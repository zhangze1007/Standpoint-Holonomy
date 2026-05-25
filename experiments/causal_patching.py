"""
Experiment: Causal Activation Patching
=======================================
Tests whether attention patterns causally drive holonomy deviation by
swapping attention matrices between scenarios.

Design:
1. Cross-scenario patching: replace attention in scenario B with attention
   from scenario A, recompute holonomy. If holonomy shifts toward A's
   pattern, attention causally drives the geometric signal.
2. Layer-specific patching: patch attention at specific layers only, to
   identify which layers carry the causal signal.
3. Standpoint-group patching: patch attention for heads in specific
   standpoint groups (γ-defined), to identify which groups matter.

Requires: activations.npz, grouping.npz, value_matrices.

Outputs:
- Per-patch holonomy deviation curves
- Causal effect sizes (how much patching shifts holonomy)
- Layer-wise and group-wise causal importance
"""

import gc
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.config import (
    RESULTS_DIR, CACHE_DIR, LAYER_NAMES, ALPHA, MODELS, ModelConfig,
)
from experiments.curvature.compute import (
    _build_P_stack,
    _batched_transport,
    _batched_curvature,
    compute_projection_bases,
    compute_projection_bases_gpu,
    build_layer_cache,
    EPSILON_INV,
)


def _load_data(model_name: str, device: torch.device):
    """Load activations, gamma, value matrices. Returns (activations, gamma, V, config)."""
    config = MODELS[model_name]
    cache = CACHE_DIR / model_name

    # Load gamma
    gamma = np.load(cache / f"{model_name}_grouping.npz")["gamma"]

    # Try split files first (memory-efficient)
    attn_path = cache / "attention_only.npz"
    vdir = cache / "value_layers"

    if attn_path.exists() and vdir.exists():
        attn_raw = np.load(attn_path, allow_pickle=False)
        activations = {}
        for key in attn_raw.files:
            parts = key.rsplit("/", 1)
            if len(parts) >= 2 and "/test/" in parts[0]:
                activations[parts[0]] = {"attention": attn_raw[key]}
        attn_raw.close()
        n_layers = len(list(vdir.glob("layer_*.npy")))
        V = _VPerLayer(vdir, n_layers)
    else:
        raw = np.load(cache / "activations.npz", allow_pickle=False)
        activations = {}
        for key in raw.files:
            parts = key.rsplit("/", 1)
            if len(parts) >= 2 and "/test/" in parts[0] and parts[1] == "attention":
                activations[parts[0]] = {"attention": np.array(raw[key])}
        V = raw["value_matrices"]
        n_layers = V.shape[0]

    return activations, gamma, V, config


class _VPerLayer:
    """Thin wrapper to load V per-layer on demand."""
    def __init__(self, layer_dir, n_layers):
        self._dir = layer_dir
        self.shape = (n_layers, 32, 4096, 128)
    def __getitem__(self, idx):
        return np.load(self._dir / f"layer_{idx:02d}.npy")


def _compute_holonomy_batch(
    activations: Dict,
    conv_ids: List[str],
    gamma: np.ndarray,
    V,
    t1_ids: List[str],
    device: torch.device,
    layer: Optional[int] = None,
    batch_size: int = 8,
    layer_cache: list = None,
) -> pd.DataFrame:
    """Compute holonomy deviation for a set of conversations.

    Uses GPU-batched transport/curvature.
    If layer_cache is provided, skips recomputing P_stack/proj_bases/U_exp_inv.
    If layer is specified, compute only for that layer.
    """
    n_layers = V.shape[0]
    gamma_t = torch.as_tensor(gamma, device=device, dtype=torch.long)
    layers_to_compute = [layer] if layer is not None else list(range(n_layers))

    rows = []
    for lay in layers_to_compute:
        if layer_cache is not None:
            P_stack_t = layer_cache[lay]["P_stack"]
            proj_bases = layer_cache[lay]["proj_bases"]
            U_exp_inv = layer_cache[lay]["U_exp_inv"]
        else:
            V_layer = np.array(V[lay]).astype(np.float32)
            V_layer_3d = V_layer[np.newaxis, ...]
            V_layer_t = torch.as_tensor(V_layer_3d, device=device, dtype=torch.float32)
            P_stack_t = _build_P_stack(V_layer_t, gamma_t, 0, device)
            proj_bases = compute_projection_bases_gpu(V_layer_3d, gamma, 0, device)
            t1_attn_list = []
            for cid in t1_ids:
                attn_np = activations[cid]["attention"].astype(np.float32)
                t1_attn_list.append(torch.as_tensor(attn_np, device=device, dtype=torch.float32))
            t1_attn_batch = torch.stack(t1_attn_list, dim=0)
            del t1_attn_list
            u12_t1 = _batched_transport(t1_attn_batch, P_stack_t, gamma_t, 0, 1, lay)
            u23_t1 = _batched_transport(t1_attn_batch, P_stack_t, gamma_t, 1, 2, lay)
            U_exp = torch.bmm(u12_t1, u23_t1).mean(dim=0)
            del t1_attn_batch, u12_t1, u23_t1
            U_exp_inv = torch.linalg.inv(U_exp + EPSILON_INV * torch.eye(
                U_exp.shape[0], device=device, dtype=torch.float32
            ))

        # --- Batched conversation processing ---
        for batch_start in range(0, len(conv_ids), batch_size):
            batch_ids = conv_ids[batch_start : batch_start + batch_size]
            B = len(batch_ids)

            attn_list = []
            for cid in batch_ids:
                attn_np = activations[cid]["attention"].astype(np.float32)
                attn_list.append(torch.as_tensor(attn_np, device=device, dtype=torch.float32))
            attn_batch = torch.stack(attn_list, dim=0)
            del attn_list

            U_12 = _batched_transport(attn_batch, P_stack_t, gamma_t, 0, 1, lay)
            U_23 = _batched_transport(attn_batch, P_stack_t, gamma_t, 1, 2, lay)
            del attn_batch

            U_exp_inv_batch = U_exp_inv.unsqueeze(0).expand(B, -1, -1)

            F, block_norms_list = _batched_curvature(
                U_12, U_23, U_exp_inv_batch, proj_bases, device
            )
            del U_12, U_23, U_exp_inv_batch, F

            for i, cid in enumerate(batch_ids):
                scenario = cid.split("/")[0]
                row = {"conversation_id": cid, "scenario": scenario, "layer": lay}
                curvatures = {}
                for idx, name in enumerate(LAYER_NAMES):
                    curvatures[name] = float(block_norms_list[idx][i])
                curvature_total = float(np.sqrt(sum(v ** 2 for v in curvatures.values())))
                for name in LAYER_NAMES:
                    row[f"curvature_{name}"] = curvatures[name]
                row["curvature_total"] = curvature_total
                rows.append(row)

        if layer_cache is None:
            del P_stack_t, U_exp, U_exp_inv, V_layer_t, V_layer, V_layer_3d
        gc.collect()

    del gamma_t
    return pd.DataFrame(rows)


def cross_scenario_patching(
    activations: Dict,
    gamma: np.ndarray,
    V,
    device: torch.device,
    n_pairs: int = 10,
    seed: int = 42,
    layer_cache: list = None,
) -> pd.DataFrame:
    """Patch attention from scenario A into scenario B, measure holonomy shift.

    For each (source, target) pair, replace target's attention with source's,
    compute holonomy, compare to original.
    """
    rng = np.random.default_rng(seed)

    # Group conversations by scenario
    by_scenario = {}
    for cid in activations:
        sc = cid.split("/")[0]
        if sc.startswith("T"):
            by_scenario.setdefault(sc, []).append(cid)

    scenarios = sorted(by_scenario.keys())
    t1_ids = sorted(by_scenario.get("T1", []))

    if not t1_ids:
        print("  No T1 conversations found for U_exp baseline")
        return pd.DataFrame()

    # Sample pairs: (source_scenario, target_conv)
    pairs = []
    failure_scenarios = [s for s in scenarios if s != "T1"]
    for target_sc in failure_scenarios:
        target_convs = by_scenario[target_sc]
        source_convs = by_scenario.get("T1", [])
        n = min(n_pairs, len(target_convs), len(source_convs))
        for i in range(n):
            pairs.append((source_convs[i], target_convs[i], target_sc))

    print(f"  Running {len(pairs)} cross-scenario patches ...")

    rows = []
    for src_cid, tgt_cid, tgt_sc in pairs:
        # Original holonomy for target
        df_orig = _compute_holonomy_batch(
            activations, [tgt_cid], gamma, V, t1_ids, device, layer_cache=layer_cache
        )
        orig_total = df_orig["curvature_total"].iloc[0]

        # Patched: replace target's attention with source's
        patched_acts = dict(activations)
        patched_acts[tgt_cid] = {"attention": activations[src_cid]["attention"]}

        df_patched = _compute_holonomy_batch(
            patched_acts, [tgt_cid], gamma, V, t1_ids, device, layer_cache=layer_cache
        )
        patched_total = df_patched["curvature_total"].iloc[0]

        # Also get source's original holonomy
        df_src = _compute_holonomy_batch(
            activations, [src_cid], gamma, V, t1_ids, device, layer_cache=layer_cache
        )
        src_total = df_src["curvature_total"].iloc[0]

        rows.append({
            "source": src_cid,
            "target": tgt_cid,
            "target_scenario": tgt_sc,
            "source_holonomy": src_total,
            "target_holonomy_orig": orig_total,
            "target_holonomy_patched": patched_total,
            "delta": patched_total - orig_total,
            "shifted_toward_source": abs(patched_total - src_total) < abs(orig_total - src_total),
        })

    return pd.DataFrame(rows)


def layer_specific_patching(
    activations: Dict,
    gamma: np.ndarray,
    V,
    device: torch.device,
    n_samples: int = 5,
    seed: int = 42,
    layer_cache: list = None,
) -> pd.DataFrame:
    """Patch attention at specific layers to identify causal layers.

    For each layer l, replace T2 attention at layer l with T1 attention,
    keep other layers original, measure holonomy shift.
    """
    rng = np.random.default_rng(seed)
    n_layers = V.shape[0]

    # Get T1 and T2 conversations
    t1_ids = sorted([c for c in activations if c.startswith("T1/")])
    t2_ids = sorted([c for c in activations if c.startswith("T2/")])

    if not t1_ids or not t2_ids:
        print("  Need T1 and T2 conversations for layer patching")
        return pd.DataFrame()

    n = min(n_samples, len(t1_ids), len(t2_ids))
    t1_sample = t1_ids[:n]
    t2_sample = t2_ids[:n]

    print(f"  Layer-specific patching: {n_layers} layers, {n} samples each ...")

    rows = []
    for t2_cid, t1_cid in zip(t2_sample, t1_sample):
        # Original T2 holonomy
        df_orig = _compute_holonomy_batch(
            activations, [t2_cid], gamma, V, t1_ids, device
        )
        orig_total = df_orig["curvature_total"].iloc[0]

        # T1 original
        df_t1 = _compute_holonomy_batch(
            activations, [t1_cid], gamma, V, t1_ids, device
        )
        t1_total = df_t1["curvature_total"].iloc[0]

        # Patch each layer individually
        for lay in range(n_layers):
            # Create patched attention: T2 with layer l replaced by T1
            attn_orig = activations[t2_cid]["attention"].copy()
            attn_patched = attn_orig.copy()
            attn_patched[lay] = activations[t1_cid]["attention"][lay]

            patched_acts = dict(activations)
            patched_acts[t2_cid] = {"attention": attn_patched}

            df_patched = _compute_holonomy_batch(
                patched_acts, [t2_cid], gamma, V, t1_ids, device, layer=lay
            )
            patched_total = df_patched["curvature_total"].iloc[0]

            # Get per-layer original and patched
            df_orig_layer = _compute_holonomy_batch(
                activations, [t2_cid], gamma, V, t1_ids, device, layer=lay
            )
            orig_layer_total = df_orig_layer["curvature_total"].iloc[0]

            rows.append({
                "conversation": t2_cid,
                "layer": lay,
                "orig_total": orig_total,
                "orig_layer_total": orig_layer_total,
                "patched_layer_total": patched_total,
                "layer_delta": patched_total - orig_layer_total,
                "t1_layer_total": df_t1["curvature_total"].iloc[0] if lay == 0 else None,
            })

    return pd.DataFrame(rows)


def standpoint_group_patching(
    activations: Dict,
    gamma: np.ndarray,
    V,
    device: torch.device,
    n_samples: int = 5,
    seed: int = 42,
    layer_cache: list = None,
) -> pd.DataFrame:
    """Patch attention for heads in specific standpoint groups.

    For each group k, replace attention for heads in group k with T1 attention,
    measure holonomy shift.
    """
    rng = np.random.default_rng(seed)
    n_standpoint = len(LAYER_NAMES)

    t1_ids = sorted([c for c in activations if c.startswith("T1/")])
    t2_ids = sorted([c for c in activations if c.startswith("T2/")])

    if not t1_ids or not t2_ids:
        print("  Need T1 and T2 conversations for group patching")
        return pd.DataFrame()

    n = min(n_samples, len(t1_ids), len(t2_ids))
    t1_sample = t1_ids[:n]
    t2_sample = t2_ids[:n]

    # Identify head indices per group
    groups = {}
    for k in range(n_standpoint):
        groups[k] = np.where(gamma == k)[0]

    print(f"  Group patching: {n_standpoint} groups, {n} samples ...")

    rows = []
    for t2_cid, t1_cid in zip(t2_sample, t1_sample):
        attn_t2 = activations[t2_cid]["attention"].copy()
        attn_t1 = activations[t1_cid]["attention"]

        # Original T2 holonomy
        df_orig = _compute_holonomy_batch(
            activations, [t2_cid], gamma, V, t1_ids, device
        )
        orig_total = df_orig["curvature_total"].iloc[0]

        for k in range(n_standpoint):
            group_name = LAYER_NAMES[k]
            head_indices = groups[k]

            if len(head_indices) == 0:
                continue

            # Patch: replace attention for heads in group k with T1's
            attn_patched = attn_t2.copy()
            for h in head_indices:
                attn_patched[:, h, :, :] = attn_t1[:, h, :, :]

            patched_acts = dict(activations)
            patched_acts[t2_cid] = {"attention": attn_patched}

            df_patched = _compute_holonomy_batch(
                patched_acts, [t2_cid], gamma, V, t1_ids, device
            )
            patched_total = df_patched["curvature_total"].iloc[0]

            rows.append({
                "conversation": t2_cid,
                "group": group_name,
                "group_idx": k,
                "n_heads": len(head_indices),
                "orig_total": orig_total,
                "patched_total": patched_total,
                "delta": patched_total - orig_total,
                "relative_shift": (patched_total - orig_total) / orig_total if orig_total > 0 else 0,
            })

    return pd.DataFrame(rows)


def run_causal_patching(model_name: str, results_dir: Path = RESULTS_DIR) -> dict:
    """Run all causal patching experiments."""
    print(f"=== Causal Activation Patching ({model_name}) ===\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    activations, gamma, V, config = _load_data(model_name, device)
    print(f"Loaded {len(activations)} conversations, {len(gamma)} heads")

    # Pre-load V into RAM to avoid mmap I/O overhead
    print("Pre-loading value matrices into RAM ...")
    V_ram = np.array(V).astype(np.float32)

    # Build layer cache once (P_stack + proj_bases + U_exp_inv for all layers)
    t1_activations = {k: v for k, v in activations.items() if k.startswith("T1/")}
    layer_cache = build_layer_cache(V_ram, gamma, t1_activations, device)

    results = {"model": model_name, "n_conversations": len(activations)}

    # 1. Cross-scenario patching
    print("\n--- Cross-Scenario Patching ---")
    df_cross = cross_scenario_patching(activations, gamma, V_ram, device, layer_cache=layer_cache)
    if not df_cross.empty:
        mean_delta = df_cross["delta"].mean()
        shifted_pct = df_cross["shifted_toward_source"].mean() * 100
        print(f"  Mean holonomy delta: {mean_delta:.4f}")
        print(f"  Shifted toward source: {shifted_pct:.1f}%")
        results["cross_scenario"] = {
            "n_pairs": len(df_cross),
            "mean_delta": float(mean_delta),
            "shifted_toward_source_pct": float(shifted_pct),
            "per_pair": df_cross.to_dict(orient="records"),
        }
        df_cross.to_csv(results_dir / f"{model_name}_causal_cross_scenario.csv", index=False)

    # 2. Layer-specific patching
    print("\n--- Layer-Specific Patching ---")
    df_layer = layer_specific_patching(activations, gamma, V_ram, device, layer_cache=layer_cache)
    if not df_layer.empty:
        layer_deltas = df_layer.groupby("layer")["layer_delta"].agg(["mean", "std"])
        most_causal_layer = layer_deltas["mean"].abs().idxmax()
        print(f"  Most causal layer: {most_causal_layer} (mean delta: {layer_deltas.loc[most_causal_layer, 'mean']:.4f})")
        results["layer_specific"] = {
            "most_causal_layer": int(most_causal_layer),
            "per_layer_mean_delta": {
                str(k): float(v) for k, v in layer_deltas["mean"].items()
            },
        }
        df_layer.to_csv(results_dir / f"{model_name}_causal_layer_specific.csv", index=False)

    # 3. Standpoint-group patching
    print("\n--- Standpoint-Group Patching ---")
    df_group = standpoint_group_patching(activations, gamma, V_ram, device, layer_cache=layer_cache)
    if not df_group.empty:
        group_deltas = df_group.groupby("group")["delta"].agg(["mean", "std"])
        most_causal_group = group_deltas["mean"].abs().idxmax()
        print(f"  Most causal group: {most_causal_group} (mean delta: {group_deltas.loc[most_causal_group, 'mean']:.4f})")
        results["group_patching"] = {
            "most_causal_group": most_causal_group,
            "per_group_mean_delta": {
                k: float(v) for k, v in group_deltas["mean"].items()
            },
        }
        df_group.to_csv(results_dir / f"{model_name}_causal_group_patching.csv", index=False)

    # Save summary
    output_path = results_dir / f"{model_name}_causal_patching.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    return results


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "llama-7b"
    run_causal_patching(model_name)
