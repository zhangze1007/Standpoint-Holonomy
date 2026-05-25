"""
LCESA Ablation Studies (GPU-Accelerated)
=========================================
Tests robustness of curvature signals by:
1. Dropping model layers (keeping evenly-spaced subset)
2. Shortening event sequences (truncating from the end)

GPU optimizations mirror compute.py:
- Pre-compute P_k projection matrices per layer
- Batch conversations via torch.einsum
- torch.bmm for batched curvature tensor computation
"""
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from experiments.config import (
    LAYER_NAMES, MODELS, CACHE_DIR, RESULTS_DIR, DATA_DIR,
    ABLATION_LAYER_COUNTS, ABLATION_LENGTHS,
)
from experiments.curvature.compute import (
    compute_projection_bases,
    compute_projection_bases_gpu,
    _build_P_stack,
    _batched_transport,
    _batched_curvature,
    EPSILON_PROJ,
    EPSILON_INV,
)


def _pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ablate_layers(
    attention: np.ndarray,
    value_matrices: np.ndarray,
    n_keep: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Keep evenly-spaced model layers, drop the rest."""
    n_layers = attention.shape[0]
    if n_keep > n_layers:
        raise ValueError(f"n_keep={n_keep} > n_layers={n_layers}")
    indices = np.linspace(0, n_layers - 1, n_keep, dtype=int)
    return attention[indices], value_matrices[indices]


def ablate_sequence_length(
    attention: np.ndarray,
    n_events: int,
) -> np.ndarray:
    """Reduce event sequence to first n_events."""
    return attention[:, :, :n_events, :n_events]


def _gpu_curvature_for_condition(
    test_activations: Dict[str, Dict[str, np.ndarray]],
    attention_key: str,  # key to get attention from fields
    attn_transform,      # function(fields["attention"]) -> ablated attention
    value_matrices: np.ndarray,
    gamma: np.ndarray,
    n_layers_cond: int,
    device: torch.device,
    batch_size: int = 16,
) -> List[dict]:
    """Run curvature computation for one ablation condition using GPU batching.

    Parameters
    ----------
    test_activations : dict of conv_id -> fields
    attention_key : not used (kept for API compat)
    attn_transform : callable that transforms attention arrays
    value_matrices : (n_layers, n_heads, d_model, d_head) for this condition
    gamma : (n_heads,) standpoint assignment
    n_layers_cond : number of layers in this ablation condition
    device : torch device
    batch_size : GPU batch size

    Returns
    -------
    list of row dicts
    """
    gamma_t = torch.as_tensor(gamma, device=device, dtype=torch.long)
    V_t = torch.as_tensor(value_matrices.astype(np.float32), device=device, dtype=torch.float32)

    all_conv_ids = sorted(test_activations.keys())
    rows = []

    for layer in range(n_layers_cond):
        # Pre-compute P_k stack for this layer
        P_stack = _build_P_stack(V_t, gamma_t, layer, device)

        # Projection bases for block norms
        V_layer_slice = value_matrices[layer] if value_matrices.ndim == 4 else value_matrices[0]
        V_3d = V_layer_slice[np.newaxis, ...].astype(np.float32)
        proj_bases = compute_projection_bases_gpu(V_3d, gamma, 0, device)

        # Compute U_exp from T1 baseline
        t1_ids = [c for c in all_conv_ids if c.startswith("T1/")]
        if not t1_ids:
            raise ValueError("No T1 conversations found for baseline.")

        t1_products = []
        for conv_id in t1_ids:
            attn_abl = attn_transform(test_activations[conv_id]["attention"])
            attn_t = torch.as_tensor(attn_abl.astype(np.float32), device=device, dtype=torch.float32).unsqueeze(0)
            u12 = _batched_transport(attn_t, P_stack, gamma_t, 0, 1, layer)
            u23 = _batched_transport(attn_t, P_stack, gamma_t, 1, 2, layer)
            t1_products.append(torch.bmm(u12, u23).squeeze(0))

        U_exp = torch.stack(t1_products, dim=0).mean(dim=0)
        U_exp_reg = U_exp + EPSILON_INV * torch.eye(U_exp.shape[0], device=device, dtype=torch.float32)
        U_exp_inv = torch.linalg.inv(U_exp_reg)

        # Batch curvature for all conversations
        for batch_start in range(0, len(all_conv_ids), batch_size):
            batch_ids = all_conv_ids[batch_start:batch_start + batch_size]
            B = len(batch_ids)

            attn_list = []
            for conv_id in batch_ids:
                attn_abl = attn_transform(test_activations[conv_id]["attention"])
                attn_list.append(torch.as_tensor(attn_abl.astype(np.float32), device=device, dtype=torch.float32))
            attn_batch = torch.stack(attn_list, dim=0)

            U_12 = _batched_transport(attn_batch, P_stack, gamma_t, 0, 1, layer)
            U_23 = _batched_transport(attn_batch, P_stack, gamma_t, 1, 2, layer)
            U_exp_inv_batch = U_exp_inv.unsqueeze(0).expand(B, -1, -1)

            _, block_norms_list = _batched_curvature(U_12, U_23, U_exp_inv_batch, proj_bases, device)

            for i, conv_id in enumerate(batch_ids):
                scenario = conv_id.split("/")[0]
                curvatures = {}
                for idx, name in enumerate(LAYER_NAMES):
                    curvatures[name] = float(block_norms_list[idx][i])
                curvature_total = float(np.sqrt(sum(v**2 for v in curvatures.values())))

                rows.append({
                    "conversation_id": conv_id,
                    "scenario": scenario,
                    "layer": layer,
                    **{f"curvature_{name}": curvatures[name] for name in LAYER_NAMES},
                    "curvature_total": curvature_total,
                })

    return rows


def run_ablation_study(
    model_name: str,
    activations_path: Path,
    gamma_path: Path,
    output_dir: Path = RESULTS_DIR,
    batch_size: int = 16,
) -> pd.DataFrame:
    """Run full ablation study: layer count x sequence length grid."""
    model_config = MODELS[model_name]
    device = _pick_device()
    print(f"Ablation using device: {device}")

    raw = np.load(activations_path, allow_pickle=False)
    grouping = np.load(gamma_path, allow_pickle=False)
    gamma = grouping["gamma"]

    # Reorganize test activations
    test_activations: Dict[str, Dict[str, np.ndarray]] = {}
    conv_ids_seen: set = set()
    for key in raw.files:
        parts = key.rsplit("/", 1)
        if len(parts) < 2:
            continue
        conv_id, field = parts[0], parts[1]
        if "/test/" not in conv_id:
            continue
        conv_ids_seen.add(conv_id)
    for conv_id in conv_ids_seen:
        fields: Dict[str, np.ndarray] = {}
        for field in ("attention", "residuals"):
            full_key = f"{conv_id}/{field}"
            if full_key in raw.files:
                fields[field] = raw[full_key]
        test_activations[conv_id] = fields

    # Load shared value_matrices
    shared_V = None
    if "value_matrices" in raw.files:
        shared_V = raw["value_matrices"]
    else:
        for fields in test_activations.values():
            if "value_matrices" in fields:
                shared_V = fields["value_matrices"]
                break

    all_rows = []

    # --- Layer ablation ---
    for n_keep in ABLATION_LAYER_COUNTS:
        if n_keep > model_config.n_layers:
            continue
        print(f"  Layer ablation: n_keep={n_keep} (GPU batched)")

        # Ablate value matrices (select evenly-spaced layers)
        indices = np.linspace(0, model_config.n_layers - 1, n_keep, dtype=int)
        V_abl = shared_V[indices]
        gamma_adj = gamma[:V_abl.shape[1]]  # gamma stays same (head assignment is model-level)

        def attn_transform_layer(attn, _indices=indices):
            return attn[_indices]

        rows = _gpu_curvature_for_condition(
            test_activations, "attention", attn_transform_layer,
            V_abl, gamma_adj, n_keep, device, batch_size,
        )
        for r in rows:
            r["model"] = model_name
            r["ablation_type"] = "layer_count"
            r["param_value"] = n_keep
        all_rows.extend(rows)

    # --- Sequence length ablation ---
    for n_events in ABLATION_LENGTHS:
        if n_events > 5:
            print(f"  Skipping sequence_length={n_events}: stimuli have only 5 events")
            continue
        print(f"  Sequence ablation: n_events={n_events} (GPU batched)")

        def attn_transform_seq(attn, _n=n_events):
            return attn[:, :, :_n, :_n]

        rows = _gpu_curvature_for_condition(
            test_activations, "attention", attn_transform_seq,
            shared_V, gamma, model_config.n_layers, device, batch_size,
        )
        for r in rows:
            r["model"] = model_name
            r["ablation_type"] = "sequence_length"
            r["param_value"] = n_events
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    output_path = output_dir / f"{model_name}_ablation.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Ablation results: {len(df)} rows -> {output_path}")
    return df
