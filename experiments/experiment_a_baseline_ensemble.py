"""
Experiment A: Baseline Ensemble
================================
Verifies H3/H7 robustness to U_exp selection by recomputing holonomy
deviation from 5 random subsets of T1 conversations.

Requires raw activations (activations.npz) and grouping (gamma.npz).
If raw data is unavailable, falls back to bootstrap analysis on existing
curvature results.

Outputs:
- H3 ε² across 5 baselines
- H7 Cohen's d and T0 rank across 5 baselines
- Robustness summary
"""

import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.config import RESULTS_DIR, LAYER_NAMES, ALPHA, CACHE_DIR


def load_curvature(model_name: str, results_dir: Path = RESULTS_DIR) -> pd.DataFrame:
    csv_path = results_dir / f"{model_name}_curvature.csv"
    return pd.read_csv(csv_path)


def h3_scenario_discrimination(df: pd.DataFrame) -> float:
    """Compute Kruskal-Wallis ε² for scenario discrimination."""
    curvature_cols = [
        c for c in df.columns
        if c.startswith("curvature_") and c != "curvature_total"
    ]

    grouped = df.groupby(["scenario", "conversation_id"])[curvature_cols].mean()
    groups = []
    for scenario in sorted(df["scenario"].unique()):
        mask = grouped.index.get_level_values("scenario") == scenario
        values = grouped.loc[mask, curvature_cols[0]].dropna().values
        if len(values) > 0:
            groups.append(values)

    if len(groups) < 2:
        return 0.0

    kw = stats.kruskal(*groups)
    n_total = sum(len(g) for g in groups)
    return float(kw.statistic / (n_total - 1)) if n_total > 1 else 0.0


def h7_t0_analysis(df: pd.DataFrame) -> dict:
    """Compute H7 statistics: T0 vs T1."""
    t0 = df.loc[df["scenario"] == "T0", "curvature_total"].dropna().values
    t1 = df.loc[df["scenario"] == "T1", "curvature_total"].dropna().values
    failure = df.loc[df["scenario"].isin(["T2", "T3", "T4", "T5"]), "curvature_total"].dropna().values

    if len(t0) == 0 or len(t1) == 0:
        return {"error": "insufficient data"}

    mw = stats.mannwhitneyu(t0, t1, alternative="greater")
    pooled_std = np.sqrt((t0.var() + t1.var()) / 2)
    cohens_d = float((t0.mean() - t1.mean()) / pooled_std) if pooled_std > 0 else 0.0

    # Compute per-scenario means to check T0 rank
    scenario_means = {}
    for sc in ["T0", "T1", "T2", "T3", "T4", "T5"]:
        vals = df.loc[df["scenario"] == sc, "curvature_total"].dropna().values
        if len(vals) > 0:
            scenario_means[sc] = float(vals.mean())

    # Rank scenarios by mean curvature (highest = rank 1)
    ranked = sorted(scenario_means.items(), key=lambda x: x[1], reverse=True)
    t0_rank = next(i + 1 for i, (sc, _) in enumerate(ranked) if sc == "T0")

    return {
        "t0_mean": float(t0.mean()),
        "t1_mean": float(t1.mean()),
        "cohens_d": cohens_d,
        "p_value": float(mw.pvalue),
        "t0_rank": t0_rank,
        "scenario_means": scenario_means,
    }


def bootstrap_baseline_ensemble(
    df: pd.DataFrame,
    n_subsets: int = 5,
    subset_size: int = 6,
    n_bootstrap: int = 20,
    seed: int = 42,
) -> dict:
    """Simulate baseline ensemble by bootstrap resampling T1 curvature values.

    This is a simplified approximation when raw activations are unavailable.
    It resamples T1 conversations and recomputes the baseline mean curvature,
    then checks if H3/H7 conclusions hold across resamples.

    Parameters
    ----------
    df : pd.DataFrame
        Curvature results.
    n_subsets : int
        Number of random subsets.
    subset_size : int
        Size of each T1 subset.
    n_bootstrap : int
        Number of bootstrap iterations per subset.
    seed : int
        Random seed.

    Returns
    -------
    dict with per-subset and aggregate results.
    """
    rng = np.random.default_rng(seed)

    # Get T1 conversation IDs
    t1_convs = df[df["scenario"] == "T1"]["conversation_id"].unique()
    n_t1 = len(t1_convs)
    print(f"  T1 conversations available: {n_t1}")

    if n_t1 < subset_size * n_subsets:
        print(f"  Warning: Not enough T1 conversations for {n_subsets} non-overlapping subsets of size {subset_size}")
        print(f"  Using overlapping subsets with replacement")
        subsets = [rng.choice(t1_convs, size=subset_size, replace=False) for _ in range(n_subsets)]
    else:
        # Non-overlapping subsets
        perm = rng.permutation(n_t1)
        subsets = [t1_convs[perm[i*subset_size:(i+1)*subset_size]] for i in range(n_subsets)]

    results_per_subset = []

    for m, subset in enumerate(subsets):
        # For each bootstrap iteration, resample the T1 subset
        h3_values = []
        h7_values = []
        t0_ranks = []

        for b in range(n_bootstrap):
            # Resample T1 subset with replacement
            resampled_t1 = rng.choice(subset, size=len(subset), replace=True)

            # Create modified dataframe where only the resampled T1 conversations
            # are used as "baseline" (we approximate by checking if conclusions hold
            # when we vary which T1 conversations contribute to the mean)
            t1_mask = df["conversation_id"].isin(resampled_t1)

            # For H3: use all scenarios but with resampled T1
            df_resampled = pd.concat([
                df[df["scenario"] != "T1"],
                df[t1_mask],
            ])

            # Compute H3 on resampled data
            h3_eps = h3_scenario_discrimination(df_resampled)
            h3_values.append(h3_eps)

            # Compute H7 on resampled data
            h7 = h7_t0_analysis(df_resampled)
            if "cohens_d" in h7:
                h7_values.append(h7["cohens_d"])
                t0_ranks.append(h7["t0_rank"])

        subset_result = {
            "subset_id": m,
            "subset_size": len(subset),
            "t1_ids": subset.tolist(),
            "h3_epsilon_sq": {
                "mean": float(np.mean(h3_values)),
                "std": float(np.std(h3_values)),
                "min": float(np.min(h3_values)),
                "max": float(np.max(h3_values)),
            },
            "h7_cohens_d": {
                "mean": float(np.mean(h7_values)),
                "std": float(np.std(h7_values)),
                "min": float(np.min(h7_values)),
                "max": float(np.max(h7_values)),
            },
            "t0_rank": {
                "mode": int(stats.mode(t0_ranks, keepdims=True).mode[0]),
                "always_first": all(r == 1 for r in t0_ranks),
                "ranks": [int(r) for r in t0_ranks],
            },
        }
        results_per_subset.append(subset_result)

    # Aggregate across subsets
    all_h3_means = [s["h3_epsilon_sq"]["mean"] for s in results_per_subset]
    all_h7_means = [s["h7_cohens_d"]["mean"] for s in results_per_subset]
    all_t0_first = [s["t0_rank"]["always_first"] for s in results_per_subset]

    return {
        "n_subsets": n_subsets,
        "subset_size": subset_size,
        "n_bootstrap": n_bootstrap,
        "per_subset": results_per_subset,
        "aggregate": {
            "h3_epsilon_sq_range": [float(min(all_h3_means)), float(max(all_h3_means))],
            "h3_epsilon_sq_mean": float(np.mean(all_h3_means)),
            "h3_robust": all(v > 0.30 for v in all_h3_means),
            "h7_cohens_d_range": [float(min(all_h7_means)), float(max(all_h7_means))],
            "h7_cohens_d_mean": float(np.mean(all_h7_means)),
            "t0_always_first_in": sum(all_t0_first),
            "t0_rank_robust": sum(all_t0_first) >= 4,
        },
    }


def run_with_activations(model_name: str, results_dir: Path) -> dict:
    """Run full baseline ensemble using raw activations.

    Uses GPU-batched transport/curvature (same path as the main pipeline).
    Requires activations.npz and grouping.npz.
    """
    import torch
    from experiments.curvature.compute import (
        _build_P_stack,
        _batched_transport,
        _batched_curvature,
        compute_projection_bases,
    )

    activations_path = CACHE_DIR / model_name / "activations.npz"
    gamma_path = CACHE_DIR / model_name / f"{model_name}_grouping.npz"

    if not activations_path.exists():
        raise FileNotFoundError(
            f"Activations not found at {activations_path}. "
            f"Run extraction first: python -m experiments.extraction.extract {model_name}"
        )

    from experiments.curvature.compute import EPSILON_INV

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    raw = np.load(activations_path, allow_pickle=False)
    grouping = np.load(gamma_path, allow_pickle=False)
    gamma = grouping["gamma"]

    # Get all test conversation IDs
    all_conv_ids = set()
    t1_ids = []
    for key in raw.files:
        parts = key.rsplit("/", 1)
        if len(parts) >= 2 and "/test/" in parts[0]:
            conv_id = parts[0]
            all_conv_ids.add(conv_id)
            if conv_id.startswith("T1/"):
                t1_ids.append(conv_id)
    all_conv_ids = sorted(all_conv_ids)
    t1_ids = sorted(set(t1_ids))
    n_t1 = len(t1_ids)
    print(f"  T1 conversations: {n_t1}, total: {len(all_conv_ids)}")

    # Split T1 into 5 non-overlapping subsets
    rng = np.random.default_rng(42)
    perm = rng.permutation(n_t1)
    subset_size = n_t1 // 5
    subsets = [t1_ids[perm[i*subset_size:(i+1)*subset_size].tolist()] for i in range(5)]

    # Load value matrices
    if "value_matrices" in raw.files:
        V = raw["value_matrices"]
    else:
        for key in raw.files:
            if "value_matrices" in key:
                V = raw[key]
                break

    # Pre-load all attention tensors on GPU
    attn_gpu = {}
    for conv_id in all_conv_ids:
        key = f"{conv_id}/attention"
        if key in raw.files:
            attn_np = raw[key].astype(np.float32)
            attn_gpu[conv_id] = torch.as_tensor(attn_np, device=device, dtype=torch.float32)

    # Free raw data to reduce memory (activations.npz is ~1.4GB)
    del raw
    gc.collect()

    gamma_t = torch.as_tensor(gamma, device=device, dtype=torch.long)
    V_t = torch.as_tensor(V.astype(np.float32), device=device, dtype=torch.float32)
    layer = 0

    # Pre-compute P_stack and proj_bases (same gamma for all subsets)
    P_stack_t = _build_P_stack(V_t, gamma_t, layer, device)
    proj_bases = compute_projection_bases(V, gamma, layer)

    results_per_subset = []

    for m, subset in enumerate(subsets):
        print(f"\n  Subset {m+1}/5: {len(subset)} T1 conversations")

        # Compute U_exp from this subset using GPU batched transport
        t1_products = []
        for conv_id in subset:
            attn = attn_gpu[conv_id].unsqueeze(0)
            u12 = _batched_transport(attn, P_stack_t, gamma_t, 0, 1, layer)
            u23 = _batched_transport(attn, P_stack_t, gamma_t, 1, 2, layer)
            t1_products.append(torch.bmm(u12, u23).squeeze(0))
        U_exp = torch.stack(t1_products, dim=0).mean(dim=0)
        U_exp_inv = torch.linalg.inv(U_exp + EPSILON_INV * torch.eye(
            U_exp.shape[0], device=device, dtype=torch.float32
        ))

        # Batch compute holonomy for ALL conversations
        attn_batch = torch.stack([attn_gpu[c] for c in all_conv_ids], dim=0)
        U_12 = _batched_transport(attn_batch, P_stack_t, gamma_t, 0, 1, layer)
        U_23 = _batched_transport(attn_batch, P_stack_t, gamma_t, 1, 2, layer)
        U_exp_inv_batch = U_exp_inv.unsqueeze(0).expand(len(all_conv_ids), -1, -1)
        F_batch, block_norms_list = _batched_curvature(U_12, U_23, U_exp_inv_batch, proj_bases, device)

        # Build DataFrame
        total_norms = np.zeros(len(all_conv_ids))
        for bn in block_norms_list:
            total_norms += bn ** 2
        total_norms = np.sqrt(total_norms)

        rows = []
        for idx, conv_id in enumerate(all_conv_ids):
            scenario = conv_id.split("/")[0]
            row = {"conversation_id": conv_id, "scenario": scenario, "layer": layer}
            for k, name in enumerate(LAYER_NAMES):
                row[f"curvature_{name}"] = float(block_norms_list[k][idx])
            row["curvature_total"] = float(total_norms[idx])
            rows.append(row)

        df_subset = pd.DataFrame(rows)

        # Run H3 and H7
        h3_eps = h3_scenario_discrimination(df_subset)
        h7 = h7_t0_analysis(df_subset)

        subset_result = {
            "subset_id": m,
            "t1_ids": list(subset),
            "h3_epsilon_sq": h3_eps,
            "h7_cohens_d": h7.get("cohens_d", 0),
            "h7_p_value": h7.get("p_value", 1),
            "t0_rank": h7.get("t0_rank", -1),
            "scenario_means": h7.get("scenario_means", {}),
        }
        results_per_subset.append(subset_result)
        print(f"    H3 ε² = {h3_eps:.4f}, H7 d = {h7.get('cohens_d', 0):.4f}, T0 rank = {h7.get('t0_rank', -1)}")

        del U_exp, U_exp_inv, U_exp_inv_batch, attn_batch, U_12, U_23, F_batch

    # Cleanup GPU
    del attn_gpu, gamma_t, V_t, P_stack_t
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Aggregate
    all_h3 = [s["h3_epsilon_sq"] for s in results_per_subset]
    all_h7 = [s["h7_cohens_d"] for s in results_per_subset]
    all_ranks = [s["t0_rank"] for s in results_per_subset]

    return {
        "method": "full_recomputation",
        "n_subsets": 5,
        "per_subset": results_per_subset,
        "aggregate": {
            "h3_epsilon_sq_range": [float(min(all_h3)), float(max(all_h3))],
            "h3_epsilon_sq_mean": float(np.mean(all_h3)),
            "h3_robust": all(v > 0.30 for v in all_h3),
            "h7_cohens_d_range": [float(min(all_h7)), float(max(all_h7))],
            "h7_cohens_d_mean": float(np.mean(all_h7)),
            "t0_always_first": all(r == 1 for r in all_ranks),
            "t0_rank_robust": sum(r == 1 for r in all_ranks) >= 4,
        },
    }


def run_experiment_a(model_name: str, results_dir: Path = RESULTS_DIR) -> dict:
    """Run the baseline ensemble experiment.

    Tries full recomputation first; falls back to bootstrap if no activations.
    """
    print(f"=== Experiment A: Baseline Ensemble ({model_name}) ===\n")

    # Try full recomputation
    activations_path = CACHE_DIR / model_name / "activations.npz"
    if activations_path.exists():
        print("Raw activations found. Running full recomputation ...")
        results = run_with_activations(model_name, results_dir)
    else:
        print("Raw activations not found. Running bootstrap approximation ...")
        df = load_curvature(model_name, results_dir)
        results = bootstrap_baseline_ensemble(df)

    # Print summary
    agg = results["aggregate"]
    print(f"\n{'='*60}")
    print(f"Baseline Ensemble Summary ({model_name})")
    print(f"{'='*60}")
    print(f"  H3 ε² range: [{agg['h3_epsilon_sq_range'][0]:.4f}, {agg['h3_epsilon_sq_range'][1]:.4f}]")
    print(f"  H3 robust (all > 0.30): {agg['h3_robust']}")
    print(f"  H7 Cohen's d range: [{agg['h7_cohens_d_range'][0]:.4f}, {agg['h7_cohens_d_range'][1]:.4f}]")
    print(f"  T0 always rank 1: {agg.get('t0_always_first', 'N/A')}")
    print(f"  T0 rank robust (≥4/5): {agg.get('t0_rank_robust', 'N/A')}")
    print(f"{'='*60}")

    # Save
    output_path = results_dir / f"{model_name}_baseline_ensemble.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to {output_path}")

    return results


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "llama-7b"
    run_experiment_a(model_name)
