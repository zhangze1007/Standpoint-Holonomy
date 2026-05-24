"""
Experiment B: Null Grouping Controls
=====================================
Verifies that the learned head grouping γ provides non-trivial
diagnostic decomposition by comparing against null groupings.

Null grouping types:
1. Random: heads randomly assigned to 5 layers, preserving layer sizes
2. Shuffled: permutation of learned γ assignments
3. Layer-uniform: physical layer-based assignment

Requires raw activations (activations.npz) and learned grouping (gamma.npz).

Outputs:
- H3 ε² for learned γ and each null grouping
- Permutation test p-value
- Statistical significance of learned grouping
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.config import RESULTS_DIR, CACHE_DIR, LAYER_NAMES, ALPHA


def generate_null_groupings(
    gamma: np.ndarray,
    n_null: int = 20,
    seed: int = 42,
) -> dict:
    """Generate null grouping variants.

    Parameters
    ----------
    gamma : np.ndarray, shape (n_heads,)
        Learned head grouping.
    n_null : int
        Number of random instances per null type.
    seed : int
        Random seed.

    Returns
    -------
    dict with keys 'random', 'shuffled', 'layer_uniform', each containing
    a list of n_null gamma arrays.
    """
    rng = np.random.default_rng(seed)
    n_heads = len(gamma)
    n_standpoint = len(LAYER_NAMES)

    # Learned layer sizes
    layer_sizes = [int(np.sum(gamma == k)) for k in range(n_standpoint)]

    null_groupings = {"random": [], "shuffled": [], "layer_uniform": []}

    for _ in range(n_null):
        # Random: preserve layer sizes, randomly assign heads
        gamma_random = np.zeros(n_heads, dtype=int)
        indices = rng.permutation(n_heads)
        pos = 0
        for k, size in enumerate(layer_sizes):
            gamma_random[indices[pos:pos + size]] = k
            pos += size
        null_groupings["random"].append(gamma_random)

        # Shuffled: random permutation of learned gamma
        gamma_shuffled = rng.permutation(gamma)
        null_groupings["shuffled"].append(gamma_shuffled)

    # Layer-uniform: assign based on physical layer position
    # (first 1/5 → ψ_min, last 1/5 → ψ_pos, etc.)
    gamma_layer = np.zeros(n_heads, dtype=int)
    for h in range(n_heads):
        # Map head index to layer proportion
        proportion = h / n_heads
        layer_idx = min(int(proportion * n_standpoint), n_standpoint - 1)
        gamma_layer[h] = layer_idx
    null_groupings["layer_uniform"] = [gamma_layer] * n_null

    return null_groupings


def compute_holonomy_with_gamma(
    activations: dict,
    gamma: np.ndarray,
    value_matrices: np.ndarray,
    layer: int = 0,
) -> pd.DataFrame:
    """Compute holonomy deviation for all conversations using a given gamma.

    Parameters
    ----------
    activations : dict
        Per-conversation activation data.
    gamma : np.ndarray
        Head grouping to use.
    value_matrices : np.ndarray
        Value matrices.
    layer : int
        Model layer index.

    Returns
    -------
    pd.DataFrame with curvature results.
    """
    from experiments.curvature.compute import (
        compute_transport_operator,
        compute_curvature,
        compute_projection_bases,
    )

    # Get T1 conversations for baseline
    t1_ids = [c for c in activations if c.startswith("T1/")]
    if not t1_ids:
        raise ValueError("No T1 conversations found")

    # Compute U_exp from T1
    products = []
    for conv_id in t1_ids:
        attn = activations[conv_id]["attention"]
        U_12 = compute_transport_operator(attn, value_matrices, gamma, 0, 1, layer)
        U_23 = compute_transport_operator(attn, value_matrices, gamma, 1, 2, layer)
        products.append(U_12 @ U_23)
    U_exp = np.mean(products, axis=0)

    # Compute projection bases
    proj_bases = compute_projection_bases(value_matrices, gamma, layer)

    # Compute holonomy for all conversations
    rows = []
    for conv_id in sorted(activations.keys()):
        attn = activations[conv_id]["attention"]
        U_12 = compute_transport_operator(attn, value_matrices, gamma, 0, 1, layer)
        U_23 = compute_transport_operator(attn, value_matrices, gamma, 1, 2, layer)
        F, block_norms = compute_curvature(U_12, U_23, U_exp, proj_bases)

        scenario = conv_id.split("/")[0]
        row = {"conversation_id": conv_id, "scenario": scenario, "layer": layer}
        for name, val in block_norms.items():
            row[f"curvature_{name}"] = val
        row["curvature_total"] = float(np.sqrt(sum(v**2 for v in block_norms.values())))
        rows.append(row)

    return pd.DataFrame(rows)


def h3_epsilon_sq(df: pd.DataFrame) -> float:
    """Compute Kruskal-Wallis ε² for scenario discrimination."""
    curvature_cols = [c for c in df.columns if c.startswith("curvature_") and c != "curvature_total"]
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
    n = sum(len(g) for g in groups)
    return float(kw.statistic / (n - 1)) if n > 1 else 0.0


def run_experiment_b(model_name: str, results_dir: Path = RESULTS_DIR) -> dict:
    """Run null grouping controls experiment.

    Parameters
    ----------
    model_name : str
        Model key.
    results_dir : Path
        Results directory.

    Returns
    -------
    dict with null grouping comparison results.
    """
    print(f"=== Experiment B: Null Grouping Controls ({model_name}) ===\n")

    activations_path = CACHE_DIR / model_name / "activations.npz"
    gamma_path = CACHE_DIR / model_name / f"{model_name}_grouping.npz"

    if not activations_path.exists():
        print("Raw activations not found. Cannot run null grouping experiment.")
        print("This experiment requires recomputing holonomy deviation with different γ.")
        print(f"Please run extraction first: python -m experiments.extraction.extract {model_name}")

        # Return placeholder with instructions
        return {
            "model": model_name,
            "status": "deferred",
            "reason": "requires raw activations",
            "instructions": f"Run: python -m experiments.extraction.extract {model_name}",
            "learned_gamma_h3_epsilon_sq": None,
            "null_groupings": {},
            "permutation_p_value": None,
        }

    # Load data
    print("Loading activations and grouping ...")
    raw = np.load(activations_path, allow_pickle=False)
    grouping = np.load(gamma_path, allow_pickle=False)
    gamma = grouping["gamma"]

    # Load value matrices
    if "value_matrices" in raw.files:
        V = raw["value_matrices"]
    else:
        # Find from first conversation
        for key in raw.files:
            if "value_matrices" in key:
                V = raw[key]
                break

    # Reorganize activations
    conv_data = {}
    for key in raw.files:
        parts = key.rsplit("/", 1)
        if len(parts) < 2 or "/test/" not in parts[0]:
            continue
        conv_id, field = parts
        if conv_id not in conv_data:
            conv_data[conv_id] = {}
        conv_data[conv_id][field] = raw[key]

    # Extract attention arrays
    activations = {}
    for conv_id, fields in conv_data.items():
        if "attention" in fields:
            activations[conv_id] = {"attention": fields["attention"]}

    print(f"  Loaded {len(activations)} conversations, {len(gamma)} heads")

    # Compute learned gamma H3
    print("\nComputing holonomy with learned gamma ...")
    df_learned = compute_holonomy_with_gamma(activations, gamma, V)
    learned_eps = h3_epsilon_sq(df_learned)
    print(f"  Learned γ: H3 ε² = {learned_eps:.4f}")

    # Generate null groupings
    print("\nGenerating null groupings (20 instances each) ...")
    null_groupings = generate_null_groupings(gamma, n_null=20, seed=42)

    # Compute H3 for each null grouping
    null_results = {}
    for null_type, gammas in null_groupings.items():
        print(f"\n  Testing {null_type} null groupings ...")
        eps_values = []
        for i, gamma_null in enumerate(gammas):
            try:
                df_null = compute_holonomy_with_gamma(activations, gamma_null, V)
                eps = h3_epsilon_sq(df_null)
                eps_values.append(eps)
                if (i + 1) % 5 == 0:
                    print(f"    {i+1}/{len(gammas)}: ε² = {eps:.4f}")
            except Exception as e:
                print(f"    {i+1}/{len(gammas)}: error - {e}")

        if eps_values:
            null_results[null_type] = {
                "n_instances": len(eps_values),
                "mean_epsilon_sq": float(np.mean(eps_values)),
                "std_epsilon_sq": float(np.std(eps_values)),
                "min_epsilon_sq": float(np.min(eps_values)),
                "max_epsilon_sq": float(np.max(eps_values)),
                "epsilon_sq_values": [float(v) for v in eps_values],
            }

    # Permutation test: is learned γ significantly better than null?
    print("\nPermutation test ...")
    all_null_eps = []
    for null_type, res in null_results.items():
        all_null_eps.extend(res["epsilon_sq_values"])

    if all_null_eps:
        # Percentile of learned γ in null distribution
        percentile = float(stats.percentileofscore(all_null_eps, learned_eps))
        p_value = 1.0 - percentile / 100.0

        # One-sided test: learned > null
        n_greater = sum(1 for v in all_null_eps if v >= learned_eps)
        perm_p = (n_greater + 1) / (len(all_null_eps) + 1)
    else:
        percentile = 0.0
        p_value = 1.0
        perm_p = 1.0

    print(f"  Learned ε² = {learned_eps:.4f}")
    print(f"  Null distribution mean = {np.mean(all_null_eps):.4f}")
    print(f"  Percentile of learned: {percentile:.1f}%")
    print(f"  Permutation p-value: {perm_p:.4f}")

    # Compile results
    results = {
        "model": model_name,
        "n_conversations": len(activations),
        "n_heads": len(gamma),
        "learned_gamma": {
            "h3_epsilon_sq": learned_eps,
            "layer_sizes": [int(np.sum(gamma == k)) for k in range(len(LAYER_NAMES))],
        },
        "null_groupings": null_results,
        "permutation_test": {
            "n_null_total": len(all_null_eps),
            "null_mean": float(np.mean(all_null_eps)),
            "null_std": float(np.std(all_null_eps)),
            "learned_percentile": percentile,
            "permutation_p_value": perm_p,
            "significant": perm_p < 0.01,
        },
        "conclusion": {
            "learned_gamma_nontrivial": perm_p < 0.01,
            "interpretation": (
                "Learned γ provides statistically non-trivial diagnostic decomposition."
                if perm_p < 0.01
                else "Learned γ does not significantly outperform null groupings."
            ),
        },
    }

    # Print summary
    print(f"\n{'='*60}")
    print(f"Null Grouping Controls Summary ({model_name})")
    print(f"{'='*60}")
    print(f"  Learned γ ε²: {learned_eps:.4f}")
    for null_type, res in null_results.items():
        print(f"  {null_type} null: ε² = {res['mean_epsilon_sq']:.4f} ± {res['std_epsilon_sq']:.4f}")
    print(f"  Permutation p-value: {perm_p:.4f}")
    print(f"  Significant (p < 0.01): {perm_p < 0.01}")
    print(f"{'='*60}")

    # Save
    output_path = results_dir / f"{model_name}_null_grouping.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    return results


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "llama-7b"
    run_experiment_b(model_name)
