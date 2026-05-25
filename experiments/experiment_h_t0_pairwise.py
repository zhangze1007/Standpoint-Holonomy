"""
Experiment H: T0 Pairwise Comparisons
======================================
Computes pairwise Mann-Whitney U tests between T0 (factual retrieval)
and each individual failure scenario (T2, T3, T4, T5).

Uses existing curvature CSV data (no raw activations needed).

Outputs:
- Pairwise effect sizes (Cohen's d) for T0 vs each scenario
- Mann-Whitney U test statistics and p-values
- Per-block breakdown
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.config import RESULTS_DIR, LAYER_NAMES


def pairwise_comparison(
    values_a: np.ndarray,
    values_b: np.ndarray,
    label_a: str,
    label_b: str,
) -> dict:
    """Compute Mann-Whitney U and Cohen's d between two groups."""
    # Mann-Whitney U
    u_stat, p_val = stats.mannwhitneyu(values_a, values_b, alternative='two-sided')

    # Cohen's d
    pooled_std = np.sqrt(
        ((len(values_a) - 1) * values_a.std(ddof=1) ** 2
         + (len(values_b) - 1) * values_b.std(ddof=1) ** 2)
        / (len(values_a) + len(values_b) - 2)
    )
    cohens_d = (values_a.mean() - values_b.mean()) / pooled_std if pooled_std > 0 else 0.0

    # Rank-biserial correlation (effect size from U)
    n_a, n_b = len(values_a), len(values_b)
    r_rb = 1 - (2 * u_stat) / (n_a * n_b)

    return {
        "label_a": label_a,
        "label_b": label_b,
        "n_a": n_a,
        "n_b": n_b,
        "mean_a": float(values_a.mean()),
        "mean_b": float(values_b.mean()),
        "median_a": float(np.median(values_a)),
        "median_b": float(np.median(values_b)),
        "u_statistic": float(u_stat),
        "p_value": float(p_val),
        "cohens_d": float(cohens_d),
        "rank_biserial_r": float(r_rb),
        "significant_at_005": p_val < 0.05,
        "significant_at_001": p_val < 0.01,
    }


def run_t0_pairwise(model_name: str, results_dir: Path = RESULTS_DIR) -> dict:
    """Run T0 vs T2-T5 pairwise comparisons."""
    print(f"=== T0 Pairwise Comparisons ({model_name}) ===\n")

    # Load curvature data
    curvature_path = results_dir / f"{model_name}_curvature.csv"
    df = pd.read_csv(curvature_path)

    # Aggregate across layers: use mean curvature_total per conversation
    agg = df.groupby(["conversation_id", "scenario"])["curvature_total"].mean().reset_index()

    # Get T0 values
    t0_vals = agg[agg["scenario"] == "T0"]["curvature_total"].values
    print(f"T0: n={len(t0_vals)}, mean={t0_vals.mean():.2f}, median={np.median(t0_vals):.2f}")

    results = {
        "model": model_name,
        "n_conversations": len(agg),
        "t0_stats": {
            "n": len(t0_vals),
            "mean": float(t0_vals.mean()),
            "std": float(t0_vals.std()),
            "median": float(np.median(t0_vals)),
        },
        "pairwise": {},
    }

    # Pairwise comparisons: T0 vs each of T1-T5
    for scenario in ["T1", "T2", "T3", "T4", "T5"]:
        scenario_vals = agg[agg["scenario"] == scenario]["curvature_total"].values
        print(f"{scenario}: n={len(scenario_vals)}, mean={scenario_vals.mean():.2f}, median={np.median(scenario_vals):.2f}")

        comparison = pairwise_comparison(t0_vals, scenario_vals, "T0", scenario)
        results["pairwise"][f"T0_vs_{scenario}"] = comparison

        sig = "***" if comparison["p_value"] < 0.001 else "**" if comparison["p_value"] < 0.01 else "*" if comparison["p_value"] < 0.05 else "ns"
        print(f"  T0 vs {scenario}: d={comparison['cohens_d']:+.3f}, U={comparison['u_statistic']:.0f}, p={comparison['p_value']:.2e} {sig}")

    # Per-block breakdown for T0 vs T2-T5
    print("\n--- Per-block breakdown ---")
    block_cols = [c for c in df.columns if c.startswith("curvature_") and c != "curvature_total"]
    results["per_block"] = {}

    for block_col in block_cols:
        block_name = block_col.replace("curvature_", "")
        block_agg = df.groupby(["conversation_id", "scenario"])[block_col].mean().reset_index()
        t0_block = block_agg[block_agg["scenario"] == "T0"][block_col].values

        block_results = {}
        for scenario in ["T2", "T3", "T4", "T5"]:
            sc_block = block_agg[block_agg["scenario"] == scenario][block_col].values
            comp = pairwise_comparison(t0_block, sc_block, "T0", scenario)
            block_results[f"T0_vs_{scenario}"] = {
                "cohens_d": comp["cohens_d"],
                "p_value": comp["p_value"],
            }
            sig = "***" if comp["p_value"] < 0.001 else "**" if comp["p_value"] < 0.01 else "*" if comp["p_value"] < 0.05 else "ns"
            print(f"  {block_name}: T0 vs {scenario} d={comp['cohens_d']:+.3f} {sig}")

        results["per_block"][block_name] = block_results

    # Summary
    print(f"\n{'='*60}")
    print(f"T0 Pairwise Summary ({model_name})")
    print(f"{'='*60}")
    for key, comp in results["pairwise"].items():
        direction = "T0 >" if comp["cohens_d"] > 0 else "T0 <"
        sig = "✓" if comp["significant_at_005"] else "✗"
        print(f"  {key}: d={comp['cohens_d']:+.3f} ({direction} {comp['label_b']}) [{sig}]")
    print(f"{'='*60}")

    # Save
    output_path = results_dir / f"{model_name}_t0_pairwise.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    return results


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "llama-7b"
    run_t0_pairwise(model_name)
