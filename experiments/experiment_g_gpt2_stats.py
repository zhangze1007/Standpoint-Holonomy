#!/usr/bin/env python3
"""
Experiment G: GPT-2 Descriptive Statistics and H1a

Computes descriptive statistics and H1a (block sensitivity) for GPT-2.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"


def load_curvature_data(model_name):
    """Load curvature data for a model."""
    path = RESULTS_DIR / f"{model_name}_curvature.csv"
    return pd.read_csv(path)


def compute_descriptive_stats(df):
    """Compute descriptive statistics by scenario at the last layer."""
    # Get last layer for each model
    max_layer = df["layer"].max()
    df_last = df[df["layer"] == max_layer]

    results = {}
    for scenario in sorted(df_last["scenario"].unique()):
        subset = df_last[df_last["scenario"] == scenario]
        values = subset["curvature_total"].values

        # Compute 95% CI for median using bootstrap
        n_bootstrap = 1000
        boot_medians = []
        for _ in range(n_bootstrap):
            boot_sample = np.random.choice(values, size=len(values), replace=True)
            boot_medians.append(np.median(boot_sample))
        ci_low = np.percentile(boot_medians, 2.5)
        ci_high = np.percentile(boot_medians, 97.5)

        results[scenario] = {
            "n": len(values),
            "median": float(np.median(values)),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "iqr": float(np.percentile(values, 75) - np.percentile(values, 25)),
            "ci_95_low": float(ci_low),
            "ci_95_high": float(ci_high),
        }

    return results


def compute_h1a(df, gamma_path):
    """Compute H1a: block sensitivity (above-mean ratio)."""
    # Load gamma
    grouping = np.load(gamma_path)
    gamma = grouping["gamma"]

    # Get last layer
    max_layer = df["layer"].max()
    df_last = df[df["layer"] == max_layer]

    results = {}
    for scenario in ["T2", "T3", "T4", "T5"]:
        subset = df_last[df_last["scenario"] == scenario]

        # For each conversation, check if the target block has above-mean curvature
        target_map = {"T2": "nar", "T3": "mor", "T4": "soc", "T5": "pos"}
        target_block = target_map[scenario]

        above_mean_count = 0
        total_count = len(subset)

        for _, row in subset.iterrows():
            block_values = {
                "min": row["curvature_min"],
                "nar": row["curvature_nar"],
                "soc": row["curvature_soc"],
                "mor": row["curvature_mor"],
                "pos": row["curvature_pos"],
            }
            mean_val = np.mean(list(block_values.values()))
            if block_values[target_block] > mean_val:
                above_mean_count += 1

        ratio = above_mean_count / total_count if total_count > 0 else 0
        results[scenario] = {
            "target_layer": target_block,
            "n_conversations": total_count,
            "above_mean_count": above_mean_count,
            "above_mean_ratio": ratio,
            "passed": ratio > 0.5  # Above chance (0.2 for 5 blocks)
        }

    return results


def main():
    print("=" * 60)
    print("GPT-2 Descriptive Statistics and H1a")
    print("=" * 60)

    # Load data
    print("\n[1/3] Loading GPT-2 curvature data...")
    df = load_curvature_data("gpt2")
    print(f"  Total rows: {len(df)}")
    print(f"  Layers: {sorted(df['layer'].unique())}")
    print(f"  Scenarios: {sorted(df['scenario'].unique())}")

    # Descriptive statistics
    print("\n[2/3] Computing descriptive statistics...")
    desc_stats = compute_descriptive_stats(df)

    print("\n  Per-scenario descriptive statistics (last layer):")
    for scenario, stats_dict in desc_stats.items():
        print(f"\n  {scenario} (n={stats_dict['n']}):")
        print(f"    Median: {stats_dict['median']:.2f} [{stats_dict['ci_95_low']:.2f}, {stats_dict['ci_95_high']:.2f}]")
        print(f"    Mean ± SD: {stats_dict['mean']:.2f} ± {stats_dict['std']:.2f}")
        print(f"    IQR: {stats_dict['iqr']:.2f}")

    # H1a
    print("\n[3/3] Computing H1a (block sensitivity)...")
    gamma_path = ROOT / "data" / "gpt2_grouping.npz"
    h1a_results = compute_h1a(df, gamma_path)

    print("\n  H1a Results:")
    for scenario, result in h1a_results.items():
        status = "PASS" if result["passed"] else "FAIL"
        print(f"    {scenario}: target={result['target_layer']}, "
              f"ratio={result['above_mean_ratio']:.3f} ({status})")

    # Save results
    output = {
        "model": "gpt2",
        "descriptive_stats": desc_stats,
        "h1a_results": h1a_results
    }

    output_path = RESULTS_DIR / "gpt2_descriptive_stats.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to: {output_path}")

    return output


if __name__ == "__main__":
    main()
