#!/usr/bin/env python3
"""
Experiment E: T0 Surface Separability Regression Analysis

Tests whether T0's higher holonomy deviation is explained by surface features
(length, lexical overlap, entity count) or reflects genuine geometric complexity.

Model: C = β₀ + β₁·Scenario + β₂·Length + β₃·LexicalOverlap + β₄·EntityCount + ε
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import re

# Paths
ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"
DATA_DIR = ROOT / "data"


def load_curvature_data():
    """Load Llama-7b curvature data."""
    df = pd.read_csv(RESULTS_DIR / "llama-7b_curvature.csv")
    return df


def load_stimuli():
    """Load stimuli conversations."""
    with open(DATA_DIR / "stimuli.json", "r") as f:
        stimuli = json.load(f)
    return stimuli


def extract_surface_features(conversation):
    """
    Extract surface features from a conversation.

    Returns:
        total_length: total character count
        avg_turn_length: average turn length
        lexical_overlap: Jaccard similarity between consecutive turns
        entity_count: rough entity count (capitalized words)
    """
    events = conversation["events"]

    # Total length
    total_length = sum(len(e["content"]) for e in events)

    # Average turn length
    avg_turn_length = total_length / len(events) if events else 0

    # Lexical overlap between consecutive assistant turns
    assistant_contents = [e["content"].lower() for e in events if e["role"] == "assistant"]
    if len(assistant_contents) >= 2:
        words_sets = [set(re.findall(r'\w+', c)) for c in assistant_contents]
        overlaps = []
        for i in range(len(words_sets) - 1):
            intersection = len(words_sets[i] & words_sets[i+1])
            union = len(words_sets[i] | words_sets[i+1])
            if union > 0:
                overlaps.append(intersection / union)
        lexical_overlap = np.mean(overlaps) if overlaps else 0.0
    else:
        lexical_overlap = 0.0

    # Entity count (rough: capitalized words not at sentence start)
    entity_count = 0
    for e in events:
        sentences = e["content"].split('.')
        for sent in sentences:
            words = sent.strip().split()
            for i, w in enumerate(words):
                if i > 0 and w[0:1].isupper() and w.isalpha():
                    entity_count += 1

    return {
        "total_length": total_length,
        "avg_turn_length": avg_turn_length,
        "lexical_overlap": lexical_overlap,
        "entity_count": entity_count
    }


def build_regression_data():
    """Build regression dataset combining curvature and surface features."""
    # Load curvature
    curvature_df = load_curvature_data()

    # Load stimuli
    stimuli = load_stimuli()

    # Build surface features lookup: scenario -> index -> features
    # conversation_id format: T0/test/0, T1/train/5, etc.
    surface_lookup = {}
    for scenario, data in stimuli.items():
        surface_lookup[scenario] = {}
        for idx, conv in enumerate(data["grouping"]):
            features = extract_surface_features(conv)
            surface_lookup[scenario][idx] = features

    # Merge curvature with surface features
    records = []
    for _, row in curvature_df.iterrows():
        scenario = row["scenario"]
        conv_id = row["conversation_id"]  # e.g., "T0/test/0"

        # Get curvature total at layer 31 (last layer for Llama-7b)
        if row["layer"] == 31:
            # Extract index from conversation_id
            parts = conv_id.split("/")
            if len(parts) >= 3:
                try:
                    idx = int(parts[2])
                    if scenario in surface_lookup and idx in surface_lookup[scenario]:
                        features = surface_lookup[scenario][idx]
                        records.append({
                            "scenario": scenario,
                            "conversation_id": conv_id,
                            "curvature_total": row["curvature_total"],
                            "curvature_min": row["curvature_min"],
                            "curvature_nar": row["curvature_nar"],
                            "curvature_soc": row["curvature_soc"],
                            "curvature_mor": row["curvature_mor"],
                            "curvature_pos": row["curvature_pos"],
                            **features
                        })
                except ValueError:
                    continue

    return pd.DataFrame(records)


def run_regression(df):
    """
    Run OLS regression: C = β₀ + β₁·Scenario + β₂·Length + β₃·LexicalOverlap + β₄·EntityCount + ε

    Returns regression results and partial R² for scenario after controlling for surface features.
    """
    import statsmodels.api as sm
    from statsmodels.formula.api import ols

    # Create scenario dummies (T1 as reference)
    df["is_T0"] = (df["scenario"] == "T0").astype(int)
    df["is_T2"] = (df["scenario"] == "T2").astype(int)
    df["is_T3"] = (df["scenario"] == "T3").astype(int)
    df["is_T4"] = (df["scenario"] == "T4").astype(int)
    df["is_T5"] = (df["scenario"] == "T5").astype(int)

    # Normalize surface features
    for col in ["total_length", "avg_turn_length", "lexical_overlap", "entity_count"]:
        df[f"{col}_z"] = (df[col] - df[col].mean()) / df[col].std()

    # Model 1: Scenario only
    formula1 = "curvature_total ~ is_T0 + is_T2 + is_T3 + is_T4 + is_T5"
    model1 = ols(formula1, data=df).fit()

    # Model 2: Surface features only
    formula2 = "curvature_total ~ total_length_z + avg_turn_length_z + lexical_overlap_z + entity_count_z"
    model2 = ols(formula2, data=df).fit()

    # Model 3: Full model (scenario + surface features)
    formula3 = "curvature_total ~ is_T0 + is_T2 + is_T3 + is_T4 + is_T5 + total_length_z + avg_turn_length_z + lexical_overlap_z + entity_count_z"
    model3 = ols(formula3, data=df).fit()

    # Partial R² for scenario after controlling for surface features
    # = (RSS_reduced - RSS_full) / RSS_reduced
    # where reduced = surface only, full = surface + scenario
    rss_reduced = model2.resid.values
    rss_full = model3.resid.values
    ss_reduced = np.sum(rss_reduced ** 2)
    ss_full = np.sum(rss_full ** 2)
    partial_r2 = (ss_reduced - ss_full) / ss_reduced if ss_reduced > 0 else 0

    return {
        "model1_scenario_only": {
            "r2": model1.rsquared,
            "adj_r2": model1.rsquared_adj,
            "f_pvalue": model1.f_pvalue,
            "params": model1.params.to_dict(),
            "pvalues": model1.pvalues.to_dict()
        },
        "model2_surface_only": {
            "r2": model2.rsquared,
            "adj_r2": model2.rsquared_adj,
            "f_pvalue": model2.f_pvalue,
            "params": model2.params.to_dict(),
            "pvalues": model2.pvalues.to_dict()
        },
        "model3_full": {
            "r2": model3.rsquared,
            "adj_r2": model3.rsquared_adj,
            "f_pvalue": model3.f_pvalue,
            "params": model3.params.to_dict(),
            "pvalues": model3.pvalues.to_dict()
        },
        "partial_r2_scenario_given_surface": partial_r2
    }


def compute_descriptive_surface(df):
    """Compute surface feature statistics by scenario."""
    results = {}
    for scenario in sorted(df["scenario"].unique()):
        subset = df[df["scenario"] == scenario]
        results[scenario] = {
            "n": len(subset),
            "total_length_mean": subset["total_length"].mean(),
            "total_length_std": subset["total_length"].std(),
            "lexical_overlap_mean": subset["lexical_overlap"].mean(),
            "lexical_overlap_std": subset["lexical_overlap"].std(),
            "entity_count_mean": subset["entity_count"].mean(),
            "entity_count_std": subset["entity_count"].std()
        }
    return results


def main():
    print("=" * 70)
    print("Experiment E: T0 Surface Separability Regression Analysis")
    print("=" * 70)

    # Build dataset
    print("\n[1/4] Building regression dataset...")
    df = build_regression_data()
    print(f"  Total observations: {len(df)}")
    print(f"  Scenarios: {sorted(df['scenario'].unique())}")
    print(f"  Observations per scenario: {df['scenario'].value_counts().to_dict()}")

    # Descriptive statistics of surface features
    print("\n[2/4] Surface feature descriptive statistics...")
    surface_stats = compute_descriptive_surface(df)
    for scenario, stats_dict in surface_stats.items():
        print(f"\n  {scenario} (n={stats_dict['n']}):")
        print(f"    Length: {stats_dict['total_length_mean']:.0f} ± {stats_dict['total_length_std']:.0f}")
        print(f"    Lexical overlap: {stats_dict['lexical_overlap_mean']:.3f} ± {stats_dict['lexical_overlap_std']:.3f}")
        print(f"    Entity count: {stats_dict['entity_count_mean']:.1f} ± {stats_dict['entity_count_std']:.1f}")

    # Run regressions
    print("\n[3/4] Running regression models...")
    results = run_regression(df)

    print("\n  Model 1 (Scenario only):")
    print(f"    R² = {results['model1_scenario_only']['r2']:.4f}")
    print(f"    Adj R² = {results['model1_scenario_only']['adj_r2']:.4f}")
    print(f"    F p-value = {results['model1_scenario_only']['f_pvalue']:.2e}")

    print("\n  Model 2 (Surface features only):")
    print(f"    R² = {results['model2_surface_only']['r2']:.4f}")
    print(f"    Adj R² = {results['model2_surface_only']['adj_r2']:.4f}")
    print(f"    F p-value = {results['model2_surface_only']['f_pvalue']:.2e}")

    print("\n  Model 3 (Full: Scenario + Surface):")
    print(f"    R² = {results['model3_full']['r2']:.4f}")
    print(f"    Adj R² = {results['model3_full']['adj_r2']:.4f}")
    print(f"    F p-value = {results['model3_full']['f_pvalue']:.2e}")

    print(f"\n  Partial R² (Scenario | Surface) = {results['partial_r2_scenario_given_surface']:.4f}")

    # T0 coefficient in full model
    t0_coef = results['model3_full']['params'].get('is_T0', 0)
    t0_pval = results['model3_full']['pvalues'].get('is_T0', 1)
    print(f"\n  T0 coefficient (full model): β = {t0_coef:.4f}, p = {t0_pval:.2e}")

    # Save results
    print("\n[4/4] Saving results...")
    output = {
        "description": "T0 surface separability regression analysis",
        "n_observations": len(df),
        "scenarios": sorted(df["scenario"].unique()),
        "regression_results": results,
        "surface_descriptive": surface_stats
    }

    output_path = RESULTS_DIR / "llama-7b_t0_separability.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Results saved to: {output_path}")

    # Interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    partial_r2 = results['partial_r2_scenario_given_surface']
    if partial_r2 > 0.1:
        print(f"\n  Scenario type explains {partial_r2*100:.1f}% of variance AFTER controlling")
        print(f"  for surface features. T0's higher holonomy is NOT fully explained by")
        print(f"  surface features --- genuine geometric complexity difference.")
    elif partial_r2 > 0.01:
        print(f"\n  Scenario type explains {partial_r2*100:.1f}% of variance AFTER controlling")
        print(f"  for surface features. Partial support: some geometric complexity remains,")
        print(f"  but surface features explain substantial variance.")
    else:
        print(f"\n  Scenario type explains only {partial_r2*100:.1f}% of variance AFTER controlling")
        print(f"  for surface features. T0's higher holonomy may be largely explained by")
        print(f"  surface feature differences.")

    return results


if __name__ == "__main__":
    main()
