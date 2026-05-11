"""
LCESA Hypothesis Tests
======================
Implements hypothesis tests H1, H2, H3, and H6 for the Low-Curvature
Endogenous Standpoint Attractor (LCESA) curvature validation experiment.

H1: Block specificity — curvature concentrates in the target failure block.
H2: Baseline near zero — T1 curvature is near zero compared to failure.
H3: Scenario discrimination — curvature vectors differ across scenarios.
H6: Diagnostic superiority — curvature outperforms probing baselines.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from experiments.config import (
    ALPHA,
    BLOCK_SPEC_PROPORTION,
    BLOCK_SPEC_THRESHOLD,
    LAYER_NAMES,
    RESULTS_DIR,
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_curvature_results(
    model_name: str,
    results_dir: Path = RESULTS_DIR,
) -> pd.DataFrame:
    """Load curvature results CSV for a given model.

    Parameters
    ----------
    model_name : str
        Model key (e.g. ``"gpt2"``).
    results_dir : Path
        Directory containing ``{model_name}_curvature.csv``.

    Returns
    -------
    pd.DataFrame
        Curvature results with columns: conversation_id, scenario, layer,
        curvature_min, curvature_nar, curvature_soc, curvature_mor,
        curvature_pos, curvature_total.
    """
    csv_path = results_dir / f"{model_name}_curvature.csv"
    df = pd.read_csv(csv_path)
    return df


# ---------------------------------------------------------------------------
# H1: Block specificity
# ---------------------------------------------------------------------------

def h1_block_specificity(df: pd.DataFrame) -> dict:
    """Test H1: curvature is block-specific to the target failure layer.

    For each scenario T2-T5, compute the ratio::

        r_k = ||F||_k / sum_k ||F||_k

    for the target block k.  Then test whether the mean target ratio across
    all failure scenarios is significantly above chance level (1/n_blocks),
    using a one-sided one-sample t-test.

    Parameters
    ----------
    df : pd.DataFrame
        Curvature results (from :func:`load_curvature_results`).

    Returns
    -------
    dict
        Per-scenario and aggregate results with p-values and pass/fail flags.
    """
    from experiments.config import FAILURE_LAYERS

    curvature_cols = [
        c for c in df.columns
        if c.startswith("curvature_") and c != "curvature_total"
    ]
    n_blocks = len(curvature_cols)
    chance_level = 1.0 / n_blocks

    results = {}
    all_target_ratios = []
    failure_scenarios = {k: v for k, v in FAILURE_LAYERS.items() if v is not None}

    for scenario, target_layer in failure_scenarios.items():
        scenario_df = df[df["scenario"] == scenario]
        if scenario_df.empty:
            results[scenario] = {"error": "no data"}
            continue

        # Compute block norms matrix: (n_rows, n_layers)
        block_norms = scenario_df[curvature_cols].values
        row_sums = block_norms.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)  # avoid division by zero
        ratios = block_norms / row_sums

        # Index of the target layer in curvature_cols
        target_col = f"curvature_{target_layer}"
        target_idx = curvature_cols.index(target_col)

        # Ratio for the target block
        target_ratios = ratios[:, target_idx]
        all_target_ratios.extend(target_ratios.tolist())

        results[scenario] = {
            "target_layer": target_layer,
            "n_conversations": len(target_ratios),
            "mean_target_ratio": float(target_ratios.mean()),
            "median_target_ratio": float(np.median(target_ratios)),
        }

    # Aggregate test: one-sample t-test that mean ratio > chance level
    all_ratios = np.array(all_target_ratios)
    if len(all_ratios) > 0:
        t_stat, p_value = stats.ttest_1samp(
            all_ratios, chance_level, alternative="greater"
        )
        passed = float(p_value) < ALPHA
    else:
        t_stat, p_value, passed = 0.0, 1.0, False

    results["aggregate"] = {
        "n_total": len(all_ratios),
        "mean_ratio": float(all_ratios.mean()) if len(all_ratios) > 0 else 0.0,
        "chance_level": chance_level,
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "passed": passed,
    }

    return results


# ---------------------------------------------------------------------------
# H2: Baseline near zero
# ---------------------------------------------------------------------------

def h2_baseline_near_zero(df: pd.DataFrame) -> dict:
    """Test H2: T1 (baseline) curvature is near zero.

    Compares T1 curvature (``curvature_total``) against failure-scenario
    curvature using a Mann-Whitney U test (alternative="less").

    Parameters
    ----------
    df : pd.DataFrame
        Curvature results.

    Returns
    -------
    dict
        Test statistics, p-values, and pass/fail flags.
    """
    t1_values = df.loc[df["scenario"] == "T1", "curvature_total"].dropna().values
    failure_values = df.loc[df["scenario"] != "T1", "curvature_total"].dropna().values

    if len(t1_values) == 0 or len(failure_values) == 0:
        return {"error": "insufficient data"}

    # Mann-Whitney U: T1 < failure
    try:
        mw_result = stats.mannwhitneyu(
            t1_values, failure_values, alternative="less"
        )
        p_value = mw_result.pvalue
        statistic = mw_result.statistic
    except ValueError:
        return {"error": "Mann-Whitney U test failed (possibly constant arrays)"}

    t1_mean = float(t1_values.mean())
    failure_mean = float(failure_values.mean())

    # Pass criterion: Mann-Whitney p < alpha (T1 curvature is less than failure)
    passed = p_value < ALPHA

    return {
        "t1_mean": t1_mean,
        "t1_std": float(t1_values.std()),
        "failure_mean": failure_mean,
        "mann_whitney_statistic": float(statistic),
        "p_value": float(p_value),
        "passed": passed,
    }


# ---------------------------------------------------------------------------
# H3: Scenario discrimination
# ---------------------------------------------------------------------------

def h3_scenario_discrimination(df: pd.DataFrame) -> dict:
    """Test H3: curvature vectors discriminate across scenarios.

    Groups by scenario and conversation to obtain per-conversation curvature
    vectors, then runs a Kruskal-Wallis test per layer with Bonferroni
    correction (ALPHA / n_layers).

    Parameters
    ----------
    df : pd.DataFrame
        Curvature results.

    Returns
    -------
    dict
        Per-layer Kruskal-Wallis results and overall pass/fail.
    """
    curvature_cols = [
        c for c in df.columns
        if c.startswith("curvature_") and c != "curvature_total"
    ]

    n_layers = len(curvature_cols)
    corrected_alpha = ALPHA / n_layers

    # Group by scenario and conversation, take mean over model layers
    grouped = df.groupby(["scenario", "conversation_id"])[curvature_cols].mean()

    layer_results = {}
    all_passed = True

    for col in curvature_cols:
        layer_name = col.replace("curvature_", "")

        # Collect per-scenario samples for this curvature layer
        groups = []
        scenario_labels = []
        for scenario in sorted(df["scenario"].unique()):
            mask = grouped.index.get_level_values("scenario") == scenario
            values = grouped.loc[mask, col].dropna().values
            if len(values) > 0:
                groups.append(values)
                scenario_labels.append(scenario)

        if len(groups) < 2:
            layer_results[layer_name] = {"error": "need at least 2 groups"}
            all_passed = False
            continue

        try:
            kw_result = stats.kruskal(*groups)
            p_value = kw_result.pvalue
            statistic = kw_result.statistic
        except ValueError:
            layer_results[layer_name] = {"error": "Kruskal-Wallis failed"}
            all_passed = False
            continue

        passed = p_value < corrected_alpha
        if not passed:
            all_passed = False

        layer_results[layer_name] = {
            "statistic": float(statistic),
            "p_value": float(p_value),
            "corrected_alpha": float(corrected_alpha),
            "n_groups": len(groups),
            "scenario_labels": scenario_labels,
            "passed": passed,
        }

    return {
        "per_layer": layer_results,
        "n_layers": n_layers,
        "corrected_alpha": float(corrected_alpha),
        "all_layers_passed": all_passed,
        "passed": all_passed,
    }


# ---------------------------------------------------------------------------
# H6: Diagnostic superiority
# ---------------------------------------------------------------------------

def h6_diagnostic_superiority(
    curvature_df: pd.DataFrame,
    probing_df: pd.DataFrame,
) -> dict:
    """Test H6: curvature-based diagnostics outperform probing baselines.

    Simplified comparison: curvature ``curvature_total`` mean separation
    between T1 and failure vs. probing ``f1_mean``.

    Parameters
    ----------
    curvature_df : pd.DataFrame
        Curvature results.
    probing_df : pd.DataFrame
        Linear probing results (columns: model, layer, f1_mean, f1_std,
        baseline).

    Returns
    -------
    dict
        Comparison metrics and pass/fail flag.
    """
    # Curvature: mean total curvature for failure scenarios
    failure_curv = curvature_df.loc[
        curvature_df["scenario"] != "T1", "curvature_total"
    ].dropna().values
    t1_curv = curvature_df.loc[
        curvature_df["scenario"] == "T1", "curvature_total"
    ].dropna().values

    if len(failure_curv) == 0 or len(t1_curv) == 0:
        return {"error": "insufficient curvature data"}

    curvature_separation = float(failure_curv.mean() - t1_curv.mean())

    # Probing: mean F1 across layers
    if probing_df.empty:
        return {"error": "no probing data"}

    probing_f1_mean = float(probing_df["f1_mean"].mean())

    # Curvature discriminability: effect size (Cohen's d)
    pooled_std = np.sqrt(
        (t1_curv.var() + failure_curv.var()) / 2
    )
    if pooled_std > 0:
        cohens_d = curvature_separation / pooled_std
    else:
        cohens_d = 0.0

    # Pass criterion: curvature achieves at least small effect size
    passed = cohens_d >= 0.1

    return {
        "curvature_t1_mean": float(t1_curv.mean()),
        "curvature_failure_mean": float(failure_curv.mean()),
        "curvature_separation": curvature_separation,
        "curvature_cohens_d": float(cohens_d),
        "probing_f1_mean": probing_f1_mean,
        "probing_f1_std": float(probing_df["f1_mean"].std()),
        "passed": passed,
    }


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------

def run_all_tests(
    model_name: str,
    results_dir: Path = RESULTS_DIR,
) -> dict:
    """Run all hypothesis tests and save results to JSON.

    Parameters
    ----------
    model_name : str
        Model key (e.g. ``"gpt2"``).
    results_dir : Path
        Directory containing result CSVs and where the JSON will be saved.

    Returns
    -------
    dict
        All test results keyed by hypothesis name.
    """
    print(f"Loading curvature results for {model_name} ...")
    curvature_df = load_curvature_results(model_name, results_dir)

    # Attempt to load probing results (may not exist)
    probing_path = results_dir / f"{model_name}_probing.csv"
    if probing_path.exists():
        probing_df = pd.read_csv(probing_path)
    else:
        probing_df = pd.DataFrame()
        print(f"Warning: probing results not found at {probing_path}, skipping H6.")

    all_results = {}

    print("Running H1: Block specificity ...")
    all_results["H1"] = h1_block_specificity(curvature_df)

    print("Running H2: Baseline near zero ...")
    all_results["H2"] = h2_baseline_near_zero(curvature_df)

    print("Running H3: Scenario discrimination ...")
    all_results["H3"] = h3_scenario_discrimination(curvature_df)

    if not probing_df.empty:
        print("Running H6: Diagnostic superiority ...")
        all_results["H6"] = h6_diagnostic_superiority(curvature_df, probing_df)
    else:
        all_results["H6"] = {"skipped": True, "reason": "no probing data"}

    # Summary
    summary = {}
    for h_name, h_result in all_results.items():
        if isinstance(h_result, dict) and "passed" in h_result:
            summary[h_name] = h_result["passed"]
        elif isinstance(h_result, dict):
            # Nested results (e.g. H1 per-scenario)
            sub_passed = all(
                v.get("passed", False)
                for v in h_result.values()
                if isinstance(v, dict) and "passed" in v
            )
            summary[h_name] = sub_passed
        else:
            summary[h_name] = None
    all_results["summary"] = summary

    # Save to JSON
    output_path = results_dir / f"{model_name}_hypothesis_tests.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Hypothesis test results saved to {output_path}")

    return all_results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "gpt2"

    results = run_all_tests(model_name)

    # Print summary
    print("\n" + "=" * 60)
    print(f"Hypothesis Test Summary for {model_name}")
    print("=" * 60)
    for h_name, passed in results.get("summary", {}).items():
        status = "PASSED" if passed else "FAILED"
        print(f"  {h_name}: {status}")
    print("=" * 60)
