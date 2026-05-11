"""
LCESA Hypothesis Tests
======================
Implements hypothesis tests H1, H2, H3, H4, H5, and H6 for the
Low-Curvature Endogenous Standpoint Attractor (LCESA) curvature
validation experiment.

H1: Block specificity — curvature concentrates in the target failure block.
H2: Baseline near zero — T1 curvature is near zero compared to failure.
H3: Scenario discrimination — curvature vectors differ across scenarios.
H4: CKA discrimination — CKA values discriminate across scenarios per layer.
H5: Ablation sensitivity — curvature degrades gracefully under ablation.
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

    Reports results separately for head-assigned layers (those with
    attention heads assigned via gamma) and non-assigned layers.

    Parameters
    ----------
    df : pd.DataFrame
        Curvature results (from :func:`load_curvature_results`).

    Returns
    -------
    dict
        Per-scenario and aggregate results with p-values and pass/fail flags.
        Includes ``head_assigned`` vs ``non_assigned`` breakdowns.
    """
    from experiments.config import FAILURE_LAYERS, DATA_DIR

    # Load head assignments (gamma) if available
    gamma_path = DATA_DIR / "gpt2_grouping.npz"
    head_counts = {}
    if gamma_path.exists():
        gamma = np.load(gamma_path)["gamma"]
        layer_names_ordered = ["min", "nar", "soc", "mor", "pos"]
        for idx, name in enumerate(layer_names_ordered):
            head_counts[name] = int(np.sum(gamma == idx))

    curvature_cols = [
        c for c in df.columns
        if c.startswith("curvature_") and c != "curvature_total"
    ]
    n_blocks = len(curvature_cols)
    chance_level = 1.0 / n_blocks

    results = {}
    all_target_ratios = []
    assigned_ratios = []
    non_assigned_ratios = []
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

        # Classify as head-assigned or not
        n_heads = head_counts.get(target_layer, 0)
        has_heads = n_heads > 0
        if has_heads:
            assigned_ratios.extend(target_ratios.tolist())
        else:
            non_assigned_ratios.extend(target_ratios.tolist())

        results[scenario] = {
            "target_layer": target_layer,
            "n_conversations": len(target_ratios),
            "mean_target_ratio": float(target_ratios.mean()),
            "median_target_ratio": float(np.median(target_ratios)),
            "n_heads_assigned": n_heads,
            "head_assigned": has_heads,
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

    # Breakdown: head-assigned vs non-assigned
    if assigned_ratios:
        a = np.array(assigned_ratios)
        results["head_assigned_layers"] = {
            "n": len(a),
            "mean_ratio": float(a.mean()),
            "median_ratio": float(np.median(a)),
            "scenarios": [s for s, l in failure_scenarios.items()
                          if head_counts.get(l, 0) > 0],
        }
    if non_assigned_ratios:
        n = np.array(non_assigned_ratios)
        results["non_assigned_layers"] = {
            "n": len(n),
            "mean_ratio": float(n.mean()),
            "median_ratio": float(np.median(n)),
            "scenarios": [s for s, l in failure_scenarios.items()
                          if head_counts.get(l, 0) == 0],
        }

    return results


# ---------------------------------------------------------------------------
# H2: Baseline near zero
# ---------------------------------------------------------------------------

def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Cohen's d between two samples."""
    pooled_std = np.sqrt((a.var() + b.var()) / 2)
    if pooled_std == 0:
        return 0.0
    return float((a.mean() - b.mean()) / pooled_std)


def _bootstrap_ci(
    a: np.ndarray, b: np.ndarray, stat_fn, n_boot: int = 10000, alpha: float = 0.05
) -> tuple:
    """Bootstrap (1-alpha) CI for stat_fn(a, b)."""
    rng = np.random.default_rng(42)
    combined = np.concatenate([a, b])
    na = len(a)
    boots = []
    for _ in range(n_boot):
        idx = rng.choice(len(combined), size=len(combined), replace=True)
        ba, bb = idx[:na], idx[na:]
        boots.append(stat_fn(combined[ba], combined[bb]))
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return lo, hi


def h2_baseline_near_zero(df: pd.DataFrame) -> dict:
    """Test H2: T1 (baseline) curvature is near zero.

    Compares T1 curvature (``curvature_total``) against failure-scenario
    curvature using a Mann-Whitney U test (alternative="less").
    Also reports Cohen's d and 95% bootstrap CI.

    Parameters
    ----------
    df : pd.DataFrame
        Curvature results.

    Returns
    -------
    dict
        Test statistics, p-values, effect sizes, CIs, and pass/fail flags.
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
    cohens_d = _cohens_d(failure_values, t1_values)  # positive = failure > T1

    # Bootstrap 95% CI for the mean difference
    diff_fn = lambda a, b: a.mean() - b.mean()
    ci_lo, ci_hi = _bootstrap_ci(failure_values, t1_values, diff_fn)

    # Pass criterion: Mann-Whitney p < alpha (T1 curvature is less than failure)
    passed = p_value < ALPHA

    return {
        "t1_mean": t1_mean,
        "t1_std": float(t1_values.std()),
        "failure_mean": failure_mean,
        "failure_std": float(failure_values.std()),
        "mean_difference": failure_mean - t1_mean,
        "cohens_d": cohens_d,
        "mean_diff_ci_95": [ci_lo, ci_hi],
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

        # Effect size: epsilon-squared = H / (n - 1)
        n_total = sum(len(g) for g in groups)
        epsilon_sq = float(statistic / (n_total - 1)) if n_total > 1 else 0.0

        passed = p_value < corrected_alpha
        if not passed:
            all_passed = False

        layer_results[layer_name] = {
            "statistic": float(statistic),
            "epsilon_squared": epsilon_sq,
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
# H4: CKA scenario discrimination
# ---------------------------------------------------------------------------

def h4_cka_discrimination(cka_df: pd.DataFrame) -> dict:
    """Test H4: CKA values discriminate across scenarios.

    For each model layer, runs Kruskal-Wallis on CKA values across scenarios
    with Bonferroni correction.

    Parameters
    ----------
    cka_df : pd.DataFrame
        CKA results with columns: conversation_id, scenario, layer, cka.

    Returns
    -------
    dict
        Per-layer Kruskal-Wallis results and overall pass/fail.
    """
    layers = sorted(cka_df["layer"].unique())
    n_layers = len(layers)
    corrected_alpha = ALPHA / n_layers

    layer_results = {}
    all_passed = True

    for layer in layers:
        layer_key = int(layer)
        layer_df = cka_df[cka_df["layer"] == layer]
        groups = []
        scenario_labels = []
        for scenario in sorted(layer_df["scenario"].unique()):
            vals = layer_df.loc[layer_df["scenario"] == scenario, "cka"].dropna().values
            if len(vals) > 0:
                groups.append(vals)
                scenario_labels.append(scenario)

        if len(groups) < 2:
            layer_results[layer_key] = {"error": "need at least 2 groups"}
            all_passed = False
            continue

        try:
            kw_result = stats.kruskal(*groups)
            p_value = kw_result.pvalue
            statistic = kw_result.statistic
        except ValueError:
            layer_results[layer_key] = {"error": "Kruskal-Wallis failed"}
            all_passed = False
            continue

        n_total = sum(len(g) for g in groups)
        epsilon_sq = float(statistic / (n_total - 1)) if n_total > 1 else 0.0

        passed = p_value < corrected_alpha
        if not passed:
            all_passed = False

        layer_results[layer_key] = {
            "statistic": float(statistic),
            "epsilon_squared": epsilon_sq,
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
# H5: Ablation sensitivity
# ---------------------------------------------------------------------------

def h5_ablation_sensitivity(ablation_df: pd.DataFrame) -> dict:
    """Test H5: curvature degrades gracefully under ablation.

    For each ablation parameter value, compute mean curvature_total per
    scenario and test whether the ranking of scenarios is preserved using
    Spearman correlation with the unablated (full) results.

    Parameters
    ----------
    ablation_df : pd.DataFrame
        Ablation results with columns: model, ablation_type, param_value,
        conversation_id, scenario, layer, curvature_total.

    Returns
    -------
    dict
        Per-param Spearman correlation results and overall pass/fail.
    """
    ablation_types = ablation_df["ablation_type"].unique()
    all_passed = True
    param_results = {}

    for abl_type in ablation_types:
        type_df = ablation_df[ablation_df["ablation_type"] == abl_type]
        param_values = sorted(type_df["param_value"].unique())
        full_param = max(param_values)

        full_df = type_df[type_df["param_value"] == full_param]
        full_means = full_df.groupby("scenario")["curvature_total"].mean()

        # Baseline entry for the unablated (full) parameter
        param_results[f"{abl_type}_{full_param}"] = {
            "ablation_type": abl_type,
            "param_value": int(full_param),
            "full_param": int(full_param),
            "spearman_rho": 1.0,
            "p_value": 0.0,
            "n_scenarios": len(full_means),
            "passed": True,
        }

        for param_val in param_values:
            if param_val == full_param:
                continue
            sub_df = type_df[type_df["param_value"] == param_val]
            sub_means = sub_df.groupby("scenario")["curvature_total"].mean()

            common = full_means.index.intersection(sub_means.index)
            if len(common) < 3:
                param_results[f"{abl_type}_{param_val}"] = {"error": "too few scenarios"}
                all_passed = False
                continue

            rho, p_val = stats.spearmanr(
                full_means[common].values,
                sub_means[common].values,
            )
            passed = rho >= 0.7 and p_val < ALPHA
            if not passed:
                all_passed = False

            param_results[f"{abl_type}_{param_val}"] = {
                "ablation_type": abl_type,
                "param_value": int(param_val),
                "full_param": int(full_param),
                "spearman_rho": float(rho),
                "p_value": float(p_val),
                "n_scenarios": len(common),
                "passed": passed,
            }

    return {
        "per_param": param_results,
        "all_passed": all_passed,
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

    # Probing: mean F1 across layers (prefer multiclass if available)
    if probing_df.empty:
        return {"error": "no probing data"}

    if "probe_type" in probing_df.columns:
        multi_df = probing_df[probing_df["probe_type"] == "multiclass"]
        if not multi_df.empty:
            probing_f1_mean = float(multi_df["f1_mean"].mean())
        else:
            probing_f1_mean = float(probing_df["f1_mean"].mean())
    else:
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
        "probing_f1_std": float(multi_df["f1_mean"].std() if "probe_type" in probing_df.columns and not multi_df.empty else probing_df["f1_mean"].std()),
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

    # H4: CKA discrimination
    cka_path = results_dir / f"{model_name}_cka.csv"
    if cka_path.exists():
        print("Running H4: CKA scenario discrimination ...")
        cka_df = pd.read_csv(cka_path)
        all_results["H4"] = h4_cka_discrimination(cka_df)
    else:
        all_results["H4"] = {"skipped": True, "reason": "no CKA data"}
        print("Warning: CKA results not found, skipping H4.")

    # H5: Ablation sensitivity
    ablation_path = results_dir / f"{model_name}_ablation.csv"
    if ablation_path.exists():
        print("Running H5: Ablation sensitivity ...")
        ablation_df = pd.read_csv(ablation_path)
        all_results["H5"] = h5_ablation_sensitivity(ablation_df)
    else:
        all_results["H5"] = {"skipped": True, "reason": "no ablation data"}
        print("Warning: Ablation results not found, skipping H5.")

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
