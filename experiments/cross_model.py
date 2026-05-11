"""
LCESA Cross-Model Comparison
=============================
Compares hypothesis test results across models (GPT-2, Llama-7b, etc.).
Generates comparison tables and summary statistics.
"""
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from experiments.config import RESULTS_DIR, LAYER_NAMES


def load_model_results(
    model_name: str,
    results_dir: Path = RESULTS_DIR,
) -> Dict:
    """Load hypothesis test JSON for a model."""
    path = results_dir / f"{model_name}_hypothesis_tests.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def load_model_curvature(
    model_name: str,
    results_dir: Path = RESULTS_DIR,
) -> pd.DataFrame:
    """Load curvature CSV for a model."""
    path = results_dir / f"{model_name}_curvature.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def build_comparison_table(
    model_names: List[str],
    results_dir: Path = RESULTS_DIR,
) -> pd.DataFrame:
    """Build a comparison table of hypothesis test results across models.

    Returns DataFrame with columns: model, hypothesis, passed, p_value, effect_size, notes
    """
    rows = []
    for model_name in model_names:
        results = load_model_results(model_name, results_dir)
        for h_name in ["H1", "H2", "H3", "H4", "H5", "H6"]:
            h_result = results.get(h_name, {})
            if not h_result or h_result.get("skipped"):
                rows.append({
                    "model": model_name,
                    "hypothesis": h_name,
                    "passed": None,
                    "p_value": None,
                    "effect_size": None,
                    "notes": h_result.get("reason", "no data"),
                })
                continue

            passed = h_result.get("passed")
            p_val = h_result.get("p_value")

            # Extract effect size depending on hypothesis
            effect = None
            if h_name == "H1":
                effect = h_result.get("aggregate", {}).get("mean_ratio")
            elif h_name == "H2":
                effect = h_result.get("cohens_d")
            elif h_name == "H3":
                eps = [v.get("epsilon_squared", 0) for v in h_result.get("per_layer", {}).values()
                       if isinstance(v, dict) and "epsilon_squared" in v]
                effect = np.mean(eps) if eps else None
            elif h_name == "H6":
                effect = h_result.get("curvature_cohens_d")

            rows.append({
                "model": model_name,
                "hypothesis": h_name,
                "passed": passed,
                "p_value": p_val,
                "effect_size": effect,
                "notes": "",
            })

    return pd.DataFrame(rows)


def build_curvature_comparison(
    model_names: List[str],
    results_dir: Path = RESULTS_DIR,
) -> pd.DataFrame:
    """Compare per-scenario mean curvature_total across models."""
    rows = []
    for model_name in model_names:
        df = load_model_curvature(model_name, results_dir)
        if df.empty:
            continue
        for scenario in sorted(df["scenario"].unique()):
            sdf = df[df["scenario"] == scenario]
            rows.append({
                "model": model_name,
                "scenario": scenario,
                "mean_total": float(sdf["curvature_total"].mean()),
                "std_total": float(sdf["curvature_total"].std()),
                "n": len(sdf),
            })
    return pd.DataFrame(rows)


def generate_comparison_report(
    model_names: List[str],
    output_path: Path = None,
    results_dir: Path = RESULTS_DIR,
) -> str:
    """Generate a markdown comparison report across models."""
    hyp_table = build_comparison_table(model_names, results_dir)
    curv_table = build_curvature_comparison(model_names, results_dir)

    lines = [
        "# Cross-Model Comparison Report",
        "",
        f"Models: {', '.join(model_names)}",
        "",
        "## Hypothesis Tests",
        "",
    ]

    if not hyp_table.empty:
        lines.append("| Model | Hypothesis | Passed | p-value | Effect Size |")
        lines.append("|-------|-----------|--------|---------|-------------|")
        for _, row in hyp_table.iterrows():
            if row["passed"] is not None and not pd.isna(row["passed"]):
                passed_str = str(row["passed"])
            else:
                passed_str = "N/A"
            if isinstance(row["p_value"], (int, float)) and not pd.isna(row["p_value"]):
                p_str = f"{row['p_value']:.2e}"
            else:
                p_str = "N/A"
            if isinstance(row["effect_size"], (int, float)) and not pd.isna(row["effect_size"]):
                e_str = f"{row['effect_size']:.3f}"
            else:
                e_str = "N/A"
            lines.append(f"| {row['model']} | {row['hypothesis']} | {passed_str} | {p_str} | {e_str} |")
    else:
        lines.append("No hypothesis test results available.")

    lines.extend(["", "## Mean Curvature Total by Scenario", ""])

    if not curv_table.empty:
        lines.append("| Model | Scenario | Mean | Std | n |")
        lines.append("|-------|---------|------|-----|---|")
        for _, row in curv_table.iterrows():
            lines.append(
                f"| {row['model']} | {row['scenario']} | "
                f"{row['mean_total']:.2f} | {row['std_total']:.2f} | {row['n']} |"
            )
    else:
        lines.append("No curvature results available.")

    report = "\n".join(lines)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(report)
        print(f"Comparison report saved to {output_path}")

    return report


if __name__ == "__main__":
    import sys
    models = sys.argv[1:] if len(sys.argv) > 1 else ["gpt2"]
    report = generate_comparison_report(models)
    print(report)
