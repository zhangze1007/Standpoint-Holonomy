"""
Experiment D: Competitive Baselines
====================================
Computes three alternative measures and compares their scenario
discrimination against holonomy deviation.

Alternative measures:
1. Layer-wise attention distance: mean Frobenius distance between
   adjacent layer attention matrices
2. Mean attention entropy: mean Shannon entropy of attention distributions
3. Residual stream norm change: ||h_x3 - h_x1||_2

Requires raw activations (activations.npz).

Outputs:
- Kruskal-Wallis ε² for each alternative measure
- Comparison with holonomy deviation ε²
- Spearman correlation between alternatives and holonomy
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.config import RESULTS_DIR, CACHE_DIR, ALPHA


def attention_entropy(attn: np.ndarray) -> float:
    """Compute mean Shannon entropy across all heads and layers.

    Parameters
    ----------
    attn : np.ndarray, shape (n_layers, n_heads, n_events, n_events)
        Attention weights.

    Returns
    -------
    float : Mean entropy.
    """
    n_layers, n_heads, n_events, _ = attn.shape
    entropies = []
    for l in range(n_layers):
        for h in range(n_heads):
            p = attn[l, h]  # (n_events, n_events)
            # Each row is a distribution over attended positions
            for row in range(n_events):
                p_row = p[row]
                p_row = p_row[p_row > 0]  # avoid log(0)
                if len(p_row) > 0:
                    entropies.append(-np.sum(p_row * np.log2(p_row + 1e-10)))
    return float(np.mean(entropies)) if entropies else 0.0


def attention_distance(attn: np.ndarray) -> float:
    """Compute mean Frobenius distance between adjacent layer attention matrices.

    Parameters
    ----------
    attn : np.ndarray, shape (n_layers, n_heads, n_events, n_events)
        Attention weights.

    Returns
    -------
    float : Mean adjacent-layer distance.
    """
    n_layers = attn.shape[0]
    distances = []
    for l in range(n_layers - 1):
        # Average over heads for each layer
        attn_l = attn[l].mean(axis=0)  # (n_events, n_events)
        attn_l1 = attn[l + 1].mean(axis=0)
        dist = np.linalg.norm(attn_l - attn_l1, "fro")
        distances.append(float(dist))
    return float(np.mean(distances)) if distances else 0.0


def residual_norm_change(residuals: np.ndarray) -> float:
    """Compute ||h_x3 - h_x1||_2.

    Parameters
    ----------
    residuals : np.ndarray, shape (n_events, d_model) or (n_layers, n_events, d_model)
        Residual stream activations.

    Returns
    -------
    float : Norm of difference between first and last event.
    """
    if residuals.ndim == 3:
        # Use last layer
        residuals = residuals[-1]
    # residuals: (n_events, d_model)
    h1 = residuals[0]
    h3 = residuals[-1]  # or residuals[2] if 3 events
    return float(np.linalg.norm(h3 - h1))


def kruskal_epsilon_sq(values: np.ndarray, labels: np.ndarray) -> float:
    """Compute Kruskal-Wallis ε² effect size."""
    groups = []
    for sc in sorted(set(labels)):
        g = values[labels == sc]
        if len(g) > 0:
            groups.append(g)
    if len(groups) < 2:
        return 0.0
    kw = stats.kruskal(*groups)
    n = sum(len(g) for g in groups)
    return float(kw.statistic / (n - 1)) if n > 1 else 0.0


def run_experiment_d(model_name: str, results_dir: Path = RESULTS_DIR) -> dict:
    """Run competitive baselines experiment.

    Parameters
    ----------
    model_name : str
        Model key (e.g. "llama-7b").
    results_dir : Path
        Directory containing results.

    Returns
    -------
    dict with comparison results.
    """
    print(f"=== Experiment D: Competitive Baselines ({model_name}) ===\n")

    activations_path = CACHE_DIR / model_name / "activations.npz"
    if not activations_path.exists():
        print("Raw activations not found. Cannot compute alternative measures.")
        print("Returning holonomy-only results for reference.")

        # Load holonomy results for comparison
        holonomy_df = pd.read_csv(results_dir / f"{model_name}_curvature.csv")
        curvature_cols = [c for c in holonomy_df.columns if c.startswith("curvature_") and c != "curvature_total"]
        grouped = holonomy_df.groupby(["scenario", "conversation_id"])[curvature_cols].mean()
        scenarios = [s for s, _ in grouped.index]
        values = grouped.values
        # Use first block as representative
        hol_eps = kruskal_epsilon_sq(values[:, 0], np.array(scenarios))

        return {
            "model": model_name,
            "holonomy_epsilon_sq": hol_eps,
            "alternatives": {
                "attention_distance": {"epsilon_sq": None, "note": "requires raw activations"},
                "attention_entropy": {"epsilon_sq": None, "note": "requires raw activations"},
                "residual_norm_change": {"epsilon_sq": None, "note": "requires raw activations"},
            },
            "note": "Full competitive analysis requires raw activations. Run extraction first.",
        }

    # Load activations
    print("Loading activations ...")
    raw = np.load(activations_path, allow_pickle=False)

    # Reorganize into per-conversation dicts
    conv_data = {}
    for key in raw.files:
        parts = key.rsplit("/", 1)
        if len(parts) < 2 or "/test/" not in parts[0]:
            continue
        conv_id, field = parts
        if conv_id not in conv_data:
            conv_data[conv_id] = {}
        conv_data[conv_id][field] = raw[key]

    print(f"  Found {len(conv_data)} test conversations")

    # Compute alternative measures
    alt_measures = {"attention_distance": {}, "attention_entropy": {}, "residual_norm_change": {}}
    conv_ids = sorted(conv_data.keys())
    scenarios = []
    holonomy_totals = []

    # Load holonomy results for comparison
    holonomy_df = pd.read_csv(results_dir / f"{model_name}_curvature.csv")

    for conv_id in conv_ids:
        scenario = conv_id.split("/")[0]
        scenarios.append(scenario)

        fields = conv_data[conv_id]

        if "attention" in fields:
            attn = fields["attention"]
            alt_measures["attention_distance"][conv_id] = attention_distance(attn)
            alt_measures["attention_entropy"][conv_id] = attention_entropy(attn)

        if "residuals" in fields:
            alt_measures["residual_norm_change"][conv_id] = residual_norm_change(fields["residuals"])

        # Get holonomy total for this conversation
        h_row = holonomy_df[holonomy_df["conversation_id"] == conv_id]
        if not h_row.empty:
            holonomy_totals.append(float(h_row["curvature_total"].mean()))
        else:
            holonomy_totals.append(0.0)

    scenarios = np.array(scenarios)

    # Compute ε² for each measure
    results = {"model": model_name, "n_conversations": len(conv_ids)}

    # Holonomy
    hol_eps = kruskal_epsilon_sq(np.array(holonomy_totals), scenarios)
    results["holonomy"] = {"epsilon_sq": hol_eps}
    print(f"  Holonomy deviation ε² = {hol_eps:.4f}")

    # Alternative measures
    for measure_name, measure_data in alt_measures.items():
        if not measure_data:
            results[measure_name] = {"epsilon_sq": None, "note": "no data"}
            continue

        values = np.array([measure_data.get(c, 0) for c in conv_ids])
        eps = kruskal_epsilon_sq(values, scenarios)

        # Spearman correlation with holonomy
        rho, p_val = stats.spearmanr(values, np.array(holonomy_totals))

        results[measure_name] = {
            "epsilon_sq": eps,
            "spearman_rho_with_holonomy": float(rho),
            "spearman_p_value": float(p_val),
            "mean_per_scenario": {
                sc: float(values[scenarios == sc].mean())
                for sc in sorted(set(scenarios))
            },
        }
        print(f"  {measure_name}: ε² = {eps:.4f}, ρ(holonomy) = {rho:.4f}")

    # Comparison summary
    print(f"\n{'='*60}")
    print("Competitive Baselines Summary")
    print(f"{'='*60}")
    print(f"  Holonomy ε²: {hol_eps:.4f}")
    for measure_name in alt_measures:
        eps = results.get(measure_name, {}).get("epsilon_sq")
        if eps is not None:
            ratio = eps / hol_eps if hol_eps > 0 else float("inf")
            print(f"  {measure_name} ε²: {eps:.4f} (ratio to holonomy: {ratio:.2f})")
    print(f"{'='*60}")

    # Save
    output_path = results_dir / f"{model_name}_competitive_baselines.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    return results


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "llama-7b"
    run_experiment_d(model_name)
