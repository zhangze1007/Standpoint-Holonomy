"""
Experiment C: PCA Decomposition of Block Holonomy
==================================================
Quantifies the "one dominant mode" claim by performing PCA on the
block holonomy matrix (conversations × 5 standpoint blocks).

Outputs:
- Explained variance ratio for each PC
- PC1-PC5 loading vectors
- PC1 scenario discrimination (Kruskal-Wallis ε²)
- Residual PC scenario discrimination after controlling for PC1
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.config import RESULTS_DIR, LAYER_NAMES, ALPHA


def load_block_holonomy_matrix(model_name: str, results_dir: Path = RESULTS_DIR) -> tuple:
    """Load curvature results and build the block holonomy matrix.

    Returns
    -------
    X : np.ndarray, shape (n_conversations, 5)
        Block holonomy norms per conversation (averaged across model layers).
    conv_ids : list of str
        Conversation IDs corresponding to rows.
    scenarios : list of str
        Scenario labels corresponding to rows.
    """
    csv_path = results_dir / f"{model_name}_curvature.csv"
    df = pd.read_csv(csv_path)

    curvature_cols = [
        c for c in df.columns
        if c.startswith("curvature_") and c != "curvature_total"
    ]

    # Average across model layers for each conversation
    grouped = df.groupby(["conversation_id", "scenario"])[curvature_cols].mean().reset_index()

    conv_ids = grouped["conversation_id"].tolist()
    scenarios = grouped["scenario"].tolist()
    X = grouped[curvature_cols].values

    return X, conv_ids, scenarios


def run_pca(X: np.ndarray) -> dict:
    """Run PCA on the block holonomy matrix.

    Parameters
    ----------
    X : np.ndarray, shape (n, 5)
        Block holonomy norms.

    Returns
    -------
    dict with:
        - explained_variance_ratio
        - loadings (PC components)
        - X_pca (projected data)
        - singular_values
    """
    # Z-score normalize each column
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    X_z = (X - mean) / std

    # SVD-based PCA
    U, s, Vt = np.linalg.svd(X_z, full_matrices=False)

    # Explained variance
    var = s ** 2 / (len(X_z) - 1)
    total_var = var.sum()
    explained_ratio = var / total_var

    # Project data onto PCs
    X_pca = U * s  # equivalent to X_z @ Vt.T

    return {
        "explained_variance_ratio": explained_ratio.tolist(),
        "cumulative_variance": np.cumsum(explained_ratio).tolist(),
        "loadings": Vt.tolist(),  # each row is a PC loading vector
        "X_pca": X_pca,
        "singular_values": s.tolist(),
        "n_samples": len(X_z),
        "n_features": X_z.shape[1],
    }


def pc_scenario_discrimination(X_pca: np.ndarray, scenarios: list, pc_idx: int = 0) -> dict:
    """Test whether a single PC discriminates across scenarios.

    Uses Kruskal-Wallis test, reports ε² effect size.
    """
    pc_scores = X_pca[:, pc_idx]
    groups = []
    labels = []
    for sc in sorted(set(scenarios)):
        vals = pc_scores[np.array(scenarios) == sc]
        if len(vals) > 0:
            groups.append(vals)
            labels.append(sc)

    if len(groups) < 2:
        return {"error": "need at least 2 groups"}

    kw = stats.kruskal(*groups)
    n_total = sum(len(g) for g in groups)
    epsilon_sq = float(kw.statistic / (n_total - 1)) if n_total > 1 else 0.0

    return {
        "pc": pc_idx + 1,
        "kruskal_wallis_statistic": float(kw.statistic),
        "epsilon_squared": epsilon_sq,
        "p_value": float(kw.pvalue),
        "n_groups": len(groups),
        "scenario_labels": labels,
        "passed": kw.pvalue < ALPHA,
    }


def residual_discrimination(X_pca: np.ndarray, scenarios: list) -> dict:
    """Test whether residual PCs (PC2-PC5) discriminate after controlling for PC1.

    Uses partial Kruskal-Wallis: regress out PC1, then test residuals.
    """
    # Regress each PC2-PC5 on PC1, take residuals
    pc1 = X_pca[:, 0]
    residual_results = {}

    for pc_idx in range(1, X_pca.shape[1]):
        pc_scores = X_pca[:, pc_idx]

        # Simple linear regression: pc_scores = a * pc1 + b + residual
        slope, intercept = np.polyfit(pc1, pc_scores, 1)
        residuals = pc_scores - (slope * pc1 + intercept)

        # Kruskal-Wallis on residuals
        groups = []
        labels = []
        for sc in sorted(set(scenarios)):
            vals = residuals[np.array(scenarios) == sc]
            if len(vals) > 0:
                groups.append(vals)
                labels.append(sc)

        if len(groups) < 2:
            residual_results[f"PC{pc_idx + 1}"] = {"error": "need at least 2 groups"}
            continue

        kw = stats.kruskal(*groups)
        n_total = sum(len(g) for g in groups)
        epsilon_sq = float(kw.statistic / (n_total - 1)) if n_total > 1 else 0.0

        residual_results[f"PC{pc_idx + 1}"] = {
            "kruskal_wallis_statistic": float(kw.statistic),
            "epsilon_squared": epsilon_sq,
            "p_value": float(kw.pvalue),
            "n_groups": len(groups),
            "scenario_labels": labels,
            "passed": kw.pvalue < ALPHA,
        }

    return residual_results


def per_scenario_pc1_means(X_pca: np.ndarray, scenarios: list) -> dict:
    """Compute mean PC1 score per scenario."""
    pc1 = X_pca[:, 0]
    result = {}
    for sc in sorted(set(scenarios)):
        vals = pc1[np.array(scenarios) == sc]
        result[sc] = {
            "mean": float(vals.mean()),
            "std": float(vals.std()),
            "n": len(vals),
        }
    return result


def run_experiment_c(model_name: str, results_dir: Path = RESULTS_DIR) -> dict:
    """Run the full PCA decomposition experiment.

    Parameters
    ----------
    model_name : str
        Model key (e.g. "llama-7b").
    results_dir : Path
        Directory containing curvature results.

    Returns
    -------
    dict with all PCA results.
    """
    print(f"Loading block holonomy matrix for {model_name} ...")
    X, conv_ids, scenarios = load_block_holonomy_matrix(model_name, results_dir)
    print(f"  Matrix shape: {X.shape} ({len(set(scenarios))} scenarios)")

    print("Running PCA ...")
    pca_results = run_pca(X)

    print(f"  Explained variance ratio: {[f'{v:.4f}' for v in pca_results['explained_variance_ratio']]}")
    print(f"  Cumulative variance: {[f'{v:.4f}' for v in pca_results['cumulative_variance']]}")

    # PC1 loadings interpretation
    block_names = LAYER_NAMES
    loadings = pca_results["loadings"]
    print("\n  PC1 loadings (z-scored block weights):")
    for i, name in enumerate(block_names):
        print(f"    {name}: {loadings[0][i]:.4f}")

    # PC1 scenario discrimination
    print("\n  Testing PC1 scenario discrimination ...")
    pc1_disc = pc_scenario_discrimination(pca_results["X_pca"], scenarios, pc_idx=0)
    print(f"    ε² = {pc1_disc['epsilon_squared']:.4f}, p = {pc1_disc['p_value']:.2e}")

    # Residual discrimination
    print("  Testing residual PC discrimination (after controlling for PC1) ...")
    resid_disc = residual_discrimination(pca_results["X_pca"], scenarios)
    for pc_name, res in resid_disc.items():
        if "epsilon_squared" in res:
            print(f"    {pc_name}: ε² = {res['epsilon_squared']:.4f}, p = {res['p_value']:.2e}")

    # Per-scenario PC1 means
    scenario_means = per_scenario_pc1_means(pca_results["X_pca"], scenarios)
    print("\n  PC1 mean per scenario:")
    for sc, vals in scenario_means.items():
        print(f"    {sc}: {vals['mean']:.4f} ± {vals['std']:.4f}")

    # Compile results
    results = {
        "model": model_name,
        "n_samples": len(conv_ids),
        "n_scenarios": len(set(scenarios)),
        "block_names": block_names,
        "explained_variance_ratio": pca_results["explained_variance_ratio"],
        "cumulative_variance": pca_results["cumulative_variance"],
        "loadings": pca_results["loadings"],
        "pc1_scenario_discrimination": pc1_disc,
        "residual_discrimination": resid_disc,
        "per_scenario_pc1_means": scenario_means,
        "singular_values": pca_results["singular_values"],
    }

    # Save
    output_path = results_dir / f"{model_name}_pca_analysis.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove X_pca from saved results (not JSON serializable, large)
    save_results = {k: v for k, v in results.items() if k != "X_pca"}
    with open(output_path, "w") as f:
        json.dump(save_results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    return results


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "llama-7b"
    run_experiment_c(model_name)
