"""
LCESA Visualization Module
===========================
Publication-quality plots for curvature analysis results.
"""

from pathlib import Path
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from experiments.config import LAYER_NAMES, LAYER_DISPLAY, FIGURES_DIR, DPI


# Column names for curvature by standpoint layer
CURVATURE_COLS = [f"curvature_{name}" for name in LAYER_NAMES]


def _display_labels() -> list[str]:
    """Return display-name labels for the five standpoint layers."""
    return [LAYER_DISPLAY[name] for name in LAYER_NAMES]


def plot_curvature_heatmap(
    df: pd.DataFrame,
    model_name: str,
    output_dir: Path = FIGURES_DIR,
) -> Path:
    """Plot a heatmap of mean curvature per scenario and standpoint layer.

    Parameters
    ----------
    df : pd.DataFrame
        Curvature results with columns ``scenario``, ``curvature_min``,
        ``curvature_nar``, etc.
    model_name : str
        Model identifier (used in the filename and title).
    output_dir : Path, optional
        Directory for the saved figure (default ``FIGURES_DIR``).

    Returns
    -------
    Path
        Path to the saved PNG file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Aggregate: mean curvature per scenario per layer
    pivot = df.groupby("scenario")[CURVATURE_COLS].mean()
    pivot.columns = _display_labels()

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(
        pivot,
        cmap="YlOrRd",
        annot=True,
        fmt=".3f",
        ax=ax,
        linewidths=0.5,
        cbar_kws={"label": "Mean Block Curvature"},
    )
    ax.set_title(f"Block-Specific Curvature — {model_name}")
    ax.set_ylabel("Scenario")
    ax.set_xlabel("Standpoint Layer")

    fig.tight_layout()
    out_path = output_dir / f"{model_name}_curvature_heatmap.png"
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved heatmap to {out_path}")
    return out_path


def plot_curvature_boxplots(
    df: pd.DataFrame,
    model_name: str,
    output_dir: Path = FIGURES_DIR,
) -> Path:
    """Plot boxplots of curvature distributions, one subplot per layer.

    Parameters
    ----------
    df : pd.DataFrame
        Curvature results with columns ``scenario``, ``conversation_id``,
        and the curvature columns.
    model_name : str
        Model identifier (used in the filename and title).
    output_dir : Path, optional
        Directory for the saved figure (default ``FIGURES_DIR``).

    Returns
    -------
    Path
        Path to the saved PNG file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Melt to long form
    melted = df.melt(
        id_vars=["scenario", "conversation_id"],
        value_vars=CURVATURE_COLS,
        var_name="layer",
        value_name="curvature",
    )
    # Map column names to display labels
    col_to_display = {
        f"curvature_{name}": LAYER_DISPLAY[name] for name in LAYER_NAMES
    }
    melted["layer"] = melted["layer"].map(col_to_display)

    fig, axes = plt.subplots(1, 5, figsize=(20, 4), sharey=True)
    display = _display_labels()

    for idx, (ax, layer_label) in enumerate(zip(axes, display)):
        subset = melted[melted["layer"] == layer_label]
        sns.boxplot(data=subset, x="scenario", y="curvature", ax=ax, palette="Set2")
        ax.set_title(layer_label)
        ax.set_xlabel("Scenario")
        if idx == 0:
            ax.set_ylabel("Block Curvature")
        else:
            ax.set_ylabel("")

    fig.suptitle(f"Curvature Distributions by Layer — {model_name}", y=1.02)
    fig.tight_layout()
    out_path = output_dir / f"{model_name}_curvature_boxplots.png"
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved boxplots to {out_path}")
    return out_path


def plot_model_comparison(
    results_by_model: Dict[str, pd.DataFrame],
    output_dir: Path = FIGURES_DIR,
) -> Path:
    """Plot a side-by-side heatmap comparison across multiple models.

    Parameters
    ----------
    results_by_model : Dict[str, pd.DataFrame]
        ``{model_name: curvature_df}`` — one DataFrame per model.
    output_dir : Path, optional
        Directory for the saved figure (default ``FIGURES_DIR``).

    Returns
    -------
    Path
        Path to the saved PNG file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_models = len(results_by_model)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    if n_models == 1:
        axes = [axes]

    for ax, (model_name, df) in zip(axes, results_by_model.items()):
        pivot = df.groupby("scenario")[CURVATURE_COLS].mean()
        pivot.columns = _display_labels()

        sns.heatmap(
            pivot,
            cmap="YlOrRd",
            annot=True,
            fmt=".3f",
            ax=ax,
            linewidths=0.5,
            cbar_kws={"label": "Mean Block Curvature"},
        )
        ax.set_title(model_name)
        ax.set_ylabel("Scenario")
        ax.set_xlabel("Standpoint Layer")

    fig.suptitle("Cross-Model Curvature Comparison", y=1.02)
    fig.tight_layout()
    out_path = output_dir / "model_comparison.png"
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved model comparison to {out_path}")
    return out_path
