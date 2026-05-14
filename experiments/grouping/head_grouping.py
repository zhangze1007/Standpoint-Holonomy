"""
LCESA Head Grouping via Attention Differentials
================================================
Assign each attention head to the standpoint layer where it exhibits
the largest differential attention signal, then measure subspace overlap
between layer-specific head groups.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Tuple

from experiments.config import LAYER_NAMES, MODELS, CACHE_DIR, DATA_DIR


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def compute_attention_differential(
    grouping_activations: Dict[str, Dict[str, np.ndarray]],
    scenario_positive: str,
    scenario_negative: str = "T1",
) -> np.ndarray:
    """Compute the attention differential between two scenario groups.

    For each conversation the mean attention from event 0 to event 2
    (``attn[:, :, 2, 0]``) is extracted.  The differential is

        delta = mean(positive) - mean(negative)

    Parameters
    ----------
    grouping_activations : dict
        ``{conv_id: {"attention": (L, H, 5, 5), ...}}`` where *conv_id*
        encodes the scenario name as its first path component (e.g.
        ``"T2/grouping/0"``).
    scenario_positive : str
        Scenario treated as the positive condition (e.g. ``"T2"``).
    scenario_negative : str, optional
        Scenario treated as the negative/baseline condition (default ``"T1"``).

    Returns
    -------
    np.ndarray
        Shape ``(n_layers, n_heads)`` — the mean attention differential.
    """
    positive_samples = []
    negative_samples = []

    for conv_id, fields in grouping_activations.items():
        scenario = conv_id.split("/")[0]
        attn = fields["attention"]  # (L, H, 5, 5)
        # Mean attention from event 0 to event 2
        signal = attn[:, :, 2, 0]  # (L, H)

        if scenario == scenario_positive:
            positive_samples.append(signal)
        elif scenario == scenario_negative:
            negative_samples.append(signal)

    if not positive_samples:
        raise ValueError(
            f"No grouping samples found for positive scenario '{scenario_positive}'"
        )
    if not negative_samples:
        raise ValueError(
            f"No grouping samples found for negative scenario '{scenario_negative}'"
        )

    mean_positive = np.mean(positive_samples, axis=0)  # (L, H)
    mean_negative = np.mean(negative_samples, axis=0)  # (L, H)

    return mean_positive - mean_negative


def assign_heads_to_layers(
    deltas: Dict[str, np.ndarray],
    n_heads: int,
) -> np.ndarray:
    """Assign each attention head to the standpoint layer with the largest
    absolute differential.

    Parameters
    ----------
    deltas : dict
        ``{"min": array(L, H), "nar": array(L, H), ...}`` — one
        differential matrix per standpoint layer.
    n_heads : int
        Number of attention heads.

    Returns
    -------
    np.ndarray
        Shape ``(n_heads,)`` — integer layer index (0-4) for each head.
    """
    # Stack into (n_standpoint_layers, n_model_layers, n_heads)
    stacked = np.stack([deltas[name] for name in LAYER_NAMES], axis=0)
    # Mean over model layers → (n_standpoint_layers, n_heads)
    mean_over_layers = stacked.mean(axis=1)

    # For each head, pick the standpoint layer with largest |delta|
    gamma = np.argmax(np.abs(mean_over_layers), axis=0)  # (n_heads,)
    return gamma


def compute_subspace_overlap(
    gamma: np.ndarray,
    value_matrices: np.ndarray,
    n_heads: int,
) -> float:
    """Compute the maximum pairwise subspace overlap between standpoint layers.

    For each standpoint layer *k*, the value vectors of its assigned heads
    are collected, mean-subtracted, flattened, and reduced via SVD to an
    orthonormal basis :math:`Q_k`.  The overlap between layers *k* and *l*
    is :math:`\\|Q_k^T Q_l\\|_2` (largest singular value).  The function
    returns the maximum such value over all pairs :math:`k \\neq l`.

    Parameters
    ----------
    gamma : np.ndarray
        Shape ``(n_heads,)`` — standpoint-layer assignment for each head.
    value_matrices : np.ndarray
        Shape ``(n_layers, n_heads, d_model, d_head)`` — the value weight
        matrices (typically taken from the last model layer).
    n_heads : int
        Number of attention heads.

    Returns
    -------
    float
        Maximum pairwise subspace overlap δ.
    """
    n_standpoint = len(LAYER_NAMES)
    # Use the last model layer's value matrices
    V = value_matrices[-1]  # (n_heads, d_model, d_head)

    bases: list[np.ndarray] = []

    for k in range(n_standpoint):
        head_indices = np.where(gamma == k)[0]
        if len(head_indices) == 0:
            # No heads assigned — use a zero-dimensional basis
            bases.append(np.zeros((V.shape[1], 0)))
            continue

        # Collect value vectors for assigned heads: each head contributes
        # d_model * d_head features when flattened
        vectors = []
        for h in head_indices:
            vectors.append(V[h].flatten())  # (d_model * d_head,)
        mat = np.stack(vectors, axis=0)  # (n_assigned, d_model * d_head)

        # Mean-subtract and compute SVD for orthonormal basis
        mat = mat - mat.mean(axis=0, keepdims=True)
        if mat.shape[0] == 0 or mat.shape[1] == 0:
            bases.append(np.zeros((mat.shape[1], 0)))
            continue
        U, s, _ = np.linalg.svd(mat, full_matrices=False)
        # Keep components with non-negligible singular values
        rank = np.sum(s > 1e-8 * s[0]) if s[0] > 0 else 0
        Q = U[:, :rank]  # (n_assigned, rank) — but we want the column space
        # The orthonormal basis is the right singular vectors projected:
        # Q_k = V^T basis from SVD of (n_assigned, features)
        # More precisely, the column space of mat^T is spanned by the
        # right singular vectors of mat, which are the columns of Vt^T.
        # But for overlap we need Q_k^T Q_l where Q_k has orthonormal columns.
        # From SVD: mat = U S Vt, so columns of Vt^T (rows of Vt) span the
        # row space of mat.  The orthonormal basis for the column space of
        # mat^T is Vt^T[:, :rank].
        _, _, Vt = np.linalg.svd(mat, full_matrices=False)
        Q = Vt[:rank].T  # (features, rank)
        bases.append(Q)

    # Compute pairwise overlaps
    max_overlap = 0.0
    for k in range(n_standpoint):
        for l in range(k + 1, n_standpoint):
            Qk = bases[k]
            Ql = bases[l]
            if Qk.shape[1] == 0 or Ql.shape[1] == 0:
                continue
            product = Qk.T @ Ql  # (rank_k, rank_l)
            svs = np.linalg.svd(product, compute_uv=False)
            overlap = float(svs[0]) if len(svs) > 0 else 0.0
            if overlap > max_overlap:
                max_overlap = overlap

    return max_overlap


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_head_grouping(
    model_name: str,
    activations_path: Path,
    output_dir: Path = DATA_DIR,
) -> Tuple[np.ndarray, float]:
    """Run the full head-grouping pipeline for a single model.

    Steps:
        1. Load activations from the npz archive.
        2. Reorganize into ``{conv_id: {field: array}}`` format.
        3. Compute attention differentials for each standpoint layer.
        4. Assign heads to layers via ``assign_heads_to_layers``.
        5. Compute subspace overlap via ``compute_subspace_overlap``.
        6. Save results and print a summary.

    Parameters
    ----------
    model_name : str
        Key into ``MODELS`` (e.g. ``"gpt2"``).
    activations_path : Path
        Path to the ``activations.npz`` file produced by extraction.
    output_dir : Path, optional
        Directory for the output file (default ``DATA_DIR``).

    Returns
    -------
    gamma : np.ndarray
        Shape ``(n_heads,)`` — standpoint-layer assignment per head.
    delta_overlap : float
        Maximum pairwise subspace overlap.
    """
    if model_name not in MODELS:
        raise ValueError(
            f"Unknown model '{model_name}'. Available: {list(MODELS.keys())}"
        )
    model_config = MODELS[model_name]
    n_heads = model_config.n_heads

    # ------------------------------------------------------------------
    # 1. Load activations
    # ------------------------------------------------------------------
    print(f"Loading activations from {activations_path} ...")
    raw = np.load(activations_path, allow_pickle=False)

    # ------------------------------------------------------------------
    # 2. Reorganize into {conv_id: {field: array}}
    # ------------------------------------------------------------------
    grouping_activations: Dict[str, Dict[str, np.ndarray]] = {}
    # Keys look like "T2/grouping/0/attention"
    conv_ids_seen: set = set()
    for key in raw.files:
        parts = key.rsplit("/", 1)
        conv_id, field = parts[0], parts[1]
        conv_ids_seen.add(conv_id)

    for conv_id in conv_ids_seen:
        fields: Dict[str, np.ndarray] = {}
        for field in ("attention", "residuals", "value_matrices"):
            full_key = f"{conv_id}/{field}"
            if full_key in raw.files:
                fields[field] = raw[full_key]
        grouping_activations[conv_id] = fields

    # Also collect value matrices (use first available, they share weights)
    # Check top-level key first (optimized format), then per-conversation
    value_matrices = None
    if "value_matrices" in raw.files:
        value_matrices = raw["value_matrices"]
    else:
        for fields in grouping_activations.values():
            if "value_matrices" in fields:
                value_matrices = fields["value_matrices"]
                break
    if value_matrices is None:
        raise ValueError("No value_matrices found in activations archive.")

    # ------------------------------------------------------------------
    # 3. Compute attention differentials
    # ------------------------------------------------------------------
    print("Computing attention differentials ...")
    deltas: Dict[str, np.ndarray] = {}

    # Targeted layers: positive scenario vs T1 baseline
    scenario_map = {
        "nar": "T2",
        "mor": "T3",
        "soc": "T4",
        "pos": "T5",
    }
    for layer_name, scenario in scenario_map.items():
        deltas[layer_name] = compute_attention_differential(
            grouping_activations, scenario_positive=scenario, scenario_negative="T1"
        )

    # Minimal layer: split T1 grouping samples into high-overlap and
    # low-overlap halves based on mean attention from event 0 → event 2.
    t1_signals = []
    t1_conv_ids = []
    for conv_id, fields in grouping_activations.items():
        if conv_id.startswith("T1/"):
            attn = fields["attention"]
            signal = attn[:, :, 2, 0].mean()  # scalar summary
            t1_signals.append(signal)
            t1_conv_ids.append(conv_id)

    if len(t1_conv_ids) < 2:
        raise ValueError("Need at least 2 T1 grouping samples for min-layer split.")

    median_signal = np.median(t1_signals)
    high_overlap = []
    low_overlap = []
    for conv_id, sig in zip(t1_conv_ids, t1_signals):
        attn = grouping_activations[conv_id]["attention"][:, :, 2, 0]  # (L, H)
        if sig >= median_signal:
            high_overlap.append(attn)
        else:
            low_overlap.append(attn)

    deltas["min"] = np.mean(high_overlap, axis=0) - np.mean(low_overlap, axis=0)

    # ------------------------------------------------------------------
    # 4. Assign heads to layers
    # ------------------------------------------------------------------
    print("Assigning heads to standpoint layers ...")
    gamma = assign_heads_to_layers(deltas, n_heads)

    # ------------------------------------------------------------------
    # 5. Compute subspace overlap
    # ------------------------------------------------------------------
    print("Computing subspace overlap ...")
    delta_overlap = compute_subspace_overlap(gamma, value_matrices, n_heads)

    # ------------------------------------------------------------------
    # 6. Save and report
    # ------------------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / f"{model_name}_grouping.npz"

    # Convert deltas dict to a flat saveable form
    save_dict = {"gamma": gamma, "delta_overlap": np.array(delta_overlap)}
    for name, arr in deltas.items():
        save_dict[f"delta_{name}"] = arr

    np.savez_compressed(save_path, **save_dict)
    print(f"Results saved to {save_path}")

    # Summary
    print(f"\n{'='*50}")
    print(f"Head Grouping Summary — {model_name}")
    print(f"{'='*50}")
    for idx, name in enumerate(LAYER_NAMES):
        count = int(np.sum(gamma == idx))
        print(f"  {name:>3s} (layer {idx}): {count:>3d} heads")
    print(f"  Subspace overlap δ = {delta_overlap:.4f}")
    print(f"{'='*50}")

    return gamma, delta_overlap


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    model_name = sys.argv[1] if len(sys.argv) > 1 else "gpt2"
    activations_path = CACHE_DIR / model_name / "activations.npz"
    run_head_grouping(model_name, activations_path)
