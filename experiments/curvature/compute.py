"""
LCESA Transport Operators and Block-Specific Curvature
=======================================================
Computes the gauge-covariant transport operators and curvature tensors
used in the Low-Curvature Endogenous Standpoint Attractor (LCESA) framework.

Mathematical formulation:
    Transport operator:  U_{ij}^{(l)} = sum_k  alpha_bar_{ji}^{(k)}  P_{W_k}
    Curvature:           F_{ijk} = U_12 . U_23 . (U_13^exp)^{-1}
    Block norm:          ||F_{ijk}||_k = ||Q_k^T (F - I) Q_k||_F
"""

import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from experiments.config import LAYER_NAMES, MODELS, CACHE_DIR, RESULTS_DIR

# Regularization constants
EPSILON_PROJ = 1e-6   # added to each P_k during U construction
EPSILON_INV = 1e-4    # added to U_exp before inversion (Tikhonov)


# ---------------------------------------------------------------------------
# Core transport / curvature functions
# ---------------------------------------------------------------------------

def compute_transport_operator(
    attention: np.ndarray,
    value_matrices: np.ndarray,
    gamma: np.ndarray,
    event_from: int,
    event_to: int,
    layer: int,
) -> np.ndarray:
    """Compute the transport operator U_{ij}^{(l)} for a single model layer.

    The transport operator is built by summing over standpoint layers k = 0..4:

        U_{ij}^{(l)} = sum_k  alpha_bar_{ji}^{(k)}  P_{W_k}

    where alpha_bar^{(k)} is the mean attention weight over heads assigned to
    standpoint layer k, and P_{W_k} is the projection onto the value subspace
    spanned by those heads.  A small diagonal regularization (EPSILON_PROJ) is
    added to each P_k to prevent ill-conditioning at the source.

    Parameters
    ----------
    attention : np.ndarray
        Shape ``(n_layers, n_heads, n_events, n_events)`` — raw attention
        patterns extracted from the model.
    value_matrices : np.ndarray
        Shape ``(n_layers, n_heads, d_model, d_head)`` — value weight matrices.
    gamma : np.ndarray
        Shape ``(n_heads,)`` — integer standpoint-layer assignment (0-4) for
        each head.
    event_from : int
        Index of the source event (column index in attention).
    event_to : int
        Index of the target event (row index in attention).
    layer : int
        Model layer index at which to read attention and value matrices.

    Returns
    -------
    np.ndarray
        Shape ``(d_model, d_model)`` — the transport operator.
    """
    n_heads = attention.shape[1]
    d_model = value_matrices.shape[2]
    n_standpoint = len(LAYER_NAMES)

    U = np.zeros((d_model, d_model), dtype=np.float64)
    I_d = np.eye(d_model, dtype=np.float64)

    for k in range(n_standpoint):
        # Heads assigned to standpoint layer k
        head_mask = np.where(gamma == k)[0]
        if len(head_mask) == 0:
            continue

        # Mean attention from event_from to event_to over assigned heads
        # attention[layer, head, event_to, event_from]
        alpha_k = attention[layer, head_mask, event_to, event_from].mean()

        # Projection P_k: mean of V_h @ V_h^T over assigned heads
        # value_matrices[layer, head] is (d_model, d_head)
        V_heads = value_matrices[layer, head_mask]  # (n_assigned, d_model, d_head)
        # V @ V^T for each head -> (n_assigned, d_model, d_model)
        projections = np.einsum("hij,hkj->hik", V_heads, V_heads)
        P_k = projections.mean(axis=0)  # (d_model, d_model)

        # Level 1 regularization: stabilize P_k before accumulation
        P_k += EPSILON_PROJ * I_d

        U += alpha_k * P_k

    return U


def compute_projection_bases(
    value_matrices: np.ndarray,
    gamma: np.ndarray,
    layer: int,
    energy_threshold: float = 0.9,
) -> List[np.ndarray]:
    """Compute gamma-aligned projection bases for each standpoint layer.

    For each standpoint layer k, collect value matrices V_h for all heads h
    where gamma[h] == k, concatenate their column spaces, and compute an
    orthonormal basis via SVD.

    Parameters
    ----------
    value_matrices : np.ndarray
        Shape ``(n_layers, n_heads, d_model, d_head)``.
    gamma : np.ndarray
        Shape ``(n_heads,)`` with values in 0..4.
    layer : int
        Model layer at which to read value matrices.
    energy_threshold : float
        Fraction of singular value energy to retain.

    Returns
    -------
    list[np.ndarray]
        One ``(d_model, rank_k)`` orthonormal basis per standpoint layer.
    """
    d_model = value_matrices.shape[2]
    n_standpoint = len(LAYER_NAMES)
    bases = []

    for k in range(n_standpoint):
        head_indices = np.where(gamma == k)[0]
        if len(head_indices) == 0:
            bases.append(np.eye(d_model, dtype=np.float64))
            continue

        # Collect V_h columns: each head contributes (d_model, d_head)
        # Concatenate horizontally: (d_model, n_heads * d_head)
        V_concat = np.concatenate(
            [value_matrices[layer, h] for h in head_indices], axis=1
        )

        # SVD to get orthonormal basis for column space
        U, s, _ = np.linalg.svd(V_concat, full_matrices=False)

        # Retain components accounting for energy_threshold of total energy
        energy = np.cumsum(s ** 2) / np.sum(s ** 2)
        rank = int(np.searchsorted(energy, energy_threshold)) + 1
        rank = min(rank, len(s))

        Q_k = U[:, :rank]  # (d_model, rank_k)
        bases.append(Q_k)

    return bases


def compute_curvature(
    U_12: np.ndarray,
    U_23: np.ndarray,
    U_exp: np.ndarray,
    projection_bases: List[np.ndarray] | None = None,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Compute the curvature tensor F and its block-specific norms.

    The curvature is defined as:

        F_{ijk} = U_{12} . U_{23} . (U_{13}^exp)^{-1}

    When ``projection_bases`` is provided, the block-specific norm for layer k
    is computed by projecting ``(F - I)`` onto the gamma-aligned subspace Q_k::

        ||F||_k = ||Q_k^T (F - I) Q_k||_F

    Otherwise, falls back to equal-width diagonal blocks (legacy behavior).

    Parameters
    ----------
    U_12 : np.ndarray
        Shape ``(d_model, d_model)`` — transport operator from event 1 to 2.
    U_23 : np.ndarray
        Shape ``(d_model, d_model)`` — transport operator from event 2 to 3.
    U_exp : np.ndarray
        Shape ``(d_model, d_model)`` — expected (direct) transport operator
        from event 1 to 3.
    projection_bases : list[np.ndarray] or None
        Gamma-aligned orthonormal bases per standpoint layer.  If None,
        uses legacy equal-width block partition.

    Returns
    -------
    F : np.ndarray
        Shape ``(d_model, d_model)`` — the curvature tensor.
    block_norms : dict
        ``{layer_name: float}`` — Frobenius norm of each block minus identity.
    """
    d_model = U_12.shape[0]
    n_standpoint = len(LAYER_NAMES)

    # Level 2 regularization: Tikhonov damping before inversion
    U_exp_reg = U_exp + EPSILON_INV * np.eye(d_model, dtype=U_exp.dtype)

    try:
        U_exp_inv = np.linalg.inv(U_exp_reg)
    except np.linalg.LinAlgError:
        U_exp_inv = np.linalg.pinv(U_exp_reg)

    # Curvature tensor
    F = U_12 @ U_23 @ U_exp_inv

    # Block-specific norms
    block_norms: Dict[str, float] = {}
    deviation = F - np.eye(d_model, dtype=F.dtype)

    for idx, name in enumerate(LAYER_NAMES):
        if projection_bases is not None:
            Q_k = projection_bases[idx]  # (d_model, rank_k)
            F_projected = Q_k.T @ deviation @ Q_k  # (rank_k, rank_k)
            norm = float(np.linalg.norm(F_projected, "fro"))
        else:
            # Fallback: equal-width blocks (legacy behavior)
            block_size = d_model // n_standpoint
            start = idx * block_size
            end = d_model if idx == n_standpoint - 1 else (idx + 1) * block_size
            block_dim = end - start
            F_block = F[start:end, start:end]
            I_block = np.eye(block_dim, dtype=F.dtype)
            norm = float(np.linalg.norm(F_block - I_block, "fro"))

        block_norms[name] = norm

    return F, block_norms


def compute_baseline_transport(
    test_activations: Dict[str, Dict[str, np.ndarray]],
    scenario: str,
    layer: int,
    gamma: np.ndarray,
    value_matrices: np.ndarray,
) -> np.ndarray:
    """Compute the mean transport product U_12 @ U_23 for T1 test conversations.

    This serves as the expected transport U_{13}^exp under the null (baseline)
    condition where no standpoint-specific failure is induced.

    Parameters
    ----------
    test_activations : dict
        ``{conv_id: {"attention": ..., ...}}`` —
        reorganised activations for the test split.  Only T1 entries are used.
    scenario : str
        The scenario identifier (used for logging; the function always filters
        for T1 conversations).
    layer : int
        Model layer index at which to evaluate transport operators.
    value_matrices : np.ndarray
        Shared value projection weights ``(n_layers, n_heads, d_model, d_head)``.

    Returns
    -------
    np.ndarray
        Shape ``(d_model, d_model)`` — mean product U_12 @ U_23 over all T1
        test conversations.
    """
    products = []

    for conv_id, fields in test_activations.items():
        # Only use T1 (baseline) conversations
        if not conv_id.startswith("T1/"):
            continue

        attn = fields["attention"]
        V = value_matrices

        # Events 1->2 (indices 0->1) and 2->3 (indices 1->2) in a 5-event conv
        U_12 = compute_transport_operator(attn, V, gamma, 0, 1, layer)
        U_23 = compute_transport_operator(attn, V, gamma, 1, 2, layer)
        products.append(U_12 @ U_23)

    if not products:
        raise ValueError(
            f"No T1 test conversations found for baseline computation "
            f"(scenario filter: '{scenario}')."
        )

    return np.mean(products, axis=0)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_curvature_computation(
    model_name: str,
    activations_path: Path,
    gamma_path: Path,
    output_dir: Path = RESULTS_DIR,
) -> Path:
    """Run the full curvature computation pipeline for a single model.

    Steps:
        1. Load activations.npz and grouping gamma.
        2. Reorganise test-split activations into per-conversation dicts.
        3. For each model layer, compute baseline transport from T1 test data.
        4. For EVERY test conversation (including T1) at each layer, compute
           curvature using gamma-aligned projection bases.
        5. Save results to CSV.

    Parameters
    ----------
    model_name : str
        Key into ``MODELS`` (e.g. ``"gpt2"``).
    activations_path : Path
        Path to the ``activations.npz`` produced by extraction.
    gamma_path : Path
        Path to the ``{model_name}_grouping.npz`` produced by head grouping.
    output_dir : Path, optional
        Directory for the output CSV (default ``RESULTS_DIR``).

    Returns
    -------
    Path
        Path to the saved results CSV.
    """
    if model_name not in MODELS:
        raise ValueError(
            f"Unknown model '{model_name}'. Available: {list(MODELS.keys())}"
        )
    model_config = MODELS[model_name]
    n_layers = model_config.n_layers

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print(f"Loading activations from {activations_path} ...")
    raw = np.load(activations_path, allow_pickle=False)

    print(f"Loading gamma from {gamma_path} ...")
    grouping = np.load(gamma_path, allow_pickle=False)
    gamma = grouping["gamma"]  # (n_heads,)

    # ------------------------------------------------------------------
    # 2. Reorganise test-split activations into {conv_id: {field: array}}
    # ------------------------------------------------------------------
    print("Reorganising test activations ...")
    test_activations: Dict[str, Dict[str, np.ndarray]] = {}
    conv_ids_seen: set = set()

    for key in raw.files:
        parts = key.rsplit("/", 1)
        if len(parts) < 2:
            continue  # skip top-level keys like "value_matrices"
        conv_id, field = parts[0], parts[1]
        # Only process test split
        if "/test/" not in conv_id:
            continue
        conv_ids_seen.add(conv_id)

    for conv_id in conv_ids_seen:
        fields: Dict[str, np.ndarray] = {}
        for field in ("attention", "residuals", "value_matrices"):
            full_key = f"{conv_id}/{field}"
            if full_key in raw.files:
                fields[field] = raw[full_key]
        test_activations[conv_id] = fields

    if not test_activations:
        raise ValueError("No test-split activations found in the archive.")

    # ------------------------------------------------------------------
    # 3-4. Compute curvature for EVERY conversation at every layer
    # ------------------------------------------------------------------
    all_conv_ids = sorted(test_activations.keys())

    # Pre-compute projection bases using value matrices
    # Check top-level key first (optimized format), then per-conversation
    sample_V = None
    if "value_matrices" in raw.files:
        sample_V = raw["value_matrices"]
    else:
        for fields in test_activations.values():
            if "value_matrices" in fields:
                sample_V = fields["value_matrices"]
                break

    print(f"Computing curvature for {len(all_conv_ids)} test conversations "
          f"across {n_layers} layers (including T1 baseline) ...")

    rows = []
    total = n_layers * len(all_conv_ids)
    done = 0
    warn_count = 0

    for layer in range(n_layers):
        # Compute baseline transport from T1 test conversations at this layer
        U_exp = compute_baseline_transport(test_activations, "T1", layer, gamma, sample_V)

        # Compute gamma-aligned projection bases for this layer
        proj_bases = compute_projection_bases(sample_V, gamma, layer)

        for conv_id in all_conv_ids:
            fields = test_activations[conv_id]
            attn = fields["attention"]
            V = sample_V  # value_matrices are model weights, same for all convs

            # Transport operators for events 1->2 and 2->3
            U_12 = compute_transport_operator(attn, V, gamma, 0, 1, layer)
            U_23 = compute_transport_operator(attn, V, gamma, 1, 2, layer)

            # Curvature tensor and block norms
            F, block_norms = compute_curvature(U_12, U_23, U_exp, proj_bases)

            # Level 3 sanity check: warn if F has extreme singular values
            if warn_count < 5:
                sv = np.linalg.svd(F, compute_uv=False)
                if sv[0] > 1e8:
                    cond = sv[0] / max(sv[-1], 1e-30)
                    print(f"  WARNING: layer={layer} conv={conv_id} "
                          f"max_sv={sv[0]:.2e}, cond={cond:.2e}")
                    warn_count += 1

            # Extract scenario from conv_id
            scenario = conv_id.split("/")[0]

            # Total curvature (L2 norm of block norms)
            curvature_total = float(np.sqrt(sum(v ** 2 for v in block_norms.values())))

            row = {"conversation_id": conv_id, "scenario": scenario, "layer": layer}
            for name in LAYER_NAMES:
                row[f"curvature_{name}"] = block_norms[name]
            row["curvature_total"] = curvature_total
            rows.append(row)

            done += 1
            if done % 50 == 0:
                print(f"  ... processed {done}/{total} combinations")

    # ------------------------------------------------------------------
    # 5. Save results to CSV
    # ------------------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{model_name}_curvature.csv"

    fieldnames = ["conversation_id", "scenario", "layer"]
    for name in LAYER_NAMES:
        fieldnames.append(f"curvature_{name}")
    fieldnames.append("curvature_total")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Curvature results saved to {csv_path} ({len(rows)} rows)")
    return csv_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "gpt2"

    activations_path = CACHE_DIR / model_name / "activations.npz"
    gamma_path = CACHE_DIR / model_name / f"{model_name}_grouping.npz"

    # Fall back to DATA_DIR for grouping if not in cache
    if not gamma_path.exists():
        gamma_path = Path("data") / f"{model_name}_grouping.npz"

    run_curvature_computation(model_name, activations_path, gamma_path)
