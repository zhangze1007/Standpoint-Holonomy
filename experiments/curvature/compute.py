"""
LCESA Transport Operators and Block-Specific Curvature (GPU-Accelerated)
========================================================================
Computes the gauge-covariant transport operators and curvature tensors
used in the Low-Curvature Endogenous Standpoint Attractor (LCESA) framework.

Mathematical formulation:
    Transport operator:  U_{ij}^{(l)} = sum_k  alpha_bar_{ji}^{(k)}  P_{W_k}
    Curvature:           F_{ijk} = U_12 . U_23 . (U_13^exp)^{-1}
    Block norm:          ||F_{ijk}||_k = ||Q_k^T (F - I) Q_k||_F

GPU optimizations:
    - Pre-compute P_k projection matrices per layer (shared across conversations)
    - Batch conversations via torch.einsum for transport operators
    - torch.bmm for batched curvature tensor computation
    - torch.linalg.inv for batched matrix inversion
"""

import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from experiments.config import LAYER_NAMES, MODELS, CACHE_DIR, RESULTS_DIR

# Regularization constants
EPSILON_PROJ = 1e-6   # added to each P_k during U construction
EPSILON_INV = 1e-4    # added to U_exp before inversion (Tikhonov)


def _pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Core transport / curvature functions
# ---------------------------------------------------------------------------

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
            bases.append(np.eye(d_model, dtype=np.float32))
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


def _build_P_stack(
    value_matrices_t: torch.Tensor,
    gamma_t: torch.Tensor,
    layer: int,
    device: torch.device,
) -> torch.Tensor:
    """Pre-compute P_k projection matrices for a given layer.

    Returns shape ``(n_standpoint, d_model, d_model)`` — one P_k per
    standpoint layer, with small diagonal regularization already added.
    """
    n_standpoint = len(LAYER_NAMES)
    d_model = value_matrices_t.shape[2]
    I_d = torch.eye(d_model, device=device, dtype=torch.float32)
    P_list = []

    for k in range(n_standpoint):
        head_mask = torch.where(gamma_t == k)[0]
        if len(head_mask) == 0:
            P_list.append(EPSILON_PROJ * I_d)
            continue

        # V_heads: (n_assigned, d_model, d_head)
        V_heads = value_matrices_t[layer, head_mask]
        # V @ V^T for each head -> (n_assigned, d_model, d_model)
        projections = torch.einsum("hij,hkj->hik", V_heads, V_heads)
        P_k = projections.mean(dim=0)  # (d_model, d_model)
        P_k = P_k + EPSILON_PROJ * I_d
        P_list.append(P_k)

    return torch.stack(P_list, dim=0)  # (n_standpoint, d_model, d_model)


def _batched_transport(
    attention_batch_t: torch.Tensor,
    P_stack_t: torch.Tensor,
    gamma_t: torch.Tensor,
    event_from: int,
    event_to: int,
    layer: int,
) -> torch.Tensor:
    """Batched transport operator computation on GPU.

    Parameters
    ----------
    attention_batch_t : (B, n_heads, n_events, n_events) on GPU
    P_stack_t : (n_standpoint, d_model, d_model) on GPU
    gamma_t : (n_heads,) on GPU
    event_from, event_to, layer : int

    Returns
    -------
    (B, d_model, d_model) on GPU
    """
    # Select the layer: attention may be (B, n_layers, n_heads, n_events, n_events)
    # or already (B, n_heads, n_events, n_events)
    if attention_batch_t.dim() == 5:
        attn_layer = attention_batch_t[:, layer]          # (B, n_heads, n_events, n_events)
    else:
        attn_layer = attention_batch_t                     # (B, n_heads, n_events, n_events)

    # alpha: (B, n_heads) — extract attention weight at (event_to, event_from)
    n_events = attn_layer.shape[2]
    flat_idx = event_to * n_events + event_from
    raw = attn_layer.reshape(attn_layer.shape[0], attn_layer.shape[1], -1)[:, :, flat_idx]  # (B, n_heads)

    n_standpoint = P_stack_t.shape[0]
    B = raw.shape[0]
    alpha = torch.stack(
        [raw[:, gamma_t == k].mean(dim=1) if (gamma_t == k).any()
         else torch.zeros(B, device=raw.device, dtype=raw.dtype)
         for k in range(n_standpoint)],
        dim=1,
    )
    # U = sum_k alpha_k * P_k  via einsum
    return torch.einsum("ck,klm->clm", alpha, P_stack_t)


def _batched_curvature(
    U_12: torch.Tensor,
    U_23: torch.Tensor,
    U_exp_inv: torch.Tensor,
    proj_bases: List[np.ndarray],
    device: torch.device,
) -> Tuple[torch.Tensor, List[np.ndarray]]:
    """Batched curvature tensor and block norms on GPU.

    Parameters
    ----------
    U_12, U_23, U_exp_inv : (B, d_model, d_model) on GPU
    proj_bases : list of (d_model, rank_k) numpy arrays

    Returns
    -------
    F : (B, d_model, d_model) on GPU
    block_norms : list of (B,) numpy arrays
    """
    F = torch.bmm(torch.bmm(U_12, U_23), U_exp_inv)  # (B, d_model, d_model)
    deviation = F - torch.eye(F.shape[1], device=device, dtype=torch.float32)

    # Pre-convert proj_bases to GPU tensors once
    proj_bases_t = [torch.as_tensor(Q_k, device=device, dtype=torch.float32) for Q_k in proj_bases]

    block_norms = []
    for Q_t in proj_bases_t:
        # deviation @ Q_k -> (B, d_model, rank_k)
        # Q_k^T @ (deviation @ Q_k) -> (B, rank_k, rank_k)
        proj_dev = torch.bmm(
            Q_t.T.unsqueeze(0).expand(deviation.shape[0], -1, -1),
            torch.bmm(deviation, Q_t.unsqueeze(0).expand(deviation.shape[0], -1, -1)),
        )
        # Frobenius norm per batch element
        norms = torch.linalg.vector_norm(proj_dev.reshape(proj_dev.shape[0], -1), dim=1)
        block_norms.append(norms.cpu().numpy())

    return F, block_norms


# ---------------------------------------------------------------------------
# CPU functions (backward-compatible, used by ablation & tests)
# ---------------------------------------------------------------------------

def compute_transport_operator(
    attention: np.ndarray,
    value_matrices: np.ndarray,
    gamma: np.ndarray,
    event_from: int,
    event_to: int,
    layer: int,
) -> np.ndarray:
    """Compute the transport operator U_{ij}^{(l)} for a single model layer (CPU).

    Used by ablation study and unit tests. For the main pipeline, the GPU-
    accelerated ``_batched_transport`` is used instead.
    """
    n_heads = attention.shape[1]
    d_model = value_matrices.shape[2]
    n_standpoint = len(LAYER_NAMES)

    U = np.zeros((d_model, d_model), dtype=np.float32)
    I_d = np.eye(d_model, dtype=np.float32)

    for k in range(n_standpoint):
        head_mask = np.where(gamma == k)[0]
        if len(head_mask) == 0:
            continue
        alpha_k = attention[layer, head_mask, event_to, event_from].mean()
        V_heads = value_matrices[layer, head_mask]
        projections = np.einsum("hij,hkj->hik", V_heads, V_heads)
        P_k = projections.mean(axis=0)
        P_k += EPSILON_PROJ * I_d
        U += alpha_k * P_k

    return U


def compute_curvature(
    U_12: np.ndarray,
    U_23: np.ndarray,
    U_exp: np.ndarray,
    projection_bases: List[np.ndarray] | None = None,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Compute the curvature tensor F and its block-specific norms (CPU).

    Used by ablation study and unit tests. For the main pipeline, the GPU-
    accelerated ``_batched_curvature`` is used instead.
    """
    d_model = U_12.shape[0]
    n_standpoint = len(LAYER_NAMES)

    U_exp_reg = U_exp + EPSILON_INV * np.eye(d_model, dtype=np.float32)

    try:
        U_exp_inv = np.linalg.inv(U_exp_reg)
    except np.linalg.LinAlgError:
        U_exp_inv = np.linalg.pinv(U_exp_reg)

    F = U_12 @ U_23 @ U_exp_inv

    block_norms: Dict[str, float] = {}
    deviation = F - np.eye(d_model, dtype=np.float32)

    for idx, name in enumerate(LAYER_NAMES):
        if projection_bases is not None:
            Q_k = projection_bases[idx]
            F_projected = Q_k.T @ deviation @ Q_k
            norm = float(np.linalg.norm(F_projected, "fro"))
        else:
            block_size = d_model // n_standpoint
            start = idx * block_size
            end = d_model if idx == n_standpoint - 1 else (idx + 1) * block_size
            block_dim = end - start
            F_block = F[start:end, start:end]
            I_block = np.eye(block_dim, dtype=np.float32)
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
    """Compute the mean transport product U_12 @ U_23 for T1 test conversations (CPU).

    Used by ablation study. For the main pipeline, the GPU-accelerated path
    in ``run_curvature_computation`` is used instead.
    """
    products = []

    for conv_id, fields in test_activations.items():
        if not conv_id.startswith("T1/"):
            continue

        attn = fields["attention"]
        V = value_matrices

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
# Full pipeline (GPU-accelerated)
# ---------------------------------------------------------------------------

def run_curvature_computation(
    model_name: str,
    activations_path: Path,
    gamma_path: Path,
    output_dir: Path = RESULTS_DIR,
    batch_size: int = 16,
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
    batch_size : int
        Number of conversations to process simultaneously on GPU.

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

    device = _pick_device()
    print(f"Using device: {device}")

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

    # GPU tensors shared across layers
    gamma_t = torch.as_tensor(gamma, device=device, dtype=torch.long)
    value_matrices_t = torch.as_tensor(
        sample_V.astype(np.float32), device=device, dtype=torch.float32
    )

    rows = []
    total = n_layers * len(all_conv_ids)
    done = 0

    for layer in range(n_layers):
        # Pre-compute P_k stack for this layer (shared across all conversations)
        P_stack_t = _build_P_stack(value_matrices_t, gamma_t, layer, device)

        # Gamma-aligned projection bases for block norms
        proj_bases = compute_projection_bases(
            sample_V.astype(np.float32), gamma, layer
        )

        # --- Compute U_exp from T1 baseline conversations ---
        t1_ids = [c for c in all_conv_ids if c.startswith("T1/")]
        if not t1_ids:
            raise ValueError("No T1 test conversations found for baseline computation.")

        t1_products = []
        for conv_id in t1_ids:
            attn = torch.as_tensor(
                test_activations[conv_id]["attention"].astype(np.float32),
                device=device, dtype=torch.float32,
            )
            u12 = _batched_transport(
                attn.unsqueeze(0), P_stack_t, gamma_t, 0, 1, layer
            )
            u23 = _batched_transport(
                attn.unsqueeze(0), P_stack_t, gamma_t, 1, 2, layer
            )
            t1_products.append(torch.bmm(u12, u23).squeeze(0))

        U_exp = torch.stack(t1_products, dim=0).mean(dim=0)  # (d_model, d_model)
        U_exp_reg = U_exp + EPSILON_INV * torch.eye(
            U_exp.shape[0], device=device, dtype=torch.float32
        )
        U_exp_inv = torch.linalg.inv(U_exp_reg)  # (d_model, d_model)

        # --- Batch curvature computation for ALL conversations ---
        for batch_start in range(0, len(all_conv_ids), batch_size):
            batch_ids = all_conv_ids[batch_start : batch_start + batch_size]
            B = len(batch_ids)

            # Stack attention tensors: (B, n_heads, n_events, n_events)
            attn_list = []
            for conv_id in batch_ids:
                attn_np = test_activations[conv_id]["attention"].astype(np.float32)
                attn_list.append(torch.as_tensor(attn_np, device=device, dtype=torch.float32))
            attn_batch = torch.stack(attn_list, dim=0)

            # Batched transport operators
            U_12 = _batched_transport(attn_batch, P_stack_t, gamma_t, 0, 1, layer)
            U_23 = _batched_transport(attn_batch, P_stack_t, gamma_t, 1, 2, layer)

            # Expand U_exp_inv for batched matmul
            U_exp_inv_batch = U_exp_inv.unsqueeze(0).expand(B, -1, -1)

            # Curvature and block norms
            F, block_norms_list = _batched_curvature(
                U_12, U_23, U_exp_inv_batch, proj_bases, device
            )

            # Extract results
            for i, conv_id in enumerate(batch_ids):
                scenario = conv_id.split("/")[0]
                curvatures = {}
                for idx, name in enumerate(LAYER_NAMES):
                    curvatures[name] = float(block_norms_list[idx][i])
                curvature_total = float(
                    np.sqrt(sum(v ** 2 for v in curvatures.values()))
                )

                row = {"conversation_id": conv_id, "scenario": scenario, "layer": layer}
                for name in LAYER_NAMES:
                    row[f"curvature_{name}"] = curvatures[name]
                row["curvature_total"] = curvature_total
                rows.append(row)

            done += B
            print(f"  ... processed {done}/{total} combinations", flush=True)

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


def compute_projection_bases_gpu(
    value_matrices: np.ndarray,
    gamma: np.ndarray,
    layer: int,
    device: torch.device,
    energy_threshold: float = 0.9,
) -> list:
    """GPU-accelerated version of compute_projection_bases using torch.linalg.svd."""
    d_model = value_matrices.shape[2]
    n_standpoint = len(LAYER_NAMES)
    bases = []
    for k in range(n_standpoint):
        head_indices = np.where(gamma == k)[0]
        if len(head_indices) == 0:
            bases.append(np.eye(d_model, dtype=np.float32))
            continue
        V_concat = np.concatenate(
            [value_matrices[layer, h] for h in head_indices], axis=1
        )
        V_t = torch.as_tensor(V_concat, device=device, dtype=torch.float32)
        U, s, _ = torch.linalg.svd(V_t, full_matrices=False)
        s_np = s.cpu().numpy()
        energy = np.cumsum(s_np ** 2) / np.sum(s_np ** 2)
        rank = int(np.searchsorted(energy, energy_threshold)) + 1
        rank = min(rank, len(s_np))
        Q_k = U[:, :rank].cpu().numpy()
        bases.append(Q_k)
        del V_t, U, s
    return bases


def build_layer_cache(
    V: np.ndarray,
    gamma: np.ndarray,
    t1_activations: dict,
    device: torch.device,
) -> list:
    """Pre-compute P_stack, proj_bases, U_exp_inv for all layers.
    
    Use when gamma is fixed (causal_patching, ablation) to avoid
    recomputing these expensive quantities on every holonomy call.
    
    Returns list of dicts, one per layer, each with keys:
        P_stack, proj_bases, U_exp_inv
    """
    n_layers = V.shape[0]
    gamma_t = torch.as_tensor(gamma, device=device, dtype=torch.long)
    t1_ids = sorted(t1_activations.keys())

    cache = []
    print(f"  Building layer cache ({n_layers} layers) ...")
    for layer in range(n_layers):
        V_layer = np.array(V[layer]).astype(np.float32)
        V_layer_3d = V_layer[np.newaxis, ...]
        V_layer_t = torch.as_tensor(V_layer_3d, device=device, dtype=torch.float32)

        P_stack = _build_P_stack(V_layer_t, gamma_t, 0, device)
        proj_bases = compute_projection_bases_gpu(V_layer_3d, gamma, 0, device)

        # Pre-compute U_exp_inv from T1 baseline
        t1_attn_list = [
            torch.as_tensor(t1_activations[cid]["attention"].astype(np.float32), device=device)
            for cid in t1_ids
        ]
        t1_batch = torch.stack(t1_attn_list, dim=0)
        del t1_attn_list

        u12 = _batched_transport(t1_batch, P_stack, gamma_t, 0, 1, layer)
        u23 = _batched_transport(t1_batch, P_stack, gamma_t, 1, 2, layer)
        U_exp = torch.bmm(u12, u23).mean(dim=0)
        del t1_batch, u12, u23

        U_exp_inv = torch.linalg.inv(
            U_exp + EPSILON_INV * torch.eye(U_exp.shape[0], device=device, dtype=torch.float32)
        )
        del U_exp, V_layer_t, V_layer, V_layer_3d

        cache.append({
            "P_stack": P_stack,
            "proj_bases": proj_bases,
            "U_exp_inv": U_exp_inv,
        })
        if (layer + 1) % 8 == 0:
            print(f"    cached {layer+1}/{n_layers} layers")

    del gamma_t
    torch.cuda.empty_cache()
    print(f"  Layer cache ready.")
    return cache
