"""
LCESA Ablation Studies
======================
Tests robustness of curvature signals by:
1. Dropping model layers (keeping evenly-spaced subset)
2. Shortening event sequences (removing middle events)
"""
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from experiments.config import (
    LAYER_NAMES, MODELS, CACHE_DIR, RESULTS_DIR, DATA_DIR,
    ABLATION_LAYER_COUNTS, ABLATION_LENGTHS,
)
from experiments.curvature.compute import (
    compute_transport_operator,
    compute_curvature,
    compute_baseline_transport,
    compute_projection_bases,
)


def ablate_layers(
    attention: np.ndarray,
    value_matrices: np.ndarray,
    n_keep: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Keep evenly-spaced model layers, drop the rest."""
    n_layers = attention.shape[0]
    if n_keep > n_layers:
        raise ValueError(f"n_keep={n_keep} > n_layers={n_layers}")
    indices = np.linspace(0, n_layers - 1, n_keep, dtype=int)
    return attention[indices], value_matrices[indices]


def ablate_sequence_length(
    attention: np.ndarray,
    n_events: int,
) -> np.ndarray:
    """Reduce event sequence to first n_events."""
    return attention[:, :, :n_events, :n_events]


def run_ablation_study(
    model_name: str,
    activations_path: Path,
    gamma_path: Path,
    output_dir: Path = RESULTS_DIR,
) -> pd.DataFrame:
    """Run full ablation study: layer count x sequence length grid."""
    model_config = MODELS[model_name]
    raw = np.load(activations_path, allow_pickle=False)
    grouping = np.load(gamma_path, allow_pickle=False)
    gamma = grouping["gamma"]

    # Reorganize test activations
    test_activations: Dict[str, Dict[str, np.ndarray]] = {}
    conv_ids_seen: set = set()
    for key in raw.files:
        parts = key.rsplit("/", 1)
        conv_id, field = parts[0], parts[1]
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

    rows = []

    # --- Layer ablation ---
    for n_keep in ABLATION_LAYER_COUNTS:
        if n_keep > model_config.n_layers:
            continue
        print(f"  Layer ablation: n_keep={n_keep}")
        for conv_id in sorted(test_activations.keys()):
            fields = test_activations[conv_id]
            attn_abl, V_abl = ablate_layers(fields["attention"], fields["value_matrices"], n_keep)
            gamma_adj = gamma[:attn_abl.shape[1]]

            t1_fields = next(
                (v for k, v in test_activations.items() if k.startswith("T1/")),
                None,
            )
            if t1_fields is None:
                continue
            t1_attn, t1_V = ablate_layers(t1_fields["attention"], t1_fields["value_matrices"], n_keep)

            for layer in range(n_keep):
                U_exp = compute_baseline_transport(
                    {"T1/tmp": {"attention": t1_attn, "value_matrices": t1_V}},
                    "T1", layer, gamma_adj,
                )
                proj_bases = compute_projection_bases(V_abl, gamma_adj, layer)
                U_12 = compute_transport_operator(attn_abl, V_abl, gamma_adj, 0, 1, layer)
                U_23 = compute_transport_operator(attn_abl, V_abl, gamma_adj, 1, 2, layer)
                _, block_norms = compute_curvature(U_12, U_23, U_exp, proj_bases)

                scenario = conv_id.split("/")[0]
                row = {
                    "model": model_name,
                    "ablation_type": "layer_count",
                    "param_value": n_keep,
                    "conversation_id": conv_id,
                    "scenario": scenario,
                    "layer": layer,
                }
                for name in LAYER_NAMES:
                    row[f"curvature_{name}"] = block_norms[name]
                row["curvature_total"] = float(np.sqrt(sum(v**2 for v in block_norms.values())))
                rows.append(row)

    # --- Sequence length ablation ---
    for n_events in ABLATION_LENGTHS:
        if n_events > 5:
            continue
        print(f"  Sequence ablation: n_events={n_events}")
        for conv_id in sorted(test_activations.keys()):
            fields = test_activations[conv_id]
            attn_abl = ablate_sequence_length(fields["attention"], n_events)
            V_full = fields["value_matrices"]

            t1_fields = next(
                (v for k, v in test_activations.items() if k.startswith("T1/")),
                None,
            )
            if t1_fields is None:
                continue
            t1_attn_abl = ablate_sequence_length(t1_fields["attention"], n_events)

            for layer in range(model_config.n_layers):
                U_exp = compute_baseline_transport(
                    {"T1/tmp": {"attention": t1_attn_abl, "value_matrices": t1_fields["value_matrices"]}},
                    "T1", layer, gamma,
                )
                proj_bases = compute_projection_bases(V_full, gamma, layer)
                U_12 = compute_transport_operator(attn_abl, V_full, gamma, 0, 1, layer)
                last_event = min(2, n_events - 1)
                U_23 = compute_transport_operator(attn_abl, V_full, gamma, 1, last_event, layer)
                _, block_norms = compute_curvature(U_12, U_23, U_exp, proj_bases)

                scenario = conv_id.split("/")[0]
                row = {
                    "model": model_name,
                    "ablation_type": "sequence_length",
                    "param_value": n_events,
                    "conversation_id": conv_id,
                    "scenario": scenario,
                    "layer": layer,
                }
                for name in LAYER_NAMES:
                    row[f"curvature_{name}"] = block_norms[name]
                row["curvature_total"] = float(np.sqrt(sum(v**2 for v in block_norms.values())))
                rows.append(row)

    df = pd.DataFrame(rows)
    output_path = output_dir / f"{model_name}_ablation.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Ablation results: {len(df)} rows -> {output_path}")
    return df
