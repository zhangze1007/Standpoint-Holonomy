"""Tests for H4 and H5 hypothesis tests."""
import numpy as np
import pandas as pd
from experiments.stats.hypothesis_tests import h4_cka_discrimination, h5_ablation_sensitivity


def _make_cka_df(n_per_scenario=10):
    """Create synthetic CKA DataFrame."""
    rows = []
    for scenario in ["T1", "T2", "T3", "T4", "T5"]:
        for i in range(n_per_scenario):
            for layer in range(12):
                cka = np.random.uniform(0.1, 0.9)
                rows.append({
                    "conversation_id": f"{scenario}/test/{i}",
                    "scenario": scenario,
                    "layer": layer,
                    "cka": cka,
                })
    return pd.DataFrame(rows)


def _make_ablation_df():
    """Create synthetic ablation DataFrame."""
    rows = []
    for n_keep in [3, 5, 7, 12]:
        for scenario in ["T1", "T2", "T3", "T4", "T5"]:
            for i in range(5):
                rows.append({
                    "model": "gpt2",
                    "ablation_type": "layer_count",
                    "param_value": n_keep,
                    "conversation_id": f"{scenario}/test/{i}",
                    "scenario": scenario,
                    "layer": 0,
                    "curvature_total": np.random.uniform(30, 50),
                })
    return pd.DataFrame(rows)


def test_h4_returns_per_layer_results():
    """H4 should return Kruskal-Wallis results per CKA layer."""
    df = _make_cka_df()
    result = h4_cka_discrimination(df)
    assert "per_layer" in result
    assert len(result["per_layer"]) == 12


def test_h5_returns_per_ablation_results():
    """H5 should return results per ablation level."""
    df = _make_ablation_df()
    result = h5_ablation_sensitivity(df)
    assert "per_param" in result
    assert len(result["per_param"]) == 4
