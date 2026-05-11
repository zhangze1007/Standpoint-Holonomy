"""Tests for ablation studies."""
import numpy as np
from experiments.ablation import ablate_layers, ablate_sequence_length


def test_ablate_layers_reduces_layer_count():
    """Ablating layers should produce curvature with fewer layer groups."""
    attn = np.random.rand(12, 4, 5, 5).astype(np.float32)
    V = np.random.rand(12, 4, 768, 64).astype(np.float32)
    gamma = np.array([0, 1, 2, 3])
    attn_abl, V_abl = ablate_layers(attn, V, n_keep=5)
    assert attn_abl.shape[0] == 5
    assert V_abl.shape[0] == 5


def test_ablate_sequence_length_shortens():
    """Sequence ablation should reduce event count."""
    attn = np.random.rand(12, 4, 5, 5).astype(np.float32)
    attn_abl = ablate_sequence_length(attn, n_events=3)
    assert attn_abl.shape[2] == 3
    assert attn_abl.shape[3] == 3


def test_ablate_layers_n_keep_must_be_less_than_total():
    """Requesting more layers than available should raise."""
    attn = np.random.rand(12, 4, 5, 5).astype(np.float32)
    V = np.random.rand(12, 4, 768, 64).astype(np.float32)
    try:
        ablate_layers(attn, V, n_keep=15)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
