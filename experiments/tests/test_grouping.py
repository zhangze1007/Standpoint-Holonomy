"""Unit tests for head grouping."""
import numpy as np
import pytest

from experiments.grouping.head_grouping import (
    compute_attention_differential,
    assign_heads_to_layers,
)

def test_attention_differential_shape():
    """Differential should have shape (n_layers, n_heads)."""
    n_layers, n_heads, n_events = 2, 4, 3
    activations = {}

    for i in range(3):
        attn = np.random.rand(n_layers, n_heads, n_events, n_events) * 0.1
        activations[f"T2/grouping/{i}"] = {"attention": attn}

    for i in range(3):
        attn = np.random.rand(n_layers, n_heads, n_events, n_events) * 0.1
        activations[f"T1/grouping/{i}"] = {"attention": attn}

    delta = compute_attention_differential(activations, "T2", "T1")
    assert delta.shape == (n_layers, n_heads)

def test_attention_differential_identifies_active_heads():
    """Heads with higher attention in positive scenario should have positive differential."""
    n_layers, n_heads, n_events = 1, 4, 3
    activations = {}

    for i in range(5):
        attn = np.random.rand(n_layers, n_heads, n_events, n_events) * 0.1
        attn[:, 0:2, 2, 0] = 0.8  # heads 0,1 have high attention
        activations[f"T2/grouping/{i}"] = {"attention": attn}

    for i in range(5):
        attn = np.random.rand(n_layers, n_heads, n_events, n_events) * 0.1
        activations[f"T1/grouping/{i}"] = {"attention": attn}

    delta = compute_attention_differential(activations, "T2", "T1")
    # Heads 0,1 should have higher differential than heads 2,3
    assert np.mean(delta[:, 0:2]) > np.mean(delta[:, 2:4])

def test_assignment_balanced():
    """Assignment should distribute heads to their strongest layer."""
    n_heads = 20
    deltas = {
        "min": np.random.randn(1, n_heads) * 0.1,
        "nar": np.zeros((1, n_heads)),
        "soc": np.zeros((1, n_heads)),
        "mor": np.zeros((1, n_heads)),
        "pos": np.zeros((1, n_heads)),
    }
    deltas["nar"][0, :4] = 1.0   # heads 0-3 -> nar
    deltas["mor"][0, 4:8] = 1.0  # heads 4-7 -> mor

    gamma = assign_heads_to_layers(deltas, n_heads)
    assert len(gamma) == n_heads
    assert np.all(gamma[:4] == 1)   # nar = index 1
    assert np.all(gamma[4:8] == 3)  # mor = index 3

def test_assignment_returns_valid_indices():
    """All gamma values should be valid layer indices (0-4)."""
    n_heads = 12
    deltas = {name: np.random.randn(1, n_heads) for name in ["min", "nar", "soc", "mor", "pos"]}
    gamma = assign_heads_to_layers(deltas, n_heads)
    assert np.all(gamma >= 0) and np.all(gamma < 5)
