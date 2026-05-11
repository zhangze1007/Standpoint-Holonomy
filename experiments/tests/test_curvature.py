"""Unit tests for curvature computation."""
import numpy as np
import pytest

from experiments.curvature.compute import (
    compute_transport_operator,
    compute_curvature,
)

def test_transport_operator_shape():
    """Transport operator should have correct shape."""
    n_layers, n_heads, n_events = 1, 4, 3
    d_model, d_head = 8, 2

    attention = np.ones((n_layers, n_heads, n_events, n_events)) / n_events
    value_matrices = np.zeros((n_layers, n_heads, d_model, d_head))
    for h in range(n_heads):
        value_matrices[0, h, :d_head, :d_head] = np.eye(d_head)

    gamma = np.array([0, 0, 1, 1])
    U = compute_transport_operator(attention, value_matrices, gamma, 0, 2, 0)
    assert U.shape == (d_model, d_model)

def test_transport_operator_nonzero():
    """Transport operator should be non-zero with non-trivial input."""
    n_layers, n_heads, n_events = 1, 4, 3
    d_model, d_head = 8, 2

    attention = np.ones((n_layers, n_heads, n_events, n_events)) / n_events
    value_matrices = np.zeros((n_layers, n_heads, d_model, d_head))
    for h in range(n_heads):
        value_matrices[0, h, :d_head, :d_head] = np.eye(d_head)

    gamma = np.array([0, 0, 1, 1])
    U = compute_transport_operator(attention, value_matrices, gamma, 0, 2, 0)
    assert np.linalg.norm(U) > 0

def test_curvature_identity_transport():
    """When U_12 = U_23 = U_exp = I, curvature F = I@I@I^{-1} = I."""
    d = 8
    U = np.eye(d)
    F, block_norms = compute_curvature(U, U, U)
    np.testing.assert_allclose(F, np.eye(d), atol=1e-10)

def test_curvature_nonzero():
    """When transport differs from expected, curvature should be non-zero."""
    d = 8
    U_exp = np.eye(d) * 0.5
    U_diff = np.eye(d) * 0.8
    F, block_norms = compute_curvature(U_diff, U_diff, U_exp)
    assert np.linalg.norm(F - np.eye(d)) > 0.1

def test_block_norms_nonnegative():
    """All block norms should be non-negative."""
    d = 8
    np.random.seed(42)
    U = np.eye(d) * 0.5 + np.random.randn(d, d) * 0.1
    F, block_norms = compute_curvature(U, U, U)
    for k_name, norm in block_norms.items():
        assert norm >= 0, f"Block norm for {k_name} should be non-negative"

def test_block_norms_keys():
    """Block norms should have keys matching LAYER_NAMES."""
    from experiments.config import LAYER_NAMES
    d = 10  # divisible by 5
    U = np.eye(d)
    _, block_norms = compute_curvature(U, U, U)
    assert set(block_norms.keys()) == set(LAYER_NAMES)
