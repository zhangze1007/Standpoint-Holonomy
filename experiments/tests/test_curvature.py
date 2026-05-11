"""Unit tests for curvature computation."""
import numpy as np
import pytest

from experiments.curvature.compute import (
    compute_transport_operator,
    compute_curvature,
    compute_projection_bases,
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
    """When U_12 = U_23 = U_exp = I, curvature F ≈ I (within regularization)."""
    d = 8
    U = np.eye(d)
    F, block_norms = compute_curvature(U, U, U, projection_bases=None)
    # Tikhonov regularization shifts U_exp to (1 + eps)*I, so F ≈ (1/(1+eps))*I
    np.testing.assert_allclose(F, np.eye(d), atol=1e-3)


def test_curvature_nonzero():
    """When transport differs from expected, curvature should be non-zero."""
    d = 8
    U_exp = np.eye(d) * 0.5
    U_diff = np.eye(d) * 0.8
    F, block_norms = compute_curvature(U_diff, U_diff, U_exp, projection_bases=None)
    assert np.linalg.norm(F - np.eye(d)) > 0.1


def test_block_norms_nonnegative():
    """All block norms should be non-negative."""
    d = 8
    np.random.seed(42)
    U = np.eye(d) * 0.5 + np.random.randn(d, d) * 0.1
    F, block_norms = compute_curvature(U, U, U, projection_bases=None)
    for k_name, norm in block_norms.items():
        assert norm >= 0, f"Block norm for {k_name} should be non-negative"


def test_block_norms_keys():
    """Block norms should have keys matching LAYER_NAMES."""
    from experiments.config import LAYER_NAMES
    d = 10  # divisible by 5
    U = np.eye(d)
    _, block_norms = compute_curvature(U, U, U, projection_bases=None)
    assert set(block_norms.keys()) == set(LAYER_NAMES)


def test_curvature_gamma_aligned_blocks():
    """Gamma-aligned blocks should produce different norms than equal-width."""
    d = 8
    n_layers, n_heads, d_head = 1, 4, 2
    np.random.seed(42)

    # Create value matrices where heads 0,1 span dims 0-3 and heads 2,3 span dims 4-7
    V = np.zeros((n_layers, n_heads, d, d_head))
    V[0, 0, :4, :] = np.random.randn(4, d_head) * 0.5
    V[0, 1, :4, :] = np.random.randn(4, d_head) * 0.5
    V[0, 2, 4:, :] = np.random.randn(4, d_head) * 0.5
    V[0, 3, 4:, :] = np.random.randn(4, d_head) * 0.5

    gamma = np.array([0, 0, 1, 1])

    proj_bases = compute_projection_bases(V, gamma, layer=0, energy_threshold=0.9)

    # Each basis should have correct row dimension
    for k, Q in enumerate(proj_bases):
        assert Q.shape[0] == d
        assert Q.shape[1] <= d  # rank cannot exceed d_model

    # Compute curvature with aligned blocks
    U = np.eye(d) * 0.5 + np.random.randn(d, d) * 0.1
    _, block_norms_aligned = compute_curvature(U, U, U, proj_bases)
    _, block_norms_equal = compute_curvature(U, U, U, projection_bases=None)

    # The norms should differ (not be identical across block methods)
    aligned_vals = list(block_norms_aligned.values())
    equal_vals = list(block_norms_equal.values())
    assert not np.allclose(aligned_vals, equal_vals, atol=1e-6)


def test_curvature_regularized_inversion():
    """Regularized inversion should produce finite, reasonable curvature."""
    d = 8
    # Near-singular U_exp (ill-conditioned)
    U_exp = np.eye(d) * 1e-8
    U_diff = np.eye(d) * 0.5
    F, block_norms = compute_curvature(U_diff, U_diff, U_exp, projection_bases=None)
    assert np.all(np.isfinite(F))
    for name, norm in block_norms.items():
        assert np.isfinite(norm), f"Block norm for {name} should be finite"
        assert norm < 1e10, f"Block norm for {name} should be bounded (got {norm})"


def test_projection_bases_orthonormal():
    """Projection bases should be orthonormal (Q^T Q = I)."""
    d = 8
    n_layers, n_heads, d_head = 1, 4, 2
    np.random.seed(42)

    V = np.random.randn(n_layers, n_heads, d, d_head)
    gamma = np.array([0, 0, 1, 1])

    bases = compute_projection_bases(V, gamma, layer=0, energy_threshold=0.9)

    for Q in bases:
        # Q^T Q should be close to identity
        gram = Q.T @ Q
        np.testing.assert_allclose(gram, np.eye(Q.shape[1]), atol=1e-10)
