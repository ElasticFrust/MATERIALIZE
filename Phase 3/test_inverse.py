"""
Tests for Phase 3 inverse optimization.

Verifies that:
1. Softplus parameterization is invertible and stays positive.
2. A single optimization run can recover unit rigidities from the ground-truth tensor.
3. Multi-start campaign converges and produces consistent solutions.
4. Round-trip validation passes (forward(optimized_k) ≈ target).
"""

import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "Phase 2"))
sys.path.insert(0, str(PROJECT_ROOT / "Phase 3"))

import Disc_2_Cont_optimized as D2C
from forward_solver_torch import from_triangulation
from inverse_optimize import (
    raw_to_rigidities,
    rigidities_to_raw,
    run_single_optimization,
    run_campaign,
    validate_round_trip,
    tensor_mse_loss,
)


def _make_solver(size=(3, 3), eta=0.2, seed=7):
    np.random.seed(seed)
    tri = D2C.generate_foam_points(size=size, eta=eta)
    solver, default_rigs, default_rl = from_triangulation(tri)
    return solver, default_rigs, len(tri.simplices)


# ---- Test 1: Softplus parameterization ----

def test_softplus_roundtrip():
    """softplus(inv_softplus(k)) == k for positive k."""
    print("Test 1: Softplus roundtrip ... ", end="", flush=True)
    k = torch.tensor([0.01, 0.1, 0.5, 1.0, 2.0, 10.0], dtype=torch.float64)
    raw = rigidities_to_raw(k)
    k_back = raw_to_rigidities(raw)
    err = (k - k_back).abs().max().item()
    assert err < 1e-10, f"Roundtrip error: {err}"
    assert (raw_to_rigidities(torch.randn(100, dtype=torch.float64)) > 0).all(), \
        "softplus should always be positive"
    print(f"PASS  (max error = {err:.2e})")


# ---- Test 2: Single optimization recovers unit rigidities ----

def test_single_optimization():
    """Starting from random k, can we recover the tensor from unit rigidities?"""
    print("Test 2: Single optimization ... ", end="", flush=True)
    solver, default_rigs, n_tri = _make_solver()

    with torch.no_grad():
        gt = solver(default_rigs, rest_lengths=None)
    target = gt['elastic_tensor']

    result = run_single_optimization(
        solver=solver,
        target_tensor=target,
        n_triangles=n_tri,
        max_iter=500,
        lr=0.05,
        tol=1e-10,
        optimizer_type='lbfgs',
        seed=0,
        verbose=False,
    )

    assert result['rel_error'] < 1e-3, \
        f"Relative error too high: {result['rel_error']:.3e}"
    print(f"PASS  (rel_error = {result['rel_error']:.3e}, "
          f"loss = {result['final_loss']:.3e}, "
          f"iters = {result['iterations']})")


# ---- Test 3: Adam optimizer also works ----

def test_adam_optimization():
    """Adam should also converge, though perhaps more slowly."""
    print("Test 3: Adam optimizer ... ", end="", flush=True)
    solver, default_rigs, n_tri = _make_solver()

    with torch.no_grad():
        gt = solver(default_rigs, rest_lengths=None)
    target = gt['elastic_tensor']

    result = run_single_optimization(
        solver=solver,
        target_tensor=target,
        n_triangles=n_tri,
        max_iter=1000,
        lr=0.01,
        tol=1e-8,
        optimizer_type='adam',
        seed=0,
        verbose=False,
    )

    # Adam may not converge as tightly, allow looser bound
    assert result['rel_error'] < 0.05, \
        f"Adam relative error too high: {result['rel_error']:.3e}"
    print(f"PASS  (rel_error = {result['rel_error']:.3e}, "
          f"loss = {result['final_loss']:.3e})")


# ---- Test 4: Round-trip validation ----

def test_round_trip():
    """Forward(optimized_k) should reproduce the target tensor."""
    print("Test 4: Round-trip validation ... ", end="", flush=True)
    solver, default_rigs, n_tri = _make_solver()

    with torch.no_grad():
        gt = solver(default_rigs, rest_lengths=None)
    target = gt['elastic_tensor']

    result = run_single_optimization(
        solver=solver,
        target_tensor=target,
        n_triangles=n_tri,
        max_iter=500,
        tol=1e-10,
        optimizer_type='lbfgs',
        seed=42,
        verbose=False,
    )

    rt = validate_round_trip(solver, result['rigidities'], target.numpy())
    assert rt['rel_error'] < 1e-3, \
        f"Round-trip error too high: {rt['rel_error']:.3e}"
    print(f"PASS  (round-trip rel_error = {rt['rel_error']:.3e})")


# ---- Test 5: Mini campaign ----

def test_mini_campaign():
    """Small campaign (5 starts) to verify the full pipeline."""
    print("Test 5: Mini campaign (5 starts) ... ", end="", flush=True)
    solver, default_rigs, n_tri = _make_solver()

    with torch.no_grad():
        gt = solver(default_rigs, rest_lengths=None)
    target = gt['elastic_tensor']

    campaign = run_campaign(
        solver=solver,
        target_tensor=target,
        n_triangles=n_tri,
        n_starts=5,
        max_iter=300,
        tol=1e-10,
        optimizer_type='lbfgs',
        verbose=False,
    )

    assert campaign['n_converged'] >= 1, "At least one run should converge"
    assert campaign['mean_rel_error'] < 0.01, \
        f"Mean relative error too high: {campaign['mean_rel_error']:.3e}"
    print(f"PASS  (converged {campaign['n_converged']}/5, "
          f"clusters = {campaign['n_clusters']}, "
          f"mean_rel_err = {campaign['mean_rel_error']:.3e})")


# ---- Test 6: Gradient flows correctly ----

def test_gradient_flow():
    """Verify that gradients from the loss flow back to raw parameters."""
    print("Test 6: Gradient flow ... ", end="", flush=True)
    solver, default_rigs, n_tri = _make_solver()

    with torch.no_grad():
        gt = solver(default_rigs, rest_lengths=None)
    target = gt['elastic_tensor'].detach()

    raw = torch.randn(n_tri, 3, dtype=torch.float64, requires_grad=True)
    k = raw_to_rigidities(raw)
    result = solver(k, rest_lengths=None)
    loss = tensor_mse_loss(result['elastic_tensor'], target)
    loss.backward()

    assert raw.grad is not None, "No gradient computed"
    assert raw.grad.shape == (n_tri, 3), f"Wrong grad shape: {raw.grad.shape}"
    assert not torch.all(raw.grad == 0), "Gradient is all zeros"
    print(f"PASS  (grad norm = {raw.grad.norm().item():.3e})")


# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("=" * 60)
    print("Phase 3 — Inverse Optimization Tests")
    print("=" * 60)

    test_softplus_roundtrip()
    test_single_optimization()
    test_adam_optimization()
    test_round_trip()
    test_mini_campaign()
    test_gradient_flow()

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
