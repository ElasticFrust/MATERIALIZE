"""Minimal regression test for the forward solver (Phase 2/forward_solver_torch.py).

Anchors the current behaviour of the default `method='intrinsic'` solve and the
legacy `method='woodbury'` path. Run: python "Phase 2/test_forward_solver.py"

Checks:
  1. Regular triangular lattice, uniform k -> nu = 1/3, E finite/positive, no NaN.
  2. Disordered foam mesh -> finite outputs, no NaN.
  3. Differentiability: gradients flow through the dense intrinsic path (<=600 tri).
  4. Autograd matches finite differences for d(nu)/d(k) (coarse tol).
  5. Legacy method='woodbury' still runs.
  6. Large-N (>600 tri) adjoint path: grad-path forward == no-grad forward, and the adjoint
     gradient matches finite differences (Component A).
"""

import sys, os
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                       # forward_solver_torch
sys.path.insert(0, os.path.dirname(HERE))      # Disc_2_Cont_optimized

import Disc_2_Cont_optimized as D2C
from forward_solver_torch import from_triangulation


def _finite(t):
    return bool(torch.isfinite(t).all())


def test_regular_lattice_nu_third():
    np.random.seed(0)
    tri = D2C.generate_cryratl_points((3, 3), (1, 1), 0)
    solver, k, l0 = from_triangulation(tri)
    assert len(tri.simplices) <= 600, "expected dense differentiable path"
    res = solver(k, l0)                          # method='intrinsic' by default
    nu, E = res['poisson'], res['young']
    assert _finite(nu) and _finite(E), "NaN/inf in regular-lattice output"
    assert abs(nu.item() - 1.0 / 3.0) < 1e-3, f"nu={nu.item()} != 1/3"
    assert E.item() > 0, f"E={E.item()} not positive"
    print(f"  [1] regular lattice: nu={nu.item():.6f} (~0.3333), E={E.item():.4f}  OK")


def test_foam_finite():
    np.random.seed(1)
    tri = D2C.generate_foam_points((3, 3), 0.2)
    solver, k, l0 = from_triangulation(tri)
    res = solver(k, l0)
    assert _finite(res['poisson']) and _finite(res['young'])
    assert _finite(res['W']) and _finite(res['elastic_tensor'])
    print(f"  [2] foam eta=0.2 ({len(tri.simplices)} tri): "
          f"nu={res['poisson'].item():.4f}, E={res['young'].item():.4f}  OK")


def test_gradients_flow():
    np.random.seed(2)
    tri = D2C.generate_foam_points((3, 3), 0.15)
    solver, k, l0 = from_triangulation(tri)
    k = k.clone().requires_grad_(True)
    res = solver(k, l0)
    res['poisson'].backward()
    assert k.grad is not None and _finite(k.grad), "no/NaN gradient"
    assert k.grad.abs().sum().item() > 0, "gradient is all-zero"
    print(f"  [3] intrinsic gradient d(nu)/dk: finite, |grad|_1={k.grad.abs().sum().item():.3e}  OK")


def test_autograd_vs_fd():
    np.random.seed(3)
    tri = D2C.generate_foam_points((3, 3), 0.15)
    solver, k0, l0 = from_triangulation(tri)

    k = k0.clone().requires_grad_(True)
    nu = solver(k, l0)['poisson']
    nu.backward()
    g_auto = k.grad[0, 0].item()

    eps = 1e-6
    with torch.no_grad():
        kp = k0.clone(); kp[0, 0] += eps
        km = k0.clone(); km[0, 0] -= eps
        nup = solver(kp, l0)['poisson'].item()
        num = solver(km, l0)['poisson'].item()
    g_fd = (nup - num) / (2 * eps)

    denom = max(1.0, abs(g_fd))
    assert abs(g_auto - g_fd) / denom < 1e-4, f"autograd {g_auto} vs FD {g_fd}"
    print(f"  [4] autograd vs finite-diff: {g_auto:.6e} vs {g_fd:.6e}  OK")


def test_adjoint_large_N():
    import forward_solver_torch as fst
    np.random.seed(6)
    tri = D2C.generate_foam_points((10, 10), 0.15)
    solver, k0, l0 = from_triangulation(tri)
    ntri = len(tri.simplices)
    assert ntri > fst.INTRINSIC_DENSE_MAX, f"need >{fst.INTRINSIC_DENSE_MAX} tri, got {ntri}"

    with torch.no_grad():                            # forward-only path
        nu_ng = solver(k0, l0)['poisson'].item()
    k = k0.clone().requires_grad_(True)
    nu = solver(k, l0)['poisson']                    # differentiable adjoint path
    assert abs(nu.item() - nu_ng) < 1e-12, "grad-path forward != no-grad forward"
    nu.backward()
    assert k.grad is not None and _finite(k.grad) and k.grad.abs().sum().item() > 0

    g_auto = k.grad[0, 0].item()
    eps = 1e-6
    with torch.no_grad():
        kp = k0.clone(); kp[0, 0] += eps
        km = k0.clone(); km[0, 0] -= eps
        g_fd = (solver(kp, l0)['poisson'].item() - solver(km, l0)['poisson'].item()) / (2 * eps)
    assert abs(g_auto - g_fd) / max(1.0, abs(g_fd)) < 1e-4, f"adjoint {g_auto} vs FD {g_fd}"
    print(f"  [6] adjoint large-N ({ntri} tri): forward identical, autograd={g_auto:.4e} "
          f"vs FD={g_fd:.4e}  OK")


def test_woodbury_legacy_runs():
    np.random.seed(4)
    tri = D2C.generate_foam_points((3, 3), 0.2)
    solver, k, l0 = from_triangulation(tri)
    res = solver(k, l0, method='woodbury')
    assert _finite(res['poisson']) and _finite(res['young'])
    print(f"  [5] legacy woodbury: nu={res['poisson'].item():.4f}  OK")


if __name__ == '__main__':
    torch.manual_seed(0)
    tests = [
        test_regular_lattice_nu_third,
        test_foam_finite,
        test_gradients_flow,
        test_autograd_vs_fd,
        test_adjoint_large_N,
        test_woodbury_legacy_runs,
    ]
    print("forward_solver_torch regression test")
    failed = 0
    for t in tests:
        try:
            t()
        except Exception as e:
            failed += 1
            print(f"  FAIL {t.__name__}: {e}")
    print("ALL PASSED" if failed == 0 else f"{failed} TEST(S) FAILED")
    sys.exit(1 if failed else 0)
