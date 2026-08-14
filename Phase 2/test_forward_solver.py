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
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)                       # forward_solver_torch
sys.path.insert(0, ROOT)                       # Disc_2_Cont_optimized
sys.path.insert(0, os.path.join(ROOT, 'verification_tools'))   # independent physical oracle [7]

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


def test_elastic_tensor_vs_energy_hessian():
    """[7] The homogenisation contraction vs an INDEPENDENT physical oracle, COMPONENT BY COMPONENT.

    Why this test exists: the regular lattice has W ≡ 0 identically, so C(s) = A(s) there and the
    ν=1/3 / E=2/√3 gate is structurally blind to any error in how W is contracted. Scalar ν,E is
    weakly sensitive to the shear-shear entry, so it hides such an error too. This checks the whole
    TENSOR against `physical_homog.energy_C` — the Hessian of the relaxed energy of the same spring
    network, a genuinely different code path — on DISORDERED meshes, where W ≠ 0.

    `_compute_actual_elastic_tensor` is fed the SIMULATION's measured W (not the solver's own), so
    the check isolates the contraction: same A(s), same W, two independent routes to C_eff.
    """
    import scipy.sparse.linalg as spla
    import forward_solver_torch as fst
    import physical_homog as PH
    import test_cluster_rigidity as TR
    import test_cluster_Ceff as CE
    import test_cluster_VD as VD

    Dgt = [F.T @ F - np.eye(2) for F in PH.Fk]                      # the 3 macro metric-change modes
    Dinv = np.linalg.inv(np.stack([CE.vec3(g) for g in Dgt], 1))    # vec3 mode matrix, inverted once

    # (N, eta, seed, VD contrast or None for uniform k); eta=0 uniform is the BLIND control (W=0)
    cases = [(8, 0.00, 0, None), (8, 0.20, 1, None), (8, 0.20, 1, 5),
             (8, 0.35, 2, None), (12, 0.30, 3, 10)]
    worst = 0.0
    for N, eta, seed, vd in cases:
        mesh = VD.build_geometry(N, eta, seed)
        if vd is None:
            mesh['bond_k'] = np.ones(len(mesh['bond_R']))
            mesh['tri_k'] = mesh['bond_k'][mesh['tri_bond']]
        else:
            VD.set_VD(mesh, vd)                                     # k = 1 + tanh(a·(|R|−1))
        nn, nt = len(mesh['pts']), len(mesh['simplices'])
        free = np.arange(2, 2 * nn)                                 # pin node 0

        # W measured from the SIM's relaxation: delta_g(s) = W(s) Delta_g  ->  W3 = D · Dinv
        u_modes = PH.relax(mesh, free, TR.assemble_K_faff)
        D = np.zeros((nt, 3, 3))
        for j, (F, u) in enumerate(zip(PH.Fk, u_modes)):
            D[:, :, j] = CE.vec3(
                CE.tri_metric_change(mesh['edge_vecs'], mesh['simplices'], F, u) - Dgt[j])
        W3 = D @ Dinv

        c6 = fst._compute_actual_elastic_tensor(
            torch.as_tensor(TR.bare_tensor(mesh)),
            torch.as_tensor(W3.reshape(-1, 9))).numpy()
        c = c6.mean(0) * (8.0 * nt / mesh['areas'].sum())           # unweighted mean → physical units
        C_solver = np.array([[c[0], c[2], c[1]], [c[2], c[5], c[4]], [c[1], c[4], c[3]]])
        C_phys = PH.energy_C(mesh, free, TR.assemble_K_faff)        # INDEPENDENT oracle

        err = np.abs(C_solver - C_phys).max() / np.abs(C_phys).max()
        worst = max(worst, err)
        tag = f"eta={eta}" + (f" VD{vd:+d}" if vd else " k=1")
        assert err < 1e-2, (f"C_eff disagrees with the energy Hessian ({tag}, N={N}): {err:.3e}\n"
                            f"  solver:\n{C_solver}\n  physical:\n{C_phys}")
    print(f"  [7] C_eff vs energy Hessian (component-wise, {len(cases)} disordered/VD meshes): "
          f"worst {worst:.2e}  OK")


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
        test_elastic_tensor_vs_energy_hessian,
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
