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


# Tolerance for the tensor gates [7] and [8], under the PER-COMPONENT metric `_rel_err`.
# NOT a loosened version of the old 1e-2/2e-2: those were calibrated for the weaker max-normalised
# measure, and a stricter metric needs its own threshold. Placed from the measured separation
# between a clean run and the A-0 shear defect over the five gate meshes (2026-08-15):
#
#            clean      with A-0     separation
#   bulk     1.65e-02   5.82e+00     353x
#   per-tri  9.74e-03   2.67e+00     275x
#   regional 1.67e-02   7.61e+00     455x
#
# 5e-2 sits ~3x above the clean worst and 50-150x below the defect signal. The clean numbers are
# dominated by components with SMALL |C| (absolute errors stay <=8.4e-4 against a tensor scale of
# 0.1-1.3) — which is the point of normalising each component by its own magnitude.
TOL_C = 5e-2


def _rel_err(A, B, floor=0.05):
    """PER-COMPONENT relative error between tensors A and B, max over components.

    `A`, `B` are (...,3,3) Voigt [xx,yy,xy] (a bulk tensor, or a stack over triangles). Each
    component is normalised by ITS OWN scale, not by the largest component of the tensor:

        err = max_c  max|A_c − B_c| / max( max|B_c| , floor·max|B| )

    Why not the simpler max|A−B|/max|B|: that normalises everything by the LARGEST component, so a
    small component could be badly wrong and barely register — the same shape of blindness as the
    ν=1/3 gate. Measured 2026-08-15, the difference is up to ~1.3× on real meshes (e.g. per-triangle
    η=0.30 VD+10: 7.74e-3 by the old measure, 9.74e-3 per-component), so it is a real if bounded
    understatement.

    Why not plain element-wise (A−B)/B: tensor components genuinely pass through zero (C_xxxy and
    C_yyxy change sign, and per-triangle values span both signs), where that ratio is undefined and
    explodes. The `floor` — 5% of the tensor's largest component, matching the eps_nu convention in
    CLAUDE.md §3 — bounds the denominator for components that are small everywhere, while leaving
    significant components at their true relative error."""
    A = np.asarray(A); B = np.asarray(B)
    ax = tuple(range(A.ndim - 2))                       # reduce over triangles, keep (3,3)
    num = np.abs(A - B).max(axis=ax) if ax else np.abs(A - B)
    den = np.abs(B).max(axis=ax) if ax else np.abs(B)
    return float((num / np.maximum(den, floor * np.abs(B).max())).max())


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
    import metric_ops as MO                    # core-adjacent: vec3, tri_metric_change, bare_tensor
    import mesh_build as MB                    # core-adjacent: build_geometry, set_VD
    import physical_homog as PH                # ORACLE side of this comparison
    import sim_assembly as SA                  # ORACLE side: the sim's assembler

    Dgt = [F.T @ F - np.eye(2) for F in PH.Fk]                      # the 3 macro metric-change modes
    Dinv = np.linalg.inv(np.stack([MO.vec3(g) for g in Dgt], 1))    # vec3 mode matrix, inverted once

    # (N, eta, seed, VD contrast or None for uniform k); eta=0 uniform is the BLIND control (W=0)
    cases = [(8, 0.00, 0, None), (8, 0.20, 1, None), (8, 0.20, 1, 5),
             (8, 0.35, 2, None), (12, 0.30, 3, 10)]
    worst = worst_r = 0.0
    for N, eta, seed, vd in cases:
        mesh = MB.build_geometry(N, eta, seed)
        if vd is None:
            mesh['bond_k'] = np.ones(len(mesh['bond_R']))
            mesh['tri_k'] = mesh['bond_k'][mesh['tri_bond']]
        else:
            MB.set_VD(mesh, vd)                                     # k = 1 + tanh(a·(|R|−1))
        nn, nt = len(mesh['pts']), len(mesh['simplices'])
        free = np.arange(2, 2 * nn)                                 # pin node 0

        # W measured from the SIM's relaxation: delta_g(s) = W(s) Delta_g  ->  W3 = D · Dinv
        u_modes = PH.relax(mesh, free, SA.assemble_K_faff)
        D = np.zeros((nt, 3, 3))
        for j, (F, u) in enumerate(zip(PH.Fk, u_modes)):
            D[:, :, j] = MO.vec3(
                MO.tri_metric_change(mesh['edge_vecs'], mesh['simplices'], F, u) - Dgt[j])
        W3 = D @ Dinv

        c6 = fst._compute_actual_elastic_tensor(
            torch.as_tensor(MO.bare_tensor(mesh)),
            torch.as_tensor(W3.reshape(-1, 9))).numpy()
        c = c6.mean(0) * (8.0 * nt / mesh['areas'].sum())           # unweighted mean → physical units
        C_solver = np.array([[c[0], c[2], c[1]], [c[2], c[5], c[4]], [c[1], c[4], c[3]]])
        C_phys = PH.energy_C(mesh, free, SA.assemble_K_faff)        # INDEPENDENT oracle

        err = _rel_err(C_solver, C_phys)                            # PER-COMPONENT (see _rel_err)
        worst = max(worst, err)
        tag = f"eta={eta}" + (f" VD{vd:+d}" if vd else " k=1")
        assert err < TOL_C, (f"C_eff disagrees with the energy Hessian ({tag}, N={N}): {err:.3e}\n"
                            f"  solver:\n{C_solver}\n  physical:\n{C_phys}")
    print(f"  [7] C_eff vs energy Hessian (component-wise, {len(cases)} disordered/VD meshes): "
          f"worst {worst:.2e}  OK")


# per-triangle C(s) [xx,yy,xy] (acting on g, F = I + [[g0,g2/2],[g2/2,g1]])
#   = 16 × the solver's per-triangle tensor (acting on Δg = FᵀF − I, vec3 [xx,xy,yy]).
# The two differ by the change of variables Δg ≈ (2g0, g2, 2g1) plus the Voigt reordering, which is
# a congruence and NOT obviously a scalar — so this constant is not asserted from algebra. It is
# CALIBRATED on crystals where W ≡ 0 and therefore C(s) = A(s) exactly, and test [8] re-derives it
# from `metric_ops.bare_tensor` on every run rather than trusting the literal.
_G_TO_DG = 16.0


def test_per_triangle_C_vs_energy_hessian():
    """[8] PER-TRIANGLE C(s), component-wise, against an INDEPENDENT per-triangle energy Hessian.

    The one-level-down analogue of [7], and the close of audit A-9. Until this existed, every LOCAL
    quantity (`_common.sim_per_triangle_C6`, `region_phys_C6`) was obtained by taking the sim's
    relaxation and pushing it back through the solver's own `_compute_actual_elastic_tensor` — the
    very contraction under test. `physical_homog.energy_C_per_triangle` instead differentiates each
    triangle's own relaxed spring energy twice w.r.t. the macro strain: it never forms W, never does
    the 4-index contraction, and touches no solver code.

    Structure mirrors [7]: `_compute_actual_elastic_tensor` is fed the SIMULATION's measured W, so
    the check isolates the contraction — same A(s), same W, two independent routes to C(s).

    Three parts:
      (a) CONVENTION — on crystals (W≡0 ⇒ C(s)=A(s)) the oracle must reproduce `bare_tensor` up to
          the single constant `_G_TO_DG`, re-derived here. Includes SHEARED crystals, without which
          the shear-coupling entries are 0/0 and the constant is not pinned there.
      (b) INTERNAL CONSISTENCY — `energy_C_region(None)` must reproduce the already-trusted bulk
          `energy_C` exactly (the ½ undoing the double-counting of shared bonds).
      (c) THE GATE — per-triangle agreement on DISORDERED / VD meshes, where W ≠ 0.

    Tolerance: the residual is O(DELTA), from the strain measure — `tri_metric_change` is
    geometrically exact while the energy uses the linearised bond extension. Verified to scale
    LINEARLY in DELTA (7.74e-3 → 2.32e-3 → 7.73e-4 → 2.32e-4 for DELTA 1e-3 → 3e-4 → 1e-4 → 3e-5).
    Worst at the default DELTA=1e-3 is 9.7e-3 under the per-component metric; see TOL_C above for
    how the threshold is placed against the A-0 defect signal.
    """
    import forward_solver_torch as fst
    import physical_homog as PH
    import sim_assembly as SA
    import metric_ops as MO
    import mesh_build as MB

    # ---- (a) convention, on crystals where W ≡ 0 so C(s) = A(s) exactly --------------------
    for tag, aff in [('regular', None),
                     ('sheared', np.array([[1.0, 0.45], [0.0, 1.0]])),
                     ('stretch+shear', np.array([[1.3, 0.35], [0.15, 0.8]]))]:
        geo = MB.build_geometry(8, 0.0, 0); MB.set_VD(geo, 0)
        if aff is not None:                       # affine keeps every triangle congruent ⇒ W still 0
            geo['pts'] = geo['pts'] @ aff.T
            geo['edge_vecs'] = geo['edge_vecs'] @ aff.T
            geo['bond_R'] = geo['bond_R'] @ aff.T
            geo['actual_len2'] = (geo['edge_vecs'] ** 2).sum(2)
            e01, e02 = geo['edge_vecs'][:, 0], geo['edge_vecs'][:, 1]
            geo['areas'] = 0.5 * np.abs(e01[:, 0] * e02[:, 1] - e01[:, 1] * e02[:, 0])
            MB.set_VD(geo, 0)
        a = MO.bare_tensor(geo)
        assert np.allclose(a, a[0], rtol=1e-12), f"{tag}: A(s) not uniform, so W≠0 — bad calibration case"
        A_mat = np.array([[a[0, 0], a[0, 2], a[0, 1]],
                          [a[0, 2], a[0, 4], a[0, 3]],
                          [a[0, 1], a[0, 3], a[0, 2]]])                     # A(s) in [xx,yy,xy]
        Cs = PH.energy_C_per_triangle(geo, np.arange(2, 2 * len(geo['pts'])), SA.assemble_K_faff)
        # compare against _G_TO_DG·A rather than forming a ratio: the REGULAR crystal has exact
        # zeros in the shear-coupling entries, where a ratio is 0/0.
        err_c = np.abs(Cs[0] - _G_TO_DG * A_mat).max() / np.abs(_G_TO_DG * A_mat).max()
        assert err_c < 1e-8, (f"{tag}: per-triangle oracle != {_G_TO_DG}·A(s) where W≡0 "
                              f"(rel {err_c:.3e}) — CONVENTION WRONG, do not tune the constant")
        assert np.allclose(Cs, Cs[0], rtol=1e-9), f"{tag}: C(s) not uniform on a crystal"

    # ---- (b) internal consistency: the region form must reproduce the trusted bulk oracle ----
    geo = MB.build_geometry(8, 0.20, 1); MB.set_VD(geo, 5)
    free = np.arange(2, 2 * len(geo['pts']))
    C_bulk = PH.energy_C(geo, free, SA.assemble_K_faff)
    C_s = PH.energy_C_per_triangle(geo, free, SA.assemble_K_faff)
    err_b = np.abs(PH.energy_C_region(geo, free, SA.assemble_K_faff, None, C_s=C_s)
                   - C_bulk).max() / np.abs(C_bulk).max()
    assert err_b < 1e-12, f"energy_C_region(None) != energy_C: {err_b:.3e}"

    # ---- (c) the gate: per-triangle, component-wise, on W ≠ 0 meshes -------------------------
    Dgt = [F.T @ F - np.eye(2) for F in PH.Fk]
    Dinv = np.linalg.inv(np.stack([MO.vec3(g) for g in Dgt], 1))
    cases = [(8, 0.00, 0, None), (8, 0.20, 1, None), (8, 0.20, 1, 5),
             (8, 0.35, 2, None), (12, 0.30, 3, 10)]
    worst = worst_r = 0.0
    for N, eta, seed, vd in cases:
        mesh = MB.build_geometry(N, eta, seed)
        if vd is None:
            mesh['bond_k'] = np.ones(len(mesh['bond_R']))
            mesh['tri_k'] = mesh['bond_k'][mesh['tri_bond']]
        else:
            MB.set_VD(mesh, vd)
        nn, nt = len(mesh['pts']), len(mesh['simplices'])
        free = np.arange(2, 2 * nn)

        u_modes = PH.relax(mesh, free, SA.assemble_K_faff)          # the SIM's relaxation
        D = np.zeros((nt, 3, 3))
        for j, (F, u) in enumerate(zip(PH.Fk, u_modes)):
            D[:, :, j] = MO.vec3(
                MO.tri_metric_change(mesh['edge_vecs'], mesh['simplices'], F, u) - Dgt[j])
        c6 = fst._compute_actual_elastic_tensor(
            torch.as_tensor(MO.bare_tensor(mesh)),
            torch.as_tensor((D @ Dinv).reshape(-1, 9))).numpy()
        C_solver = np.stack([np.stack([c6[:, 0], c6[:, 2], c6[:, 1]], -1),
                             np.stack([c6[:, 2], c6[:, 5], c6[:, 4]], -1),
                             np.stack([c6[:, 1], c6[:, 4], c6[:, 3]], -1)], -2)   # (N,3,3) [xx,yy,xy]
        C_ind = PH.energy_C_per_triangle(mesh, free, SA.assemble_K_faff) / _G_TO_DG  # INDEPENDENT

        err = _rel_err(C_solver, C_ind)                             # PER-COMPONENT (see _rel_err)
        worst = max(worst, err)
        tag = f"eta={eta}" + (f" VD{vd:+d}" if vd else " k=1")
        assert err < TOL_C, (f"per-triangle C(s) disagrees with the energy Hessian ({tag}, N={N}): "
                            f"{err:.3e}")

        # ---- (d) genuine SUB-REGIONS -------------------------------------------------------
        # region=None in (b) only exercises the whole-cell branch; the `idx` branch needs its own
        # check. NB **no _G_TO_DG here**, unlike the per-triangle comparison above: the ×16 between
        # the two conventions is exactly cancelled by their normalisations. `energy_C_region` is
        # ½·ΣC_s/A_r = 8·ΣM_s/A_r, and `region_phys_C6` is mean(c6)·8n/A_r = 8·ΣM_s/A_r — the same
        # quantity. (Dividing by 16 here fails by a factor 16, which is how this was caught.)
        C_s_ind = PH.energy_C_per_triangle(mesh, free, SA.assemble_K_faff)
        cen = mesh['pts'][mesh['simplices']].mean(1)
        halves = [np.where(cen[:, 0] < np.median(cen[:, 0]))[0],          # left half
                  np.where(cen[:, 1] > np.median(cen[:, 1]))[0],          # top half
                  np.arange(0, nt, 3)]                                    # a scattered third
        for r, region in enumerate(halves):
            idx = np.asarray(region)
            c6r = c6[idx].mean(0) * (8.0 * len(idx) / mesh['areas'][idx].sum())
            R_solver = np.array([[c6r[0], c6r[2], c6r[1]],
                                 [c6r[2], c6r[5], c6r[4]],
                                 [c6r[1], c6r[4], c6r[3]]])
            R_ind = PH.energy_C_region(mesh, free, SA.assemble_K_faff,
                                       region=idx, C_s=C_s_ind)
            e_r = _rel_err(R_solver, R_ind)
            worst_r = max(worst_r, e_r)
            assert e_r < TOL_C, (f"regional C disagrees with the energy Hessian "
                                f"({tag}, N={N}, region {r}, {len(idx)} tri): {e_r:.3e}")
    print(f"  [8] per-triangle C(s) vs energy Hessian (component-wise, {len(cases)} meshes): "
          f"worst {worst:.2e}; regional (3 sub-regions each): worst {worst_r:.2e}  OK")


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
        test_per_triangle_C_vs_energy_hessian,
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
