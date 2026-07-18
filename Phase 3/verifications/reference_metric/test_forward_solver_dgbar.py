"""
Verification for forward_solver_dgbar.DGBarSolver — hard assertions.

  1. reduction     ḡ=I must equal the ordinary Phase 2 core solve (ν=1/3, E=E0, W=0, σ⁰=0).
  2. covariant     two descriptions of the same material collapse: elongated net (ḡ=I) vs regular
                   net (full ḡ=FᵀF) give the same covariant ν(θ), E(θ).
  3. sphere        incompatible spherical ḡ → the pinned residual-stress values (regression guard).

Run:  python test_forward_solver_dgbar.py   → prints PASS/FAIL per check and exits nonzero on failure.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..')))
import _common as C
from forward_solver_dgbar import DGBarSolver, forward_dgbar


def affine(geo, M):
    g = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in geo.items()}
    for key in ('pts', 'edge_vecs', 'bond_R', 'centroids', 'tri_verts'):
        if key in g:
            g[key] = geo[key] @ M.T
    g['actual_len2'] = (g['edge_vecs'] ** 2).sum(-1)
    g['areas'] = geo['areas'] * abs(np.linalg.det(M))
    g['BL1'] = M @ geo['BL1']; g['BL2'] = M @ geo['BL2']
    return g


def _check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}   {detail}")
    return ok


def test_reduction():
    """ḡ=I reproduces the ordinary core solve exactly (no reference offset)."""
    reg = C.make_lattice(1.0, 1.0, half=6.0); n = len(reg['simplices'])
    out = forward_dgbar(reg, 1.0, np.eye(2))
    E0 = 2.0 / np.sqrt(3.0)                                   # 2D triangular-lattice modulus, k=1
    ok = True
    ok &= _check("nu = 1/3", abs(out['poisson'] - 1.0 / 3.0) < 1e-6, f"nu={out['poisson']:.6f}")
    ok &= _check("E = E0", abs(out['young'] - E0) < 1e-4, f"E={out['young']:.6f} (E0={E0:.6f})")
    ok &= _check("W = 0", np.abs(out['W']).max() < 1e-9, f"|W|={np.abs(out['W']).max():.1e}")
    ok &= _check("sigma0 = 0", np.abs(out['sigma0']).max() < 1e-9, f"|s0|={np.abs(out['sigma0']).max():.1e}")
    # covariant readout must coincide with the flat one at ḡ=I
    ok &= _check("covariant==flat at ḡ=I", abs(out['poisson'] - out['poisson_flat']) < 1e-9
                 and abs(out['young'] - out['young_flat']) < 1e-9)
    return ok


def test_covariant_collapse():
    """Same material, two descriptions → covariant ν(θ), E(θ) coincide."""
    psi = 1.3; F = np.diag([1.0, psi])
    reg = C.make_lattice(1.0, 1.0, half=6.0); n = len(reg['simplices'])
    elong = affine(reg, F); ne = len(elong['simplices'])
    th = np.linspace(0.0, np.pi, 181)
    he = forward_dgbar(elong, 1.0, np.eye(2), thetas=th)                 # ḡ=I, anisotropy in network
    rf = forward_dgbar(reg, 1.0, F.T @ F, thetas=th)                     # regular, anisotropy in ḡ
    dnu = np.max(np.abs(he['nu_theta'] - rf['nu_theta']))
    dE = np.max(np.abs(he['E_theta'] - rf['E_theta']))
    ok = True
    ok &= _check("covariant nu(theta) collapse", dnu < 1e-10, f"max|dnu|={dnu:.1e}")
    ok &= _check("covariant E(theta) collapse", dE < 1e-10, f"max|dE|={dE:.1e}")
    return ok


def test_sphere_residual():
    """Incompatible spherical ḡ → pinned residual stress (regression)."""
    reg = C.make_lattice(1.0, 1.0, half=6.0); n = len(reg['simplices'])
    L = np.array([reg['BL1'][0], reg['BL2'][1]]); c0 = L / 2.0
    d = reg['centroids'] - c0; d -= L * np.round(d / L); r = np.hypot(d[:, 0], d[:, 1])
    lam = 1.0 / (1.0 + (1.0 / 64.0) * r ** 2 / 4.0) ** 2
    gbar = np.zeros((n, 2, 2)); gbar[:, 0, 0] = lam; gbar[:, 1, 1] = lam
    out = forward_dgbar(reg, 1.0, gbar)
    s0 = out['sigma0']; p = 0.5 * (out['stress_field'][:, 0] + out['stress_field'][:, 2])
    rms = np.sqrt((p ** 2).mean())
    ok = True
    ok &= _check("sigma0 isotropic +0.0723", abs(s0[0] - 0.0723) < 2e-3 and abs(s0[2] - 0.0723) < 2e-3
                 and abs(s0[1]) < 1e-6, f"s0=[{s0[0]:.4f},{s0[1]:.4f},{s0[2]:.4f}]")
    ok &= _check("residual pressure rms 0.0742", abs(rms - 0.0742) < 2e-3, f"rms={rms:.4f}")
    return ok


if __name__ == '__main__':
    print("test_forward_solver_dgbar:")
    results = [test_reduction(), test_covariant_collapse(), test_sphere_residual()]
    print(f"\n{sum(results)}/{len(results)} checks passed.")
    sys.exit(0 if all(results) else 1)
