"""
Does linearising the kinematics cost anything? — the exact-kinematics reference, and the answer.

`physical_homog` (the oracle) uses LINEAR springs with the extension **linearised** in displacement:
`assemble_K_faff` builds S = k·R̂⊗R̂ and `energy_C` measures the extension as (R·δu)/ℓ. That is the
tangent problem at the reference configuration, not a finite-deformation simulation. The obvious
worry is that the oracle is therefore validating the solver against an approximation.

This script implements the missing reference — **model (1): linear (Hookean) springs with EXACT
kinematics**, U = ½Σ k(‖xᵢ−xⱼ‖ − ℓ₀)², relaxed by L-BFGS on the true bond lengths with the analytic
gradient — and measures the difference. Terminology, per `CLAUDE.md` §3: every model here has a
LINEAR constitutive law; "nonlinear" always means geometric/kinematic.

**Result (measured, N=6 η=0.25 VD a=+3): the difference is O(h) in the probe amplitude and vanishes
as h → 0** — 6.62e-3, 2.00e-3, 6.66e-4, 2.00e-4 at h = 1e-2, 3e-3, 1e-3, 3e-4. So the two models
have the SAME tangent modulus, and the linearisation costs **nothing** for C_eff.

Why that is so, and not a coincidence: at an UNSTRESSED reference (ℓ₀ = ℓ, so bond tension T = 0)
the terms in ∂²U/∂ε² carrying second derivatives of ℓ are multiplied by δℓ = 0 and drop out, leaving
exactly the linearised expression. The tangent stiffness of a spring is
k·R̂⊗R̂ + (T/ℓ)(I − R̂⊗R̂); the second, GEOMETRIC term vanishes at T = 0.

**Where it does bite** — the same argument says where the oracle stops being exact:
  - **prestress / residual stress (T ≠ 0).** Then the geometric term survives, and
    `sim_assembly.assemble_K_faff` cannot represent it: it takes no rest-length argument at all
    (ℓ₀ is implicitly |R|), so a prestressed tangent is outside what it can express. This is the
    incompatible-ḡ direction (FUTURE_DIRECTIONS #1), and it is why `CLAUDE.md` §3 already marks the
    oracle "flat-compatible only".
  - **finite strain**, beyond the tangent — but the framework only claims the tangent about the
    reference, so this is a limit of scope, not an error.
  - **near a mechanism**, where the tangent is nearly singular and the response at any usable strain
    is dominated by what the tangent omits.

Run:  python exact_kinematics_check.py [--N 6] [--eta 0.25] [--vd 3]
"""
import argparse
import os
import sys

import numpy as np
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, 'Phase 2'))

import physical_homog as PH
import sim_assembly as SA
import mesh_build as MB


def _bond_state(mesh):
    R = mesh['bond_R']
    return (mesh['bond_u'], mesh['bond_v'], R, mesh['bond_k'], np.sqrt((R ** 2).sum(1)))


def energy_exact(uflat, F, mesh, free):
    """U = ½Σ k(‖xᵢ−xⱼ‖ − ℓ₀)² — LINEAR springs, EXACT kinematics (true lengths)."""
    bu, bv, R, kb, L0 = _bond_state(mesh)
    nn = len(mesh['pts'])
    u = np.zeros(2 * nn); u[free] = uflat; u = u.reshape(nn, 2)
    V = R + R @ (F - np.eye(2)).T + (u[bv] - u[bu])
    return 0.5 * np.sum(kb * (np.sqrt((V ** 2).sum(1)) - L0) ** 2)


def grad_exact(uflat, F, mesh, free):
    """Analytic gradient of `energy_exact` (nodal forces from bond tensions)."""
    bu, bv, R, kb, L0 = _bond_state(mesh)
    nn = len(mesh['pts'])
    u = np.zeros(2 * nn); u[free] = uflat; u = u.reshape(nn, 2)
    V = R + R @ (F - np.eye(2)).T + (u[bv] - u[bu])
    Ln = np.sqrt((V ** 2).sum(1))
    fb = (kb * (Ln - L0) / Ln)[:, None] * V
    g = np.zeros((nn, 2)); np.add.at(g, bv, fb); np.add.at(g, bu, -fb)
    return g.ravel()[free]


def energy_C_exact(mesh, free, h=1e-3):
    """C_eff from the EXACT-kinematics model — same Voigt [xx,yy,xy] convention as `energy_C`.

    Each probe strain is relaxed by L-BFGS on the true spring energy (not a linear solve), so this
    is model (1). Tight tolerances: the second differences amplify any residual force."""
    A = mesh['areas'].sum()

    def e(g):
        F = np.eye(2) + np.array([[g[0], g[2] / 2], [g[2] / 2, g[1]]])
        r = minimize(energy_exact, np.zeros(len(free)), args=(F, mesh, free), jac=grad_exact,
                     method='L-BFGS-B', options=dict(maxiter=5000, ftol=1e-18, gtol=1e-14))
        return r.fun / A

    C = np.zeros((3, 3)); diag = []
    for a in range(3):
        ga = np.zeros(3); ga[a] = h
        d = 2 * e(ga) / h ** 2; C[a, a] = d; diag.append(d)
    for a in range(3):
        for b in range(a + 1, 3):
            g = np.zeros(3); g[a] = h; g[b] = h
            C[a, b] = C[b, a] = (2 * e(g) / h ** 2 - diag[a] - diag[b]) / 2
    return C


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--N', type=int, default=6)
    ap.add_argument('--eta', type=float, default=0.25)
    ap.add_argument('--vd', type=float, default=3)
    a = ap.parse_args()

    mesh = MB.build_geometry(a.N, a.eta, 1); MB.set_VD(mesh, a.vd)
    free = np.arange(2, 2 * len(mesh['pts']))
    C_lin = PH.energy_C(mesh, free, SA.assemble_K_faff)          # model 3 (linearised kinematics)

    print(f"N={a.N}, eta={a.eta}, VD a={a.vd}  ({len(mesh['simplices'])} triangles)")
    print("model 1 (EXACT kinematics, L-BFGS on true lengths) vs model 3 (linearised, one solve)")
    print(f"   {'probe h':>9} {'rel difference':>16} {'ratio':>8}")
    prev = None
    for h in [1e-2, 3e-3, 1e-3, 3e-4]:
        d = np.abs(energy_C_exact(mesh, free, h) - C_lin).max() / np.abs(C_lin).max()
        print(f"   {h:>9.0e} {d:>16.3e} {('%.2f' % (prev / d)) if prev else '-':>8}")
        prev = d
    print("\n   -> difference is O(h) and vanishes as h->0: the two models share a tangent modulus,")
    print("      so linearising the kinematics costs NOTHING for C_eff at an unstressed reference.")


if __name__ == '__main__':
    main()
