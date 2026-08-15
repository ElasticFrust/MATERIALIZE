"""
Finite amplitude: what does the solver's constitutive approximation cost?

The solver is exact in GEOMETRY (Δg = FᵀF − ḡ) but its energy is a quadratic form in Δg:

    U_sol = ½ Σ_e (k_e/4ℓ_e²)(q_e·Δg)²

Since q_e·Δg = ℓ′² − ℓ² **exactly**, this equals

    U_sol = ½ Σ_e k_e (δℓ_e)² · ((ℓ′_e+ℓ_e)/2ℓ_e)²   ≠   ½ Σ_e k_e (δℓ_e)²

so it differs from a real Hookean spring by the factor ((ℓ′+ℓ)/2ℓ)² ≈ (1 + δℓ/ℓ) — **first order
in the strain**, not second. That is a CONSTITUTIVE difference, and it is invisible to every check
in the repo so far, because all of them are tangent (infinitesimal) quantities where it vanishes.

This script measures it at finite amplitude λ, and separates it from a second, independent
approximation — that the solver's response W is linear in Δg. Three relaxed energies of the SAME
network under the SAME macro deformation F(λ), all with EXACT kinematics (true node positions):

  U1(λ)  linear Hookean springs, the physical reference:   ½ Σ k (ℓ′ − ℓ₀)²
  U2(λ)  the solver's ENERGY FUNCTIONAL, relaxed over the same nodal DOF:
         ½ Σ (k/4ℓ²)(ℓ′² − ℓ²)²
  U3(λ)  the solver's actual PREDICTION: ½ Δg(λ) : C_eff : Δg(λ) · A_tot, with C_eff from
         `forward()` and Δg(λ) the exact macro metric change — i.e. its linear response
         extrapolated to amplitude λ

Then:
    U2 − U1  is the CONSTITUTIVE error alone (same relaxation, same kinematics, different energy)
    U3 − U2  is the RESPONSE-LINEARISATION error alone (same energy, linear vs relaxed response)
    U3 − U1  is the total, which is what a user of the solver actually incurs at amplitude λ

All three → 0 as λ → 0 (they share a tangent modulus); the point is how fast they grow, and which
term dominates. Uniaxial and shear are both probed since the shear channel behaves differently.

Terminology (CLAUDE.md §3): every model here has a LINEAR Hookean constitutive law except U2, whose
energy is quadratic in Δg. "Nonlinear" always means geometric/kinematic.

Run:  python finite_amplitude_check.py [--N 6] [--eta 0.25] [--vd 3]
Out:  verification_tools/plots/finite_amplitude/
"""
import argparse
import os
import sys

import numpy as np
import torch
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, 'Phase 2'))
sys.path.insert(0, ROOT)

import mesh_build as MB
import solver_build as SB
import plotting as P
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)
OUTDIR = os.path.join(HERE, 'plots', 'finite_amplitude')


def _state(mesh):
    R = mesh['bond_R']
    return mesh['bond_u'], mesh['bond_v'], R, mesh['bond_k'], np.sqrt((R ** 2).sum(1))


def _lengths(uflat, F, mesh, free):
    """True deformed bond vectors under macro F plus nodal fluctuation — EXACT kinematics."""
    bu, bv, R, kb, L0 = _state(mesh)
    nn = len(mesh['pts'])
    u = np.zeros(2 * nn); u[free] = uflat; u = u.reshape(nn, 2)
    V = R + R @ (F - np.eye(2)).T + (u[bv] - u[bu])
    return V, np.sqrt((V ** 2).sum(1)), kb, L0, bu, bv, nn


def U_hooke(uflat, F, mesh, free):
    """U1 — real linear springs: ½Σk(ℓ′−ℓ₀)²."""
    _, Ln, kb, L0, *_ = _lengths(uflat, F, mesh, free)
    return 0.5 * np.sum(kb * (Ln - L0) ** 2)


def grad_hooke(uflat, F, mesh, free):
    V, Ln, kb, L0, bu, bv, nn = _lengths(uflat, F, mesh, free)
    fb = (kb * (Ln - L0) / Ln)[:, None] * V
    g = np.zeros((nn, 2)); np.add.at(g, bv, fb); np.add.at(g, bu, -fb)
    return g.ravel()[free]


def U_metric(uflat, F, mesh, free):
    """U2 — the SOLVER'S energy functional: ½Σ(k/4ℓ²)(ℓ′²−ℓ²)², exact kinematics."""
    _, Ln, kb, L0, *_ = _lengths(uflat, F, mesh, free)
    return 0.5 * np.sum(kb * (Ln ** 2 - L0 ** 2) ** 2 / (4.0 * L0 ** 2))


def grad_metric(uflat, F, mesh, free):
    V, Ln, kb, L0, bu, bv, nn = _lengths(uflat, F, mesh, free)
    fb = (kb * (Ln ** 2 - L0 ** 2) / L0 ** 2)[:, None] * V        # d/dV of ½k(|V|²−ℓ²)²/(4ℓ²)
    g = np.zeros((nn, 2)); np.add.at(g, bv, fb); np.add.at(g, bu, -fb)
    return g.ravel()[free]


def relax(fun, jac, F, mesh, free):
    r = minimize(fun, np.zeros(len(free)), args=(F, mesh, free), jac=jac, method='L-BFGS-B',
                 options=dict(maxiter=20000, ftol=1e-18, gtol=1e-14))
    return r.fun


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--N', type=int, default=6)
    ap.add_argument('--eta', type=float, default=0.25)
    ap.add_argument('--vd', type=float, default=3)
    a = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)

    mesh = MB.build_geometry(a.N, a.eta, 1); MB.set_VD(mesh, a.vd)
    free = np.arange(2, 2 * len(mesh['pts']))
    A_tot = mesh['areas'].sum()

    sv = SB.make_solver(mesh, MB.kkt_from_tri_bond(mesh['tri_bond'], mesh['edge_vecs']))
    out = sv.forward(torch.as_tensor(mesh['tri_k']),
                     rest_lengths=torch.as_tensor(np.sqrt(mesh['actual_len2'])),
                     method='intrinsic', physical_units=True)
    c6 = out['elastic_tensor'].detach().numpy()
    Csol = np.array([[c6[0], c6[1], c6[2]], [c6[1], c6[3], c6[4]], [c6[2], c6[4], c6[5]]])
    # Two convention factors, both CALIBRATED on the η=0 crystal (where ν=1/3, E=2/√3 are known)
    # rather than assumed — see ACCURACY docs. (i) vec3 carries shear ONCE but Δg:C:Δg counts the
    # xy slot four times (xy,yx)×(xy,yx) ⇒ SH = diag(1,2,1). (ii) `physical_units=True` returns C in
    # the standard ½ε:C:ε convention, and Δg = 2ε to leading order ⇒ contract with Δg/2, keeping Δg
    # itself geometrically EXACT so only the response is linearised.
    SH = np.diag([1.0, 2.0, 1.0])
    Csol = SH @ Csol @ SH

    modes = {'uniaxial x': np.array([[1.0, 0.0], [0.0, 0.0]]),
             'simple shear': np.array([[0.0, 1.0], [0.0, 0.0]])}
    lams = np.array([1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 2e-1])
    res = {m: {k: [] for k in ('U1', 'U2', 'U3')} for m in modes}

    print(f"N={a.N} eta={a.eta} VD a={a.vd}  ({len(mesh['simplices'])} triangles)")
    for mname, M in modes.items():
        for lam in lams:
            F = np.eye(2) + lam * M
            dg = F.T @ F - np.eye(2)
            v = 0.5 * np.array([dg[0, 0], dg[0, 1], dg[1, 1]])    # vec3 [xx,xy,yy]; Δg/2 = ε + O(λ²)
            res[mname]['U1'].append(relax(U_hooke, grad_hooke, F, mesh, free))
            res[mname]['U2'].append(relax(U_metric, grad_metric, F, mesh, free))
            res[mname]['U3'].append(0.5 * float(v @ Csol @ v) * A_tot)
        print(f"  {mname} done", flush=True)

    fig, axes = plt.subplots(1, 2, figsize=(P.STYLE.PANEL[0] * 2.2, P.STYLE.PANEL[1] * 1.4),
                             squeeze=False)
    for ax, (mname, _) in zip(axes[0], modes.items()):
        U1 = np.array(res[mname]['U1']); U2 = np.array(res[mname]['U2']); U3 = np.array(res[mname]['U3'])
        ax.loglog(lams, np.abs(U2 - U1) / U1, 'o-', color='tab:blue', lw=1.9,
                  label='CONSTITUTIVE:  |U2−U1|/U1')
        ax.loglog(lams, np.abs(U3 - U2) / U1, 's--', color='tab:orange', lw=1.9,
                  label='RESPONSE LINEARISATION:  |U3−U2|/U1')
        ax.loglog(lams, np.abs(U3 - U1) / U1, '^-', color='tab:red', lw=2.2,
                  label='TOTAL as used:  |U3−U1|/U1')
        ax.loglog(lams, lams, ':', color='0.5', lw=1.2, label='slope 1 (O(λ)) reference')
        ax.set_title(mname, fontsize=10); ax.set_xlabel('amplitude λ'); ax.grid(alpha=0.25, which='both')
    axes[0][0].set_ylabel('relative energy error vs real Hookean springs')
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=2, fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Finite amplitude: the solver's constitutive approximation vs real linear springs\n"
                 '(both geometrically exact; all errors vanish as λ→0 because they share a tangent)',
                 fontsize=11)
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    P.save_fig(fig, os.path.join(OUTDIR, 'finite_amplitude.png'))

    print(f"\n{'mode':<14}{'lambda':>9}{'constitutive':>15}{'response-lin':>15}{'total':>12}")
    for mname in modes:
        U1 = np.array(res[mname]['U1']); U2 = np.array(res[mname]['U2']); U3 = np.array(res[mname]['U3'])
        for i, lam in enumerate(lams):
            print(f"{mname:<14}{lam:>9.0e}{abs(U2[i]-U1[i])/U1[i]:>15.3e}"
                  f"{abs(U3[i]-U2[i])/U1[i]:>15.3e}{abs(U3[i]-U1[i])/U1[i]:>12.3e}")
    print(f"\nwrote {OUTDIR}")


if __name__ == '__main__':
    main()
