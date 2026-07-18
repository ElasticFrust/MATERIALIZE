"""
Random (generically incompatible) reference: perturb every bond's rest length by a small random
factor and compare the solver's residual stress to a direct prestressed simulation.
Both get the SAME per-bond perturbation:
  sim    — bond rest lengths ℓ0,b = |edge_b|·f_b directly;
  solver — per-triangle ḡ_s reconstructed from its 3 edge rest lengths (ℓ0² = ḡ_s : q_e).
"""
import os
import sys

import numpy as np
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..')))
import _common as C
from forward_solver_dgbar import forward_dgbar

reg = C.make_lattice(1.0, 1.0, half=6.0); n = len(reg['simplices'])
ev = reg['edge_vecs']                                             # (n,3,2)
elen = np.hypot(ev[:, :, 0], ev[:, :, 1])                         # (n,3) edge lengths
bu, bv, bR = reg['bond_u'], reg['bond_v'], reg['bond_R']
nb = len(bR)

# --- random per-bond rest-length perturbation (fixed seed) ---
eta = 0.05
rng = np.random.default_rng(0)
f = 1.0 + eta * rng.standard_normal(nb)                           # per-bond factor
l0_tri = elen * f[reg['tri_bond']]                               # (n,3) rest length per triangle-edge

# --- solver: reconstruct ḡ_s per triangle from its 3 edge rest lengths (ℓ0² = ḡ:q) ---
q = np.stack([ev[:, :, 0] ** 2, 2 * ev[:, :, 0] * ev[:, :, 1], ev[:, :, 1] ** 2], -1)   # (n,3,3)
gbar_v = np.linalg.solve(q, (l0_tri ** 2)[..., None])[..., 0]     # (n,3) = [gxx,gxy,gyy]
gbar = np.zeros((n, 2, 2))
gbar[:, 0, 0] = gbar_v[:, 0]; gbar[:, 0, 1] = gbar[:, 1, 0] = gbar_v[:, 1]; gbar[:, 1, 1] = gbar_v[:, 2]
th = np.linspace(0.0, np.pi, 181)
out = forward_dgbar(reg, 1.0, gbar, thetas=th)
s = out['stress_field']; p_solver = 0.5 * (s[:, 0] + s[:, 2])

# --- simulation: relax the prestressed spring net (per-bond rest lengths) ---
pts = reg['pts'].copy(); N = len(pts)
offset = bR - (pts[bv] - pts[bu])
l0_bond = np.hypot(bR[:, 0], bR[:, 1]) * f
def eg(x):
    X = x.reshape(N, 2); dd = X[bv] - X[bu] + offset
    Lb = np.hypot(dd[:, 0], dd[:, 1]); dL = Lb - l0_bond
    fr = (dL / Lb)[:, None] * dd; g = np.zeros((N, 2)); np.add.at(g, bv, fr); np.add.at(g, bu, -fr)
    return 0.5 * (dL ** 2).sum(), g.ravel()
res = minimize(eg, pts.ravel(), jac=True, method='L-BFGS-B', options=dict(maxiter=20000, gtol=1e-10))
X = res.x.reshape(N, 2); dd = X[bv] - X[bu] + offset; Lb = np.hypot(dd[:, 0], dd[:, 1]); t = Lb - l0_bond
p_sim = np.zeros(n)
for si in range(n):
    sig = np.zeros((2, 2))
    for ei in reg['tri_bond'][si]:
        v = dd[ei]; sig += t[ei] * np.outer(v, v) / Lb[ei]
    p_sim[si] = 0.5 * (sig[0, 0] + sig[1, 1]) / (2.0 * reg['areas'][si])

scale = np.dot(p_solver, p_sim) / np.dot(p_sim, p_sim)
corr = np.corrcoef(p_solver, p_sim)[0, 1]
print(f"random reference (η={eta}, seed 0):  sim relaxed E={res.fun:.3e}, |grad|={np.abs(res.jac).max():.1e}")
print(f"RESIDUAL STRESS  corr(solver, sim) = {corr:.4f}   best-fit scale = {scale:.4f}")
print(f"  solver rms = {np.sqrt((p_solver**2).mean()):.4f}   sim rms = {np.sqrt((p_sim**2).mean()):.4f}")

# --- rigidity (E) and Poisson ratio: solver C_eff vs sim finite-difference C_eff ---
A0 = np.array([reg['BL1'][0], reg['BL2'][1]]); A0 = A0[0] * A0[1]

def relax_stress(Fdef, x_init):
    off = offset @ Fdef.T
    def eg2(x):
        X = x.reshape(N, 2); ddd = X[bv] - X[bu] + off
        Lb2 = np.hypot(ddd[:, 0], ddd[:, 1]); dL = Lb2 - l0_bond
        fr = (dL / Lb2)[:, None] * ddd; g = np.zeros((N, 2)); np.add.at(g, bv, fr); np.add.at(g, bu, -fr)
        return 0.5 * (dL ** 2).sum(), g.ravel()
    rr = minimize(eg2, (x_init @ Fdef.T).ravel(), jac=True, method='L-BFGS-B', options=dict(maxiter=20000, gtol=1e-11))
    Xr = rr.x.reshape(N, 2); ddd = Xr[bv] - Xr[bu] + off; Lb2 = np.hypot(ddd[:, 0], ddd[:, 1]); tt = Lb2 - l0_bond
    A = A0 * abs(np.linalg.det(Fdef))
    return Xr, (tt[:, None, None] / Lb2[:, None, None] * (ddd[:, :, None] * ddd[:, None, :])).sum(0) / A

X0, sig0 = relax_stress(np.eye(2), X)                            # base (already-relaxed) state
dlt = 1e-4
Ct = np.zeros((3, 3))
for j, e in enumerate([np.array([[1., 0], [0, 0]]), np.array([[0, 0], [0, 1.]]), np.array([[0, 1.], [1., 0]])]):
    _, sp_ = relax_stress(np.eye(2) + dlt * e, X0); _, sm_ = relax_stress(np.eye(2) - dlt * e, X0)
    Ct[:, j] = (np.array([sp_[0, 0], sp_[1, 1], sp_[0, 1]]) - np.array([sm_[0, 0], sm_[1, 1], sm_[0, 1]])) / (2 * dlt)
Ct = 0.5 * (Ct + Ct.T)
C6_sim = np.array([Ct[0, 0], Ct[0, 2], Ct[0, 1], Ct[2, 2], Ct[1, 2], Ct[1, 1]])
nu_sim_th, E_sim_th = C.nu_E_theta(C6_sim, th)
nu_solver = out['nu_theta'].mean(); E_solver = out['E_theta'].mean()
nu_sim = nu_sim_th.mean(); E_sim = E_sim_th.mean()
print(f"RIGIDITY / POISSON (angle-averaged; regular baseline ν=0.3333, E=1.1547):")
print(f"  solver:  ν={nu_solver:+.4f}  E={E_solver:.5f}")
print(f"  sim   :  ν={nu_sim:+.4f}  E={E_sim:.5f}")
print(f"  Δν={abs(nu_solver-nu_sim):.4f}   ΔE/E={abs(E_solver-E_sim)/E_sim:.4f}")

L = np.array([reg['BL1'][0], reg['BL2'][1]]); frac = (reg['centroids'] / L) % 1.0
fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
vlim = np.abs(p_solver).max()
for a, fld, ttl in ((ax[0], p_solver, 'solver'), (ax[1], p_sim * scale, f'simulation ×{scale:.3f}')):
    sc = a.scatter(frac[:, 0], frac[:, 1], c=fld, s=24, cmap='RdBu_r', vmin=-vlim, vmax=vlim)
    a.set_title(ttl); a.set_aspect('equal'); a.set_xlim(0, 1); a.set_ylim(0, 1); plt.colorbar(sc, ax=a, shrink=.8)
ax[2].scatter(p_sim * scale, p_solver, s=12, alpha=.5)
lim = [min(p_solver.min(), (p_sim * scale).min()), max(p_solver.max(), (p_sim * scale).max())]
ax[2].plot(lim, lim, 'k--', lw=1); ax[2].set_xlabel('sim (scaled)'); ax[2].set_ylabel('solver')
ax[2].set_title(f'corr={corr:.4f}'); ax[2].set_aspect('equal')
fig.suptitle(f'Random reference-length perturbation: residual pressure, solver vs simulation  (η={eta})')
plt.tight_layout(rect=[0, 0, 1, 0.95])
p = os.path.join(HERE, 'random_ref_compare.png')
plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
print(f"saved {os.path.basename(p)}")
