"""
Validate the intrinsic residual stress (forward_dgbar) against a DIRECT simulation:
relax a real prestressed spring network whose rest lengths come from the SAME spherical reference
ḡ=λ(r)I, then read the per-triangle residual pressure and compare fields.
"""
import os, sys
import numpy as np
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..')))
import _common as C
from forward_solver_dgbar import forward_dgbar

reg = C.make_lattice(1.0, 1.0, half=6.0); n = len(reg['simplices'])
pts = reg['pts'].copy(); N = len(pts)
bu, bv, bR = reg['bond_u'], reg['bond_v'], reg['bond_R']
offset = bR - (pts[bv] - pts[bu])                             # fixed periodic offset per bond
L = np.array([reg['BL1'][0], reg['BL2'][1]]); c0 = L / 2.0
R = 8.0; K = 1.0 / R ** 2


def lam_at(xy):
    d = xy - c0; d -= L * np.round(d / L)
    r2 = (d ** 2).sum(-1)
    return 1.0 / (1.0 + K * r2 / 4.0) ** 2


# --- spherical reference: per-triangle ḡ (for the intrinsic solve) and per-bond rest length (sim) ---
gbar = np.zeros((n, 2, 2))
lam_tri = lam_at(reg['centroids']); gbar[:, 0, 0] = lam_tri; gbar[:, 1, 1] = lam_tri
mid = pts[bu] + bR / 2.0
l0 = np.sqrt(lam_at(mid)) * np.hypot(bR[:, 0], bR[:, 1])      # ℓ0 = √λ · |bond|  (ḡ=λI)

# --- intrinsic (relaxed residual stress) ---
out = forward_dgbar(reg, 1.0, gbar)
s = out['stress_field']; p_intr = 0.5 * (s[:, 0] + s[:, 2])

# --- direct simulation: minimize ½Σ k(|d_b|-ℓ0)² over node positions, box (offsets) fixed ---
def energy_grad(x):
    X = x.reshape(N, 2)
    d = X[bv] - X[bu] + offset
    Lb = np.hypot(d[:, 0], d[:, 1]); dL = Lb - l0
    E = 0.5 * (dL ** 2).sum()
    f = (dL / Lb)[:, None] * d                                # per-bond force direction * magnitude
    g = np.zeros((N, 2))
    np.add.at(g, bv, f); np.add.at(g, bu, -f)
    return E, g.ravel()

res = minimize(energy_grad, pts.ravel(), jac=True, method='L-BFGS-B',
               options=dict(maxiter=20000, ftol=1e-14, gtol=1e-10))
X = res.x.reshape(N, 2)
d = X[bv] - X[bu] + offset; Lb = np.hypot(d[:, 0], d[:, 1]); t = Lb - l0   # bond tension (k=1)
print(f"sim relaxed: E={res.fun:.4e}  |grad|={np.abs(res.jac).max():.2e}  iters={res.nit}")

# per-triangle residual pressure from bond tensions:  σ_s = (1/2a)Σ_e t_e (v⊗v)/|v|
p_sim = np.zeros(n)
for si in range(n):
    e = reg['tri_bond'][si]
    sig = np.zeros((2, 2))
    for ei in e:
        v = d[ei]; sig += t[ei] * np.outer(v, v) / Lb[ei]
    sig /= (2.0 * reg['areas'][si])
    p_sim[si] = 0.5 * (sig[0, 0] + sig[1, 1])

# --- compare intrinsic residual stress vs simulation ---
scale = np.dot(p_intr, p_sim) / np.dot(p_sim, p_sim)         # best-fit scale (metric vs engineering)
corr = np.corrcoef(p_intr, p_sim)[0, 1]
print(f"residual pressure  corr(intrinsic, sim) = {corr:.4f}   best-fit scale = {scale:.4f}")
print(f"  intrinsic rms = {np.sqrt((p_intr**2).mean()):.4f}   sim rms = {np.sqrt((p_sim**2).mean()):.4f}")

frac = (reg['centroids'] / L) % 1.0
fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
for a, f, ttl in ((ax[0], p_intr, 'intrinsic  (forward_dgbar)'), (ax[1], p_sim * scale, 'simulation  (× best-fit scale)')):
    sc = a.scatter(frac[:, 0], frac[:, 1], c=f, s=24, cmap='RdBu_r',
                   vmin=-np.abs(p_intr).max(), vmax=np.abs(p_intr).max())
    a.set_title(ttl); a.set_aspect('equal'); a.set_xlim(0, 1); a.set_ylim(0, 1); plt.colorbar(sc, ax=a, shrink=.8)
ax[2].scatter(p_sim * scale, p_intr, s=14, alpha=.6)
lim = [min(p_intr.min(), (p_sim * scale).min()), max(p_intr.max(), (p_sim * scale).max())]
ax[2].plot(lim, lim, 'k--', lw=1); ax[2].set_xlabel('sim (scaled)'); ax[2].set_ylabel('intrinsic')
ax[2].set_title(f'corr={corr:.4f}'); ax[2].set_aspect('equal')
fig.suptitle('Residual pressure: intrinsic solver vs direct simulation (spherical reference)')
plt.tight_layout(rect=[0, 0, 1, 0.95])
pth = os.path.join(HERE, 'sphere_sim_compare.png')
plt.savefig(pth, dpi=150, bbox_inches='tight'); plt.close()
print(f"saved {os.path.basename(pth)}")
