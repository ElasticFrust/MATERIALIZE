"""
Do the intrinsic solver and the direct simulation give the same elastic RESPONSE for the spherical
reference?  Intrinsic C_eff from forward_dgbar; simulation C_eff by finite-difference homogenisation
of the relaxed prestressed network (apply macroscopic strain modes, re-relax, read macroscopic stress
— naturally includes geometric/prestress stiffening).  Compare ν(θ), E(θ).
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
offset = bR - (pts[bv] - pts[bu])
L = np.array([reg['BL1'][0], reg['BL2'][1]]); c0 = L / 2.0; A0 = L[0] * L[1]
R = 8.0; K = 1.0 / R ** 2


def lam_at(xy):
    d = xy - c0; d -= L * np.round(d / L)
    return 1.0 / (1.0 + K * (d ** 2).sum(-1) / 4.0) ** 2


gbar = np.zeros((n, 2, 2))
lt = lam_at(reg['centroids']); gbar[:, 0, 0] = lt; gbar[:, 1, 1] = lt
l0 = np.sqrt(lam_at(pts[bu] + bR / 2.0)) * np.hypot(bR[:, 0], bR[:, 1])

# ---- intrinsic ----
oi = forward_dgbar(reg, 1.0, gbar)
C6_intr = np.asarray(oi['elastic_tensor'])

# ---- simulation: relax at a given macroscopic F, return macroscopic stress ----
def relax_stress(Fdef, x_init):
    off = offset @ Fdef.T
    def eg(x):
        X = x.reshape(N, 2); d = X[bv] - X[bu] + off
        Lb = np.hypot(d[:, 0], d[:, 1]); dL = Lb - l0
        f = (dL / Lb)[:, None] * d
        g = np.zeros((N, 2)); np.add.at(g, bv, f); np.add.at(g, bu, -f)
        return 0.5 * (dL ** 2).sum(), g.ravel()
    res = minimize(eg, (x_init @ Fdef.T).ravel(), jac=True, method='L-BFGS-B',
                   options=dict(maxiter=20000, ftol=1e-15, gtol=1e-11))
    X = res.x.reshape(N, 2); d = X[bv] - X[bu] + off
    Lb = np.hypot(d[:, 0], d[:, 1]); t = Lb - l0
    A = A0 * abs(np.linalg.det(Fdef))
    sig = (t[:, None, None] / Lb[:, None, None] * (d[:, :, None] * d[:, None, :])).sum(0) / A
    return X, sig


X0, sig0 = relax_stress(np.eye(2), pts)                       # base prestressed state
dlt = 1e-4
modes = [np.array([[1., 0.], [0., 0.]]), np.array([[0., 0.], [0., 1.]]), np.array([[0., 1.], [1., 0.]])]
Ct = np.zeros((3, 3))                                         # [sxx,syy,sxy] <- [exx,eyy,exy(tensor)]
for j, e in enumerate(modes):
    _, sp = relax_stress(np.eye(2) + dlt * e, X0)
    _, sm = relax_stress(np.eye(2) - dlt * e, X0)
    ds = (np.array([sp[0, 0], sp[1, 1], sp[0, 1]]) - np.array([sm[0, 0], sm[1, 1], sm[0, 1]])) / (2 * dlt)
    Ct[:, j] = ds
Ct = 0.5 * (Ct + Ct.T)
C6_sim = np.array([Ct[0, 0], Ct[0, 2], Ct[0, 1], Ct[2, 2], Ct[1, 2], Ct[1, 1]])

th = np.linspace(0.0, np.pi, 361)
nu_i, E_i = C.nu_E_theta(C6_intr, th)
nu_s, E_s = C.nu_E_theta(C6_sim, th)
sE = np.dot(E_i, E_s) / np.dot(E_s, E_s)                      # E convention factor
print(f"spherical reference K=1/{R:.0f}²   (sim base prestress σ0=[{sig0[0,0]:+.4f},{sig0[1,1]:+.4f},{sig0[0,1]:+.4f}])")
print(f"  intrinsic (linear D2C, no geometric stiffening):  ν≈{nu_i.mean():+.4f}  E≈{E_i.mean():.4f}")
print(f"  simulation (full tangent, WITH geom. stiffening):  ν≈{nu_s.mean():+.4f}  E≈{E_s.mean():.4f}")
print(f"ν(θ):  max|intrinsic − sim| = {np.max(np.abs(nu_i - nu_s)):.4f}")
print(f"E(θ):  best-fit scale intrinsic/sim = {sE:.4f}")

fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.8))
ax[0].plot(np.degrees(th), nu_i, color='#1f77b4', lw=2.4, label='intrinsic')
ax[0].plot(np.degrees(th), nu_s, color='#d62728', lw=1.6, ls='--', label='simulation')
ax[0].axhline(1 / 3, color='0.7', lw=.6); ax[0].set_ylabel('ν(θ)'); ax[0].legend()
ax[1].plot(np.degrees(th), E_i, color='#1f77b4', lw=2.4, label='intrinsic')
ax[1].plot(np.degrees(th), E_s * sE, color='#d62728', lw=1.6, ls='--', label=f'simulation ×{sE:.3f}')
ax[1].set_ylabel('E(θ)'); ax[1].legend()
for a in ax:
    a.set_xlabel('θ (deg)'); a.set_xlim(0, 180); a.set_xticks(range(0, 181, 30)); a.grid(alpha=.3)
fig.suptitle('Elastic response for the spherical reference: intrinsic vs simulation')
plt.tight_layout(rect=[0, 0, 1, 0.95])
pth = os.path.join(HERE, 'sphere_response_compare.png')
plt.savefig(pth, dpi=150, bbox_inches='tight'); plt.close()
print(f"saved {os.path.basename(pth)}")
