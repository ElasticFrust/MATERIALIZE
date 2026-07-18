"""
INCOMPATIBLE reference: a spherical (constant positive Gaussian curvature) reference metric on the
regular network. Conformal form ḡ = λ(r) I, λ(r) = 1/(1 + K r²/4)²  (Gaussian curvature = K > 0),
r = distance to the nearest periodic image of the cell centre. A sphere cannot be laid flat, so the
reference is frustrated → a residual stress field that cannot be relaxed away.
Run the inhomogeneous solver and show the per-triangle residual stress.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..')))
import _common as C
from forward_solver_dgbar import forward_dgbar

reg = C.make_lattice(1.0, 1.0, half=6.0); n = len(reg['simplices'])
L = np.array([reg['BL1'][0], reg['BL2'][1]])                 # periodic box [12, 12.12]
c0 = L / 2.0
d = reg['centroids'] - c0
d -= L * np.round(d / L)                                     # wrap to nearest image
r = np.hypot(d[:, 0], d[:, 1])

R = 8.0; K = 1.0 / R ** 2                                    # Gaussian curvature K>0 (sphere)
lam = 1.0 / (1.0 + K * r ** 2 / 4.0) ** 2                    # conformal factor
gbar = np.zeros((n, 2, 2)); gbar[:, 0, 0] = lam; gbar[:, 1, 1] = lam

out = forward_dgbar(reg, 1.0, gbar)
s = out['stress_field']                                      # (n,3) [xx,xy,yy]
p = 0.5 * (s[:, 0] + s[:, 2])                                # residual pressure
mises = np.sqrt((s[:, 0] - s[:, 2]) ** 2 + 4 * s[:, 1] ** 2) # deviatoric magnitude

print(f"spherical reference: K={K:.4f} (R={R}), λ∈[{lam.min():.3f},{lam.max():.3f}]")
print(f"residual pressure  : rms={np.sqrt((p**2).mean()):.4f}  |max|={np.abs(p).max():.4f}")
print(f"residual deviatoric: rms={np.sqrt((mises**2).mean()):.4f}  |max|={mises.max():.4f}")
print(f"macroscopic σ0     = [{out['sigma0'][0]:+.4f}, {out['sigma0'][1]:+.4f}, {out['sigma0'][2]:+.4f}]")

frac = (reg['centroids'] / L) % 1.0                          # fractional coords in [0,1]^2
fig, ax = plt.subplots(1, 2, figsize=(11, 5))
sc0 = ax[0].scatter(frac[:, 0], frac[:, 1], c=lam, s=22, cmap='viridis')
ax[0].set_title('reference conformal factor λ (sphere cap)'); plt.colorbar(sc0, ax=ax[0], shrink=.8)
sc1 = ax[1].scatter(frac[:, 0], frac[:, 1], c=p, s=22, cmap='RdBu_r')
ax[1].set_title('residual pressure  ½(σxx+σyy)'); plt.colorbar(sc1, ax=ax[1], shrink=.8)
for a in ax:
    a.set_aspect('equal'); a.set_xlim(0, 1); a.set_ylim(0, 1); a.set_xlabel('x'); a.set_ylabel('y')
fig.suptitle(f'Spherical (incompatible) reference: residual stress   K=1/{R:.0f}²')
plt.tight_layout(rect=[0, 0, 1, 0.95])
pth = os.path.join(HERE, 'sphere_reference.png')
plt.savefig(pth, dpi=150, bbox_inches='tight'); plt.close()
print(f"saved {os.path.basename(pth)}")
