r"""Design run: a HIGHLY ANISOTROPIC directional Poisson-ratio response, 10 different networks.

Target: nu(theta) = A*cos(2 theta)  ->  nu = +A along theta=0, nu = -A along theta=90deg
(a strongly direction-dependent Poisson ratio). E(theta) held ~flat but DOWN-WEIGHTED so the
designer prioritises the anisotropic nu shape. Keeps 10 distinct trustworthy designs, each
independently simulated, and draws them.

Run:  "C:\Users\doron\anaconda3\python.exe" "Phase 5\verifications\run_anisotropic.py"
"""
import os, sys
import numpy as np
import torch
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))   # verifications/ -> repo
sys.path.insert(0, os.path.join(REPO, 'Phase 3', 'verifications'))
import _common as C
from inverse_design import ANG
torch.set_default_dtype(torch.float64)
sys.path.insert(0, os.path.join(REPO, 'Phase 5'))
import designer, seeds, gallery

# ---- target -----------------------------------------------------------------------------------
A = 0.30
nu_target = A * np.cos(2 * ANG)          # +A at 0 deg, -A at 90 deg (period pi)
E_target = np.ones_like(ANG)             # flat stiffness ~1 (down-weighted below)
i0, i90 = 0, len(ANG) // 2               # ANG=linspace(0,pi,37): idx 18 ~ 90 deg

print("=" * 96)
print(f"TARGET: highly anisotropic nu(theta) = {A}*cos(2 theta)  ->  nu(0deg)={+A:+.2f}, "
      f"nu(90deg)={-A:+.2f}  (anisotropy {2*A:.2f}); E~1 (down-weighted)")
print("=" * 96)

# ---- a rich pool, heavy on ANISOTROPIC lattices (good raw material for directional nu) ---------
pool = designer.topology_pool(n_random=12, n_nodes=90, n_flip_variants=6, flips=6, seed=1)
for name, geo in seeds.seed_bravais(phi_vals=(0.5, 0.7, 1.4, 1.8), psi_vals=(0.6, 1.5),
                                    etas=(0.0, 0.2), seeds=(0,), half=6):
    pool.append(designer._tag(geo, name))
print(f"pool: {len(pool)} candidate topologies "
      f"({sum(designer._is_nondelaunay(g) for g in pool)} non-Delaunay / flipped)")

# ---- design: prioritise nu (nu_weight >> E_weight), keep 10 -----------------------------------
reports = designer.design(nu_target, E_target, tag='aniso', pool=pool, keep=10,
                          n_iter=100, n_restarts=2, reg=0.01,
                          nu_weight=3.0, E_weight=0.3)

# ---- results table ---------------------------------------------------------------------------
print("\n" + "-" * 96)
print(f"{'#':>2} {'network shape':32s} {'nonDel':>6} {'nu(0)':>7} {'nu(90)':>7} "
      f"{'anisotropy':>10} {'tgt_err':>8} {'gap':>7}")
print("-" * 96)
for r in reports:
    nu0, nu90 = r['nu_sim'][i0], r['nu_sim'][i90]
    print(f"{r['rank']:>2} {r['seed_name'][:32]:32s} {str(r['is_nonDelaunay']):>6} "
          f"{nu0:>+7.3f} {nu90:>+7.3f} {nu0 - nu90:>+10.3f} "
          f"{r['target_err_sim']:>8.3f} {r['solver_sim_gap']:>7.3f}")
print("-" * 96)
print(f"target anisotropy nu(0)-nu(90) = {2*A:+.2f}")

best = max(reports, key=lambda r: r['nu_sim'][i0] - r['nu_sim'][i90])   # most anisotropic
print(f"\nMOST anisotropic design: '{best['seed_name']}' "
      f"nu(0)={best['nu_sim'][i0]:+.3f}, nu(90)={best['nu_sim'][i90]:+.3f} "
      f"(anisotropy {best['nu_sim'][i0]-best['nu_sim'][i90]:+.3f}), "
      f"trustworthy={best['trustworthy']}")
print(f"kept {len(reports)} designs; all saved to Phase 5/networks/design_aniso_*.npz")

# ---- draw all designs ------------------------------------------------------------------------
paths = [r['path'] for r in reports]
out = os.path.join(REPO, 'Phase 5', 'anisotropic_gallery.png')
gallery.gallery(paths, out, ncols=5)
print(f"saved {out}")
print("ANISOTROPIC RUN DONE")
