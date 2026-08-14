r"""Design run: highly anisotropic — but REALIZABLE — directional Poisson ratio, 10 networks.

Physics note (why this replaces run_anisotropic.py's target): compliance-tensor symmetry
(Maxwell-Betti reciprocity) forces  nu(0)/E(0) = nu(90)/E(90)  for ANY stable material, so
nu(0)=+A, nu(90)=-A (a sign flip between the two axes) is UNREALIZABLE — and with flat E it even
forces nu(0)=nu(90) exactly.  The first run's designs all parked on this constraint surface
(nu/E equal at 0 and 90 to ~1%), correctly finding the closest physical material to an unphysical
request.  The realizable "highly anisotropic nu" is the 4-theta harmonic: the sign flip happens at
45 deg, which reciprocity does NOT forbid:

    nu(theta) = A*cos(4 theta)   ->  nu(0)=nu(90)=+A,  nu(45)=-A   (peak-to-peak 2A)

Run:  "C:\Users\doron\anaconda3\python.exe" "Phase 5\verifications\run_anisotropic4.py"
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
import designer, seeds, gallery, plot_responses

# ---- target: REALIZABLE strong nu anisotropy (sign flip at 45 deg, not 90) --------------------
A = 0.30
nu_target = A * np.cos(4 * ANG)          # +A at 0 and 90 deg, -A at 45 deg
E_target = np.ones_like(ANG)             # flat E is COMPATIBLE with this target (nu(0)=nu(90))
i0, i45, i90 = 0, len(ANG) // 4, len(ANG) // 2

print("=" * 96)
print(f"TARGET (realizable): nu(theta) = {A}*cos(4 theta)  ->  nu(0)=nu(90)={+A:+.2f}, "
      f"nu(45)={-A:+.2f}  (peak-to-peak {2*A:.2f}); E~1 (down-weighted)")
print("=" * 96)

# ---- rich pool, heavy on anisotropic lattices -------------------------------------------------
pool = designer.topology_pool(n_random=12, n_nodes=90, n_flip_variants=6, flips=6, seed=1)
for name, geo in seeds.seed_bravais(phi_vals=(0.5, 0.7, 1.4, 1.8), psi_vals=(0.6, 1.5),
                                    etas=(0.0, 0.2), seeds=(0,), half=6):
    pool.append(designer._tag(geo, name))
print(f"pool: {len(pool)} candidate topologies "
      f"({sum(designer._is_nondelaunay(g) for g in pool)} non-Delaunay / flipped)")

# ---- design: prioritise nu, keep 10 -----------------------------------------------------------
reports = designer.design(nu_target, E_target, tag='aniso4', pool=pool, keep=10,
                          n_iter=100, n_restarts=2, reg=0.01,
                          nu_weight=3.0, E_weight=0.3)

# ---- results table ----------------------------------------------------------------------------
print("\n" + "-" * 96)
print(f"{'#':>2} {'network shape':32s} {'nonDel':>6} {'nu(0)':>7} {'nu(45)':>7} {'nu(90)':>7} "
      f"{'pk-pk':>7} {'tgt_err':>8} {'gap':>7}")
print("-" * 96)
for r in reports:
    nu0, nu45, nu90 = r['nu_sim'][i0], r['nu_sim'][i45], r['nu_sim'][i90]
    print(f"{r['rank']:>2} {r['seed_name'][:32]:32s} {str(r['is_nonDelaunay']):>6} "
          f"{nu0:>+7.3f} {nu45:>+7.3f} {nu90:>+7.3f} "
          f"{max(nu0, nu90) - nu45:>+7.3f} "
          f"{r['target_err_sim']:>8.3f} {r['solver_sim_gap']:>7.3f}")
print("-" * 96)
print(f"target: nu(0)={+A:+.2f}, nu(45)={-A:+.2f}  (peak-to-peak {2*A:.2f})")

best = max(reports, key=lambda r: (r['nu_sim'][i0] + r['nu_sim'][i90]) / 2 - r['nu_sim'][i45])
print(f"\nMOST anisotropic design: '{best['seed_name']}' "
      f"nu(0)={best['nu_sim'][i0]:+.3f}, nu(45)={best['nu_sim'][i45]:+.3f}, "
      f"nu(90)={best['nu_sim'][i90]:+.3f}, trustworthy={best['trustworthy']}")
print(f"kept {len(reports)} designs; saved to Phase 5/networks/design_aniso4_*.npz")

# ---- figures: montage + angular response (project rule: ALWAYS both) --------------------------
paths = [r['path'] for r in reports]
gallery.gallery(paths, os.path.join(REPO, 'Phase 5', 'aniso4_gallery.png'), ncols=5)
plot_responses.plot_responses('aniso4')
print("ANISOTROPIC-4THETA RUN DONE")
