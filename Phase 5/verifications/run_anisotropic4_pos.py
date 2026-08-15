r"""cos(4theta) anisotropy WITH vertex-position optimization — does moving nodes close the gap?

Takes the top-5 already-saved cos(4theta) designs (design_aniso4_*.npz) as starting topologies and
runs the full k + POSITION optimization (positions.design_with_positions) on each, then re-verifies
with the independent sim and compares nu(0), nu(45) and target error BEFORE (k-only) vs AFTER
(k + positions).  Saves design_aniso4pos_*.npz + montage + angular-response figure.

Run:  "C:\Users\doron\anaconda3\python.exe" "Phase 5\verifications\run_anisotropic4_pos.py"
"""
import os, sys, glob
import numpy as np
import torch
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'Phase 3', 'verifications'))
import _common as C
from inverse_design import ANG
torch.set_default_dtype(torch.float64)
sys.path.insert(0, os.path.join(REPO, 'Phase 5'))
import designer, positions, gallery, plot_responses

A = 0.30
nu_target = A * np.cos(4 * ANG)
E_target = np.ones_like(ANG)
i0, i45, i90 = 0, len(ANG) // 4, len(ANG) // 2
NDIR = os.path.join(REPO, 'Phase 5', 'networks')

# heavier-than-default position budget (CPU is free now).  CRUCIAL: match the k-design's
# nu-prioritised weighting (nu_weight=3, E_weight=0.3) so the position polish optimises the SAME
# objective — otherwise it flattens the down-weighted E at the expense of the nu anisotropy we want.
BUDGET = dict(n_outer=3, spsa_steps=35, n_iter=70, n_restarts=1, nu_weight=3.0, E_weight=0.3)

print("=" * 100)
print(f"cos(4theta) anisotropy + POSITION optimization (budget {BUDGET})")
print("target: nu(0)=nu(90)=+0.30, nu(45)=-0.30")
print("=" * 100)

srcs = sorted(glob.glob(os.path.join(NDIR, 'design_aniso4_*.npz')))[:5]
rows = []
for i, src in enumerate(srcs):
    geo, k0, C6p, meta = C.load_network(src)
    geo['seed_name'] = str(meta.get('seed_name', f'aniso4_{i}'))
    nu_b, _ = C.nu_E_theta(C.sim_bulk_C6(geo), ANG)          # BEFORE (k-only)
    err_b = float(meta.get('target_err_sim', np.nan))

    geoP, kP, hist = positions.design_with_positions(nu_target, E_target, geo,
                                                     verbose=False, **BUDGET)
    repP = designer.verify(geoP, kP, nu_target, E_target)                  # AFTER (k+positions)
    nu_a = repP['nu_sim']

    out = os.path.join(NDIR, f'design_aniso4pos_{i}.npz')
    C.apply_k_to_geo(geoP, kP)
    C.save_network(out, geoP, kP, C6_per=repP['C6_per'],
                   target_nu=nu_target.tolist(), target_E=E_target.tolist(),
                   solver_sim_gap=repP['solver_sim_gap'], target_err_sim=repP['target_err_sim'],
                   seed_name=geo['seed_name'] + '+pos')
    rows.append((i, geo['seed_name'], nu_b[i0], nu_a[i0], nu_b[i45], nu_a[i45],
                 err_b, repP['target_err_sim'], repP['solver_sim_gap']))
    print(f"  [{i}] {geo['seed_name'][:26]:26s}  "
          f"nu(0) {nu_b[i0]:+.3f}->{nu_a[i0]:+.3f}   nu(45) {nu_b[i45]:+.3f}->{nu_a[i45]:+.3f}   "
          f"err {err_b:.3f}->{repP['target_err_sim']:.3f}  gap {repP['solver_sim_gap']:.3f}")

print("\n" + "-" * 100)
print(f"{'#':>2} {'topology':26s} {'nu0 before':>10} {'nu0 after':>10} "
      f"{'nu45 before':>12} {'nu45 after':>11} {'err before':>11} {'err after':>10}")
print("-" * 100)
for (i, nm, nb0, na0, nb45, na45, eb, ea, gap) in rows:
    print(f"{i:>2} {nm[:26]:26s} {nb0:>+10.3f} {na0:>+10.3f} {nb45:>+12.3f} {na45:>+11.3f} "
          f"{eb:>11.3f} {ea:>10.3f}")
print("-" * 100)
d_err = np.mean([eb - ea for (_, _, _, _, _, _, eb, ea, _) in rows])
d_nu0 = np.mean([na0 - nb0 for (_, _, nb0, na0, _, _, _, _, _) in rows])
print(f"mean target-error change: {d_err:+.4f}  (positive = positions HELPED)")
print(f"mean nu(0) change: {d_nu0:+.4f}  (target +0.30; was ~+0.235 k-only)")

paths = [os.path.join(NDIR, f'design_aniso4pos_{i}.npz') for i in range(len(rows))]
gallery.gallery(paths, os.path.join(REPO, 'Phase 5', 'aniso4pos_gallery.png'), ncols=len(paths))
plot_responses.plot_responses('aniso4pos')
print("ANISOTROPIC-4THETA + POSITIONS DONE")
