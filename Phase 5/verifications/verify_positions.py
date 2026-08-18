r"""Verification: VERTEX-POSITION OPTIMIZATION (Phase 5/positions.py) actually helps.

Target: isotropic near-auxetic nu = -0.15, E = 1.0 on a small regular lattice.
Checks:
  (a) the design loss after position optimization (design_with_positions) is LOWER than the
      k-only design loss on the same starting topology (same k-design budget/seed);
  (b) the final positioned design still passes the honesty check — INDEPENDENT sim via
      designer.verify with solver_sim_gap < 0.05;
  (c) prints sim mean nu/E vs target before (k-only) and after (k+positions).
Saves the positioned design + the two companion figures (network picture + angular response).

Run:  "C:\Users\doron\anaconda3\python.exe" "Phase 5\verifications\verify_positions.py"
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
import designer, positions, gallery, plot_responses

nu_target, E_target = -0.15, 1.0
N_ITER, N_RESTARTS, REG = 60, 1, 0.02          # small budget — CPU is contended

print("=" * 88)
print(f"POSITION-OPTIMIZATION VERIFICATION — target nu={nu_target}, E={E_target} "
      f"(flat profiles), lattice half=5")
print("=" * 88)

# ---- baseline: k-only design on the starting topology -----------------------------------------
geo0 = C.make_lattice(1.0, 1.0, half=5)
print(f"start topology: {len(geo0['simplices'])} triangles, {len(geo0['bond_u'])} bonds")
_, _, res0 = designer.design_on_topology(geo0, nu_target, E_target,
                                         n_iter=N_ITER, n_restarts=N_RESTARTS, reg=REG)
L0 = positions.loss_at(geo0, res0['k'], nu_target, E_target)
rep0 = designer.verify(geo0, res0['k'], nu_target, E_target)
print(f"\nk-only design:      loss = {L0:.4e}   (solver_sim_gap {rep0['solver_sim_gap']:.4f})")
print(f"  sim mean nu = {rep0['nu_sim'].mean():+.4f}  vs target {nu_target:+.4f}")
print(f"  sim mean E  = {rep0['E_sim'].mean():+.4f}  vs target {E_target:+.4f}")

# ---- k + positions: the alternating SPSA loop -------------------------------------------------
print("\nrunning design_with_positions (n_outer=2, spsa_steps=25) ...")
geoP, kP, hist = positions.design_with_positions(
    nu_target, E_target, geo0, n_outer=2, spsa_steps=25,
    n_iter=N_ITER, n_restarts=N_RESTARTS, reg=REG, seed=0, verbose=True)
LP = positions.loss_at(geoP, kP, nu_target, E_target)
print(f"\nk+positions design: loss = {LP:.4e}   "
      f"({len(geoP['simplices'])} tri, {len(geoP['bond_u'])} bonds)")
print("loss history:", ", ".join(f"{st}[{o}]={L:.3e}" for st, o, L in hist))

ok_a = LP < L0
print(f"\n(a) position optimization lowers the design loss: {L0:.4e} -> {LP:.4e}  "
      f"({'-' if ok_a else '+'}{abs(1 - LP / L0) * 100:.1f}%)")
print(("PASSED" if ok_a else "FAILED") + " (a): k+positions loss < k-only loss")

# ---- honesty check: independent sim on the positioned design ----------------------------------
repP = designer.verify(geoP, kP, nu_target, E_target)
ok_b = repP['solver_sim_gap'] < 0.05
print(f"\n(b) independent-sim honesty check: solver_sim_gap = {repP['solver_sim_gap']:.4f} "
      f"(tol 0.05), target_err_sim = {repP['target_err_sim']:.4f}")
print(("PASSED" if ok_b else "FAILED") + " (b): solver_sim_gap < 0.05")

print(f"\n(c) sim mean nu/E vs target:")
print(f"    before (k-only):      nu = {rep0['nu_sim'].mean():+.4f}, "
      f"E = {rep0['E_sim'].mean():+.4f}")
print(f"    after (k+positions):  nu = {repP['nu_sim'].mean():+.4f}, "
      f"E = {repP['E_sim'].mean():+.4f}")
print(f"    target:               nu = {nu_target:+.4f}, E = {E_target:+.4f}")

# ---- save the positioned design + the two companion figures -----------------------------------
nu_t = np.broadcast_to(np.asarray(nu_target, float), ANG.shape)
E_t = np.broadcast_to(np.asarray(E_target, float), ANG.shape)
npz = os.path.join(REPO, 'Phase 5', 'networks', 'design_posdemo_0.npz')
C.apply_k_to_geo(geoP, kP)
C.save_network(npz, geoP, kP, C6_per=repP['C6_per'],
               target_nu=nu_t.tolist(), target_E=E_t.tolist(),
               loss=LP, loss_k_only=L0,
               solver_sim_gap=repP['solver_sim_gap'],
               target_err_sim=repP['target_err_sim'],
               seed_name='bravais_reg_half5+pos')
print(f"\nsaved -> {npz}")
gallery.gallery([npz], os.path.join(REPO, 'Phase 5', 'posdemo_gallery.png'), ncols=1)
plot_responses.plot_responses('posdemo')

# ---- (d) default path: plain designer.design() runs the position polish automatically --------
print("\n" + "=" * 88)
print("(d) DEFAULT PATH — plain designer.design() on a tiny pool (optimize_positions defaults ON)")
print("=" * 88)
import seeds
tiny_pool = [designer._tag(C.make_lattice(1.0, 1.0, half=4), 'bravais_reg_half4')]
rec = seeds.random_patch(40, seed=0, process='poisson_disk')
tiny_pool.append(designer._tag(rec['geo'], rec['name']))
reports = designer.design(nu_target, E_target, tag='posdefault', pool=tiny_pool, keep=1,
                          n_iter=40, n_restarts=1, reg=REG,
                          pos_budget=dict(n_outer=1, spsa_steps=10, n_iter=40, n_restarts=1))
r0 = reports[0]
ok_d = r0['solver_sim_gap'] < 0.05
print(f"top design: '{r0['seed_name']}'  (position-polished: {r0['seed_name'].endswith('+pos')})")
print(f"  loss={r0['loss']:.4e}  target_err_sim={r0['target_err_sim']:.4f}  "
      f"solver_sim_gap={r0['solver_sim_gap']:.4f}")
print(("PASSED" if ok_d else "FAILED") + " (d): default design() path, solver_sim_gap < 0.05")

ok_all = ok_a and ok_b and ok_d
print("\nVERIFY_POSITIONS " + ("PASSED" if ok_all else "FAILED"))

# EXIT NON-ZERO when a check fails. Without this the script printed FAILED and still
# returned 0, so the clean-validation runner recorded a genuinely failing check as a
# success -- (a) position polish RAISED the loss, (b) solver_sim_gap 0.0646 > 0.05.
sys.exit(0 if ok_all else 1)
