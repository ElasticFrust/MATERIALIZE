"""
Pure MF comparison: standard vs area-weighted, distort-first triangulation.

Mesh order: perturb vertices → Delaunay triangulate → trim edge triangles (85%).
No KKT correction in either case.

Outputs:
  mf_distort_first_data.npz   — raw arrays
  mf_distort_first.png        — E/E0 vs η, both MF variants
"""
import sys, os, time
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'Phase 2'))

import Disc_2_Cont_optimized as D2C
import forward_solver_torch as fst

ETA_VALUES = np.linspace(0.0, 0.5, 11)
N_TRIALS   = 10
SIZE       = (20, 20)
TRIM_FRAC  = 0.85
CACHE      = os.path.join(os.path.dirname(__file__), 'mf_distort_first_data.npz')
OUT        = os.path.join(os.path.dirname(__file__), 'mf_distort_first.png')

n_eta = len(ETA_VALUES)

if os.path.exists(CACHE):
    print(f"Loading cached data from {CACHE}")
    d = np.load(CACHE)
    Ex_std = d['Ex_std']
    Ex_aw  = d['Ex_aw']
    etas   = d['etas']
else:
    Ex_std = np.full((n_eta, N_TRIALS), np.nan)
    Ex_aw  = np.full((n_eta, N_TRIALS), np.nan)
    t0 = time.time()

    for i_eta, eta in enumerate(ETA_VALUES):
        for trial in range(N_TRIALS):
            seed = 100 * i_eta + trial
            np.random.seed(seed)

            DT = D2C.generate_foam_distort_first(SIZE, eta, trim_frac=TRIM_FRAC)
            solver, rigs, rl = fst.from_triangulation(DT)

            res_std = solver.forward(rigs, rl, area_weighted=False, use_kkt=False)
            res_aw  = solver.forward(rigs, rl, area_weighted=True,  use_kkt=False)

            Ex_std[i_eta, trial] = res_std['young'].item()
            Ex_aw [i_eta, trial] = res_aw ['young'].item()

            elapsed = time.time() - t0
            print(f"  η={eta:.2f} t={trial:2d}  "
                  f"E_std={Ex_std[i_eta,trial]:.5f}  "
                  f"E_aw={Ex_aw[i_eta,trial]:.5f}  "
                  f"{elapsed:.0f}s", flush=True)
        print()

    etas = np.asarray(ETA_VALUES)
    np.savez(CACHE, Ex_std=Ex_std, Ex_aw=Ex_aw, etas=etas)
    print(f"Data saved to {CACHE}")

# ── Normalise ─────────────────────────────────────────────────────────────────
E0 = np.nanmedian(Ex_std[0])   # Young's modulus of uniform network
Estd_n = Ex_std / E0
Eaw_n  = Ex_aw  / E0

print(f"\nE0 (std MF, η=0) = {E0:.5f}")
print(f"{'eta':>6}  {'std_med':>8}  {'aw_med':>8}")
for i, eta in enumerate(etas):
    print(f"{eta:6.2f}  {np.nanmedian(Estd_n[i]):8.3f}  {np.nanmedian(Eaw_n[i]):8.3f}")

# ── Plot ──────────────────────────────────────────────────────────────────────
jitter = [-0.005, 0.005]
fig, ax = plt.subplots(figsize=(8, 5))

for arr, label, color, marker, ls, jit in [
    (Estd_n, 'Standard MF',      'C0', 'o', '-',  jitter[0]),
    (Eaw_n,  'Area-weighted MF', 'C1', 's', '--', jitter[1]),
]:
    med = np.nanmedian(arr, axis=1)
    q1  = np.nanpercentile(arr, 25, axis=1)
    q3  = np.nanpercentile(arr, 75, axis=1)
    ax.errorbar(
        etas + jit, med,
        yerr=[np.clip(med - q1, 0, None), np.clip(q3 - med, 0, None)],
        fmt=f'{marker}{ls}', color=color, capsize=4,
        markersize=6, label=label, alpha=0.9, lw=1.8,
    )
    ax.fill_between(etas, q1, q3, alpha=0.12, color=color)

ax.set_xlabel(r'Disorder $\eta$', fontsize=13)
ax.set_ylabel(r'$E / E_0$', fontsize=13)
ax.set_xlim(-0.02, 0.52)
ax.axhline(1.0, color='gray', lw=0.8, ls=':')
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
ax.set_title(
    rf"Young's modulus vs disorder — pure MF, no KKT"
    "\n"
    rf"distort-first triangulation, trim={TRIM_FRAC}, "
    rf"20×20 network, {N_TRIALS} trials",
    fontsize=11)

fig.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches='tight')
print(f"\nSaved: {OUT}")
