"""
Poisson ratio ν vs η — mean-field vs KKT — 50×50 network, 20 trials.
Saves cache and plot with _50x50 suffix.
"""
import sys, os, copy, time
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
N_TRIALS   = 20
SIZE       = (50, 50)

CACHE = os.path.join(os.path.dirname(__file__), 'poisson_vs_eta_kkt_50x50_data.npz')
print(f"size={SIZE}  n_trials={N_TRIALS}  n_eta={len(ETA_VALUES)}")
print(f"eta: {np.round(ETA_VALUES, 3).tolist()}\n")

t_start = time.time()

if os.path.exists(CACHE):
    print(f"Loading cached data from {CACHE}")
    d = np.load(CACHE)
    old_arr = d['old_arr']
    new_arr = d['new_arr']
    etas    = d['etas']
else:
    nu_old_all = []
    nu_new_all = []

    for i_eta, eta in enumerate(ETA_VALUES):
        row_old, row_new = [], []
        for trial in range(N_TRIALS):
            seed = 100 * i_eta + trial
            np.random.seed(seed)

            DT = D2C.generate_foam_points(SIZE, eta)

            DT_old = copy.deepcopy(DT)
            D2C.analyze_elastic_struct(DT_old)
            nu_o = DT_old.PoissonsRatio

            DT_new = copy.deepcopy(DT)
            solver, rigs, rl = fst.from_triangulation(DT_new)
            with torch.no_grad():
                nu_n = solver(rigs, rl)['poisson'].item()

            row_old.append(nu_o)
            row_new.append(nu_n)
            elapsed = time.time() - t_start
            print(f"  η={eta:.2f}  trial={trial:2d}  old={nu_o:+.5f}  new={nu_n:+.5f}"
                  f"  Δ={nu_n-nu_o:+.5f}  elapsed={elapsed:.0f}s", flush=True)
        print()
        nu_old_all.append(row_old)
        nu_new_all.append(row_new)

    etas    = np.asarray(ETA_VALUES)
    old_arr = np.array(nu_old_all)
    new_arr = np.array(nu_new_all)
    np.savez(CACHE, etas=etas, old_arr=old_arr, new_arr=new_arr)
    print(f"Data saved to {CACHE}")

old_mean = old_arr.mean(1); old_std = old_arr.std(1)
new_mean = new_arr.mean(1); new_std = new_arr.std(1)

YMIN, YMAX = -1.0, 0.45
ETA_STABLE = 0.35
old_clip = np.clip(old_arr, YMIN, YMAX)
new_clip = np.clip(new_arr, YMIN, YMAX)

old_med = np.median(old_clip, axis=1)
new_med = np.median(new_clip, axis=1)
old_q1  = np.percentile(old_clip, 25, axis=1)
old_q3  = np.percentile(old_clip, 75, axis=1)
new_q1  = np.percentile(new_clip, 25, axis=1)
new_q3  = np.percentile(new_clip, 75, axis=1)

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
jitter = 0.005

ax = axes[0]
for i, eta in enumerate(etas):
    kw_o = dict(marker='o', facecolors='none', edgecolors='C0', s=28, linewidths=1.2,
                label='Mean-field (no KKT)' if i == 0 else None)
    kw_n = dict(marker='s', facecolors='none', edgecolors='C1', s=28, linewidths=1.2,
                label='KKT-corrected' if i == 0 else None)
    ax.scatter(np.full(N_TRIALS, eta) - jitter, old_clip[i], **kw_o)
    ax.scatter(np.full(N_TRIALS, eta) + jitter, new_clip[i], **kw_n)

ax.axvline(ETA_STABLE, color='gray', lw=1.2, ls=':', alpha=0.7)
ax.text(ETA_STABLE + 0.003, YMIN + 0.07, 'unstable', color='gray',
        fontsize=9, rotation=90, va='bottom')
ax.set_ylim(YMIN, YMAX)
ax.set_xlabel(r'$\eta$', fontsize=13)
ax.set_ylabel(r"Poisson's ratio $\nu$", fontsize=13)
ax.set_title(f'Raw scatter ({N_TRIALS} trials per $\\eta$, clipped to [{YMIN},{YMAX}])', fontsize=12)
ax.legend(fontsize=11)
ax.grid(alpha=0.3)

ax = axes[1]
delta_med = new_med - old_med
ax.errorbar(etas - 0.006, old_med,
            yerr=[old_med - old_q1, old_q3 - old_med],
            fmt='o-', color='C0', capsize=4, markersize=6, label='Mean-field (no KKT)')
ax.errorbar(etas + 0.006, new_med,
            yerr=[new_med - new_q1, new_q3 - new_med],
            fmt='s--', color='C1', capsize=4, markersize=6, label='KKT-corrected')
ax.fill_between(etas, old_q1, old_q3, alpha=0.10, color='C0')
ax.fill_between(etas, new_q1, new_q3, alpha=0.10, color='C1')

ax2 = ax.twinx()
ax2.bar(etas, delta_med, width=0.012, alpha=0.25, color='gray', label=r'$\Delta\nu$ (new−old)')
ax2.set_ylabel(r'$\Delta\nu$ (KKT − MF)', fontsize=11, color='gray')
ax2.tick_params(axis='y', colors='gray')
ax2.axhline(0, color='gray', lw=0.8, ls='--')
ax2.legend(loc='lower left', fontsize=9)

ax.axvline(ETA_STABLE, color='gray', lw=1.2, ls=':', alpha=0.7)
ax.set_ylim(YMIN, YMAX)
ax.set_xlabel(r'$\eta$', fontsize=13)
ax.set_title(r'Median $\pm$ IQR  (robust)', fontsize=12)
ax.legend(fontsize=11)
ax.grid(alpha=0.3)

fig.suptitle(
    rf"$\nu(\eta)$: mean-field vs KKT correction — "
    rf"{SIZE[0]}$\times${SIZE[1]} network (~12650 vertices), {N_TRIALS} trials",
    fontsize=13, y=1.02,
)
fig.tight_layout()

out_path = os.path.join(os.path.dirname(__file__), 'poisson_vs_eta_kkt_50x50_new.png')
fig.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\nSaved: {out_path}")

print(f"\n{'η':>6}  {'ν_old':>9}  {'σ_old':>7}  {'ν_new':>9}  {'σ_new':>7}  {'Δν(new-old)':>12}")
print('-' * 58)
for i, eta in enumerate(etas):
    print(f"  {eta:.2f}   {old_mean[i]:+.5f}  {old_std[i]:.5f}   "
          f"{new_mean[i]:+.5f}  {new_std[i]:.5f}   {new_mean[i]-old_mean[i]:+.5f}")

total = time.time() - t_start
print(f"\nTotal time: {total/60:.1f} min")
