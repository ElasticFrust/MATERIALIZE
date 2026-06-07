"""
aaa_all_comparison — Poisson ratio and Young's modulus for all 4 solver variants.

Cases:
  1. Regular MF          — arithmetic mean, no KKT
  2. Area-weighted MF    — volume-weighted mean, no KKT
  3. Regular MF + KKT    — arithmetic mean + edge-compatibility correction
  4. Area-wtd MF + KKT   — volume-weighted mean + edge-compatibility correction

2×2 figure:
  Top left:    ν scatter (all trials)
  Top right:   ν median ± IQR
  Bottom left: E scatter (all trials)
  Bottom right:E median ± IQR
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CACHE = os.path.join(os.path.dirname(__file__), 'all_models_comp_data.npz')
OUT   = os.path.join(os.path.dirname(__file__), 'aaa_all_comparison.png')

CASES = [
    ('Regular MF',        'C0', 'o',  '-'),
    ('Area-weighted MF',  'C1', 's',  '--'),
    ('Regular MF + KKT',  'C2', '^',  '-'),
    ('Area-wtd MF + KKT', 'C3', 'D',  '--'),
]

d        = np.load(CACHE)
etas     = d['etas']
nu_arr   = 0.5 * (d['nuxy_arr'] + d['nuyx_arr'])   # scalar average (4, 11, 10)
E_arr    = 0.5 * (d['Ex_arr']   + d['Ey_arr'])

N_TRIALS  = nu_arr.shape[2]
ETA_STABLE = 0.35
jitter     = np.linspace(-0.009, 0.009, len(CASES))

NU_LO, NU_HI = -1.0, 0.45
nu_clip = np.clip(nu_arr, NU_LO, NU_HI)

valid_E = E_arr[np.isfinite(E_arr)]
E_lo    = max(0.0, float(np.nanpercentile(valid_E, 2)))
E_hi    = float(np.nanpercentile(valid_E, 98)) * 1.1
E_clip  = np.clip(E_arr, E_lo, E_hi)

fig, axes = plt.subplots(2, 2, figsize=(16, 10))

for row, (arr, clip_arr, ylabel, ylo, yhi) in enumerate([
    (nu_arr, nu_clip, r"Poisson's ratio $\nu$", NU_LO, NU_HI),
    (E_arr,  E_clip,  r"Young's modulus $E$",   E_lo,  E_hi),
]):
    for col in range(2):
        ax = axes[row, col]

        for c, (label, color, marker, ls) in enumerate(CASES):
            data = clip_arr[c]   # (11, 10)
            jit  = jitter[c]

            if col == 0:   # scatter
                for i_eta, eta in enumerate(etas):
                    ax.scatter(
                        np.full(N_TRIALS, eta) + jit, data[i_eta],
                        marker=marker, facecolors='none', edgecolors=color,
                        s=22, linewidths=1.1, alpha=0.7,
                        label=label if i_eta == 0 else None,
                    )
            else:          # median ± IQR
                med = np.nanmedian(data, axis=1)
                q1  = np.nanpercentile(data, 25, axis=1)
                q3  = np.nanpercentile(data, 75, axis=1)
                ax.errorbar(
                    etas + jit, med,
                    yerr=[np.clip(med - q1, 0, None), np.clip(q3 - med, 0, None)],
                    fmt=f'{marker}{ls}', color=color, capsize=3,
                    markersize=5, label=label, alpha=0.9,
                )
                ax.fill_between(etas, q1, q3, alpha=0.08, color=color)

        ax.axvline(ETA_STABLE, color='gray', lw=1.2, ls=':', alpha=0.6)
        ax.set_xlabel(r'Disorder $\eta$', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_ylim(ylo, yhi)
        ax.legend(fontsize=9, loc='best')
        ax.grid(alpha=0.3)

        if col == 0:
            ax.set_title(f'Raw scatter — {N_TRIALS} trials per η', fontsize=11)
        else:
            ax.set_title('Median ± IQR', fontsize=11)

fig.suptitle(
    rf"$\nu$ and $E$ vs disorder $\eta$ — 4 solver variants  "
    rf"(20×20 network, {N_TRIALS} trials each)",
    fontsize=14, y=1.01,
)
fig.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches='tight')
print(f"Saved: {OUT}")
