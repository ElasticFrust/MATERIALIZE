"""
Combined summary plot: ν and E vs η for all 4 solver variants.

Loads all_models_comp_data.npz and produces a single 2-panel figure:
  Left:  Poisson ratio ν = (ν_xy + ν_yx)/2  — median ± IQR
  Right: Young's modulus E = (E_x + E_y)/2  — median ± IQR
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CACHE = os.path.join(os.path.dirname(__file__), 'all_models_comp_data.npz')
OUT   = os.path.join(os.path.dirname(__file__), 'all_models_comp_combined.png')

CASES = [
    ('Regular MF',        'C0', 'o',  '-'),
    ('Area-weighted MF',  'C1', 's',  '--'),
    ('Regular MF + KKT',  'C2', '^',  '-'),
    ('Area-wtd MF + KKT', 'C3', 'D',  '--'),
]

d = np.load(CACHE)
etas     = d['etas']
nuxy_arr = d['nuxy_arr']
nuyx_arr = d['nuyx_arr']
Ex_arr   = d['Ex_arr']
Ey_arr   = d['Ey_arr']

# Scalar averages
nu_arr = 0.5 * (nuxy_arr + nuyx_arr)   # (4, 11, 10)
E_arr  = 0.5 * (Ex_arr   + Ey_arr)

N_TRIALS = nu_arr.shape[2]
ETA_STABLE = 0.35
jitter = np.linspace(-0.009, 0.009, len(CASES))

NU_LO, NU_HI = -1.0, 0.45
nu_clip = np.clip(nu_arr, NU_LO, NU_HI)

valid_E = E_arr[np.isfinite(E_arr)]
E_lo = max(0.0, float(np.nanpercentile(valid_E, 2)))
E_hi = float(np.nanpercentile(valid_E, 98)) * 1.1
E_clip = np.clip(E_arr, E_lo, E_hi)

fig, (ax_nu, ax_E) = plt.subplots(1, 2, figsize=(14, 5))

for ax, arr, ylabel, ylo, yhi in [
    (ax_nu, nu_clip, r"Poisson's ratio $\nu$",  NU_LO, NU_HI),
    (ax_E,  E_clip,  r"Young's modulus $E$",    E_lo,  E_hi),
]:
    for c, (label, color, marker, ls) in enumerate(CASES):
        data = arr[c]                         # (n_eta, n_trials)
        med = np.nanmedian(data, axis=1)
        q1  = np.nanpercentile(data, 25, axis=1)
        q3  = np.nanpercentile(data, 75, axis=1)
        ax.errorbar(
            etas + jitter[c], med,
            yerr=[np.clip(med - q1, 0, None), np.clip(q3 - med, 0, None)],
            fmt=f'{marker}{ls}', color=color, capsize=3,
            markersize=5, label=label, alpha=0.9,
        )
        ax.fill_between(etas, q1, q3, alpha=0.08, color=color)

    ax.axvline(ETA_STABLE, color='gray', lw=1.2, ls=':', alpha=0.6,
               label=rf'$\eta={ETA_STABLE}$ (stability)')
    ax.set_xlabel(r'Disorder $\eta$', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_ylim(ylo, yhi)
    ax.legend(fontsize=9, loc='best')
    ax.grid(alpha=0.3)

n_size, n_eta = 20, len(etas)
fig.suptitle(
    rf"Elastic constants vs disorder $\eta$ — 4 solver variants  "
    rf"(20×20 network, {N_TRIALS} trials each)",
    fontsize=13, y=1.02,
)
fig.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches='tight')
print(f"Saved: {OUT}")
