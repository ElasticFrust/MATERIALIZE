"""
Young's modulus comparison: regular MF vs volume-weighted MF (paper Sec. II D).

Paper formula: replace arithmetic constraint Σ_s δg(s)=0 with
volume-weighted constraint Σ_s √ḡ_s δg(s)=0, where √ḡ_s is the
metric determinant density (∝ area_s in 2D Euclidean). This gives
normalized weights w_s = area_s / total_area in the Woodbury formula.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CACHE = os.path.join(os.path.dirname(__file__), 'all_models_comp_data.npz')
OUT   = os.path.join(os.path.dirname(__file__), 'young_mf_comparison.png')

d    = np.load(CACHE)
etas = d['etas']

# Cases 0=Regular MF (arithmetic), 1=Area-weighted MF (volume-weighted, paper)
# Use Ex (x-loading Young's modulus); network is nearly isotropic at low η
Ex_reg = d['Ex_arr'][0]   # (n_eta, n_trials)
Ex_aw  = d['Ex_arr'][1]

N_TRIALS = Ex_reg.shape[1]
E0 = np.nanmedian(Ex_reg[0])   # Young's modulus at η=0 (normalisation)

# Normalise
Ereg_n = Ex_reg / E0
Eaw_n  = Ex_aw  / E0

jitter = [-0.005, 0.005]
fig, ax = plt.subplots(figsize=(8, 5))

for arr, label, color, marker, ls, jit in [
    (Ereg_n, 'Regular MF',          'C0', 'o', '-',  jitter[0]),
    (Eaw_n,  'Volume-weighted MF',  'C1', 's', '--', jitter[1]),
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
ax.set_ylabel(r'$Y / Y_0$', fontsize=13)
ax.set_xlim(-0.01, 0.51)
ax.set_ylim(0, 1.05)
ax.axhline(1.0, color='gray', lw=0.8, ls=':')
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
ax.set_title(
    rf"Young's modulus vs disorder — regular MF vs volume-weighted MF"
    "\n"
    rf"20×20 network, {N_TRIALS} trials, $Y_0={E0:.4f}$",
    fontsize=11)

fig.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches='tight')
print(f"Saved: {OUT}")
