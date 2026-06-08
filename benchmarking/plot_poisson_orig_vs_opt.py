"""Plot 1: Poisson ratio original vs optimized (scatter + zoomed mean+std)."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import Disc_2_Cont_original as orig
import Disc_2_Cont_optimized as opt

eta_values = np.linspace(0.1, 0.49, 10)
n_trials = 7
size = (7, 7)

nu_orig_all = []
nu_opt_all = []
eta_list = []

for eta in eta_values:
    for trial in range(n_trials):
        seed = int(eta * 10000) + trial
        np.random.seed(seed)
        DT_orig = orig.generate_foam_points(size, eta)
        DT_opt = copy.deepcopy(DT_orig)
        orig.analyze_elastic_struct(DT_orig)
        opt.analyze_elastic_struct(DT_opt)
        nu_orig_all.append(DT_orig.PoissonsRatio)
        nu_opt_all.append(DT_opt.PoissonsRatio)
        eta_list.append(eta)
        print(f"  eta={eta:.3f} trial={trial} done", flush=True)

nu_orig_all = np.array(nu_orig_all)
nu_opt_all = np.array(nu_opt_all)
eta_arr = np.array(eta_list)

# Reshape to (10, 7)
nu_orig_2d = nu_orig_all.reshape(len(eta_values), n_trials)
nu_opt_2d = nu_opt_all.reshape(len(eta_values), n_trials)

# ---- Figure 1: nu_orig vs nu_opt scatter + zoomed mean+std ----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left: scatter of nu_orig vs nu_opt, colored by eta
sc = ax1.scatter(nu_orig_all, nu_opt_all, c=eta_arr, cmap='viridis',
                 s=50, edgecolors='k', linewidths=0.5, zorder=3)
lims = [min(nu_orig_all.min(), nu_opt_all.min()) * 1.05,
        max(nu_orig_all.max(), nu_opt_all.max()) * 1.05]
ax1.plot(lims, lims, 'r--', linewidth=1, label='y = x', zorder=1)
ax1.set_xlim(lims)
ax1.set_ylim(lims)
ax1.set_xlabel(r"$\nu$ (Original solver)", fontsize=13)
ax1.set_ylabel(r"$\nu$ (Optimized / Woodbury)", fontsize=13)
ax1.set_title(r"$\nu_{\mathrm{orig}}$ vs $\nu_{\mathrm{opt}}$ — all 70 runs", fontsize=13)
ax1.legend(fontsize=11, loc='upper left')
ax1.set_aspect('equal')
ax1.grid(alpha=0.3)
cb = fig.colorbar(sc, ax=ax1, label=r'$\eta$')

# Right: zoomed mean+std in range [-0.2, 0.5]
nu_orig_mean = np.mean(nu_orig_2d, axis=1)
nu_orig_std = np.std(nu_orig_2d, axis=1)
nu_opt_mean = np.mean(nu_opt_2d, axis=1)
nu_opt_std = np.std(nu_opt_2d, axis=1)

ax2.errorbar(eta_values - 0.004, nu_orig_mean, yerr=nu_orig_std, fmt='o-',
             color='C0', capsize=4, markersize=6, label='Original')
ax2.errorbar(eta_values + 0.004, nu_opt_mean, yerr=nu_opt_std, fmt='s--',
             color='C1', capsize=4, markersize=6, label='Optimized (Woodbury)')
ax2.set_ylim(-0.2, 0.5)
ax2.set_xlabel(r'$\eta$', fontsize=13)
ax2.set_ylabel(r"Poisson's ratio $\nu$", fontsize=13)
ax2.set_title(r"Mean $\pm$ std (zoomed $\nu \in [-0.2, 0.5]$)", fontsize=13)
ax2.legend(fontsize=11)
ax2.grid(alpha=0.3)
ax2.axhline(y=1/3, color='gray', linestyle=':', alpha=0.5, label=r'$\nu=1/3$')

fig.suptitle(r"Poisson's ratio comparison — 7$\times$7 network, 7 trials per $\eta$",
             fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig('/home/user/MATERIALIZE/poisson_orig_vs_opt.png', dpi=150, bbox_inches='tight')
print("\nSaved: poisson_orig_vs_opt.png", flush=True)
