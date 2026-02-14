"""Plot Poisson ratio vs eta for original and optimized solvers."""
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

nu_orig_all = {eta: [] for eta in eta_values}
nu_opt_all  = {eta: [] for eta in eta_values}

for eta in eta_values:
    for trial in range(n_trials):
        seed = int(eta * 10000) + trial
        np.random.seed(seed)

        DT_orig = orig.generate_foam_points(size, eta)
        DT_opt = copy.deepcopy(DT_orig)

        orig.analyze_elastic_struct(DT_orig)
        opt.analyze_elastic_struct(DT_opt)

        nu_orig_all[eta].append(DT_orig.PoissonsRatio)
        nu_opt_all[eta].append(DT_opt.PoissonsRatio)

        print(f"  eta={eta:.3f} trial={trial}: nu_orig={DT_orig.PoissonsRatio:.6f}, nu_opt={DT_opt.PoissonsRatio:.6f}", flush=True)

# Collect into arrays
etas = np.array(sorted(nu_orig_all.keys()))
nu_orig_raw = np.array([nu_orig_all[e] for e in etas])   # (10, 7)
nu_opt_raw  = np.array([nu_opt_all[e]  for e in etas])   # (10, 7)
nu_orig_mean = np.mean(nu_orig_raw, axis=1)
nu_orig_std  = np.std(nu_orig_raw, axis=1)
nu_opt_mean  = np.mean(nu_opt_raw, axis=1)
nu_opt_std   = np.std(nu_opt_raw, axis=1)

# --- Figure: two subplots side by side ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

# Left panel: raw data
for i, eta in enumerate(etas):
    jitter = 0.003
    ax1.scatter(np.full(n_trials, eta) - jitter, nu_orig_raw[i],
                marker='o', facecolors='none', edgecolors='C0', s=40, linewidths=1.2,
                label='Original' if i == 0 else None)
    ax1.scatter(np.full(n_trials, eta) + jitter, nu_opt_raw[i],
                marker='s', facecolors='none', edgecolors='C1', s=40, linewidths=1.2,
                label='Optimized (Woodbury)' if i == 0 else None)

ax1.set_xlabel(r'$\eta$', fontsize=13)
ax1.set_ylabel(r"Poisson's ratio $\nu$", fontsize=13)
ax1.set_title('Raw data (7 trials per $\\eta$)', fontsize=13)
ax1.legend(fontsize=11)
ax1.grid(alpha=0.3)

# Right panel: mean +/- std error bars
ax2.errorbar(etas - 0.004, nu_orig_mean, yerr=nu_orig_std, fmt='o-',
             color='C0', capsize=4, markersize=6, label='Original')
ax2.errorbar(etas + 0.004, nu_opt_mean, yerr=nu_opt_std, fmt='s--',
             color='C1', capsize=4, markersize=6, label='Optimized (Woodbury)')
ax2.set_xlabel(r'$\eta$', fontsize=13)
ax2.set_title(r'Mean $\pm$ std', fontsize=13)
ax2.legend(fontsize=11)
ax2.grid(alpha=0.3)

fig.suptitle(r"Poisson's ratio vs disorder $\eta$ — 7$\times$7 network, 7 trials",
             fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig('/home/user/MATERIALIZE/poisson_vs_eta.png', dpi=150, bbox_inches='tight')
print("\nSaved: poisson_vs_eta.png")
