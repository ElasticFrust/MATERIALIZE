"""Run optimized solver on 50x50 networks.
10 eta values, 30 trials each."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import Disc_2_Cont_optimized as opt

eta_values = np.linspace(0.1, 0.49, 10)
n_trials = 30
size = (50, 50)

nu_data = np.zeros((len(eta_values), n_trials))

print("=" * 60, flush=True)
print("  50x50 network", flush=True)
print("=" * 60, flush=True)
t0_all = time.perf_counter()
for i, eta in enumerate(eta_values):
    for trial in range(n_trials):
        seed = int(eta * 10000) + trial + 50000
        np.random.seed(seed)
        DT = opt.generate_foam_points(size, eta)
        opt.analyze_elastic_struct(DT)
        nu_data[i, trial] = DT.PoissonsRatio
    print(f"  eta={eta:.3f}: nu={np.mean(nu_data[i]):.4f} +/- {np.std(nu_data[i]):.4f}", flush=True)
t_total = time.perf_counter() - t0_all
print(f"  Total: {t_total:.1f}s ({t_total/300*1000:.0f}ms/run)", flush=True)

# --- Plot ---
fig, ax = plt.subplots(figsize=(8, 6))

mu = np.mean(nu_data, axis=1)
sigma = np.std(nu_data, axis=1)
ax.errorbar(eta_values, mu, yerr=sigma, fmt='o-',
            color='C0', capsize=4, markersize=6, linewidth=1.5)
ax.set_ylim(-0.5, 0.5)
ax.set_xlabel(r'$\eta$', fontsize=13)
ax.set_ylabel(r"Poisson's ratio $\nu$", fontsize=13)
ax.set_title(f'50x50  (~23k triangles, {t_total:.0f}s total)', fontsize=13)
ax.grid(alpha=0.3)
ax.axhline(y=1/3, color='gray', linestyle=':', alpha=0.5)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.4)

fig.suptitle(r"Poisson's ratio vs $\eta$ — 50$\times$50, 30 trials, mean $\pm$ std",
             fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(os.path.dirname(__file__), 'poisson_50x50.png'),
            dpi=150, bbox_inches='tight')
print("\nSaved: benchmarking/poisson_50x50.png", flush=True)
