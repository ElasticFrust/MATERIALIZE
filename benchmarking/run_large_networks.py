"""Run optimized solver on larger networks: 10x10, 11x11, 13x13.
10 eta values (0.1 to 0.49), 10 trials each. Plot results and report timing."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import Disc_2_Cont_optimized as opt

eta_values = np.linspace(0.1, 0.49, 10)
n_trials = 10
sizes = [(10, 10), (11, 11), (13, 13)]

all_results = {}

for size in sizes:
    label = f"{size[0]}x{size[1]}"
    print(f"\n{'='*60}", flush=True)
    print(f"  Network size: {label}", flush=True)
    print(f"{'='*60}", flush=True)

    nu_data = np.zeros((len(eta_values), n_trials))
    Y_data = np.zeros((len(eta_values), n_trials))
    n_tri_data = np.zeros((len(eta_values), n_trials), dtype=int)

    t_total_start = time.perf_counter()

    for i, eta in enumerate(eta_values):
        for trial in range(n_trials):
            seed = int(eta * 10000) + trial + size[0] * 1000
            np.random.seed(seed)

            DT = opt.generate_foam_points(size, eta)
            n_tri = len(DT.simplices)

            opt.analyze_elastic_struct(DT)

            nu_data[i, trial] = DT.PoissonsRatio
            Y_data[i, trial] = DT.YoungsModulus
            n_tri_data[i, trial] = n_tri

        print(f"  eta={eta:.3f}: mean nu={np.mean(nu_data[i]):.4f} "
              f"std={np.std(nu_data[i]):.4f}  "
              f"N_tri={n_tri_data[i, 0]}", flush=True)

    t_total = time.perf_counter() - t_total_start
    print(f"\n  Total time for {label}: {t_total:.2f}s "
          f"({len(eta_values)*n_trials} runs, "
          f"~{t_total/(len(eta_values)*n_trials)*1000:.1f}ms/run)", flush=True)

    all_results[label] = {
        'nu': nu_data, 'Y': Y_data, 'n_tri': n_tri_data,
        'time': t_total, 'size': size
    }

# ---- Plotting ----
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

for ax, (label, res) in zip(axes, all_results.items()):
    nu = res['nu']  # (10, n_trials)
    n_tri = res['n_tri'][0, 0]
    t = res['time']

    # Raw data scatter
    for i, eta in enumerate(eta_values):
        ax.scatter(np.full(n_trials, eta), nu[i],
                   marker='o', facecolors='none', edgecolors='C0',
                   s=25, linewidths=0.8, alpha=0.6,
                   label='Raw data' if i == 0 else None)

    # Mean + std errorbars
    nu_mean = np.mean(nu, axis=1)
    nu_std = np.std(nu, axis=1)
    ax.errorbar(eta_values, nu_mean, yerr=nu_std, fmt='s-',
                color='C1', capsize=4, markersize=5, linewidth=1.5,
                label=r'Mean $\pm$ std', zorder=5)

    ax.set_xlabel(r'$\eta$', fontsize=13)
    ax.set_title(f'{label}  ({n_tri} triangles, {t:.1f}s total)', fontsize=12)
    ax.legend(fontsize=10, loc='lower left')
    ax.grid(alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.4)

axes[0].set_ylabel(r"Poisson's ratio $\nu$", fontsize=13)

fig.suptitle(r"Poisson's ratio vs $\eta$ — Optimized solver, 10 trials per $\eta$",
             fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig('/home/user/MATERIALIZE/poisson_large_networks.png', dpi=150, bbox_inches='tight')
print(f"\nSaved: poisson_large_networks.png", flush=True)

# ---- Summary ----
print(f"\n{'='*60}")
print("TIMING SUMMARY")
print(f"{'='*60}")
for label, res in all_results.items():
    n_tri = res['n_tri'][0, 0]
    t = res['time']
    n_runs = len(eta_values) * n_trials
    print(f"  {label:>6s}: {n_tri:5d} triangles, {t:7.2f}s total, "
          f"{t/n_runs*1000:6.1f}ms/run  ({n_runs} runs)")
