"""Run optimized solver on 13x13, 15x15, 20x20 networks.
10 eta values (0.1 to 0.49), 30 trials each. Plot with nu in [-0.5, 0.5]."""
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
sizes = [(13, 13), (15, 15), (20, 20)]

all_results = {}

for size in sizes:
    label = f"{size[0]}x{size[1]}"
    print(f"\n{'='*60}", flush=True)
    print(f"  Network size: {label}", flush=True)
    print(f"{'='*60}", flush=True)

    nu_data = np.zeros((len(eta_values), n_trials))
    n_tri_first = 0

    t_total_start = time.perf_counter()

    for i, eta in enumerate(eta_values):
        for trial in range(n_trials):
            seed = int(eta * 10000) + trial + size[0] * 1000
            np.random.seed(seed)
            DT = opt.generate_foam_points(size, eta)
            if i == 0 and trial == 0:
                n_tri_first = len(DT.simplices)
            opt.analyze_elastic_struct(DT)
            nu_data[i, trial] = DT.PoissonsRatio

        print(f"  eta={eta:.3f}: mean nu={np.mean(nu_data[i]):.4f} "
              f"std={np.std(nu_data[i]):.4f}", flush=True)

    t_total = time.perf_counter() - t_total_start
    n_runs = len(eta_values) * n_trials
    print(f"\n  Total time for {label}: {t_total:.2f}s "
          f"({n_runs} runs, ~{t_total/n_runs*1000:.1f}ms/run, "
          f"~{n_tri_first} triangles)", flush=True)

    all_results[label] = {
        'nu': nu_data, 'n_tri': n_tri_first, 'time': t_total
    }

# ---- Plot ----
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

for ax, (label, res) in zip(axes, all_results.items()):
    nu = res['nu']
    nu_mean = np.mean(nu, axis=1)
    nu_std = np.std(nu, axis=1)

    ax.errorbar(eta_values, nu_mean, yerr=nu_std, fmt='o-',
                color='C0', capsize=4, markersize=6, linewidth=1.5,
                ecolor='C0', elinewidth=1.2, capthick=1.2)

    ax.set_ylim(-0.5, 0.5)
    ax.set_xlabel(r'$\eta$', fontsize=13)
    ax.set_title(f'{label}  ({res["n_tri"]} tri, {res["time"]:.1f}s)',
                 fontsize=12)
    ax.grid(alpha=0.3)
    ax.axhline(y=1/3, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.4)

axes[0].set_ylabel(r"Poisson's ratio $\nu$", fontsize=13)

fig.suptitle(r"Poisson's ratio vs $\eta$ — Optimized solver, "
             f"30 trials, mean $\\pm$ std",
             fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig('/home/user/MATERIALIZE/benchmarking/poisson_13_15_20.png',
            dpi=150, bbox_inches='tight')
print(f"\nSaved: benchmarking/poisson_13_15_20.png", flush=True)

print(f"\n{'='*60}")
print("TIMING SUMMARY")
print(f"{'='*60}")
for label, res in all_results.items():
    n_runs = len(eta_values) * n_trials
    print(f"  {label:>6s}: {res['n_tri']:5d} triangles, "
          f"{res['time']:7.2f}s total, "
          f"{res['time']/n_runs*1000:6.1f}ms/run  ({n_runs} runs)")
