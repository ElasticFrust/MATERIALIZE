"""Compare full vs edge-trimmed Poisson ratio on 20x20 networks.
Trimming: keep only triangles whose centroids are within 85% of max dimensions.
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
size = (20, 20)
trim_frac = 0.85


def compute_poisson_youngs(elastic_tensor):
    """Compute Poisson ratio and Young's modulus from 6-component tensor."""
    C = elastic_tensor
    nu = (C[2]*C[3] - C[1]*C[4]) / (C[0]*C[3] - C[1]**2)
    Y = (C[2]**2*C[3] - 2*C[1]*C[2]*C[4] + C[1]**2*C[5] +
         C[0]*(C[4]**2 - C[3]*C[5])) / (C[1]**2 - C[0]*C[3])
    return nu, Y


nu_full = np.zeros((len(eta_values), n_trials))
nu_trim = np.zeros((len(eta_values), n_trials))

t0 = time.perf_counter()

for i, eta in enumerate(eta_values):
    for trial in range(n_trials):
        seed = int(eta * 10000) + trial + size[0] * 1000
        np.random.seed(seed)

        DT = opt.generate_foam_points(size, eta)
        opt.analyze_elastic_struct(DT)

        # Full (all triangles)
        nu_full[i, trial] = DT.PoissonsRatio

        # Trimmed: keep interior triangles
        centroids = np.mean(DT.points[DT.simplices], axis=1)
        xmin, xmax = centroids[:, 0].min(), centroids[:, 0].max()
        ymin, ymax = centroids[:, 1].min(), centroids[:, 1].max()
        half_w = (xmax - xmin) / 2
        half_h = (ymax - ymin) / 2
        cx = (xmax + xmin) / 2
        cy = (ymax + ymin) / 2
        mask = ((np.abs(centroids[:, 0] - cx) <= trim_frac * half_w) &
                (np.abs(centroids[:, 1] - cy) <= trim_frac * half_h))

        trimmed_tensor = np.mean(DT.ActualElasticTensor[mask], axis=0)
        nu_t, _ = compute_poisson_youngs(trimmed_tensor)
        nu_trim[i, trial] = nu_t

    n_kept = mask.sum()
    n_total = len(DT.simplices)
    print(f"  eta={eta:.3f}: full={np.mean(nu_full[i]):.4f}±{np.std(nu_full[i]):.4f}  "
          f"trimmed={np.mean(nu_trim[i]):.4f}±{np.std(nu_trim[i]):.4f}  "
          f"({n_kept}/{n_total} tri kept)", flush=True)

t_total = time.perf_counter() - t0
print(f"\nTotal time: {t_total:.1f}s", flush=True)

# ---- Plots ----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left: both curves on same axes, zoomed
nu_full_mean = np.mean(nu_full, axis=1)
nu_full_std = np.std(nu_full, axis=1)
nu_trim_mean = np.mean(nu_trim, axis=1)
nu_trim_std = np.std(nu_trim, axis=1)

ax1.errorbar(eta_values - 0.004, nu_full_mean, yerr=nu_full_std, fmt='o-',
             color='C0', capsize=4, markersize=6, linewidth=1.5,
             label='Full (all triangles)')
ax1.errorbar(eta_values + 0.004, nu_trim_mean, yerr=nu_trim_std, fmt='s--',
             color='C1', capsize=4, markersize=6, linewidth=1.5,
             label=f'Trimmed (interior {int(trim_frac*100)}%)')
ax1.set_ylim(-0.5, 0.5)
ax1.set_xlabel(r'$\eta$', fontsize=13)
ax1.set_ylabel(r"Poisson's ratio $\nu$", fontsize=13)
ax1.set_title(r'Mean $\pm$ std ($\nu \in [-0.5, 0.5]$)', fontsize=13)
ax1.legend(fontsize=11)
ax1.grid(alpha=0.3)
ax1.axhline(y=1/3, color='gray', linestyle=':', alpha=0.5)
ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.4)

# Right: difference (trimmed - full)
diff_mean = nu_trim_mean - nu_full_mean
diff_std = np.sqrt(nu_trim_std**2 + nu_full_std**2)  # propagated error (conservative)

ax2.errorbar(eta_values, diff_mean, yerr=diff_std, fmt='D-',
             color='C2', capsize=4, markersize=6, linewidth=1.5)
ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.6)
ax2.set_xlabel(r'$\eta$', fontsize=13)
ax2.set_ylabel(r"$\Delta\nu$ (trimmed $-$ full)", fontsize=13)
ax2.set_title('Difference: trimmed minus full', fontsize=13)
ax2.grid(alpha=0.3)

fig.suptitle(f"20x20 network — full vs edge-trimmed ({int(trim_frac*100)}% interior), "
             f"30 trials per $\\eta$",
             fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(os.path.dirname(__file__), 'poisson_trimmed_vs_full.png'),
            dpi=150, bbox_inches='tight')
print("Saved: benchmarking/poisson_trimmed_vs_full.png", flush=True)
