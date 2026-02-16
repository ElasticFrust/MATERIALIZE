"""
50x50 network with adapted topology (Delaunay AFTER perturbation).
30 trials x 10 eta values.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import scipy as sp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import Disc_2_Cont_optimized as D2C


def generate_perturbed_then_delaunay(size, eta):
    """Mathematica approach: perturb points FIRST, then Delaunay."""
    v1 = np.array([1, 0])
    v2 = np.array([0.5, np.sqrt(3)/2])
    MaxPos = int(round(2 * (max(size) / min([np.sqrt(3)/2, 1]))))

    points = []
    for n in range(-MaxPos, MaxPos):
        for m in range(-MaxPos, MaxPos):
            theta = 2 * np.pi * np.random.rand()
            pt = n*v1 + m*v2 + eta * np.array([np.cos(theta), np.sin(theta)])
            points.append(pt)
    points = np.array(points)

    mask = ((points[:, 0] <= size[0]+2) & (points[:, 0] >= -(size[0]+2)) &
            (points[:, 1] <= size[1]+2) & (points[:, 1] >= -(size[1]+2)))
    points = points[mask]

    DM = sp.spatial.Delaunay(points)

    centroids = np.mean(DM.points[DM.simplices], axis=1)
    goods = ((np.abs(centroids[:, 0]) <= size[0]) &
             (np.abs(centroids[:, 1]) <= size[1]))
    DM.all_simplices = DM.simplices
    DM.simplices = DM.simplices[np.where(goods)]

    return DM


network_size = (50, 50)
etas = np.linspace(0.1, 0.49, 10)
n_trials = 30

results = {eta: [] for eta in etas}

print("=" * 60)
print(f"  50x50 adapted topology (Delaunay after perturbation)")
print("=" * 60)

t0 = time.time()

for eta in etas:
    t1 = time.time()
    for trial in range(n_trials):
        np.random.seed(1000 * trial + int(eta * 1000))
        DT = generate_perturbed_then_delaunay(network_size, eta)
        D2C.analyze_elastic_struct(DT)
        results[eta].append(DT.PoissonsRatio)

    nu = np.array(results[eta])
    dt = time.time() - t1
    print(f"  eta={eta:.3f}: nu={np.mean(nu):+.4f} +/- {np.std(nu):.4f}  ({dt:.1f}s)")

total_time = time.time() - t0
print(f"\nTotal: {total_time:.1f}s")

# --- Plot ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

eta_arr = np.array(list(etas))
means = np.array([np.mean(results[e]) for e in etas])
stds  = np.array([np.std(results[e])  for e in etas])

ax = axes[0]
ax.errorbar(eta_arr, means, yerr=stds, fmt='o-', color='blue', capsize=4)
ax.set_xlabel('eta')
ax.set_ylabel('Poisson ratio')
ax.set_title('50x50 adapted topology — Mean +/- Std')
ax.set_ylim(-0.5, 0.5)
ax.axhline(y=1/3, color='gray', ls='--', alpha=0.5, label='nu=1/3')
ax.axhline(y=0, color='gray', ls=':', alpha=0.5)
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
for eta in etas:
    ax.scatter([eta]*n_trials, results[eta], marker='o', c='blue',
               alpha=0.3, s=12, edgecolors='none')
ax.set_xlabel('eta')
ax.set_ylabel('Poisson ratio')
ax.set_title('50x50 adapted topology — Raw data')
ax.set_ylim(-0.5, 0.5)
ax.axhline(y=1/3, color='gray', ls='--', alpha=0.5)
ax.axhline(y=0, color='gray', ls=':', alpha=0.5)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('benchmarking/poisson_50x50_adapted.png', dpi=150)
print(f"Saved: benchmarking/poisson_50x50_adapted.png")
