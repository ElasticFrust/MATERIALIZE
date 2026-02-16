"""
Compare Poisson ratio stability:
  1) Crystal topology: Delaunay BEFORE perturbation (current Python approach)
  2) Adapted topology: Delaunay AFTER perturbation (Mathematica approach)

Both use the same optimized Woodbury solver.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import scipy as sp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import Disc_2_Cont_optimized as D2C

def generate_crystal_then_perturb(size, eta):
    """Current Python approach: Delaunay on crystal, THEN perturb."""
    return D2C.generate_foam_points(size, eta)

def generate_perturbed_then_delaunay(size, eta):
    """Mathematica approach: perturb points FIRST, then Delaunay."""
    v1 = np.array([1, 0])
    v2 = np.array([0.5, np.sqrt(3)/2])
    MaxPos = int(round(2 * (max(size) / min([np.sqrt(3)/2, 1]))))

    # Generate perturbed points directly (like Mathematica)
    points = []
    for n in range(-MaxPos, MaxPos):
        for m in range(-MaxPos, MaxPos):
            theta = 2 * np.pi * np.random.rand()
            pt = n*v1 + m*v2 + eta * np.array([np.cos(theta), np.sin(theta)])
            points.append(pt)
    points = np.array(points)

    # Cut to size+2 for triangulation buffer
    mask = ((points[:, 0] <= size[0]+2) & (points[:, 0] >= -(size[0]+2)) &
            (points[:, 1] <= size[1]+2) & (points[:, 1] >= -(size[1]+2)))
    points = points[mask]

    # Delaunay on perturbed points
    DM = sp.spatial.Delaunay(points)

    # Keep interior triangles (centroids within size box)
    centroids = np.mean(DM.points[DM.simplices], axis=1)
    goods = ((np.abs(centroids[:, 0]) <= size[0]) &
             (np.abs(centroids[:, 1]) <= size[1]))
    DM.all_simplices = DM.simplices
    DM.simplices = DM.simplices[np.where(goods)]

    return DM


# --- Parameters ---
network_size = (20, 20)
etas = np.linspace(0.1, 0.49, 10)
n_trials = 30

results_crystal = {eta: [] for eta in etas}
results_adapted = {eta: [] for eta in etas}

print("=" * 70)
print(f"  Delaunay BEFORE vs AFTER perturbation  ({network_size[0]}x{network_size[1]} network)")
print("=" * 70)

t0 = time.time()

for eta in etas:
    for trial in range(n_trials):
        # --- Method 1: Crystal topology (Delaunay before perturbation) ---
        np.random.seed(1000 * trial + int(eta * 1000))
        DT1 = generate_crystal_then_perturb(network_size, eta)
        D2C.analyze_elastic_struct(DT1)
        results_crystal[eta].append(DT1.PoissonsRatio)

        # --- Method 2: Adapted topology (Delaunay after perturbation) ---
        np.random.seed(1000 * trial + int(eta * 1000))
        DT2 = generate_perturbed_then_delaunay(network_size, eta)
        D2C.analyze_elastic_struct(DT2)
        results_adapted[eta].append(DT2.PoissonsRatio)

    nu_cryst = np.array(results_crystal[eta])
    nu_adapt = np.array(results_adapted[eta])
    print(f"  eta={eta:.3f}: crystal nu={np.mean(nu_cryst):+.4f} +/- {np.std(nu_cryst):.4f}  |  "
          f"adapted nu={np.mean(nu_adapt):+.4f} +/- {np.std(nu_adapt):.4f}")

total_time = time.time() - t0
print(f"\nTotal: {total_time:.1f}s")

# --- Plot ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: mean + std errorbars
ax = axes[0]
eta_arr = np.array(list(etas))

means_c = np.array([np.mean(results_crystal[e]) for e in etas])
stds_c  = np.array([np.std(results_crystal[e])  for e in etas])
means_a = np.array([np.mean(results_adapted[e])  for e in etas])
stds_a  = np.array([np.std(results_adapted[e])   for e in etas])

ax.errorbar(eta_arr - 0.003, means_c, yerr=stds_c, fmt='s-', color='red',
            capsize=4, label='Crystal topology (Python)')
ax.errorbar(eta_arr + 0.003, means_a, yerr=stds_a, fmt='o-', color='blue',
            capsize=4, label='Adapted topology (Mathematica-style)')
ax.set_xlabel('eta')
ax.set_ylabel('Poisson ratio')
ax.set_title(f'Mean +/- Std  ({network_size[0]}x{network_size[1]}, {n_trials} trials)')
ax.set_ylim(-0.5, 0.5)
ax.axhline(y=1/3, color='gray', ls='--', alpha=0.5, label='nu=1/3 (crystal)')
ax.axhline(y=0, color='gray', ls=':', alpha=0.5)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Right: raw scatter
ax = axes[1]
for eta in etas:
    ax.scatter([eta]*n_trials, results_crystal[eta], marker='s', c='red',
               alpha=0.25, s=15, edgecolors='none')
    ax.scatter([eta]*n_trials, results_adapted[eta], marker='o', c='blue',
               alpha=0.25, s=15, edgecolors='none')

ax.scatter([], [], marker='s', c='red', label='Crystal topology')
ax.scatter([], [], marker='o', c='blue', label='Adapted topology')
ax.set_xlabel('eta')
ax.set_ylabel('Poisson ratio')
ax.set_title(f'Raw data  ({network_size[0]}x{network_size[1]}, {n_trials} trials)')
ax.set_ylim(-1.5, 0.5)
ax.axhline(y=1/3, color='gray', ls='--', alpha=0.5)
ax.axhline(y=0, color='gray', ls=':', alpha=0.5)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('benchmarking/poisson_delaunay_comparison.png', dpi=150)
print(f"Saved: benchmarking/poisson_delaunay_comparison.png")
