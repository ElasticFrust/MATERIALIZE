"""
Ground-truth comparison: spring simulation vs D2C analytical.

Key change from v1: D2C is computed on a SQUARE sub-region cut from the
exact same strip triangulation used by the simulation. This ensures
identical mesh, vertices, edges, and perturbation pattern.

Strip test: apply uniaxial strain, minimize elastic energy (L-BFGS-B),
measure Poisson ratio from transverse contraction in the middle.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import scipy as sp
import scipy.optimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import Disc_2_Cont_optimized as D2C
import copy


# =========================================================================
# Mesh generation
# =========================================================================

def gen_crystal_then_perturb(size, eta):
    """Pre-perturbation: Delaunay on crystal lattice, THEN perturb positions."""
    v1 = np.array([1, 0])
    v2 = np.array([0.5, np.sqrt(3)/2])
    MaxPos = int(round(2 * (max(size) / min([np.sqrt(3)/2, 1]))))

    points = []
    for n in range(-MaxPos, MaxPos):
        for m in range(-MaxPos, MaxPos):
            pt = n*v1 + m*v2
            points.append(pt)
    points = np.array(points)
    mask = ((points[:,0] <= size[0]+2) & (points[:,0] >= -(size[0]+2)) &
            (points[:,1] <= size[1]+2) & (points[:,1] >= -(size[1]+2)))
    points = points[mask]

    DM = sp.spatial.Delaunay(points)
    centroids = np.mean(DM.points[DM.simplices], axis=1)
    goods = (np.abs(centroids[:,0]) <= size[0]) & (np.abs(centroids[:,1]) <= size[1])
    DM.all_simplices = DM.simplices
    DM.simplices = DM.simplices[np.where(goods)]

    # Perturb AFTER triangulation
    for i in range(len(DM.points)):
        theta = 2 * np.pi * np.random.rand()
        DM.points[i] += eta * np.array([np.cos(theta), np.sin(theta)])
    return DM


def gen_perturbed_then_delaunay(size, eta):
    """Post-perturbation: perturb positions FIRST, then Delaunay."""
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
    mask = ((points[:,0] <= size[0]+2) & (points[:,0] >= -(size[0]+2)) &
            (points[:,1] <= size[1]+2) & (points[:,1] >= -(size[1]+2)))
    points = points[mask]

    DM = sp.spatial.Delaunay(points)
    centroids = np.mean(DM.points[DM.simplices], axis=1)
    goods = (np.abs(centroids[:,0]) <= size[0]) & (np.abs(centroids[:,1]) <= size[1])
    DM.all_simplices = DM.simplices
    DM.simplices = DM.simplices[np.where(goods)]
    return DM


def extract_square_region(DT, half_side):
    """Extract a square sub-region from a triangulation for D2C analysis.

    Returns a new triangulation-like object containing only simplices
    whose centroids fall within [-half_side, half_side]^2.
    Same points array (shared vertices), just fewer simplices.
    """
    centroids = np.mean(DT.points[DT.simplices], axis=1)
    in_square = ((np.abs(centroids[:,0]) <= half_side) &
                 (np.abs(centroids[:,1]) <= half_side))

    sub = copy.copy(DT)
    sub.all_simplices = DT.simplices
    sub.simplices = DT.simplices[np.where(in_square)]
    return sub


# =========================================================================
# Spring network simulation
# =========================================================================

def build_edges(DT):
    """Extract unique edges and rest lengths from trimmed simplices."""
    edge_set = set()
    for tri in DT.simplices:
        for i in range(3):
            for j in range(i+1, 3):
                a, b = int(tri[i]), int(tri[j])
                edge_set.add((min(a,b), max(a,b)))
    edges = np.array(sorted(edge_set))
    rest_lengths = np.linalg.norm(
        DT.points[edges[:,0]] - DT.points[edges[:,1]], axis=1)
    return edges, rest_lengths


def simulate_poisson(DT, strain=0.01):
    """
    Uniaxial strip test.
    - Fix x-coords at left/right boundaries, apply strain to right.
    - Minimize elastic energy E = sum k/2 (|r_ij| - l0)^2.
    - Measure transverse strain in the middle via linear regression.
    """
    pts0 = DT.points.copy()
    edges, rest_lengths = build_edges(DT)
    n_all = len(pts0)

    # Vertices actually in the mesh
    used = np.unique(DT.simplices.ravel())
    x_used = pts0[used, 0]
    x_min, x_max = x_used.min(), x_used.max()
    L_x = x_max - x_min

    # Boundary: vertices within 0.8 of the left/right edges
    tol = 0.8
    is_left = np.zeros(n_all, dtype=bool)
    is_right = np.zeros(n_all, dtype=bool)
    is_left[used] = pts0[used, 0] < x_min + tol
    is_right[used] = pts0[used, 0] > x_max - tol

    # Fixed DOFs: x of boundary vertices + y of one vertex to kill translation
    fixed_dofs = []
    for i in used:
        if is_left[i] or is_right[i]:
            fixed_dofs.append(2*i)      # fix x
    left_verts = used[is_left[used]]
    pin = left_verts[np.argmin(np.abs(pts0[left_verts, 1]))]
    fixed_dofs.append(2*pin + 1)        # fix y of one vertex
    fixed_dofs = np.array(sorted(set(fixed_dofs)))

    all_dofs = np.sort(np.concatenate([2*used, 2*used+1]))
    free_dofs = np.setdiff1d(all_dofs, fixed_dofs)

    # Apply affine strain as initial guess: x -> x + strain*(x - x_min)
    pos = pts0.copy()
    pos[:, 0] += strain * (pts0[:, 0] - x_min)

    pos_flat = pos.ravel()
    fixed_vals = pos_flat[fixed_dofs]

    def energy_grad(free_vals):
        full = pos_flat.copy()
        full[free_dofs] = free_vals
        p = full.reshape(-1, 2)

        dr = p[edges[:,0]] - p[edges[:,1]]
        lengths = np.linalg.norm(dr, axis=1)
        stretch = lengths - rest_lengths

        E = 0.5 * np.sum(stretch**2)

        fmag = stretch / lengths
        f = fmag[:, None] * dr
        g = np.zeros_like(p)
        np.add.at(g, edges[:,0], f)
        np.add.at(g, edges[:,1], -f)

        return E, g.ravel()[free_dofs]

    x0 = pos_flat[free_dofs]
    res = scipy.optimize.minimize(
        energy_grad, x0, jac=True, method='L-BFGS-B',
        options={'maxiter': 50000, 'ftol': 1e-16, 'gtol': 1e-13}
    )

    # Reconstruct final positions
    final_flat = pos_flat.copy()
    final_flat[free_dofs] = res.x
    final_pos = final_flat.reshape(-1, 2)

    # Measure transverse strain in middle 20% of strip
    mid_x = (x_min + x_max) / 2
    mid_w = L_x * 0.1
    mid_mask = ((pts0[used,0] > mid_x - mid_w) &
                (pts0[used,0] < mid_x + mid_w))
    mid_verts = used[mid_mask]

    if len(mid_verts) < 4:
        return np.nan

    # Linear regression: y_final = (1 + eps_yy) * y_initial
    y0 = pts0[mid_verts, 1]
    yf = final_pos[mid_verts, 1]
    eps_yy = np.dot(y0, yf) / np.dot(y0, y0) - 1.0

    return -eps_yy / strain


# =========================================================================
# Main comparison
# =========================================================================

strip_size = (20, 6)     # long strip for simulation
square_half = 6.0        # D2C square = [-6,6]^2 cut from the strip center
etas = np.linspace(0.1, 0.49, 8)
n_trials = 10
strain = 0.01

print("=" * 70)
print("  Spring simulation vs D2C (same mesh, square sub-region)")
print("=" * 70)
print(f"  Strip: {strip_size},  D2C square: [{-square_half},{square_half}]^2")
print(f"  Trials: {n_trials}, strain={strain}")
print()

results = {k: {e: [] for e in etas}
           for k in ['sim_crystal','sim_adapted','d2c_crystal','d2c_adapted']}

t_total = time.time()

for eta in etas:
    t0 = time.time()

    for trial in range(n_trials):
        seed = 1000*trial + int(eta*1000)

        # --- Crystal topology ---
        np.random.seed(seed)
        DT = gen_crystal_then_perturb(strip_size, eta)
        results['sim_crystal'][eta].append(simulate_poisson(DT, strain))

        sub = extract_square_region(DT, square_half)
        D2C.analyze_elastic_struct(sub)
        results['d2c_crystal'][eta].append(sub.PoissonsRatio)

        # --- Adapted topology ---
        np.random.seed(seed)
        DT = gen_perturbed_then_delaunay(strip_size, eta)
        results['sim_adapted'][eta].append(simulate_poisson(DT, strain))

        sub = extract_square_region(DT, square_half)
        D2C.analyze_elastic_struct(sub)
        results['d2c_adapted'][eta].append(sub.PoissonsRatio)

    sc = np.mean(results['sim_crystal'][eta])
    sa = np.mean(results['sim_adapted'][eta])
    dc = np.mean(results['d2c_crystal'][eta])
    da = np.mean(results['d2c_adapted'][eta])
    dt = time.time() - t0
    print(f"  eta={eta:.3f}:  sim_cryst={sc:+.4f}  sim_adapt={sa:+.4f}  "
          f"d2c_cryst={dc:+.4f}  d2c_adapt={da:+.4f}   ({dt:.1f}s)")

print(f"\nTotal time: {time.time()-t_total:.1f}s")

# =========================================================================
# Plot
# =========================================================================
eta_arr = np.array(list(etas))

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, label, sk, dk in [
    (axes[0], 'Crystal topology (Delaunay before perturbation)',
     'sim_crystal', 'd2c_crystal'),
    (axes[1], 'Adapted topology (Delaunay after perturbation)',
     'sim_adapted', 'd2c_adapted'),
]:
    sim_mean = np.array([np.mean(results[sk][e]) for e in etas])
    sim_std  = np.array([np.std(results[sk][e])  for e in etas])
    d2c_mean = np.array([np.mean(results[dk][e]) for e in etas])
    d2c_std  = np.array([np.std(results[dk][e])  for e in etas])

    ax.errorbar(eta_arr, sim_mean, yerr=sim_std, fmt='o-', color='red',
                capsize=4, label='Simulation (strip)')
    ax.errorbar(eta_arr + 0.003, d2c_mean, yerr=d2c_std, fmt='s--', color='blue',
                capsize=4, label='D2C (square from same mesh)')
    ax.set_xlabel('eta')
    ax.set_ylabel('Poisson ratio')
    ax.set_title(label)
    ax.axhline(y=1/3, color='gray', ls=':', alpha=0.5, label='nu=1/3')
    ax.axhline(y=0, color='gray', ls='--', alpha=0.3)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.5, 0.5)

plt.tight_layout()
plt.savefig('benchmarking/sim_vs_d2c_comparison.png', dpi=150)
print(f"\nSaved: benchmarking/sim_vs_d2c_comparison.png")
