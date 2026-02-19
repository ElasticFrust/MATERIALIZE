"""
Comparison: simulation vs D2C (pre-fix 1/16) vs D2C (post-fix area-normalized).

All three methods run on the SAME mesh for each trial/eta.
Plots all results on a single figure.
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
# Mesh generation (same as run_sim_vs_d2c.py)
# =========================================================================

def gen_crystal_then_perturb(size, eta):
    v1 = np.array([1, 0])
    v2 = np.array([0.5, np.sqrt(3)/2])
    MaxPos = int(round(2 * (max(size) / min([np.sqrt(3)/2, 1]))))
    points = []
    for n in range(-MaxPos, MaxPos):
        for m in range(-MaxPos, MaxPos):
            points.append(n*v1 + m*v2)
    points = np.array(points)
    mask = ((points[:,0] <= size[0]+2) & (points[:,0] >= -(size[0]+2)) &
            (points[:,1] <= size[1]+2) & (points[:,1] >= -(size[1]+2)))
    points = points[mask]
    DM = sp.spatial.Delaunay(points)
    centroids = np.mean(DM.points[DM.simplices], axis=1)
    goods = (np.abs(centroids[:,0]) <= size[0]) & (np.abs(centroids[:,1]) <= size[1])
    DM.all_simplices = DM.simplices
    DM.simplices = DM.simplices[np.where(goods)]
    for i in range(len(DM.points)):
        theta = 2 * np.pi * np.random.rand()
        DM.points[i] += eta * np.array([np.cos(theta), np.sin(theta)])
    return DM


def gen_perturbed_then_delaunay(size, eta):
    v1 = np.array([1, 0])
    v2 = np.array([0.5, np.sqrt(3)/2])
    MaxPos = int(round(2 * (max(size) / min([np.sqrt(3)/2, 1]))))
    points = []
    for n in range(-MaxPos, MaxPos):
        for m in range(-MaxPos, MaxPos):
            theta = 2 * np.pi * np.random.rand()
            points.append(n*v1 + m*v2 + eta * np.array([np.cos(theta), np.sin(theta)]))
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
    pts0 = DT.points.copy()
    edges, rest_lengths = build_edges(DT)
    n_all = len(pts0)
    used = np.unique(DT.simplices.ravel())
    x_used = pts0[used, 0]
    x_min, x_max = x_used.min(), x_used.max()
    L_x = x_max - x_min
    tol = 0.8
    is_left = np.zeros(n_all, dtype=bool)
    is_right = np.zeros(n_all, dtype=bool)
    is_left[used] = pts0[used, 0] < x_min + tol
    is_right[used] = pts0[used, 0] > x_max - tol
    fixed_dofs = []
    for i in used:
        if is_left[i] or is_right[i]:
            fixed_dofs.append(2*i)
    left_verts = used[is_left[used]]
    pin = left_verts[np.argmin(np.abs(pts0[left_verts, 1]))]
    fixed_dofs.append(2*pin + 1)
    fixed_dofs = np.array(sorted(set(fixed_dofs)))
    all_dofs = np.sort(np.concatenate([2*used, 2*used+1]))
    free_dofs = np.setdiff1d(all_dofs, fixed_dofs)
    pos = pts0.copy()
    pos[:, 0] += strain * (pts0[:, 0] - x_min)
    pos_flat = pos.ravel()

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
        options={'maxiter': 50000, 'ftol': 1e-16, 'gtol': 1e-13})
    final_flat = pos_flat.copy()
    final_flat[free_dofs] = res.x
    final_pos = final_flat.reshape(-1, 2)
    mid_x = (x_min + x_max) / 2
    mid_w = L_x * 0.1
    mid_mask = ((pts0[used,0] > mid_x - mid_w) & (pts0[used,0] < mid_x + mid_w))
    mid_verts = used[mid_mask]
    if len(mid_verts) < 4:
        return np.nan
    y0 = pts0[mid_verts, 1]
    yf = final_pos[mid_verts, 1]
    eps_yy = np.dot(y0, yf) / np.dot(y0, y0) - 1.0
    return -eps_yy / strain


# =========================================================================
# OLD D2C (pre-fix): uses 1/16 and uniform 1/N weights
# =========================================================================

def d2c_old_prefix(triangulation):
    """D2C with the OLD 1/16 normalization and uniform coupling."""
    D2C.add_edges_to_triangulation(triangulation)

    # --- bare tensors with 1/16 ---
    N = len(triangulation.simplices)
    edges = triangulation.edges
    positions = triangulation.points
    node_a = edges[:, :, 0]
    node_b = edges[:, :, 1]
    vecs = positions[node_a] - positions[node_b]
    vx = vecs[:, :, 0]
    vy = vecs[:, :, 1]
    rigs = np.ones((N, 3))
    for i, r in enumerate(triangulation.rigidities):
        if len(r) > 0:
            rigs[i] = r
    length2 = np.sum(vecs**2, axis=2)
    for i, rl in enumerate(triangulation.rest_lenghts):
        if len(rl) > 0:
            length2[i] = np.array(rl)**2

    factor = rigs / length2 / 16.0  # OLD: fixed 1/16
    bare = np.column_stack([
        np.sum(factor * vx**4,         axis=1),
        np.sum(factor * vx**3 * vy,    axis=1),
        np.sum(factor * vx**2 * vy**2, axis=1),
        np.sum(factor * vx * vy**3,    axis=1),
        np.sum(factor * vy**4,         axis=1),
    ])
    triangulation.BareElasticTensor = bare

    # --- delta with simple mean ---
    mean_tensor = np.mean(bare, axis=0)  # OLD: simple mean
    delta = bare - mean_tensor
    triangulation.delta_tensor = delta

    A_blocks = D2C._batch_to_9x9(bare)
    B_blocks = D2C._batch_to_9x9(delta)
    dA_vecs  = D2C._batch_to_9vec(delta)

    # --- Woodbury with uniform 1/N weights ---
    triangulation.Ws = D2C._woodbury_solve(
        A_blocks, B_blocks, dA_vecs, area_weights=None)  # None => 1/N

    # --- actual elastic tensor ---
    actual = D2C._compute_actual_elastic_tensor_vectorized(bare, triangulation.Ws)
    triangulation.ActualElasticTensor = actual

    # --- simple mean for total ---
    C = np.mean(actual, axis=0)  # OLD: simple mean
    triangulation.totalElasticTensor = C
    triangulation.PoissonsRatio = (C[2]*C[3] - C[1]*C[4]) / (C[0]*C[3] - C[1]**2)


# =========================================================================
# Main comparison
# =========================================================================

strip_size = (20, 6)
square_half = 6.0
etas = np.linspace(0.0, 0.49, 20)
n_trials = 10
strain = 0.01

print("=" * 78)
print("  Sim vs D2C (old 1/16) vs D2C (area-fixed)  —  crystal topology only")
print("=" * 78)
print(f"  Strip: {strip_size},  D2C square: [-{square_half},{square_half}]^2")
print(f"  Trials: {n_trials}, eta points: {len(etas)}")
print()

keys = ['sim_crystal', 'sim_adapted',
        'd2c_old_crystal',
        'd2c_new_crystal']
results = {k: {e: [] for e in etas} for k in keys}

t_total = time.time()

for eta in etas:
    t0 = time.time()
    for trial in range(n_trials):
        seed = 1000*trial + int(eta*1000)

        # --- Crystal topology ---
        np.random.seed(seed)
        DT = gen_crystal_then_perturb(strip_size, eta)
        results['sim_crystal'][eta].append(simulate_poisson(DT, strain))

        sub_old = extract_square_region(DT, square_half)
        d2c_old_prefix(sub_old)
        results['d2c_old_crystal'][eta].append(sub_old.PoissonsRatio)

        sub_new = extract_square_region(DT, square_half)
        D2C.analyze_elastic_struct(sub_new)
        results['d2c_new_crystal'][eta].append(sub_new.PoissonsRatio)

        # --- Adapted topology (simulation only, for reference) ---
        np.random.seed(seed)
        DT = gen_perturbed_then_delaunay(strip_size, eta)
        results['sim_adapted'][eta].append(simulate_poisson(DT, strain))

    dt = time.time() - t0
    sc = np.mean(results['sim_crystal'][eta])
    sa = np.mean(results['sim_adapted'][eta])
    oc = np.mean(results['d2c_old_crystal'][eta])
    nc = np.mean(results['d2c_new_crystal'][eta])
    print(f"  eta={eta:.3f}  |  sim_cryst={sc:+.3f}  sim_adapt={sa:+.3f}  "
          f"old_d2c={oc:+.3f}  new_d2c={nc:+.3f}  ({dt:.1f}s)")

print(f"\nTotal time: {time.time()-t_total:.1f}s")


# =========================================================================
# Plot — single figure, 4 curves
# =========================================================================

eta_arr = np.array(list(etas))
dx = 0.003  # horizontal offset to avoid overlapping error bars

fig, ax = plt.subplots(figsize=(12, 7))

for label, key, color, marker, ls, offset in [
    ('Simulation (crystal)',       'sim_crystal',      'C3', 'o', '-',  -dx),
    ('Simulation (adapted)',       'sim_adapted',      'C1', 'o', '-',  0),
    ('D2C old 1/16 (crystal)',     'd2c_old_crystal',  'C0', 's', '--', dx),
    ('D2C area-fixed (crystal)',   'd2c_new_crystal',  'C0', 'D', '-',  2*dx),
]:
    means = np.array([np.mean(results[key][e]) for e in etas])
    stds  = np.array([np.std(results[key][e])  for e in etas])
    ax.errorbar(eta_arr + offset, means, yerr=stds, fmt=marker+ls,
                color=color, capsize=3, markersize=5, linewidth=1.5,
                label=label)

ax.axhline(y=1/3, color='gray', ls=':', alpha=0.5, label=r'$\nu=1/3$')
ax.axhline(y=0,   color='gray', ls='--', alpha=0.3)
ax.set_xlabel(r'$\eta$', fontsize=14)
ax.set_ylabel(r"Poisson's ratio $\nu$", fontsize=14)
ax.set_title('Crystal topology: Simulation vs D2C (old 1/16 vs area-fixed)', fontsize=14)
ax.legend(fontsize=11, loc='lower left')
ax.grid(True, alpha=0.3)
ax.set_ylim(-1.5, 0.5)

fig.tight_layout()
outpath = os.path.join(os.path.dirname(__file__), 'area_fix_comparison.png')
fig.savefig(outpath, dpi=150, bbox_inches='tight')
print(f"\nSaved: {outpath}")
