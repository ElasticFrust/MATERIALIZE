#!/usr/bin/env python3
"""
Ribbon simulation: 4:1 aspect-ratio strip, uniaxial elongation,
measure Poisson ratio from transverse contraction in the middle.

Uses the SAME approach as benchmarking/run_area_fix_comparison.py:
  - 4:1 strip (strip_size = (24, 6))
  - Fix left/right edges in x, pin one y-DOF
  - Apply 1% uniaxial strain along x
  - Relax interior nodes via L-BFGS-B
  - Measure transverse strain in the middle 20% of the strip

Configurations:
  A) Regular lattice, uniform k=1
  B) Regular lattice, VD rigidities (eta=0.15, a=10)
  C) Deformed lattice (eta=0.15), uniform k=1
  D) Deformed lattice + VD rigidities (eta=0.15, a=10)
"""

import sys, os, time, copy
import numpy as np
import scipy as sp
import scipy.optimize

sys.path.insert(0, 'Phase 3')
sys.path.insert(0, 'Phase 2')

import Disc_2_Cont_optimized as D2C
import torch
from forward_solver_torch import ElasticSolver, from_triangulation

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import TwoSlopeNorm, Normalize


# ═══════════════════════════════════════════════════════════════════════════
# Mesh generation
# ═══════════════════════════════════════════════════════════════════════════

def gen_crystal_strip(size, seed=None):
    """Regular triangular lattice on a rectangular strip, Delaunay triangulated."""
    if seed is not None:
        np.random.seed(seed)
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
    return DM


def perturb_points(points, eta, seed=None):
    """Apply random perturbation of magnitude eta to each node."""
    if seed is not None:
        np.random.seed(seed)
    deformed = points.copy()
    thetas = 2 * np.pi * np.random.rand(len(deformed))
    deformed[:, 0] += eta * np.cos(thetas)
    deformed[:, 1] += eta * np.sin(thetas)
    return deformed


# ═══════════════════════════════════════════════════════════════════════════
# Spring network helpers
# ═══════════════════════════════════════════════════════════════════════════

def build_edges(DT):
    """Extract unique edges and rest lengths from triangulation."""
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


def build_edges_with_k(points, simplices, k_per_tri):
    """Build unique edges with per-edge rigidity (averaged for shared edges)."""
    edge_dict = {}
    for ti, sv in enumerate(simplices):
        for ei, (i, j) in enumerate([(sv[0],sv[1]), (sv[0],sv[2]), (sv[1],sv[2])]):
            key = (min(i,j), max(i,j))
            if key not in edge_dict:
                edge_dict[key] = []
            edge_dict[key].append(k_per_tri[ti, ei])
    keys = sorted(edge_dict.keys())
    edges = np.array(keys, dtype=int)
    k_vals = np.array([np.mean(edge_dict[k]) for k in keys])
    rest_lengths = np.linalg.norm(points[edges[:,0]] - points[edges[:,1]], axis=1)
    return edges, rest_lengths, k_vals


def vd_rigidities_for_strip(original_points, deformed_points, simplices, a=10.0):
    """Compute VD rigidities for a strip: k = 1 + tanh(a * (l_def - l0))."""
    edges_tri = np.array([
        [(s[i], s[j]) for i in range(3) for j in range(i+1, 3)]
        for s in simplices
    ])
    node_a = edges_tri[:, :, 0]
    node_b = edges_tri[:, :, 1]
    orig_vecs = original_points[node_a] - original_points[node_b]
    l0 = np.sqrt(np.sum(orig_vecs**2, axis=2))
    def_vecs = deformed_points[node_a] - deformed_points[node_b]
    l_def = np.sqrt(np.sum(def_vecs**2, axis=2))
    return 1.0 + np.tanh(a * (l_def - l0))


# ═══════════════════════════════════════════════════════════════════════════
# Ribbon simulation  (the original approach)
# ═══════════════════════════════════════════════════════════════════════════

def simulate_poisson_ribbon(points, edges, rest_lengths, k, simplices,
                            strain=0.01):
    """
    Elongate a 4:1 strip along x, measure Poisson ratio from transverse
    contraction in the middle.

    Boundary conditions:
      - Left and right edges (within 0.8 of x-boundary): x-DOF fixed
      - One node on left edge: y-DOF pinned (prevent rigid translation)
      - All other DOFs: free

    Returns: (poisson_ratio, final_positions, convergence_info)
    """
    pts0 = points.copy()
    n_all = len(pts0)
    used = np.unique(simplices.ravel())

    x_used = pts0[used, 0]
    x_min, x_max = x_used.min(), x_used.max()
    L_x = x_max - x_min
    tol = 0.8

    # Identify boundary nodes
    is_left = np.zeros(n_all, dtype=bool)
    is_right = np.zeros(n_all, dtype=bool)
    is_left[used] = pts0[used, 0] < x_min + tol
    is_right[used] = pts0[used, 0] > x_max - tol

    # Fixed DOFs: x-component for left/right edges
    fixed_dofs = []
    for i in used:
        if is_left[i] or is_right[i]:
            fixed_dofs.append(2*i)  # fix x

    # Pin one y-DOF on the left edge
    left_verts = used[is_left[used]]
    pin = left_verts[np.argmin(np.abs(pts0[left_verts, 1]))]
    fixed_dofs.append(2*pin + 1)
    fixed_dofs = np.array(sorted(set(fixed_dofs)))

    all_dofs = np.sort(np.concatenate([2*used, 2*used+1]))
    free_dofs = np.setdiff1d(all_dofs, fixed_dofs)

    # Apply affine strain along x
    pos = pts0.copy()
    pos[:, 0] += strain * (pts0[:, 0] - x_min)
    pos_flat = pos.ravel()

    def energy_grad(free_vals):
        full = pos_flat.copy()
        full[free_dofs] = free_vals
        p = full.reshape(-1, 2)
        dr = p[edges[:,0]] - p[edges[:,1]]
        lengths = np.linalg.norm(dr, axis=1)
        lengths_safe = np.maximum(lengths, 1e-15)
        stretch = lengths - rest_lengths
        E = 0.5 * np.sum(k * stretch**2)
        fmag = k * stretch / lengths_safe
        f = fmag[:, None] * dr
        g = np.zeros_like(p)
        np.add.at(g, edges[:,0], f)
        np.add.at(g, edges[:,1], -f)
        return E, g.ravel()[free_dofs]

    x0 = pos_flat[free_dofs]
    res = sp.optimize.minimize(
        energy_grad, x0, jac=True, method='L-BFGS-B',
        options={'maxiter': 50000, 'ftol': 1e-16, 'gtol': 1e-13})

    final_flat = pos_flat.copy()
    final_flat[free_dofs] = res.x
    final_pos = final_flat.reshape(-1, 2)

    # Measure transverse strain in middle 20% of strip
    mid_x = (x_min + x_max) / 2
    mid_w = L_x * 0.1
    mid_mask = ((pts0[used,0] > mid_x - mid_w) & (pts0[used,0] < mid_x + mid_w))
    mid_verts = used[mid_mask]

    if len(mid_verts) < 4:
        return np.nan, final_pos, res

    y0 = pts0[mid_verts, 1]
    yf = final_pos[mid_verts, 1]
    eps_yy = np.dot(y0, yf) / np.dot(y0, y0) - 1.0
    nu = -eps_yy / strain

    return nu, final_pos, res


# ═══════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════

def plot_ribbon(ax, ref_pts, def_pts, edges, k, rest_lengths, title='',
                magnify=20.0):
    """Draw ribbon with displacements magnified and coloured by edge strain."""
    disp = def_pts - ref_pts
    vis_pts = ref_pts + magnify * disp

    dr = def_pts[edges[:,0]] - def_pts[edges[:,1]]
    l_def = np.sqrt(np.sum(dr**2, axis=1))
    strain_vals = (l_def - rest_lengths) / rest_lengths

    vmax = max(np.abs(strain_vals).max(), 1e-8)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    segments = [[vis_pts[e[0]], vis_pts[e[1]]] for e in edges]
    lc = LineCollection(segments, cmap='coolwarm', norm=norm, linewidths=0.5)
    lc.set_array(strain_vals)
    ax.add_collection(lc)
    ax.set_xlim(vis_pts[:,0].min()-1, vis_pts[:,0].max()+1)
    ax.set_ylim(vis_pts[:,1].min()-1, vis_pts[:,1].max()+1)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=8)
    plt.colorbar(lc, ax=ax, shrink=0.5, pad=0.02, label='edge strain')


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    strip_size = (24, 6)   # 4:1 aspect ratio
    eta = 0.15
    a = 10
    strain = 0.01
    seed = 42

    print('=' * 78)
    print('  RIBBON SIMULATION  --  4:1 strip, uniaxial elongation')
    print(f'  Strip size: {strip_size}  |  strain: {strain}  |  eta: {eta}  |  a: {a}')
    print('=' * 78)

    # ── Generate the regular lattice strip ──
    DT_reg = gen_crystal_strip(strip_size, seed=seed)
    pts_reg = DT_reg.points.copy()
    simps = DT_reg.simplices

    # ── Generate deformed points (same topology) ──
    pts_def = perturb_points(pts_reg, eta, seed=seed)

    # ── Compute VD rigidities ──
    k_vd_per_tri = vd_rigidities_for_strip(pts_reg, pts_def, simps, a=a)

    # ── Build edges for each config ──
    edges_u, rl_reg, _ = build_edges_with_k(pts_reg, simps, np.ones((len(simps), 3)))
    _, rl_def, _ = build_edges_with_k(pts_def, simps, np.ones((len(simps), 3)))
    _, _, k_vd_edges = build_edges_with_k(pts_reg, simps, k_vd_per_tri)
    k_ones = np.ones(len(edges_u))

    n_nodes = len(pts_reg)
    n_tri = len(simps)
    n_edges = len(edges_u)
    print(f'  Nodes: {n_nodes}  |  Triangles: {n_tri}  |  Unique edges: {n_edges}')

    # ── Analytical homogenization (for comparison) ──
    edges_tri = np.array([
        [(s[i], s[j]) for i in range(3) for j in range(i+1, 3)]
        for s in simps
    ])
    solver_reg = ElasticSolver(pts_reg, simps, edges_tri)
    solver_def = ElasticSolver(pts_def, simps, edges_tri)
    rigs_vd = torch.tensor(k_vd_per_tri, dtype=torch.float64)
    rigs_one = torch.ones(n_tri, 3, dtype=torch.float64)

    analytical = {}
    for lbl, solver, rigs in [
        ('A', solver_reg, rigs_one),
        ('B', solver_reg, rigs_vd),
        ('C', solver_def, rigs_one),
        ('D', solver_def, rigs_vd),
    ]:
        with torch.no_grad():
            r = solver(rigs)
        analytical[lbl] = r['poisson'].item()

    # ── 4 configurations ──
    configs = [
        ('A) Regular, k=1',       pts_reg, edges_u, rl_reg, k_ones),
        ('B) Regular, VD rigs',   pts_reg, edges_u, rl_reg, k_vd_edges),
        ('C) Deformed, k=1',      pts_def, edges_u, rl_def, k_ones),
        ('D) Deformed + VD rigs', pts_def, edges_u, rl_def, k_vd_edges),
    ]

    results = []
    for label, pts, edges, rl, k in configs:
        t0 = time.time()
        nu, final_pos, res = simulate_poisson_ribbon(
            pts, edges, rl, k, simps, strain=strain)
        dt = time.time() - t0

        key = label[0]
        nu_an = analytical[key]
        err = abs(nu - nu_an) / max(abs(nu_an), 1e-15)

        print(f'\n  {label}')
        print(f'    nu (ribbon sim) = {nu:.6f}')
        print(f'    nu (analytical) = {nu_an:.6f}')
        print(f'    error           = {err:.1%}')
        print(f'    converged: {res.success}  |  E_final: {res.fun:.6e}  |  {dt:.2f}s')

        results.append(dict(
            label=label, key=key, nu_sim=nu, nu_an=nu_an, err=err,
            points=pts, edges=edges, rl=rl, k=k,
            final_pos=final_pos, dt=dt,
        ))

    # ── Summary ──
    print('\n' + '=' * 78)
    print('  RIBBON SIMULATION SUMMARY')
    print('=' * 78)
    print(f'  {"Config":<28} {"nu(ribbon)":>10} {"nu(analyt)":>10} {"error":>7}')
    print(f'  {"─" * 58}')
    for r in results:
        print(f'  {r["label"]:<28} {r["nu_sim"]:10.5f} {r["nu_an"]:10.5f} {r["err"]:6.1%}')
    print('=' * 78)

    # ── Figure ──
    fig, axes = plt.subplots(4, 1, figsize=(16, 14))
    fig.suptitle(
        f'Ribbon Simulation: 4:1 strip, strain={strain}, eta={eta}\n'
        f'Displacements magnified x20, coloured by edge strain',
        fontsize=12, fontweight='bold')

    for i, r in enumerate(results):
        plot_ribbon(axes[i], r['points'], r['final_pos'],
                    r['edges'], r['k'], r['rl'],
                    title=f'{r["label"]}   nu_sim={r["nu_sim"]:.4f}   nu_ana={r["nu_an"]:.4f}',
                    magnify=20.0)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    outpath = 'Phase 3/ribbon_simulation.png'
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'\n  Figure saved -> {outpath}')

    # ── Append to results_summary.txt ──
    with open('Phase 3/results_summary.txt', 'a') as f:
        f.write('\n\n')
        f.write('='*80 + '\n')
        f.write('5.  RIBBON SIMULATION  (4:1 strip, transverse contraction in middle)\n')
        f.write('='*80 + '\n')
        f.write(f'\n  Strip size: {strip_size}  |  strain: {strain}  |  eta: {eta}  |  a: {a}\n')
        f.write(f'  Nodes: {n_nodes}  |  Triangles: {n_tri}  |  Edges: {n_edges}\n')
        f.write(f'\n  Method: Fix left/right edges in x, pin one y-DOF.\n')
        f.write(f'  Apply {strain*100:.0f}% uniaxial strain along x, relax interior (L-BFGS-B).\n')
        f.write(f'  Measure eps_yy in middle 20% of strip -> nu = -eps_yy/eps_xx.\n')
        f.write(f'\n  {"Config":<28} {"nu(ribbon)":>10} {"nu(analyt)":>10} {"error":>7}\n')
        f.write(f'  {"─"*58}\n')
        for r in results:
            f.write(f'  {r["label"]:<28} {r["nu_sim"]:10.5f} {r["nu_an"]:10.5f} {r["err"]:6.1%}\n')
        f.write('\n  Key findings:\n')
        f.write(f'  - Config A (baseline): nu = {results[0]["nu_sim"]:.4f} vs analytical 0.3333\n')
        f.write(f'  - Config D (both effects): nu = {results[3]["nu_sim"]:.4f} vs analytical {results[3]["nu_an"]:.4f}\n')
        f.write(f'  - The ribbon strip simulation directly shows the transverse contraction\n')
        f.write(f'    (or lack thereof for low-nu configs) in a physically intuitive setup.\n')

    print('  --> Appended to Phase 3/results_summary.txt')


if __name__ == '__main__':
    main()
