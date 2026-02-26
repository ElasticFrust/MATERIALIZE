#!/usr/bin/env python3
"""
Mechanical simulation: spring elastic-energy minimization.

For each of the 4 network configurations (A-D), applies macroscopic strains
to the spring network, minimises total elastic energy with boundary nodes
affinely constrained (KUBC), and extracts effective elastic moduli.

Configurations
--------------
  A) Regular lattice, uniform k=1        (baseline)
  B) Regular lattice, VD rigidities      (eta=0.15, a=10)
  C) Deformed geometry (eta=0.15), k=1
  D) Deformed geometry + VD rigidities   (eta=0.15, a=10)

Method
------
- Kinematic Uniform Boundary Conditions (KUBC): boundary nodes displaced
  affinely, interior nodes relax via L-BFGS-B minimization of total spring
  energy  E = Sum 1/2 k (|ri-rj| - l0)^2.
- Full 3x3 Voigt stiffness extracted from 6 independent strain tests.
- Poisson ratio extracted from compliance matrix (normalization-independent).
- Direct measurement of lateral contraction under uniaxial strain.
"""

import sys, time
import numpy as np
from scipy.optimize import minimize
import torch

sys.path.insert(0, 'Phase 3')
sys.path.insert(0, 'Phase 2')

import Disc_2_Cont_optimized as D2C
from forward_solver_torch import ElasticSolver, from_triangulation
from sweep_utils import virtual_distortion_rigidities

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, TwoSlopeNorm


# ═══════════════════════════════════════════════════════════════════════════
# Mesh helpers
# ═══════════════════════════════════════════════════════════════════════════

def build_unique_edges(simplices, rigidities_per_tri):
    """Extract unique edges with averaged spring constants."""
    edge_dict = {}
    for ti, sv in enumerate(simplices):
        for ei, (i, j) in enumerate([(sv[0], sv[1]),
                                      (sv[0], sv[2]),
                                      (sv[1], sv[2])]):
            key = (min(i, j), max(i, j))
            if key not in edge_dict:
                edge_dict[key] = []
            edge_dict[key].append(rigidities_per_tri[ti, ei])

    keys = sorted(edge_dict.keys())
    edges = np.array(keys, dtype=int)
    k_vals = np.array([np.mean(edge_dict[k]) for k in keys])
    return edges, k_vals


def compute_rest_lengths(points, edges):
    """Rest length = distance in reference config."""
    dr = points[edges[:, 0]] - points[edges[:, 1]]
    return np.sqrt(np.sum(dr ** 2, axis=1))


def classify_boundary(points, margin_frac=0.10):
    """Nodes within margin_frac of the bounding-box edge -> boundary."""
    lo = points.min(axis=0)
    hi = points.max(axis=0)
    m = (hi - lo) * margin_frac
    return (
        (points[:, 0] < lo[0] + m[0]) | (points[:, 0] > hi[0] - m[0]) |
        (points[:, 1] < lo[1] + m[1]) | (points[:, 1] > hi[1] - m[1])
    )


def mesh_area(points, simplices):
    """Total area of the triangulation."""
    p = points[simplices]
    return 0.5 * np.abs(
        (p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1]) -
        (p[:, 2, 0] - p[:, 0, 0]) * (p[:, 1, 1] - p[:, 0, 1])
    ).sum()


# ═══════════════════════════════════════════════════════════════════════════
# Energy minimisation
# ═══════════════════════════════════════════════════════════════════════════

def energy_and_grad(free_flat, fixed_pos, free_idx, fixed_idx,
                    edges, k, rest_lengths, n_nodes):
    """Spring energy and analytical gradient over free DOFs."""
    pos = np.empty((n_nodes, 2))
    pos[fixed_idx] = fixed_pos
    pos[free_idx] = free_flat.reshape(-1, 2)

    dr = pos[edges[:, 0]] - pos[edges[:, 1]]
    lengths = np.sqrt(np.sum(dr ** 2, axis=1))
    lengths_safe = np.maximum(lengths, 1e-15)
    ext = lengths - rest_lengths

    energy = 0.5 * np.dot(k, ext ** 2)

    fac = k * ext / lengths_safe
    forces = np.zeros((n_nodes, 2))
    fdr = fac[:, None] * dr
    np.add.at(forces, edges[:, 0], fdr)
    np.add.at(forces, edges[:, 1], -fdr)

    return energy, forces[free_idx].ravel().copy()


def simulate_strain(points, edges, k, rest_lengths, boundary_mask,
                    strain_tensor, tol=1e-12):
    """Apply affine strain to boundary, relax interior, return energy + positions."""
    n_nodes = len(points)
    displacement = points @ strain_tensor.T
    deformed = points + displacement

    free_idx = np.where(~boundary_mask)[0]
    fixed_idx = np.where(boundary_mask)[0]
    fixed_pos = deformed[fixed_idx]

    x0 = deformed[free_idx].ravel()

    result = minimize(
        energy_and_grad, x0,
        args=(fixed_pos, free_idx, fixed_idx, edges, k, rest_lengths, n_nodes),
        method='L-BFGS-B', jac=True,
        options={'maxiter': 10000, 'ftol': 1e-15, 'gtol': tol},
    )

    final_pos = np.empty((n_nodes, 2))
    final_pos[fixed_idx] = fixed_pos
    final_pos[free_idx] = result.x.reshape(-1, 2)

    dr = final_pos[edges[:, 0]] - final_pos[edges[:, 1]]
    lengths = np.sqrt(np.sum(dr ** 2, axis=1))
    ext = lengths - rest_lengths
    energy = 0.5 * np.dot(k, ext ** 2)

    return energy, final_pos, result


# ═══════════════════════════════════════════════════════════════════════════
# Elastic-tensor extraction (6 energy measurements -> full Voigt 3x3)
# ═══════════════════════════════════════════════════════════════════════════

def extract_elastic_tensor(points, simplices, edges, k, rest_lengths,
                           boundary_mask, delta=0.005):
    """Return (C_voigt_3x3, energies_dict, mesh_area, time, deformed_positions)."""
    A = mesh_area(points, simplices)

    strain_states = {
        'xx':    np.array([[1, 0], [0, 0]]),
        'yy':    np.array([[0, 0], [0, 1]]),
        'xy':    np.array([[0, .5], [.5, 0]]),
        'xx+yy': np.array([[1, 0], [0, 1]]),
        'xx+xy': np.array([[1, .5], [.5, 0]]),
        'yy+xy': np.array([[0, .5], [.5, 1]]),
    }

    energies = {}
    deformed_pos = {}
    t0 = time.time()
    for name, eps0 in strain_states.items():
        eps = delta * eps0
        E_val, final_pos, res = simulate_strain(points, edges, k, rest_lengths,
                                                boundary_mask, eps)
        energies[name] = E_val
        deformed_pos[name] = final_pos
    dt = time.time() - t0

    d2 = delta ** 2
    C = np.zeros((3, 3))
    C[0, 0] = 2 * energies['xx'] / (A * d2)
    C[1, 1] = 2 * energies['yy'] / (A * d2)
    C[2, 2] = 2 * energies['xy'] / (A * d2)
    C[0, 1] = C[1, 0] = (2 * energies['xx+yy'] / (A * d2) - C[0, 0] - C[1, 1]) / 2
    C[0, 2] = C[2, 0] = (2 * energies['xx+xy'] / (A * d2) - C[0, 0] - C[2, 2]) / 2
    C[1, 2] = C[2, 1] = (2 * energies['yy+xy'] / (A * d2) - C[1, 1] - C[2, 2]) / 2

    return C, energies, A, dt, deformed_pos


def per_edge_strain_energy(points, edges, k, rest_lengths, def_points):
    """Compute per-edge elastic energy 0.5 k (l - l0)^2."""
    dr = def_points[edges[:, 0]] - def_points[edges[:, 1]]
    l_cur = np.sqrt(np.sum(dr ** 2, axis=1))
    return 0.5 * k * (l_cur - rest_lengths) ** 2


# ═══════════════════════════════════════════════════════════════════════════
# Build the 4 configurations
# ═══════════════════════════════════════════════════════════════════════════

def build_configs(size=(14, 14), eta=0.15, a=10, seed=42):
    """Return list of dicts, one per configuration."""
    vd = virtual_distortion_rigidities(size=size, eta=eta, a=a, seed=seed)
    tri_reg = vd['tri']
    pts_reg = tri_reg.points
    simps = tri_reg.simplices
    pts_def = vd['deformed_points']

    edges_uniq, _ = build_unique_edges(simps, np.ones((len(simps), 3)))
    _, k_vd = build_unique_edges(simps, vd['rigidities_np'])
    k_uniform = np.ones(len(edges_uniq))

    edges_tri = np.array([
        [(s[i], s[j]) for i in range(3) for j in range(i + 1, 3)]
        for s in simps
    ])
    solver_def = ElasticSolver(pts_def, simps, edges_tri)

    configs = [
        dict(label='A) Regular, k=1',
             points=pts_reg, simplices=simps, edges=edges_uniq,
             k=k_uniform.copy(),
             rest_lengths=compute_rest_lengths(pts_reg, edges_uniq),
             solver=vd['solver'],
             rigs_torch=torch.ones(len(simps), 3, dtype=torch.float64)),
        dict(label='B) Regular, VD rigs',
             points=pts_reg, simplices=simps, edges=edges_uniq,
             k=k_vd.copy(),
             rest_lengths=compute_rest_lengths(pts_reg, edges_uniq),
             solver=vd['solver'],
             rigs_torch=vd['rigidities']),
        dict(label='C) Deformed, k=1',
             points=pts_def, simplices=simps, edges=edges_uniq,
             k=k_uniform.copy(),
             rest_lengths=compute_rest_lengths(pts_def, edges_uniq),
             solver=solver_def,
             rigs_torch=torch.ones(len(simps), 3, dtype=torch.float64)),
        dict(label='D) Deformed + VD rigs',
             points=pts_def, simplices=simps, edges=edges_uniq,
             k=k_vd.copy(),
             rest_lengths=compute_rest_lengths(pts_def, edges_uniq),
             solver=solver_def,
             rigs_torch=vd['rigidities']),
    ]
    return configs


# ═══════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════

def plot_deformed_mesh(ax, ref_points, def_points, edges, k, rest_lengths,
                       title='', magnify=10.0):
    """Draw edges coloured by strain, with displacements magnified."""
    disp = def_points - ref_points
    vis_pts = ref_points + magnify * disp

    dr_def = def_points[edges[:, 0]] - def_points[edges[:, 1]]
    l_def = np.sqrt(np.sum(dr_def ** 2, axis=1))
    strain = (l_def - rest_lengths) / rest_lengths

    vmax = max(np.abs(strain).max(), 1e-8)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    segments = [[vis_pts[e[0]], vis_pts[e[1]]] for e in edges]
    lc = LineCollection(segments, cmap='coolwarm', norm=norm, linewidths=0.6)
    lc.set_array(strain)
    ax.add_collection(lc)
    ax.set_xlim(vis_pts[:, 0].min() - 1, vis_pts[:, 0].max() + 1)
    ax.set_ylim(vis_pts[:, 1].min() - 1, vis_pts[:, 1].max() + 1)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=7)
    plt.colorbar(lc, ax=ax, shrink=0.65, pad=0.02, label='edge strain')


def plot_reference_mesh(ax, points, edges, k, title=''):
    """Draw undeformed mesh, edges coloured by log-rigidity."""
    k_safe = np.maximum(k, 1e-10)
    logk = np.log10(k_safe)
    norm = Normalize(vmin=logk.min() - 0.1, vmax=logk.max() + 0.1)

    segments = [[points[e[0]], points[e[1]]] for e in edges]
    lws = 0.3 + 1.7 * (k_safe / k_safe.max())
    lc = LineCollection(segments, cmap='inferno', norm=norm, linewidths=lws)
    lc.set_array(logk)
    ax.add_collection(lc)
    ax.set_xlim(points[:, 0].min() - 1, points[:, 0].max() + 1)
    ax.set_ylim(points[:, 1].min() - 1, points[:, 1].max() + 1)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=7)
    plt.colorbar(lc, ax=ax, shrink=0.65, pad=0.02, label='log10(k)')


def plot_energy_map(ax, ref_points, def_points, edges, k, rest_lengths,
                    title='', magnify=50.0):
    """Draw edges coloured by per-edge elastic energy, displacements magnified."""
    dr = def_points[edges[:, 0]] - def_points[edges[:, 1]]
    l_cur = np.sqrt(np.sum(dr ** 2, axis=1))
    energy_per_edge = 0.5 * k * (l_cur - rest_lengths) ** 2

    disp = def_points - ref_points
    vis_pts = ref_points + magnify * disp

    emax = max(energy_per_edge.max(), 1e-15)
    norm = Normalize(vmin=0, vmax=emax)

    segments = [[vis_pts[e[0]], vis_pts[e[1]]] for e in edges]
    lc = LineCollection(segments, cmap='hot', norm=norm, linewidths=0.6)
    lc.set_array(energy_per_edge)
    ax.add_collection(lc)
    ax.set_xlim(vis_pts[:, 0].min() - 1, vis_pts[:, 0].max() + 1)
    ax.set_ylim(vis_pts[:, 1].min() - 1, vis_pts[:, 1].max() + 1)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=7)
    plt.colorbar(lc, ax=ax, shrink=0.65, pad=0.02, label='energy')


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print('=' * 78)
    print('  MECHANICAL SIMULATION  --  spring elastic-energy minimisation')
    print('  Boundary: KUBC (10% margin)  |  Strain: delta = 0.005')
    print('=' * 78)

    configs = build_configs()
    delta = 0.005

    results = []

    for cfg in configs:
        label = cfg['label']
        pts = cfg['points']
        simps = cfg['simplices']
        edges = cfg['edges']
        k = cfg['k']
        rl = cfg['rest_lengths']
        bnd = classify_boundary(pts, margin_frac=0.10)

        n_free = (~bnd).sum()
        n_bnd = bnd.sum()
        n_tri = len(simps)
        A_mesh = mesh_area(pts, simps)

        print(f'\n{"─" * 74}')
        print(f'  {label}')
        print(f'  Nodes: {len(pts)}  ({n_bnd} boundary, {n_free} interior)')
        print(f'  Edges: {len(edges)}  |  Triangles: {n_tri}  |  Area: {A_mesh:.1f}')
        print(f'{"─" * 74}')

        # ── Run all 6 strain tests ──
        C_sim, energies, _, dt_sim, def_pos = extract_elastic_tensor(
            pts, simps, edges, k, rl, bnd, delta=delta,
        )

        # ── Compliance matrix -> Poisson & Young ──
        try:
            S = np.linalg.inv(C_sim)
            nu_sim_x = -S[0, 1] / S[0, 0]   # Poisson for x-loading
            nu_sim_y = -S[1, 0] / S[1, 1]   # Poisson for y-loading
            E_sim_x = 1.0 / S[0, 0]
            E_sim_y = 1.0 / S[1, 1]
        except np.linalg.LinAlgError:
            nu_sim_x = nu_sim_y = E_sim_x = E_sim_y = float('nan')

        # ── Analytical (homogenisation) ──
        with torch.no_grad():
            r_an = cfg['solver'](cfg['rigs_torch'])
        C_an_6 = r_an['elastic_tensor'].numpy()
        nu_an = r_an['poisson'].item()
        E_an = r_an['young'].item()

        # Best Poisson estimate = average of x- and y-load compliance values
        # (Note: direct displacement measurement is invalid under KUBC because
        #  all boundaries are affinely fixed, suppressing visible lateral contraction.
        #  The Poisson effect is encoded in the energy ratios, not displacements.)
        nu_sim = 0.5 * (nu_sim_x + nu_sim_y)

        # ── Print results ──
        print(f'\n  Simulation Voigt stiffness C (3x3, absolute units):')
        for row in C_sim:
            print(f'    [{row[0]:12.6e}  {row[1]:12.6e}  {row[2]:12.6e}]')

        print(f'\n  Simulation isotropy: C11/C22 = {C_sim[0,0]/C_sim[1,1]:.4f}'
              f'   (1.0 = isotropic)')

        print(f'\n  Poisson ratio (from compliance matrix):')
        print(f'    nu_x (x-load) = {nu_sim_x:.6f}')
        print(f'    nu_y (y-load) = {nu_sim_y:.6f}')
        print(f'    nu_avg (sim)  = {nu_sim:.6f}')
        print(f'    nu (analytic) = {nu_an:.6f}')
        err_nu = abs(nu_sim - nu_an) / max(abs(nu_an), 1e-15)
        print(f'    error         = {err_nu:.1%}')

        print(f'\n  Time: {dt_sim:.2f}s  ({6} strain tests, L-BFGS-B)')

        results.append(dict(
            label=label, C_sim=C_sim,
            nu_sim=nu_sim, nu_an=nu_an,
            nu_comp_x=nu_sim_x, nu_comp_y=nu_sim_y,
            E_sim_x=E_sim_x, E_an=E_an,
            points=pts, simplices=simps, edges=edges, k=k, rl=rl,
            def_xx=def_pos['xx'], def_xy=def_pos['xy'],
            energies=energies, dt=dt_sim,
        ))

    # ══════════════════════════════════════════════════════════════════════
    # Summary tables
    # ══════════════════════════════════════════════════════════════════════
    print('\n\n' + '=' * 82)
    print('  SUMMARY: Poisson Ratio  --  Simulation (compliance) vs Analytical')
    print('=' * 82)
    print(f'  {"Config":<26} {"nu_x(sim)":>10} {"nu_y(sim)":>10} '
          f'{"nu_avg":>10} {"nu(analyt)":>10} {"error":>7}')
    print(f'  {"─" * 80}')
    for r in results:
        nu_a = r['nu_an']
        err = abs(r['nu_sim'] - nu_a) / max(abs(nu_a), 1e-15)
        print(f'  {r["label"]:<26} {r["nu_comp_x"]:10.5f} {r["nu_comp_y"]:10.5f} '
              f'{r["nu_sim"]:10.5f} {nu_a:10.5f} {err:6.1%}')
    print('=' * 82)

    print(f'\n  Relative stiffness (normalized to config A):')
    E_a_sim = results[0]['E_sim_x']
    E_a_an = results[0]['E_an']
    print(f'  {"Config":<26} {"E/E_A (sim)":>12} {"E/E_A (ana)":>12} {"ratio":>8}')
    print(f'  {"─" * 60}')
    for r in results:
        rs = r['E_sim_x'] / E_a_sim
        ra = r['E_an'] / E_a_an
        ratio = rs / ra if abs(ra) > 1e-15 else float('nan')
        print(f'  {r["label"]:<26} {rs:12.5f} {ra:12.5f} {ratio:8.4f}')

    print(f'\n  Energy under applied strains (absolute, delta={delta}):')
    print(f'  {"Config":<26} {"E(exx)":>11} {"E(eyy)":>11} {"E(exy)":>11}')
    print(f'  {"─" * 62}')
    for r in results:
        exx = r['energies']['xx']
        eyy = r['energies']['yy']
        exy = r['energies']['xy']
        print(f'  {r["label"]:<26} {exx:11.6e} {eyy:11.6e} {exy:11.6e}')

    # ══════════════════════════════════════════════════════════════════════
    # Figure
    # ══════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(4, 3, figsize=(18, 22))

    fig.suptitle(
        'Mechanical Simulation: Spring Energy Minimisation (KUBC)\n'
        'Left: reference mesh (rigidity)   |   Centre: uniaxial '
        r'$\varepsilon_{xx}$'
        f'={delta} (strain, x50)   |   Right: shear '
        r'$\varepsilon_{xy}$'
        f'={delta} (energy, x50)',
        fontsize=11, y=0.998)

    magnify = 50.0
    for i, r in enumerate(results):
        plot_reference_mesh(
            axes[i, 0], r['points'], r['edges'], r['k'],
            title=f'{r["label"]}  --  reference mesh')

        plot_deformed_mesh(
            axes[i, 1], r['points'], r['def_xx'],
            r['edges'], r['k'], r['rl'],
            title=(f'uniaxial exx   '
                   f'nu_sim={r["nu_sim"]:.3f}  nu_ana={r["nu_an"]:.3f}'),
            magnify=magnify)

        plot_energy_map(
            axes[i, 2], r['points'], r['def_xy'],
            r['edges'], r['k'], r['rl'],
            title=f'shear exy  (energy map)',
            magnify=magnify)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    outpath = 'Phase 3/mechanical_simulation.png'
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'\n  Figure saved -> {outpath}')


if __name__ == '__main__':
    main()
