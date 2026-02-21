#!/usr/bin/env python
"""Large sweep: compare different network topologies for isotropic Poisson targeting.

Topologies (9 total, all 20x20):
  0: Isotropic crystal   — perfect hexagonal lattice (eta=0, shape=(1,1))
  1: Anisotropic crystal  — stretched hexagonal lattice (eta=0, shape=(1, 1.8))
  2-3: Foam eta=0.2       — perturbed hex lattice, moderate disorder (2 seeds)
  4-5: Foam eta=0.45      — perturbed hex lattice, high disorder (2 seeds)
  6-8: Poisson random     — Poisson point process + Delaunay (3 seeds)

For each topology we run the isotropic sweep across Poisson ratio targets,
optimising rigidities, rest_lengths, and both.

For rest_lengths and both:  we also compute a *residual energy* — the elastic
energy stored in the network when it relaxes to mechanical equilibrium with the
optimised rest lengths (measuring how far from a stress-free state the solution is).
"""

import sys, os, json, time, itertools
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'Phase 2'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.environ['MPLBACKEND'] = 'Agg'

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import TwoSlopeNorm, LogNorm, Normalize
import matplotlib.cm as cm
from scipy.spatial import Delaunay

import Disc_2_Cont_optimized as D2C
from forward_solver_torch import ElasticSolver, from_triangulation
from inverse_optimize import run_property_optimization

OUT = os.path.dirname(os.path.abspath(__file__))

# ── Configuration ─────────────────────────────────────────────────────────
POISSON_TARGETS = [-0.9, -0.7, -0.5, -0.3, -0.1, 0.0, 0.1, 0.3, 0.5, 0.7, 0.9]
DESIGN_VARS = ['rigidities', 'rest_lengths', 'both']
OPT_SEEDS = [7]  # single restart for speed (20x20 meshes are large)
MESH_SIZE = (20, 20)
MAX_ITER = 500
LR = 0.05
TOL = 1e-14

# ══════════════════════════════════════════════════════════════════════════
# Topology generation
# ══════════════════════════════════════════════════════════════════════════

def generate_poisson_random_network(size, n_target_nodes, seed):
    """Generate a random network via Poisson point process + Delaunay.

    Scatter n_target_nodes uniformly in [-size[0], size[0]] x [-size[1], size[1]]
    (with a border margin), Delaunay-triangulate, then crop triangles whose
    centroid falls within the box — exactly like the lattice-based generators.

    Args:
        size: (width, height) half-extents
        n_target_nodes: approximate number of interior nodes
        seed: random seed

    Returns:
        scipy.spatial.Delaunay object with .simplices cropped to good triangles
    """
    rng = np.random.RandomState(seed)
    # Scatter in a box slightly larger than the target to have border nodes
    margin = 2.0
    x_lo, x_hi = -size[0] - margin, size[0] + margin
    y_lo, y_hi = -size[1] - margin, size[1] + margin
    area_total = (x_hi - x_lo) * (y_hi - y_lo)
    area_inner = (2 * size[0]) * (2 * size[1])
    # Scale up the number of points to account for border
    n_pts = int(n_target_nodes * area_total / area_inner * 1.05)
    points = np.column_stack([
        rng.uniform(x_lo, x_hi, n_pts),
        rng.uniform(y_lo, y_hi, n_pts),
    ])
    DM = Delaunay(points)
    # Crop: keep triangles whose centroid is inside [-size, size]
    centroids = np.mean(DM.points[DM.simplices], axis=1)
    good = (
        (np.abs(centroids[:, 0]) <= size[0]) &
        (np.abs(centroids[:, 1]) <= size[1])
    )
    DM.all_simplices = DM.simplices.copy()
    DM.simplices = DM.simplices[good]
    DM.centroids = centroids[good]
    DM.goods_bool = good
    DM.good_idxs = np.where(good)
    return DM


def count_lattice_nodes(size):
    """Count the approximate number of nodes the hex-lattice generator produces."""
    tri = D2C.generate_cryratl_points(size, (1, 1), 0)
    return len(tri.points)


def build_topologies():
    """Build the 9 topologies and return a list of dicts."""
    topos = []
    n_ref_nodes = count_lattice_nodes(MESH_SIZE)
    print(f"Reference hex-lattice node count for {MESH_SIZE}: {n_ref_nodes}")

    # 0: Isotropic crystal (perfect hexagonal)
    print("\n[Topo 0] Isotropic crystal (perfect hex, eta=0)")
    np.random.seed(0)
    tri = D2C.generate_cryratl_points(MESH_SIZE, (1, 1), 0)
    D2C.add_edges_to_triangulation(tri)
    solver, _, _ = from_triangulation(tri)
    n_tri = len(tri.simplices)
    topos.append({
        'name': 'Isotropic crystal', 'short': 'iso_crystal',
        'tri': tri, 'solver': solver, 'n_tri': n_tri,
        'actual_rl': solver.actual_length2.sqrt().numpy(),
        'description': 'Perfect hexagonal lattice (eta=0, shape=(1,1))',
    })
    print(f"  {n_tri} triangles, {len(tri.points)} nodes")

    # 1: Anisotropic crystal (stretched hex)
    print("\n[Topo 1] Anisotropic crystal (stretched hex, shape=(1, 1.8))")
    np.random.seed(1)
    tri = D2C.generate_cryratl_points(MESH_SIZE, (1, 1.8), 0)
    D2C.add_edges_to_triangulation(tri)
    solver, _, _ = from_triangulation(tri)
    n_tri = len(tri.simplices)
    topos.append({
        'name': 'Anisotropic crystal', 'short': 'aniso_crystal',
        'tri': tri, 'solver': solver, 'n_tri': n_tri,
        'actual_rl': solver.actual_length2.sqrt().numpy(),
        'description': 'Stretched hexagonal lattice (eta=0, shape=(1,1.8))',
    })
    print(f"  {n_tri} triangles, {len(tri.points)} nodes")

    # 2-3: Foam eta=0.2
    for idx, seed in enumerate([42, 137]):
        topo_id = 2 + idx
        print(f"\n[Topo {topo_id}] Foam eta=0.2, seed={seed}")
        np.random.seed(seed)
        tri = D2C.generate_foam_points(size=MESH_SIZE, eta=0.2)
        D2C.add_edges_to_triangulation(tri)
        solver, _, _ = from_triangulation(tri)
        n_tri = len(tri.simplices)
        topos.append({
            'name': f'Foam eta=0.2 (seed {seed})', 'short': f'foam_eta02_{seed}',
            'tri': tri, 'solver': solver, 'n_tri': n_tri,
            'actual_rl': solver.actual_length2.sqrt().numpy(),
            'description': f'Hex lattice + eta=0.2 perturbation, seed={seed}',
        })
        print(f"  {n_tri} triangles, {len(tri.points)} nodes")

    # 4-5: Foam eta=0.45
    for idx, seed in enumerate([256, 314]):
        topo_id = 4 + idx
        print(f"\n[Topo {topo_id}] Foam eta=0.45, seed={seed}")
        np.random.seed(seed)
        tri = D2C.generate_foam_points(size=MESH_SIZE, eta=0.45)
        D2C.add_edges_to_triangulation(tri)
        solver, _, _ = from_triangulation(tri)
        n_tri = len(tri.simplices)
        topos.append({
            'name': f'Foam eta=0.45 (seed {seed})', 'short': f'foam_eta045_{seed}',
            'tri': tri, 'solver': solver, 'n_tri': n_tri,
            'actual_rl': solver.actual_length2.sqrt().numpy(),
            'description': f'Hex lattice + eta=0.45 perturbation, seed={seed}',
        })
        print(f"  {n_tri} triangles, {len(tri.points)} nodes")

    # 6-8: Poisson random
    for idx, seed in enumerate([999, 1337, 2025]):
        topo_id = 6 + idx
        print(f"\n[Topo {topo_id}] Poisson random, seed={seed}")
        tri = generate_poisson_random_network(MESH_SIZE, n_ref_nodes, seed)
        D2C.add_edges_to_triangulation(tri)
        solver, _, _ = from_triangulation(tri)
        n_tri = len(tri.simplices)
        topos.append({
            'name': f'Poisson random (seed {seed})', 'short': f'poisson_{seed}',
            'tri': tri, 'solver': solver, 'n_tri': n_tri,
            'actual_rl': solver.actual_length2.sqrt().numpy(),
            'description': f'Poisson point process + Delaunay, seed={seed}, ~{n_ref_nodes} target nodes',
        })
        print(f"  {n_tri} triangles, {len(tri.points)} nodes")

    return topos


# ══════════════════════════════════════════════════════════════════════════
# Residual energy calculation
# ══════════════════════════════════════════════════════════════════════════

def compute_residual_energy(solver, rigidities, rest_lengths):
    """Compute the elastic energy stored in the network at its current geometry.

    The energy per spring is:  E_i = (1/2) k_i (l_actual_i - l0_i)^2
    where l_actual is the current (fixed) edge length and l0 is the rest length.
    If l0 == l_actual for all springs, energy = 0 (stress-free).

    We also return the energy normalised by the total stiffness (energy per unit
    stiffness) so that comparisons across different rigidity scales are fair.

    Args:
        solver: ElasticSolver
        rigidities: (N, 3) numpy array
        rest_lengths: (N, 3) numpy array

    Returns:
        dict with:
            'total_energy': float — sum of 0.5 * k * (l - l0)^2 over all springs
            'energy_per_spring': float — total / n_springs
            'normalised_energy': float — total / sum(k)
            'max_spring_energy': float — max over springs
            'frac_stressed': float — fraction of springs with |l-l0|/l0 > 0.01
    """
    actual_l = solver.actual_length2.sqrt().numpy()  # (N, 3)
    k = rigidities
    l0 = rest_lengths
    dl = actual_l - l0  # extension
    spring_energy = 0.5 * k * dl ** 2
    total = float(spring_energy.sum())
    n_springs = k.size
    return {
        'total_energy': total,
        'energy_per_spring': total / n_springs,
        'normalised_energy': total / (k.sum() + 1e-30),
        'max_spring_energy': float(spring_energy.max()),
        'frac_stressed': float((np.abs(dl) / (l0 + 1e-30) > 0.01).mean()),
    }


# ══════════════════════════════════════════════════════════════════════════
# Coordination number analysis
# ══════════════════════════════════════════════════════════════════════════

def compute_coordination_stats(tri):
    """Compute coordination number statistics for the triangulation."""
    simplices = tri.simplices
    n_nodes = len(tri.points)
    coord = np.zeros(n_nodes, dtype=int)
    # Count edges per node (from triangles)
    edge_set = set()
    for s in simplices:
        for i in range(3):
            for j in range(i + 1, 3):
                a, b = min(s[i], s[j]), max(s[i], s[j])
                edge_set.add((a, b))
    for a, b in edge_set:
        coord[a] += 1
        coord[b] += 1
    # Only count nodes that appear in at least one triangle
    active = coord > 0
    return {
        'mean': float(coord[active].mean()),
        'std': float(coord[active].std()),
        'min': int(coord[active].min()),
        'max': int(coord[active].max()),
        'n_active_nodes': int(active.sum()),
        'n_edges': len(edge_set),
    }


# ══════════════════════════════════════════════════════════════════════════
# Main sweep
# ══════════════════════════════════════════════════════════════════════════

def run_sweep(topologies):
    """Run the full isotropic sweep for all topologies."""
    N_TOPO = len(topologies)
    total = len(POISSON_TARGETS) * N_TOPO * len(DESIGN_VARS)
    done = 0
    t_start = time.time()

    all_results = {}

    for target_nu in POISSON_TARGETS:
        all_results[target_nu] = {}
        for topo_idx, topo in enumerate(topologies):
            all_results[target_nu][topo_idx] = {}
            for dv in DESIGN_VARS:
                best = None
                for opt_seed in OPT_SEEDS:
                    try:
                        r = run_property_optimization(
                            solver=topo['solver'],
                            n_triangles=topo['n_tri'],
                            target_poisson=target_nu,
                            weight_poisson=1.0,
                            design_variable=dv,
                            isotropic=True,
                            weight_isotropy=10.0,
                            max_iter=MAX_ITER,
                            lr=LR,
                            tol=TOL,
                            optimizer_type='lbfgs',
                            seed=opt_seed,
                            verbose=False,
                        )
                        if best is None or r['final_loss'] < best['final_loss']:
                            best = r
                    except Exception as e:
                        print(f"  FAILED: nu*={target_nu}, topo={topo_idx} "
                              f"({topo['short']}), dv={dv}, seed={opt_seed}: {e}")

                # Compute residual energy for rest_lengths / both
                if best is not None and dv in ('rest_lengths', 'both'):
                    energy = compute_residual_energy(
                        topo['solver'], best['rigidities'], best['rest_lengths'])
                    best['residual_energy'] = energy

                all_results[target_nu][topo_idx][dv] = best
                done += 1
                elapsed = time.time() - t_start
                eta_s = (elapsed / done) * (total - done) if done > 0 else 0
                status = "OK" if best is not None else "FAIL"
                if best is not None:
                    nu_str = f"nu_xy={best['nu_xy']:+.4f}"
                    loss_str = f"loss={best['final_loss']:.2e}"
                else:
                    nu_str = loss_str = "---"
                print(f"  [{done:3d}/{total}] nu*={target_nu:+.1f}  "
                      f"topo={topo_idx}({topo['short']:20s})  dv={dv:13s}  "
                      f"{nu_str}  {loss_str}  [{status}]  [ETA {eta_s:.0f}s]")

    return all_results


# ══════════════════════════════════════════════════════════════════════════
# Visualization helpers
# ══════════════════════════════════════════════════════════════════════════

def make_rigidity_cmap():
    """Create a perceptually uniform diverging colormap for log-rigidities.

    Uses a blue-white-red scheme on log(k) centered at k=1, giving better
    contrast than viridis for seeing which springs are stiffened vs softened.
    """
    from matplotlib.colors import LinearSegmentedColormap
    colors = ['#2166ac', '#67a9cf', '#d1e5f0', '#f7f7f7',
              '#fddbc7', '#ef8a62', '#b2182b']
    return LinearSegmentedColormap.from_list('rigidity_div', colors, N=256)


RIG_CMAP = make_rigidity_cmap()


def plot_mesh_edges(ax, points, simplices, vals_per_tri, cmap, norm, lw=0.5):
    """Draw mesh edges colored by per-edge values."""
    segments, colors = [], []
    for ti, sv in enumerate(simplices):
        for e_idx, (a, b) in enumerate([(sv[0], sv[1]), (sv[0], sv[2]), (sv[1], sv[2])]):
            segments.append([points[a], points[b]])
            colors.append(vals_per_tri[ti, e_idx])
    lc = LineCollection(segments, cmap=cmap, norm=norm, linewidths=lw)
    lc.set_array(np.array(colors))
    ax.add_collection(lc)
    ax.set_xlim(points[:, 0].min() - 0.5, points[:, 0].max() + 0.5)
    ax.set_ylim(points[:, 1].min() - 0.5, points[:, 1].max() + 0.5)
    ax.set_aspect('equal')
    return lc


def generate_per_case_plots(all_results, topologies):
    """Generate mesh + hist/polar plots per (target, topology)."""
    N_TOPO = len(topologies)

    for target_nu in POISSON_TARGETS:
        tag = f"nu{target_nu:+.1f}".replace('+', 'p').replace('-', 'm').replace('.', '')
        subdir = os.path.join(OUT, tag)
        os.makedirs(subdir, exist_ok=True)

        for topo_idx, topo in enumerate(topologies):
            tri_obj = topo['tri']
            actual_rl = topo['actual_rl']
            points = tri_obj.points
            simplices = tri_obj.simplices

            # ── Mesh plots ──────────────────────────────────────────
            fig, axes = plt.subplots(2, 3, figsize=(18, 11))
            for col, dv in enumerate(DESIGN_VARS):
                r = all_results[target_nu][topo_idx][dv]
                if r is None:
                    axes[0, col].text(0.5, 0.5, 'FAILED',
                                      transform=axes[0, col].transAxes,
                                      ha='center', va='center', fontsize=14, color='red')
                    axes[1, col].set_visible(False)
                    continue

                # Row 0: rigidities — use diverging log colormap centered at k=1
                rigs = r['rigidities']
                log_rigs = np.log10(np.clip(rigs, 1e-6, None))
                abs_max = max(abs(log_rigs.min()), abs(log_rigs.max()), 0.5)
                k_norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0.0, vmax=abs_max)
                plot_mesh_edges(axes[0, col], points, simplices, log_rigs,
                                RIG_CMAP, k_norm, lw=0.5)
                info = (f'{dv}\n'
                        f'nu_xy={r["nu_xy"]:+.4f}  nu_yx={r["nu_yx"]:+.4f}\n'
                        f'E_x={r["E_x"]:.3e}  E_y={r["E_y"]:.3e}')
                if 'residual_energy' in r:
                    info += f'\nResid. energy/spring={r["residual_energy"]["energy_per_spring"]:.2e}'
                axes[0, col].set_title(info, fontsize=7)
                cb = fig.colorbar(cm.ScalarMappable(norm=k_norm, cmap=RIG_CMAP),
                                  ax=axes[0, col], label='log10(k)', shrink=0.7, pad=0.02)

                # Row 1: rest-length ratio
                rl_ratio = r['rest_lengths'] / actual_rl
                rl_norm = TwoSlopeNorm(
                    vmin=min(rl_ratio.min(), 0.3), vcenter=1.0,
                    vmax=max(rl_ratio.max(), 3.0))
                plot_mesh_edges(axes[1, col], points, simplices, rl_ratio,
                                'coolwarm', rl_norm, lw=0.5)
                axes[1, col].set_title('l0/l_actual', fontsize=9)
                fig.colorbar(cm.ScalarMappable(norm=rl_norm, cmap='coolwarm'),
                             ax=axes[1, col], label='l0/l_actual', shrink=0.7, pad=0.02)

            fig.suptitle(
                f'Target nu = {target_nu:+.1f}  |  Topo {topo_idx}: {topo["name"]}',
                fontsize=12, y=1.01)
            axes[0, 0].set_ylabel('Rigidities (log10)', fontsize=10)
            axes[1, 0].set_ylabel('Rest-length ratio', fontsize=10)
            fig.tight_layout()
            fig.savefig(os.path.join(subdir, f'mesh_topo{topo_idx}_{topo["short"]}.png'),
                        dpi=120, bbox_inches='tight')
            plt.close(fig)

            # ── Histogram + Polar ────────────────────────────────────
            fig2 = plt.figure(figsize=(18, 9))
            for col, dv in enumerate(DESIGN_VARS):
                r = all_results[target_nu][topo_idx][dv]
                if r is None:
                    continue

                rl_ratio = (r['rest_lengths'] / actual_rl).ravel()
                rigs_flat = r['rigidities'].ravel()

                # Histogram
                ax_h = fig2.add_subplot(2, 3, col + 1)
                ax_h.hist(rl_ratio, bins=50, alpha=0.6, color='steelblue',
                          edgecolor='none', density=True, label='l0/l_actual')
                ax_h.axvline(1.0, color='k', ls='--', lw=0.8)
                ax_h.set_xlabel('l0 / l_actual')
                ax_h.set_ylabel('Density')
                ax_h.set_title(f'{dv}', fontsize=9)
                ax_h.set_xlim(0, max(4, np.percentile(rl_ratio, 99) * 1.1))
                ax_h2 = ax_h.twinx()
                ax_h2.hist(rigs_flat, bins=50, alpha=0.4, color='orange',
                           edgecolor='none', density=True, label='k')
                ax_h2.set_ylabel('Density (k)', color='orange')
                lines1, labels1 = ax_h.get_legend_handles_labels()
                lines2, labels2 = ax_h2.get_legend_handles_labels()
                ax_h.legend(lines1 + lines2, labels1 + labels2, fontsize=7)

                # Polar
                ax_p = fig2.add_subplot(2, 3, 4 + col, projection='polar')
                rl_ratio_2d = r['rest_lengths'] / actual_rl
                angles, ratios_flat = [], []
                for ti, sv in enumerate(simplices):
                    for e_idx, (a, b) in enumerate([(sv[0], sv[1]), (sv[0], sv[2]),
                                                     (sv[1], sv[2])]):
                        dx = points[b, 0] - points[a, 0]
                        dy = points[b, 1] - points[a, 1]
                        angles.append(np.arctan2(dy, dx))
                        ratios_flat.append(rl_ratio_2d[ti, e_idx])
                angles = np.array(angles)
                ratios_flat = np.array(ratios_flat)
                rl_norm_p = TwoSlopeNorm(
                    vmin=min(ratios_flat.min(), 0.3), vcenter=1.0,
                    vmax=max(ratios_flat.max(), 3.0))
                ax_p.scatter(angles, ratios_flat, c=ratios_flat, cmap='coolwarm',
                             norm=rl_norm_p, s=1, alpha=0.3)
                ax_p.set_ylim(0, min(5, np.percentile(ratios_flat, 99) * 1.2))
                ax_p.set_title(f'{dv}', fontsize=9, pad=12)

            fig2.suptitle(
                f'Target nu = {target_nu:+.1f}  |  Topo {topo_idx}: {topo["name"]}',
                fontsize=11, y=1.03)
            fig2.tight_layout()
            fig2.savefig(os.path.join(subdir, f'hist_polar_topo{topo_idx}_{topo["short"]}.png'),
                         dpi=120, bbox_inches='tight')
            plt.close(fig2)

        print(f"  nu*={target_nu:+.1f}: saved mesh/hist/polar for {N_TOPO} topos")


def generate_summary_plots(all_results, topologies):
    """Generate summary comparison plots across all topologies."""
    N_TOPO = len(topologies)
    targets = np.array(POISSON_TARGETS)

    # Assign colors by topology category
    cat_colors = {
        'iso_crystal': '#1b9e77',
        'aniso_crystal': '#d95f02',
        'foam_eta02': '#7570b3',
        'foam_eta045': '#e7298a',
        'poisson': '#66a61e',
    }

    def topo_color(t):
        s = t['short']
        for prefix, c in cat_colors.items():
            if s.startswith(prefix):
                return c
        return 'gray'

    def collect(field):
        out = {}
        for dv in DESIGN_VARS:
            arr = np.full((len(POISSON_TARGETS), N_TOPO), np.nan)
            for i, nu in enumerate(POISSON_TARGETS):
                for j in range(N_TOPO):
                    r = all_results[nu][j][dv]
                    if r is not None and field in r:
                        arr[i, j] = r[field]
            out[dv] = arr
        return out

    nu_xy = collect('nu_xy')
    nu_yx = collect('nu_yx')
    loss_arr = collect('final_loss')

    # ── Figure 1: Achieved nu per topology, faceted by design variable ──
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    for col, dv in enumerate(DESIGN_VARS):
        ax = axes[col]
        for j, topo in enumerate(topologies):
            vals = nu_xy[dv][:, j]
            ax.plot(targets, vals, 'o-', color=topo_color(topo),
                    label=topo['name'], markersize=3, alpha=0.7)
        ax.plot([-1, 1], [-1, 1], 'k--', lw=1, label='Perfect')
        ax.set_xlabel('Target nu')
        ax.set_ylabel('Achieved nu_xy')
        ax.set_title(dv, fontsize=11)
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.legend(fontsize=6, loc='upper left')
    fig.suptitle('Achieved Poisson ratio by topology (isotropic constraint)',
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'summary_achieved_nu.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ── Figure 2: Isotropy quality |nu_xy - nu_yx| ──
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    for col, dv in enumerate(DESIGN_VARS):
        ax = axes[col]
        for j, topo in enumerate(topologies):
            gap = np.abs(nu_xy[dv][:, j] - nu_yx[dv][:, j])
            ax.plot(targets, gap, 'o-', color=topo_color(topo),
                    label=topo['name'], markersize=3, alpha=0.7)
        ax.set_xlabel('Target nu')
        ax.set_ylabel('|nu_xy - nu_yx|')
        ax.set_title(dv, fontsize=11)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.legend(fontsize=6, loc='upper left')
    fig.suptitle('Isotropy quality: |nu_xy - nu_yx| gap by topology',
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'summary_isotropy_gap.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ── Figure 3: Optimization loss by topology ──
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    for col, dv in enumerate(DESIGN_VARS):
        ax = axes[col]
        for j, topo in enumerate(topologies):
            ax.plot(targets, loss_arr[dv][:, j], 'o-', color=topo_color(topo),
                    label=topo['name'], markersize=3, alpha=0.7)
        ax.set_xlabel('Target nu')
        ax.set_ylabel('Final loss')
        ax.set_title(dv, fontsize=11)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.legend(fontsize=6, loc='upper left')
    fig.suptitle('Optimization loss by topology', fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'summary_loss.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ── Figure 4: Residual energy for rest_lengths and both ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for col, dv in enumerate(['rest_lengths', 'both']):
        ax = axes[col]
        for j, topo in enumerate(topologies):
            energies = []
            for i, nu in enumerate(POISSON_TARGETS):
                r = all_results[nu][j][dv]
                if r is not None and 'residual_energy' in r:
                    energies.append(r['residual_energy']['energy_per_spring'])
                else:
                    energies.append(np.nan)
            ax.plot(targets, energies, 'o-', color=topo_color(topo),
                    label=topo['name'], markersize=3, alpha=0.7)
        ax.set_xlabel('Target nu')
        ax.set_ylabel('Residual energy / spring')
        ax.set_title(dv, fontsize=11)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.legend(fontsize=6, loc='upper left')
    fig.suptitle('Residual elastic energy per spring after optimization\n'
                 '(lower = closer to stress-free)',
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'summary_residual_energy.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ── Figure 5: Coordination number comparison ──
    fig, ax = plt.subplots(figsize=(10, 5))
    names = [t['name'] for t in topologies]
    means = []
    stds = []
    colors_bar = [topo_color(t) for t in topologies]
    for t in topologies:
        cs = compute_coordination_stats(t['tri'])
        means.append(cs['mean'])
        stds.append(cs['std'])
    x = np.arange(len(topologies))
    ax.bar(x, means, yerr=stds, color=colors_bar, alpha=0.8, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('Mean coordination number')
    ax.set_title('Coordination number by topology')
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'summary_coordination.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    print("  Saved all summary plots")


# ══════════════════════════════════════════════════════════════════════════
# Save results to JSON
# ══════════════════════════════════════════════════════════════════════════

def save_results(all_results, topologies):
    """Save sweep results (excluding large arrays) to JSON."""
    N_TOPO = len(topologies)
    summary = {
        'config': {
            'mesh_size': list(MESH_SIZE),
            'poisson_targets': POISSON_TARGETS,
            'design_vars': DESIGN_VARS,
            'max_iter': MAX_ITER,
            'n_topologies': N_TOPO,
        },
        'topologies': [
            {'name': t['name'], 'short': t['short'],
             'description': t['description'], 'n_tri': t['n_tri'],
             'coord_stats': compute_coordination_stats(t['tri'])}
            for t in topologies
        ],
        'results': {},
    }
    for target_nu in POISSON_TARGETS:
        key = f"{target_nu:+.1f}"
        summary['results'][key] = {}
        for topo_idx in range(N_TOPO):
            summary['results'][key][str(topo_idx)] = {}
            for dv in DESIGN_VARS:
                r = all_results[target_nu][topo_idx][dv]
                if r is None:
                    summary['results'][key][str(topo_idx)][dv] = None
                    continue
                entry = {
                    'nu_xy': r.get('nu_xy', float('nan')),
                    'nu_yx': r.get('nu_yx', float('nan')),
                    'E_x': r.get('E_x', float('nan')),
                    'E_y': r.get('E_y', float('nan')),
                    'poisson': r['poisson'],
                    'young': r['young'],
                    'final_loss': r['final_loss'],
                    'converged': r['converged'],
                    'iterations': r['iterations'],
                }
                if 'residual_energy' in r:
                    entry['residual_energy'] = r['residual_energy']
                summary['results'][key][str(topo_idx)][dv] = entry

    path = os.path.join(OUT, 'large_sweep_results.json')
    with open(path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Saved {path}")


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 70)
    print("LARGE SWEEP: 9 topologies x 11 Poisson targets x 3 design vars")
    print("=" * 70)

    topologies = build_topologies()

    # Print coordination number overview
    print("\n" + "=" * 70)
    print("Coordination number overview")
    print("=" * 70)
    for i, t in enumerate(topologies):
        cs = compute_coordination_stats(t['tri'])
        print(f"  Topo {i} ({t['name']:30s}): "
              f"z_mean={cs['mean']:.2f} +/- {cs['std']:.2f}, "
              f"z_range=[{cs['min']}, {cs['max']}], "
              f"{cs['n_active_nodes']} nodes, {cs['n_edges']} edges, "
              f"{t['n_tri']} tris")

    print("\n" + "=" * 70)
    print("Running isotropic sweep ...")
    print("=" * 70)
    all_results = run_sweep(topologies)

    print("\n" + "=" * 70)
    print("Saving results ...")
    print("=" * 70)
    save_results(all_results, topologies)

    print("\n" + "=" * 70)
    print("Generating per-case visualizations ...")
    print("=" * 70)
    generate_per_case_plots(all_results, topologies)

    print("\n" + "=" * 70)
    print("Generating summary plots ...")
    print("=" * 70)
    generate_summary_plots(all_results, topologies)

    print("\n" + "=" * 70)
    print("ALL DONE")
    print("=" * 70)
