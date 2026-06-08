"""Generate and plot all 16 topologies with Poisson's ratio in both directions."""

import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "Phase 2"))
sys.path.insert(0, str(PROJECT_ROOT / "Phase 3"))
sys.path.insert(0, str(PROJECT_ROOT / "Phase 4"))

from data.topology_generators import generate_topology, TOPOLOGY_GENERATORS
from forward_solver_torch import from_triangulation
import torch


def poisson_both_directions(C):
    """Extract nu_xy and nu_yx from 6-component elastic tensor.

    C = [C₁₁₁₁, C₁₁₁₂, C₁₁₂₂, C₁₂₁₂, C₂₂₁₂, C₂₂₂₂]

    Voigt stiffness matrix (2D):
        [[C[0], C[2], C[1]],
         [C[2], C[5], C[4]],
         [C[1], C[4], C[3]]]

    Compliance S = inv(Voigt).
    nu_xy = -S[0,1]/S[0,0]   (lateral contraction in y under x-load)
    nu_yx = -S[0,1]/S[1,1]   (lateral contraction in x under y-load)
    """
    voigt = np.array([
        [C[0], C[2], C[1]],
        [C[2], C[5], C[4]],
        [C[1], C[4], C[3]],
    ])
    S = np.linalg.inv(voigt)
    nu_xy = -S[0, 1] / S[0, 0]
    nu_yx = -S[0, 1] / S[1, 1]
    return nu_xy, nu_yx


def plot_topology(ax, tri_result, title=''):
    """Plot a triangulation with hard edges thick/dark, soft edges thin/light."""
    points = tri_result.points
    simplices = tri_result.simplices

    hard_segments = []
    soft_segments = []

    seen = set()
    for tri in simplices:
        edges = [(tri[0], tri[1]), (tri[0], tri[2]), (tri[1], tri[2])]
        for a, b in edges:
            key = (min(a, b), max(a, b))
            if key in seen:
                continue
            seen.add(key)
            seg = [points[a], points[b]]
            if tri_result.edge_is_hard(a, b):
                hard_segments.append(seg)
            else:
                soft_segments.append(seg)

    if soft_segments:
        lc_soft = LineCollection(soft_segments, colors='#cccccc',
                                 linewidths=0.2, alpha=0.4)
        ax.add_collection(lc_soft)

    if hard_segments:
        lc_hard = LineCollection(hard_segments, colors='#2c3e50',
                                 linewidths=0.5)
        ax.add_collection(lc_hard)

    if tri_result.hard_edge_set is not None:
        hard_nodes = set()
        for edge in tri_result.hard_edge_set:
            hard_nodes.update(edge)
        active_nodes = set(simplices.ravel())
        hard_active = np.array(sorted(hard_nodes & active_nodes))
        soft_active = np.array(sorted(active_nodes - hard_nodes))

        if len(hard_active) > 0:
            ax.scatter(points[hard_active, 0], points[hard_active, 1],
                       s=1.5, c='#2c3e50', zorder=5)
        if len(soft_active) > 0:
            ax.scatter(points[soft_active, 0], points[soft_active, 1],
                       s=0.8, c='#aaaaaa', zorder=4)
    else:
        active_nodes = np.unique(simplices.ravel())
        ax.scatter(points[active_nodes, 0], points[active_nodes, 1],
                   s=1.5, c='#2c3e50', zorder=5)

    ax.set_aspect('equal')
    ax.set_title(title, fontsize=7, fontweight='bold')
    ax.tick_params(labelsize=4)
    ax.autoscale_view()


def main():
    size = (10, 10)
    names = list(TOPOLOGY_GENERATORS.keys())
    n = len(names)
    ncols = 6
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(30, 10 * nrows))
    axes = axes.ravel()

    print(f"{'Topology':<22} {'pts':>5} {'tri':>5} {'hard':>5} {'soft':>5}"
          f"  {'nu_xy':>8} {'nu_yx':>8}")
    print("-" * 75)

    # Use different seeds for foam variants so they don't share random state
    seed_overrides = {'foam_eta045': 137}

    for i, name in enumerate(names):
        print(f"Generating {name}...", end=" ", flush=True)
        seed = seed_overrides.get(name, 42)
        tri = generate_topology(name, size, seed=seed)

        nu_xy = nu_yx = float('nan')
        try:
            compat = tri.to_delaunay_compat()
            solver, _, _ = from_triangulation(compat)
            rigs = torch.tensor(tri.get_default_rigidities(), dtype=torch.float64)
            with torch.no_grad():
                result = solver(rigs)
            C = result['elastic_tensor'].detach().cpu().numpy()
            nu_xy, nu_yx = poisson_both_directions(C)
        except Exception as e:
            print(f"ERROR: {e}")

        mask = tri.get_hard_edge_mask_per_triangle()
        n_hard = int(mask.sum())
        n_soft = int((~mask).sum())

        print(f"{name:<22} {tri.n_points:>5} {tri.n_tri:>5} {n_hard:>5} {n_soft:>5}"
              f"  {nu_xy:>+8.4f} {nu_yx:>+8.4f}")

        title = (f"{name}\n"
                 f"{tri.n_points} pts, {tri.n_tri} tri, "
                 f"hard={n_hard}, soft={n_soft}\n"
                 f"nu_xy={nu_xy:+.4f}  nu_yx={nu_yx:+.4f}")

        plot_topology(axes[i], tri, title)

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('Phase 4 Topology Catalog (size=(10,10), seed=42)\n'
                 'nu_xy = Poisson under x-load, nu_yx = Poisson under y-load',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    out_path = Path(__file__).parent / 'outputs' / 'topology_catalog_10x10.png'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    main()
