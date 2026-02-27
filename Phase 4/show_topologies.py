"""Generate and plot all 16 topologies for visual review."""

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


def plot_topology(ax, tri_result, title=''):
    """Plot a triangulation with hard edges thick/dark, soft edges thin/light."""
    points = tri_result.points
    simplices = tri_result.simplices

    # Collect edges with hard/soft distinction
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

    # Plot soft edges first (behind)
    if soft_segments:
        lc_soft = LineCollection(soft_segments, colors='#cccccc',
                                 linewidths=0.3, alpha=0.5)
        ax.add_collection(lc_soft)

    # Plot hard edges
    if hard_segments:
        lc_hard = LineCollection(hard_segments, colors='#2c3e50',
                                 linewidths=0.8)
        ax.add_collection(lc_hard)

    # Plot nodes
    # Color hard-edge nodes darker
    if tri_result.hard_edge_set is not None:
        hard_nodes = set()
        for edge in tri_result.hard_edge_set:
            hard_nodes.update(edge)
        # Only plot nodes that appear in simplices
        active_nodes = set(simplices.ravel())
        hard_active = np.array(sorted(hard_nodes & active_nodes))
        soft_active = np.array(sorted(active_nodes - hard_nodes))

        if len(hard_active) > 0:
            ax.scatter(points[hard_active, 0], points[hard_active, 1],
                       s=4, c='#2c3e50', zorder=5)
        if len(soft_active) > 0:
            ax.scatter(points[soft_active, 0], points[soft_active, 1],
                       s=2, c='#aaaaaa', zorder=4)
    else:
        active_nodes = np.unique(simplices.ravel())
        ax.scatter(points[active_nodes, 0], points[active_nodes, 1],
                   s=4, c='#2c3e50', zorder=5)

    ax.set_aspect('equal')
    ax.set_title(title, fontsize=8, fontweight='bold')
    ax.tick_params(labelsize=5)
    ax.autoscale_view()


def main():
    size = (4, 4)
    names = list(TOPOLOGY_GENERATORS.keys())
    n = len(names)
    ncols = 4
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 5 * nrows))
    axes = axes.ravel()

    for i, name in enumerate(names):
        print(f"Generating {name}...")
        tri = generate_topology(name, size, seed=42)

        # Get Poisson ratio
        try:
            compat = tri.to_delaunay_compat()
            solver, _, _ = from_triangulation(compat)
            rigs = torch.tensor(tri.get_default_rigidities(), dtype=torch.float64)
            with torch.no_grad():
                result = solver(rigs)
            nu = result['poisson'].item()
        except Exception as e:
            nu = float('nan')

        mask = tri.get_hard_edge_mask_per_triangle()
        n_hard = int(mask.sum())
        n_soft = int((~mask).sum())

        title = (f"{name}\n"
                 f"{tri.n_points} pts, {tri.n_tri} tri, "
                 f"nu={nu:+.3f}\n"
                 f"hard={n_hard}, soft={n_soft}")

        plot_topology(axes[i], tri, title)

    # Hide unused axes
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('Phase 4 Topology Catalog (size=(4,4), seed=42)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    out_path = Path(__file__).parent / 'outputs' / 'topology_catalog.png'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    main()
