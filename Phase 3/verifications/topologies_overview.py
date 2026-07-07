"""
Overview of the topologies used in all cases, drawn in SQUARE (fractional / periodic-wrapped)
coordinates so anisotropic (parallelogram) boxes render as squares and are directly comparable.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C

HALF = 6                        # square half-size (small enough that individual bonds are visible)


def main():
    fig, axes = plt.subplots(1, len(C.TOPOS), figsize=(3.1 * len(C.TOPOS), 3.4))
    for ax, (topo, label, _) in zip(axes, C.TOPOS):
        geo = C.make_topology(topo, HALF)
        C.draw_network(ax, geo, np.ones(len(geo['bond_u'])), cmap='Greys')
        for coll in ax.collections:
            coll.set_linewidth(0.6); coll.set_color('#333333')
        ax.set_title(f'{label}\n({len(geo["simplices"])} tri)', fontsize=9)
    fig.suptitle('Topologies (real geometry, square frame + periodic-box outline) — used in every case',
                 fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'topologies_overview.png')
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print('saved', p)


if __name__ == '__main__':
    main()
