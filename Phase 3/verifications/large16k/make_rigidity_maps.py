"""
Spring-rigidity maps for the large-16k designs: each network drawn in real geometry with every EDGE
colored by its designed rigidity k (loaded from the saved networks). 4 types x 2 topologies.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import _common as C

ND = os.path.join(HERE, 'networks')
KINDS = ['iso', 'aniso', 'indep', 'patch']
TOPOS = ['regular', 'disorder_hi']


def main():
    fig, axes = plt.subplots(len(KINDS), len(TOPOS), figsize=(6.4 * len(TOPOS), 6.2 * len(KINDS)),
                             squeeze=False)
    for r, kind in enumerate(KINDS):
        for c, topo in enumerate(TOPOS):
            geo, k, C6, meta = C.load_network(os.path.join(ND, f'{kind}__{topo}.npz'))
            lc = C.draw_network(axes[r, c], geo, k, cmap='viridis', lw_scale=0.9)  # thin at this density
            lo, hi = np.percentile(np.asarray(k), [3, 97])       # clip outliers so the bulk contrast shows
            lc.set_clim(lo, hi)
            C.mark_region(axes[r, c], meta.get('region'))
            plt.colorbar(lc, ax=axes[r, c], fraction=0.046, extend='both', label='k')
            axes[r, c].set_title(f'{kind} · {topo}  (k∈[{k.min():.2f},{k.max():.2f}], '
                                 f'shown {lo:.2f}–{hi:.2f})', fontsize=10)
    fig.suptitle('LARGE 16k designs — spring rigidity k per edge (real geometry)', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.985])
    plt.savefig(os.path.join(HERE, 'large16k_rigidity.png'), dpi=150, bbox_inches='tight')
    plt.close(); print('saved large16k_rigidity.png')


if __name__ == '__main__':
    main()
