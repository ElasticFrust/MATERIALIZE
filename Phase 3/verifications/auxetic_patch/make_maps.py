"""
Local nu/E maps for ALL patch cases x ALL topologies, built by LOADING the saved networks
(networks/{case}__{topo}.npz), no re-optimising. Two views:
  PER-TOPOLOGY : one figure per topology, rows = cases, cols = [rigidity k | local nu | local E]
                 with the region marked (design_detail_figure).
  PER-CASE     : one figure per case, cols = topologies, rows = [local nu | local E], region marked.
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
TOPOS = C.TOPO_IDS
CASES = ['disc_auxetic', 'square_auxetic', 'triangle_auxetic', 'ring_auxetic',
         'normal_in_auxetic', 'stiffE_in_soft', 'softE_in_stiff', 'decoupled']


def load(case, topo):
    return C.load_network(os.path.join(ND, f'{case}__{topo}.npz'))       # geo, k, C6, meta


def per_topology():
    for topo in TOPOS:
        entries = []
        for case in CASES:
            geo, k, C6, meta = load(case, topo)
            entries.append((case, geo, k, C6, meta.get('region')))
        C.design_detail_figure(os.path.join(HERE, f'auxetic_patch_bytopo_{topo}.png'), entries,
                               f'auxetic_patch — ALL cases on "{topo}" [rigidity k | local ν | local E], region in lime')
        print('saved bytopo', topo)


def per_case():
    for case in CASES:
        fig, axes = plt.subplots(2, len(TOPOS), figsize=(3.0 * len(TOPOS), 6.2), squeeze=False)
        for col, topo in enumerate(TOPOS):
            geo, k, C6, meta = load(case, topo)
            reg = meta.get('region')
            nu = C.local_field_smooth(geo, C6, 'nu')
            pnu = C.fill_local_map(axes[0, col], geo, nu, cmap='RdBu_r', sym=True, vlim=0.6)
            C.draw_box(axes[0, col], geo); C.mark_region(axes[0, col], reg)
            axes[0, col].set_title(topo, fontsize=9)
            E = C.local_field_smooth(geo, C6, 'E')
            pE = C.fill_local_map(axes[1, col], geo, E, cmap='viridis')
            pE.set_clim(0, np.nanpercentile(E, 97))
            C.draw_box(axes[1, col], geo); C.mark_region(axes[1, col], reg)
        axes[0, 0].set_ylabel('local ν', fontsize=10); axes[1, 0].set_ylabel('local E', fontsize=10)
        fig.suptitle(f'auxetic_patch case "{case}" — local ν (top) & E (bottom) across topologies '
                     f'(region marked)', fontsize=12)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(os.path.join(HERE, f'auxetic_patch_bycase_{case}.png'), dpi=140, bbox_inches='tight')
        plt.close(); print('saved bycase', case)


def main():
    per_topology()
    per_case()
    print('done')


if __name__ == '__main__':
    main()
