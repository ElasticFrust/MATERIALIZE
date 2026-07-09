"""
Local nu/E maps for ALL patch cases x ALL topologies, built by LOADING the saved networks
(networks/{case}__{topo}.npz), no re-optimising. Two views:
  PER-TOPOLOGY : one figure per topology, rows = cases, cols = [rigidity k | local nu | local E]
                 with the region marked (design_detail_figure).
  PER-CASE     : one figure per case, cols = topologies, rows = [local nu | local E], region marked.
"""
import os, sys
import matplotlib
matplotlib.use('Agg')

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
        entries = []
        for topo in TOPOS:
            geo, k, C6, meta = load(case, topo)
            entries.append((topo, None, geo, C6, meta.get('region')))
        C.nuE_row_grid(os.path.join(HERE, f'auxetic_patch_bycase_{case}.png'),
                       f'auxetic_patch case "{case}" — local ν (top) & E (bottom) across topologies '
                       f'(region marked)', entries, figsize=(3.0, 6.2))
        print('saved bycase', case)


def main():
    per_topology()
    per_case()
    print('done')


if __name__ == '__main__':
    main()
