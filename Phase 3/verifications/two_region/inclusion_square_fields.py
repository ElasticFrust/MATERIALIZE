"""
Local field maps for the glued matrix+inclusion designs (loaded from the saved inclusion_sq_*.npz,
no re-optimising): the DESIGNED intrinsic local Poisson ratio and Young's modulus (per-triangle, from
the relaxation already used to verify each design), and the local stress magnitude ||sigma|| actually
measured under the open x-stretch (sqrt-scaled -- stress spans orders of magnitude at the floppy/soft
auxetic bonds, sqrt compresses that range so the map isn't dominated by outliers). One row per
rigidity case (stiff / same / soft), one column per field, each with its own colorbar.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(HERE, '..', '..', '..', 'Phase 2'))
import _common as C
import demo as D
import response_fields as RF

CASES = ['stiff', 'same', 'soft']


def main():
    fig, axes = plt.subplots(len(CASES), 3, figsize=(19, 5.6 * len(CASES)), squeeze=False)
    for r, tag in enumerate(CASES):
        geo, k, C6, meta = C.load_network(os.path.join(HERE, f'inclusion_sq_{tag}.npz'))
        spec = meta['region']
        _, nu_v = C.local_scalar_field(geo, C6, quantity='nu')
        _, E_v = C.local_scalar_field(geo, C6, quantity='E')

        u, nwt = D.cut_stretch(geo, axis=0)
        _, _, smag = RF.fields(geo, u)
        sscaled = np.sqrt(smag)

        gp_all = {'tri_verts': geo['tri_verts'], 'BL1': geo['BL1'], 'BL2': geo['BL2']}
        gp_open = {'tri_verts': np.asarray(geo['tri_verts'])[nwt], 'BL1': geo['BL1'], 'BL2': geo['BL2']}

        pc0 = C.fill_local_map(axes[r, 0], gp_all, nu_v, cmap='RdBu_r', sym=True, vlim=0.6)
        pc1 = C.fill_local_map(axes[r, 1], gp_all, E_v, cmap='viridis')
        pc1.set_clim(0, np.nanpercentile(E_v, 98))
        pc2 = C.fill_local_map(axes[r, 2], gp_open, sscaled[nwt], cmap='magma')
        pc2.set_clim(0, np.nanpercentile(sscaled[nwt], 97))

        titles = ['local Poisson ratio ν  (designed, intrinsic)',
                  "local Young's modulus E  (designed, intrinsic)",
                  '√||stress|| under open x-stretch  (sqrt-scaled)']
        for c, (ax, pc, title) in enumerate(zip(axes[r], [pc0, pc1, pc2], titles)):
            C.draw_box(ax, geo); C.mark_region(ax, spec)
            plt.colorbar(pc, ax=ax, fraction=0.046)
            if r == 0:
                ax.set_title(title, fontsize=11)
        axes[r, 0].set_ylabel(f'{tag.upper()} inclusion', fontsize=11)
    fig.suptitle('Glued matrix + square auxetic inclusion — local field maps (stiff / same / soft)',
                 fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(HERE, 'inclusion_square_fields.png'), dpi=145, bbox_inches='tight')
    plt.close(); print('saved inclusion_square_fields.png')


if __name__ == '__main__':
    main()
