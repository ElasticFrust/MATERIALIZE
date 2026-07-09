"""
Local field maps for the glued auxetic|regular ribbon (loaded from the saved ribbon.npz, no
re-optimising): the DESIGNED intrinsic local Poisson ratio and Young's modulus (per-triangle, from
the one full PBC relaxation already used to verify the design), and the local stress magnitude
‖σ‖ actually measured under the open x-stretch (sqrt-scaled -- stress spans orders of magnitude at
the floppy/soft auxetic bonds, sqrt compresses that range so the map isn't dominated by outliers).
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
import response_fields as RF


def main():
    geo, k, C6, meta = C.load_network(os.path.join(HERE, 'ribbon.npz'))
    mid = meta['mid']
    _, nu_v = C.local_scalar_field(geo, C6, quantity='nu')
    _, E_v = C.local_scalar_field(geo, C6, quantity='E')

    u, nwt = C.open_stretch(geo, axis=0, regularize=True)
    _, _, smag = RF.fields(geo, u)
    sscaled = np.sqrt(smag)

    gp_all = {'tri_verts': geo['tri_verts'], 'BL1': geo['BL1'], 'BL2': geo['BL2']}
    gp_open = {'tri_verts': np.asarray(geo['tri_verts'])[nwt], 'BL1': geo['BL1'], 'BL2': geo['BL2']}

    fig, axes = plt.subplots(1, 3, figsize=(19, 6.2))
    pc0 = C.fill_local_map(axes[0], gp_all, nu_v, cmap='RdBu_r', sym=True, vlim=0.6)
    pc1 = C.fill_local_map(axes[1], gp_all, E_v, cmap='viridis')
    pc1.set_clim(0, np.nanpercentile(E_v, 98))
    pc2 = C.fill_local_map(axes[2], gp_open, sscaled[nwt], cmap='magma')
    pc2.set_clim(0, np.nanpercentile(sscaled[nwt], 97))
    titles = ['local Poisson ratio ν  (designed, intrinsic)',
              "local Young's modulus E  (designed, intrinsic)",
              '√‖stress‖ under open x-stretch  (sqrt-scaled)']
    for ax, pc, title in zip(axes, [pc0, pc1, pc2], titles):
        C.draw_box(ax, geo); ax.axvline(mid, color='k', ls='--', lw=1)
        plt.colorbar(pc, ax=ax, fraction=0.046)
        ax.set_title(title, fontsize=11)
    fig.suptitle('Glued auxetic|regular ribbon — local field maps', fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(HERE, 'ribbon_fields.png'), dpi=150, bbox_inches='tight')
    plt.close(); print('saved ribbon_fields.png')


if __name__ == '__main__':
    main()
