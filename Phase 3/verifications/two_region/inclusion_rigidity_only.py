"""
SANITY CHECK: a 30x30 matrix and a 10x10 square inclusion that are the SAME material (nu=1/3, the
plain regular triangular lattice's own Poisson ratio, k=1 uniform -- no design at all needed since
nu is scale-invariant under uniform k-rescaling) but with very DIFFERENT rigidity: one case with a
much STIFFER inclusion (k x6), one with a much SOFTER inclusion (k x1/6). Glued (C.glue) and cut into
an open sheet, stretched along x. Since nu is identical everywhere, there should be NO lateral-strain
sign difference -- only a redistribution of strain magnitude (stiff region strains less, soft region
strains more) and of stress (stiff region carries more stress). This is the expected, boring composite
behaviour with no Poisson-ratio effect at all -- a control for the auxetic-inclusion experiments.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(HERE, '..', '..', '..', 'Phase 2'))
import _common as C
import ribbon as RB                      # reuse strain()
import response_fields as RF             # reuse coarse()

HALF_M, HALF_I = 15.0, 5.0                # matrix 30x30, inclusion 10x10, ordered (eta=0)
CASES = [('stiff', 6.0), ('soft', 1.0 / 6.0)]


def build_uniform(half, kscale):
    """Plain regular lattice at uniform bond stiffness kscale -- nu stays 1/3 (scale-invariant),
    E scales with kscale. No optimisation needed."""
    geo = C.make_lattice(1.0, 1.0, half=half, eta=0.0, seed=0)
    k = np.full(len(geo['bond_u']), kscale)
    geo['bond_k'] = k; geo['tri_k'] = k[geo['tri_bond']]
    C6 = C.sim_per_triangle_C6(geo)
    nu, E = C.c6_nuE(C.sim_bulk_C6(geo))
    return geo, k, nu, E


def main():
    geoM, kM, nm, Em = build_uniform(HALF_M, 1.0)
    print(f"  matrix: nu={nm:+.3f} (expect +0.333) E={Em:.3f}", flush=True)
    Lx, Ly = float(geoM['BL1'][0]), float(geoM['BL2'][1]); cx, cy = Lx / 2, Ly / 2

    fig, axes = plt.subplots(len(CASES), 2, figsize=(13, 5.0 * len(CASES)), squeeze=False)
    for r, (tag, kscale) in enumerate(CASES):
        geoI, kI, ni, Ei = build_uniform(HALF_I, kscale)
        print(f"  case {tag}: inclusion k={kscale:.3f}  nu={ni:+.3f} (expect +0.333) E={Ei:.3f}", flush=True)
        geo, glued, spec, C6, disc, out, nd, Ed, no, Eo, u, nwt = C.glue_square_hole(
            geoM, kM, geoI, kI, Lx, Ly, cx, cy)
        cen = np.asarray(geo['centroids'])
        C.save_network(os.path.join(HERE, f'inclusion_rigidity_{tag}.npz'), geo, geo['bond_k'], C6,
                       region=spec, disc=disc.tolist())

        exx, eyy, smag = RF.fields(geo, u)
        exxC = RF.coarse(geo, exx, nwt, ncell=24); smagC = RF.coarse(geo, smag, nwt, ncell=24)
        e_in = np.nanmean(exxC[nwt & np.isin(np.arange(len(cen)), disc)])
        e_out = np.nanmean(exxC[nwt & np.isin(np.arange(len(cen)), out)])
        s_in = np.nanmean(smagC[nwt & np.isin(np.arange(len(cen)), disc)])
        s_out = np.nanmean(smagC[nwt & np.isin(np.arange(len(cen)), out)])
        print(f"    under x-stretch: <exx>_inclusion={e_in:+.4f}  <exx>_matrix={e_out:+.4f}   "
              f"<|stress|>_inclusion={s_in:.4f}  <|stress|>_matrix={s_out:.4f}", flush=True)

        Lxb, Lyb = float(geo['BL1'][0]), float(geo['BL2'][1]); mg = 0.08 * Lyb
        tv = np.asarray(geo['tri_verts'])[nwt]; sx = np.asarray(geo['simplices'])[nwt]
        dtv = tv + 3.0 * u[sx]
        v = np.nanpercentile(np.abs(exxC[nwt]), 94)
        cols = plt.cm.RdBu_r(0.5 + 0.5 * np.clip(np.nan_to_num(exxC[nwt]) / v, -1, 1))
        a = axes[r, 0]
        a.add_collection(PolyCollection(list(tv), facecolors='none', edgecolors='0.88', lw=0.1))
        a.add_collection(PolyCollection(list(dtv), facecolors=cols, edgecolors='0.6', lw=0.08, alpha=0.9))
        a.set_xlim(-mg, Lxb + mg); a.set_ylim(-mg, Lyb + mg); a.set_aspect('equal')
        a.set_xticks([]); a.set_yticks([])
        a.set_title(f'{tag.upper()} inclusion (deform x3, colour=axial strain exx): '
                    f'disc E={Ed:.2f} nu={nd:+.2f} - matrix E={Eo:.2f} nu={no:+.2f}', fontsize=9)
        gp = {'tri_verts': tv, 'BL1': geo['BL1'], 'BL2': geo['BL2']}
        pc = C.fill_local_map(axes[r, 1], gp, smagC[nwt], cmap='magma')
        pc.set_clim(0, np.nanpercentile(smagC[nwt], 97))
        C.draw_box(axes[r, 1], geo); C.mark_region(axes[r, 1], spec)
        plt.colorbar(pc, ax=axes[r, 1], fraction=0.046)
        axes[r, 1].set_title(f'{tag.upper()}: stress magnitude ||sigma||  <|s|>disc={s_in:.3f} '
                             f'<|s|>matrix={s_out:.3f}', fontsize=9)
    fig.suptitle('SANITY CHECK: matrix + inclusion, SAME nu=1/3 everywhere, rigidity contrast only '
                 '(stiff x6 / soft x1/6) -- no sign-flip expected, only strain/stress redistribution',
                 fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(HERE, 'inclusion_rigidity_only.png'), dpi=145, bbox_inches='tight')
    plt.close(); print('saved inclusion_rigidity_only.png')


if __name__ == '__main__':
    main()
