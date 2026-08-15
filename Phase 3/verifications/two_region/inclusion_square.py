"""
Build a 30x30 REGULAR (ν=+0.5) matrix and a square AUXETIC (ν=-0.5) inclusion SEPARATELY (each its
own independent whole-patch design), at three inclusion rigidities vs the matrix (E_matrix=1):
STIFF (E=5) · SAME (E=1) · SOFT (E=0.2). Then punch a hole matching the inclusion's own box out of
the matrix's centre and GLUE the two (C.glue: retriangulate the union; the new matrix/inclusion
interface bonds default to k=1, untouched by either optimisation). Cut the glued design into an open
sheet and stretch along x (clamp only x on the two ends, all else incl. clamp-y free); plot the
macroscopic deformed shape and the lateral-strain field ε_yy (red = expands = auxetic). Networks saved.
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
import ribbon as RB                      # reuse strain()/profile()
import response_fields as RF             # reuse coarse()

HALF_M, HALF_I, REG = 15.0, 5.0, 1e-4                   # matrix 30x30, inclusion 10x10, ordered
CASES = [('stiff', 5.0), ('same', 1.0), ('soft', 0.2)]


def design_whole(half, nu_t, E_t, tag, n_iter=220, n_restarts=1):
    geo = C.make_lattice(1.0, 1.0, half=half, eta=0.0, seed=0)
    prob = C.DesignProblem.from_geo(geo)
    r = C.optimize(prob, [C.Objective('nu', nu_t, weight=3.0), C.Objective('E', E_t, weight=1.0)],
                   mode='k', n_iter=n_iter, reg=REG, n_restarts=n_restarts, verbose=False)
    k = r['k'].detach().numpy()
    C.apply_k_to_geo(geo, r['k']); C6 = C.sim_per_triangle_C6(geo)
    nu, E = C.c6_nuE(C.sim_bulk_C6(geo))
    print(f"    {tag}: nu={nu:+.3f} (tgt {nu_t:+.2f}) E={E:.2f} (tgt {E_t})   "
          f"k min={k.min():.3f} mean={k.mean():.3f}", flush=True)
    return geo, k


def main():
    geoM, kM = design_whole(HALF_M, 0.5, 1.0, 'matrix (regular, solo)')
    Lx, Ly = float(geoM['BL1'][0]), float(geoM['BL2'][1])
    cx, cy = Lx / 2, Ly / 2

    fig, axes = plt.subplots(len(CASES), 2, figsize=(13, 5.0 * len(CASES)), squeeze=False)
    for r, (tag, Ein) in enumerate(CASES):
        print(f"  case {tag}:", flush=True)
        niter = 500 if tag == 'stiff' else 220           # stiff (nu=-0.5 & E=5) needs more optimisation
        rest = 4 if tag == 'stiff' else 1
        geoI, kI = design_whole(HALF_I, -0.5, Ein, f'inclusion (auxetic, solo, E={Ein})',
                                n_iter=niter, n_restarts=rest)
        geo, glued, spec, C6, disc, out, nd, Ed, no, Eo, u, nwt = C.glue_square_hole(
            geoM, kM, geoI, kI, Lx, Ly, cx, cy)
        cen = np.asarray(geo['centroids'])
        C.save_network(os.path.join(HERE, f'inclusion_sq_{tag}.npz'), geo, geo['bond_k'], C6,
                       region=spec, disc=disc.tolist())

        exx, eyy = RB.strain(geo, u)
        exxC = RF.coarse(geo, exx, nwt, ncell=24); eyyC = RF.coarse(geo, eyy, nwt, ncell=24)
        ein = np.nanmean(eyyC[nwt & np.isin(np.arange(len(cen)), disc)])
        eou = np.nanmean(eyyC[nwt & np.isin(np.arange(len(cen)), out)])
        print(f"    under x-stretch: <eyy>_inclusion={ein:+.4f} (auxetic, expect >0)  "
              f"<eyy>_matrix={eou:+.4f}", flush=True)

        Lxb, Lyb = float(geo['BL1'][0]), float(geo['BL2'][1]); mg = 0.08 * Lyb
        tv = np.asarray(geo['tri_verts'])[nwt]; sx = np.asarray(geo['simplices'])[nwt]
        dtv = tv + 3.0 * u[sx]
        v = np.nanpercentile(np.abs(eyyC[nwt]), 94)
        cols = plt.cm.RdBu_r(0.5 + 0.5 * np.clip(np.nan_to_num(eyyC[nwt]) / v, -1, 1))
        a = axes[r, 0]
        a.add_collection(PolyCollection(list(tv), facecolors='none', edgecolors='0.88', lw=0.1))
        a.add_collection(PolyCollection(list(dtv), facecolors=cols, edgecolors='0.6', lw=0.08, alpha=0.9))
        a.set_xlim(-mg, Lxb + mg); a.set_ylim(-mg, Lyb + mg); a.set_aspect('equal')
        a.set_xticks([]); a.set_yticks([])
        a.set_title(f'{tag.upper()} inclusion (deform ×3): disc ν={nd:+.2f} E={Ed:.1f} · '
                    f'matrix ν={no:+.2f} E={Eo:.1f}', fontsize=9)
        gp = {'tri_verts': tv, 'BL1': geo['BL1'], 'BL2': geo['BL2']}
        pc = C.fill_local_map(axes[r, 1], gp, eyyC[nwt], cmap='RdBu_r', sym=True, vlim=v)
        C.draw_box(axes[r, 1], geo); C.mark_region(axes[r, 1], spec)
        plt.colorbar(pc, ax=axes[r, 1], fraction=0.046)
        axes[r, 1].set_title(f'{tag.upper()}: lateral strain εyy  [red = expands = auxetic]  '
                             f'⟨εyy⟩disc={ein:+.3f}', fontsize=9)
    fig.suptitle('GLUED 30x30 regular matrix (ν=+0.5) + square auxetic inclusion (ν=-0.5), designed '
                 'SEPARATELY — actual x-stretch response at three inclusion rigidities', fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(os.path.join(HERE, 'inclusion_square.png'), dpi=145, bbox_inches='tight')
    plt.close(); print('saved inclusion_square.png')


if __name__ == '__main__':
    main()
