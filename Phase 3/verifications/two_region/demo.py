"""
Two direct "does it behave differently" demonstrations — design, then OPEN cut-and-stretch, drawing
the (exaggerated) deformed shape.

  BAR  : top half auxetic (ν=-0.5), bottom half regular (ν=+0.5), E=1 everywhere. Stretch along x
         (parallel to the interface): the auxetic top should WIDEN in y, the regular bottom NARROW.
  INCL : a stiff, strongly-auxetic disc (ν≈-0.85, E≈5) in a soft normal matrix (ν=+0.5, E=1),
         locally isotropic. Stretch and see the inclusion's distinct response.

All targets are isotropic scalars ('nu'/'E'). Networks saved.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import _common as C

N, NITER, REG = 14, 150, 1e-4


def cut_stretch(geo, axis):
    """Open cut-and-stretch, regularized (this module pins floppy/dangling nodes so the deformed-
    shape plots stay well-behaved even for under-constrained disordered lattices)."""
    return C.open_stretch(geo, axis=axis, regularize=True)


def draw_deformed(ax, geo, u, nwt, facecolors, scale, title):
    sx = np.asarray(geo['simplices'])[nwt]; tv = np.asarray(geo['tri_verts'])[nwt]
    Lx, Ly = float(geo['BL1'][0]), float(geo['BL2'][1]); mg = 0.25 * max(Lx, Ly)
    dtv = tv + scale * u[sx]
    dtv[..., 0] = np.clip(dtv[..., 0], -mg, Lx + mg)                    # clip runaway nodes to the frame
    dtv[..., 1] = np.clip(dtv[..., 1], -mg, Ly + mg)
    ax.add_collection(PolyCollection(list(tv), facecolors='none', edgecolors='0.85', lw=0.2))  # undeformed
    ax.add_collection(PolyCollection(list(dtv), facecolors=facecolors[nwt], edgecolors='none', alpha=0.9))
    ax.set_xlim(-mg, Lx + mg); ax.set_ylim(-mg, Ly + mg); ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=10)


def eyy_of(geo, u, sel):
    import test_cluster_Ceff as CE
    tv = np.asarray(geo['tri_verts']); p0, p1, p2 = tv[:, 0], tv[:, 1], tv[:, 2]
    ev = np.stack([p1 - p0, p2 - p0, p2 - p1], 1)
    eps = CE.tri_metric_change(ev, np.asarray(geo['simplices']), np.eye(2), u)
    return eps[sel, 1, 1].mean(), eps[sel, 0, 0].mean()


def bar():
    prob, geo = C.make_case('disorder_hi', N)
    cen = prob.centroids; mid = cen[:, 1].mean()
    top = np.where(cen[:, 1] > mid)[0]; bot = np.where(cen[:, 1] <= mid)[0]
    objs = [C.Objective('E', 1.0, weight=1.0),
            C.Objective('nu', -0.5, region=top, weight=3.0),
            C.Objective('nu', 0.5, region=bot, weight=3.0)]
    r = C.optimize(prob, objs, mode='k', n_iter=NITER, reg=REG, verbose=False)
    C.apply_k_to_geo(geo, r['k']); C6 = C.sim_per_triangle_C6(geo)
    nt = C.c6_nuE(C.region_phys_C6(geo, C6, top))[0]; nb = C.c6_nuE(C.region_phys_C6(geo, C6, bot))[0]
    print(f"  BAR: top nu={nt:+.3f} (tgt -0.5)  bottom nu={nb:+.3f} (tgt +0.5)", flush=True)
    C.save_network(os.path.join(HERE, 'bar.npz'), geo, r['k'], C6, top=top.tolist(), bot=bot.tolist())
    u, nwt = cut_stretch(geo, axis=0)                                   # stretch along x (‖ interface)
    fc = np.where(cen[:, 1] > mid, '#1f77b4', '#d62728')                # blue=auxetic top, red=regular bottom
    scale = 0.12 * float(geo['BL1'][0])
    et = eyy_of(geo, u, cen[:, 1] > mid)[0]; eb = eyy_of(geo, u, cen[:, 1] <= mid)[0]
    fig, ax = plt.subplots(figsize=(9, 8))
    draw_deformed(ax, geo, u, nwt, fc, scale,
                  f'BAR stretched → along x (deform ×{scale:.0f})\nblue=auxetic top (ν={nt:+.2f}, '
                  f'lateral εyy={et:+.3f}) · red=regular bottom (ν={nb:+.2f}, εyy={eb:+.3f})')
    plt.tight_layout(); plt.savefig(os.path.join(HERE, 'bar.png'), dpi=150, bbox_inches='tight'); plt.close()
    print('saved bar.png')


def inclusion(disc_E, tag):
    prob, geo = C.make_case('disorder_hi', 16)
    cen = prob.centroids; ctr = cen.mean(0); rad = 0.22 * (cen[:, 0].max() - cen[:, 0].min())
    disc = prob.region_in_circle(ctr, rad); out = np.setdiff1d(np.arange(prob.n_tri), disc)
    objs = [C.Objective('nu', 0.5, region=out, weight=1.0), C.Objective('E', 1.0, region=out, weight=1.0),
            C.Objective('nu', -0.85, region=disc, weight=4.0), C.Objective('E', disc_E, region=disc, weight=3.0)]
    r = C.optimize(prob, objs, mode='k', n_iter=180, reg=REG, verbose=False)
    C.apply_k_to_geo(geo, r['k']); C6 = C.sim_per_triangle_C6(geo)
    nd, Ed = C.c6_nuE(C.region_phys_C6(geo, C6, disc)); no, Eo = C.c6_nuE(C.region_phys_C6(geo, C6, out))
    kind = 'STIFF' if disc_E > 1.5 else 'SAME-RIGIDITY'
    print(f"  INCL[{tag}]: disc nu={nd:+.3f} E={Ed:.2f} (tgt -0.85 / {disc_E})  "
          f"matrix nu={no:+.3f} E={Eo:.2f} (tgt 0.5 / 1)", flush=True)
    spec = {'kind': 'circle', 'center': tuple(map(float, ctr)), 'radius': float(rad), 'color': 'lime'}
    C.save_network(os.path.join(HERE, f'inclusion_{tag}.npz'), geo, r['k'], C6, region=spec, disc=disc.tolist())
    u, nwt = cut_stretch(geo, axis=0)
    scale = 0.12 * float(geo['BL1'][0])
    fc = np.where(np.isin(np.arange(prob.n_tri), disc), '#2ca02c', '#bbbbbb')
    fig, ax = plt.subplots(figsize=(9, 8))
    draw_deformed(ax, geo, u, nwt, fc, scale,
                  f'{kind} AUXETIC inclusion stretched → (deform ×{scale:.0f})\n'
                  f'green disc: ν={nd:+.2f}, E={Ed:.1f} · grey matrix: ν={no:+.2f}, E={Eo:.1f}')
    C.mark_region(ax, spec)
    plt.tight_layout(); plt.savefig(os.path.join(HERE, f'inclusion_{tag}.png'), dpi=150, bbox_inches='tight')
    plt.close(); print(f'saved inclusion_{tag}.png')


def main():
    os.makedirs(HERE, exist_ok=True)
    bar()
    inclusion(5.0, 'stiff')          # auxetic disc, stiff (E=5) vs matrix E=1
    inclusion(1.0, 'same')           # auxetic disc, SAME rigidity as matrix (E=1)
    print('done')


if __name__ == '__main__':
    main()
