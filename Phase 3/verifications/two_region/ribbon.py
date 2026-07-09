"""
Build an auxetic patch and a regular patch SEPARATELY (each its own independent periodic design,
whole-patch objective: left ν=-0.5, right ν=+0.5, E=1), then physically GLUE them side by side
(C.glue: retriangulate the union point cloud; interface bonds default to k=1, untouched by either
optimisation -- no shared-bond mechanism exploit). Then stretch along x by clamping ONLY the
x-component of the left-most and right-most nodes; every other DOF is free, INCLUDING the y-motion
of the clamped nodes. The auxetic half should bulge OUT in y, the regular half should neck IN.
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
import response_fields as RF                  # reuse coarse-graining
import test_cluster_Ceff as CE

# each patch is its own 30x30 ORDERED (eta=0) periodic lattice, designed independently
HALF, NITER, REG = 15.0, 220, 1e-4


def strain(geo, u):
    tv = np.asarray(geo['tri_verts']); p0, p1, p2 = tv[:, 0], tv[:, 1], tv[:, 2]
    ev = np.stack([p1 - p0, p2 - p0, p2 - p1], 1)
    eps = CE.tri_metric_change(ev, np.asarray(geo['simplices']), np.eye(2), u)
    return eps[:, 0, 0], eps[:, 1, 1]          # exx, eyy per triangle


def profile(cen, val, nwt, nbin=26):
    """mean of a per-triangle scalar vs x (over the open triangles only)."""
    x = cen[:, 0]; edges = np.linspace(x.min(), x.max(), nbin + 1)
    xc = 0.5 * (edges[:-1] + edges[1:]); out = np.full(nbin, np.nan)
    for i in range(nbin):
        m = nwt & (x >= edges[i]) & (x < edges[i + 1])
        if m.any():
            out[i] = val[m].mean()
    return xc, out


def design_patch(nu_target, tag):
    geo = C.make_lattice(1.0, 1.0, half=HALF, eta=0.0, seed=0)
    prob = C.DesignProblem.from_geo(geo)
    r = C.optimize(prob, [C.Objective('E', 1.0, weight=1.0), C.Objective('nu', nu_target, weight=3.0)],
                   mode='k', n_iter=NITER, reg=REG, verbose=False)
    k = r['k'].detach().numpy()
    C.apply_k_to_geo(geo, r['k']); C6 = C.sim_per_triangle_C6(geo)
    nu, E = C.c6_nuE(C.region_phys_C6(geo, C6, None))
    print(f"  {tag} patch (solo, independent design): nu={nu:+.3f} (tgt {nu_target:+.2f}) E={E:.3f}   "
          f"k: min={k.min():.3f} mean={k.mean():.3f} frac(k<0.05)={np.mean(k < 0.05):.3f}", flush=True)
    return geo, k, nu, E


def main():
    geoL, kL, nl, El = design_patch(-0.5, 'AUXETIC (left)')
    geoR, kR, nr, Er = design_patch(0.5, 'regular (right)')
    Lx, Ly = float(geoL['BL1'][0]), float(geoL['BL2'][1])
    assert abs(Lx - float(geoR['BL1'][0])) < 1e-6 and abs(Ly - float(geoR['BL2'][1])) < 1e-6

    ptsL = np.asarray(geoL['pts']); ptsR = np.asarray(geoR['pts']) + [Lx, 0.0]
    pieces = [dict(pts_keep=ptsL, pts_full=ptsL, bond_u=geoL['bond_u'], bond_v=geoL['bond_v'],
                   bond_R=geoL['bond_R'], k=kL),
              dict(pts_keep=ptsR, pts_full=ptsR, bond_u=geoR['bond_u'], bond_v=geoR['bond_v'],
                   bond_R=geoR['bond_R'], k=kR)]
    geo, glued = C.glue(pieces, 2 * Lx, Ly)
    mid = Lx
    print(f"  GLUED: {geo['tri_bond'].shape[0]} bonds, {glued.sum()} default/interface bonds "
          f"({glued.mean()*100:.1f}%)", flush=True)
    C6 = C.sim_per_triangle_C6(geo); cen = np.asarray(geo['centroids'])
    left = np.where(cen[:, 0] < mid)[0]; right = np.where(cen[:, 0] >= mid)[0]
    nl2 = C.c6_nuE(C.region_phys_C6(geo, C6, left))[0]; nr2 = C.c6_nuE(C.region_phys_C6(geo, C6, right))[0]
    print(f"  after gluing: left nu={nl2:+.3f} (solo was {nl:+.3f})  right nu={nr2:+.3f} (solo was {nr:+.3f})",
          flush=True)
    C.save_network(os.path.join(HERE, 'ribbon.npz'), geo, geo['bond_k'], C6, mid=float(mid))

    u, nwt = C.open_stretch(geo, axis=0, regularize=True)  # clamp x only on left/right; all else free
    exx_raw, eyy_raw = strain(geo, u)
    exx, eyy = RF.coarse(geo, exx_raw, nwt, ncell=24), RF.coarse(geo, eyy_raw, nwt, ncell=24)
    el = np.nanmean(eyy[nwt & (cen[:, 0] < mid)]); er = np.nanmean(eyy[nwt & (cen[:, 0] >= mid)])
    print(f"  under x-stretch: <eyy>_left ={el:+.4f} (auxetic, expect >0)  "
          f"<eyy>_right={er:+.4f} (regular, expect <0)", flush=True)

    for scale, tag in [(2.0, ''), (5.0, '_large')]:
        draw(geo, u, nwt, cen, mid, exx, eyy, nl2, nr2, scale, tag)


def draw(geo, uc, nwt, cen, mid, exx, eyy, nl, nr, scale, tag):
    fig, (a0, a1) = plt.subplots(2, 1, figsize=(13, 9), gridspec_kw={'height_ratios': [1.35, 1]})
    Lx, Ly = float(geo['BL1'][0]), float(geo['BL2'][1]); mg = 0.55 * Ly
    sx = np.asarray(geo['simplices'])[nwt]; tv = np.asarray(geo['tri_verts'])[nwt]
    dtv = tv + scale * uc[sx]
    v = np.nanpercentile(np.abs(eyy[nwt]), 92)
    cols = plt.cm.RdBu_r(0.5 + 0.5 * np.clip(np.nan_to_num(eyy[nwt]) / v, -1, 1))
    a0.add_collection(PolyCollection(list(tv), facecolors='none', edgecolors='0.88', lw=0.12))  # undeformed
    a0.add_collection(PolyCollection(list(dtv), facecolors=cols, edgecolors='0.5', lw=0.1, alpha=0.9))
    a0.axvline(mid, color='k', ls='--', lw=1)
    a0.set_xlim(-mg, Lx + mg); a0.set_ylim(-mg, Ly + mg)
    a0.set_aspect('equal'); a0.set_xticks([]); a0.set_yticks([])
    a0.set_title(f'Macroscopic deformed shape (×{scale:.0f}) — colour = lateral strain εyy [red = expands]\n'
                 f'LEFT auxetic patch ν={nl:+.2f}: bulges OUT   ·   RIGHT regular patch ν={nr:+.2f}: necks IN',
                 fontsize=10)
    xc, ey = profile(cen, eyy, nwt); _, ex = profile(cen, exx, nwt)
    nu = np.clip(-ey / ex, -1.2, 1.2)
    a1.axhline(0, color='k', lw=.6); a1.axvline(mid, color='k', ls='--', lw=1)
    a1.axvspan(cen[:, 0].min(), mid, color='#d62728', alpha=0.06)
    a1.axvspan(mid, cen[:, 0].max(), color='#1f77b4', alpha=0.06)
    a1.plot(xc, nu, '-s', ms=4, color='#2c3e50', label='local ν(x) = -εyy/εxx (from simulation)')
    a1.plot(xc, ey, '-o', ms=3, color='#c0392b', alpha=0.6, label='lateral strain ⟨εyy⟩(x)')
    a1.axhline(-0.5, color='#d62728', ls=':', lw=1.2); a1.axhline(0.5, color='#1f77b4', ls=':', lw=1.2)
    a1.text(cen[:, 0].min() + 0.4, -0.9, 'AUXETIC patch (left)  ν→−0.5', color='#c0392b', fontsize=9)
    a1.text(mid + 0.4, 0.8, 'regular patch (right)  ν→+0.5', color='#1f77b4', fontsize=9)
    a1.set_xlabel('x  (stretch direction →)'); a1.set_ylabel('response')
    a1.set_ylim(-1.05, 1.05)
    a1.legend(fontsize=9, loc='lower right'); a1.grid(alpha=.3)
    a1.set_title('Measured response across the ribbon: local ν jumps from ≈−0.5 to ≈+0.5 at the glued '
                 'interface', fontsize=10)
    big = '  (LARGE deformation)' if tag else ''
    fig.suptitle('GLUED auxetic|regular patches — clamp x of the two ends only, everything else free' + big,
                 fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(os.path.join(HERE, f'ribbon{tag}.png'), dpi=150, bbox_inches='tight')
    plt.close(); print(f'saved ribbon{tag}.png')


if __name__ == '__main__':
    main()
