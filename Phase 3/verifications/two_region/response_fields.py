"""
Plot the actual simulation RESPONSE FIELDS of the two-region designs under an open x-stretch:
per-triangle axial strain ε_xx, lateral strain ε_yy (the auxetic tell: ε_yy>0 under x-stretch ⇒
expands laterally ⇒ auxetic), and stress magnitude ‖σ‖. Rows: BAR, stiff inclusion, same-rigidity
inclusion. Loaded from the saved networks (bar.npz, inclusion_{stiff,same}.npz).
"""
import os, sys
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(HERE, '..', '..', '..', 'verification_tools'))
sys.path.insert(0, os.path.join(HERE, '..', '..', '..', 'Phase 2'))
import _common as C
import test_cluster_Ceff as CE
import test_cluster_rigidity as TR

DESIGNS = [('bar', 'BAR: top auxetic / bottom regular'),
           ('inclusion_stiff', 'STIFF auxetic disc in soft matrix'),
           ('inclusion_same', 'SAME-rigidity auxetic disc')]


def nwb_of(geo):
    pts = np.asarray(geo['pts'])
    return np.abs(np.asarray(geo['bond_R']) - (pts[geo['bond_v']] - pts[geo['bond_u']])).max(1) < 1e-6


def Kmat(npts, a, b, R, kap):
    L = np.sqrt((R ** 2).sum(1)); nx, ny = R[:, 0] / L, R[:, 1] / L
    bxx, bxy, byy = kap * nx * nx, kap * nx * ny, kap * ny * ny
    a0, a1, b0, b1 = 2 * a, 2 * a + 1, 2 * b, 2 * b + 1
    r = np.concatenate([a0, a0, a1, a1, b0, b0, b1, b1, a0, a0, a1, a1, b0, b0, b1, b1])
    c = np.concatenate([a0, a1, a0, a1, b0, b1, b0, b1, b0, b1, b0, b1, a0, a1, a0, a1])
    v = np.concatenate([bxx, bxy, bxy, byy, bxx, bxy, bxy, byy,
                        -bxx, -bxy, -bxy, -byy, -bxx, -bxy, -bxy, -byy])
    return sp.coo_matrix((v, (r, c)), shape=(2 * npts, 2 * npts)).tocsr()


def cut_stretch(geo):
    nwb = nwb_of(geo); pts = np.asarray(geo['pts']); n = len(pts); m = 1.3
    nwt = nwb[geo['tri_bond']].all(1)
    K = Kmat(n, geo['bond_u'][nwb], geo['bond_v'][nwb], np.asarray(geo['bond_R'])[nwb],
             np.asarray(geo['bond_k'])[nwb]) + 1e-3 * sp.identity(2 * n)
    lo = np.where(pts[:, 0] < pts[:, 0].min() + m)[0]; hi = np.where(pts[:, 0] > pts[:, 0].max() - m)[0]
    fix = np.concatenate([2 * lo, 2 * hi]); uf = np.concatenate([np.zeros(len(lo)), np.ones(len(hi))])
    u = np.zeros(2 * n); u[fix] = uf; free = np.setdiff1d(np.arange(2 * n), fix)
    u[free] = spla.spsolve(K[free][:, free].tocsc(), -(K[free][:, fix] @ uf))
    return u.reshape(n, 2), nwt


def fields(geo, u):
    tv = np.asarray(geo['tri_verts']); p0, p1, p2 = tv[:, 0], tv[:, 1], tv[:, 2]
    ev = np.stack([p1 - p0, p2 - p0, p2 - p1], 1)
    eps = CE.tri_metric_change(ev, np.asarray(geo['simplices']), np.eye(2), u)
    bare = TR.bare_tensor({**geo, 'edge_vecs': ev, 'actual_len2': (ev ** 2).sum(2)})
    A0, A1, A2, A3, A4 = (bare[:, i] for i in range(5))
    exx, eyy, exy = eps[:, 0, 0], eps[:, 1, 1], eps[:, 0, 1]
    sxx = A0 * exx + 2 * A1 * exy + A2 * eyy; sxy = A1 * exx + 2 * A2 * exy + A3 * eyy
    syy = A2 * exx + 2 * A3 * exy + A4 * eyy
    smag = np.sqrt(sxx ** 2 + 2 * sxy ** 2 + syy ** 2)
    return exx, eyy, smag


def coarse(geo, scal, nwt, ncell=18):
    """area-weighted block-average of a per-triangle scalar over the open (nwt) domain."""
    frac = C.to_square(geo, np.asarray(geo['centroids'])); ar = np.asarray(geo['areas'])
    ix = np.clip((frac[:, 0] * ncell).astype(int), 0, ncell - 1)
    iy = np.clip((frac[:, 1] * ncell).astype(int), 0, ncell - 1)
    b = ix * ncell + iy; out = np.full(len(scal), np.nan)
    for bb in np.unique(b[nwt]):
        m = (b == bb) & nwt
        out[m] = (scal[m] * ar[m]).sum() / ar[m].sum()
    return out


def main():
    fig, axes = plt.subplots(len(DESIGNS), 3, figsize=(16, 5.2 * len(DESIGNS)), squeeze=False)
    for r, (name, title) in enumerate(DESIGNS):
        geo, k, C6, meta = C.load_network(os.path.join(HERE, f'{name}.npz'))
        u, nwt = cut_stretch(geo)
        exx, eyy, smag = fields(geo, u)
        exx, eyy, smag = (coarse(geo, f, nwt) for f in (exx, eyy, smag))   # coarse-grain (drop outliers)
        gp = {'tri_verts': np.asarray(geo['tri_verts'])[nwt], 'BL1': geo['BL1'], 'BL2': geo['BL2']}
        reg = meta.get('region')
        panels = [(exx[nwt], 'axial strain ε_xx', 'RdBu_r', True),
                  (eyy[nwt], 'lateral strain ε_yy  (>0 ⇒ expands ⇒ AUXETIC)', 'RdBu_r', True),
                  (smag[nwt], '‖stress‖', 'magma', False)]
        for col, (fld, lab, cmap, sym) in enumerate(panels):
            ax = axes[r, col]
            if sym:
                v = np.nanpercentile(np.abs(fld), 97)
                pc = C.fill_local_map(ax, gp, fld, cmap=cmap, sym=True, vlim=v)
            else:
                pc = C.fill_local_map(ax, gp, fld, cmap=cmap); pc.set_clim(0, np.nanpercentile(fld, 97))
            C.draw_box(ax, geo)
            if reg:
                C.mark_region(ax, reg)
            else:                                                       # bar: draw the interface line
                mid = float(np.asarray(geo['centroids'])[:, 1].mean())
                ax.axhline(mid, color='k', ls='--', lw=1)
            plt.colorbar(pc, ax=ax, fraction=0.046)
            if r == 0:
                ax.set_title(lab, fontsize=10)
        axes[r, 0].set_ylabel(title, fontsize=10)
    fig.suptitle('Two-region designs — actual simulation response under open x-stretch → '
                 '(coarse-grained ε_xx, ε_yy, ‖stress‖)', fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(os.path.join(HERE, 'response_fields.png'), dpi=140, bbox_inches='tight')
    plt.close(); print('saved response_fields.png')


if __name__ == '__main__':
    main()
