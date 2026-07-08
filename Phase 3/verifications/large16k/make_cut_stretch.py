"""
FAST edge-forcing check: CUT the periodic patch design into an OPEN sheet and STRETCH it along the
boundary only. Clamp the left edge (u_x=0), pull the right edge (u_x=Δ), leave top/bottom and the
whole interior free, and relax as a linear spring truss (axial stiffness κ_e=k_e). This applies the
load as an EXTERNAL EDGE effect (not a bulk/periodic affine strain). We then draw the per-triangle
strain & stress and check the auxetic region's lateral response (ε_yy same sign as ε_xx ⇒ auxetic).
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

TOPOS = ['regular', 'disorder_hi']
DELTA = 1.0                        # boundary stretch (linear ⇒ relative)


def open_truss(geo):
    """Keep only non-wrapping bonds/triangles; return (nw_bond mask, nw_tri mask, edge_vecs)."""
    pts = np.asarray(geo['pts']); bu, bv, bR = geo['bond_u'], geo['bond_v'], np.asarray(geo['bond_R'])
    exp = pts[bv] - pts[bu]
    nwb = np.abs(bR - exp).max(1) < 1e-6                       # non-wrapping bonds
    nwt = nwb[geo['tri_bond']].all(1)                         # triangles with all edges non-wrapping
    tv = np.asarray(geo['tri_verts']); p0, p1, p2 = tv[:, 0], tv[:, 1], tv[:, 2]
    geo['edge_vecs'] = np.stack([p1 - p0, p2 - p0, p2 - p1], 1)
    geo['actual_len2'] = (geo['edge_vecs'] ** 2).sum(2)
    geo['simplices'] = np.asarray(geo['simplices'])
    return nwb, nwt


def assemble_open_K(geo, nwb):
    pts = np.asarray(geo['pts']); n = len(pts)
    a = geo['bond_u'][nwb]; b = geo['bond_v'][nwb]; R = np.asarray(geo['bond_R'])[nwb]
    L = np.sqrt((R ** 2).sum(1)); nx, ny = R[:, 0] / L, R[:, 1] / L; kap = np.asarray(geo['bond_k'])[nwb]
    bxx, bxy, byy = kap * nx * nx, kap * nx * ny, kap * ny * ny
    a0, a1, b0, b1 = 2 * a, 2 * a + 1, 2 * b, 2 * b + 1
    r = np.concatenate([a0, a0, a1, a1, b0, b0, b1, b1, a0, a0, a1, a1, b0, b0, b1, b1])
    c = np.concatenate([a0, a1, a0, a1, b0, b1, b0, b1, b0, b1, b0, b1, a0, a1, a0, a1])
    v = np.concatenate([bxx, bxy, bxy, byy, bxx, bxy, bxy, byy,
                        -bxx, -bxy, -bxy, -byy, -bxx, -bxy, -bxy, -byy])
    return sp.coo_matrix((v, (r, c)), shape=(2 * n, 2 * n)).tocsr()


def solve_stretch(geo, nwb):
    pts = np.asarray(geo['pts']); n = len(pts); Lx = float(geo['BL1'][0])
    K = assemble_open_K(geo, nwb)
    margin = 1.3
    left = np.where(pts[:, 0] < margin)[0]; right = np.where(pts[:, 0] > Lx - margin)[0]
    fix = np.concatenate([2 * left, 2 * right]); uf = np.concatenate([np.zeros(len(left)),
                                                                      DELTA * np.ones(len(right))])
    u = np.zeros(2 * n); u[fix] = uf
    free = np.setdiff1d(np.arange(2 * n), fix)
    rhs = -(K[free][:, fix] @ uf)
    u[free] = spla.spsolve(K[free][:, free].tocsc(), rhs)
    return u.reshape(n, 2)


def stress(bare, eps):
    A0, A1, A2, A3, A4 = (bare[:, i] for i in range(5))
    exx, eyy, exy = eps[:, 0, 0], eps[:, 1, 1], eps[:, 0, 1]; s = np.zeros_like(eps)
    s[:, 0, 0] = A0 * exx + 2 * A1 * exy + A2 * eyy
    s[:, 0, 1] = s[:, 1, 0] = A1 * exx + 2 * A2 * exy + A3 * eyy
    s[:, 1, 1] = A2 * exx + 2 * A3 * exy + A4 * eyy
    return s


def mag(t):
    return np.sqrt(t[:, 0, 0] ** 2 + 2 * t[:, 0, 1] ** 2 + t[:, 1, 1] ** 2)


def main():
    fig, axes = plt.subplots(len(TOPOS), 2, figsize=(13, 6.2 * len(TOPOS)), squeeze=False)
    for r, topo in enumerate(TOPOS):
        geo, k, C6, meta = C.load_network(os.path.join(HERE, 'networks', f'patch__{topo}.npz'))
        nwb, nwt = open_truss(geo)
        u = solve_stretch(geo, nwb)
        eps = CE.tri_metric_change(geo['edge_vecs'], geo['simplices'], np.eye(2), u)  # (N,2,2)
        sig = stress(TR.bare_tensor(geo), eps)
        cen = np.asarray(geo['centroids']); RE, RN = meta['region']
        inRE = ((cen - RE['center']) ** 2).sum(1) < RE['radius'] ** 2
        inRN = ((cen - RN['center']) ** 2).sum(1) < RN['radius'] ** 2
        for nm, sel in [('whole', nwt), ('R_E stiff', nwt & inRE), ('R_nu aux', nwt & inRN)]:
            exx = eps[sel, 0, 0].mean(); eyy = eps[sel, 1, 1].mean()
            print(f"  {topo:11s} {nm:10s}: <exx>={exx:+.4f} <eyy>={eyy:+.4f}  local nu=-eyy/exx={-eyy/exx:+.3f}"
                  f"  {'AUXETIC' if eyy > 0 else ''}", flush=True)
        gp = {'tri_verts': np.asarray(geo['tri_verts'])[nwt], 'BL1': geo['BL1'], 'BL2': geo['BL2']}
        for col, (fld, lab) in enumerate([(mag(eps)[nwt], '‖strain‖'), (mag(sig)[nwt], '‖stress‖')]):
            pc = C.fill_local_map(axes[r, col], gp, fld, cmap='magma')
            pc.set_clim(0, np.nanpercentile(fld, 98)); C.draw_box(axes[r, col], geo)
            C.mark_region(axes[r, col], meta.get('region'))
            plt.colorbar(pc, ax=axes[r, col], fraction=0.046)
            axes[r, col].set_title(f'{topo}: {lab} (open, edge-stretched →)', fontsize=10)
    fig.suptitle('CUT & STRETCH along the boundary (open sheet, left clamped / right pulled) — actual '
                 'simulation response\ncyan = stiff-E patch, lime = auxetic-ν patch', fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(HERE, 'large16k_cut_stretch.png'), dpi=140, bbox_inches='tight')
    plt.close(); print('saved large16k_cut_stretch.png')


if __name__ == '__main__':
    main()
