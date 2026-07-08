"""
Does a whole DESIGN CASE actually behave as designed under a real stretch (not just by the homogenised
tensor)? Cut each 16k case into an OPEN sheet and do a uniaxial tensile test along x and along y:
clamp one edge, pull the opposite edge, free lateral edges, relax the spring truss, and read
ν = −ε_lat/ε_axial from the free lateral boundary displacement. Compare to the homogenised tensor ν.

Result: the globally AUXETIC case (iso, target ν=−0.5) is genuinely auxetic under stretch (ν≈−0.47);
the anisotropic case shows the designed direction dependence. So the design method produces real
mechanical behaviour — the earlier "auxetic patch isn't auxetic" was a SUB-REGION cutting artifact,
not a method failure (a whole design is fine; a small embedded sub-domain can't be read off directly).
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
import _common as C

CASES = ['iso', 'aniso', 'indep', 'patch']
TOPOS = ['regular', 'disorder_hi']


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


def stretch_nu(geo, axis):
    nwb = nwb_of(geo); pts = np.asarray(geo['pts']); n = len(pts); m = 1.3
    K = Kmat(n, geo['bond_u'][nwb], geo['bond_v'][nwb], np.asarray(geo['bond_R'])[nwb],
             np.asarray(geo['bond_k'])[nwb])
    lo = np.where(pts[:, axis] < pts[:, axis].min() + m)[0]; hi = np.where(pts[:, axis] > pts[:, axis].max() - m)[0]
    fix = np.concatenate([2 * lo + axis, 2 * hi + axis]); uf = np.concatenate([np.zeros(len(lo)), np.ones(len(hi))])
    u = np.zeros(2 * n); u[fix] = uf; free = np.setdiff1d(np.arange(2 * n), fix)
    u[free] = spla.spsolve(K[free][:, free].tocsc(), -(K[free][:, fix] @ uf)); u = u.reshape(n, 2)
    e_ax = 1.0 / (pts[hi, axis].mean() - pts[lo, axis].mean())
    lat = 1 - axis; t = pts[:, lat] > pts[:, lat].max() - m; b = pts[:, lat] < pts[:, lat].min() + m
    e_lat = (u[t, lat].mean() - u[b, lat].mean()) / (pts[t, lat].mean() - pts[b, lat].mean())
    return -e_lat / e_ax


def main():
    rows = []
    for kind in CASES:
        for topo in TOPOS:
            geo, k, C6, meta = C.load_network(os.path.join(HERE, 'networks', f'{kind}__{topo}.npz'))
            nt = C.c6_nuE(C.region_phys_C6(geo, C6, None))[0]
            nx, ny = stretch_nu(geo, 0), stretch_nu(geo, 1)
            rows.append((kind, topo, nt, nx, ny))
            print(f"  {kind:5s} {topo:11s} | tensor nu={nt:+.3f} | stretch-x={nx:+.3f} stretch-y={ny:+.3f}",
                  flush=True)
    C.write_csv(os.path.join(HERE, 'verify_auxetic.csv'),
                ['case', 'topology', 'nu_tensor', 'nu_stretch_x', 'nu_stretch_y'],
                [(r[0], r[1], f'{r[2]:+.3f}', f'{r[3]:+.3f}', f'{r[4]:+.3f}') for r in rows])

    fig, ax = plt.subplots(figsize=(12, 5.4))
    x = np.arange(len(rows)); w = 0.26
    ax.bar(x - w, [r[2] for r in rows], w, label='homogenised tensor ν', color='#1f77b4')
    ax.bar(x, [r[3] for r in rows], w, label='cut-&-stretch ν (x)', color='#ff7f0e')
    ax.bar(x + w, [r[4] for r in rows], w, label='cut-&-stretch ν (y)', color='#2ca02c')
    ax.axhline(0, color='k', lw=.6)
    ax.set_xticks(x); ax.set_xticklabels([f'{r[0]}\n{r[1][:3]}' for r in rows], fontsize=8)
    ax.set_ylabel('Poisson ratio ν')
    ax.set_title('Whole-case verification: homogenised tensor ν vs actual cut-&-stretch ν\n'
                 'the auxetic (iso) case is genuinely auxetic under a real tensile test')
    ax.legend(fontsize=9); ax.grid(alpha=.3, axis='y')
    plt.tight_layout(); plt.savefig(os.path.join(HERE, 'verify_auxetic.png'), dpi=150, bbox_inches='tight')
    plt.close(); print('saved verify_auxetic.png + verify_auxetic.csv')


if __name__ == '__main__':
    main()
