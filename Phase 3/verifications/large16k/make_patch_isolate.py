"""
Two clean ways to measure a patch's Poisson ratio under stretching, vs the intrinsic tensor value:
  (1) COARSE regional ν: from the full open-sheet edge stretch, block-average the strain (kills the
      floppy-bond outliers), then region-average ⟨ε⟩ → ν = −⟨ε_yy⟩/⟨ε_xx⟩.
  (2) ISOLATE & stretch: cut out just the region's disc as its own open specimen, clamp its left edge,
      pull its right edge, and measure ν from the free top/bottom lateral displacement (a real tensile
      test of that patch alone).
Compared to the intrinsic homogenised-tensor ν (Method A). Auxetic ⇒ ν<0.
"""
import os, sys
import numpy as np
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
NCELL = 22


def stretch_x(pts, K, margin):
    """clamp left (u_x=0), pull right (u_x=Δ), top/bottom & interior free; return u (n,2), gauge W."""
    n = len(pts); xmin, xmax = pts[:, 0].min(), pts[:, 0].max()
    left = np.where(pts[:, 0] < xmin + margin)[0]; right = np.where(pts[:, 0] > xmax - margin)[0]
    fix = np.concatenate([2 * left, 2 * right]); uf = np.concatenate([np.zeros(len(left)), np.ones(len(right))])
    u = np.zeros(2 * n); u[fix] = uf; free = np.setdiff1d(np.arange(2 * n), fix)
    u[free] = spla.spsolve(K[free][:, free].tocsc(), -(K[free][:, fix] @ uf))
    W = pts[right, 0].mean() - pts[left, 0].mean()
    return u.reshape(n, 2), W, left, right


def coarse(geo, tens):
    frac = C.to_square(geo, np.asarray(geo['centroids'])); ar = np.asarray(geo['areas'])
    ix = np.clip((frac[:, 0] * NCELL).astype(int), 0, NCELL - 1)
    iy = np.clip((frac[:, 1] * NCELL).astype(int), 0, NCELL - 1)
    b = ix * NCELL + iy; out = np.zeros_like(tens)
    for bb in np.unique(b):
        m = b == bb; out[m] = (tens[m] * ar[m, None, None]).sum(0) / ar[m].sum()
    return out


def poisson_from_edges(pts, u, H_axis=1):
    """ν from a tensile test: ε_xx from the clamp gauge (set outside); here return lateral ε_yy/height."""
    top = pts[:, 1] > pts[:, 1].max() - 1.3; bot = pts[:, 1] < pts[:, 1].min() + 1.3
    H = pts[top, 1].mean() - pts[bot, 1].mean()
    return (u[top, 1].mean() - u[bot, 1].mean()) / H


def isolate_nu(geo, spec, nwb):
    cen = np.asarray(geo['centroids']); sx = geo['simplices']
    tri = np.where(((cen - spec['center']) ** 2).sum(1) < spec['radius'] ** 2)[0]
    nodes = np.unique(sx[tri]); inn = np.zeros(len(geo['pts']), bool); inn[nodes] = True
    bu, bv = geo['bond_u'], geo['bond_v']
    bsel = nwb & inn[bu] & inn[bv]
    remap = -np.ones(len(geo['pts']), int); remap[nodes] = np.arange(len(nodes))
    pts = np.asarray(geo['pts'])[nodes]
    a = remap[bu[bsel]]; b = remap[bv[bsel]]; R = np.asarray(geo['bond_R'])[bsel]; kap = np.asarray(geo['bond_k'])[bsel]
    K = C.spring_K(len(nodes), a, b, R, kap)
    u, W, left, right = stretch_x(pts, K, margin=1.3)
    exx = 1.0 / W; eyy = poisson_from_edges(pts, u)
    return -eyy / exx, len(tri)


def main():
    rows = []
    for topo in TOPOS:
        geo, k, C6, meta = C.load_network(os.path.join(HERE, 'networks', f'patch__{topo}.npz'))
        nwb, _ = C.nonwrap_mask(geo); RE, RN = meta['region']
        # full-sheet edge stretch -> coarse regional nu
        u, _ = C.open_stretch(geo, axis=0)
        eps = CE.tri_metric_change(geo['edge_vecs'], geo['simplices'], np.eye(2), u)
        epsC = coarse(geo, eps); cen = np.asarray(geo['centroids'])
        for name, spec in [('R_E stiff', RE), ('R_nu aux', RN)]:
            idx = np.where(((cen - spec['center']) ** 2).sum(1) < spec['radius'] ** 2)[0]
            exx, eyy = epsC[idx, 0, 0].mean(), epsC[idx, 1, 1].mean()
            nu_coarse = -eyy / exx
            nu_iso, ntri = isolate_nu(geo, spec, nwb)
            nu_tensor = C.c6_nuE(C.region_phys_C6(geo, C6, idx))[0]
            rows.append((topo, name, f'{nu_tensor:+.3f}', f'{nu_coarse:+.3f}', f'{nu_iso:+.3f}'))
            print(f"  {topo:11s} {name:10s} | intrinsic(tensor)={nu_tensor:+.3f}  "
                  f"coarse-regional={nu_coarse:+.3f}  ISOLATED-specimen={nu_iso:+.3f}", flush=True)
    C.write_csv(os.path.join(HERE, 'patch_isolate.csv'),
                ['topology', 'region', 'nu_intrinsic_tensor', 'nu_coarse_edgestretch', 'nu_isolated'], rows)

    fig, ax = plt.subplots(figsize=(9, 5.2)); labels = [f'{t[:3]}\n{n}' for t, n, *_ in rows]
    x = np.arange(len(rows)); w = 0.26
    ax.bar(x - w, [float(r[2]) for r in rows], w, label='intrinsic (tensor)', color='#1f77b4')
    ax.bar(x, [float(r[3]) for r in rows], w, label='coarse regional (edge stretch)', color='#ff7f0e')
    ax.bar(x + w, [float(r[4]) for r in rows], w, label='isolated specimen', color='#2ca02c')
    ax.axhline(0, color='k', lw=.6); ax.axhline(-0.3, color='gray', ls=':', label='R_ν target −0.3')
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel('Poisson ratio ν')
    ax.set_title('Patch ν three ways: intrinsic tensor vs coarse-regional-under-edge-stretch vs isolated specimen')
    ax.legend(fontsize=8); ax.grid(alpha=.3, axis='y')
    plt.tight_layout(); plt.savefig(os.path.join(HERE, 'patch_isolate.png'), dpi=150, bbox_inches='tight')
    plt.close(); print('saved patch_isolate.png + patch_isolate.csv')


if __name__ == '__main__':
    main()
