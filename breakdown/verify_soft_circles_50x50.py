"""
50x50, two circular soft regions in the middle, three different network structures:
  (a) perfect periodic triangular lattice (eta=0.0)
  (b) disordered periodic triangular lattice (eta=0.3)
  (c) periodic Delaunay triangulation of a Poisson point process with the same number of
      points (N*N=2500) as the lattice (same mean density: unit-cell area sqrt(3)/2).

In each, all bonds whose midpoint falls within either of two circles (radius r, centred in
the box) are softened to k=SOFT_K; all other bonds k=1. (No eta-disorder *sweep* and no
tanh-VD contrast -- a fixed local soft-inclusion pattern on three structurally different
periodic networks.)

For each network we compare the per-triangle response W3(s) (delta_g(s)=W3(s)@Delta_g for
the 3 macro modes xx/yy/xy) between the PBC simulation (ground truth) and
ElasticSolver.forward(method='intrinsic'), driven exactly (rigidities=tri_k,
rest_lengths=reference edge length).
"""
import os, sys, time
import numpy as np
import scipy.sparse.linalg as spla
from scipy.spatial import Delaunay
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..', 'Phase 2'))
import test_cluster_VD as VD
import test_cluster_rigidity as TR
import test_cluster_Ceff as CE
from test_intrinsic_VD import kkt_from_tri_bond
import verify_solver_sweep as svs       # make_solver, _mount
torch.set_default_dtype(torch.float64)

N = 50
SOFT_K = 0.05
CIRCLE_FRAC_DX = 0.15     # circle-centre offset from box centre, as a fraction of box width
CIRCLE_FRAC_R = 0.05      # circle radius, as a fraction of box width
Fk   = [np.eye(2) + CE.DELTA * M for M in CE.MODES]
Dgt  = [F.T @ F - np.eye(2) for F in Fk]
Dinv = np.linalg.inv(np.stack([CE.vec3(g) for g in Dgt], 1))


def build_poisson_geometry(N, seed):
    """Periodic Delaunay triangulation of N*N Poisson points (3x3 tiling trick)."""
    rng = np.random.default_rng(seed)
    n_points = N * N
    Lx, Ly = float(N), N * np.sqrt(3) / 2     # same box / density as the lattice
    pts = rng.uniform([0, 0], [Lx, Ly], size=(n_points, 2))
    box = np.array([Lx, Ly])
    shifts = np.array([(i, j) for i in (-1, 0, 1) for j in (-1, 0, 1)])
    tiled = np.concatenate([pts + s * box for s in shifts], axis=0)
    shift_of = np.repeat(shifts, n_points, axis=0)

    simp_t = Delaunay(tiled).simplices
    cen = tiled[simp_t].mean(axis=1)
    keep = (cen[:, 0] >= 0) & (cen[:, 0] < Lx) & (cen[:, 1] >= 0) & (cen[:, 1] < Ly)
    simp_t = simp_t[keep]
    n_tri = len(simp_t)
    canon = simp_t % n_points
    sft = shift_of[simp_t]

    p0, p1, p2 = tiled[simp_t[:, 0]], tiled[simp_t[:, 1]], tiled[simp_t[:, 2]]
    edge_vecs = np.stack([p1 - p0, p2 - p0, p2 - p1], 1)
    l2 = (edge_vecs ** 2).sum(2)
    areas = 0.5 * np.abs(edge_vecs[:, 0, 0] * edge_vecs[:, 1, 1]
                          - edge_vecs[:, 0, 1] * edge_vecs[:, 1, 0])

    pairs = [(0, 1, 0), (0, 2, 1), (1, 2, 2)]
    keymap = {}; bonds = []; tri_bond = np.zeros((n_tri, 3), np.int64)
    for ti in range(n_tri):
        for ka, kb, ei in pairs:
            ca, cb = int(canon[ti, ka]), int(canon[ti, kb])
            d = sft[ti, kb] - sft[ti, ka]; dp = (int(d[0]), int(d[1]))
            R = edge_vecs[ti, ei]
            if (ca, dp[0], dp[1]) <= (cb, -dp[0], -dp[1]):
                key, Rk = (ca, cb, dp[0], dp[1]), R
            else:
                key, Rk = (cb, ca, -dp[0], -dp[1]), -R
            if key not in keymap:
                keymap[key] = len(bonds); bonds.append((key[0], key[1], Rk))
            tri_bond[ti, ei] = keymap[key]

    return dict(N=N, pts=pts, simplices=canon, edge_vecs=edge_vecs, actual_len2=l2,
                 bond_u=np.array([b[0] for b in bonds], np.int64),
                 bond_v=np.array([b[1] for b in bonds], np.int64),
                 bond_R=np.array([b[2] for b in bonds], float),
                 tri_bond=tri_bond, areas=areas)


def soft_circles_mask(geo):
    """Two circular soft regions centred (symmetrically) in the bond-midpoint cloud."""
    mid = geo['pts'][geo['bond_u']] + geo['bond_R'] / 2
    xr = geo['pts'][:, 0].max() - geo['pts'][:, 0].min()
    yr = geo['pts'][:, 1].max() - geo['pts'][:, 1].min()
    cx = (geo['pts'][:, 0].max() + geo['pts'][:, 0].min()) / 2
    cy = (geo['pts'][:, 1].max() + geo['pts'][:, 1].min()) / 2
    dx, r = CIRCLE_FRAC_DX * xr, CIRCLE_FRAC_R * xr
    centres = [(cx - dx, cy), (cx + dx, cy)]
    mask = np.zeros(len(mid), bool)
    for c in centres:
        mask |= ((mid[:, 0] - c[0]) ** 2 + (mid[:, 1] - c[1]) ** 2) < r ** 2
    return mask, centres, r


def sim_W3(geo):
    ev, sx = geo['edge_vecs'], geo['simplices']; nn = len(geo['pts']); nt = len(sx)
    K, _ = TR.assemble_K_faff(geo, np.eye(2)); free = np.arange(2, 2 * nn)
    Kff = K[free][:, free].tocsc()
    D = np.zeros((nt, 3, 3))
    for k, F in enumerate(Fk):
        fa = TR.assemble_K_faff(geo, F)[1]
        u = np.zeros(2 * nn); u[free] = spla.spsolve(Kff, -fa[free])
        D[:, :, k] = CE.vec3(CE.tri_metric_change(ev, sx, F, u.reshape(nn, 2)) - Dgt[k])
    return D @ Dinv


def solver_W3(geo):
    kkt = kkt_from_tri_bond(geo['tri_bond'], geo['edge_vecs'])
    sv = svs.make_solver(geo, kkt)
    rl = torch.as_tensor(np.sqrt(geo['actual_len2']), dtype=torch.float64)
    out = sv.forward(torch.as_tensor(geo['tri_k'], dtype=torch.float64),
                     rest_lengths=rl, method='intrinsic')
    return out['W'].detach().numpy().reshape(-1, 3, 3)


def main():
    os.makedirs(os.path.join(HERE, 'plots'), exist_ok=True)
    cases = [
        ('perfect lattice (eta=0.0)', lambda: VD.build_geometry(N, 0.0, seed=0)),
        ('disordered lattice (eta=0.3)', lambda: VD.build_geometry(N, 0.3, seed=0)),
        ('Poisson Delaunay (N*N pts)', lambda: build_poisson_geometry(N, seed=0)),
    ]

    fig, axes = plt.subplots(3, 4, figsize=(18, 12))
    for row, (label, builder) in enumerate(cases):
        t0 = time.time()
        geo = builder()
        mask, centres, r = soft_circles_mask(geo)
        bond_k = np.ones(len(geo['bond_R'])); bond_k[mask] = SOFT_K
        geo['bond_k'] = bond_k; geo['tri_k'] = bond_k[geo['tri_bond']]
        soft_tris = np.where(np.isin(geo['tri_bond'], np.where(mask)[0]).any(axis=1))[0]

        Wsim = sim_W3(geo)
        Wint = solver_W3(geo)
        flat_s, flat_i = Wsim.reshape(-1, 9), Wint.reshape(-1, 9)
        corr = np.corrcoef(flat_s.ravel(), flat_i.ravel())[0, 1]
        rmse = np.sqrt(np.mean((flat_s - flat_i) ** 2))
        rmse_soft = np.sqrt(np.mean((flat_s[soft_tris] - flat_i[soft_tris]) ** 2))
        print(f"{label}: n_tri={len(geo['simplices'])}, n_bonds={len(geo['bond_R'])}, "
              f"n_soft_tri={len(soft_tris)}, corr={corr:.5f}, rmse_all={rmse:.4e}, "
              f"rmse_soft_tri={rmse_soft:.4e}  [{time.time()-t0:.1f}s]", flush=True)

        ax = axes[row, 0]
        ax.scatter(flat_s.ravel(), flat_i.ravel(), s=2, alpha=0.2, color='#1f77b4', label='all tri')
        ax.scatter(flat_s[soft_tris].ravel(), flat_i[soft_tris].ravel(), s=10, color='#d62728',
                   label='soft-region tri', zorder=3)
        lo, hi = flat_s.min(), flat_s.max()
        ax.plot([lo, hi], [lo, hi], 'k--', lw=1)
        ax.set_xlabel('sim  W3(s)'); ax.set_ylabel('intrinsic solver  W3(s)')
        ax.set_title(f'{label}\ncorr={corr:.4f}'); ax.legend(fontsize=7); ax.grid(alpha=0.3)

        cen = geo['pts'][geo['simplices']].mean(axis=1)
        dil_sim = Wsim[:, 0, 0] + Wsim[:, 2, 0]
        dil_int = Wint[:, 0, 0] + Wint[:, 2, 0]
        for col, (vals, ttl) in enumerate([(dil_sim, 'sim: dilation resp. to e_xx'),
                                            (dil_int, 'intrinsic: dilation resp. to e_xx'),
                                            (dil_sim - dil_int, 'sim - intrinsic')], start=1):
            ax = axes[row, col]
            vmax = np.abs(vals).max()
            sc = ax.scatter(cen[:, 0], cen[:, 1], c=vals, cmap='RdBu_r', vmin=-vmax, vmax=vmax, s=6)
            for c in centres:
                ax.add_patch(plt.Circle(c, r, facecolor='none', edgecolor='lime', lw=1.5))
            ax.set_title(ttl, fontsize=9); ax.set_aspect('equal'); plt.colorbar(sc, ax=ax, fraction=0.046)
    fig.suptitle(f'Per-triangle response W3(s) (delta_g(s)=W3(s)@Delta_g): sim vs forward '
                 f'solver (intrinsic), two soft circles (k={SOFT_K}, r={CIRCLE_FRAC_R}*Lx), '
                 f'N={N}', fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    p = os.path.join(HERE, 'plots', 'dg_soft_circles_50x50.png')
    plt.savefig(p, dpi=130, bbox_inches='tight'); plt.close()
    print('saved', p)


if __name__ == '__main__':
    main()
