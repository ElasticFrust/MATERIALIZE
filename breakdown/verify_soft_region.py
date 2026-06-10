"""
Soft-region test: a small cluster of bonds (all 6 bonds around one central vertex) is made
much softer (k=0.05) than the rest of the lattice (k=1), embedded in (a) a perfect periodic
triangular lattice (eta=0) and (b) a disordered one (eta=0.15). Same bond indices are
softened in both (bond ordering depends only on lattice connectivity, not on eta).

For each network we compare the per-triangle RESPONSE -- W3(s), the linear map from the
macroscopic metric change Delta_g (3 modes: xx, yy, xy) to the triangle's local metric-change
deviation delta_g(s) -- between the PBC simulation (ground truth) and the forward solver
(ElasticSolver.forward, method='intrinsic'), driven exactly (rigidities=tri_k,
rest_lengths=reference edge length).
"""
import os, sys
import numpy as np
import scipy.sparse.linalg as spla
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

N = 16
SOFT_K = 0.05
Fk   = [np.eye(2) + CE.DELTA * M for M in CE.MODES]
Dgt  = [F.T @ F - np.eye(2) for F in Fk]
Dinv = np.linalg.inv(np.stack([CE.vec3(g) for g in Dgt], 1))
MODE_NAMES = ['xx', 'yy', 'xy']
COMP_NAMES = ['g_xx', 'g_xy', 'g_yy']


def build_case(eta, seed):
    geo = VD.build_geometry(N, eta, seed=seed)
    centre = (N // 2) * N + (N // 2)
    mask = (geo['bond_u'] == centre) | (geo['bond_v'] == centre)
    bond_k = np.ones(len(geo['bond_R']))
    bond_k[mask] = SOFT_K
    geo['bond_k'] = bond_k
    geo['tri_k'] = bond_k[geo['tri_bond']]
    return geo, mask


def sim_W3(geo):
    """Per-triangle response W3(s) (3,3): delta_g(s) = W3(s) @ Delta_g  -- PBC ground truth."""
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
    """Per-triangle response W3(s) (3,3) from ElasticSolver.forward(method='intrinsic')."""
    kkt = kkt_from_tri_bond(geo['tri_bond'], geo['edge_vecs'])
    sv = svs.make_solver(geo, kkt)
    rl = torch.as_tensor(np.sqrt(geo['actual_len2']), dtype=torch.float64)
    out = sv.forward(torch.as_tensor(geo['tri_k'], dtype=torch.float64),
                     rest_lengths=rl, method='intrinsic')
    return out['W'].detach().numpy().reshape(-1, 3, 3)


def main():
    os.makedirs(os.path.join(HERE, 'plots'), exist_ok=True)
    cases = [('perfect lattice (eta=0.0)', 0.0, 0), ('disordered (eta=0.15)', 0.15, 0)]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for row, (label, eta, seed) in enumerate(cases):
        geo, bmask = build_case(eta, seed)
        soft_tris = np.where(np.isin(geo['tri_bond'], np.where(bmask)[0]).any(axis=1))[0]
        Wsim = sim_W3(geo)
        Wint = solver_W3(geo)

        flat_s, flat_i = Wsim.reshape(-1, 9), Wint.reshape(-1, 9)
        corr = np.corrcoef(flat_s.ravel(), flat_i.ravel())[0, 1]
        rmse = np.sqrt(np.mean((flat_s - flat_i) ** 2))
        rmse_soft = np.sqrt(np.mean((flat_s[soft_tris] - flat_i[soft_tris]) ** 2))
        print(f"{label}: n_tri={len(geo['simplices'])}, n_soft_tri={len(soft_tris)}, "
              f"corr(W3 sim,intrinsic)={corr:.5f}, rmse_all={rmse:.4e}, "
              f"rmse_soft_tri={rmse_soft:.4e}", flush=True)

        # scatter, all components, all triangles
        ax = axes[row, 0]
        ax.scatter(flat_s.ravel(), flat_i.ravel(), s=4, alpha=0.3, color='#1f77b4', label='all tri')
        ax.scatter(flat_s[soft_tris].ravel(), flat_i[soft_tris].ravel(), s=14, color='#d62728',
                   label='soft-region tri', zorder=3)
        lo, hi = flat_s.min(), flat_s.max()
        ax.plot([lo, hi], [lo, hi], 'k--', lw=1)
        ax.set_xlabel('sim  W3(s)'); ax.set_ylabel('intrinsic solver  W3(s)')
        ax.set_title(f'{label}\ncorr={corr:.4f}'); ax.legend(fontsize=7); ax.grid(alpha=0.3)

        # spatial maps: dilation response (g_xx+g_yy) to mode 'xx', sim / solver / |diff|
        cen = geo['pts'][geo['simplices']].mean(axis=1)
        dil_sim = Wsim[:, 0, 0] + Wsim[:, 2, 0]
        dil_int = Wint[:, 0, 0] + Wint[:, 2, 0]
        for col, (vals, ttl) in enumerate([(dil_sim, 'sim: dilation resp. to e_xx'),
                                            (dil_int, 'intrinsic: dilation resp. to e_xx'),
                                            (dil_sim - dil_int, 'sim - intrinsic')], start=1):
            ax = axes[row, col]
            vmax = np.abs(vals).max()
            sc = ax.scatter(cen[:, 0], cen[:, 1], c=vals, cmap='RdBu_r', vmin=-vmax, vmax=vmax, s=18)
            ax.scatter(cen[soft_tris, 0], cen[soft_tris, 1], facecolors='none',
                       edgecolors='lime', s=60, lw=1.5, label='soft-region tri')
            ax.set_title(ttl, fontsize=9); ax.set_aspect('equal'); plt.colorbar(sc, ax=ax, fraction=0.046)
            if col == 1:
                ax.legend(fontsize=7)
    fig.suptitle(f'Per-triangle response W3(s) (delta_g(s)=W3(s)@Delta_g): sim vs forward '
                 f'solver (intrinsic), soft bonds k={SOFT_K} around one vertex (N={N})', fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    p = os.path.join(HERE, 'plots', 'dg_soft_region_per_triangle.png')
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print('saved', p)


if __name__ == '__main__':
    main()
