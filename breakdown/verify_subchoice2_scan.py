"""
Scan verification of the area-weighted chi-elimination  dA(s) = A_s - [A]*S_s/[S_s]
in the (A-B) Woodbury form, across eta, for TWO cases (geometric disorder; VD rigidity
contrast), each in THREE constraint configurations:
    none  : (A-B) only (chi eliminated; area-normalisation implicit) = area-weighted single-site MF
    edge  : + edge-length KKT
    all   : + edge + angle/curvature (zero discrete Gaussian curvature)
Reports per-triangle dg corr / overshoot vs the PBC simulation (single uniaxial mode).
"""
import os, sys
import numpy as np
import scipy.sparse.linalg as spla
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..', 'Phase 2'))
import pbc_dg_analysis as pda
import test_cluster_Ceff as CE
import test_cluster_rigidity as TR
import test_cluster_VD as VD
from test_mean_isolation import Hblocks
from test_intrinsic_metric import edge_op, curv_op
from test_intrinsic_VD import Hblocks_vd, kkt_from_tri_bond

N = 10
DELTA = 1e-3
H = np.array([[1.0, 0.0], [0.0, 0.0]])
ETAS = [0.1, 0.2, 0.3, 0.4, 0.5]
VD_CONTRAST = 10


def solve_AmB(A3, S, ev, sx, n_node, kkt, config, F):
    """Solve (A-B) dg + constraints = -dA.Dg with user's dA = A_s - [A] S_s/[S_s]."""
    n_tri = A3.shape[0]; V = S.sum()
    Asum = A3.sum(0)
    dA = A3 - (S[:, None, None] / V) * Asum                       # user's area-scaled deviation
    # (B dg)(s) = (S_s/V) sum_s' dA(s') dg(s')  -> dense (3N,3N)
    Bdense = np.zeros((3*n_tri, 3*n_tri)); Hblk = np.zeros((3*n_tri, 3*n_tri))
    dA_row = dA.transpose(1, 0, 2).reshape(3, 3*n_tri)            # [comp_s, (s'*3+comp')]
    for s in range(n_tri):
        Bdense[3*s:3*s+3, :] = (S[s]/V) * dA_row
        Hblk[3*s:3*s+3, 3*s:3*s+3] = A3[s]
    AmB = Hblk - Bdense
    Dgv = CE.vec3(pda.macroscopic_dg(F))
    rhs_top = -(dA @ Dgv).reshape(3*n_tri)
    if config == 'none':
        dg = np.linalg.lstsq(AmB, rhs_top, rcond=None)[0].reshape(n_tri, 3)
        return dg
    blocks = [edge_op(kkt, n_tri).toarray()]
    if config == 'all':
        blocks.append(curv_op(sx, ev, n_node, n_tri).toarray())
    Cc = np.vstack(blocks); nC = Cc.shape[0]
    K = np.block([[AmB, Cc.T], [Cc, np.zeros((nC, nC))]])
    rhs = np.concatenate([rhs_top, np.zeros(nC)])
    sol = np.linalg.lstsq(K, rhs, rcond=None)[0]
    return sol[:3*n_tri].reshape(n_tri, 3)


def metrics(dg, simv):
    corr = np.corrcoef(dg.ravel(), simv.ravel())[0, 1]
    over = np.median(np.linalg.norm(dg, axis=1) / np.maximum(np.linalg.norm(simv, axis=1), 1e-30))
    return corr, over


def geometric(eta):
    mesh = pda.build_periodic_tf_mesh(N, eta, seed=0)
    ev, sx = mesh['edge_vecs'], mesh['simplices']; nn = len(mesh['pts'])
    A3 = Hblocks(ev); S = mesh['areas']; kkt = mesh['kkt_arrays']
    F = np.eye(2) + DELTA*H
    K, faff = pda._assemble_K_and_faff(mesh, F); free = np.arange(2, 2*nn)
    u = np.zeros(2*nn); u[free] = spla.spsolve(K[free][:, free].tocsc(), -faff[free])
    sim = pda.triangle_metric_change(ev, sx, F, u.reshape(nn, 2)) - pda.macroscopic_dg(F)
    return A3, S, ev, sx, nn, kkt, F, CE.vec3(sim)


def vd(eta, a):
    geo = VD.build_geometry(N, eta, seed=0); VD.set_VD(geo, a)
    ev, sx = geo['edge_vecs'], geo['simplices']; nn = len(geo['pts'])
    A3 = Hblocks_vd(ev, geo['tri_k']); S = geo['areas']
    kkt = kkt_from_tri_bond(geo['tri_bond'], ev)
    F = np.eye(2) + DELTA*H
    K, faff = TR.assemble_K_faff(geo, F); free = np.arange(2, 2*nn)
    u = np.zeros(2*nn); u[free] = spla.spsolve(K[free][:, free].tocsc(), -faff[free])
    sim = CE.tri_metric_change(ev, sx, F, u.reshape(nn, 2)) - (F.T@F - np.eye(2))
    return A3, S, ev, sx, nn, kkt, F, CE.vec3(sim)


def run(name, builder):
    print(f"\n=== {name} ===  per-triangle dg corr / overshoot vs sim  (dA = A_s - [A] S_s/[S_s])")
    print(f"{'eta':>5} | {'none corr':>9} {'over':>5} | {'edge corr':>9} {'over':>5} | {'all corr':>9} {'over':>5}")
    res = {c: {'corr': [], 'over': []} for c in ('none', 'edge', 'all')}
    for eta in ETAS:
        A3, S, ev, sx, nn, kkt, F, simv = builder(eta)
        row = []
        for cfg in ('none', 'edge', 'all'):
            dg = solve_AmB(A3, S, ev, sx, nn, kkt, cfg, F)
            c, o = metrics(dg, simv)
            res[cfg]['corr'].append(c); res[cfg]['over'].append(o); row.append((c, o))
        print(f"{eta:>5.1f} | {row[0][0]:>9.4f} {row[0][1]:>5.2f} | {row[1][0]:>9.4f} {row[1][1]:>5.2f} | "
              f"{row[2][0]:>9.4f} {row[2][1]:>5.2f}", flush=True)
    return res


def main():
    cases = [('GEOMETRIC disorder (k=1)', geometric),
             (f'VD rigidity contrast a={VD_CONTRAST}', lambda e: vd(e, VD_CONTRAST))]
    results = [(name, run(name, b)) for name, b in cases]

    style = {'none': ('#d62728', '-s', 'none (area-wtd MF)'),
             'edge': ('#ff7f0e', '-^', 'edge KKT'),
             'all':  ('#1f77b4', '-o', 'edge + angle (all)')}
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), squeeze=False)
    for i, (name, res) in enumerate(results):
        for j, (key, ttl, lo, hi) in enumerate([('corr', 'per-triangle δg corr vs sim', -0.2, 1.05),
                                                ('over', 'overshoot  ‖δg‖/‖δg_sim‖', 0.9, 1.35)]):
            ax = axes[i, j]
            for cfg in ('none', 'edge', 'all'):
                col, mk, lab = style[cfg]
                ax.plot(ETAS, res[cfg][key], mk, color=col, ms=5, label=lab)
            if key == 'corr':
                ax.axhline(1.0, color='gray', lw=0.6, ls=':')
            else:
                ax.axhline(1.0, color='gray', lw=0.6, ls=':')
            ax.set_ylim(lo, hi); ax.set_xlabel('η'); ax.set_title(f'{name}\n{ttl}', fontsize=10)
            ax.grid(alpha=0.3)
            if i == 0 and j == 0:
                ax.legend(fontsize=8, loc='lower left')
    fig.suptitle('Area-weighted χ-elimination  δA = A_s − [A]·S_s/[S_s]  (the (A−B) form) vs simulation',
                 fontsize=12)
    plt.tight_layout()
    p = os.path.join(HERE, 'plots', 'dg_subchoice2_scan.png')
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print('\nsaved', p)


if __name__ == '__main__':
    main()
