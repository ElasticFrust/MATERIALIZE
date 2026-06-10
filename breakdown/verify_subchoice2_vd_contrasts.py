"""
Finer eta scan: area-weighted chi-elimination  dA = A_s - [A]*S_s/[S_s]  vs the regular
UNWEIGHTED G&B  dA = A_s - [A]/N , both in the (A-B) form with edge + angle (curvature),
for VD rigidity contrasts alpha = -2, 2, 10.  Homogenised nu and E vs the PBC simulation.
"""
import os, sys
import numpy as np
import scipy.sparse.linalg as spla
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..', 'Phase 2'))
import test_cluster_Ceff as CE
import test_cluster_rigidity as TR
import test_cluster_VD as VD
from test_intrinsic_metric import edge_op, curv_op
from test_intrinsic_VD import Hblocks_vd, kkt_from_tri_bond

N = 10
DELTA = CE.DELTA
MODES = CE.MODES
ETAS = np.round(np.arange(0.0, 0.5001, 0.025), 4)        # finer grid (21 points)
CONTRASTS = [-2, 2, 10]


def W3_AmB(A3, S, ev, sx, n_node, kkt, weight):
    """W3 (n_tri,3,3) from (A-B)+edge+angle. weight: 'area' (S_s-weighted chi-elimination)
    or 'unweighted' (regular G&B, dA = A_s - [A]/N)."""
    n_tri = A3.shape[0]; V = S.sum()
    if weight == 'area':
        dA = A3 - (S[:, None, None] / V) * A3.sum(0)
        pref = S / V
    else:
        dA = A3 - A3.mean(0)
        pref = np.full(n_tri, 1.0 / n_tri)
    Hblk = np.zeros((3*n_tri, 3*n_tri)); Bdense = np.zeros((3*n_tri, 3*n_tri))
    dA_row = dA.transpose(1, 0, 2).reshape(3, 3*n_tri)
    for s in range(n_tri):
        Hblk[3*s:3*s+3, 3*s:3*s+3] = A3[s]
        Bdense[3*s:3*s+3, :] = pref[s] * dA_row
    AmB = Hblk - Bdense
    Cc = np.vstack([edge_op(kkt, n_tri).toarray(), curv_op(sx, ev, n_node, n_tri).toarray()])
    nC = Cc.shape[0]
    K = np.block([[AmB, Cc.T], [Cc, np.zeros((nC, nC))]])
    Dg_k = [CE.vec3((np.eye(2)+DELTA*M).T @ (np.eye(2)+DELTA*M) - np.eye(2)) for M in MODES]
    Dinv = np.linalg.inv(np.stack(Dg_k, axis=1))
    D = np.zeros((n_tri, 3, 3))
    for k, dgv in enumerate(Dg_k):
        rhs = np.concatenate([-(dA @ dgv).reshape(3*n_tri), np.zeros(nC)])
        D[:, :, k] = np.linalg.lstsq(K, rhs, rcond=None)[0][:3*n_tri].reshape(n_tri, 3)
    return D @ Dinv


def sim_W3(geo):
    ev, sx = geo['edge_vecs'], geo['simplices']; nn = len(geo['pts']); nt = len(sx)
    Fk = [np.eye(2) + DELTA*M for M in MODES]
    Dg_k = [CE.vec3(F.T@F - np.eye(2)) for F in Fk]
    Dinv = np.linalg.inv(np.stack(Dg_k, axis=1))
    K, _ = TR.assemble_K_faff(geo, np.eye(2)); free = np.arange(2, 2*nn)
    D = np.zeros((nt, 3, 3))
    for k, F in enumerate(Fk):
        fa = TR.assemble_K_faff(geo, F)[1]
        u = np.zeros(2*nn); u[free] = spla.spsolve(K[free][:, free].tocsc(), -fa[free])
        D[:, :, k] = CE.vec3(CE.tri_metric_change(ev, sx, F, u.reshape(nn, 2)) - (F.T@F-np.eye(2)))
    return D @ Dinv


def main():
    res = {a: {k: [] for k in ['nu_s', 'nu_a', 'nu_u', 'E_s', 'E_a', 'E_u']} for a in CONTRASTS}
    for a in CONTRASTS:
        print(f"\n=== VD alpha={a:+d} ===  nu / E:  sim | area-weighted | unweighted")
        for eta in ETAS:
            geo = VD.build_geometry(N, float(eta), seed=0); VD.set_VD(geo, a)
            bare = TR.bare_tensor(geo); A3 = Hblocks_vd(geo['edge_vecs'], geo['tri_k'])
            kkt = kkt_from_tri_bond(geo['tri_bond'], geo['edge_vecs'])
            args = (A3, geo['areas'], geo['edge_vecs'], geo['simplices'], len(geo['pts']), kkt)
            ns, Es = CE.Ceff_nuE(geo, sim_W3(geo), bare)
            na, Ea = CE.Ceff_nuE(geo, W3_AmB(*args, 'area'), bare)
            nu, Eu = CE.Ceff_nuE(geo, W3_AmB(*args, 'unweighted'), bare)
            for k, v in zip(res[a], [ns, na, nu, Es, Ea, Eu]):
                res[a][k].append(v)
        print(f"  (eta 0..0.5, {len(ETAS)} pts) sample eta=0.3: "
              f"nu {res[a]['nu_s'][12]:+.3f}/{res[a]['nu_a'][12]:+.3f}/{res[a]['nu_u'][12]:+.3f}", flush=True)

    nC = len(CONTRASTS)
    fig, axes = plt.subplots(nC, 2, figsize=(12, 3.2*nC), squeeze=False)
    for i, a in enumerate(CONTRASTS):
        for j, (key, ttl) in enumerate([('nu', "Poisson ratio ν"), ('E', "Young's modulus E")]):
            ax = axes[i, j]
            ax.plot(ETAS, res[a][f'{key}_s'], 'k-', lw=2.6, label='sim (truth)', zorder=5)
            ax.plot(ETAS, res[a][f'{key}_a'], '-', color='#1f77b4', lw=1.8, label='area-weighted  A_s−[A]·S_s/[S_s]')
            ax.plot(ETAS, res[a][f'{key}_u'], '--', color='#d62728', lw=1.8, label='unweighted  A_s−[A]/N')
            if key == 'nu':
                ax.axhline(0, color='gray', lw=0.5, ls=':')
            ax.set_ylabel(f'{ttl}\n(α={a:+d})'); ax.grid(alpha=0.3)
            if i == 0 and j == 0:
                ax.legend(fontsize=8)
            if i == nC-1:
                ax.set_xlabel('η')
    fig.suptitle('VD contrasts α=−2,+2,+10:  area-weighted vs unweighted χ-elimination (edge+angle) vs sim',
                 fontsize=12)
    plt.tight_layout()
    p = os.path.join(HERE, 'plots', 'dg_subchoice2_vd_contrasts.png')
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print('\nsaved', p)


if __name__ == '__main__':
    main()
