"""
Homogenised nu and E (Young's) vs eta for the area-weighted chi-elimination
    dA(s) = A_s - [A]*S_s/[S_s]   in the (A-B) form with edge + angle (curvature),
for GEOMETRIC disorder and VD rigidity contrast, vs the PBC simulation and single-site MF.
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
import test_cluster_Ceff_rigidity as RG
import test_cluster_VD as VD
from test_mean_isolation import Hblocks
from test_intrinsic_metric import edge_op, curv_op
from test_intrinsic_VD import Hblocks_vd, kkt_from_tri_bond

N = 10
DELTA = CE.DELTA
MODES = CE.MODES
ETAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
VD_CONTRAST = 10


def W3_AmB(A3, S, ev, sx, n_node, kkt):
    """Loading-independent W3 (n_tri,3,3) from (A-B)+edge+angle with dA = A_s - [A] S_s/[S_s]."""
    n_tri = A3.shape[0]; V = S.sum()
    dA = A3 - (S[:, None, None] / V) * A3.sum(0)
    Hblk = np.zeros((3*n_tri, 3*n_tri)); Bdense = np.zeros((3*n_tri, 3*n_tri))
    dA_row = dA.transpose(1, 0, 2).reshape(3, 3*n_tri)
    for s in range(n_tri):
        Hblk[3*s:3*s+3, 3*s:3*s+3] = A3[s]
        Bdense[3*s:3*s+3, :] = (S[s]/V) * dA_row
    AmB = Hblk - Bdense
    Cc = np.vstack([edge_op(kkt, n_tri).toarray(), curv_op(sx, ev, n_node, n_tri).toarray()])
    nC = Cc.shape[0]
    K = np.block([[AmB, Cc.T], [Cc, np.zeros((nC, nC))]])
    Fk = [np.eye(2) + DELTA*M for M in MODES]
    Dg_k = [CE.vec3(F.T@F - np.eye(2)) for F in Fk]
    Dinv = np.linalg.inv(np.stack(Dg_k, axis=1))
    D = np.zeros((n_tri, 3, 3))
    for k, dgv in enumerate(Dg_k):
        rhs = np.concatenate([-(dA @ dgv).reshape(3*n_tri), np.zeros(nC)])
        sol = np.linalg.lstsq(K, rhs, rcond=None)[0]
        D[:, :, k] = sol[:3*n_tri].reshape(n_tri, 3)
    return D @ Dinv


def sim_W3(mesh, A_assemble, tmc, dgt_full):
    ev, sx = mesh['edge_vecs'], mesh['simplices']; nn = len(mesh['pts']); nt = len(sx)
    Fk = [np.eye(2) + DELTA*M for M in MODES]
    Dg_k = [CE.vec3(F.T@F - np.eye(2)) for F in Fk]
    Dinv = np.linalg.inv(np.stack(Dg_k, axis=1))
    K, _ = A_assemble(mesh, np.eye(2)); free = np.arange(2, 2*nn)
    D = np.zeros((nt, 3, 3))
    for k, F in enumerate(Fk):
        fa = A_assemble(mesh, F)[1]
        u = np.zeros(2*nn); u[free] = spla.spsolve(K[free][:, free].tocsc(), -fa[free])
        D[:, :, k] = CE.vec3(tmc(ev, sx, F, u.reshape(nn, 2)) - (F.T@F-np.eye(2)))
    return D @ Dinv


def geometric(eta):
    mesh = pda.build_periodic_tf_mesh(N, eta, seed=0)
    bare = CE.bare_tensor(mesh); A3 = Hblocks(mesh['edge_vecs'])
    w3_s = sim_W3(mesh, pda._assemble_K_and_faff, pda.triangle_metric_change, None)
    w3_a = W3_AmB(A3, mesh['areas'], mesh['edge_vecs'], mesh['simplices'], len(mesh['pts']), mesh['kkt_arrays'])
    w3_m = RG.mf_W3(bare)
    return mesh, bare, w3_s, w3_a, w3_m


def vd(eta, a):
    geo = VD.build_geometry(N, eta, seed=0); VD.set_VD(geo, a)
    bare = TR.bare_tensor(geo); A3 = Hblocks_vd(geo['edge_vecs'], geo['tri_k'])
    kkt = kkt_from_tri_bond(geo['tri_bond'], geo['edge_vecs'])
    w3_s = sim_W3(geo, TR.assemble_K_faff, CE.tri_metric_change, None)
    w3_a = W3_AmB(A3, geo['areas'], geo['edge_vecs'], geo['simplices'], len(geo['pts']), kkt)
    w3_m = RG.mf_W3(bare)
    return geo, bare, w3_s, w3_a, w3_m


def run(name, builder):
    print(f"\n=== {name} ===  nu / E:  sim | area-wtd (A-B)+edge+angle | single-site MF")
    out = {k: [] for k in ['nu_s', 'nu_a', 'nu_m', 'E_s', 'E_a', 'E_m']}
    for eta in ETAS:
        mesh, bare, w3_s, w3_a, w3_m = builder(eta)
        ns, Es = CE.Ceff_nuE(mesh, w3_s, bare)
        na, Ea = CE.Ceff_nuE(mesh, w3_a, bare)
        nm, Em = CE.Ceff_nuE(mesh, w3_m, bare)
        for k, v in zip(out, [ns, na, nm, Es, Ea, Em]):
            out[k].append(v)
        print(f"  eta={eta:.1f}: nu {ns:+.3f}/{na:+.3f}/{nm:+.3f}   E {Es:.4f}/{Ea:.4f}/{Em:.4f}", flush=True)
    return out


def main():
    cases = [('GEOMETRIC disorder (k=1)', geometric),
             (f'VD rigidity contrast a={VD_CONTRAST}', lambda e: vd(e, VD_CONTRAST))]
    results = [(name, run(name, b)) for name, b in cases]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), squeeze=False)
    for i, (name, o) in enumerate(results):
        for j, (key, ttl) in enumerate([('nu', "Poisson ratio ν"), ('E', "Young's modulus E")]):
            ax = axes[i, j]
            ax.plot(ETAS, o[f'{key}_s'], 'k-o', lw=2.4, ms=5, label='sim (truth)')
            ax.plot(ETAS, o[f'{key}_a'], '-^', color='#1f77b4', ms=6, label='area-wtd (A−B)+edge+angle')
            ax.plot(ETAS, o[f'{key}_m'], '--s', color='#d62728', ms=4, label='single-site MF')
            if key == 'nu':
                ax.axhline(0, color='gray', lw=0.5, ls=':')
            ax.set_xlabel('η'); ax.set_title(f'{name}\n{ttl}', fontsize=10); ax.grid(alpha=0.3)
            if i == 0 and j == 0:
                ax.legend(fontsize=8)
    fig.suptitle('Homogenised ν and E:  area-weighted χ-elimination  δA = A_s − [A]·S_s/[S_s]  vs simulation',
                 fontsize=12)
    plt.tight_layout()
    p = os.path.join(HERE, 'plots', 'dg_subchoice2_nuE.png')
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print('\nsaved', p)


if __name__ == '__main__':
    main()
