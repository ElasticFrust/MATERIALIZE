"""
Verify sub-choice 2: eliminating chi with the AREA-weighted condition gives
    dA(s) = A_s - [A] * S_s / [S_s]            ([.] = unweighted sum, S_s = triangle area)
and the (A - B) operator with row prefactor S_s/[S_s],  (B dg)(s) = (S_s/[S_s]) sum_s' dA(s') dg(s').
Solve (A-B) dg + edge + curvature KKT (chi already eliminated) and compare per-triangle dg to
the PBC simulation. Also contrast with the SHIPPED 'weights' delta = A_s - <A>_S (constant mean).
"""
import os, sys
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pbc_dg_analysis as pda
import test_cluster_Ceff as CE
from test_mean_isolation import Hblocks
from test_intrinsic_metric import edge_op, curv_op

DELTA = 1e-3
H = np.array([[1.0, 0.0], [0.0, 0.0]])             # uniaxial macro strain direction


def solve_AmB(A3, Svec, dg_drive, J, C):
    """Solve (A-B) dg + J^T l + C^T k = -dA.Dg,  J dg=0, C dg=0,  for a given loading.
    dA built with the chosen delta rule; returns dg (n_tri,3). chi is eliminated."""
    N = A3.shape[0]
    Hblk = sp.block_diag([A3[i] for i in range(N)]).tocsc()
    # B as a dense rank-3 operator: (B dg)(s) = (S_s/V) sum_s' dA(s') dg(s')
    # represented via the 3N x 3N matrix B[s,s'] = (S_s/V) dA(s')
    # We assemble A-B densely (N small).
    return Hblk  # placeholder, real work below


def run(rule):
    """rule: 'area_scaled' (user's dA = A_s - [A] S_s/[S_s]) or 'const_mean' (shipped)."""
    mesh = pda.build_periodic_tf_mesh(10, ETA, seed=0)
    ev, sx = mesh['edge_vecs'], mesh['simplices']
    n_node = len(mesh['pts']); n_tri = len(sx); S = mesh['areas']; V = S.sum()
    A3 = Hblocks(ev)                                          # (n_tri,3,3) block-diag stiffness
    Asum = A3.sum(0)                                          # [A] unweighted sum (3,3)
    if rule == 'area_scaled':
        dA = A3 - (S[:, None, None] / V) * Asum               # A_s - [A] S_s/[S_s]
    else:  # shipped constant area-weighted mean
        Abar_S = (S[:, None, None] / V * A3).sum(0)           # <A>_S (3,3 constant)
        dA = A3 - Abar_S[None]

    # B[s,s'] = (S_s/V) dA(s')  -> dense (3N,3N)
    pref = S / V                                              # (n_tri,)
    Bdense = np.zeros((3*n_tri, 3*n_tri))
    dA_stack = dA.reshape(n_tri, 3, 3)
    for s in range(n_tri):
        # row-block s: (S_s/V) * [dA(0) dA(1) ... ] horizontally
        Bdense[3*s:3*s+3, :] = pref[s] * dA_stack.transpose(1, 0, 2).reshape(3, 3*n_tri)
    Hblk = np.zeros((3*n_tri, 3*n_tri))
    for s in range(n_tri):
        Hblk[3*s:3*s+3, 3*s:3*s+3] = A3[s]
    AmB = Hblk - Bdense                                       # (A-B), 3N x 3N

    # constraints: edge + curvature (chi already eliminated)
    J = edge_op(mesh['kkt_arrays'], n_tri).toarray()
    C = curv_op(sx, ev, n_node, n_tri).toarray()
    Cc = np.vstack([J, C])
    nC = Cc.shape[0]

    # loading
    F = np.eye(2) + DELTA*H
    Dgv = CE.vec3(pda.macroscopic_dg(F))                     # (3,)
    rhs_top = -(dA_stack @ Dgv).reshape(3*n_tri)             # -dA.Dg per triangle
    K = np.block([[AmB, Cc.T], [Cc, np.zeros((nC, nC))]])
    rhs = np.concatenate([rhs_top, np.zeros(nC)])
    sol = np.linalg.lstsq(K, rhs, rcond=None)[0]
    dg = sol[:3*n_tri].reshape(n_tri, 3)

    # simulation dg for this mesh / loading
    Km, faff = pda._assemble_K_and_faff(mesh, F)
    free = np.arange(2, 2*n_node); u = np.zeros(2*n_node)
    u[free] = spla.spsolve(Km[free][:, free].tocsc(), -faff[free])
    sim = pda.triangle_metric_change(ev, sx, F, u.reshape(n_node, 2)) - pda.macroscopic_dg(F)
    simv = CE.vec3(sim)
    corr = np.corrcoef(dg.ravel(), simv.ravel())[0, 1]
    over = np.median(np.linalg.norm(dg, axis=1) / np.maximum(np.linalg.norm(simv, axis=1), 1e-30))
    # area-weighted normalisation residual of the solution
    awres = np.linalg.norm((S[:, None] * dg).sum(0))
    return corr, over, awres


ETA = 0.3
if __name__ == '__main__':
    print(f"N=10, eta={ETA}, (A-B)+edge+curvature, chi eliminated. corr / overshoot vs sim")
    for rule in ('area_scaled', 'const_mean'):
        c, o, r = run(rule)
        tag = "USER's dA = A_s - [A] S_s/[S_s]" if rule == 'area_scaled' else "shipped dA = A_s - <A>_S"
        print(f"  {tag:34s}: corr={c:.4f}  overshoot={o:.3f}  |Sum S_s dg|={r:.2e}")
