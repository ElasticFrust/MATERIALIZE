"""
Isolate the mean/normalisation constraint as THE failure of the metric-space solve, and
verify the fix holds at large eta.  Also check the area-conservation identity the user raised.

We change exactly ONE thing between two solves on the SAME compatible subspace (delta_g = B u),
with the SAME loading (the simulation's affine spring force), differing only by the mean
constraint <delta_g> = 0:

  V_A  (the fix)   : minimise the per-triangle metric energy  1/2 sum_s dG_s^T H_s dG_s
                     over the compatible subspace, NO mean constraint.
  V_B  (+mean)     : identical, but add the 3 constraints  (Mean . B) u = 0  via KKT.

Reported per eta:
  corr / overshoot of V_A vs sim,  corr of V_B vs sim,  corr of V_B vs sim after
  removing the spatial mean (to show the damage is not a uniform offset),
  and the mean residual <delta_g> of the SIMULATION itself (nonzero => zero-mean is wrong).

Area identity check: the global affine area change vs the sum of per-triangle deformed areas.
"""
import os, sys
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import compat_projection as CP
import pbc_dg_analysis as pda
from test_curvature_operator import dvec, compatible_resolve

DATA = os.path.join(HERE, 'dg_analysis_data')


def Hblocks(ev):
    """Per-triangle metric-energy Hessian H_s = sum_edges (1/4 l^2) q q^T, q=[vx^2,2vxvy,vy^2]."""
    n_tri = ev.shape[0]
    H = np.zeros((n_tri, 3, 3))
    for e in range(3):
        vx, vy = ev[:, e, 0], ev[:, e, 1]; l2 = vx**2 + vy**2
        q = np.stack([vx**2, 2*vx*vy, vy**2], 1)
        H += (1.0/(4*np.maximum(l2, 1e-30)))[:, None, None] * np.einsum('ni,nj->nij', q, q)
    return H


def mean_op(n_tri):
    rows, cols, data = [], [], []
    for i in range(3):
        for s in range(n_tri):
            rows.append(i); cols.append(3*s+i); data.append(1.0)
    return sp.csr_matrix((data, (rows, cols)), shape=(3, 3*n_tri))


def resolve_with_mean(s):
    """V_B: same compatible metric-energy minimisation as compatible_resolve (V_A),
    but with the mean constraint (Mean . B) u = 0 enforced via KKT."""
    ev, sx, F = s['edge_vecs'], s['simplices'], s['F']
    n_node = len(s['pts']); n_tri = len(sx)
    H = Hblocks(ev)
    B = CP.build_B(ev, sx, n_node)
    Km = (B.T @ sp.block_diag([H[i] for i in range(n_tri)]).tocsc() @ B).tocsc()
    mesh = {'bond_u': s['bond_u'], 'bond_v': s['bond_v'], 'bond_R': s['bond_R'], 'pts': s['pts']}
    _, faff = pda._assemble_K_and_faff(mesh, F)
    MB = (mean_op(n_tri) @ B).tocsc()                       # 3 x 2n_node
    free = np.array([i for i in range(2*n_node) if i not in (0, 1, 2)])
    Kff = Km[free][:, free].tocsc(); MBf = MB[:, free]
    nf = len(free); nc = MBf.shape[0]
    KKT = sp.bmat([[Kff, MBf.T], [MBf, None]]).tocsc()
    rhs = np.concatenate([(-2*faff)[free], np.zeros(nc)])
    sol = spla.spsolve(KKT, rhs)
    u = np.zeros(2*n_node); u[free] = sol[:nf]
    return pda.triangle_metric_change(ev, sx, F, u.reshape(n_node, 2)) - pda.macroscopic_dg(F)


def cc(A, Bm):
    a = dvec(A).ravel(); b = dvec(Bm).ravel(); m = np.isfinite(a) & np.isfinite(b)
    return np.corrcoef(a[m], b[m])[0, 1]


def fro(a):
    return np.sqrt((a*a).sum((1, 2)))


def main():
    print("Isolation of the mean constraint (V_A = no mean, V_B = +mean), on stored N=40 data\n")
    print(f"{'eta':>5} | {'V_A corr':>9} {'V_A over':>9} | {'V_B corr':>9} {'V_B(-mean) corr':>16} |"
          f" {'sim <dg> resid':>15}")
    print("-"*82)
    for eta in [0.1, 0.2, 0.3, 0.4, 0.5]:
        s = np.load(os.path.join(DATA, f'sample_eta{eta:.2f}_trial0.npz'), allow_pickle=True)
        sim = s['dg_sim']
        dgA = compatible_resolve(s)                 # V_A
        dgB = resolve_with_mean(s)                  # V_B
        dgB_dm = dgB - dgB.mean(0, keepdims=True)   # V_B with spatial mean removed
        sim_mean_resid = np.linalg.norm(dvec(sim).sum(0))   # <delta_g> of the simulation
        print(f"{eta:5.1f} | {cc(dgA, sim):9.4f} {np.median(fro(dgA)/np.maximum(fro(sim),1e-30)):9.3f} |"
              f" {cc(dgB, sim):9.4f} {cc(dgB_dm, sim):16.4f} | {sim_mean_resid:15.3e}", flush=True)

    # ---- area identity the user raised ----
    # Global affine deformed area = det(F) * A_ref_total.  Sum of per-triangle deformed areas
    # = sum_s A_ref_s * sqrt(det(g_def_s))/sqrt(det(g_ref_s)).  These need NOT be equal, because
    # the non-affine fluctuation redistributes area; on the torus the TOTAL is fixed by det(F),
    # but it is enforced by the periodic boundary (a configuration/gradient condition), not by a
    # per-triangle metric mean.
    print("\nArea identity (global affine det(F)*A_ref  vs  sum of deformed triangle areas):")
    for eta in [0.0, 0.2, 0.4, 0.5]:
        s = np.load(os.path.join(DATA, f'sample_eta{eta:.2f}_trial0.npz'), allow_pickle=True)
        ev, sx, F, u = s['edge_vecs'], s['simplices'], s['F'], s['u_fluct']
        areas = s['areas']; A_ref = areas.sum()
        A_aff_global = abs(np.linalg.det(F)) * A_ref
        n_node = len(s['pts']); du = u.reshape(n_node, 2)
        # deformed area of triangle s = A_ref_s * det(F_s),  with g_def_s = F_s^T F_s = tmc + I,
        # det(F_s) = sqrt(det(g_def_s)).  (triangle_metric_change returns F_s^T F_s - I.)
        g_def_full = pda.triangle_metric_change(ev, sx, F, du) + np.eye(2)
        detFs = np.sqrt(np.maximum(np.linalg.det(g_def_full), 0))
        A_def_sum = (areas * detFs).sum()
        print(f"  eta={eta:.1f}:  det(F)*A_ref = {A_aff_global:.4f}   sum A_def = {A_def_sum:.4f}"
              f"   rel diff = {abs(A_def_sum-A_aff_global)/A_aff_global:.2e}"
              f"   <det F_s>-det F = {detFs.mean()-abs(np.linalg.det(F)):+.3e}")


if __name__ == '__main__':
    main()
