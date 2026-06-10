"""
Are ALL the compatibility constraints expressible in the metric description?

Build the three metric-space constraint operators on the per-triangle field delta_g (3N):
  J_edge : edge-length agreement,  q.(dg_s1 - dg_s2) = 0           (E_int rows)
  Cv     : zero discrete curvature, sum_{s at v} a_s.dg_s = 0       (n_int rows)
  Mean   : zero mean (affine/non-affine split), sum_s dg_s = 0      (3 rows)
and the displacement->strain operator B (range B = compatible, realisable non-affine fields).

Checks:
 (A) each constraint annihilates compatible fields:  ||J @ B|| ~ 0.
 (B) completeness via rank: dim ker([J_edge; Cv; Mean]) == rank(B). Show the progression
     edge -> +curvature -> +mean closing the gap down to the compatible subspace.
If equal, the metric description has EXACTLY the compatibility content of the configuration
description -- so imposing these on W recovers the simulation/cluster (verified separately
in test_curvature_operator: compatible re-solve = sim, corr 0.9999).
"""
import os, sys
import numpy as np
import scipy.sparse as sp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compat_projection as CP
import test_angle_response as AR
import pbc_dg_analysis as pda

N = 8


def edge_operator(kkt, n_tri):
    s1, s2, q = kkt
    rows, cols, data = [], [], []
    for e in range(len(s1)):
        for loc in range(3):
            rows += [e, e]; cols += [3*s1[e]+loc, 3*s2[e]+loc]; data += [q[e, loc], -q[e, loc]]
    return sp.csr_matrix((data, (rows, cols)), shape=(len(s1), 3*n_tri))


def curvature_operator(sx, ev, n_node, n_tri):
    va, sv, av, n_int = AR.build_angle_arrays(sx, ev, n_node, half_shear=False)
    rows, cols, data = [], [], []
    for j in range(len(va)):
        for loc in range(3):
            rows.append(va[j]); cols.append(3*sv[j]+loc); data.append(av[j, loc])
    return sp.csr_matrix((data, (rows, cols)), shape=(n_node, 3*n_tri))


def mean_operator(n_tri):
    rows, cols, data = [], [], []
    for i in range(3):
        for s in range(n_tri):
            rows.append(i); cols.append(3*s+i); data.append(1.0)
    return sp.csr_matrix((data, (rows, cols)), shape=(3, 3*n_tri))


def main():
    mesh = pda.build_periodic_tf_mesh(N, 0.3, seed=0)
    ev, sx = mesh['edge_vecs'], mesh['simplices']
    n_node, n_tri = len(mesh['pts']), len(sx)
    B = CP.build_B(ev, sx, n_node)                      # (3n_tri, 2n_node)
    Jedge = edge_operator(mesh['kkt_arrays'], n_tri)
    Cv = curvature_operator(sx, ev, n_node, n_tri)
    Mean = mean_operator(n_tri)

    nB = sp.linalg.norm(B)
    print(f"N={N}: n_node={n_node}, n_tri={n_tri}, 3*n_tri={3*n_tri}, 2*n_node={2*n_node}")
    print("\n(A) does each constraint annihilate compatible fields (range B)?  ||J@B||/(||J|| ||B||):")
    for name, J in [('edge', Jedge), ('curvature', Cv), ('mean', Mean)]:
        r = sp.linalg.norm(J @ B) / max(sp.linalg.norm(J)*nB, 1e-30)
        print(f"   {name:10s}: {r:.2e}")

    print("\n(B) completeness via rank (dim ker of constraint set vs dim of compatible subspace):")
    rankB = np.linalg.matrix_rank(B.toarray(), tol=1e-9)
    print(f"   rank(B) = dim(compatible non-affine subspace) = {rankB}   (= 2*n_node - 3 = {2*n_node-3})")
    for name, blocks in [('edge', [Jedge]), ('edge+curv', [Jedge, Cv]),
                         ('edge+curv+mean', [Jedge, Cv, Mean])]:
        J = sp.vstack(blocks).toarray()
        dim_ker = 3*n_tri - np.linalg.matrix_rank(J, tol=1e-9)
        flag = "== rank(B)  (COMPLETE)" if dim_ker == rankB else f"(gap {dim_ker-rankB})"
        print(f"   dim ker[{name:14s}] = {dim_ker:4d}   {flag}")


if __name__ == '__main__':
    main()
