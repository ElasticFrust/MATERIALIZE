"""
VERIFY the claim: does G&B's original Woodbury formulation reproduce the simulation IF we
  (1) use the AREA-weighted mean (Sum_s S_s dg = 0),
  (2) add the edge (KKT) constraint, and
  (3) add the vertex-angle (curvature) constraint -- with the tensor-consistent convention?

We call the PRODUCTION solver function fst._woodbury_kkt_sparse_combined directly with
area weights + edge kkt + angle arrays built with half_shear=False (the tensor convention
consistent with q and with the W=[dg11,dg12,dg22] output). This bypasses ONLY the buggy
'/2' in _build_vertex_angle_constraints; everything else is the shipped G&B/Woodbury code.

Compares per-triangle dg vs sim (stored N=40) and homogenised nu/E vs sim (fresh meshes),
against the failing single-site MF and (cross-check) the standalone intrinsic solve.
"""
import os, sys
import numpy as np
import scipy.sparse.linalg as spla
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..', 'Phase 2'))
import forward_solver_torch as fst
import pbc_dg_analysis as pda
import test_angle_response as AR
import test_cluster_Ceff as CE
from test_intrinsic_metric import intrinsic_solve
import metric_ops as MO
torch.set_default_dtype(torch.float64)
DATA = os.path.join(HERE, 'dg_analysis_data')


def gb_W(bare, area_w, kkt, angle):
    """Production Woodbury KKT solver: area-weighted mean + edge + angle. Returns (N,9)."""
    mean = (bare * area_w[:, None]).sum(0)
    db = bare - mean
    A = fst._batch_to_9x9(torch.as_tensor(bare))
    B = fst._batch_to_9x9(torch.as_tensor(db))
    dA = fst._batch_to_9vec(torch.as_tensor(db))
    return fst._woodbury_kkt_sparse_combined(A, B, dA, kkt, angle, weights=area_w)


def mf_W(bare):
    db = bare - bare.mean(0)
    A = fst._batch_to_9x9(torch.as_tensor(bare))
    B = fst._batch_to_9x9(torch.as_tensor(db))
    dA = fst._batch_to_9vec(torch.as_tensor(db))
    return fst._woodbury_solve(A, B, dA, J=None, weights=None).detach().numpy()


def dgvec(W9, Dg_vec):
    return (W9.reshape(-1, 3, 3) @ Dg_vec)


def cc(a3, dg2):
    b = MO.vec3(dg2).ravel(); a = a3.ravel(); m = np.isfinite(a) & np.isfinite(b)
    return np.corrcoef(a[m], b[m])[0, 1]


def main():
    # ---- Part A: per-triangle dg vs sim (stored N=40, single mode) ----
    print("Part A: per-triangle dg corr vs PBC sim (stored N=40)")
    print(f"{'eta':>5} | {'G&B area+edge+angle':>20} | {'plain MF':>9} | {'intrinsic':>10}")
    for eta in [0.1, 0.2, 0.3, 0.4, 0.5]:
        s = np.load(os.path.join(DATA, f'sample_eta{eta:.2f}_trial0.npz'), allow_pickle=True)
        ev, sx = s['edge_vecs'], s['simplices']; nn = len(s['pts'])
        bare = AR.bare_from_edges(ev, s['actual_len2'])
        aw = s['areas'] / s['areas'].sum()
        kkt = (s['kkt_s1'], s['kkt_s2'], s['kkt_q'])
        angle = AR.build_angle_arrays(sx, ev, nn, half_shear=False)
        Dgv = MO.vec3(pda.macroscopic_dg(s['F']))
        dg_gb = dgvec(gb_W(bare, aw, kkt, angle), Dgv)
        dg_mf = dgvec(mf_W(bare), Dgv)
        mesh = dict(edge_vecs=ev, simplices=sx, pts=s['pts'], areas=s['areas'],
                    kkt_arrays=kkt)
        dg_in = intrinsic_solve(mesh, Dgv, weighted=True)
        print(f"{eta:>5.1f} | {cc(dg_gb, s['dg_sim']):>20.4f} | {cc(dg_mf, s['dg_sim']):>9.4f} | "
              f"{cc(dg_in, s['dg_sim']):>10.4f}")

    # ---- Part B: homogenised nu/E vs sim (fresh N=16, 3 modes) ----
    print("\nPart B: homogenised nu / E vs eta:  sim | G&B(area+edge+angle) | MF")
    print(f"{'eta':>5} | {'nu_sim':>7} {'nu_GB':>7} {'nu_MF':>7} | {'E_sim':>8} {'E_GB':>8} {'E_MF':>8}")
    N = 16
    Fk = [np.eye(2) + CE.DELTA*M for M in CE.MODES]
    Dg_k = [MO.vec3(F.T@F - np.eye(2)) for F in Fk]
    Dinv = np.linalg.inv(np.stack(Dg_k, axis=1))
    for eta in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        mesh = pda.build_periodic_tf_mesh(N, eta, seed=0)
        ev, sx = mesh['edge_vecs'], mesh['simplices']; nn = len(mesh['pts']); nt = len(sx)
        bare = MO.bare_tensor(mesh); aw = mesh['areas'] / mesh['areas'].sum()
        kkt = mesh['kkt_arrays']; angle = AR.build_angle_arrays(sx, ev, nn, half_shear=False)
        K, _ = pda._assemble_K_and_faff(mesh, np.eye(2)); free = np.arange(2, 2*nn)
        Ds = np.zeros((nt, 3, 3))
        for k, F in enumerate(Fk):
            fa = pda._assemble_K_and_faff(mesh, F)[1]
            u = np.zeros(2*nn); u[free] = spla.spsolve(K[free][:, free].tocsc(), -fa[free])
            Ds[:, :, k] = MO.vec3(MO.tri_metric_change(ev, sx, F, u.reshape(nn, 2)) - (F.T@F-np.eye(2)))
        nu_s, E_s = CE.Ceff_nuE(mesh, Ds @ Dinv, bare)
        W9 = gb_W(bare, aw, kkt, angle)
        nu_g, E_g = CE.Ceff_nuE(mesh, W9.reshape(nt, 3, 3), bare)
        nu_m, E_m = CE.Ceff_nuE(mesh, mf_W(bare).reshape(nt, 3, 3), bare)
        print(f"{eta:>5.1f} | {nu_s:>7.3f} {nu_g:>7.3f} {nu_m:>7.3f} | "
              f"{E_s:>8.4f} {E_g:>8.4f} {E_m:>8.4f}", flush=True)


if __name__ == '__main__':
    main()
