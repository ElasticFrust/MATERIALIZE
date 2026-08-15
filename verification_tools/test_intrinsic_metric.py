"""
The intrinsic, configuration-free metric solve, end to end.

Variables: the per-triangle metric fluctuation field  delta_g  (3*n_tri numbers) -- NO node
displacements anywhere.  Minimise the metric elastic energy

    E(delta_g) = 1/2 sum_s (Dg + delta_g_s)^T H_s (Dg + delta_g_s)

(Dg = macroscopic strain, the loading; H_s = per-triangle metric Hessian) subject ONLY to the
intrinsic metric constraints
    C = [ edge agreement ; curvature inc(delta_g)=0 ; AREA-weighted mean  sum_s A_s delta_g_s=0 ]
and NOT the plain mean.  Solved as a sparse saddle point
    [ H   C^T ] [delta_g]   [ -H Dg_rep ]
    [ C  -eI  ] [ lambda ] = [    0      ].

Part A (stored N=40 data, the one stored macro mode): per-triangle corr of the intrinsic
delta_g vs the PBC simulation, for C using the area-weighted mean vs the plain mean.
Part B (fresh meshes, 3 macro modes): homogenised C_eff -> nu, E vs eta, for the intrinsic
metric solve vs the simulation vs single-site MF.  This is the deliverable: the metric theory,
with the right constraints and NO plain mean, reproduces the simulation's Hessian.
"""
import os, sys
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..', 'Phase 2'))
import pbc_dg_analysis as pda
import test_angle_response as AR
import test_cluster_Ceff as CE
from test_mean_isolation import Hblocks
import metric_ops as MO

DATA = os.path.join(HERE, 'dg_analysis_data')


def edge_op(kkt, n_tri):
    s1, s2, q = kkt
    rows, cols, data = [], [], []
    for e in range(len(s1)):
        for loc in range(3):
            rows += [e, e]; cols += [3*s1[e]+loc, 3*s2[e]+loc]; data += [q[e, loc], -q[e, loc]]
    return sp.csr_matrix((data, (rows, cols)), shape=(len(s1), 3*n_tri))


def curv_op(sx, ev, n_node, n_tri):
    va, sv, av, _ = AR.build_angle_arrays(sx, ev, n_node, half_shear=False)
    rows, cols, data = [], [], []
    for j in range(len(va)):
        for loc in range(3):
            rows.append(va[j]); cols.append(3*sv[j]+loc); data.append(av[j, loc])
    return sp.csr_matrix((data, (rows, cols)), shape=(n_node, 3*n_tri))


def mean_op(n_tri, w):
    rows, cols, data = [], [], []
    for i in range(3):
        for s in range(n_tri):
            rows.append(i); cols.append(3*s+i); data.append(w[s])
    return sp.csr_matrix((data, (rows, cols)), shape=(3, 3*n_tri))


def intrinsic_solve(mesh, Dg_vec, weighted=True, eps=1e-10):
    """Solve the constrained metric energy minimisation for the non-affine field delta_g.
    Returns delta_g as (n_tri,3) vectors [dg11,dg12,dg22].  No B, no node displacements."""
    ev, sx = mesh['edge_vecs'], mesh['simplices']
    n_node = len(mesh['pts']); n_tri = len(sx)
    Hs = Hblocks(ev)                                     # (n_tri,3,3) metric Hessian
    Hblk = sp.block_diag([Hs[i] for i in range(n_tri)]).tocsc()
    w = mesh['areas'] if weighted else np.ones(n_tri)
    C = sp.vstack([edge_op(mesh['kkt_arrays'], n_tri),
                   curv_op(sx, ev, n_node, n_tri),
                   mean_op(n_tri, w)]).tocsc()
    Dg_rep = np.tile(Dg_vec, n_tri)                      # affine strain replicated per triangle
    b = Hblk @ Dg_rep                                    # forcing  -H Dg
    nC = C.shape[0]
    KKT = sp.bmat([[Hblk, C.T], [C, -eps*sp.eye(nC)]]).tocsc()
    rhs = np.concatenate([-b, np.zeros(nC)])
    sol = spla.spsolve(KKT, rhs)
    return sol[:3*n_tri].reshape(n_tri, 3)


def main():
    # ---------- Part A: per-triangle delta_g vs sim, on stored N=40 data ----------
    def cc(a3, dg2):
        b = MO.vec3(dg2).ravel(); a = a3.ravel()
        m = np.isfinite(a) & np.isfinite(b)
        return np.corrcoef(a[m], b[m])[0, 1]
    print("Part A: intrinsic metric solve (no B) vs PBC sim, per-triangle delta_g  (stored N=40)")
    print(f"{'eta':>5} | {'area-weighted mean':>19} | {'plain mean':>11}")
    for eta in [0.1, 0.2, 0.3, 0.4, 0.5]:
        s = np.load(os.path.join(DATA, f'sample_eta{eta:.2f}_trial0.npz'), allow_pickle=True)
        mesh = dict(edge_vecs=s['edge_vecs'], simplices=s['simplices'], pts=s['pts'],
                    areas=s['areas'], kkt_arrays=(s['kkt_s1'], s['kkt_s2'], s['kkt_q']))
        Dg_vec = MO.vec3(pda.macroscopic_dg(s['F']))
        dgi_w = intrinsic_solve(mesh, Dg_vec, weighted=True)
        dgi_p = intrinsic_solve(mesh, Dg_vec, weighted=False)
        print(f"{eta:>5.1f} | {cc(dgi_w, s['dg_sim']):>19.4f} | {cc(dgi_p, s['dg_sim']):>11.4f}")

    # ---------- Part B: homogenised C_eff -> nu, E vs eta (3 modes), fresh meshes ----------
    print("\nPart B: homogenised nu / E vs eta:  sim (truth) | intrinsic metric | single-site MF")
    print(f"{'eta':>5} | {'nu_sim':>7} {'nu_int':>7} {'nu_MF':>7} | {'E_sim':>8} {'E_int':>8} {'E_MF':>8}")
    N = 16
    ETAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    Fk = [np.eye(2) + CE.DELTA*M for M in CE.MODES]
    Dg_k = [MO.vec3(F.T @ F - np.eye(2)) for F in Fk]
    Dgmat_inv = np.linalg.inv(np.stack(Dg_k, axis=1))
    out = {k: [] for k in ['nu_s', 'nu_i', 'nu_m', 'E_s', 'E_i', 'E_m']}
    for eta in ETAS:
        mesh = pda.build_periodic_tf_mesh(N, eta, seed=0)
        ev, sx = mesh['edge_vecs'], mesh['simplices']
        n_node = len(mesh['pts']); n_tri = len(sx); bare = MO.bare_tensor(mesh)
        K, _ = pda._assemble_K_and_faff(mesh, np.eye(2)); free = np.arange(2, 2*n_node)
        # sim: relax whole network, 3 modes
        D_sim = np.zeros((n_tri, 3, 3)); D_int = np.zeros((n_tri, 3, 3))
        for k, F in enumerate(Fk):
            fa = pda._assemble_K_and_faff(mesh, F)[1]
            u = np.zeros(2*n_node); u[free] = spla.spsolve(K[free][:, free].tocsc(), -fa[free])
            dg = MO.tri_metric_change(ev, sx, F, u.reshape(n_node, 2)) - (F.T@F - np.eye(2))
            D_sim[:, :, k] = MO.vec3(dg)
            D_int[:, :, k] = intrinsic_solve(mesh, Dg_k[k], weighted=True)   # metric-only
        W3_s = D_sim @ Dgmat_inv; W3_i = D_int @ Dgmat_inv
        W3_m = pda.woodbury_W(mesh, area_weighted=False, use_kkt=False).reshape(n_tri, 3, 3)
        nu_s, E_s = CE.Ceff_nuE(mesh, W3_s, bare)
        nu_i, E_i = CE.Ceff_nuE(mesh, W3_i, bare)
        nu_m, E_m = CE.Ceff_nuE(mesh, W3_m, bare)
        for kk, vv in zip(out, [nu_s, nu_i, nu_m, E_s, E_i, E_m]):
            out[kk].append(vv)
        print(f"{eta:>5.1f} | {nu_s:>7.3f} {nu_i:>7.3f} {nu_m:>7.3f} | "
              f"{E_s:>8.4f} {E_i:>8.4f} {E_m:>8.4f}", flush=True)

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for a, key, ttl in [(ax[0], 'nu', "Poisson ratio ν"), (ax[1], 'E', "Young's modulus E")]:
        a.plot(ETAS, out[f'{key}_s'], 'k-o', lw=2.5, label='simulation (truth)')
        a.plot(ETAS, out[f'{key}_i'], '-^', color='#1f77b4', ms=7, label='intrinsic metric (edge+curv+area-mean)')
        a.plot(ETAS, out[f'{key}_m'], '--s', color='#d62728', label='single-site MF (plain mean)')
        a.set_xlabel('η'); a.set_title(ttl); a.legend(fontsize=9); a.grid(alpha=0.3)
    ax[0].axhline(0, color='gray', lw=0.5, ls=':')
    fig.suptitle('Configuration-free metric solve reproduces the simulation Hessian', fontsize=12)
    plt.tight_layout()
    p = os.path.join(HERE, 'plots', 'dg_intrinsic_metric.png')
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print('saved', p)


if __name__ == '__main__':
    main()
