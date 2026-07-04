"""
Discrete incompatibility (curvature / disclination) operator, and the
'compatible re-solve = simulation' check that addresses: can the correct conditions on W
(compatibility) recover the truth in the metric description, without the cluster?

(A) Curvature operator Cv: per interior vertex, the linearised angle-deficit change
       (Cv delta_g)_v = sum_{s at v} a_s . [dg11,dg12,dg22]_s     (a_s = d(theta)/dg, tensor)
    = the discrete Gaussian curvature change = St-Venant incompatibility = disclination
    density. Compatible (node-realisable) fields have Cv delta_g = 0.
    Show: ||Cv delta_g_sim|| ~ 0 (compatible) vs ||Cv delta_g_MF|| large (spurious disclinations).

(B) Compatible re-solve: minimise the SAME metric energy 1/2 sum_s dG_s^T M_s dG_s over the
    COMPATIBLE subspace (delta_g = B u, u = node displacements), i.e. impose exact
    compatibility on the field. Compare to the simulation. This is the 'correct conditions
    on W' route -- a single global solve, of which the cluster is the local truncation.
    NB: projecting the MF onto compatible (compat_projection) does NOT fix it; re-MINIMISING
    over the compatible subspace does. (Corrects the earlier projection-based statement.)
"""
import os, sys
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import test_angle_response as AR       # angle_grad, build_angle_arrays
import compat_projection as CP          # build_B
import pbc_dg_analysis as pda           # _assemble_K_and_faff, triangle_metric_change, macroscopic_dg
DATA = os.path.join(HERE, 'dg_analysis_data')


def dvec(dg):
    return np.stack([dg[:, 0, 0], dg[:, 0, 1], dg[:, 1, 1]], 1)


def curvature_residual(dg, sx, ev, n_node):
    """||Cv dg|| normalised by RMS of the per-(vertex,triangle) terms."""
    va, sv, av, n_int = AR.build_angle_arrays(sx, ev, n_node, half_shear=False)
    dv = dvec(dg)
    term = (av * dv[sv]).sum(1)                 # a_s . dg_s for each (vertex, triangle) pair
    Cv = np.bincount(va, weights=term, minlength=n_node)   # per-vertex angle-deficit change
    return np.sqrt(np.mean(Cv**2)) / max(np.sqrt(np.mean(term**2)), 1e-30), Cv


def Mblocks(bare):
    a0, a1, a2, a3, a4 = [bare[:, i] for i in range(5)]
    M = np.stack([np.stack([a0, 2*a1, a2], -1),
                  np.stack([a1, 2*a2, a3], -1),
                  np.stack([a2, 2*a3, a4], -1)], -2)        # (n,3,3) acts on [dg11,dg12,dg22]
    return M


def compatible_resolve(s):
    """Minimise the metric energy over the COMPATIBLE subspace (delta_g = B u) and return
    the non-affine delta_g (n,2,2). This imposes exact compatibility on the metric field;
    the result equals the simulation.

    Energy Hessian on the compatible subspace:  K_m = B^T (blkdiag H_s) B,  with
    H_s = sum_edges (k/4 l^2) q q^T  (q=[vx^2,2vxvy,vy^2]); on the torus K_m = 2K (each edge
    in two triangles). The affine residual force is the spring-network affine force (x2 for
    the double count); deriving it purely per-triangle is an assembly detail, so we use the
    consistent spring force here."""
    ev, sx, F = s['edge_vecs'], s['simplices'], s['F']
    n_node = len(s['pts']); n_tri = len(sx)
    H = np.zeros((n_tri, 3, 3))
    for e in range(3):
        vx, vy = ev[:, e, 0], ev[:, e, 1]; l2 = vx**2 + vy**2
        q = np.stack([vx**2, 2*vx*vy, vy**2], 1)
        H += (1.0/(4*np.maximum(l2, 1e-30)))[:, None, None] * np.einsum('ni,nj->nij', q, q)
    B = CP.build_B(ev, sx, n_node)                          # (3n_tri, 2n_node)
    Km = (B.T @ sp.block_diag([H[i] for i in range(n_tri)]).tocsc() @ B).tocsc()
    mesh = {'bond_u': s['bond_u'], 'bond_v': s['bond_v'], 'bond_R': s['bond_R'], 'pts': s['pts']}
    _, faff = pda._assemble_K_and_faff(mesh, F)
    free = np.array([i for i in range(2*n_node) if i not in (0, 1, 2)])
    u = np.zeros(2*n_node)
    u[free] = spla.spsolve(Km[free][:, free].tocsc(), (-2*faff)[free])   # Km=2K, force=-2 faff
    return pda.triangle_metric_change(ev, sx, F, u.reshape(n_node, 2)) - pda.macroscopic_dg(F)


def main():
    # (A) curvature residual: sim vs MF, across eta
    print("(A) curvature (disclination) residual ||Cv dg|| / RMS:  sim vs single-site MF")
    etas = [0.1, 0.2, 0.3, 0.4]
    rs_sim, rs_mf = [], []
    for eta in etas:
        s = np.load(os.path.join(DATA, f'sample_eta{eta:.2f}_trial0.npz'), allow_pickle=True)
        nn = len(s['pts'])
        r_sim, _ = curvature_residual(s['dg_sim'], s['simplices'], s['edge_vecs'], nn)
        r_mf, _ = curvature_residual(s['dg_Std'], s['simplices'], s['edge_vecs'], nn)
        rs_sim.append(r_sim); rs_mf.append(r_mf)
        print(f"   eta={eta}:  sim {r_sim:.2e}   MF {r_mf:.2e}")

    # (B) compatible re-solve = simulation (re-MINIMISE over compatible),  vs
    #     projecting MF onto compatible (which does NOT fix it).  project != minimise.
    print("\n(B) metric energy minimised over the COMPATIBLE subspace (= correct conditions on W):")
    def cc(A, Bm):
        a = dvec(A).ravel(); b = dvec(Bm).ravel(); m = np.isfinite(a) & np.isfinite(b)
        return np.corrcoef(a[m], b[m])[0, 1]
    def fro(a):
        return np.sqrt((a*a).sum((1, 2)))
    for eta in [0.1, 0.2, 0.3, 0.4]:
        s = np.load(os.path.join(DATA, f'sample_eta{eta:.2f}_trial0.npz'), allow_pickle=True)
        nn = len(s['pts']); ev = s['edge_vecs']; sim, mf = s['dg_sim'], s['dg_Std']
        dg_re = compatible_resolve(s)
        # projection of MF onto compatible (for contrast)
        B = CP.build_B(ev, s['simplices'], nn)
        free = np.array([i for i in range(2*nn) if i not in (0, 1, 2)])
        spl = spla.splu((B.T @ B).tocsc()[free][:, free])
        d = dvec(mf).ravel(); u = np.zeros(2*nn); u[free] = spl.solve((B.T @ d)[free])
        proj = (B @ u).reshape(-1, 3)
        mfp = np.zeros_like(mf)
        mfp[:, 0, 0] = proj[:, 0]; mfp[:, 1, 1] = proj[:, 2]; mfp[:, 0, 1] = mfp[:, 1, 0] = proj[:, 1]
        print(f"   eta={eta}:  compat re-solve  corr={cc(dg_re, sim):.4f} overshoot="
              f"{np.median(fro(dg_re)/np.maximum(fro(sim),1e-30)):.3f}"
              f"   |  MF raw overshoot={np.median(fro(mf)/np.maximum(fro(sim),1e-30)):.2f}"
              f"  MF->projected overshoot={np.median(fro(mfp)/np.maximum(fro(sim),1e-30)):.2f}")

    # plot: curvature residual + a disclination-density map for MF at eta=0.3
    s = np.load(os.path.join(DATA, 'sample_eta0.30_trial0.npz'), allow_pickle=True)
    nn = len(s['pts'])
    _, Cv_mf = curvature_residual(s['dg_Std'], s['simplices'], s['edge_vecs'], nn)
    _, Cv_sim = curvature_residual(s['dg_sim'], s['simplices'], s['edge_vecs'], nn)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].semilogy(etas, rs_mf, '-s', color='#d62728', label='single-site MF')
    ax[0].semilogy(etas, rs_sim, '-o', color='k', label='simulation')
    ax[0].set_xlabel('η'); ax[0].set_ylabel('curvature residual  ||Cv δg|| / RMS')
    ax[0].set_title('Spurious disclination content of δg'); ax[0].legend(); ax[0].grid(alpha=0.3, which='both')
    pts = s['pts']
    sc = ax[1].scatter(pts[:, 0], pts[:, 1], c=Cv_mf, cmap='coolwarm', s=14,
                       vmin=-np.percentile(np.abs(Cv_mf), 98), vmax=np.percentile(np.abs(Cv_mf), 98))
    plt.colorbar(sc, ax=ax[1], label='per-vertex angle-deficit change (MF)')
    ax[1].set_aspect('equal'); ax[1].set_title('MF disclination density map (η=0.3)')
    ax[1].set_xticks([]); ax[1].set_yticks([])
    plt.tight_layout()
    p = os.path.join(HERE, 'plots', 'dg_curvature_operator.png')
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print('\nsaved', p)


if __name__ == '__main__':
    main()
