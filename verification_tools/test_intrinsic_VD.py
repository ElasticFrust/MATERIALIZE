"""
Virtual-distortion (VD) rigidity contrast: homogenised nu / E vs eta at several signed
contrasts, using the CORRECT intrinsic metric solve (edge + curvature + AREA-weighted mean),
compared against the PBC simulation (truth) and the single-site MF.

Bond rigidity  k = 1 + tanh(a*(|R|-1))  (a=0 uniform; a>0 stiffer where stretched; a<0 stiffer
where compressed). The intrinsic solve uses the same k-weighted per-triangle metric Hessian
A(s)=H_s as the network, with NO plain mean -- only the S_triangle-weighted normalisation.
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
import test_cluster_rigidity as TR          # k-aware bare_tensor, assemble_K_faff
import test_cluster_Ceff as CE              # vec3, tri_metric_change, Ceff_nuE, DELTA, MODES
import test_cluster_Ceff_rigidity as RG     # mf_W3
import test_cluster_VD as VD                # build_geometry, set_VD
from test_intrinsic_metric import edge_op, curv_op, mean_op

N = 14
CONTRASTS = [-10, -5, 0, 5, 10]
ETAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
DELTA = CE.DELTA
MODES = CE.MODES


def kkt_from_tri_bond(tri_bond, edge_vecs):
    """Interior-edge (s1,s2,q) arrays from the triangle->bond map (torus: every bond shared)."""
    occ = {}
    for ti in range(len(tri_bond)):
        for ei in range(3):
            occ.setdefault(int(tri_bond[ti, ei]), []).append((ti, ei))
    s1, s2, q = [], [], []
    for b, lst in occ.items():
        if len(lst) == 2:
            (t1, e1), (t2, e2) = lst
            v = edge_vecs[t1, e1]
            s1.append(t1); s2.append(t2); q.append([v[0]**2, 2*v[0]*v[1], v[1]**2])
    return np.array(s1, np.int64), np.array(s2, np.int64), np.array(q, float)


def Hblocks_vd(ev, tri_k):
    """k-weighted per-triangle metric Hessian A(s)=H_s = sum_e (k_e/4 l_e^2) q_e q_e^T."""
    n_tri = ev.shape[0]; H = np.zeros((n_tri, 3, 3))
    for e in range(3):
        vx, vy = ev[:, e, 0], ev[:, e, 1]; l2 = vx**2 + vy**2
        q = np.stack([vx**2, 2*vx*vy, vy**2], 1)
        H += (tri_k[:, e]/(4*np.maximum(l2, 1e-30)))[:, None, None] * np.einsum('ni,nj->nij', q, q)
    return H


def intrinsic_W3(mesh, eps=1e-10):
    """3x3 strain-concentration W per triangle from the intrinsic metric KKT
    [A ; edge ; curvature ; area-weighted mean], NO plain mean. No node displacements."""
    ev, sx = mesh['edge_vecs'], mesh['simplices']
    n_node = len(mesh['pts']); n_tri = len(sx)
    Hblk = sp.block_diag([h for h in Hblocks_vd(ev, mesh['tri_k'])]).tocsc()
    kkt = kkt_from_tri_bond(mesh['tri_bond'], ev)
    C = sp.vstack([edge_op(kkt, n_tri),
                   curv_op(sx, ev, n_node, n_tri),
                   mean_op(n_tri, mesh['areas'])]).tocsc()
    nC = C.shape[0]
    KKT = sp.bmat([[Hblk, C.T], [C, -eps*sp.eye(nC)]]).tocsc()
    Fk = [np.eye(2) + DELTA*M for M in MODES]
    Dg_k = [CE.vec3(F.T@F - np.eye(2)) for F in Fk]
    Dinv = np.linalg.inv(np.stack(Dg_k, axis=1))
    D = np.zeros((n_tri, 3, 3))
    for k, dgv in enumerate(Dg_k):
        rhs = np.concatenate([-(Hblk @ np.tile(dgv, n_tri)), np.zeros(nC)])
        sol = spla.spsolve(KKT, rhs)
        D[:, :, k] = sol[:3*n_tri].reshape(n_tri, 3)
    return D @ Dinv


def nuE(mesh):
    """PHYSICAL (nu,E): PBC simulation (virial=energy, truth) vs the production forward solver
    (forward(method='intrinsic', physical_units=True)). Lazy import of make_solver avoids a
    circular import with verify_solver_sweep (which imports kkt_from_tri_bond from here)."""
    import torch
    import physical_homog as PH
    from verify_solver_sweep import make_solver
    free = np.arange(2, 2 * len(mesh['pts']))
    nu_s, E_s = PH.sim_nuE(mesh, free, TR.assemble_K_faff)
    kkt = kkt_from_tri_bond(mesh['tri_bond'], mesh['edge_vecs'])
    sv = make_solver(mesh, kkt)
    rl = torch.as_tensor(np.sqrt(mesh['actual_len2']), dtype=torch.float64)
    out = sv.forward(torch.as_tensor(mesh['tri_k'], dtype=torch.float64),
                     rest_lengths=rl, method='intrinsic', physical_units=True)
    return nu_s, E_s, float(out['poisson']), float(out['young'])


def main():
    res = {a: {k: [] for k in ['nu_s', 'E_s', 'nu_i', 'E_i']} for a in CONTRASTS}
    print(f"N={N}, VD k=1+tanh(a*(|R|-1)); PHYSICAL forward solver vs physical PBC simulation")
    for eta in ETAS:
        geo = VD.build_geometry(N, eta, seed=0)
        for a in CONTRASTS:
            VD.set_VD(geo, a)
            ns, Es, ni, Ei = nuE(geo)
            for kk, vv in zip(['nu_s', 'E_s', 'nu_i', 'E_i'], [ns, Es, ni, Ei]):
                res[a][kk].append(vv)
            print(f"  eta={eta:.1f} a={a:>+3}: nu sim/solver = {ns:+.3f}/{ni:+.3f}"
                  f"   E = {Es:.3f}/{Ei:.3f}", flush=True)

    nC = len(CONTRASTS)
    fig, axes = plt.subplots(nC, 2, figsize=(11, 3.0*nC), squeeze=False)
    for i, a in enumerate(CONTRASTS):
        for j, (key, ttl) in enumerate([('nu', 'ν'), ('E', 'E')]):
            ax = axes[i, j]
            ax.plot(ETAS, res[a][f'{key}_s'], 'k-o', lw=2.2, ms=4, label='physical sim (truth)')
            ax.plot(ETAS, res[a][f'{key}_i'], '-^', color='#1f77b4', ms=6, label='forward solver')
            if key == 'nu':
                ax.axhline(0, color='gray', lw=0.5, ls=':')
            ax.set_ylabel(f'{ttl}  (a={a:+d})'); ax.grid(alpha=0.3)
            if i == 0:
                ax.set_title(f'{ttl} vs η'); ax.legend(fontsize=8)
            if i == nC-1:
                ax.set_xlabel('η')
    fig.suptitle('ν and E vs η at several VD rigidity contrasts — PHYSICAL forward solver vs simulation',
                 fontsize=12)
    plt.tight_layout()
    p = os.path.join(HERE, 'plots', 'dg_intrinsic_VD_eta.png')
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print('saved', p)


if __name__ == '__main__':
    main()
