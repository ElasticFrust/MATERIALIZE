"""
Task 2: cluster forward solver -> homogenised C_eff / nu / E, vs the PBC simulation.

For each triangle we obtain its loading-independent response W (3x3 metric-space) three ways:
  - MF      : the single-site Woodbury W (existing solver)
  - cluster : relax a radius-d node patch (boundary affine) for the 3 macro strain modes
  - sim     : relax the WHOLE network (= cluster radius -> infinity) for the 3 modes
Then the SAME homogenisation C_s = (I+W)^T A (I+W) (fst._compute_actual_elastic_tensor),
area-weighted-averaged, gives C_eff -> (nu, E). Apples-to-apples; sim is the ground truth.
Validation: eta=0 regular lattice -> nu = 1/3 for all three.
"""
import os, sys
import numpy as np
import scipy.sparse.linalg as spla
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, 'Phase 2'))
import pbc_dg_analysis as pda
import forward_solver_torch as fst
# vec3 / tri_metric_change / bare_tensor MOVED to the core layer by the A-7b re-layering
# (Phase 2/metric_ops.py) — the design layer must not depend on this retireable oracle layer.
# Re-exported here so this script and its peers keep working unchanged. NB the single
# metric_ops.bare_tensor subsumes the k-less variant that used to live here: it reads `tri_k` if
# the mesh carries one and defaults to 1, and pbc_dg_analysis meshes carry none — bit-identical.
from metric_ops import vec3, tri_metric_change, bare_tensor    # noqa: F401
torch.set_default_dtype(torch.float64)

DELTA = 1e-3
MODES = [np.array([[1., 0.], [0., 0.]]), np.array([[0., 0.], [0., 1.]]),
         np.array([[0., .5], [.5, 0.]])]                        # xx, yy, xy
ETAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
N = 16
DRAD = 2


def W3_to_W9(W3):
    return W3.reshape(W3.shape[0], 9)                            # [3*loc+k] = W3[loc,k]


def c6_to_nuE(c6):
    CV = np.array([[c6[0], c6[2], c6[1]], [c6[2], c6[5], c6[4]], [c6[1], c6[4], c6[3]]])
    try:
        S = np.linalg.inv(CV); Ex, Ey = 1/S[0, 0], 1/S[1, 1]
        return 0.5*(-S[1, 0]*Ex - S[0, 1]*Ey), 0.5*(Ex + Ey)
    except np.linalg.LinAlgError:
        return np.nan, np.nan


def Ceff_nuE(mesh, W3, bare):
    # REMOVED 2026-08-10 — the legacy AREA-WEIGHTED metric average is physically WRONG: it biases
    # nu on unequal-area (disordered/anisotropic) meshes and overstates auxeticity. The correct
    # homogenisation is the UNWEIGHTED mean of C(s) (= physical energy=virial modulus). Use
    # verification_tools/physical_homog.py (sim_nuE / virial_nuE / energy_nuE) as ground truth.
    # Original body preserved for historical restore:
    #     W9 = torch.as_tensor(W3_to_W9(W3)); A = torch.as_tensor(bare)
    #     C6 = fst._compute_actual_elastic_tensor(A, W9).numpy()       # (N,6)
    #     w = mesh['areas'] / mesh['areas'].sum()
    #     return c6_to_nuE((C6 * w[:, None]).sum(0))
    raise NotImplementedError(
        "Ceff_nuE (legacy area-weighted metric average) removed as physically wrong; "
        "use verification_tools/physical_homog.py (sim_nuE / virial_nuE / energy_nuE).")


def main():
    print(f"N={N}, cluster radius d={DRAD}. nu / E:  sim (truth) | MF | cluster")
    print(f"{'eta':>5} | {'nu_sim':>7} {'nu_MF':>7} {'nu_cl':>7} | "
          f"{'E_sim':>8} {'E_MF':>8} {'E_cl':>8}")
    out = {k: [] for k in ['nu_s', 'nu_m', 'nu_c', 'E_s', 'E_m', 'E_c']}
    for eta in ETAS:
        mesh = pda.build_periodic_tf_mesh(N, eta, seed=0)
        ev, sx = mesh['edge_vecs'], mesh['simplices']
        n_node = len(mesh['pts']); n_tri = len(sx)
        bare = bare_tensor(mesh)
        K, _ = pda._assemble_K_and_faff(mesh, np.eye(2))         # K independent of F
        # per mode: F_k, macro Dg_k, affine force
        Fk = [np.eye(2) + DELTA*M for M in MODES]
        Dg_k = [vec3(F.T @ F - np.eye(2)) for F in Fk]
        Dgmat = np.stack(Dg_k, axis=1)                           # (3 loc, 3 modes)
        Dgmat_inv = np.linalg.inv(Dgmat)
        faff_k = [pda._assemble_K_and_faff(mesh, F)[1] for F in Fk]

        # ---- sim: relax all nodes (pin node 0) ----
        free = np.arange(2, 2*n_node)
        D_sim = np.zeros((n_tri, 3, 3))
        for k, (F, fa) in enumerate(zip(Fk, faff_k)):
            u = np.zeros(2*n_node)
            u[free] = spla.spsolve(K[free][:, free].tocsc(), -fa[free])
            dg = tri_metric_change(ev, sx, F, u.reshape(n_node, 2)) - (F.T@F - np.eye(2))
            D_sim[:, :, k] = vec3(dg)
        W3_sim = D_sim @ Dgmat_inv

        # ---- MF: single-site Woodbury ----
        W9_mf = pda.woodbury_W(mesh, area_weighted=False, use_kkt=False)
        W3_mf = W9_mf.reshape(n_tri, 3, 3)

        # ---- cluster: radius-d patch per triangle, 3 modes ----
        adj = [set() for _ in range(n_node)]
        for a_, b_ in zip(mesh['bond_u'], mesh['bond_v']):
            adj[int(a_)].add(int(b_)); adj[int(b_)].add(int(a_))
        def ring(seed_n, d):
            seen = set(seed_n); fr = set(seed_n)
            for _ in range(d):
                nx = set()
                for x in fr:
                    nx |= adj[x]
                nx -= seen; seen |= nx; fr = nx
            return seen
        D_cl = np.zeros((n_tri, 3, 3))
        for c in range(n_tri):
            n0, n1, n2 = int(sx[c, 0]), int(sx[c, 1]), int(sx[c, 2])
            fr = np.array(sorted(ring([n0, n1, n2], DRAD)))
            fdof = np.sort(np.concatenate([2*fr, 2*fr + 1]))
            Kff = K[fdof][:, fdof].tocsc()
            a_ref, b_ref = ev[c, 0], ev[c, 1]
            Eref = np.array([[a_ref[0], b_ref[0]], [a_ref[1], b_ref[1]]])
            Erefi = np.linalg.inv(Eref)
            for k, F in enumerate(Fk):
                u = np.zeros((n_node, 2))
                u.ravel()[fdof] = spla.spsolve(Kff, -faff_k[k][fdof])
                ad = a_ref @ F.T + (u[n1] - u[n0]); bd = b_ref @ F.T + (u[n2] - u[n0])
                Fc = np.array([[ad[0], bd[0]], [ad[1], bd[1]]]) @ Erefi
                dg = Fc.T @ Fc - np.eye(2) - (F.T@F - np.eye(2))
                D_cl[c, :, k] = vec3(dg)
        W3_cl = D_cl @ Dgmat_inv

        nu_s, E_s = Ceff_nuE(mesh, W3_sim, bare)
        nu_m, E_m = Ceff_nuE(mesh, W3_mf, bare)
        nu_c, E_c = Ceff_nuE(mesh, W3_cl, bare)
        print(f"{eta:>5.1f} | {nu_s:>7.3f} {nu_m:>7.3f} {nu_c:>7.3f} | "
              f"{E_s:>8.4f} {E_m:>8.4f} {E_c:>8.4f}")
        for kk, vv in zip(['nu_s', 'nu_m', 'nu_c', 'E_s', 'E_m', 'E_c'],
                          [nu_s, nu_m, nu_c, E_s, E_m, E_c]):
            out[kk].append(vv)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for a, key, ttl in [(ax[0], 'nu', "Poisson ratio ν"), (ax[1], 'E', "Young's modulus E")]:
        a.plot(ETAS, out[f'{key}_s'], 'k-o', lw=2.5, label='simulation (truth)')
        a.plot(ETAS, out[f'{key}_m'], '--s', color='#d62728', label='single-site MF')
        a.plot(ETAS, out[f'{key}_c'], '-^', color='#2ca02c', label=f'cluster d={DRAD}')
        a.set_xlabel('η'); a.set_title(ttl); a.legend(fontsize=10); a.grid(alpha=0.3)
    ax[0].axhline(0, color='gray', lw=0.5, ls=':')
    fig.suptitle('Homogenised properties: cluster forward solver vs single-site MF vs simulation',
                 fontsize=12)
    plt.tight_layout()
    p = os.path.join(HERE, 'plots', 'dg_cluster_Ceff.png')
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print('saved', p)


if __name__ == '__main__':
    main()
