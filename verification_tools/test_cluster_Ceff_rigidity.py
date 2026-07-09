"""
ny / E vs rigidity contrast: cluster forward solver vs single-site MF vs simulation.
Regular lattice (eta=0), random binary bond rigidities k in {1, 1/ratio}. Homogenised
nu and E via the same C=(I+W)^T A (I+W) pipeline, W from MF / cluster(d) / full relax.
"""
import os
import numpy as np
import scipy.sparse.linalg as spla
import torch
import _bootstrap
import matplotlib.pyplot as plt

import test_cluster_rigidity as TR
import test_cluster_Ceff as CE
import forward_solver_torch as fst
torch.set_default_dtype(torch.float64)

HERE = _bootstrap.HERE

N = 16
RATIOS = [1.0, 2.0, 3.0, 5.0, 10.0, 30.0, 100.0]
DS = [2, 4]
DELTA = CE.DELTA
MODES = CE.MODES


def mf_W3(bare):
    db = bare - bare.mean(0)
    A = fst._batch_to_9x9(torch.as_tensor(bare)); B = fst._batch_to_9x9(torch.as_tensor(db))
    dA = fst._batch_to_9vec(torch.as_tensor(db))
    W9 = fst._woodbury_solve(A, B, dA, J=None, weights=None).detach().numpy()
    return W9.reshape(-1, 3, 3)


def central_dg(ev, c, F, u, n0, n1, n2, Dg_t):
    a, b = ev[c, 0], ev[c, 1]
    Eref = np.array([[a[0], b[0]], [a[1], b[1]]])
    ad = a @ F.T + (u[n1] - u[n0]); bd = b @ F.T + (u[n2] - u[n0])
    Fc = np.array([[ad[0], bd[0]], [ad[1], bd[1]]]) @ np.linalg.inv(Eref)
    return Fc.T @ Fc - np.eye(2) - Dg_t


def main():
    Fk = [np.eye(2) + DELTA * M for M in MODES]
    Dg_t = [F.T @ F - np.eye(2) for F in Fk]
    Dgmat_inv = np.linalg.inv(np.stack([CE.vec3(g) for g in Dg_t], axis=1))
    out = {k: [] for k in ['nu_s', 'E_s', 'nu_m', 'E_m'] + [f'nu_d{d}' for d in DS] + [f'E_d{d}' for d in DS]}

    for ratio in RATIOS:
        # average a couple of disorder realisations
        acc = {k: [] for k in out}
        for seed in range(2):
            mesh = TR.build_mesh_k(N, ratio, seed)
            ev, sx = mesh['edge_vecs'], mesh['simplices']
            e01, e02 = ev[:, 0], ev[:, 1]
            mesh['areas'] = 0.5 * np.abs(e01[:, 0]*e02[:, 1] - e01[:, 1]*e02[:, 0])
            nn = len(mesh['pts']); nt = len(sx)
            bare = TR.bare_tensor(mesh)
            K, _ = TR.assemble_K_faff(mesh, np.eye(2))
            faff = [TR.assemble_K_faff(mesh, F)[1] for F in Fk]

            # sim (relax all)
            free = np.arange(2, 2*nn)
            Ds = np.zeros((nt, 3, 3))
            for k, F in enumerate(Fk):
                u = np.zeros(2*nn)
                u[free] = spla.spsolve(K[free][:, free].tocsc(), -faff[k][free])
                dg = CE.tri_metric_change(ev, sx, F, u.reshape(nn, 2)) - Dg_t[k]
                Ds[:, :, k] = CE.vec3(dg)
            nu_s, E_s = CE.Ceff_nuE(mesh, Ds @ Dgmat_inv, bare)

            # MF
            nu_m, E_m = CE.Ceff_nuE(mesh, mf_W3(bare), bare)

            # cluster d
            adj = [set() for _ in range(nn)]
            for a_, b_ in zip(mesh['bond_u'], mesh['bond_v']):
                adj[int(a_)].add(int(b_)); adj[int(b_)].add(int(a_))
            def ring(sd, d):
                seen = set(sd); fr = set(sd)
                for _ in range(d):
                    nx = set()
                    for x in fr:
                        nx |= adj[x]
                    nx -= seen; seen |= nx; fr = nx
                return seen
            nuE_d = {}
            for d in DS:
                Dc = np.zeros((nt, 3, 3))
                for c in range(nt):
                    n0, n1, n2 = int(sx[c, 0]), int(sx[c, 1]), int(sx[c, 2])
                    fr = np.array(sorted(ring([n0, n1, n2], d)))
                    fdof = np.sort(np.concatenate([2*fr, 2*fr+1]))
                    Kff = K[fdof][:, fdof].tocsc()
                    for k, F in enumerate(Fk):
                        u = np.zeros((nn, 2))
                        u.ravel()[fdof] = spla.spsolve(Kff, -faff[k][fdof])
                        Dc[c, :, k] = CE.vec3(central_dg(ev, c, F, u, n0, n1, n2, Dg_t[k]))
                nuE_d[d] = CE.Ceff_nuE(mesh, Dc @ Dgmat_inv, bare)

            acc['nu_s'].append(nu_s); acc['E_s'].append(E_s)
            acc['nu_m'].append(nu_m); acc['E_m'].append(E_m)
            for d in DS:
                acc[f'nu_d{d}'].append(nuE_d[d][0]); acc[f'E_d{d}'].append(nuE_d[d][1])
        for k in out:
            out[k].append(np.nanmean(acc[k]))
        print(f"ratio={ratio:>5.0f}: nu sim/MF/d2/d4 = {out['nu_s'][-1]:+.3f}/{out['nu_m'][-1]:+.3f}/"
              f"{out['nu_d2'][-1]:+.3f}/{out['nu_d4'][-1]:+.3f}   "
              f"E = {out['E_s'][-1]:.4f}/{out['E_m'][-1]:.4f}/{out['E_d2'][-1]:.4f}/{out['E_d4'][-1]:.4f}",
              flush=True)

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for a, key, ttl in [(ax[0], 'nu', 'Poisson ratio ν'), (ax[1], 'E', "Young's modulus E")]:
        a.plot(RATIOS, out[f'{key}_s'], 'k-o', lw=2.5, label='simulation (truth)')
        a.plot(RATIOS, out[f'{key}_m'], '--s', color='#d62728', label='single-site MF')
        a.plot(RATIOS, out[f'{key}_d2'], '-^', color='#2ca02c', label='cluster d=2')
        a.plot(RATIOS, out[f'{key}_d4'], '-v', color='#1f77b4', label='cluster d=4')
        a.set_xscale('log'); a.set_xlabel('rigidity contrast  k_stiff / k_soft')
        a.set_title(ttl); a.legend(fontsize=9); a.grid(alpha=0.3, which='both')
    ax[0].axhline(0, color='gray', lw=0.5, ls=':')
    fig.suptitle('Homogenised ν and E vs rigidity contrast (regular lattice, η=0)', fontsize=12)
    plt.tight_layout()
    p = os.path.join(HERE, 'plots', 'dg_cluster_Ceff_rigidity.png')
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print('saved', p)


if __name__ == '__main__':
    main()
