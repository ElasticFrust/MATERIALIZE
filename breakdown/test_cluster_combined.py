"""
Combined disorder: fixed rigidity contrast, several eta. Homogenised nu/E, code vs sim.
Perturbed periodic lattice (eta) + random binary bond rigidities k in {1, 1/ratio}.
Compares the cluster forward solver (code) and single-site MF against the PBC simulation.
"""
import os, sys
import numpy as np
import scipy.sparse.linalg as spla
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..', 'Phase 2'))
import test_cluster_rigidity as TR
import test_cluster_Ceff as CE
import test_cluster_Ceff_rigidity as RG
import forward_solver_torch as fst
torch.set_default_dtype(torch.float64)

N = 16
RATIO = 10.0          # fixed rigidity contrast
ETAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
DCL = 4
DELTA = CE.DELTA
MODES = CE.MODES


def build_mesh_eta_k(N, eta, ratio, seed):
    """Perturbed periodic triangular lattice (eta) + random binary bond rigidities."""
    rng = np.random.default_rng(seed)
    L1 = np.array([1.0, 0.0]); L2 = np.array([0.5, np.sqrt(3)/2])
    BL1, BL2 = N*L1, N*L2
    nn, mm = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
    ref = nn.ravel()[:, None]*L1 + mm.ravel()[:, None]*L2
    ang = rng.uniform(0, 2*np.pi, N*N)
    pts = ref + eta*np.stack([np.cos(ang), np.sin(ang)], 1)            # geometric disorder
    idx = lambda a, b: (a % N)*N + (b % N)
    simp, img = [], []
    for a in range(N):
        for b in range(N):
            simp.append([idx(a, b), idx(a+1, b), idx(a, b+1)])
            img.append([[0, 0], [1 if a+1 >= N else 0, 0], [0, 1 if b+1 >= N else 0]])
            simp.append([idx(a+1, b), idx(a+1, b+1), idx(a, b+1)])
            img.append([[1 if a+1 >= N else 0, 0],
                        [1 if a+1 >= N else 0, 1 if b+1 >= N else 0],
                        [0, 1 if b+1 >= N else 0]])
    simp = np.array(simp, np.int64); img = np.array(img, np.int64); n_tri = len(simp)
    vpos = lambda k: pts[simp[:, k]] + img[:, k, 0:1]*BL1 + img[:, k, 1:2]*BL2
    p0, p1, p2 = vpos(0), vpos(1), vpos(2)
    edge_vecs = np.stack([p1-p0, p2-p0, p2-p1], 1); l2 = (edge_vecs**2).sum(2)
    pairs = [(0, 1, 0), (0, 2, 1), (1, 2, 2)]
    keymap = {}; tri_bond = np.zeros((n_tri, 3), np.int64); bonds = []
    for ti in range(n_tri):
        for ka, kb, ei in pairs:
            va, vb = int(simp[ti, ka]), int(simp[ti, kb])
            d = img[ti, ka]-img[ti, kb]; dp = (int(d[0]), int(d[1]))
            if (va, dp[0], dp[1]) <= (vb, -dp[0], -dp[1]):
                key, R = (va, vb, dp[0], dp[1]), edge_vecs[ti, ei]
            else:
                key, R = (vb, va, -dp[0], -dp[1]), -edge_vecs[ti, ei]
            if key not in keymap:
                keymap[key] = len(bonds); bonds.append((key[0], key[1], R))
            tri_bond[ti, ei] = keymap[key]
    bu = np.array([b[0] for b in bonds], np.int64); bv = np.array([b[1] for b in bonds], np.int64)
    bR = np.array([b[2] for b in bonds], float)
    bond_k = np.where(rng.random(len(bonds)) < 0.5, 1.0, 1.0/ratio)
    e01, e02 = edge_vecs[:, 0], edge_vecs[:, 1]
    return dict(N=N, pts=pts, simplices=simp, edge_vecs=edge_vecs, actual_len2=l2,
                bond_u=bu, bond_v=bv, bond_R=bR, bond_k=bond_k, tri_k=bond_k[tri_bond],
                areas=0.5*np.abs(e01[:, 0]*e02[:, 1] - e01[:, 1]*e02[:, 0]))


def main():
    Fk = [np.eye(2)+DELTA*M for M in MODES]; Dgt = [F.T@F-np.eye(2) for F in Fk]
    Dinv = np.linalg.inv(np.stack([CE.vec3(g) for g in Dgt], 1))
    out = {k: [] for k in ['nu_s', 'E_s', 'nu_m', 'E_m', 'nu_c', 'E_c']}
    print(f"N={N}, rigidity contrast={RATIO:.0f}, cluster d={DCL}.  nu / E  (sim | MF | cluster)")
    for eta in ETAS:
        acc = {k: [] for k in out}
        for seed in range(2):
            mesh = build_mesh_eta_k(N, eta, RATIO, seed)
            ev, sx = mesh['edge_vecs'], mesh['simplices']
            nn = len(mesh['pts']); nt = len(sx); bare = TR.bare_tensor(mesh)
            K, _ = TR.assemble_K_faff(mesh, np.eye(2))
            faff = [TR.assemble_K_faff(mesh, F)[1] for F in Fk]
            free = np.arange(2, 2*nn)
            Ds = np.zeros((nt, 3, 3))
            for k, F in enumerate(Fk):
                u = np.zeros(2*nn); u[free] = spla.spsolve(K[free][:, free].tocsc(), -faff[k][free])
                Ds[:, :, k] = CE.vec3(CE.tri_metric_change(ev, sx, F, u.reshape(nn, 2)) - Dgt[k])
            nu_s, E_s = CE.Ceff_nuE(mesh, Ds @ Dinv, bare)
            nu_m, E_m = CE.Ceff_nuE(mesh, RG.mf_W3(bare), bare)
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
            Dc = np.zeros((nt, 3, 3))
            for c in range(nt):
                n0, n1, n2 = int(sx[c, 0]), int(sx[c, 1]), int(sx[c, 2])
                fr = np.array(sorted(ring([n0, n1, n2], DCL)))
                fdof = np.sort(np.concatenate([2*fr, 2*fr+1])); Kff = K[fdof][:, fdof].tocsc()
                for k, F in enumerate(Fk):
                    u = np.zeros((nn, 2)); u.ravel()[fdof] = spla.spsolve(Kff, -faff[k][fdof])
                    Dc[c, :, k] = CE.vec3(RG.central_dg(ev, c, F, u, n0, n1, n2, Dgt[k]))
            nu_c, E_c = CE.Ceff_nuE(mesh, Dc @ Dinv, bare)
            for kk, vv in zip(out, [nu_s, E_s, nu_m, E_m, nu_c, E_c]):
                acc[kk].append(vv)
        for kk in out:
            out[kk].append(np.nanmean(acc[kk]))
        print(f"  eta={eta:.1f}: nu {out['nu_s'][-1]:+.3f}/{out['nu_m'][-1]:+.3f}/{out['nu_c'][-1]:+.3f}"
              f"   E {out['E_s'][-1]:.4f}/{out['E_m'][-1]:.4f}/{out['E_c'][-1]:.4f}", flush=True)

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for a, key, ttl in [(ax[0], 'nu', 'Poisson ratio ν'), (ax[1], 'E', "Young's modulus E")]:
        a.plot(ETAS, out[f'{key}_s'], 'k-o', lw=2.5, label='simulation')
        a.plot(ETAS, out[f'{key}_m'], '--s', color='#d62728', label='single-site MF')
        a.plot(ETAS, out[f'{key}_c'], '-^', color='#2ca02c', label=f'cluster d={DCL}')
        a.set_xlabel('η'); a.set_title(ttl); a.legend(fontsize=10); a.grid(alpha=0.3)
    ax[0].axhline(0, color='gray', lw=0.5, ls=':')
    fig.suptitle(f'Combined disorder: rigidity contrast {RATIO:.0f}× + geometric η — code vs simulation',
                 fontsize=12)
    plt.tight_layout()
    p = os.path.join(HERE, 'plots', 'dg_cluster_combined.png')
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print('saved', p)


if __name__ == '__main__':
    main()
