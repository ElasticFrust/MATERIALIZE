"""
nu / E vs signed (virtual-distortion) rigidity contrast a in [-10, 10].
Perturbed periodic lattice (eta fixed); bond rigidity k = 1 + tanh(a*(|R|-1)) (a=0 uniform,
a>0 stiffer where stretched, a<0 stiffer where compressed; |a|~10 -> near-binary k in {0,2}).
Homogenised nu and E for the cluster forward solver (code) and single-site MF vs PBC sim.
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
torch.set_default_dtype(torch.float64)

N = 14
CONTRASTS = [-10, -5, 0, 5, 10]
ETAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
DCL = 3
DELTA = CE.DELTA
MODES = CE.MODES


def build_geometry(N, eta, seed):
    """Perturbed periodic triangular lattice; returns geometry + tri->bond map (no rigidity)."""
    rng = np.random.default_rng(seed)
    L1 = np.array([1.0, 0.0]); L2 = np.array([0.5, np.sqrt(3)/2]); BL1, BL2 = N*L1, N*L2
    nn, mm = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
    ref = nn.ravel()[:, None]*L1 + mm.ravel()[:, None]*L2
    ang = rng.uniform(0, 2*np.pi, N*N)
    pts = ref + eta*np.stack([np.cos(ang), np.sin(ang)], 1)
    idx = lambda a, b: (a % N)*N + (b % N)
    simp, img = [], []
    for a in range(N):
        for b in range(N):
            simp.append([idx(a, b), idx(a+1, b), idx(a, b+1)])
            img.append([[0, 0], [1 if a+1 >= N else 0, 0], [0, 1 if b+1 >= N else 0]])
            simp.append([idx(a+1, b), idx(a+1, b+1), idx(a, b+1)])
            img.append([[1 if a+1 >= N else 0, 0],
                        [1 if a+1 >= N else 0, 1 if b+1 >= N else 0], [0, 1 if b+1 >= N else 0]])
    simp = np.array(simp, np.int64); img = np.array(img, np.int64); n_tri = len(simp)
    vpos = lambda k: pts[simp[:, k]] + img[:, k, 0:1]*BL1 + img[:, k, 1:2]*BL2
    p0, p1, p2 = vpos(0), vpos(1), vpos(2)
    edge_vecs = np.stack([p1-p0, p2-p0, p2-p1], 1); l2 = (edge_vecs**2).sum(2)
    pairs = [(0, 1, 0), (0, 2, 1), (1, 2, 2)]; keymap = {}; tri_bond = np.zeros((n_tri, 3), np.int64); bonds = []
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
    e01, e02 = edge_vecs[:, 0], edge_vecs[:, 1]
    return dict(N=N, pts=pts, simplices=simp, edge_vecs=edge_vecs, actual_len2=l2,
                bond_u=np.array([b[0] for b in bonds], np.int64),
                bond_v=np.array([b[1] for b in bonds], np.int64),
                bond_R=np.array([b[2] for b in bonds], float),
                tri_bond=tri_bond,
                areas=0.5*np.abs(e01[:, 0]*e02[:, 1] - e01[:, 1]*e02[:, 0]))


def set_VD(mesh, a):
    dl = np.sqrt((mesh['bond_R']**2).sum(1)) - 1.0          # |R| - ideal spacing
    mesh['bond_k'] = 1.0 + np.tanh(a*dl)
    mesh['tri_k'] = mesh['bond_k'][mesh['tri_bond']]


def nuE_all(mesh, Fk, Dgt, Dinv):
    ev, sx = mesh['edge_vecs'], mesh['simplices']; nn = len(mesh['pts']); nt = len(sx)
    bare = TR.bare_tensor(mesh)
    K, _ = TR.assemble_K_faff(mesh, np.eye(2)); faff = [TR.assemble_K_faff(mesh, F)[1] for F in Fk]
    free = np.arange(2, 2*nn); Ds = np.zeros((nt, 3, 3))
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
        fr = np.array(sorted(ring([n0, n1, n2], DCL))); fdof = np.sort(np.concatenate([2*fr, 2*fr+1]))
        Kff = K[fdof][:, fdof].tocsc()
        for k, F in enumerate(Fk):
            u = np.zeros((nn, 2)); u.ravel()[fdof] = spla.spsolve(Kff, -faff[k][fdof])
            Dc[c, :, k] = CE.vec3(RG.central_dg(ev, c, F, u, n0, n1, n2, Dgt[k]))
    nu_c, E_c = CE.Ceff_nuE(mesh, Dc @ Dinv, bare)
    return nu_s, E_s, nu_m, E_m, nu_c, E_c


def main():
    Fk = [np.eye(2)+DELTA*M for M in MODES]; Dgt = [F.T@F-np.eye(2) for F in Fk]
    Dinv = np.linalg.inv(np.stack([CE.vec3(g) for g in Dgt], 1))
    # res[a] = dict of lists over eta
    res = {a: {k: [] for k in ['nu_s', 'E_s', 'nu_m', 'E_m', 'nu_c', 'E_c']} for a in CONTRASTS}
    print(f"N={N}, VD rigidity k=1+tanh(a*(|R|-1)), cluster d={DCL}")
    for eta in ETAS:
        geo = build_geometry(N, eta, seed=0)
        for a in CONTRASTS:
            set_VD(geo, a)
            ns, Es, nm, Em, nc, Ec = nuE_all(geo, Fk, Dgt, Dinv)
            for kk, vv in zip(res[a], [ns, Es, nm, Em, nc, Ec]):
                res[a][kk].append(vv)
            print(f"  eta={eta:.1f} a={a:>+3}: nu sim/MF/cl = {ns:+.3f}/{nm:+.3f}/{nc:+.3f}"
                  f"   E = {Es:.4f}/{Em:.4f}/{Ec:.4f}", flush=True)

    nC = len(CONTRASTS)
    fig, axes = plt.subplots(nC, 2, figsize=(11, 3.0*nC), squeeze=False)
    for i, a in enumerate(CONTRASTS):
        for j, (key, ttl) in enumerate([('nu', 'ν'), ('E', 'E')]):
            ax = axes[i, j]
            ax.plot(ETAS, res[a][f'{key}_s'], 'k-o', lw=2.2, ms=4, label='sim')
            ax.plot(ETAS, res[a][f'{key}_m'], '--s', color='#d62728', ms=4, label='MF')
            ax.plot(ETAS, res[a][f'{key}_c'], '-^', color='#2ca02c', ms=4, label=f'cluster d={DCL}')
            if key == 'nu':
                ax.axhline(0, color='gray', lw=0.5, ls=':')
            ax.set_ylabel(f'{ttl}  (a={a:+d})'); ax.grid(alpha=0.3)
            if i == 0:
                ax.set_title(f'{ttl} vs η'); ax.legend(fontsize=8)
            if i == nC-1:
                ax.set_xlabel('η')
    fig.suptitle('ν and E vs η at several VD rigidity contrasts a — code (cluster) vs simulation',
                 fontsize=12)
    plt.tight_layout()
    p = os.path.join(HERE, 'plots', 'dg_cluster_VD_eta.png')
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print('saved', p)


if __name__ == '__main__':
    main()
