"""
Heterogeneous-rigidity cluster test — the documented single-site catastrophe.

Regular periodic triangular lattice (eta=0, so geometry is uniform and the non-affine
response is driven PURELY by rigidity contrast). Each bond gets a random binary rigidity
k in {1, 1/ratio} (stiff/soft). Compare, per triangle, the non-affine metric change dg:
  - single-site MF (Woodbury W, bare tensor with the actual k)
  - cluster radius d (relax a local patch with the actual k, boundary affine)
  - simulation (full PBC relax with the actual k)
The single-site MF is documented to fail badly here (~70% error on nu); the question is
whether the local cluster recovers the simulation as it did for geometric disorder.
"""
import os, sys
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
sys.path.insert(0, os.path.join(ROOT, 'Phase 2'))
import forward_solver_torch as fst
import torch
torch.set_default_dtype(torch.float64)
DELTA = 1e-3
H = np.array([[1.0, 0.0], [0.0, 0.0]])


def build_mesh_k(N, ratio, seed):
    """Regular periodic triangular lattice + random binary bond rigidities.

    Returns mesh dict with per-bond k (bond_k) and per-triangle-edge k (tri_k, (N_tri,3))
    consistently assigned by canonical bond key."""
    rng = np.random.default_rng(seed)
    L1 = np.array([1.0, 0.0]); L2 = np.array([0.5, np.sqrt(3)/2])
    BL1, BL2 = N*L1, N*L2
    nn, mm = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
    pts = nn.ravel()[:, None]*L1 + mm.ravel()[:, None]*L2          # eta=0, regular
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
    simp = np.array(simp, np.int64); img = np.array(img, np.int64)
    n_tri = len(simp)
    vpos = lambda k: pts[simp[:, k]] + img[:, k, 0:1]*BL1 + img[:, k, 1:2]*BL2
    p0, p1, p2 = vpos(0), vpos(1), vpos(2)
    edge_vecs = np.stack([p1-p0, p2-p0, p2-p1], axis=1)
    l2 = (edge_vecs**2).sum(2)
    pairs = [(0, 1, 0), (0, 2, 1), (1, 2, 2)]
    # canonical key per (triangle, local edge)
    keymap = {}                      # key -> bond index
    tri_bond = np.zeros((n_tri, 3), np.int64)
    bond_list = []
    for ti in range(n_tri):
        for ka, kb, ei in pairs:
            va, vb = int(simp[ti, ka]), int(simp[ti, kb])
            d = img[ti, ka]-img[ti, kb]; dp = (int(d[0]), int(d[1]))
            if (va, dp[0], dp[1]) <= (vb, -dp[0], -dp[1]):
                key, R = (va, vb, dp[0], dp[1]), edge_vecs[ti, ei]
            else:
                key, R = (vb, va, -dp[0], -dp[1]), -edge_vecs[ti, ei]
            if key not in keymap:
                keymap[key] = len(bond_list)
                bond_list.append((key[0], key[1], R))
            tri_bond[ti, ei] = keymap[key]
    bu = np.array([b[0] for b in bond_list], np.int64)
    bv = np.array([b[1] for b in bond_list], np.int64)
    bR = np.array([b[2] for b in bond_list], float)
    n_bond = len(bond_list)
    bond_k = np.where(rng.random(n_bond) < 0.5, 1.0, 1.0/ratio)    # binary stiff/soft
    tri_k = bond_k[tri_bond]                                        # (n_tri,3)
    return dict(N=N, pts=pts, simplices=simp, edge_vecs=edge_vecs, actual_len2=l2,
                bond_u=bu, bond_v=bv, bond_R=bR, bond_k=bond_k, tri_k=tri_k)


def assemble_K_faff(mesh, F):
    bu, bv, R, k = mesh['bond_u'], mesh['bond_v'], mesh['bond_R'], mesh['bond_k']
    Nn = len(mesh['pts']); l2 = (R**2).sum(1)
    S = k[:, None, None]*np.einsum('bp,bq->bpq', R, R)/l2[:, None, None]   # k * Rhat⊗Rhat
    nb = len(bu); pp = np.array([0, 0, 1, 1]); qq = np.array([0, 1, 0, 1])
    def blk(i, j, sg):
        return ((2*i[:, None]+pp).ravel(), (2*j[:, None]+qq).ravel(),
                (sg*S.reshape(nb, 4)).ravel())
    R_, C_, V_ = [], [], []
    for ia, ja, sg in [(bu, bu, 1.), (bv, bv, 1.), (bu, bv, -1.), (bv, bu, -1.)]:
        r, c, v = blk(ia, ja, sg); R_.append(r); C_.append(c); V_.append(v)
    K = sp.coo_matrix((np.concatenate(V_), (np.concatenate(R_), np.concatenate(C_))),
                      shape=(2*Nn, 2*Nn)).tocsc()
    HR = R @ (F-np.eye(2)).T
    fb = np.einsum('bpq,bq->bp', S, HR)
    faff = np.zeros((Nn, 2)); np.add.at(faff, bv, fb); np.add.at(faff, bu, -fb)
    return K, faff.ravel()


def bare_tensor(mesh):
    ev, k, l2 = mesh['edge_vecs'], mesh['tri_k'], mesh['actual_len2']
    vx, vy = ev[:, :, 0], ev[:, :, 1]; fac = k/np.maximum(l2, 1e-30)/16.0
    return np.stack([(fac*vx**4).sum(1), (fac*vx**3*vy).sum(1), (fac*vx**2*vy**2).sum(1),
                     (fac*vx*vy**3).sum(1), (fac*vy**4).sum(1)], 1)


def mf_dg(mesh, Dg):
    bare = bare_tensor(mesh); db = bare-bare.mean(0)
    A = fst._batch_to_9x9(torch.as_tensor(bare)); B = fst._batch_to_9x9(torch.as_tensor(db))
    dA = fst._batch_to_9vec(torch.as_tensor(db))
    W = fst._woodbury_solve(A, B, dA, J=None, weights=None).detach().numpy()
    n = len(W); W3 = W.reshape(n, 3, 3)
    v = W3 @ np.array([Dg[0, 0], Dg[0, 1], Dg[1, 1]])
    g = np.empty((n, 2, 2)); g[:, 0, 0] = v[:, 0]; g[:, 1, 1] = v[:, 2]
    g[:, 0, 1] = g[:, 1, 0] = v[:, 1]
    return g


def central_dg(mesh, F, Dg, u, c):
    sx = mesh['simplices']; ev = mesh['edge_vecs']
    n0, n1, n2 = sx[c]; a, b = ev[c, 0], ev[c, 1]
    Eref = np.array([[a[0], b[0]], [a[1], b[1]]])
    ad = a @ F.T + (u[n1]-u[n0]); bd = b @ F.T + (u[n2]-u[n0])
    Edef = np.array([[ad[0], bd[0]], [ad[1], bd[1]]])
    Fc = Edef @ np.linalg.inv(Eref)
    return Fc.T @ Fc - np.eye(2) - Dg


def fro(a, b):
    return (a*b).sum((1, 2))


def main():
    N = 30; F = np.eye(2)+DELTA*H; Dg = 0.5*(F.T@F+F@F.T)/1  # use F^T F - I below
    Dg = F.T @ F - np.eye(2)
    print("Regular lattice (eta=0), binary rigidities; per-triangle non-affine dg vs sim")
    print(f"{'ratio':>6} | {'MF corr':>8} {'MF over':>8} | {'d=1 corr':>9} {'d=1 over':>9} "
          f"| {'d=2 corr':>9} {'d=2 over':>9}")
    for ratio in (3.0, 10.0, 100.0):
        rows = {'mf': [], 'd1': [], 'd2': []}
        for seed in range(3):
            mesh = build_mesh_k(N, ratio, seed)
            n_node = len(mesh['pts']); n_tri = len(mesh['simplices'])
            K, faff = assemble_K_faff(mesh, F); faff = faff.reshape(n_node, 2)
            # sim: full relax (pin node 0)
            free = np.arange(2, 2*n_node)
            u = np.zeros(2*n_node)
            u[free] = spla.spsolve(K[free][:, free].tocsc(), -faff.ravel()[free])
            u = u.reshape(n_node, 2)
            sim = np.array([central_dg(mesh, F, Dg, u, c) for c in range(n_tri)])
            mf = mf_dg(mesh, Dg)
            # cluster
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
            cells = np.random.default_rng(seed).choice(n_tri, size=min(120, n_tri), replace=False)
            cl = {1: [], 2: []}; simc = []; mfc = []
            for c in cells:
                sx = mesh['simplices']
                for d in (1, 2):
                    fr = np.array(sorted(ring([int(sx[c, 0]), int(sx[c, 1]), int(sx[c, 2])], d)))
                    fdof = np.sort(np.concatenate([2*fr, 2*fr+1]))
                    uf = np.zeros(2*n_node)
                    uf[fdof] = spla.spsolve(K[fdof][:, fdof].tocsc(), -faff.ravel()[fdof])
                    cl[d].append(central_dg(mesh, F, Dg, uf.reshape(n_node, 2), c))
                simc.append(sim[c]); mfc.append(mf[c])
            simc = np.array(simc); mfc = np.array(mfc)
            def cc(A):
                a = np.stack([A[:, 0, 0], A[:, 0, 1], A[:, 1, 1]], 1).ravel()
                b = np.stack([simc[:, 0, 0], simc[:, 0, 1], simc[:, 1, 1]], 1).ravel()
                return np.corrcoef(a, b)[0, 1]
            def ov(A):
                return np.median(np.sqrt(fro(A, A))/np.maximum(np.sqrt(fro(simc, simc)), 1e-30))
            rows['mf'].append((cc(mfc), ov(mfc)))
            rows['d1'].append((cc(np.array(cl[1])), ov(np.array(cl[1]))))
            rows['d2'].append((cc(np.array(cl[2])), ov(np.array(cl[2]))))
        m = {k: np.mean(v, 0) for k, v in rows.items()}
        print(f"{ratio:>6.0f} | {m['mf'][0]:>8.2f} {m['mf'][1]:>8.2f} | "
              f"{m['d1'][0]:>9.2f} {m['d1'][1]:>9.2f} | {m['d2'][0]:>9.2f} {m['d2'][1]:>9.2f}")


if __name__ == '__main__':
    main()
