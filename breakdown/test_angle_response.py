"""
Implement and test the vertex-angle (Gaussian-curvature / full St-Venant) response.

Adds the 'angle' KKT block on top of edge-KKT: for each interior vertex the sum of the
incident triangles' angle CHANGES must vanish (linearized flatness). Built to match the
Phase 2 solver convention (_angle_gradient_vec + _build_vertex_angle_constraints) and fed
to _woodbury_kkt_sparse_combined. All vertices are interior on the periodic torus.

Compares Std / +edge / +full (edge+angle):
  - edge residual   q.(dg[s1]-dg[s2])                     (edge compatibility)
  - angle residual  sum_{s at v} a.dg_s  per interior vertex (vertex/flatness compatibility)
  - overshoot ||dg_MF||/||dg_sim|| and corr vs sim, on well-shaped triangles
to see whether enforcing full compatibility closes the overshoot + direction gap.
"""
import os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
sys.path.insert(0, os.path.join(ROOT, 'Phase 2'))
import forward_solver_torch as fst  # noqa
torch.set_default_dtype(torch.float64)
DATA = os.path.join(HERE, 'dg_analysis_data')


def angle_grad(a, b):                       # == solver _angle_gradient_vec
    a2, b2, ab = a @ a, b @ b, a @ b
    la, lb = np.sqrt(a2), np.sqrt(b2)
    ct = np.clip(ab / (la * lb), -1 + 1e-10, 1 - 1e-10)
    st = np.sqrt(1 - ct ** 2)
    if st < 1e-10:
        return np.zeros(3)
    d_ab = np.array([a[0]*b[0], a[0]*b[1] + a[1]*b[0], a[1]*b[1]])
    d_a2 = np.array([a[0]**2, 2*a[0]*a[1], a[1]**2])
    d_b2 = np.array([b[0]**2, 2*b[0]*b[1], b[1]**2])
    d_cos = d_ab / (la * lb) - ct * (d_a2 / (2*a2) + d_b2 / (2*b2))
    return -d_cos / st


def build_angle_arrays(simplices, edge_vecs, n_node, half_shear=True):
    v2 = [[] for _ in range(n_node)]
    for s in range(len(simplices)):
        for vi in range(3):
            v2[int(simplices[s, vi])].append((s, vi))
    vl, sl, al = [], [], []
    for v in range(n_node):                 # all interior on the torus
        for (s, vi) in v2[v]:
            e01, e02, e12 = edge_vecs[s, 0], edge_vecs[s, 1], edge_vecs[s, 2]
            a, b = {0: (e01, e02), 1: (e12, -e01), 2: (-e02, -e12)}[vi]
            d = angle_grad(a, b).copy()
            if half_shear:
                d[1] /= 2                    # solver's "engineering" convention
            vl.append(v); sl.append(s); al.append(d)
    return (np.array(vl, np.int64), np.array(sl, np.int64), np.array(al, float), n_node)


def bare_from_edges(ev, l2):
    vx, vy = ev[:, :, 0], ev[:, :, 1]
    fac = 1.0 / np.maximum(l2, 1e-30) / 16.0
    return np.stack([(fac*vx**4).sum(1), (fac*vx**3*vy).sum(1), (fac*vx**2*vy**2).sum(1),
                     (fac*vx*vy**3).sum(1), (fac*vy**4).sum(1)], 1)


def getW(bare, areas, kkt, angle, aw):
    w = (areas / areas.sum()) if aw else None
    mean = (bare * w[:, None]).sum(0) if aw else bare.mean(0)
    db = bare - mean
    A = fst._batch_to_9x9(torch.as_tensor(bare))
    B = fst._batch_to_9x9(torch.as_tensor(db))
    dA = fst._batch_to_9vec(torch.as_tensor(db))
    if kkt is None and angle is None:
        wt = torch.as_tensor(w) if w is not None else None
        return fst._woodbury_solve(A, B, dA, J=None, weights=wt).detach().numpy()
    return fst._woodbury_kkt_sparse_combined(A, B, dA, kkt, angle, weights=w)


def dg_from_W(W, Dg):
    n = len(W); W3 = W.reshape(n, 3, 3)
    v = W3 @ np.array([Dg[0, 0], Dg[0, 1], Dg[1, 1]])
    g = np.empty((n, 2, 2))
    g[:, 0, 0] = v[:, 0]; g[:, 1, 1] = v[:, 2]; g[:, 0, 1] = g[:, 1, 0] = v[:, 1]
    return g


def minang(ev):
    e01, e02, e12 = ev[:, 0], ev[:, 1], ev[:, 2]
    def a(x, y):
        c = (x*y).sum(1) / np.sqrt(np.maximum((x**2).sum(1)*(y**2).sum(1), 1e-30))
        return np.degrees(np.arccos(np.clip(c, -1, 1)))
    a0, a1 = a(e01, e02), a(-e01, e12)
    return np.minimum(np.minimum(a0, a1), np.abs(180 - a0 - a1))


def fro(a, b):
    return (a * b).sum((1, 2))


def main():
    print(f"{'eta':>5} {'method':>9} | {'edge_res':>9} {'angle_res':>9} | "
          f"{'overshoot':>9} {'corr':>6}  (well-shaped >30deg)")
    for eta in (0.1, 0.2, 0.3, 0.4, 0.5):
        agg = {}
        for t in range(5):
            f = os.path.join(DATA, f'sample_eta{eta:.2f}_trial{t}.npz')
            if not os.path.exists(f):
                continue
            s = np.load(f, allow_pickle=True)
            ev, sx, Dg, sim = s['edge_vecs'], s['simplices'], s['Delta_g'], s['dg_sim']
            areas = s['areas']; n_node = len(s['pts'])
            kkt = (s['kkt_s1'], s['kkt_s2'], s['kkt_q'])
            ang = build_angle_arrays(sx, ev, n_node, half_shear=False)  # tensor-consistent with q
            bare = bare_from_edges(ev, s['actual_len2'])
            q = s['kkt_q']; s1, s2 = s['kkt_s1'], s['kkt_s2']
            va, sv, av, _ = ang
            g = minang(ev) > 30
            for name, kk, an in [('Std', None, None), ('+edge', kkt, None), ('+full', kkt, ang)]:
                W = getW(bare, areas, kk, an, aw=False)
                dg = dg_from_W(W, Dg)
                dv = np.stack([dg[:, 0, 0], dg[:, 0, 1], dg[:, 1, 1]], 1)
                # edge residual
                er = (q * (dv[s1] - dv[s2])).sum(1)
                esc = np.sqrt(np.mean((q @ np.array([Dg[0, 0], Dg[0, 1], Dg[1, 1]]))**2))
                eR = np.sqrt(np.mean(er**2)) / esc
                # angle residual: per vertex sum of physical d(theta)=a.[dg11,dg12,dg22]
                dth = (av * np.stack([dv[sv, 0], dv[sv, 1], dv[sv, 2]], 1)).sum(1)
                vsum = np.bincount(va, weights=dth, minlength=n_node)
                asc = np.sqrt(np.mean(dth**2))
                aR = np.sqrt(np.mean(vsum**2)) / max(asc, 1e-30)
                ov = np.sqrt(fro(dg, dg)[g]).mean() / np.sqrt(fro(sim, sim)[g]).mean()
                aa = np.stack([dg[g, 0, 0], dg[g, 0, 1], dg[g, 1, 1]], 1).ravel()
                bb = np.stack([sim[g, 0, 0], sim[g, 0, 1], sim[g, 1, 1]], 1).ravel()
                co = np.corrcoef(aa, bb)[0, 1]
                agg.setdefault(name, []).append((eR, aR, ov, co))
        for name in ('Std', '+edge', '+full'):
            v = np.array(agg[name]).mean(0)
            print(f"{eta:>5.1f} {name:>9} | {v[0]:>9.2e} {v[1]:>9.2e} | "
                  f"{v[2]:>9.2f} {v[3]:>6.2f}")
        print()


if __name__ == '__main__':
    main()
