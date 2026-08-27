"""Gate the v3 tensor-message model: equivariance, invariance, SPD -- before training."""
import os, sys, warnings
import numpy as np
R = r'c:\Users\doron\Documents\GitHub\MATERIALIZE'
sys.path.insert(0, os.path.join(R,'Phase 3','verifications')); import _common as C
sys.path.insert(0, os.path.join(R,'Phase 5')); sys.path.insert(0, os.path.join(R,'Phase 5','m2'))
import torch, seeds as S, model_v2 as M2, model_v3 as M3
warnings.simplefilter('ignore'); torch.set_num_threads(1); torch.set_default_dtype(torch.float64)


def build(geo, k, rot=0.0):
    """Everything v3 consumes, from (edge vectors, k, l0) alone."""
    R2 = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])
    bR = np.asarray(geo['bond_R']) @ R2.T
    tb = np.asarray(geo['tri_bond']).astype(np.int64)
    lbar = float(np.hypot(bR[:, 0], bR[:, 1]).mean())
    ev = bR[tb] / lbar
    Q = M2.edge_carriers(ev)
    l0 = np.hypot(bR[:, 0], bR[:, 1]) / lbar          # l0 == l here (zero prestress)
    kk = np.asarray(k) / np.mean(k)
    src, dst, bnd = M3.triangle_adjacency(tb, len(tb))
    bq = M2.edge_carriers((bR / lbar)[:, None, :])[:, :, 0]        # (n_bond, 3) vec3 carriers
    return dict(Q=Q,
                tri_scalars=torch.as_tensor(np.concatenate([kk[tb], l0[tb], np.log(kk[tb])], 1)),
                tri_src=src, tri_dst=dst, bond_q=bq[bnd],
                bond_feat=torch.as_tensor(np.stack([kk[bnd], l0[bnd]], 1)))


geo = S.random_patch(60, seed=0)['geo']
k = np.exp(np.random.default_rng(0).normal(0, .8, len(geo['bond_u'])))
net = M3.ForwardGNNv3(ns=16, nt=6, hidden=32, n_layers=2)
print('v3 parameters: %d' % sum(p.numel() for p in net.parameters()))

g0 = build(geo, k, 0.0)
with torch.no_grad():
    G0 = net(g0); C0, _ = M2.assemble(g0['Q'], G0)

print('\n[1] triangle adjacency built: %d directed edges over %d triangles (expect ~3 per triangle)'
      % (len(g0['tri_src']), len(g0['Q'])))

for th in (0.4, 1.9):
    gr = build(geo, k, th)
    with torch.no_grad():
        Gr = net(gr); Cr, _ = M2.assemble(gr['Q'], Gr)
    Rv = M2.vec3_rotation(th)
    dG = float((Gr - G0).abs().max())
    dC = float((Cr - Rv @ C0 @ Rv.T).abs().max() / C0.abs().max())
    print('[2] theta=%.2f  G INVARIANT: max|dG| = %.2e   C EQUIVARIANT: rel %.2e' % (th, dG, dC))

eig = torch.linalg.eigvalsh(0.5 * (C0 + C0.transpose(-1, -2)))
print('[3] SPD by construction: min eig(C) = %.2e' % float(eig.min()))

with torch.no_grad():
    net2 = M3.ForwardGNNv3(ns=16, nt=6, hidden=32, n_layers=0)
    g = build(geo, k, 0.0)
    print('[4] n_layers=0 runs (no message passing): %s' % (tuple(net2(g).shape),))

# does the tensor channel actually carry orientation? perturb ONE neighbour and see if G moves
geo2 = dict(geo); pts = np.asarray(geo['pts']).copy()
b = np.asarray(geo['bond_R']).copy(); b[5] = b[5] @ np.array([[0.9, .3], [-.3, 0.9]]).T
geo2['bond_R'] = b
with torch.no_grad():
    Gp = net(build(geo2, k, 0.0))
moved = (Gp - G0).abs().max(-1).values.max(-1).values
print('[5] perturbing ONE bond changes G on %d of %d triangles (max |dG| = %.2e)'
      % (int((moved > 1e-12).sum()), len(moved), float(moved.max())))
