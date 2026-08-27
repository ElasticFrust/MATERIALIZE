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

# [5] RECEPTIVE FIELD. Perturb the PREPARED tensors, not the geometry: `build` normalises lengths by
# `lbar` and k by `mean(k)`, both GLOBAL, so perturbing an input moves every triangle through the
# normaliser and says nothing about message passing. (An earlier version of this test did exactly
# that and "showed" influence on 112 of 112 triangles at n_layers=2 -- impossible, and the giveaway
# that the test, not the model, was wrong.) Perturbing the prepared arrays isolates the graph: with
# L layers the influence must reach EXACTLY the triangles within L hops and leave the rest
# bit-identical. That is also the claim `M2_LOCALITY.md` rests on, so it is worth gating.
def hop_distance(src, dst, roots, n):
    """BFS hop count over the triangle adjacency; -1 = unreachable."""
    d = np.full(n, -1)
    d[list(roots)] = 0
    src, dst = np.asarray(src), np.asarray(dst)
    for h in range(n):
        frontier = np.where(d == h)[0]
        if not len(frontier):
            break
        nbr = dst[np.isin(src, frontier)]
        d[nbr[d[nbr] < 0]] = h + 1
    return d


BOND = 5                                                   # the one bond whose k is perturbed
tri_bond = np.asarray(geo['tri_bond'])
owns = (tri_bond == BOND).any(1)                           # triangles having BOND as an edge
src_np, dst_np = np.asarray(g0['tri_src']), np.asarray(g0['tri_dst'])
# message slots whose SHARED bond is BOND: both endpoints own it
slots = np.where(owns[src_np] & owns[dst_np])[0]

for L in (1, 2):
    netL = M3.ForwardGNNv3(ns=16, nt=6, hidden=32, n_layers=L)
    with torch.no_grad():
        base = netL(g0)
        gp = dict(g0)
        # raise k on BOND wherever it enters: the owning triangles' own scalars, and the messages
        # carried across it -- exactly the two places a real change in k_BOND would appear
        gp['bond_feat'] = g0['bond_feat'].clone()
        gp['bond_feat'][slots, 0] *= 1.5
        gp['tri_scalars'] = g0['tri_scalars'].clone()
        for t in np.where(owns)[0]:
            col = int(np.where(tri_bond[t] == BOND)[0][0])
            gp['tri_scalars'][t, col] *= 1.5               # k/kbar block occupies columns 0..2
        moved = (netL(gp) - base).abs().amax(-1).amax(-1).numpy()
    d = hop_distance(g0['tri_src'], g0['tri_dst'], np.where(owns)[0], len(base))
    inside = (d >= 0) & (d <= L)
    beyond = int(((moved > 1e-12) & ~inside).sum())
    print('[5] L=%d: %d/%d triangles within %d hops moved; %d beyond the receptive field '
          '(MUST be 0)' % (L, int(((moved > 1e-12) & inside).sum()), int(inside.sum()), L, beyond))
    assert beyond == 0, 'influence leaked past the %d-hop receptive field' % L
