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
    # the CONSTRAINT channels must be built here too, or the gates silently skip them:
    # `forward` guards on `'star_tri' in g` / `'tri_batch' in g`, so a build() that omits them
    # would test the bond-only model while reporting on the constrained one.
    st_t, st_v, st_w, n_vert = M3.vertex_stars(geo['simplices'] if 'simplices' in geo
                                               else geo['tri_verts'],
                                               tb, geo['bond_u'], geo['bond_v'], bR / lbar)
    areas = np.asarray(geo['areas'], float)
    return dict(Q=Q,
                tri_scalars=torch.as_tensor(np.concatenate([kk[tb], l0[tb], np.log(kk[tb])], 1)),
                tri_src=src, tri_dst=dst, bond_q=bq[bnd],
                bond_feat=torch.as_tensor(np.stack([kk[bnd], l0[bnd]], 1)),
                star_tri=torch.as_tensor(st_t), star_vert=torch.as_tensor(st_v),
                star_w=torch.as_tensor(st_w), n_vert=int(n_vert),
                area_w=torch.as_tensor(areas / max(areas.sum(), 1e-30)),
                tri_batch=torch.zeros(len(tb), dtype=torch.long), n_graphs=1)


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

# [6] THE RESIDUAL HEAD'S DEFINING PROPERTY. `G = (I+X) G_an (I+X)^T` with the readout
# zero-initialised means X = 0 at step 0, so an UNTRAINED net must return the ANALYTIC tensor
# EXACTLY -- not approximately. This promotes `train_v3.oracle_check`'s property (which needs W=0
# data to test) into an architectural invariant testable on any mesh. If it fails, the model is no
# longer starting from the closed form and the whole point of the head is lost.
with torch.no_grad():
    fresh = M3.ForwardGNNv3(ns=16, nt=6, hidden=32, n_layers=2)
    g = build(geo, k, 0.0)
    G_pred = fresh(g)
    G_an = M3.ForwardGNNv3.analytic_G(g)
    rel = float((G_pred - G_an).abs().max() / G_an.abs().max())
    print('[6] untrained head == analytic A(s): rel %.2e  (X=0 by zero-init)' % rel)
    assert rel < 1e-14, 'residual head does not start at the analytic tensor (rel %.3e)' % rel
    # and it must MOVE once X is nonzero -- otherwise the correction is unreachable
    for p in fresh.readout[-1].parameters():
        p.add_(torch.randn_like(p) * 0.05)
    moved = float((fresh(g) - G_an).abs().max() / G_an.abs().max())
    print('    nonzero X moves it: rel %.2e  (must be >> 0)' % moved)
    assert moved > 1e-6, 'X has no effect on G -- the correction channel is dead'

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

# The reach is now the UNION of two couplings: bond adjacency (J_edge) and the vertex star
# (C_curv), because a StarMP round reaches every triangle sharing a VERTEX, not just an edge.
simp = np.asarray(geo['simplices'] if 'simplices' in geo else geo['tri_verts'], np.int64)
_vs, _vd = [], []
for _v in np.unique(simp):
    _ts = np.where((simp == _v).any(1))[0]
    for _i in _ts:
        for _j in _ts:
            if _i != _j:
                _vs.append(_i); _vd.append(_j)
UNION_SRC = np.concatenate([np.asarray(g0['tri_src']), np.array(_vs, np.int64)])
UNION_DST = np.concatenate([np.asarray(g0['tri_dst']), np.array(_vd, np.int64)])

for L in (1, 2):
    # use_global=False: M_S is a GLOBAL rank-3 constraint, so with it on every triangle is reachable
    # in one layer BY DESIGN and a finite-reach test is meaningless. Its own signature is checked
    # separately below.
    netL = M3.ForwardGNNv3(ns=16, nt=6, hidden=32, n_layers=L, use_global=False)
    # THE READOUT MUST BE UN-ZEROED FIRST. The residual head zero-initialises the last layer so
    # training starts at X = 0 -- but then X is identically zero, G = G_an depends only on each
    # triangle's OWN edges, and this test measures nothing about message passing. (Caught exactly
    # that way: the counts fell to 2/6 and 2/13, i.e. only the two triangles owning the perturbed
    # bond.) Randomising the readout puts the message path back in the output.
    with torch.no_grad():
        for _p in netL.readout[-1].parameters():
            _p.add_(torch.randn_like(_p) * 0.05)
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
    d = hop_distance(UNION_SRC, UNION_DST, np.where(owns)[0], len(base))
    # each LAYER is TensorMP (one BOND hop) followed by StarMP (one STAR hop), so a layer
    # advances up to TWO hops on the union graph -- not one. The bound being gated is that reach
    # stays finite and equals the composition, not that it equals L.
    inside = (d >= 0) & (d <= 2 * L)
    beyond = int(((moved > 1e-12) & ~inside).sum())
    print('[5] L=%d (bond+star, no M_S): %d/%d within %d union hops moved; %d beyond (MUST be 0)'
          % (L, int(((moved > 1e-12) & inside).sum()), int(inside.sum()), 2 * L, beyond))
    assert beyond == 0, 'influence leaked past the %d-hop receptive field' % L

# [7] M_S IS GLOBAL BY CONSTRUCTION -- and that is a property to state, not to discover. With the
# channel on, one bond's perturbation must reach essentially every triangle in a single layer,
# because the area-weighted mean it computes is a global quantity. Worth gating BOTH ways: it
# confirms the channel is live, and it records that the model is no longer finite-reach.
with torch.no_grad():
    netG = M3.ForwardGNNv3(ns=16, nt=6, hidden=32, n_layers=1, use_global=True)
    for _p in netG.readout[-1].parameters():
        _p.add_(torch.randn_like(_p) * 0.05)
    b = netG(g0)
    gq = dict(g0)
    gq['bond_feat'] = g0['bond_feat'].clone(); gq['bond_feat'][slots, 0] *= 1.5
    reach = int(((netG(gq) - b).abs().amax(-1).amax(-1) > 1e-12).sum())
    print('[7] with M_S on, L=1 reaches %d/%d triangles (global by construction)'
          % (reach, len(b)))
    assert reach > len(b) // 2, 'M_S channel is not actually coupling globally'

# [8] CHECKPOINT -> ARCHITECTURE ROUND TRIP. `use_star`/`use_global` are optional, so a scorer that
# builds the model from this module's DEFAULTS silently loads a DIFFERENT network than was trained
# -- which happened: `evaluate_v2.py` scored a `--no_global` checkpoint against a default-built
# model. `M3.from_checkpoint` is the single place that rule now lives, so it is gated here: every
# flag combination must round-trip, and the reloaded net must reproduce the original's output
# EXACTLY (not approximately -- same weights, same graph, so any difference is a wrong architecture).
with torch.no_grad():
    for _star, _glob in ((True, True), (True, False), (False, True), (False, False)):
        src_net = M3.ForwardGNNv3(ns=16, nt=6, hidden=32, n_layers=2,
                                  use_star=_star, use_global=_glob)
        for _p in src_net.readout[-1].parameters():          # un-zero, else every model agrees
            _p.add_(torch.randn_like(_p) * 0.05)
        ck = dict(state=src_net.state_dict(), ns=16, nt=6, hidden=32, layers=2,
                  use_star=_star, use_global=_glob)
        back = M3.from_checkpoint(ck)
        assert back.use_star == _star and back.use_global == _glob, 'flags lost in round trip'
        d = float((back(g0) - src_net(g0)).abs().max())
        # and the LEGACY path: a checkpoint predating the flags must infer them from the keys alone
        inferred = M3.from_checkpoint({k: v for k, v in ck.items()
                                       if k not in ('use_star', 'use_global')})
        assert (inferred.use_star, inferred.use_global) == (_star, _glob), \
            'architecture not recoverable from the state dict for star=%s global=%s' % (_star, _glob)
        print('[8] from_checkpoint star=%-5s global=%-5s: exact %.1e, flags inferred OK'
              % (_star, _glob, d))
        assert d == 0.0, 'reloaded model does not reproduce the original bit-for-bit'
