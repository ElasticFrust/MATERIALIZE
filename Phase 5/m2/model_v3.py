r"""M2 v3 -- TENSOR-valued message passing on TRIANGLE adjacency.

Replaces v2's scalar messages on the bond graph.  Both changes are the user's (2026-08-27), and both
follow from what the physics actually couples.

WHY v2 COULD NOT WORK, measured
-------------------------------
v2's network never saw the geometry.  `Q` was used only OUTSIDE the network, in `assemble(Q, G)`;
`forward()` received per bond a single SCALAR length, plus node-level SUMMARY statistics (min/max/std
angular gap) and three triangle angles.  Measured consequences:

  * where `W = 0`, `C(s) = A(s)` needs only each edge's own `(k, l)` -- both are in those scalars --
    and v2 learns it to 0.001 with ZERO message-passing layers.
  * where `W != 0`, triangles whose scalar features are as close as the data allows have targets
    differing by 0.716 (MAE/std), against 0.697 for predicting the global mean.  Local-feature
    proximity buys essentially NOTHING.  Depth did not help: 0, 1 and 2 layers all land at ~0.51,
    which is the "predict your own bulk" baseline.

The information that decides `W` is the RELATIVE ARRANGEMENT of adjacent triangles, and v2 destroyed
it twice over -- by reducing each bond to a length, and by mean-aggregating scalar messages between
NODES, when the physics couples TRIANGLES through shared edges (edge compatibility `J`, and the
vertex angle sum in the curvature operator `C`).

WHAT v3 PASSES -- and nothing else
----------------------------------
The solver computes `C(s)` from exactly (edge vectors, k, l0).  So does this:

    per bond   :  q_e = vec3(dx_e dx_e^T)   -- the edge vector, as a TENSOR
                  k_e,  l0_e
    per triangle: its three `q_e`, which are its full geometry

Lengths, angles, degree and gap statistics are all DERIVED from these and are not passed; the network
forms whatever it needs.  Absolute node position is not passed either: `C` is exactly
translation-invariant (measured 5.9e-16), so position carries no information about it, while
`bond_R` additionally carries the periodic wrap that `pts[v] - pts[u]` does not.  Positions remain
the upstream DESIGN variable -- `dC/dpts` flows through the edge vectors by the chain rule.

EQUIVARIANCE -- the algebra this is built on
--------------------------------------------
Hidden state is (invariant scalars `s`, vec3 TENSOR channels `T`), where `T -> R T` under a rotation
in the vec3 representation.  The permitted operations, and the only ones used:

    mix tensor CHANNELS with scalar weights      T' = W T          (W acts on the channel index)
    gate tensors by scalars                      T' = g(s) * T
    extract invariants                           <T_i, T_j>
    inject geometry as a tensor                  q_e enters messages directly

**The invariant inner product on vec3 [xx, xy, yy] is `diag(1, 2, 1)`, i.e. tr(AB) -- NOT the plain
dot product**, because vec3 carries no factor 2 on shear.  Measured: the plain dot product changes by
up to 11x under rotation; `diag(1,2,1)` is invariant to 1.1e-14.  Getting this wrong would have been
the third convention bug of the project in two days.

`G` must be INVARIANT: `C = Q G Q^T` with `Q -> R Q` already carries the rotation.  So the readout
builds `G` from tensor inner products -- which is where relative orientation finally reaches the
scalars, instead of being discarded at the input.

The head itself is unchanged from v2 (`C(s) = Q (M M^T) Q^T`), which is verified exactly expressive,
equivariant and SPD by `Phase 5/verifications/test_m2_head.py`.
"""
import numpy as np
import torch
import torch.nn as nn

from model_v2 import assemble, c6_to_sym3, edge_carriers, sym3_to_c6, vec3_rotation  # noqa: F401

#: HEAD VERSION, and it belongs in every checkpoint name. The run tag used to encode only the
#: configuration (holdout, filter, layers, width, epochs) and NOT the architecture, so the
#: residual-head run produced a tag byte-identical to the free-head run before it -- and would
#: have hit the no-overwrite guard and discarded ~24 h of training at the save step. Two
#: architectures must never share a checkpoint name.
HEAD_VERSION = 'res'          # 'res' = G = (I+X) G_an (I+X)^T ; pre-2026-08-31 was free SPD

#: metric making <A, B> = tr(AB) on vec3 [xx, xy, yy] -- the ROTATION-INVARIANT inner product
VEC3_METRIC = torch.tensor([1.0, 2.0, 1.0])


def tensor_rms_norm(T, eps=1e-8):
    """Rescale tensor channels by a SCALAR built from their own invariants -- equivariant.

    Needed because the readout consumes `inner(T, T)`, which is QUADRATIC in `T`: any growth through
    the residual updates squares into the readout input. Unnormalised, the first training step saw a
    loss of 2e33. Dividing by an invariant scalar leaves the transformation law untouched (a scalar
    commutes with `T -> R T`), so stability costs no equivariance."""
    m = VEC3_METRIC.to(T.dtype).to(T.device)
    n2 = torch.einsum('...ax,x,...ax->...a', T, m, T)        # <T_a, T_a> per channel
    return T / torch.sqrt(n2.mean(-1, keepdim=True).unsqueeze(-1) + eps)


def inner(A, B):
    """<A, B> = tr(AB) for vec3 tensors.  A: (..., a, 3), B: (..., b, 3) -> (..., a*b) invariants."""
    m = VEC3_METRIC.to(A.dtype).to(A.device)
    return torch.einsum('...ax,x,...bx->...ab', A, m, B).flatten(-2)


def triangle_adjacency(tri_bond, n_tri):
    """Directed triangle-to-triangle edges through SHARED BONDS -> (src, dst, bond).

    This is the graph the PHYSICS couples: edge compatibility requires the two triangles sharing a
    bond to agree on its length, and that pairing is exactly this adjacency.  v2 passed messages
    between NODES instead, so this relation was never a channel."""
    tb = np.asarray(tri_bond)
    owner = {}
    for t in range(n_tri):
        for b in tb[t]:
            owner.setdefault(int(b), []).append(t)
    src, dst, bnd = [], [], []
    for b, ts in owner.items():
        for i in ts:
            for j in ts:
                if i != j:
                    src.append(i); dst.append(j); bnd.append(b)
    if not src:
        return (torch.zeros(0, dtype=torch.long),) * 3
    return (torch.as_tensor(src), torch.as_tensor(dst), torch.as_tensor(bnd))


def angle_gradient_vec(a, b):
    """`d(theta)/d(g11, g12, g22)` for the angle between edge vectors `a` and `b`, as a vec3.

    A LITERAL port of `Phase 2/forward_solver_torch._angle_gradient_vec`, vectorised over a leading
    batch axis.  It is verified against that function elementwise by
    `Phase 5/verifications/test_m2_constraints.py` -- this must not be an approximation of the
    solver's convention, it must BE it, because the constraint it encodes is the one the solver
    actually imposes.

    Note the object type: the return value is a vec3 in the SAME representation as the edge carriers
    `q_e`, so it transforms the same way under rotation and can be fed straight into a tensor
    channel.

    TYPE-PRESERVING, and deliberately so (A0.1, 2026-09-15): numpy in -> numpy out, torch in ->
    torch out.  The torch path is what carries `dC/dpts` -- the star weights are a FUNCTION OF THE
    GEOMETRY, so leaving them as numpy constants silently truncates the position gradient.  Keeping
    ONE function rather than adding a torch twin is the point: the solver's version and this one are
    the only two implementations of this formula, and `test_m2_constraints.py [1]` pins them
    together.  The clamps mirror the solver's exactly (`la*lb` and `sin_th`, not `a2`/`b2`), so the
    numpy path stays bit-identical to what it was."""
    was_np = not (torch.is_tensor(a) or torch.is_tensor(b))
    a = torch.as_tensor(a, dtype=torch.float64) if not torch.is_tensor(a) else a
    b = torch.as_tensor(b, dtype=torch.float64) if not torch.is_tensor(b) else b
    a2 = (a * a).sum(-1); b2 = (b * b).sum(-1); ab = (a * b).sum(-1)
    la = torch.sqrt(a2); lb = torch.sqrt(b2)
    cos_th = torch.clamp(ab / torch.clamp_min(la * lb, 1e-300), -1.0 + 1e-10, 1.0 - 1e-10)
    sin_th = torch.sqrt(1.0 - cos_th ** 2)
    d_ab = torch.stack([a[..., 0] * b[..., 0],
                        a[..., 0] * b[..., 1] + a[..., 1] * b[..., 0],
                        a[..., 1] * b[..., 1]], -1)
    d_a2 = torch.stack([a[..., 0] ** 2, 2 * a[..., 0] * a[..., 1], a[..., 1] ** 2], -1)
    d_b2 = torch.stack([b[..., 0] ** 2, 2 * b[..., 0] * b[..., 1], b[..., 1] ** 2], -1)
    d_cos = (d_ab / torch.clamp_min(la * lb, 1e-300)[..., None]
             - cos_th[..., None] * (d_a2 / (2 * a2)[..., None] + d_b2 / (2 * b2)[..., None]))
    out = -d_cos / torch.clamp_min(sin_th, 1e-300)[..., None]
    out = torch.where((sin_th < 1e-10)[..., None], torch.zeros_like(out), out)   # degenerate -> 0
    return out.detach().numpy() if was_np else out


def vertex_stars(tri_verts, tri_bond, bond_u, bond_v, bond_R, idx=None):
    """The CURVATURE constraint's coupling, as a bipartite (triangle, vertex) incidence.

    `C_curv` has one row per interior vertex: `sum_{s in star(v)} (dtheta_v^s/dg) . dg(s) = 0`, i.e.
    it couples the ~6 triangles meeting at a vertex SIMULTANEOUSLY.  `triangle_adjacency` cannot
    express that -- it is the EDGE-compatibility pairing (two triangles per shared bond), so
    `J_edge`'s structure was already present in the model and `C_curv`'s was entirely absent.

    Returns `(star_tri, star_vert, star_w, n_vert)`:
        star_tri  (P,)    triangle index of each (triangle, corner) incidence
        star_vert (P,)    vertex index      "
        star_w    (P, 3)  the vec3 `dtheta/dg` weight for that corner
    Sign convention copied verbatim from the solver: for edge (a, b), the vector taken at vertex `v`
    is `-bond_R` if `v == a` and `+bond_R` if `v == b`.

    `bond_R` is TYPE-PRESERVING (A0.1): pass it as torch and `star_w` comes back as torch carrying
    `d/dpts`.  The INDICES are computed by `corner_index` from connectivity alone -- they do not
    depend on the coordinates, which is exactly why the split is safe: topology is discrete and
    correctly non-differentiable, the weights are not."""
    e_a, s_a, e_b, s_b, tri_idx, vert_compact, n_vert = idx if idx is not None else corner_index(
        tri_verts, tri_bond, bond_u, bond_v)
    if len(tri_idx) == 0:
        z = np.zeros(0, np.int64)
        return z, z, np.zeros((0, 3)), 0
    was_np = not torch.is_tensor(bond_R)
    bR = torch.as_tensor(np.asarray(bond_R, float)) if was_np else bond_R
    sa = torch.as_tensor(s_a, dtype=bR.dtype)[:, None]
    sb = torch.as_tensor(s_b, dtype=bR.dtype)[:, None]
    w = angle_gradient_vec(sa * bR[e_a], sb * bR[e_b])
    return tri_idx, vert_compact, (w.detach().numpy() if was_np else w), n_vert


def corner_index(tri_verts, tri_bond, bond_u, bond_v):
    """The (triangle, corner) incidence of the curvature star, as PURE CONNECTIVITY.

    Returns `(e_a, s_a, e_b, s_b, tri_idx, vert_compact, n_vert)`: for each incidence, the two bond
    indices meeting at that corner and the sign each is taken with (the solver's convention, see
    `vertex_stars`).  Depends on no coordinate, so it is computed once in numpy and reused for every
    perturbed geometry -- which is what makes `star_w` and the areas differentiable without
    recomputing the combinatorics."""
    tri_verts = np.asarray(tri_verts, np.int64)
    tri_bond = np.asarray(tri_bond, np.int64)
    bu = np.asarray(bond_u, np.int64); bv = np.asarray(bond_v, np.int64)

    e_a, s_a, e_b, s_b, tri_idx, vert_idx = [], [], [], [], [], []
    for s_ in range(len(tri_bond)):
        for v in tri_verts[s_]:
            got = []
            for i in range(3):
                e = int(tri_bond[s_, i])
                if bu[e] == v:
                    got.append((e, -1.0))
                elif bv[e] == v:
                    got.append((e, +1.0))
            if len(got) != 2:            # v is not a corner of two of this triangle's edges
                continue
            tri_idx.append(s_); vert_idx.append(int(v))
            e_a.append(got[0][0]); s_a.append(got[0][1])
            e_b.append(got[1][0]); s_b.append(got[1][1])
    if not tri_idx:
        z = np.zeros(0, np.int64)
        return z, np.zeros(0), z, np.zeros(0), z, z, 0
    # compact the vertex ids so they index a dense per-sample vertex array
    uniq, vert_compact = np.unique(np.array(vert_idx, np.int64), return_inverse=True)
    return (np.array(e_a, np.int64), np.array(s_a, float),
            np.array(e_b, np.int64), np.array(s_b, float),
            np.array(tri_idx, np.int64), vert_compact.astype(np.int64), len(uniq))


def triangle_areas(tri_verts, tri_bond, bond_u, bond_v, bond_R, idx=None, signed=False):
    """Per-triangle area, in torch, from the edge vectors -- so it carries `d/dpts`.

    `signed=True` returns the SIGNED area instead, which is the inversion detector: a node moved far
    enough flips a triangle inside out, and `abs` hides exactly that.  Any perturbed geometry -- a
    finite-difference probe, a trust-region step, a node drag -- must be checked with it.

    The stored `areas` are a NumPy constant; using them would silently truncate the position
    gradient through `area_w` (the `M_S` channel) and `phys` (the physical-units factor).  Computed
    as `|a x b| / 2` at one corner of each triangle, reusing `corner_index`'s verified sign
    convention rather than re-deriving an edge orientation here.  Gated against the stored areas at
    1e-13 by `test_m2_grad_port.py`, which is what catches a wrong orientation."""
    e_a, s_a, e_b, s_b, tri_idx, _, _ = idx if idx is not None else corner_index(
        tri_verts, tri_bond, bond_u, bond_v)
    n_tri = len(np.asarray(tri_bond, np.int64))
    bR = torch.as_tensor(np.asarray(bond_R, float)) if not torch.is_tensor(bond_R) else bond_R
    # one incidence per triangle: the first corner encountered
    first = np.full(n_tri, -1, np.int64)
    for p in range(len(tri_idx) - 1, -1, -1):        # reverse so the earliest wins
        first[tri_idx[p]] = p
    if (first < 0).any():
        raise ValueError('triangle with no complete corner: mesh connectivity is inconsistent')
    sa = torch.as_tensor(s_a[first], dtype=bR.dtype)[:, None]
    sb = torch.as_tensor(s_b[first], dtype=bR.dtype)[:, None]
    a = sa * bR[e_a[first]]; b = sb * bR[e_b[first]]
    cross = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    return 0.5 * (cross if signed else torch.abs(cross))


class TensorMP(nn.Module):
    """One equivariant message-passing round over triangle adjacency.

    A message from triangle `t` to `s` across shared bond `b` carries BOTH:
      * invariants -- <T_s, T_t>, <T_s, q_b>, <T_t, q_b>, and the bond's own (k, l0);
      * tensors    -- a scalar-gated combination of `T_t` and `q_b`.
    The tensor part is what v2 lacked: it lets orientation propagate instead of being collapsed to a
    scalar at every hop."""

    def __init__(self, ns, nt, hidden, n_bond_feat=2):
        super().__init__()
        self.ns, self.nt = ns, nt
        n_inv = nt * nt + 2 * nt + n_bond_feat          # <Ts,Tt>, <Ts,q>, <Tt,q>, bond scalars
        self.msg = nn.Sequential(nn.Linear(2 * ns + n_inv, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU())
        self.msg_s = nn.Linear(hidden, ns)              # scalar part of the message
        self.msg_t = nn.Linear(hidden, nt + 1)          # gates: nt for T_t, 1 for q_b
        self.upd_s = nn.Sequential(nn.Linear(2 * ns, hidden), nn.SiLU(), nn.Linear(hidden, ns))
        self.upd_t = nn.Linear(ns, nt)                  # scalar gate on the aggregated tensors
        self.norm_s = nn.LayerNorm(ns)

    def forward(self, s, T, src, dst, qb, bf):
        Ts, Tt, ss, st = T[dst], T[src], s[dst], s[src]
        q = qb.unsqueeze(1)                                            # (E, 1, 3)
        inv = torch.cat([inner(Ts, Tt), inner(Ts, q), inner(Tt, q), bf], -1)
        h = self.msg(torch.cat([ss, st, inv], -1))

        gate = self.msg_t(h)                                           # (E, nt+1)
        mT = gate[:, :self.nt].unsqueeze(-1) * Tt + gate[:, self.nt:].unsqueeze(-1) * q
        mS = self.msg_s(h)

        aggT = torch.zeros_like(T).index_add(0, dst, mT)
        aggS = torch.zeros_like(s).index_add(0, dst, mS)
        cnt = torch.zeros(len(s), 1, dtype=s.dtype, device=s.device)
        cnt = cnt.index_add(0, dst, torch.ones(len(dst), 1, dtype=s.dtype, device=s.device))
        cnt = cnt.clamp_min(1.0)

        s = self.norm_s(s + self.upd_s(torch.cat([s, aggS / cnt], -1)))
        T = tensor_rms_norm(T + self.upd_t(s).unsqueeze(-1) * (aggT / cnt.unsqueeze(-1)))
        return s, T


class StarMP(nn.Module):
    """One round of triangle -> VERTEX -> triangle passing: the CURVATURE constraint's coupling.

    `C_curv` couples the ~6 triangles meeting at a vertex SIMULTANEOUSLY, weighted by
    `dtheta_v^s/dg`.  `TensorMP` cannot express that: it runs on `triangle_adjacency`, which is the
    EDGE-compatibility pairing (two triangles per shared bond).  So `J_edge`'s structure was already
    in the model by construction and `C_curv`'s was entirely absent -- this module is that missing
    channel, and its weights are verified identical to the solver's operator
    (`test_m2_constraints.py`).

    The weight `w = dtheta/dg` is a vec3, the same representation as the edge carriers, so it is
    split into a UNIT TENSOR `w/|w|` (equivariant, carries the direction) and its NORM `|w|`
    (invariant, carries the magnitude).  Passing `w` raw would let its scale -- which varies over
    orders of magnitude on sliver corners -- swamp the tensor channels."""

    def __init__(self, ns, nt, hidden):
        super().__init__()
        self.ns, self.nt = ns, nt
        # the two directions consume DIFFERENT invariants: up sees <T_tri, w_hat> only (nt),
        # down sees both <T_vert, w_hat> and <T_tri, w_hat> (2nt); both carry |w|.
        n_up, n_dn = nt + 1, 2 * nt + 1
        self.up = nn.Sequential(nn.Linear(ns + n_up, hidden), nn.SiLU(),
                                nn.Linear(hidden, hidden), nn.SiLU())
        self.up_s = nn.Linear(hidden, ns)
        self.up_t = nn.Linear(hidden, nt + 1)    # gates for T_tri and w_hat
        self.down = nn.Sequential(nn.Linear(2 * ns + n_dn, hidden), nn.SiLU(),
                                  nn.Linear(hidden, hidden), nn.SiLU())
        self.down_s = nn.Linear(hidden, ns)
        self.down_t = nn.Linear(hidden, nt + 1)
        self.norm_s = nn.LayerNorm(ns)

    def forward(self, s, T, star_tri, star_vert, star_w, n_vert):
        if len(star_tri) == 0 or n_vert == 0:
            return s, T
        wn = torch.sqrt(torch.einsum('px,x,px->p', star_w,
                                     VEC3_METRIC.to(star_w.dtype).to(star_w.device),
                                     star_w).clamp_min(1e-30))
        w_hat = (star_w / wn.unsqueeze(-1)).unsqueeze(1)                  # (P, 1, 3) unit vec3
        St, Tt = s[star_tri], T[star_tri]

        # --- triangles -> vertices -------------------------------------------------------------
        inv_up = torch.cat([inner(Tt, w_hat), torch.log1p(wn).unsqueeze(-1)], -1)
        h = self.up(torch.cat([St, inv_up], -1))
        g = self.up_t(h)
        mT = g[:, :self.nt].unsqueeze(-1) * Tt + g[:, self.nt:].unsqueeze(-1) * w_hat
        cnt = torch.zeros(n_vert, 1, dtype=s.dtype, device=s.device).index_add(
            0, star_vert, torch.ones(len(star_vert), 1, dtype=s.dtype, device=s.device)).clamp_min(1.0)
        sv = torch.zeros(n_vert, self.ns, dtype=s.dtype, device=s.device).index_add(
            0, star_vert, self.up_s(h)) / cnt
        Tv = torch.zeros(n_vert, self.nt, 3, dtype=T.dtype, device=T.device).index_add(
            0, star_vert, mT) / cnt.unsqueeze(-1)

        # --- vertices -> triangles ---------------------------------------------------------------
        Sv, Tvv = sv[star_vert], Tv[star_vert]
        inv_dn = torch.cat([inner(Tvv, w_hat), inner(Tt, w_hat),
                            torch.log1p(wn).unsqueeze(-1)], -1)[:, :2 * self.nt + 1]
        h2 = self.down(torch.cat([St, Sv, inv_dn], -1))
        g2 = self.down_t(h2)
        bT = g2[:, :self.nt].unsqueeze(-1) * Tvv + g2[:, self.nt:].unsqueeze(-1) * w_hat
        cnt_t = torch.zeros(len(s), 1, dtype=s.dtype, device=s.device).index_add(
            0, star_tri, torch.ones(len(star_tri), 1, dtype=s.dtype, device=s.device)).clamp_min(1.0)
        aggS = torch.zeros_like(s).index_add(0, star_tri, self.down_s(h2)) / cnt_t
        aggT = torch.zeros_like(T).index_add(0, star_tri, bT) / cnt_t.unsqueeze(-1)
        return self.norm_s(s + aggS), tensor_rms_norm(T + aggT)


class GlobalMS(nn.Module):
    """The AREA-WEIGHTED GLOBAL MEAN constraint `M_S`:  sum_s S_s dg(s) = 0.

    Three rows -- one per vec3 component -- so it is global but only RANK 3, and its influence per
    triangle falls off as 1/N. That makes it cheap to represent exactly: the quantity the constraint
    sets to zero IS the area-weighted mean of the metric field, so handing the model that mean is
    the whole content of the constraint, not a proxy for it.

    Equivariance: a weighted mean of vec3 tensors is a vec3 tensor, and the weights (areas) are
    rotation invariant, so `Tbar` transforms exactly as `T` does.

    BATCHING: the mean must be taken PER GRAPH. Collate packs many graphs block-diagonally, so a
    naive global mean would average across unrelated networks and leak between samples -- which is
    why `tri_batch` exists rather than a plain `T.mean(0)`."""

    def __init__(self, ns, nt, hidden):
        super().__init__()
        self.nt = nt
        self.mix_s = nn.Sequential(nn.Linear(2 * ns + nt * nt, hidden), nn.SiLU(),
                                   nn.Linear(hidden, ns))
        self.gate_t = nn.Linear(ns, nt)
        self.norm_s = nn.LayerNorm(ns)

    def forward(self, s, T, tri_batch, area_w, n_graphs):
        w = area_w.unsqueeze(-1)
        wsum = torch.zeros(n_graphs, 1, dtype=s.dtype, device=s.device).index_add(
            0, tri_batch, w).clamp_min(1e-30)
        sbar = torch.zeros(n_graphs, s.shape[1], dtype=s.dtype, device=s.device).index_add(
            0, tri_batch, w * s) / wsum
        Tbar = torch.zeros(n_graphs, T.shape[1], 3, dtype=T.dtype, device=T.device).index_add(
            0, tri_batch, w.unsqueeze(-1) * T) / wsum.unsqueeze(-1)
        Sb, Tb = sbar[tri_batch], Tbar[tri_batch]
        s = self.norm_s(s + self.mix_s(torch.cat([s, Sb, inner(T, Tb)], -1)))
        return s, tensor_rms_norm(T + self.gate_t(s).unsqueeze(-1) * Tb)


class ForwardGNNv3(nn.Module):
    """Graph -> per-triangle G -> C(s) = Q G Q^T.  Tensor messages on triangle adjacency.

    RESIDUAL HEAD (2026-08-30).  `G` is not predicted from scratch; it is a CONGRUENCE of the
    ANALYTIC per-triangle tensor:

        G = (I + X) G_an (I + X)^T ,      G_an = diag(k_e / 16 l_e^2)

    `G_an` is `A(s)` in the `Q` basis -- a closed form in the triangle's own three edges, which
    `train_v3.oracle_check` verifies to machine precision wherever `W = 0`.  The network predicts
    only the dimensionless `X`.

    WHY, measured on the previous checkpoint.  The old head predicted `G` freely, so the model had to
    rediscover a closed form it could have been handed -- and it did so IMPERFECTLY: on the 254
    `W = 0` holdout networks, where the answer IS `A(s)`, it scored 0.0538 instead of 0.  Worse, `C`
    is dominated by `A` (predicting `A` alone scores 1.1747 against the label sigma, while the trained
    model scores 0.2000), so the loss is dominated by the easy term while 91 % of the error sits at
    `max|W| >= 1` -- the correction that carries the actual physics.  This head makes the easy term
    exact and leaves the network only the hard one.

    Properties, all preserved:
      * `X = 0` reproduces `A(s)` EXACTLY -- and the readout's last layer is zero-initialised, so
        training STARTS there rather than at a random tensor of the wrong scale.
      * SPD: a congruence of an SPD diagonal is PSD always, and SPD unless `det(I + X) = 0`
        (measure zero; `test_m2_head_v3.py` gate [3] detects it).
      * INVARIANCE: `G_an` is diagonal in the per-edge scalars `k`, `l`, which are rotation
        invariant, and `X` is built from invariant features -- so `G` stays invariant and
        `C = Q G Q^T` stays equivariant, unchanged from before.
    """

    def __init__(self, ns=32, nt=8, hidden=64, n_layers=3, passive=True,
                 n_tri_scalars=9, n_bond_feat=2, use_star=True, use_global=True,
                 tie=False, n_iter=0):
        """`tie` + `n_iter`: WEIGHT-TIED ITERATION instead of stacked distinct layers.

        `W(s)` is the solution of a global constrained system -- an operator INVERSE -- so a
        fixed-depth message-passing net is a fixed number of relaxation sweeps, and the sweeps a
        solve needs grow with the system's conditioning. Plain depth is the wrong way to buy more
        sweeps: it adds parameters and it does not train (depth 8 DIVERGED at lr 3e-3, depth 10
        STALLED at 0.53 against depth 5's 0.19). Tying reuses ONE block `n_iter` times, so the
        iteration count is decoupled from both the parameter count and the optimisation difficulty.

        It also makes the hypothesis DIRECTLY TESTABLE in a way nothing else here is: `n_iter` can be
        changed AFTER training (`net.n_iter = N`), so a single trained model can be swept over N at
        inference. If the iteration picture is right, error falls with N on fixed weights. Train with
        N jittered per batch so that sweep is in-distribution rather than an extrapolation.

        Defaults are OFF, so every existing checkpoint and every call site is unaffected.
        """
        super().__init__()
        self.ns, self.nt, self.passive = ns, nt, passive
        self.use_star, self.use_global = use_star, use_global
        self.tie = bool(tie)
        #: applications of the block(s) per forward pass. Untied, this MUST equal n_layers (each
        #: layer is a distinct module and is used once); tied, it is free and may be changed after
        #: training.
        self.n_iter = int(n_iter) if n_iter else int(n_layers)
        if not self.tie and self.n_iter != n_layers:
            raise ValueError('n_iter=%d != n_layers=%d is only meaningful with tie=True; untied, '
                             'each layer is a distinct module used exactly once'
                             % (self.n_iter, n_layers))
        n_blocks = 1 if self.tie else n_layers
        # initial tensor channels are the triangle's own three q_e, mixed up to `nt`
        self.t_in = nn.Linear(3, nt, bias=False)
        # initial scalars: the 9 invariants <q_i, q_j> of the triangle's own geometry,
        # plus its per-edge scalars (k, l0, log k) -- 9 by default. Sizes are explicit rather than
        # hard-coded, since a mismatch here is a silent architecture change.
        self.s_in = nn.Sequential(nn.Linear(9 + n_tri_scalars, hidden), nn.SiLU(),
                                  nn.Linear(hidden, ns))
        self.norm_in = nn.LayerNorm(ns)
        self.layers = nn.ModuleList(TensorMP(ns, nt, hidden, n_bond_feat)
                                    for _ in range(n_blocks))
        # the CURVATURE channel, one per layer, run alongside the bond channel. Flagged so it can be
        # ablated as a single variable against the bond-only model.
        self.stars = nn.ModuleList(StarMP(ns, nt, hidden) for _ in range(n_blocks))             if use_star else None
        self.globals = nn.ModuleList(GlobalMS(ns, nt, hidden) for _ in range(n_blocks))             if use_global else None
        self.readout = nn.Sequential(nn.Linear(ns + nt * nt, hidden), nn.SiLU(),
                                     nn.Linear(hidden, hidden), nn.SiLU(),
                                     nn.Linear(hidden, 9))          # X, a full invariant 3x3
        # START AT THE ANALYTIC ANSWER: zeroing the last layer gives X = 0, hence G = G_an = A(s),
        # which is EXACT wherever W = 0 and the right scale everywhere else. Without this the model
        # begins at a random tensor and spends its first epochs recovering the closed form.
        nn.init.zeros_(self.readout[-1].weight)
        nn.init.zeros_(self.readout[-1].bias)

    @staticmethod
    def analytic_G(g):
        """`G_an = diag(k_e / 16 l_e^2)` -- `A(s)` in the `Q` basis, from inputs already passed.

        `l_e^2` is the trace of each edge carrier: for `q_e = vec3(dx dx^T) = [xx, xy, yy]`,
        `xx + yy = l_e^2`, i.e. rows 0 and 2 of `Q`.  `k` is the first block of `tri_scalars`
        (`kk = k / mean(k)`; the `mean(k)` factor is carried separately in `assemble`'s
        `physical_factor`, so using the NORMALISED k here is what keeps the two consistent)."""
        l2 = g['Q'][:, 0, :] + g['Q'][:, 2, :]                         # (n_tri, 3) squared lengths
        k = g['tri_scalars'][:, :3]                                    # (n_tri, 3) k/kbar per edge
        return torch.diag_embed(k / (16.0 * l2))

    def forward(self, g):
        Q = g['Q']                                                     # (n_tri, 3, 3) carriers
        Qc = Q.transpose(-1, -2)                                       # (n_tri, 3, 3) as rows
        T = tensor_rms_norm(self.t_in(Qc.transpose(-1, -2)).transpose(-1, -2))   # 3 carriers -> nt
        gram = inner(Qc, Qc)                                           # 9 invariants of the geometry
        s = self.norm_in(self.s_in(torch.cat([gram, g['tri_scalars']], -1)))

        # TIED: one block, applied `n_iter` times (index 0 every sweep). UNTIED: the original
        # behaviour exactly -- n_iter == n_layers, so `j == i` and each distinct layer runs once.
        for i in range(self.n_iter):
            j = 0 if self.tie else i
            s, T = self.layers[j](s, T, g['tri_src'], g['tri_dst'], g['bond_q'], g['bond_feat'])
            if self.stars is not None and 'star_tri' in g:
                s, T = self.stars[j](s, T, g['star_tri'], g['star_vert'], g['star_w'],
                                     int(g['n_vert']))
            if self.globals is not None and 'tri_batch' in g:
                s, T = self.globals[j](s, T, g['tri_batch'], g['area_w'], int(g['n_graphs']))

        raw = self.readout(torch.cat([s, inner(T, T)], -1))
        return self._to_G(raw, self.analytic_G(g))

    def _to_G(self, raw, G_an):
        """`G = (I + X) G_an (I + X)^T` -- a congruence of the analytic tensor, so SPD is inherited
        from `G_an` and `X = 0` returns `A(s)` exactly."""
        n = raw.shape[0]
        X = raw.reshape(n, 3, 3)
        L = torch.eye(3, dtype=raw.dtype, device=raw.device) + X
        return L @ G_an @ L.transpose(-1, -2)


def from_checkpoint(ck, eval_mode=True):
    """Rebuild the EXACT architecture a checkpoint was trained with, and load its weights.

    The star (`C_curv`) and `M_S` channels are optional, so a model built from this module's
    DEFAULTS silently mismatches any run trained with `--no_star` / `--no_global` -- and that is not
    hypothetical: `evaluate_v2.py` scored a `--no_global` checkpoint against a default-built model
    before this existed.  Three scripts now load checkpoints (`evaluate_v2`, `m2_v3_report`,
    `m2_error_strata`), so the rule lives in ONE place.

    Checkpoints written before the flags existed carry no `use_star`/`use_global` key; for those the
    architecture is inferred from the state dict's own parameter names, which is the only honest
    source left."""
    ks = ck['state'].keys()
    # A TIED checkpoint has exactly ONE block, so `layers.1.*` is absent -- inferable from the
    # state dict for checkpoints written before the flag existed, same principle as the channels.
    tie = ck.get('tie', not any(k.startswith('layers.1.') for k in ks) and int(ck['layers']) > 1)
    net = ForwardGNNv3(ns=ck['ns'], nt=ck['nt'], hidden=ck['hidden'], n_layers=ck['layers'],
                       use_star=ck.get('use_star', any(k.startswith('stars.') for k in ks)),
                       use_global=ck.get('use_global', any(k.startswith('globals.') for k in ks)),
                       tie=tie, n_iter=ck.get('n_iter', 0))
    net.load_state_dict(ck['state'])
    if eval_mode:
        net.eval()
    return net
