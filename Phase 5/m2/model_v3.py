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


class ForwardGNNv3(nn.Module):
    """Graph -> per-triangle G -> C(s) = Q G Q^T.  Tensor messages on triangle adjacency."""

    def __init__(self, ns=32, nt=8, hidden=64, n_layers=3, passive=True,
                 n_tri_scalars=9, n_bond_feat=2):
        super().__init__()
        self.ns, self.nt, self.passive = ns, nt, passive
        # initial tensor channels are the triangle's own three q_e, mixed up to `nt`
        self.t_in = nn.Linear(3, nt, bias=False)
        # initial scalars: the 9 invariants <q_i, q_j> of the triangle's own geometry,
        # plus its per-edge scalars (k, l0, log k) -- 9 by default. Sizes are explicit rather than
        # hard-coded, since a mismatch here is a silent architecture change.
        self.s_in = nn.Sequential(nn.Linear(9 + n_tri_scalars, hidden), nn.SiLU(),
                                  nn.Linear(hidden, ns))
        self.norm_in = nn.LayerNorm(ns)
        self.layers = nn.ModuleList(TensorMP(ns, nt, hidden, n_bond_feat)
                                    for _ in range(n_layers))
        self.readout = nn.Sequential(nn.Linear(ns + nt * nt, hidden), nn.SiLU(),
                                     nn.Linear(hidden, hidden), nn.SiLU(),
                                     nn.Linear(hidden, 6 if passive else 9))

    def forward(self, g):
        Q = g['Q']                                                     # (n_tri, 3, 3) carriers
        Qc = Q.transpose(-1, -2)                                       # (n_tri, 3, 3) as rows
        T = tensor_rms_norm(self.t_in(Qc.transpose(-1, -2)).transpose(-1, -2))   # 3 carriers -> nt
        gram = inner(Qc, Qc)                                           # 9 invariants of the geometry
        s = self.norm_in(self.s_in(torch.cat([gram, g['tri_scalars']], -1)))

        for lay in self.layers:
            s, T = lay(s, T, g['tri_src'], g['tri_dst'], g['bond_q'], g['bond_feat'])

        raw = self.readout(torch.cat([s, inner(T, T)], -1))
        return self._to_G(raw)

    def _to_G(self, raw):
        n = raw.shape[0]
        if not self.passive:
            return raw.reshape(n, 3, 3)
        M = torch.zeros(n, 3, 3, dtype=raw.dtype, device=raw.device)
        idx = torch.tril_indices(3, 3)
        M[:, idx[0], idx[1]] = raw
        d = torch.arange(3)
        M[:, d, d] = nn.functional.softplus(M[:, d, d])
        return M @ M.transpose(-1, -2)
