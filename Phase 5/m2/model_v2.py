r"""M2 v2 model -- physics-shaped, equivariant, SPD-by-construction GNN surrogate.

Implements `Phase 5/m2/M2_V2_PLAN.md` section 2.  Replaces `model.py` (v1), which is kept as the
record of the earlier scaffold.

THE HEAD (section 2.1) -- why it is shaped like this
----------------------------------------------------
The solver's own structure is  C(s) = (1+W)^T A(s) (1+W)  with  A(s) = sum_e c_e q_e q_e^T  and
q_e = vec3(dx_e dx_e^T) = [dx^2, dx*dy, dy^2] the rank-1 edge carrier.  Pushing (1+W) through gives
C(s) = sum_e c_e qt_e qt_e^T with qt_e = (1+W)^T q_e -- a sum of outer products of TRANSFORMED
carriers.  So a scalar-weighted sum of the BARE q_e q_e^T is an AFFINE-ONLY ansatz and cannot
represent the non-affine response.  (An earlier draft of the plan had exactly that error.)

The exact form: the three q_e of a triangle, when linearly independent, are a basis of vec3.  Stack
them as the columns of Q(s) = [q1 q2 q3] and predict, per triangle,

    passive :  C(s) = Q (M M^T) Q^T ,  M lower-triangular with softplus diagonal   (6 scalars)
    general :  C(s) = Q    G    Q^T ,  G a free 3x3                                (9 scalars)

  * EXACTLY EXPRESSIVE -- Sym(3x3) is 6-dimensional, so the passive form reaches ANY symmetric C(s).
  * EQUIVARIANT -- under a rotation Q -> R Q, hence C -> R C R^T, the correct tensor law, while the
    network emits only rotation-INVARIANT scalars.  No capacity is spent relearning a known symmetry.
  * SPD BY CONSTRUCTION in the passive form (X X^T), so physicality is structural, not hoped for.
  * INTERPRETABLE -- at W = 0, M M^T is DIAGONAL with entries c_e, so the OFF-DIAGONALS are exactly
    the non-affine content and the model learns the correction to a known answer.
  * `passive=False` admits ODD / ACTIVE elasticity: a free G spans all of 3x3 including the
    antisymmetric part.  SPD is a passive-only assumption and so is a flag, not a hard-wired law.

UNITS.  c_e = k_e / (16 l_e^2) in the solver's INTERNAL units -- measured, not assumed: on the open
single triangle the solver's C(s) equals sum_e (k_e/16 l_e^2) q_e q_e^T to 2e-16 over three shapes x
three k vectors (`Phase 5/seeds.py: anchor_single_triangle`).  The plan states the gate as k/4l^2,
which is the PHYSICAL form -- a factor 4 apart.  `physical_factor` converts.

ASSEMBLY (section 2.2).  C_eff = (1/N) sum_s C(s), the UNWEIGHTED mean (`CLAUDE.md` section 3; an
area weight biases nu on unequal-area meshes).  A MEAN, so intensivity holds by construction --
v1 pooled `mean + sum` and the sum branch is EXTENSIVE, which C_eff is not (verified intensive to
5e-16..7e-13, `Phase 5/results/supercell_invariance/`).

SCALING SYMMETRY (section 2.2).  k -> lambda k sends A -> lambda A and A^-1 -> A^-1/lambda, so W is
UNCHANGED and C -> lambda C.  The network is therefore fed k/mean(k) and its prediction multiplied
by mean(k): nu comes out scale-invariant by construction, as the physics requires.
"""
import numpy as np
import torch
import torch.nn as nn

VEC3 = 3

#: sizes of the invariant scalar feature blocks; a checkpoint trained with different
#: values is architecturally incompatible, so they are constants rather than something to probe
N_NODE_FEAT = 4        # [degree, min gap, max gap, std gap]
N_TRI_FEAT = 3         # the triangle's sorted interior angles


# ---- geometry -> the equivariant basis --------------------------------------------------------
def edge_carriers(edge_vecs):
    """q_e = vec3(dx dx^T) = [dx^2, dx*dy, dy^2] for each of a triangle's three edges.

    Args:  edge_vecs (n_tri, 3, 2) -- the solver's own edge vectors, order (0,1), (0,2), (1,2).
    Returns: Q (n_tri, 3, 3) with the three carriers as COLUMNS, so `Q @ diag(c) @ Q.T` is exactly
    `sum_e c_e q_e q_e^T`."""
    ev = torch.as_tensor(edge_vecs)
    dx, dy = ev[..., 0], ev[..., 1]
    q = torch.stack([dx * dx, dx * dy, dy * dy], dim=-1)          # (n_tri, 3, 3): [edge, vec3]
    return q.transpose(-1, -2)                                     # carriers as columns


def vec3_rotation(theta):
    """The 3x3 representation of a 2D rotation acting on symmetric tensors in vec3 [xx, xy, yy].

    S -> R S R^T induces this on the vec3 coordinates; it is what makes the head equivariant, and
    it is what `test_m2_head.py` checks numerically rather than trusting the algebra."""
    c, s = np.cos(theta), np.sin(theta)
    return torch.tensor([[c * c, -2 * c * s, s * s],
                         [c * s, c * c - s * s, -c * s],
                         [s * s, 2 * c * s, c * c]], dtype=torch.get_default_dtype())


def closed_form_weights(k_tri, len2, internal=True):
    """The ANALYTIC `M M^T` at W = 0: diagonal with entries c_e = k_e / (16 l_e^2).

    This is the S1 gate.  `internal=False` returns the physical-convention k_e/(4 l_e^2)."""
    denom = 16.0 if internal else 4.0
    return torch.diag_embed(torch.as_tensor(k_tri) / (denom * torch.as_tensor(len2)))


def assemble(Q, G, physical_factor=None):
    """C(s) = Q G Q^T, and the UNWEIGHTED mean over triangles.

    Returns (C_per (n_tri, 3, 3), C_eff (3, 3)).  `physical_factor` = 8*n_tri/sum(areas) converts
    the solver's internal scale to physical units; both outputs are scaled by it, so the mean
    relation between them is preserved exactly."""
    C_per = Q @ G @ Q.transpose(-1, -2)
    if physical_factor is not None:
        C_per = C_per * physical_factor
    return C_per, C_per.mean(0)


def sym3_to_c6(C):
    """(..., 3, 3) symmetric vec3 tensor -> the project's 6-vector [xxxx, xxxy, xxyy, xyxy, xyyy, yyyy]."""
    return torch.stack([C[..., 0, 0], C[..., 0, 1], C[..., 0, 2],
                        C[..., 1, 1], C[..., 1, 2], C[..., 2, 2]], dim=-1)


def c6_to_sym3(c6):
    """Inverse of `sym3_to_c6`."""
    a, b, c, d, e, f = (c6[..., i] for i in range(6))
    return torch.stack([torch.stack([a, b, c], -1),
                        torch.stack([b, d, e], -1),
                        torch.stack([c, e, f], -1)], dim=-2)


# ---- invariant features -----------------------------------------------------------------------
def edge_features(k, bond_R, eps=1e-12):
    """Per-bond ROTATION-INVARIANT features: [k/kbar, log(k/kbar), l/lbar].

    Raw direction components are deliberately absent from the scalar path (v1 fed `dir_x, dir_y`,
    which is not rotation-invariant).  Direction enters ONLY through Q(s) in the head, where the
    tensor law handles it exactly."""
    k = torch.as_tensor(k)
    R = torch.as_tensor(bond_R)
    kb = k.mean().clamp_min(eps)
    ell = torch.linalg.norm(R, dim=-1)
    return torch.stack([k / kb, torch.log(k / kb + eps), ell / ell.mean().clamp_min(eps)], dim=-1)


def node_angle_features(bond_u, bond_v, bond_R, n_nodes):
    """Per-node ROTATION-INVARIANT angular features: `[degree, min gap, max gap, std gap]`, where the
    gaps are the angular spacings between the directions of the bonds incident on that node.

    The MEAN gap is deliberately absent: gaps around a node sum to 2*pi, so mean = 2*pi/degree and it
    carries nothing degree does not.

    WHY THIS EXISTS -- it was specified and then omitted.  Section 2.3 asks for "per node, the sorted
    angles between incident bonds"; the first implementation used DEGREE ALONE.  That leaves the
    scalar path with no angular information whatever, so two triangles with identical edge lengths
    and stiffnesses sitting in differently-shaped neighbourhoods get IDENTICAL features while having
    different `W`, hence different `C(s)`.  The features were degenerate with respect to the target,
    which is the natural explanation for the first runs learning the bulk response and giving up on
    the local part.  (The head still had the full geometry through `Q(s)`, but `Q` only ROTATES the
    answer -- the scalar network deciding WHAT the answer is could not see angles at all.)

    Gaps rather than raw angles because gaps are invariant under a global rotation: rotating every
    direction shifts all angles equally and leaves their differences alone.  Summary statistics
    rather than the sorted list because degree varies, and the feature vector must not."""
    R = np.asarray(bond_R, float)
    u = np.asarray(bond_u, np.int64)
    v = np.asarray(bond_v, np.int64)
    out_dirs = [[] for _ in range(n_nodes)]
    for e in range(len(u)):
        ang = np.arctan2(R[e, 1], R[e, 0])
        out_dirs[u[e]].append(ang)                       # leaving u
        out_dirs[v[e]].append(np.arctan2(-R[e, 1], -R[e, 0]))   # leaving v, i.e. reversed
    feat = np.zeros((n_nodes, 4))
    for i, angs in enumerate(out_dirs):
        d = len(angs)
        feat[i, 0] = d
        if d < 2:
            continue
        a = np.sort(np.mod(np.asarray(angs), 2 * np.pi))
        gaps = np.diff(np.concatenate([a, a[:1] + 2 * np.pi]))
        feat[i, 1:] = (gaps.min(), gaps.max(), gaps.std())
    feat[:, 0] /= max(feat[:, 0].mean(), 1.0)            # degree, normalised
    return torch.as_tensor(feat)


def triangle_angle_features(Q):
    """Per-triangle interior angles, sorted, from the edge carriers.

    `q_e = vec3(dx dx^T)` gives `|dx_e|^2 = q_xx + q_yy`, so the three squared edge lengths are read
    straight off `Q` and the angles follow from the law of cosines.  Sorted, so the feature does not
    depend on which edge the mesh happened to list first.  Rotation-invariant by construction."""
    l2 = Q[:, 0, :] + Q[:, 2, :]                          # (n_tri, 3) squared lengths
    l2 = l2.clamp_min(1e-300)
    a2, b2, c2 = l2[:, 0], l2[:, 1], l2[:, 2]
    ang = []
    for x2, y2, z2 in ((a2, b2, c2), (b2, c2, a2), (c2, a2, b2)):
        cos = ((x2 + y2 - z2) / (2 * torch.sqrt(x2 * y2))).clamp(-1.0, 1.0)
        ang.append(torch.arccos(cos))
    return torch.sort(torch.stack(ang, -1), dim=-1).values


class ForwardGNNv2(nn.Module):
    """Graph -> per-triangle `M` (or `G`) -> C(s) -> C_eff.  Plain torch, no torch_geometric.

    Message passing runs on NODES; the readout gathers each triangle's three nodes and three bonds
    and emits that triangle's head weights.  v1 pooled GLOBALLY and emitted a bulk C6 directly,
    which is both non-equivariant and unable to express per-triangle structure."""

    def __init__(self, hidden=128, n_layers=5, n_edge_feat=3, passive=True, n_node_feat=N_NODE_FEAT,
                 n_tri_feat=N_TRI_FEAT):
        super().__init__()
        self.passive, self.hidden = passive, hidden
        self.node_embed = nn.Linear(n_node_feat, hidden)
        self.edge_mlp = nn.ModuleList(
            nn.Sequential(nn.Linear(2 * hidden + n_edge_feat, hidden), nn.SiLU(),
                          nn.Linear(hidden, hidden)) for _ in range(n_layers))
        self.node_mlp = nn.ModuleList(
            nn.Sequential(nn.Linear(2 * hidden, hidden), nn.SiLU(),
                          nn.Linear(hidden, hidden)) for _ in range(n_layers))
        n_out = 6 if passive else 9
        # A triangle contributes its 3 NODE embeddings (3*hidden) and its 3 BOND features
        # (3*n_edge_feat, raw -- the bonds are not embedded), hence 3*hidden + 3*n_edge_feat.
        self.readout = nn.Sequential(
            nn.Linear(3 * hidden + 3 * n_edge_feat + n_tri_feat, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, n_out))

    def forward(self, graph):
        u, v = graph['bond_u'].long(), graph['bond_v'].long()
        ef = graph['edge_feat']
        n_nodes = graph['n_nodes']
        h = self.node_embed(graph['node_feat'])
        for emlp, nmlp in zip(self.edge_mlp, self.node_mlp):
            m = emlp(torch.cat([h[u], h[v], ef], -1))
            agg = torch.zeros_like(h)
            cnt = torch.zeros(n_nodes, 1, dtype=h.dtype, device=h.device)
            for src, dst in ((u, v), (v, u)):                      # undirected: both directions
                agg = agg.index_add(0, dst, m)
                cnt = cnt.index_add(0, dst, torch.ones_like(cnt[:1]).expand(len(dst), 1))
            h = h + nmlp(torch.cat([h, agg / cnt.clamp_min(1.0)], -1))

        tv, tb = graph['tri_verts'].long(), graph['tri_bond'].long()
        feats = torch.cat([h[tv].reshape(len(tv), -1), ef[tb].reshape(len(tb), -1),
                           graph['tri_feat']], -1)
        raw = self.readout(feats)
        return self._to_G(raw)

    def _to_G(self, raw):
        """Head weights -> G.  Passive: G = M M^T with M lower-triangular, softplus diagonal, which
        makes G symmetric positive-semidefinite and hence C(s) = Q G Q^T PSD, structurally."""
        n = raw.shape[0]
        if not self.passive:
            return raw.reshape(n, 3, 3)
        M = torch.zeros(n, 3, 3, dtype=raw.dtype, device=raw.device)
        idx = torch.tril_indices(3, 3)
        M[:, idx[0], idx[1]] = raw
        d = torch.arange(3)
        M[:, d, d] = nn.functional.softplus(M[:, d, d])
        return M @ M.transpose(-1, -2)
