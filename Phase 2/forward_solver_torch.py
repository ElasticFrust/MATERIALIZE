"""
Differentiable forward solver for 2D elastic spring networks.

Implements the D2C (Disc-to-Continuum) homogenisation with exact edge-length
compatibility constraints (KKT correction), derived in the working notes
(Grossman & Boudaoud, PRR 2026, arXiv:2309.07844 — see derivation notes).

The original mean-field Woodbury solve (paper Eq. 19-20) is augmented with a
projection step that enforces: for every interior edge e = (s1, s2, Δx),

    [W(s1) - W(s2)]^{αβ}_{μν} Δx^μ Δx^ν = 0   for all loading modes αβ

This corrects the ~70% error on heterogeneous rigidity patterns by enforcing
local metric compatibility between neighbouring triangles (not just the global
mean-field constraint Σ_s W(s) = 0).

===========================================================================
THE 7-STEP FORWARD PIPELINE
===========================================================================

Step 1 — Edge vectors
    v[n,i] = pos[edge_a] − pos[edge_b]    for triangle n, edge i ∈ {0,1,2}

Step 2 — Bare elastic tensor  A[n]  (5 independent components)
    factor[n,i] = k[n,i] / l₀²[n,i] / 16

    a₀ = Σᵢ factor·vx⁴           (C₁₁₁₁)
    a₁ = Σᵢ factor·vx³·vy        (C₁₁₁₂)
    a₂ = Σᵢ factor·vx²·vy²       (C₁₁₂₂)
    a₃ = Σᵢ factor·vx·vy³        (C₁₂₂₂)
    a₄ = Σᵢ factor·vy⁴           (C₂₂₂₂)

Step 3 — Deviation from mean
    A₀   = mean(A, over triangles)
    δA[n] = A[n] − A₀

Step 4 — Build 9×9 block matrices  kron(M, I₃)

Step 5 — Woodbury solve + KKT edge-compatibility correction
    (A_full − B_full) W = −δA   solved by Woodbury (paper Eq. 20),
    then W is projected onto ker J (compatible metric fields) via the KKT
    system with multipliers Λ (one scalar per interior edge per loading mode).

Step 6 — 4-index tensor contraction
Step 7 — Homogenize and extract ν, E

===========================================================================
"""

import numpy as np
import torch
import torch.nn as nn


class ElasticSolver(nn.Module):
    """Differentiable forward solver for 2D elastic spring networks.

    Fixed inputs (set at construction, not differentiated):
        positions  (M, 2)   — node coordinates
        simplices  (N, 3)   — triangle → node index triplets
        edges      (N, 3, 2) — triangle → edge node-index pairs

    Differentiable inputs (passed to forward()):
        rigidities   (N, 3)  — spring rigidity per edge
        rest_lengths (N, 3)  — rest length per edge  (or None → use actual)
    """

    def __init__(self, positions, simplices, edges):
        super().__init__()

        pos_np = np.asarray(positions)
        sim_np = np.asarray(simplices)

        self.register_buffer('positions', torch.as_tensor(pos_np, dtype=torch.float64))
        self.register_buffer('simplices', torch.as_tensor(sim_np, dtype=torch.long))
        self.register_buffer('edges', torch.as_tensor(edges, dtype=torch.long))

        node_a = self.edges[:, :, 0]
        node_b = self.edges[:, :, 1]
        self.register_buffer(
            'edge_vecs', self.positions[node_a] - self.positions[node_b]
        )
        self.register_buffer(
            'actual_length2', (self.edge_vecs ** 2).sum(dim=2)
        )

        # Build edge-compatibility constraint matrix J
        self._build_edge_compatibility(pos_np, sim_np)

    def _build_edge_compatibility(self, positions, simplices):
        """Find interior edges; build J (3*E_int × 9*N) for KKT correction."""
        N = len(simplices)

        # Map sorted edge key → list of (tri_idx, node_a, node_b)
        edge_map = {}
        for s, tri in enumerate(simplices):
            for i in range(3):
                na, nb = int(tri[i]), int(tri[(i + 1) % 3])
                key = (min(na, nb), max(na, nb))
                if key not in edge_map:
                    edge_map[key] = []
                edge_map[key].append((s, na, nb))

        s1_list, s2_list, q_list = [], [], []
        for entries in edge_map.values():
            if len(entries) == 2:
                s1, na, nb = entries[0]
                s2 = entries[1][0]
                dx = positions[na, 0] - positions[nb, 0]
                dy = positions[na, 1] - positions[nb, 1]
                # q_e is invariant to edge direction (dx→-dx leaves q unchanged)
                s1_list.append(s1)
                s2_list.append(s2)
                q_list.append([dx * dx, 2.0 * dx * dy, dy * dy])

        E_int = len(s1_list)
        if E_int == 0:
            self.J = None
            return

        s1 = np.array(s1_list, dtype=np.int64)
        s2 = np.array(s2_list, dtype=np.int64)
        q  = np.array(q_list,  dtype=np.float64)  # (E_int, 3)

        # J: (3*E_int, 9*N)
        # Loading group k ∈ {0,1,2} → W components at positions {k, 3+k, 6+k}
        # Row k*E_int + e: +q[e] at s1[e]*9 + {k, 3+k, 6+k}
        #                  -q[e] at s2[e]*9 + {k, 3+k, 6+k}
        J = np.zeros((3 * E_int, 9 * N), dtype=np.float64)
        for k in range(3):
            rows = np.arange(E_int) + k * E_int
            for loc, blk in enumerate([0, 3, 6]):
                cols_s1 = s1 * 9 + blk + k
                cols_s2 = s2 * 9 + blk + k
                J[rows, cols_s1] =  q[:, loc]
                J[rows, cols_s2] = -q[:, loc]

        self.J = torch.as_tensor(J, dtype=torch.float64)

    def forward(self, rigidities, rest_lengths=None):
        """Run the full forward pipeline with KKT edge-compatibility correction.

        Args:
            rigidities:   (N, 3) tensor
            rest_lengths: (N, 3) tensor or None

        Returns:
            dict: elastic_tensor (6,), poisson, young, per_triangle (N,6),
                  bare (N,5), W (N,9)
        """
        vx = self.edge_vecs[:, :, 0]
        vy = self.edge_vecs[:, :, 1]

        if rest_lengths is not None:
            length2 = rest_lengths ** 2
        else:
            length2 = self.actual_length2

        factor = rigidities / length2 / 16.0

        bare = torch.stack([
            (factor * vx ** 4).sum(dim=1),
            (factor * vx ** 3 * vy).sum(dim=1),
            (factor * vx ** 2 * vy ** 2).sum(dim=1),
            (factor * vx * vy ** 3).sum(dim=1),
            (factor * vy ** 4).sum(dim=1),
        ], dim=1)

        mean_tensor = bare.mean(dim=0)
        delta = bare - mean_tensor

        A_blocks = _batch_to_9x9(bare)
        B_blocks = _batch_to_9x9(delta)
        dA_vecs  = _batch_to_9vec(delta)

        # Move J to same device/dtype as tensors (geometry buffer)
        J = self.J
        if J is not None:
            J = J.to(dtype=bare.dtype, device=bare.device)

        W = _woodbury_solve(A_blocks, B_blocks, dA_vecs, J=J)

        actual = _compute_actual_elastic_tensor(bare, W)
        C = actual.mean(dim=0)

        poisson = (C[2] * C[3] - C[1] * C[4]) / (C[0] * C[3] - C[1] ** 2)
        young = (
            C[2] ** 2 * C[3]
            - 2 * C[1] * C[2] * C[4]
            + C[1] ** 2 * C[5]
            + C[0] * (C[4] ** 2 - C[3] * C[5])
        ) / (C[1] ** 2 - C[0] * C[3])

        return {
            'elastic_tensor': C,
            'poisson': poisson,
            'young': young,
            'per_triangle': actual,
            'bare': bare,
            'W': W,
        }


# ---------------------------------------------------------------------------
# Pure-function helpers
# ---------------------------------------------------------------------------

def _batch_to_9x9(vecs5):
    """(N, 5) → (N, 9, 9) block matrices kron(M, I₃)."""
    N = vecs5.shape[0]
    a0, a1, a2, a3, a4 = vecs5[:, 0], vecs5[:, 1], vecs5[:, 2], vecs5[:, 3], vecs5[:, 4]
    M = torch.stack([
        torch.stack([a0,    2*a1, a2  ], dim=1),
        torch.stack([a1,    2*a2, a3  ], dim=1),
        torch.stack([a2,    2*a3, a4  ], dim=1),
    ], dim=1)  # (N, 3, 3)
    I3 = torch.eye(3, dtype=vecs5.dtype, device=vecs5.device)
    return torch.einsum('nij,kl->nikjl', M, I3).reshape(N, 9, 9)


def _batch_to_9vec(vecs5):
    """(N, 5) → (N, 9) vectors  [a0,a1,a2, a1,a2,a3, a2,a3,a4]."""
    a0, a1, a2, a3, a4 = [vecs5[:, i] for i in range(5)]
    return torch.stack([a0, a1, a2, a1, a2, a3, a2, a3, a4], dim=1)


def _woodbury_solve(A_blocks, B_blocks, dA_vecs, J=None):
    """Solve (A_full − B_full) W = −dA via Woodbury, then apply KKT correction.

    Without J (J=None): recovers paper Eq. 20 exactly.
    With J (3*E_int × 9N): projects W₀ onto ker J in the P-metric via the KKT
    system  [P  Jᵀ; J  0] [W; Λ] = [-δA; 0].

    Args:
        A_blocks: (N, 9, 9)
        B_blocks: (N, 9, 9)
        dA_vecs:  (N, 9)
        J:        (3*E_int, 9*N) or None

    Returns:
        W: (N, 9)
    """
    N = A_blocks.shape[0]
    eps = 1e-14 * A_blocks.abs().max()
    I9 = torch.eye(9, dtype=A_blocks.dtype, device=A_blocks.device)
    A_reg = A_blocks + eps * I9.unsqueeze(0)
    A_inv = torch.linalg.inv(A_reg)          # (N, 9, 9)

    y   = torch.einsum('nij,nj->ni', A_inv, dA_vecs)       # (N, 9)
    Vy  = torch.einsum('nij,nj->i',  B_blocks, y)           # (9,)
    S   = torch.einsum('nij,njk->ik', B_blocks, A_inv) / N  # (9, 9)
    IminusS = I9 - S
    z   = torch.linalg.solve(IminusS, Vy)                   # (9,)
    W0  = -(y + torch.einsum('nij,j->ni', A_inv, z) / N)   # (N, 9)

    if J is None or J.shape[0] == 0:
        return W0

    # ------------------------------------------------------------------
    # KKT correction: project W₀ onto ker J in the P-metric
    #
    # P⁻¹ applied to a batch of right-hand sides (columns of Jᵀ):
    #   Jᵀ reshaped to (N, 9, M) where M = 3*E_int
    #   y_J[n]   = A_inv[n] @ Jᵀ[n]          (N, 9, M)
    #   Vy_J     = Σₙ B[n] @ y_J[n]           (9, M)
    #   z_J      = (I-S)⁻¹ @ Vy_J             (9, M)
    #   PinvJt   = y_J + (1/N) A_inv @ z_J    (9N, M)
    #
    # Then G = J @ PinvJt,  r = J @ W₀,  Λ = G⁻¹ r,  W = W₀ - PinvJt Λ
    # ------------------------------------------------------------------
    M_c = J.shape[0]  # 3 * E_int

    Jt          = J.T                               # (9N, M_c)
    Jt_batched  = Jt.reshape(N, 9, M_c)            # (N, 9, M_c)

    y_J   = torch.einsum('nij,njm->nim', A_inv, Jt_batched)          # (N, 9, M_c)
    Vy_J  = torch.einsum('nij,njm->im',  B_blocks, y_J)              # (9, M_c)
    z_J   = torch.linalg.solve(IminusS, Vy_J)                        # (9, M_c)
    corr_J = torch.einsum('nij,jm->nim', A_inv, z_J) / N             # (N, 9, M_c)
    PinvJt = (y_J + corr_J).reshape(9 * N, M_c)                      # (9N, M_c)

    G      = J @ PinvJt                                               # (M_c, M_c)
    r      = J @ W0.reshape(-1)                                       # (M_c,)
    Lambda = torch.linalg.solve(G, r)                                 # (M_c,)

    W = (W0.reshape(-1) - PinvJt @ Lambda).reshape(N, 9)
    return W


def _compute_actual_elastic_tensor(bare_tensors, Ws):
    """Compute per-triangle effective elastic tensor via 4-index contraction.

        C_{manb} = A_{manb} + A_{minj}W_{iajb} + A_{aibj}W_{imjn} + A_{kilj}W_{iajb}W_{kmln}

    Args:
        bare_tensors: (N, 5)
        Ws:           (N, 9)

    Returns:
        (N, 6) — [C₁₁₁₁, C₁₁₁₂, C₁₁₂₂, C₂₁₁₂, C₂₁₂₂, C₂₂₂₂]
    """
    N = bare_tensors.shape[0]
    a, w = bare_tensors, Ws

    A_mat = torch.zeros(N, 4, 4, dtype=a.dtype, device=a.device)
    A_mat[:, 0, 0] = a[:, 0]; A_mat[:, 0, 1] = a[:, 1]
    A_mat[:, 0, 2] = a[:, 1]; A_mat[:, 0, 3] = a[:, 2]
    A_mat[:, 1, 0] = a[:, 1]; A_mat[:, 1, 1] = a[:, 2]
    A_mat[:, 1, 2] = a[:, 2]; A_mat[:, 1, 3] = a[:, 3]
    A_mat[:, 2, 0] = a[:, 1]; A_mat[:, 2, 1] = a[:, 2]
    A_mat[:, 2, 2] = a[:, 2]; A_mat[:, 2, 3] = a[:, 3]
    A_mat[:, 3, 0] = a[:, 2]; A_mat[:, 3, 1] = a[:, 3]
    A_mat[:, 3, 2] = a[:, 3]; A_mat[:, 3, 3] = a[:, 4]

    W_mat = torch.zeros(N, 4, 4, dtype=w.dtype, device=w.device)
    W_mat[:, 0, 0] = w[:, 0]; W_mat[:, 0, 1] = w[:, 1]
    W_mat[:, 0, 2] = w[:, 3]; W_mat[:, 0, 3] = w[:, 4]
    W_mat[:, 1, 0] = w[:, 1]; W_mat[:, 1, 1] = w[:, 2]
    W_mat[:, 1, 2] = w[:, 4]; W_mat[:, 1, 3] = w[:, 5]
    W_mat[:, 2, 0] = w[:, 3]; W_mat[:, 2, 1] = w[:, 4]
    W_mat[:, 2, 2] = w[:, 6]; W_mat[:, 2, 3] = w[:, 7]
    W_mat[:, 3, 0] = w[:, 4]; W_mat[:, 3, 1] = w[:, 5]
    W_mat[:, 3, 2] = w[:, 7]; W_mat[:, 3, 3] = w[:, 8]

    A4 = A_mat.reshape(N, 2, 2, 2, 2)
    W4 = W_mat.reshape(N, 2, 2, 2, 2)

    S2 = torch.einsum('tminj,tiajb->tmanb', A4, W4)
    S3 = torch.einsum('taibj,timjn->tmanb', A4, W4)
    S4 = torch.einsum('tkilj,tiajb,tkmln->tmanb', A4, W4, W4)

    C_mat = (A4 + S2 + S3 + S4).reshape(N, 4, 4)
    return torch.stack([
        C_mat[:, 0, 0], C_mat[:, 0, 1], C_mat[:, 1, 1],
        C_mat[:, 1, 2], C_mat[:, 2, 3], C_mat[:, 3, 3],
    ], dim=1)


def from_triangulation(triangulation):
    """Create an ElasticSolver from a scipy.spatial.Delaunay object."""
    import numpy as np
    points    = triangulation.points
    simplices = triangulation.simplices
    edges = np.array([
        [(tri[i], tri[j]) for i in range(3) for j in range(i + 1, 3)]
        for tri in simplices
    ])
    solver = ElasticSolver(points, simplices, edges)
    N = len(simplices)
    default_rigs = torch.ones(N, 3, dtype=torch.float64)
    default_rl   = solver.actual_length2.sqrt()
    return solver, default_rigs, default_rl
