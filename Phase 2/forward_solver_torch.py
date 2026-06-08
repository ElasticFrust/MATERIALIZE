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


def _angle_gradient_vec(a, b):
    """∂θ(a,b)/∂(g11,g12,g22) at reference Euclidean metric. Returns 3-vector."""
    a2 = np.dot(a, a)
    b2 = np.dot(b, b)
    ab = np.dot(a, b)
    la = np.sqrt(a2)
    lb = np.sqrt(b2)
    cos_th = np.clip(ab / (la * lb), -1.0 + 1e-10, 1.0 - 1e-10)
    sin_th = np.sqrt(1.0 - cos_th ** 2)
    if sin_th < 1e-10:
        return np.zeros(3)
    d_ab = np.array([a[0]*b[0], a[0]*b[1]+a[1]*b[0], a[1]*b[1]])
    d_a2 = np.array([a[0]**2,   2*a[0]*a[1],          a[1]**2])
    d_b2 = np.array([b[0]**2,   2*b[0]*b[1],          b[1]**2])
    d_cos = d_ab / (la * lb) - cos_th * (d_a2 / (2*a2) + d_b2 / (2*b2))
    return -d_cos / sin_th


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

        v0, v1, v2 = pos_np[sim_np[:, 0]], pos_np[sim_np[:, 1]], pos_np[sim_np[:, 2]]
        areas = 0.5 * np.abs(
            (v1[:, 0] - v0[:, 0]) * (v2[:, 1] - v0[:, 1]) -
            (v1[:, 1] - v0[:, 1]) * (v2[:, 0] - v0[:, 0])
        )
        self.register_buffer(
            'area_weights', torch.as_tensor(areas / areas.sum(), dtype=torch.float64)
        )
        self.register_buffer(
            'actual_length2', (self.edge_vecs ** 2).sum(dim=2)
        )

        # Build edge-compatibility constraint matrix J
        self._build_edge_compatibility(pos_np, sim_np)

        # Build vertex-angle compatibility constraints
        self._build_vertex_angle_constraints(pos_np, sim_np)

    def _build_edge_compatibility(self, positions, simplices):
        """Find interior edges; store (s1, s2, q) for sparse KKT correction."""
        N = len(simplices)

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
                s1_list.append(s1)
                s2_list.append(s2)
                q_list.append([dx * dx, 2.0 * dx * dy, dy * dy])

        E_int = len(s1_list)
        if E_int == 0:
            self.J = None
            self.kkt_arrays = None
            return

        s1 = np.array(s1_list, dtype=np.int64)
        s2 = np.array(s2_list, dtype=np.int64)
        q  = np.array(q_list,  dtype=np.float64)  # (E_int, 3)

        # Always keep the sparse representation (used by the efficient KKT path)
        self.kkt_arrays = (s1, s2, q)

        # Build dense J only for small meshes where the O(N²) path is acceptable
        # (≤ 500 triangles → y_J tensor < ~200 MB; larger meshes use the sparse path)
        DENSE_THRESHOLD = 500
        if N <= DENSE_THRESHOLD:
            M_c = 3 * E_int
            J = np.zeros((M_c, 9 * N), dtype=np.float64)
            for k in range(3):
                rows = np.arange(E_int) + k * E_int
                for loc, blk in enumerate([0, 3, 6]):
                    J[rows, s1 * 9 + blk + k] =  q[:, loc]
                    J[rows, s2 * 9 + blk + k] = -q[:, loc]
            self.J = torch.as_tensor(J, dtype=torch.float64)
        else:
            self.J = None  # sparse path will be used in forward()

    def _build_vertex_angle_constraints(self, positions, simplices):
        """For each interior vertex compute ∂θ/∂g for each incident triangle.

        Stores self.angle_arrays = (v_int_idx, s_idx, a_vecs, n_int) or None.
          v_int_idx : (n_pairs,) int64 — remapped interior vertex index
          s_idx     : (n_pairs,) int64 — triangle index
          a_vecs    : (n_pairs, 3) float64 — angle gradient vector
          n_int     : int — number of interior vertices
        """
        n_pts = len(positions)

        # Boundary vertices touch exactly one triangle on their boundary edge
        edge_count = {}
        for tri in simplices:
            for i in range(3):
                na, nb = int(tri[i]), int(tri[(i+1) % 3])
                key = (min(na, nb), max(na, nb))
                edge_count[key] = edge_count.get(key, 0) + 1
        boundary_verts = set()
        for (na, nb), cnt in edge_count.items():
            if cnt == 1:
                boundary_verts.update([na, nb])

        v2tri = [[] for _ in range(n_pts)]
        for s, tri in enumerate(simplices):
            for v in tri:
                v2tri[v].append(s)

        v_int_list, s_list, a_list = [], [], []
        int_count = 0
        for v in range(n_pts):
            if v in boundary_verts:
                continue
            for s in v2tri[v]:
                tri = simplices[s]
                vi = list(tri).index(v)
                v1 = tri[(vi + 1) % 3]
                v2 = tri[(vi + 2) % 3]
                a = positions[v1] - positions[v]
                b = positions[v2] - positions[v]
                dth = _angle_gradient_vec(a, b)
                dth[1] /= 2   # engineering Voigt: component 1 = 2*g_12
                v_int_list.append(int_count)
                s_list.append(s)
                a_list.append(dth)
            int_count += 1

        if int_count == 0 or len(v_int_list) == 0:
            self.angle_arrays = None
            return

        self.angle_arrays = (
            np.array(v_int_list, dtype=np.int64),
            np.array(s_list,     dtype=np.int64),
            np.array(a_list,     dtype=np.float64),  # (n_pairs, 3)
            int_count,
        )

    def forward(self, rigidities, rest_lengths=None, area_weighted=False, use_kkt=True,
                use_angle_kkt=False):
        """Run the forward pipeline.

        Args:
            rigidities:     (N, 3) tensor
            rest_lengths:   (N, 3) tensor or None
            area_weighted:  if True, use volume-weighted constraint Σ w_n δg(n)=0;
                            if False use arithmetic constraint Σ δg(n)=0.
            use_kkt:        if True, apply edge-compatibility (KKT) correction;
                            if False, pure mean-field Woodbury only (no KKT).
            use_angle_kkt:  if True, apply BOTH edge and vertex-angle compatibility
                            constraints jointly (superset of use_kkt); if False,
                            falls back to use_kkt behaviour (default False).

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

        if area_weighted:
            w = self.area_weights.to(dtype=bare.dtype, device=bare.device)  # (N,)
            mean_tensor = (bare * w.unsqueeze(1)).sum(0)
        else:
            w = None
            mean_tensor = bare.mean(dim=0)

        delta    = bare - mean_tensor
        A_blocks = _batch_to_9x9(bare)
        B_blocks = _batch_to_9x9(delta)
        dA_vecs  = _batch_to_9vec(delta)

        if use_kkt and self.kkt_arrays is not None:
            # All KKT paths (edge-only or edge+angle) use the combined solver.
            # angle_arrays=None → edge-only; the per-loading-mode decoupling
            # reduces the sparse solve from 3*E_int to E_int per loading mode.
            _angle = self.angle_arrays if use_angle_kkt else None
            _weights = self.area_weights.numpy() if area_weighted else None
            W_np = _woodbury_kkt_sparse_combined(
                A_blocks, B_blocks, dA_vecs,
                self.kkt_arrays, _angle, weights=_weights)
            W = torch.as_tensor(W_np, dtype=bare.dtype, device=bare.device)
        elif use_kkt and self.J is not None:
            W = _woodbury_solve(A_blocks, B_blocks, dA_vecs,
                                J=self.J.to(dtype=bare.dtype, device=bare.device),
                                weights=w)
        else:
            W = _woodbury_solve(A_blocks, B_blocks, dA_vecs, J=None, weights=w)

        actual = _compute_actual_elastic_tensor(bare, W)
        if area_weighted:
            C = (actual * w.unsqueeze(1)).sum(0)
        else:
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


def _woodbury_solve(A_blocks, B_blocks, dA_vecs, J=None, weights=None):
    """Solve (A_full − B_full) W = −dA via Woodbury, then apply KKT correction.

    Without J (J=None): recovers paper Eq. 20 exactly.
    With J (3*E_int × 9N): projects W₀ onto ker J in the P-metric via the KKT
    system  [P  Jᵀ; J  0] [W; Λ] = [-δA; 0].

    Args:
        A_blocks: (N, 9, 9)
        B_blocks: (N, 9, 9)
        dA_vecs:  (N, 9)
        J:        (3*E_int, 9*N) or None
        weights:  (N,) area weights summing to 1, or None for uniform 1/N

    Returns:
        W: (N, 9)
    """
    N = A_blocks.shape[0]
    eps = 1e-14 * A_blocks.abs().max()
    I9 = torch.eye(9, dtype=A_blocks.dtype, device=A_blocks.device)
    A_reg = A_blocks + eps * I9.unsqueeze(0)
    A_inv = torch.linalg.inv(A_reg)          # (N, 9, 9)

    y = torch.einsum('nij,nj->ni', A_inv, dA_vecs)         # (N, 9)

    if weights is not None:
        w = weights                                         # (N,)
        Vy = torch.einsum('n,nij,nj->i', w, B_blocks, y)   # (9,)
        S  = torch.einsum('n,nij,njk->ik', w, B_blocks, A_inv)  # (9,9) no /N
        IminusS = I9 - S
        z  = torch.linalg.solve(IminusS, Vy)
        W0 = -(y + torch.einsum('nij,j->ni', A_inv, z))    # no /N
    else:
        Vy = torch.einsum('nij,nj->i',  B_blocks, y)       # (9,)
        S  = torch.einsum('nij,njk->ik', B_blocks, A_inv) / N  # (9,9)
        IminusS = I9 - S
        z  = torch.linalg.solve(IminusS, Vy)
        W0 = -(y + torch.einsum('nij,j->ni', A_inv, z) / N)

    if J is None or J.shape[0] == 0:
        return W0

    # ------------------------------------------------------------------
    # KKT correction: project W₀ onto ker J in the P-metric
    #
    # P⁻¹ applied to a batch of right-hand sides (columns of Jᵀ):
    #   Jᵀ reshaped to (N, 9, M) where M = 3*E_int
    #   y_J[n]   = A_inv[n] @ Jᵀ[n]          (N, 9, M)
    #   Vy_J     = Σₙ [w_n] B[n] @ y_J[n]    (9, M)
    #   z_J      = (I-S)⁻¹ @ Vy_J             (9, M)
    #   PinvJt   = y_J + [1/N or 1] A_inv @ z_J  (9N, M)
    # ------------------------------------------------------------------
    M_c = J.shape[0]  # 3 * E_int

    Jt         = J.T                               # (9N, M_c)
    Jt_batched = Jt.reshape(N, 9, M_c)            # (N, 9, M_c)

    y_J = torch.einsum('nij,njm->nim', A_inv, Jt_batched)   # (N, 9, M_c)

    if weights is not None:
        Vy_J = torch.einsum('n,nij,njm->im', w, B_blocks, y_J)   # (9, M_c)
    else:
        Vy_J = torch.einsum('nij,njm->im', B_blocks, y_J)         # (9, M_c)

    z_J    = torch.linalg.solve(IminusS, Vy_J)              # (9, M_c)
    corr_J = torch.einsum('nij,jm->nim', A_inv, z_J)        # (N, 9, M_c)
    if weights is None:
        corr_J = corr_J / N
    PinvJt = (y_J + corr_J).reshape(9 * N, M_c)             # (9N, M_c)

    G      = J @ PinvJt                                      # (M_c, M_c)
    r      = J @ W0.reshape(-1)                              # (M_c,)
    Lambda = torch.linalg.solve(G, r)                        # (M_c,)

    W = (W0.reshape(-1) - PinvJt @ Lambda).reshape(N, 9)
    return W


def _woodbury_kkt_sparse(A_blocks, B_blocks, dA_vecs, kkt_arrays):
    """Memory-efficient KKT correction using sparse G_local + rank-9 decomposition.

    G = J P⁻¹ Jᵀ = G_local + (1/N) H IminusS⁻¹ Kᵀ

    where:
      G_local = J A_diag⁻¹ Jᵀ  (sparse, ~9 nnz/row)
      H[i,:]  = (J A_diag⁻¹ U)[i,:]  (M_c × 9)
      K[i,:]  = (V A_diag⁻¹ Jᵀ)[i,:]  (M_c × 9)

    Solves via Woodbury on G_local (sparse LU) + rank-9 correction.
    Returns W as numpy array (N, 9) — no gradient.
    """
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    s1_arr, s2_arr, q_arr = kkt_arrays
    E_int = len(s1_arr)
    M_c   = 3 * E_int
    N     = A_blocks.shape[0]

    A_np = A_blocks.detach().double().numpy()  # (N, 9, 9)
    B_np = B_blocks.detach().double().numpy()
    dA_np = dA_vecs.detach().double().numpy()

    eps = 1e-14 * np.abs(A_np).max()
    I9  = np.eye(9)
    A_reg = A_np + eps * I9[None]
    A_inv = np.linalg.inv(A_reg)              # (N, 9, 9)

    # Woodbury base solve → W0
    y    = np.einsum('nij,nj->ni', A_inv, dA_np)
    Vy   = np.einsum('nij,nj->i',  B_np, y)
    S    = np.einsum('nij,njk->ik', B_np, A_inv) / N
    IminusS = I9 - S
    z    = np.linalg.solve(IminusS, Vy)
    W0   = -(y + np.einsum('nij,j->ni', A_inv, z) / N)  # (N, 9)

    # ── H and K  (M_c × 9) ──────────────────────────────────────────────────
    # H[k*E+e, j] = Σ_loc q[e,loc] * (A_inv[s1, 3*loc+k, j] − A_inv[s2, 3*loc+k, j])
    # K[k*E+e, j] = Σ_loc q[e,loc] * (BA_inv[s1, j, 3*loc+k] − BA_inv[s2, j, 3*loc+k])
    BA_inv = np.einsum('nij,njk->nik', B_np, A_inv)  # (N, 9, 9)

    # H[k*E+e, :] and K[k*E+e, :] each have a unique row index, so plain
    # += is safe (no duplicate row indices within one (k, loc) pass).
    H = np.zeros((M_c, 9))
    K = np.zeros((M_c, 9))
    for k in range(3):
        rk = np.arange(E_int) + k * E_int
        for loc in range(3):
            col = 3 * loc + k
            H[rk] += q_arr[:, loc:loc+1] * (A_inv[s1_arr, col, :] - A_inv[s2_arr, col, :])
            K[rk] += q_arr[:, loc:loc+1] * (BA_inv[s1_arr, :, col] - BA_inv[s2_arr, :, col])

    # ── Sparse G_local = J A_diag⁻¹ Jᵀ ────────────────────────────────────
    # G_local[k1*E+e1, k2*E+e2] = Σ_{n shared} sg1*sg2 * q[e1] @ A_sub(n,k1,k2) @ q[e2]
    # A_sub(n,k1,k2)[loc1,loc2] = A_inv[n, 3*loc1+k1, 3*loc2+k2]  (3×3 sub-block)
    from collections import defaultdict
    tri_edges = defaultdict(list)
    for e in range(E_int):
        tri_edges[s1_arr[e]].append((e, +1))
        tri_edges[s2_arr[e]].append((e, -1))

    rows_g, cols_g, data_g = [], [], []
    LOC_ROWS = [np.array([k, 3+k, 6+k]) for k in range(3)]  # row idx sets per loading group

    for n, elist in tri_edges.items():
        An = A_inv[n]  # (9, 9)
        for e1, sg1 in elist:
            for e2, sg2 in elist:
                val_sg = sg1 * sg2
                for k1 in range(3):
                    row = k1 * E_int + e1
                    for k2 in range(3):
                        col = k2 * E_int + e2
                        A_sub = An[np.ix_(LOC_ROWS[k1], LOC_ROWS[k2])]  # 3×3
                        val = val_sg * q_arr[e1] @ A_sub @ q_arr[e2]
                        rows_g.append(row)
                        cols_g.append(col)
                        data_g.append(val)

    G_local = sp.coo_matrix((data_g, (rows_g, cols_g)), shape=(M_c, M_c)).tocsc()
    reg_gl  = 1e-12 * abs(max(data_g, default=1.0))
    G_local = G_local + reg_gl * sp.eye(M_c, format='csc')

    # ── r = J W₀ ────────────────────────────────────────────────────────────
    # r[k*E+e] = Σ_{loc} q[e,loc] * (W0[s1*9+3*loc+k] - W0[s2*9+3*loc+k])
    # Each row of J has a unique index (k*E+e), so plain += is safe here.
    W0_flat = W0.reshape(-1)
    r = np.zeros(M_c)
    for k in range(3):
        rk = np.arange(E_int) + k * E_int
        for loc in range(3):
            col = 3 * loc + k
            r[rk] += q_arr[:, loc] * (W0_flat[s1_arr * 9 + col] - W0_flat[s2_arr * 9 + col])

    # ── Woodbury solve on (G_local + (1/N) H IminusS⁻¹ Kᵀ) Λ = r ──────────
    # Woodbury: (A + (1/N) H T Kᵀ)⁻¹ r  with T = IminusS⁻¹
    # C = (1/N) T  →  C⁻¹ = N IminusS  →  M_mat = N*IminusS + Kᵀ G_local⁻¹ H
    G_lu  = spla.factorized(G_local)
    Lam0  = G_lu(r)                                   # G_local⁻¹ r
    Y     = np.column_stack([G_lu(H[:, j]) for j in range(9)])  # G_local⁻¹ H (M_c,9)
    M_mat = N * IminusS + K.T @ Y                     # 9×9  (C⁻¹ + V A⁻¹ U)
    c     = np.linalg.solve(M_mat, K.T @ Lam0)        # 9-vec
    Lambda = Lam0 - Y @ c                             # M_c-vec

    # ── W = W₀ − P⁻¹ (Jᵀ Λ) ────────────────────────────────────────────────
    # Jᵀ Λ — multiple edges can map to the same (triangle, col) index,
    # so np.add.at is required to correctly accumulate duplicates.
    JtLam = np.zeros(9 * N)
    for k in range(3):
        rk = np.arange(E_int) + k * E_int
        Lk  = Lambda[rk]
        for loc in range(3):
            col = 3 * loc + k
            np.add.at(JtLam, s1_arr * 9 + col, q_arr[:, loc] * Lk)
            np.add.at(JtLam, s2_arr * 9 + col, -q_arr[:, loc] * Lk)

    JtLam_b  = JtLam.reshape(N, 9)
    AinvJtLam = np.einsum('nij,nj->ni', A_inv, JtLam_b)
    gs        = np.einsum('nij,nj->i', B_np, AinvJtLam)
    z_c       = np.linalg.solve(IminusS, gs)
    PinvJtLam = AinvJtLam + np.einsum('nij,j->ni', A_inv, z_c) / N

    return W0 - PinvJtLam


def _woodbury_kkt_sparse_aw(A_blocks, B_blocks, dA_vecs, kkt_arrays, weights):
    """Area-weighted sparse KKT Woodbury (volume-weighted constraint Σ w_n δg(n)=0).

    Identical to _woodbury_kkt_sparse except every 1/N is replaced by w_n and
    the /N correction factors in W0 and PinvJtLam are dropped.

    Args:
        weights: (N,) numpy array summing to 1

    Returns:
        W: (N, 9) numpy array
    """
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    from collections import defaultdict

    s1_arr, s2_arr, q_arr = kkt_arrays
    E_int = len(s1_arr)
    M_c   = 3 * E_int
    N     = A_blocks.shape[0]
    w     = weights  # (N,)

    A_np  = A_blocks.detach().double().numpy()
    B_np  = B_blocks.detach().double().numpy()
    dA_np = dA_vecs.detach().double().numpy()

    eps   = 1e-14 * np.abs(A_np).max()
    I9    = np.eye(9)
    A_inv = np.linalg.inv(A_np + eps * I9[None])

    y    = np.einsum('nij,nj->ni', A_inv, dA_np)
    Vy   = np.einsum('n,nij,nj->i',   w, B_np, y)
    S    = np.einsum('n,nij,njk->ik', w, B_np, A_inv)    # no /N
    IminusS = I9 - S
    z    = np.linalg.solve(IminusS, Vy)
    W0   = -(y + np.einsum('nij,j->ni', A_inv, z))       # no /N

    BA_inv_aw = np.einsum('n,nij,njk->nik', w, B_np, A_inv)

    H = np.zeros((M_c, 9))
    K = np.zeros((M_c, 9))
    for k in range(3):
        rk = np.arange(E_int) + k * E_int
        for loc in range(3):
            col = 3 * loc + k
            H[rk] += q_arr[:, loc:loc+1] * (A_inv[s1_arr, col, :] - A_inv[s2_arr, col, :])
            K[rk] += q_arr[:, loc:loc+1] * (BA_inv_aw[s1_arr, :, col] - BA_inv_aw[s2_arr, :, col])

    tri_edges = defaultdict(list)
    for e in range(E_int):
        tri_edges[s1_arr[e]].append((e, +1))
        tri_edges[s2_arr[e]].append((e, -1))

    rows_g, cols_g, data_g = [], [], []
    LOC_ROWS = [np.array([k, 3+k, 6+k]) for k in range(3)]
    for n, elist in tri_edges.items():
        An = A_inv[n]
        for e1, sg1 in elist:
            for e2, sg2 in elist:
                val_sg = sg1 * sg2
                for k1 in range(3):
                    row = k1 * E_int + e1
                    for k2 in range(3):
                        col = k2 * E_int + e2
                        val = val_sg * q_arr[e1] @ An[np.ix_(LOC_ROWS[k1], LOC_ROWS[k2])] @ q_arr[e2]
                        rows_g.append(row); cols_g.append(col); data_g.append(val)

    G_local = sp.coo_matrix((data_g, (rows_g, cols_g)), shape=(M_c, M_c)).tocsc()
    G_local = G_local + 1e-12 * abs(max(data_g, default=1.0)) * sp.eye(M_c, format='csc')

    W0_flat = W0.reshape(-1)
    r = np.zeros(M_c)
    for k in range(3):
        rk = np.arange(E_int) + k * E_int
        for loc in range(3):
            col = 3 * loc + k
            r[rk] += q_arr[:, loc] * (W0_flat[s1_arr*9+col] - W0_flat[s2_arr*9+col])

    G_lu   = spla.factorized(G_local)
    Lam0   = G_lu(r)
    Y      = np.column_stack([G_lu(H[:, j]) for j in range(9)])
    M_mat  = IminusS + K.T @ Y                   # no N factor
    c      = np.linalg.solve(M_mat, K.T @ Lam0)
    Lambda = Lam0 - Y @ c

    JtLam = np.zeros(9 * N)
    for k in range(3):
        rk = np.arange(E_int) + k * E_int
        Lk = Lambda[rk]
        for loc in range(3):
            col = 3 * loc + k
            np.add.at(JtLam, s1_arr*9+col,  q_arr[:, loc] * Lk)
            np.add.at(JtLam, s2_arr*9+col, -q_arr[:, loc] * Lk)

    JtLam_b   = JtLam.reshape(N, 9)
    AinvJtLam = np.einsum('nij,nj->ni', A_inv, JtLam_b)
    gs        = np.einsum('n,nij,nj->i', w, B_np, AinvJtLam)
    z_c       = np.linalg.solve(IminusS, gs)
    PinvJtLam = AinvJtLam + np.einsum('nij,j->ni', A_inv, z_c)   # no /N

    return W0 - PinvJtLam


def _woodbury_kkt_sparse_combined(A_blocks, B_blocks, dA_vecs,
                                   kkt_arrays, angle_arrays, weights=None):
    """KKT correction with edge-length AND vertex-angle compatibility constraints.

    Jointly enforces:
      - E_int edge constraints  (kkt_arrays: s1, s2, q)
      - n_int angle constraints (angle_arrays: v_int, s, a_vec, n_int)

    Because A = kron(M, I_3) the 3 loading modes decouple completely.
    G_0 (M_c × M_c) is factorised ONCE; 3 RHS are solved independently.
    weights: None → uniform 1/N; (N,) array → area-weighted.
    """
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    from collections import defaultdict

    s1_arr, s2_arr, q_arr = kkt_arrays
    E_int = len(s1_arr)

    if angle_arrays is not None:
        v_arr, sv_arr, a_arr, n_int = angle_arrays
    else:
        v_arr  = np.empty(0, dtype=np.int64)
        sv_arr = np.empty(0, dtype=np.int64)
        a_arr  = np.empty((0, 3), dtype=np.float64)
        n_int  = 0
    n_pairs = len(v_arr)

    M_c = E_int + n_int   # constraints per loading mode
    N   = A_blocks.shape[0]

    A_np  = A_blocks.detach().double().numpy()   # (N, 9, 9) = kron(M, I_3)
    B_np  = B_blocks.detach().double().numpy()   # (N, 9, 9) = kron(δM, I_3)
    dA_np = dA_vecs.detach().double().numpy()    # (N, 9)

    # ── Extract 3×3 blocks (A = kron(M, I_3) → M = A[0:9:3, 0:9:3]) ────────
    IDX3 = np.array([0, 3, 6])
    M_np    = A_np[:, IDX3[:, None], IDX3[None, :]]        # (N, 3, 3)
    dM_np   = B_np[:, IDX3[:, None], IDX3[None, :]]        # (N, 3, 3)
    eps3    = 1e-14 * np.abs(M_np).max()
    I3      = np.eye(3)
    M_inv   = np.linalg.inv(M_np + eps3 * I3[None])        # (N, 3, 3)

    # ── Woodbury base solve → W0 ─────────────────────────────────────────────
    # dA_np has shape (N, 9); each row [d0,d1,d2, d1,d2,d3, d2,d3,d4]
    # Loading mode k: indices LOC_ROWS[k] = {k, 3+k, 6+k} → 3-vector
    dA_3 = np.stack([dA_np[:, [k, 3+k, 6+k]] for k in range(3)], axis=2)  # (N, 3, 3)
    y3   = np.einsum('nij,njk->nik', M_inv, dA_3)          # (N, 3, 3)

    if weights is not None:
        w  = weights
        Vy3 = np.einsum('n,nij,njk->ik', w, dM_np, y3)    # (3, 3)
        S3  = np.einsum('n,nij,njk->ik', w, dM_np, M_inv) # (3, 3)
    else:
        Vy3 = np.einsum('nij,njk->ik', dM_np, y3)          # (3, 3)
        S3  = np.einsum('nij,njk->ik', dM_np, M_inv) / N  # (3, 3)

    IminusS3 = I3 - S3                                     # (3, 3)
    Z3       = np.linalg.solve(IminusS3, Vy3)              # (3, 3)

    if weights is not None:
        W0_3 = -(y3 + np.einsum('nij,jk->nik', M_inv, Z3))   # (N, 3, 3)
    else:
        W0_3 = -(y3 + np.einsum('nij,jk->nik', M_inv, Z3) / N)  # (N, 3, 3)

    # Reconstruct W0 in flat (N, 9) layout
    W0 = np.zeros((N, 9))
    for k in range(3):
        W0[:, [k, 3+k, 6+k]] = W0_3[:, :, k]

    # ── BA_inv (3×3 per triangle) for H/K assembly ──────────────────────────
    if weights is not None:
        BMA_inv = np.einsum('n,nij,njk->nik', w, dM_np, M_inv)  # (N, 3, 3)
    else:
        BMA_inv = np.einsum('nij,njk->nik', dM_np, M_inv)        # (N, 3, 3)

    # ── Sparse G_0 = C₀ M_diag⁻¹ C₀ᵀ  (M_c × M_c, shared by all k) ────────
    tri_edges = defaultdict(list)
    for e in range(E_int):
        tri_edges[s1_arr[e]].append((e, +1))
        tri_edges[s2_arr[e]].append((e, -1))

    rows_g, cols_g, data_g = [], [], []

    # (a) Edge-edge block
    for n, elist in tri_edges.items():
        Mn = M_inv[n]   # (3, 3)
        for e1, sg1 in elist:
            for e2, sg2 in elist:
                val = sg1 * sg2 * float(q_arr[e1] @ Mn @ q_arr[e2])
                rows_g.append(e1); cols_g.append(e2); data_g.append(val)

    if n_pairs > 0:
        # Sort by triangle for batch processing
        ord_  = np.argsort(sv_arr, kind='stable')
        sv_s  = sv_arr[ord_]; v_s = v_arr[ord_]; a_s = a_arr[ord_]
        bnd   = np.where(np.diff(sv_s, prepend=-1, append=-1))[0]

        # (b) Angle-angle block
        for i in range(len(bnd) - 1):
            sl  = slice(bnd[i], bnd[i+1])
            n   = sv_s[bnd[i]]
            a_g = a_s[sl]; v_g = v_s[sl]
            G_vals = a_g @ M_inv[n] @ a_g.T    # (d, d)
            ri = E_int + v_g; ci = E_int + v_g
            ri2, ci2 = np.meshgrid(ri, ci, indexing='ij')
            rows_g.extend(ri2.ravel()); cols_g.extend(ci2.ravel())
            data_g.extend(G_vals.ravel())

        # (c) Edge-angle cross block
        tri_angle = defaultdict(list)
        for j in range(n_pairs):
            tri_angle[sv_arr[j]].append(j)

        for n, jlist in tri_angle.items():
            if n not in tri_edges:
                continue
            Mn = M_inv[n]
            for (e, sg) in tri_edges[n]:
                for j in jlist:
                    val = sg * float(q_arr[e] @ Mn @ a_arr[j])
                    rows_g.append(e); cols_g.append(E_int + v_arr[j]); data_g.append(val)
                    rows_g.append(E_int + v_arr[j]); cols_g.append(e); data_g.append(val)

    G_0 = sp.coo_matrix((data_g, (rows_g, cols_g)), shape=(M_c, M_c)).tocsc()
    reg  = 1e-12 * max(abs(v) for v in data_g) if data_g else 1e-12
    G_0  = G_0 + reg * sp.eye(M_c, format='csc')
    G_lu = spla.factorized(G_0)

    # ── H_0, K_0  (M_c × 3) — shared by all loading modes ───────────────────
    # H_0[e, loc'] = Σ_loc q[e,loc] * (M_inv[s1,loc,loc'] - M_inv[s2,loc,loc'])
    # K_0[e, loc'] = Σ_loc q[e,loc] * (BMA_inv[s1,loc,loc'] - BMA_inv[s2,loc,loc'])
    # (q @ M_inv) form vectorised
    H0 = np.zeros((M_c, 3))
    K0 = np.zeros((M_c, 3))
    # edge part: Σ_l q[e,l] * M_inv[s1[e],l,m] → (E_int, 3)
    qM1  = np.einsum('el,elm->em', q_arr, M_inv[s1_arr])    # (E_int, 3)
    qM2  = np.einsum('el,elm->em', q_arr, M_inv[s2_arr])    # (E_int, 3)
    qBM1 = np.einsum('el,elm->em', q_arr, BMA_inv[s1_arr])  # (E_int, 3)
    qBM2 = np.einsum('el,elm->em', q_arr, BMA_inv[s2_arr])  # (E_int, 3)
    H0[:E_int] = qM1 - qM2
    K0[:E_int] = qBM1 - qBM2
    # angle part: scatter-add over pairs
    if n_pairs > 0:
        aM  = np.einsum('pl,plm->pm', a_arr, M_inv[sv_arr])    # (n_pairs, 3)
        aBM = np.einsum('pl,plm->pm', a_arr, BMA_inv[sv_arr])  # (n_pairs, 3)
        np.add.at(H0, E_int + v_arr, aM)
        np.add.at(K0, E_int + v_arr, aBM)

    # ── Y_0 = G_0⁻¹ H_0  (M_c × 3) ─────────────────────────────────────────
    Y0 = np.column_stack([G_lu(H0[:, j]) for j in range(3)])  # (M_c, 3)
    M_mat3 = (IminusS3 if weights is not None else N * IminusS3) + K0.T @ Y0  # (3, 3)

    # ── Per loading mode: compute r_k, solve, apply correction ───────────────
    Lambda = np.zeros((M_c, 3))
    for k in range(3):
        # r_k[e] = q · (W0[s1,k-slice] - W0[s2,k-slice])
        r_k = np.zeros(M_c)
        diff = W0_3[s1_arr, :, k] - W0_3[s2_arr, :, k]   # (E_int, 3)
        r_k[:E_int] = np.einsum('el,el->e', q_arr, diff)
        if n_pairs > 0:
            np.add.at(r_k, E_int + v_arr,
                      np.einsum('pl,pl->p', a_arr, W0_3[sv_arr, :, k]))
        lam0   = G_lu(r_k)                                     # M_c
        c_vec  = np.linalg.solve(M_mat3, K0.T @ lam0)         # 3
        Lambda[:, k] = lam0 - Y0 @ c_vec

    # ── W = W₀ − P⁻¹ (C₀ᵀ Λ) per loading mode ──────────────────────────────
    CtLam_3 = np.zeros((N, 3, 3))   # (N, loc, k)
    for k in range(3):
        Lk_e = Lambda[:E_int, k]    # (E_int,)
        # edge scatter: Σ_e q[e,loc] * Lk_e at s1, -at s2
        np.add.at(CtLam_3[:, :, k], s1_arr,  q_arr * Lk_e[:, None])
        np.add.at(CtLam_3[:, :, k], s2_arr, -q_arr * Lk_e[:, None])
        if n_pairs > 0:
            Lk_a = Lambda[E_int + v_arr, k]   # (n_pairs,)
            np.add.at(CtLam_3[:, :, k], sv_arr, a_arr * Lk_a[:, None])

    # Apply P⁻¹ = (A - B)⁻¹  per loading mode k (3×3 systems)
    AinvCt  = np.einsum('nij,njk->nik', M_inv, CtLam_3)         # (N, 3, 3)
    if weights is not None:
        gs3 = np.einsum('n,nij,njk->ik', w, dM_np, AinvCt)     # (3, 3)
        PinvCt = AinvCt + np.einsum('nij,jk->nik', M_inv,
                                     np.linalg.solve(IminusS3, gs3))
    else:
        gs3 = np.einsum('nij,njk->ik', dM_np, AinvCt)          # (3, 3)
        PinvCt = AinvCt + np.einsum('nij,jk->nik', M_inv,
                                     np.linalg.solve(IminusS3, gs3)) / N

    # Reconstruct W in (N, 9) layout
    dW = np.zeros((N, 9))
    for k in range(3):
        dW[:, [k, 3+k, 6+k]] = PinvCt[:, :, k]

    return W0 - dW


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
