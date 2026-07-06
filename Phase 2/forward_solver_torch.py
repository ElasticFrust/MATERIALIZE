"""
Differentiable forward solver for 2D elastic spring networks.

Implements the D2C (Disc-to-Continuum) homogenisation of Grossman & Boudaoud
(PRR 2026, arXiv:2309.07844). Each triangle s has a bare elastic tensor A(s);
the per-triangle strain-concentration field W(s) (metric response δg(s)=W(s)Δḡ)
is solved for, then homogenised into the effective tensor C_eff and ν, E.

Two solve methods (selected by forward(method=...)):

  'intrinsic'  (default) — the configuration-free metric solve that reproduces
      the PBC simulation. Minimises the spring energy over the per-triangle
      metric field subject to three constraints: edge compatibility, the
      curvature / incompatibility constraint (zero discrete Gaussian curvature),
      and the AREA-weighted normalisation Σ_s S_s δg(s)=0. Eliminating the
      normalisation multiplier χ gives the area-weighted (A−B) Woodbury with
      δA(s)=A(s)−[A]·S_s/[S_s] (see _woodbury_solve_aw). For ≤ INTRINSIC_DENSE_MAX
      triangles this runs as a DIFFERENTIABLE dense torch solve; larger meshes
      fall back to the NumPy sparse saddle (_intrinsic_solve_W, forward-only).

  'woodbury'   (legacy) — the original G&B mean-field (A−B)W=−δA (Eq. 19-20),
      optionally projected onto edge / vertex-angle compatibility via the KKT
      step (_woodbury_solve, _woodbury_kkt_sparse_combined).

Pipeline: edge vectors v[n,i] → bare tensor A(n) (5 components) → solve W(n)
(one of the methods above) → 4-index contraction (_compute_actual_elastic_tensor)
→ homogenise to C_eff and extract ν, E.

Note on representations: the bare tensor's 9×9 block form M (kron(M, I₃),
_batch_to_9x9) is the ASYMMETRIC mixed-Voigt matrix used by the (A−B) Woodbury
and the contraction. The intrinsic energy solve instead uses the SYMMETRIC 3×3
metric Hessian A3(s)=Σ_e (k_e/4ℓ_e²) q_e q_eᵀ — these are NOT the same matrix
(sym(M) ≠ A3), and using one where the other is required gives wrong physics.
"""

import numpy as np
import torch
import torch.nn as nn

# Max #triangles for the differentiable dense intrinsic Woodbury path; larger meshes
# fall back to the (non-differentiable) sparse saddle solve.
INTRINSIC_DENSE_MAX = 600


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

        # Build the intrinsic-solver constraint operators (edge, curvature, area-mean)
        self._build_intrinsic_constraints(sim_np)

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

    def _build_intrinsic_constraints(self, simplices):
        """Precompute the geometry-only operators for the intrinsic (explicit-multiplier)
        solver: edge compatibility J, curvature/incompatibility C (zero discrete Gaussian
        curvature, tensor convention — NO /2), and the area-weighted normalisation M_S.

        All act on the per-triangle metric field δg ∈ ℝ^{3N} ([dg11,dg12,dg22] per triangle).
        Built convention-robustly from edge vectors + node membership, so signs/periodicity
        follow self.edge_vecs (correct whenever the geometry is set up consistently).
        """
        import scipy.sparse as sp
        from collections import defaultdict
        N = len(simplices)
        ev = self.edge_vecs.numpy()                 # (N,3,2)
        edges = self.edges.numpy()                  # (N,3,2) node-index pairs

        # q per edge for the per-triangle metric Hessian A3 = Σ_e (k_e/4l_e²) q_e q_eᵀ
        vx, vy = ev[:, :, 0], ev[:, :, 1]
        self._q_geom = np.stack([vx ** 2, 2 * vx * vy, vy ** 2], axis=2)   # (N,3,3) [edge,comp]

        # (1) edge operator from the sparse KKT arrays (periodic-correct when provided)
        if self.kkt_arrays is not None:
            s1, s2, q = self.kkt_arrays
            r, c, d = [], [], []
            for e in range(len(s1)):
                for loc in range(3):
                    r += [e, e]; c += [3*int(s1[e])+loc, 3*int(s2[e])+loc]
                    d += [q[e, loc], -q[e, loc]]
            self._J_edge_sp = sp.csr_matrix((d, (r, c)), shape=(len(s1), 3*N))
        else:
            self._J_edge_sp = sp.csr_matrix((0, 3*N))

        # (2) curvature operator: per interior vertex, Σ_{s∋v} ∂θ_v^s/∂g · δg(s) = 0
        ecount = defaultdict(int)
        for s in range(N):
            for i in range(3):
                a, b = int(edges[s, i, 0]), int(edges[s, i, 1])
                ecount[(min(a, b), max(a, b))] += 1
        boundary = {v for (a, b), cnt in ecount.items() if cnt == 1 for v in (a, b)}
        r, c, d = [], [], []
        vint = {}; ni = 0
        for s in range(N):
            for v in (int(x) for x in simplices[s]):
                if v in boundary:
                    continue
                vecs = []
                for i in range(3):
                    a, b = int(edges[s, i, 0]), int(edges[s, i, 1])
                    if v == a:
                        vecs.append(-ev[s, i])
                    elif v == b:
                        vecs.append(ev[s, i])
                if len(vecs) != 2:
                    continue
                ag = _angle_gradient_vec(vecs[0], vecs[1])     # ∂θ/∂[g11,g12,g22], NO /2
                if v not in vint:
                    vint[v] = ni; ni += 1
                for loc in range(3):
                    r.append(vint[v]); c.append(3*s+loc); d.append(ag[loc])
        self._C_curv_sp = sp.csr_matrix((d, (r, c)), shape=(ni, 3*N))

        # (3) area-weighted normalisation  Σ_s S_s δg(s)_{μν} = 0
        w = self.area_weights.numpy()
        r, c, d = [], [], []
        for mu in range(3):
            for s in range(N):
                r.append(mu); c.append(3*s+mu); d.append(float(w[s]))
        self._M_S_sp = sp.csr_matrix((d, (r, c)), shape=(3, 3*N))

    def _assemble_dense_J(self, use_kkt, use_angle):
        """Dense (M_c × 3N) constraint matrix for the differentiable intrinsic Woodbury:
        edge and/or curvature rows on the per-triangle metric field [dg11,dg12,dg22]."""
        import scipy.sparse as sp
        blocks = []
        if use_kkt and self._J_edge_sp.shape[0] > 0:
            blocks.append(self._J_edge_sp)
        if use_angle and self._C_curv_sp.shape[0] > 0:
            blocks.append(self._C_curv_sp)
        if not blocks:
            return None
        return torch.as_tensor(sp.vstack(blocks).toarray(), dtype=torch.float64)

    def forward(self, rigidities, rest_lengths=None, method='intrinsic',
                area_weighted=None, use_kkt=None, use_angle_kkt=None,
                physical_units=False):
        """Run the forward pipeline.

        Args:
            rigidities:     (N, 3) tensor
            rest_lengths:   (N, 3) tensor or None
            method:         'intrinsic' (default) — the explicit-multiplier, loading-
                            independent metric solve (block-diagonal A + edge + curvature
                            + area-weighted normalisation; reproduces the PBC simulation).
                            'woodbury' — the original G&B (A−B) mean-field Woodbury path.
            area_weighted:  None → default per method (intrinsic: True, woodbury: False).
                            S_s-weighted normalisation Σ_s S_s δg(s)=0 (the correct,
                            compatibility-implied condition); False → arithmetic Σ δg(n)=0.
            use_kkt:        None → default True. Edge-length compatibility constraint.
            use_angle_kkt:  None → default per method (intrinsic: True, woodbury: False).
                            Vertex-angle / curvature (zero discrete Gaussian curvature)
                            constraint. In the intrinsic path this is the robust sparse
                            curvature operator; in the woodbury path it is the (fragile)
                            angle-Gram correction.

        Toggles can be turned off individually; the intrinsic path has all three ON by
        default. The intrinsic solve uses the area-weighted χ-elimination δA=A_s−[A]·S_s/[S_s]
        (sub-choice 2): for ≤ INTRINSIC_DENSE_MAX triangles it runs a DIFFERENTIABLE torch
        Woodbury (edge+curvature KKT); larger meshes fall back to the NumPy sparse saddle
        (forward-only). Both reproduce the simulation; only the dense path carries gradients.

        physical_units: None/False (default) → elastic_tensor & young in the internal
                        (bare-tensor /16) scale, unchanged from before. True → rescale the
                        homogenised tensor by 8·N/A_total to true physical (energy/virial)
                        units. ν (poisson) is scale-invariant and identical either way; only
                        E / elastic_tensor change. Exact for periodic meshes.

        Returns:
            dict: elastic_tensor (6,), poisson, young, per_triangle (N,6),
                  bare (N,5), W (N,9)
        """
        if method == 'intrinsic':
            if area_weighted is None: area_weighted = True
            if use_kkt is None: use_kkt = True
            if use_angle_kkt is None: use_angle_kkt = True
        elif method == 'woodbury':
            if area_weighted is None: area_weighted = False
            if use_kkt is None: use_kkt = True
            if use_angle_kkt is None: use_angle_kkt = False
        else:
            raise ValueError(f"unknown method {method!r} (expected 'intrinsic' or 'woodbury')")

        rigidities = rigidities.double()
        if rest_lengths is not None:
            rest_lengths = rest_lengths.double()

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

        if method == 'intrinsic':
            # Per-triangle SYMMETRIC metric Hessian  A(s) = Σ_e (k_e / 4ℓ_e²) q_e q_eᵀ,
            # with q_e = [vx², 2 vx vy, vy²] (self._q_geom). Built once and shared by both
            # the differentiable dense Woodbury and the sparse-saddle fallback.
            Ntri = bare.shape[0]
            q_geom = torch.as_tensor(self._q_geom, dtype=bare.dtype, device=bare.device)   # (N,3,3)
            edge_stiff = rigidities / (4.0 * torch.clamp(length2, min=1e-30))              # k_e/4ℓ_e² (N,3)
            A3 = torch.einsum('ne,nei,nej->nij', edge_stiff, q_geom, q_geom)               # (N,3,3)

            # χ-elimination weight  w_s = S_s/[S_s] (area-weighted) or 1/N (arithmetic).
            if area_weighted:
                w_eff = self.area_weights.to(dtype=bare.dtype, device=bare.device)
            else:
                w_eff = torch.full((Ntri,), 1.0 / Ntri, dtype=bare.dtype, device=bare.device)

            if Ntri <= INTRINSIC_DENSE_MAX:
                # differentiable area-weighted χ-elimination + dense edge/curvature KKT
                J3 = self._assemble_dense_J(use_kkt, use_angle_kkt)
                if J3 is not None:
                    J3 = J3.to(dtype=bare.dtype, device=bare.device)
                W = _woodbury_solve_aw(A3, w_eff, J3)
            else:
                # large-mesh fallback: explicit-multiplier sparse saddle (NumPy, non-diff)
                W_np = _intrinsic_solve_W(A3.detach().cpu().numpy(),
                                          self._J_edge_sp, self._C_curv_sp, self._M_S_sp,
                                          use_kkt, use_angle_kkt, area_weighted)
                W = torch.as_tensor(W_np, dtype=bare.dtype, device=bare.device)
        else:
            delta    = bare - mean_tensor
            A_blocks = _batch_to_9x9(bare)
            B_blocks = _batch_to_9x9(delta)
            dA_vecs  = _batch_to_9vec(delta)

            if use_kkt and self.kkt_arrays is not None:
                # All KKT paths (edge-only or edge+angle) use the combined solver.
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
        # Homogenised tensor = UNWEIGHTED mean of the per-triangle tensors. A(s) is the full
        # per-triangle Hessian (no area prefactor), so the total energy is an unweighted sum
        # and its Δg-Hessian is the unweighted sum → unweighted mean here. An area weight S_s
        # in THIS average double-counts area and biases ν on variable-area (disordered) meshes
        # (harmless only for equal-area crystals). NB: the area weight is still correct — and
        # kept — in the compatibility normalisation Σ_s S_s δg(s)=0 (the χ / mean constraint).
        C = actual.mean(dim=0)
        if physical_units:
            # Convert the internal (bare-tensor /16, per-triangle) elastic tensor to true
            # physical (energy/virial) units. Derivation: Σ_s bare_s = (1/8)·Σ_bonds on a torus
            # — the /16 in `bare` and the ×2 interior-edge double-count — and the average is 1/N
            # vs the physical 1/A_total, so C_phys = (8·N / A_total)·C. This is a pure scalar, so
            # ν is unchanged; only E / the elastic_tensor rescale. Exact for periodic meshes; on
            # open finite samples the boundary edges (counted once) make it approximate.
            e01, e02 = self.edge_vecs[:, 0], self.edge_vecs[:, 1]
            areas = 0.5 * torch.abs(e01[:, 0] * e02[:, 1] - e01[:, 1] * e02[:, 0])
            C = C * (8.0 * areas.shape[0] / areas.sum())

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
    """(N, 5) → (N, 9, 9) block matrices kron(M, I₃).

    M is the ASYMMETRIC mixed-Voigt representation (factor 2 on the columns):
    M=[[a0,2a1,a2],[a1,2a2,a3],[a2,2a3,a4]]. It is correct for the (A−B) Woodbury
    response and the 4-index contraction, but it is NOT the symmetric energy
    Hessian A3=Σ_e(k_e/4ℓ_e²)q_e q_eᵀ used by the intrinsic solve (sym(M) ≠ A3)."""
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


def _intrinsic_solve_W(A3_blocks, J_edge, C_curv, M_S,
                       use_kkt=True, use_angle=True, area_weighted=True):
    """Loading-independent strain-concentration W via the explicit-multiplier sparse saddle.

    Minimises ½(Δg+δg)ᵀ A (Δg+δg) over the per-triangle metric field with BLOCK-DIAGONAL A
    (true spring energy — no mean-field (A−B)), subject to the constraint stack
    C = [edge ; curvature ; area-weighted mean].  Stripping Δg gives, per triangle,
        A3(s) W3(s) + Cᵀ(multipliers) = −A3(s),   C W = 0,
    solved once with a 3-wide RHS as the regularised saddle
        [ A   Cᵀ ] [W]   [ -A ]
        [ C  -εI ] [Λ] = [  0 ].

    Args:
        A3_blocks: (N,3,3) symmetric per-triangle metric Hessian (numpy).
        J_edge, C_curv, M_S: scipy.sparse constraint matrices (·×3N).
        use_kkt/use_angle/area_weighted: include the edge/curvature/mean blocks.

    Returns:
        W (N,9) numpy, layout [3*loc+k] = ∂δg_loc/∂Δg_k (as _compute_actual_elastic_tensor expects).
    """
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    N = A3_blocks.shape[0]
    Hblk = sp.block_diag([A3_blocks[i] for i in range(N)], format='csc')
    cons = []
    if use_kkt and J_edge.shape[0] > 0:
        cons.append(J_edge)
    if use_angle and C_curv.shape[0] > 0:
        cons.append(C_curv)
    if area_weighted and M_S.shape[0] > 0:
        cons.append(M_S)

    rhs_top = -A3_blocks.reshape(3 * N, 3)           # block s = -A3(s)
    if cons:
        C = sp.vstack(cons).tocsc()
        nC = C.shape[0]
        eps = 1e-10 * float(np.abs(A3_blocks).max() or 1.0)
        KKT = sp.bmat([[Hblk, C.T], [C, -eps * sp.eye(nC)]], format='csc')
        rhs = np.vstack([rhs_top, np.zeros((nC, 3))])
        x = spla.splu(KKT).solve(rhs)                # (3N+nC, 3), 3 RHS columns
        W3 = np.asarray(x[:3 * N, :])
    else:
        # unconstrained: A3 W3 = -A3 ⇒ W3 = -I per triangle
        W3 = np.tile(-np.eye(3), (N, 1))             # (3N,3) stacked -I blocks
    return W3.reshape(N, 9)


def _woodbury_solve_aw(A3, w, J3=None):
    """Differentiable area-weighted χ-elimination (sub-choice 2), in 3×3 metric space.

    Solves the (A − B) system with the CORRECT area weighting
        A3(s) W(s) − w_s Σ_s' δA(s') W(s') = −δA(s),
        δA(s) = A3(s) − w_s·[A3]   (area-scaled, [A3]=Σ_s A3(s)),   B[s,s']=w_s δA(s'),
    then projects onto ker J3 (edge + curvature) via the KKT Schur complement.  The 3
    loading modes are the columns (they decouple).  Pure torch ⇒ differentiable; reduces to
    the unweighted G&B solve when w = 1/N.  A3 is the SYMMETRIC per-triangle metric Hessian
    A3(s)=Σ_e (k_e/4l_e²) q_e q_eᵀ (the spring energy), so W feeds _compute_actual_elastic_tensor.

    Closed form:  y=A_inv δA, Vy=Σ_s δA(s)y(s) [unweighted], Sw=Σ_s w_s δA(s)A_inv(s),
    Zc=(I−Sw)⁻¹Vy, W0(s)=−(y(s)+w_s A_inv(s) Zc).

    Args: A3 (N,3,3); w (N,) area weights summing to 1; J3 (M_c, 3N) or None.
    Returns: W (N,9), layout [3*loc+k] = ∂δg_loc/∂Δg_k.
    """
    N = A3.shape[0]
    eps = 1e-12 * A3.abs().max()
    I3 = torch.eye(3, dtype=A3.dtype, device=A3.device)
    dA = A3 - w.view(N, 1, 1) * A3.sum(0)                          # area-scaled deviation (N,3,3)
    A_inv = torch.linalg.inv(A3 + eps * I3)                        # (N,3,3)
    Y = torch.einsum('nij,njk->nik', A_inv, dA)                    # (N,3,3)
    Vy = torch.einsum('nij,njk->ik', dA, Y)                        # (3,3) UNWEIGHTED
    Sw = torch.einsum('n,nij,njk->ik', w, dA, A_inv)              # (3,3)
    Zc = torch.linalg.solve(I3 - Sw, Vy)                          # (3,3)
    W0 = -(Y + w.view(N, 1, 1) * torch.einsum('nij,jk->nik', A_inv, Zc))   # (N,3,3)

    if J3 is None or J3.shape[0] == 0:
        return W0.reshape(N, 9)

    M_c = J3.shape[0]
    Jt = J3.T.reshape(N, 3, M_c)                                   # (N,3,M_c)
    yJ = torch.einsum('nij,njm->nim', A_inv, Jt)                   # (N,3,M_c)
    VyJ = torch.einsum('nij,njm->im', dA, yJ)                      # (3,M_c) UNWEIGHTED
    zJ = torch.linalg.solve(I3 - Sw, VyJ)                          # (3,M_c)
    PinvJt = (yJ + w.view(N, 1, 1) * torch.einsum('nij,jm->nim', A_inv, zJ)).reshape(3*N, M_c)
    G = J3 @ PinvJt                                                # (M_c,M_c)
    r = J3 @ W0.reshape(3*N, 3)                                    # (M_c,3)
    Lam = torch.linalg.lstsq(G, r).solution                       # lstsq tolerates redundant rows
    W = (W0.reshape(3*N, 3) - PinvJt @ Lam).reshape(N, 3, 3)
    return W.reshape(N, 9)


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
    # Unweighted mean-field is just the area-weighted solve with w_n = 1/N (the constant
    # commutes through the linear solve), so route both through one weighted code path.
    N = A_blocks.shape[0]
    if weights is None:
        weights = torch.full((N,), 1.0 / N, dtype=A_blocks.dtype, device=A_blocks.device)
    w = weights                                                       # (N,)
    I9 = torch.eye(9, dtype=A_blocks.dtype, device=A_blocks.device)
    A_inv = torch.linalg.inv(A_blocks + 1e-14 * A_blocks.abs().max() * I9)   # (N, 9, 9)

    y       = torch.einsum('nij,nj->ni',    A_inv, dA_vecs)          # (N, 9)
    Vy      = torch.einsum('n,nij,nj->i',   w, B_blocks, y)          # (9,)
    S       = torch.einsum('n,nij,njk->ik', w, B_blocks, A_inv)      # (9, 9)
    IminusS = I9 - S
    z       = torch.linalg.solve(IminusS, Vy)                        # (9,)
    W0      = -(y + torch.einsum('nij,j->ni', A_inv, z))             # (N, 9)

    if J is None or J.shape[0] == 0:
        return W0

    # KKT correction: project W₀ onto ker J in the P=(A−B) metric via P⁻¹ applied to the
    # columns of Jᵀ (same Woodbury identity, reshaped to (N, 9, M_c) with M_c = 3·E_int).
    M_c    = J.shape[0]
    Jt_b   = J.T.reshape(N, 9, M_c)                                  # (N, 9, M_c)
    y_J    = torch.einsum('nij,njm->nim',  A_inv, Jt_b)             # (N, 9, M_c)
    Vy_J   = torch.einsum('n,nij,njm->im', w, B_blocks, y_J)        # (9, M_c)
    z_J    = torch.linalg.solve(IminusS, Vy_J)                      # (9, M_c)
    PinvJt = (y_J + torch.einsum('nij,jm->nim', A_inv, z_J)).reshape(9 * N, M_c)
    G      = J @ PinvJt                                             # (M_c, M_c)
    Lambda = torch.linalg.solve(G, J @ W0.reshape(-1))             # (M_c,)
    return (W0.reshape(-1) - PinvJt @ Lambda).reshape(N, 9)


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
    qBM1 = np.einsum('el,eml->em', q_arr, BMA_inv[s1_arr])  # (E_int, 3)
    qBM2 = np.einsum('el,eml->em', q_arr, BMA_inv[s2_arr])  # (E_int, 3)
    H0[:E_int] = qM1 - qM2
    K0[:E_int] = qBM1 - qBM2
    # angle part: scatter-add over pairs
    if n_pairs > 0:
        aM  = np.einsum('pl,plm->pm', a_arr, M_inv[sv_arr])    # (n_pairs, 3)
        aBM = np.einsum('pl,pml->pm', a_arr, BMA_inv[sv_arr])  # (n_pairs, 3)
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
