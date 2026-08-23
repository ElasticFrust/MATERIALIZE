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
      triangles this runs as a DIFFERENTIABLE dense torch solve; larger meshes use
      the sparse saddle, which is ALSO differentiable when gradients are requested
      (_IntrinsicSparseWFn: NumPy forward = _intrinsic_solve_W, analytic adjoint
      backward reusing the KKT factorisation) and falls back to a forward-only
      NumPy solve otherwise.

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

import warnings

import numpy as np
import torch
import torch.nn as nn

# Max #triangles for the dense intrinsic Woodbury path; larger meshes use the sparse
# saddle solve (differentiable via _IntrinsicSparseWFn's adjoint when grad is needed,
# forward-only NumPy otherwise). Both size paths carry gradients.
INTRINSIC_DENSE_MAX = 600

# audit B-1, stage 1: LOOSE trigger on the orthogonality of the intrinsic KKT residual,
# |Gᵀ(G Lam - r)| / (|G| max(|G Lam - r|, |r|)) -- zero iff Lam is a least-squares solution, and
# valid whether or not the constraint set is consistent (an inconsistent one has an irreducibly
# nonzero residual, so testing the RAW residual would fire on healthy solves -- it did).
# Its healthy floor tracks cond(G) and is mesh-dependent: 3e-15 on a clean regular lattice, up to
# ~4e-6 on designed meshes across the gate suite. 1e-3 sits ~2.5 orders above that and ~2.5 below
# the one measured failure (0.46). Only a TRIGGER -- stage 2 decides.
_B1_KKT_RTOL = 1e-3

# audit B-1, stage 2: how much the SVD re-solve must move the CORRECTION TERM PinvJt·Lam, relative
# to |W0|, before it counts as a repair rather than ill-conditioning. Measured against W and not
# against Lam because a large relative move of a near-zero Lam has no effect on the answer. The
# observed failure moved W by ~46 %; healthy re-solves agree to ~1e-14.
_B1_LAM_RTOL = 1e-6


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
        edge and/or curvature rows on the per-triangle metric field [dg11,dg12,dg22].

        CACHED per (use_kkt, use_angle) — audit B-5. The constraint Jacobian is built purely from
        `self._J_edge_sp` / `self._C_curv_sp`, both fixed for the lifetime of the solver, so this is
        loop-invariant; it was previously re-assembled (scipy `vstack` + `toarray`) on EVERY
        `forward()`, i.e. up to n_iter × n_restarts times per design. Consumers treat the result as
        read-only (`_woodbury_solve_aw` only reads `J3`), so handing back the same tensor is safe."""
        import scipy.sparse as sp
        cache = self.__dict__.setdefault('_dense_J_cache', {})
        ck = (bool(use_kkt), bool(use_angle))
        if ck in cache:
            return cache[ck]
        blocks = []
        if use_kkt and self._J_edge_sp.shape[0] > 0:
            blocks.append(self._J_edge_sp)
        if use_angle and self._C_curv_sp.shape[0] > 0:
            blocks.append(self._C_curv_sp)
        if not blocks:
            cache[ck] = None
            return None
        cache[ck] = torch.as_tensor(sp.vstack(blocks).toarray(), dtype=torch.float64)
        return cache[ck]

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
        Woodbury (edge+curvature KKT); larger meshes use the sparse saddle, which stays
        DIFFERENTIABLE when gradients are requested (adjoint backward reusing the KKT
        factorisation) and drops to a forward-only NumPy solve otherwise. Both size paths
        reproduce the simulation and carry gradients.

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
            elif torch.is_grad_enabled() and A3.requires_grad:
                # large mesh WITH gradients: differentiable sparse saddle via the adjoint
                # (forward is identical to _intrinsic_solve_W; backward reuses the factorisation)
                W = _IntrinsicSparseWFn.apply(A3, self._J_edge_sp, self._C_curv_sp, self._M_S_sp,
                                              use_kkt, use_angle_kkt, area_weighted)
            else:
                # large-mesh forward-only: explicit-multiplier sparse saddle (NumPy, unchanged)
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

        # ---- scalar reduction: DIRECTION-AVERAGED, with an isotropy measure beside it ----------
        # ν = ½(ν_xy + ν_yx), E = ½(Ex + Ey), from the compliance S = C⁻¹ in Voigt [xx,yy,xy].
        #
        # This used to be the closed form (C2·C3 − C1·C4)/(C0·C3 − C1²) etc., which returns **ν_yx
        # and E_y — the y direction alone**. Identical on an isotropic tensor, so ν=1/3 never saw
        # it, but on a plain orthotropic tensor it differs by Δν=0.081 / ΔE=21%, and on the ψ=0.6
        # crystal ν_xy=0.926 vs ν_yx=0.209 — the old value silently reported the smaller of the two
        # because it happened to be y. It also disagreed with the oracle
        # (`physical_homog._voigt_nuE`), so every solver-vs-sim ν/E comparison was mixing two
        # different quantities: measured gaps of 0.079 collapsed to 1.4e-12 once matched.
        # Audit A-10, changed 2026-08-15 with explicit approval (protected core).
        #
        # A scalar is only meaningful when the tensor is near-isotropic, so `anisotropy` travels
        # with it — 0 = isotropic, and a large value means READ ν(θ)/E(θ) INSTEAD, not this number.
        Cv = torch.stack([torch.stack([C[0], C[2], C[1]]),
                          torch.stack([C[2], C[5], C[4]]),
                          torch.stack([C[1], C[4], C[3]])])
        S = torch.linalg.inv(Cv)
        Ex, Ey = 1.0 / S[0, 0], 1.0 / S[1, 1]
        poisson = 0.5 * (-S[1, 0] * Ex - S[0, 1] * Ey)
        young = 0.5 * (Ex + Ey)
        dev = ((C[0] - C[5]) ** 2 + C[1] ** 2 + C[4] ** 2
               + (C[3] - (C[0] - C[2]) / 2) ** 2)                # deviation from isotropy
        anisotropy = dev / (C[0] ** 2 + C[5] ** 2 + C[2] ** 2 + C[3] ** 2 + 1e-12)

        return {
            'elastic_tensor': C,
            'poisson': poisson,
            'young': young,
            'anisotropy': anisotropy,       # 0 = isotropic; large ⇒ the scalars above are a poor summary
            'poisson_xy': -S[1, 0] * Ex,    # the two directional values the average is made of,
            'poisson_yx': -S[0, 1] * Ey,    # kept so the spread is always available
            'young_x': Ex,
            'young_y': Ey,
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


class _IntrinsicSparseWFn(torch.autograd.Function):
    """Differentiable large-mesh intrinsic solve (Ntri > INTRINSIC_DENSE_MAX).

    forward: identical to `_intrinsic_solve_W` — the scipy sparse KKT saddle
        [[H(A3), Cᵀ]; [C, −εI]] · [W; Λ] = [−A3_stacked; 0]  via `splu`.
    backward: the ADJOINT of that linear solve. The KKT matrix is symmetric, so the adjoint
        system uses the SAME factorisation (one extra triangular solve, ~O(N) assembly on top):
            KKT · λ = [∂L/∂W; 0],   then   ∂L/∂A3(s) = −sym( λ_W(s) + λ_W(s) · W(s)ᵀ ).
        A3(s) is the only k/l0-dependent input (the constraint operators J_edge/C_curv/M_S are
        geometry-only); torch then continues A3 → edge_stiff = k/(4ℓ²) → k, l0.
    This carries gradients for meshes above the dense cap at O(N) memory / sparse O(N^1.5) time —
    forward-only cost is unchanged; gradients add ~one reused-factorisation solve."""

    @staticmethod
    def forward(ctx, A3, J_edge, C_curv, M_S, use_kkt, use_angle, area_weighted):
        import scipy.sparse as sp
        import scipy.sparse.linalg as spla
        A3np = A3.detach().cpu().numpy()
        N = A3np.shape[0]
        Hblk = sp.block_diag([A3np[i] for i in range(N)], format='csc')
        cons = []
        if use_kkt and J_edge.shape[0] > 0:
            cons.append(J_edge)
        if use_angle and C_curv.shape[0] > 0:
            cons.append(C_curv)
        if area_weighted and M_S.shape[0] > 0:
            cons.append(M_S)
        rhs_top = -A3np.reshape(3 * N, 3)
        if cons:
            C = sp.vstack(cons).tocsc(); nC = C.shape[0]
            eps = 1e-10 * float(np.abs(A3np).max() or 1.0)
            KKT = sp.bmat([[Hblk, C.T], [C, -eps * sp.eye(nC)]], format='csc')
            rhs = np.vstack([rhs_top, np.zeros((nC, 3))])
            lu = spla.splu(KKT)
            x = lu.solve(rhs)
            W3 = np.asarray(x[:3 * N, :])
            ctx.cache = (lu, x, N, nC)
        else:
            W3 = np.tile(-np.eye(3), (N, 1))
            ctx.cache = (None, None, N, 0)
        return torch.as_tensor(W3.reshape(N, 9), dtype=A3.dtype, device=A3.device)

    @staticmethod
    def backward(ctx, grad_out):
        lu, x, N, nC = ctx.cache
        none6 = (None, None, None, None, None, None)
        if lu is None:                                    # W = −I, independent of A3
            g0 = torch.zeros((N, 3, 3), dtype=grad_out.dtype, device=grad_out.device)
            return (g0,) + none6
        gW = grad_out.detach().cpu().numpy().reshape(N, 3, 3).reshape(3 * N, 3)
        lam = lu.solve(np.vstack([gW, np.zeros((nC, 3))]))    # symmetric KKT → reuse factorisation
        lamW = lam[:3 * N, :].reshape(N, 3, 3)
        W3 = x[:3 * N, :].reshape(N, 3, 3)
        G = -(lamW + np.einsum('nij,nkj->nik', lamW, W3))     # ∂L/∂A3(s)
        G = 0.5 * (G + np.transpose(G, (0, 2, 1)))            # symmetrise (A3 = Σ_e k_e/4ℓ² q qᵀ)
        return (torch.as_tensor(G, dtype=grad_out.dtype, device=grad_out.device),) + none6


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
    # --- audit B-1 GUARD (2026-08-23) ------------------------------------------------------
    # G is SINGULAR BY CONSTRUCTION -- the constraint rows are redundant (rank 671/672,
    # cond ~3e16 on the regular lattice), which is exactly what the `lstsq` above is here to
    # tolerate. Measured once in ~700 solves, its pivoting-based CPU driver (gelsy) instead
    # returns a Lam that only PARTIALLY solves G Lam = r, leaving the KKT correction ~54 %
    # applied: W then violated J3 W = 0 by fourteen orders (6.1e-15 -> 5.4e-01) and C_eff came
    # out ~1 % over-compliant -- above the design tolerances and indistinguishable from a real
    # result. Full evidence: Phase 3/verifications/b1_dumps/B1_OVERNIGHT.md §4d.
    #
    # The test is ORTHOGONALITY, not the raw residual. Lam is a valid least-squares solution iff
    # its residual is orthogonal to range(G), i.e. Gᵀ(G Lam - r) = 0. That holds even when the
    # constraint set is INCONSISTENT (r outside range(G)), where |G Lam - r| is irreducibly
    # nonzero and J3 W != 0 legitimately -- so testing the raw residual would fire on healthy
    # solves (it did: `sanity.py` at 1.9e-01 and the hexagon gate at 4e-08, both correct).
    # Measured: the ratio below is 3e-15..4e-14 on healthy meshes spanning eta=0..0.45, while a
    # 54 %-applied Lam gives ~0.46 -- twelve orders of separation.
    # Costs two mat-vecs when healthy (~1 MFLOP against the einsums above); Lam is untouched then.
    # TWO STAGES, because the healthy floor of this ratio is MESH-DEPENDENT: it tracks cond(G),
    # measured 3e-15 on a clean regular lattice but up to ~4e-6 on designed meshes across the gate
    # suite, where re-solving barely improves it (1.04e-08 -> 8.81e-09) because that IS the best
    # achievable. So no fixed tight tolerance can separate "ill-conditioned" from "mis-solved".
    #   stage 1 -- a LOOSE trigger (1e-3), ~2.5 orders above the worst healthy value observed and
    #             ~2.5 below the one measured failure (0.46). Cheap: two mat-vecs.
    #   stage 2 -- only if triggered, re-solve with the SVD driver and ask the DEFINITIVE question:
    #             does it actually change Lam? Ill-conditioning gives the same answer twice; a
    #             mis-solve does not (the observed failure was 46 % off). A false trigger is then
    #             harmless -- one extra solve and no warning -- so this can never abort a run.
    res = (G @ Lam - r).detach()
    Gd = G.detach()
    scale = float(Gd.abs().max()) * max(float(res.abs().max()), float(r.detach().abs().max()))
    if scale > 0.0 and float((Gd.T @ res).abs().max()) > _B1_KKT_RTOL * scale:
        Lam_alt = torch.linalg.lstsq(G, r, driver='gelsd').solution   # SVD: nothing to mis-pivot
        # Measure the impact on W, NOT on Lam. W = W0 - PinvJt Lam, so what matters is how much
        # the CORRECTION TERM moves relative to W0 -- a large relative move of a negligible Lam
        # changes nothing. (Testing Lam directly reported 1.9e+06 on a healthy solve where |Lam|
        # was ~0, i.e. divide-by-almost-zero; the observed real failure moves W by ~46 %.)
        # Normalise by max(|W0|, 1): W is a correction TO THE IDENTITY -- C is built from (1+W) --
        # so its natural scale is 1, NOT its own magnitude. Dividing by |W0| alone inflates the
        # ratio without bound wherever W ~ 0, which is exactly the uniform-k regular lattice
        # (W == 0 identically there); that produced spurious 1.9e+06 and 3.9e+00 "repairs".
        dW = (PinvJt @ (Lam_alt - Lam).detach()).abs().max()
        impact = float(dW) / max(float(W0.detach().abs().max()), 1.0)
        if impact > _B1_LAM_RTOL:
            warnings.warn(f"audit B-1: KKT correction was mis-solved and has been repaired "
                          f"(W changed by {impact:.3e} relative)", RuntimeWarning)
            Lam = Lam_alt                                    # SVD solution is the trustworthy one
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


# `_woodbury_kkt_sparse` (edge-length constraints ONLY) was REMOVED 2026-08-18 - audit B-5.
# It had ZERO call sites repo-wide (re-verified by import graph and by name-string search);
# `_woodbury_kkt_sparse_combined` below is the live routine and carries edge-length AND
# vertex-angle (curvature) constraints, which is the verified default - dropping C2 is the
# classic single-site mean-field over-compliance error (CLAUDE.md section 3). Recover from
# git history if an edge-only variant is ever wanted again.



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
    """Per-triangle effective elastic tensor  C = (1+W)ᵀ A (1+W)  by 4-index contraction:

        C_{ijkl} = T_{mnij} A_{mnpq} T_{pqkl},        T = Id + W

    with every contraction over an index PAIR running over both of its indices.

    A is the fully symmetric bare tensor, A_{ijkl} = a[(i+j)+(k+l)] — 5 independent components,
    and being fully symmetric it is insensitive to the index bookkeeping below.  W is not: it maps
    symmetric 2-tensors to symmetric 2-tensors, δg_ij = W_{ijkl} Δg_kl summed over BOTH k and l, so
    lifting the vec3 operator `Ws` (layout w[3·loc + k] = ∂δg_loc/∂Δg_k, basis [xx,xy,yy], NO
    factor-2 on shear — see `_intrinsic_solve_W`) into 4 indices carries a ½ on a shear INPUT pair:

        W_xxxx Δg_xx + 2·W_xxxy Δg_xy + W_xxyy Δg_yy  =  W3[0,0] Δg_xx + W3[0,1] Δg_xy + W3[0,2] Δg_yy

    and, for the same reason, the identity inside (1+W) is the SYMMETRISED delta ½(δ_ik δ_jl +
    δ_il δ_jk), not δ_ik δ_jl.  Dropping either double-counts the shear input and over-stiffens
    C_xyxy on any network with W≠0 (2026-08 fix; the regular lattice has W≡0 identically, so no
    crystal-anchored check can see it — `test_forward_solver.py` [7] is the gate that does).

    Args:
        bare_tensors: (N, 5) — [a_xxxx, a_xxxy, a_xxyy, a_xyyy, a_yyyy]
        Ws:           (N, 9) — vec3 strain-concentration operator, entry [3·loc + k]

    Returns:
        (N, 6) — [C₁₁₁₁, C₁₁₁₂, C₁₁₂₂, C₂₁₁₂, C₂₁₂₂, C₂₂₂₂]
    """
    N = bare_tensors.shape[0]
    a, w = bare_tensors, Ws
    dev = a.device

    # index maps over the 2×2×2×2 slots (tiny, geometry-independent)
    e = torch.arange(2, device=dev)
    I, J, K, L = torch.meshgrid(e, e, e, e, indexing='ij')
    a_idx = ((I + J) + (K + L)).reshape(-1)              # → one of the 5 bare components
    w_idx = (3 * (I + J) + (K + L)).reshape(-1)          # → one of the 9 vec3 entries
    half = 1.0 - 0.5 * (K != L).to(a.dtype)              # ½ on a shear INPUT pair (k≠l)

    A4 = a[:, a_idx].reshape(N, 2, 2, 2, 2)
    W4 = w[:, w_idx].reshape(N, 2, 2, 2, 2) * half
    I2 = torch.eye(2, dtype=a.dtype, device=dev)
    Id = 0.5 * (torch.einsum('ik,jl->ijkl', I2, I2) + torch.einsum('il,jk->ijkl', I2, I2))

    T = Id + W4
    C = torch.einsum('tmnij,tmnpq,tpqkl->tijkl', T, A4, T)
    return torch.stack([
        C[:, 0, 0, 0, 0], C[:, 0, 0, 0, 1], C[:, 0, 0, 1, 1],
        C[:, 1, 0, 0, 1], C[:, 1, 0, 1, 1], C[:, 1, 1, 1, 1],
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
