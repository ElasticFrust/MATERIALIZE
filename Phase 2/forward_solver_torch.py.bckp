"""
Differentiable forward solver for 2D elastic spring networks.

This is a PyTorch port of the Woodbury-optimized NumPy solver from
Disc_2_Cont_optimized.py.  Every operation is differentiable so that
gradients flow from macroscopic elastic properties (Poisson's ratio,
Young's modulus) back to microscopic design variables (per-edge spring
rigidities and rest lengths).

===========================================================================
PHYSICS BACKGROUND  (concise)
===========================================================================

We model a 2D material as a Delaunay triangulation of randomly perturbed
hexagonal lattice points.  Each triangle has 3 edges, each edge a linear
spring with rigidity k and rest length l₀.

The goal: given the network geometry and spring parameters, compute the
*effective* (homogenized) elastic tensor of the bulk material, and from it
the macroscopic Poisson's ratio ν and Young's modulus E.

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

    This is the elastic tensor *if every triangle were identical*.

Step 3 — Deviation from mean
    A₀   = mean(A, over triangles)
    δA[n] = A[n] − A₀

Step 4 — Build 9×9 block matrices
    Each 5-component tensor maps to a 3×3-block matrix (each block is
    scalar × I₃), i.e. kron(M, I₃) where:

        M = [[a₀, 2a₁, a₂],
             [a₁, 2a₂, a₃],
             [a₂, 2a₃, a₄]]

    We build:
        A_blocks[n] = kron(M(A[n]), I₃)     — (N, 9, 9)
        B_blocks[n] = kron(M(δA[n]), I₃)    — (N, 9, 9)
        dA_vecs[n]  = flatten of δA mapping  — (N, 9)

Step 5 — Woodbury solve  (this is the key efficiency trick)
    We need to solve  (A_full − B_full) · W = −dA  where:
        A_full = block_diag(A₁, …, Aₙ)        [9N × 9N, block-diagonal]
        B_full = (1/N) · U · V                  [rank-9 perturbation]
            U = [I₉; I₉; …; I₉]   (9N × 9)
            V = [B₁, B₂, …, Bₙ]   (9 × 9N)

    Woodbury identity:
        (A − UV)⁻¹ = A⁻¹ + A⁻¹ U (I − V A⁻¹ U)⁻¹ V A⁻¹

    Concretely:
        1. Invert each 9×9 block:  A_inv[n] = inv(A_blocks[n])
        2. y[n] = A_inv[n] · dA_vecs[n]
        3. Vy   = Σₙ B_blocks[n] · y[n]                     (9-vector)
        4. S    = (1/N) Σₙ B_blocks[n] · A_inv[n]           (9×9)
        5. Solve (I₉ − S) z = Vy                            (9×9 system)
        6. W[n] = −(y[n] + (1/N) A_inv[n] · z)

    Cost: O(N) instead of O(N³) for a direct sparse solve.

Step 6 — 4-index tensor contraction
    Build rank-4 tensors  A₄[n] and W₄[n]  (each 2×2×2×2) from the
    5- and 9-component vectors, then contract:

        C₄ = A₄ + A₄·W₄ + A₄·W₄ + A₄·W₄·W₄

    (the full expression has 4 terms involving einsum over internal indices)

Step 7 — Homogenize and extract material properties
    C_eff = mean(C₄) over all triangles  →  6 independent components

    ν = (C₂·C₃ − C₁·C₄) / (C₀·C₃ − C₁²)          (Poisson's ratio)
    E = (C₂²C₃ − 2C₁C₂C₄ + C₁²C₅ + C₀(C₄²−C₃C₅))
        / (C₁² − C₀C₃)                              (Young's modulus)

===========================================================================
DIFFERENTIABLE DESIGN VARIABLES
===========================================================================

    rigidities   (N, 3)  — spring constant per edge per triangle
    rest_lengths (N, 3)  — rest length per edge per triangle

Node positions and mesh topology (simplices, edges) are *fixed* — they
describe the geometry of the foam and are not optimized.

===========================================================================
"""

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
        """
        Args:
            positions:  numpy array or tensor, shape (M, 2)
            simplices:  numpy array or tensor, shape (N, 3), int
            edges:      numpy array or tensor, shape (N, 3, 2), int
        """
        super().__init__()

        # Store fixed geometry as non-parameter buffers (moved with .to())
        self.register_buffer(
            'positions', torch.as_tensor(positions, dtype=torch.float64)
        )
        self.register_buffer(
            'simplices', torch.as_tensor(simplices, dtype=torch.long)
        )
        self.register_buffer(
            'edges', torch.as_tensor(edges, dtype=torch.long)
        )

        # Pre-compute edge vectors (fixed — geometry doesn't change)
        node_a = self.edges[:, :, 0]  # (N, 3)
        node_b = self.edges[:, :, 1]  # (N, 3)
        self.register_buffer(
            'edge_vecs', self.positions[node_a] - self.positions[node_b]
        )  # (N, 3, 2)

        # Pre-compute actual edge lengths squared (used as default rest lengths)
        self.register_buffer(
            'actual_length2',
            (self.edge_vecs ** 2).sum(dim=2)  # (N, 3)
        )

    def forward(self, rigidities, rest_lengths=None):
        """Run the full 7-step forward pipeline.

        Args:
            rigidities:   (N, 3) tensor, requires_grad=True for optimization
            rest_lengths: (N, 3) tensor or None (defaults to actual edge lengths)

        Returns:
            dict with keys:
                'elastic_tensor' — (6,) effective elastic tensor
                'poisson'        — scalar Poisson's ratio
                'young'          — scalar Young's modulus
                'per_triangle'   — (N, 6) per-triangle actual elastic tensors
                'bare'           — (N, 5) per-triangle bare elastic tensors
                'W'              — (N, 9) Woodbury correction vectors
        """
        vx = self.edge_vecs[:, :, 0]  # (N, 3)
        vy = self.edge_vecs[:, :, 1]  # (N, 3)

        # --- Step 2: Bare elastic tensor ---
        if rest_lengths is not None:
            length2 = rest_lengths ** 2
        else:
            length2 = self.actual_length2

        factor = rigidities / length2 / 16.0  # (N, 3)

        bare = torch.stack([
            (factor * vx ** 4).sum(dim=1),              # a0
            (factor * vx ** 3 * vy).sum(dim=1),         # a1
            (factor * vx ** 2 * vy ** 2).sum(dim=1),    # a2
            (factor * vx * vy ** 3).sum(dim=1),         # a3
            (factor * vy ** 4).sum(dim=1),              # a4
        ], dim=1)  # (N, 5)

        # --- Step 3: Deviation from mean ---
        mean_tensor = bare.mean(dim=0)       # (5,)
        delta = bare - mean_tensor           # (N, 5)

        # --- Step 4: Build 9×9 block matrices ---
        A_blocks = _batch_to_9x9(bare)      # (N, 9, 9)
        B_blocks = _batch_to_9x9(delta)     # (N, 9, 9)
        dA_vecs = _batch_to_9vec(delta)      # (N, 9)

        # --- Step 5: Woodbury solve ---
        W = _woodbury_solve(A_blocks, B_blocks, dA_vecs)  # (N, 9)

        # --- Step 6: 4-index tensor contraction ---
        actual = _compute_actual_elastic_tensor(bare, W)  # (N, 6)

        # --- Step 7: Homogenize ---
        C = actual.mean(dim=0)  # (6,)

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
# Pure-function helpers (no state, all differentiable)
# ---------------------------------------------------------------------------

def _batch_to_9x9(vecs5):
    """(N, 5) tensor components → (N, 9, 9) block matrices.

    Block structure (each entry × I₃):
        [[a₀,  2a₁,  a₂ ],
         [a₁,  2a₂,  a₃ ],
         [a₂,  2a₃,  a₄ ]]
    """
    N = vecs5.shape[0]
    a0, a1, a2, a3, a4 = vecs5[:, 0], vecs5[:, 1], vecs5[:, 2], vecs5[:, 3], vecs5[:, 4]

    # Build the 3×3 scalar matrix M, then kron(M, I₃)
    # M[i,j] values for each triangle  → (N,)
    # Row 0: a0, 2*a1, a2
    # Row 1: a1, 2*a2, a3
    # Row 2: a2, 2*a3, a4
    M = torch.stack([
        torch.stack([a0,     2*a1, a2  ], dim=1),
        torch.stack([a1,     2*a2, a3  ], dim=1),
        torch.stack([a2,     2*a3, a4  ], dim=1),
    ], dim=1)  # (N, 3, 3)

    # kron(M, I₃) = M[i,j] * I₃  at block (i,j)
    I3 = torch.eye(3, dtype=vecs5.dtype, device=vecs5.device)
    mat = torch.einsum('nij,kl->nikjl', M, I3).reshape(N, 9, 9)
    return mat


def _batch_to_9vec(vecs5):
    """(N, 5) tensor components → (N, 9) vectors.

    Pattern: [a0, a1, a2,  a1, a2, a3,  a2, a3, a4]

    This is the column-major flattening of the 3×3 scalar matrix M
    (same M as in _batch_to_9x9), reading one element per I₃ block.
    """
    a0, a1, a2, a3, a4 = [vecs5[:, i] for i in range(5)]
    return torch.stack([a0, a1, a2, a1, a2, a3, a2, a3, a4], dim=1)


def _woodbury_solve(A_blocks, B_blocks, dA_vecs):
    """Solve  (A_full − B_full) W = −dA  via Woodbury decomposition.

    A_full = block_diag(A₁, …, Aₙ)           — block-diagonal, O(N) to invert
    B_full = (1/N) · U · V                     — rank-9 correction
        where U = [I₉; …; I₉] (9N×9),  V = [B₁, …, Bₙ] (9×9N)

    Woodbury: (A − UV)⁻¹ = A⁻¹ + A⁻¹ U (I − V A⁻¹ U)⁻¹ V A⁻¹

    Args:
        A_blocks: (N, 9, 9)
        B_blocks: (N, 9, 9)
        dA_vecs:  (N, 9)

    Returns:
        W: (N, 9)
    """
    N = A_blocks.shape[0]

    # 1. Regularize and invert each 9×9 block
    eps = 1e-14 * A_blocks.abs().max()
    I9 = torch.eye(9, dtype=A_blocks.dtype, device=A_blocks.device)
    A_reg = A_blocks + eps * I9.unsqueeze(0)
    A_inv = torch.linalg.inv(A_reg)  # (N, 9, 9)

    # 2. y[n] = A_inv[n] @ dA_vecs[n]
    y = torch.einsum('nij,nj->ni', A_inv, dA_vecs)  # (N, 9)

    # 3. Vy = Σₙ B[n] @ y[n]
    Vy = torch.einsum('nij,nj->i', B_blocks, y)  # (9,)

    # 4. S = (1/N) Σₙ B[n] @ A_inv[n]
    S = torch.einsum('nij,njk->ik', B_blocks, A_inv) / N  # (9, 9)

    # 5. Solve (I₉ − S) z = Vy
    z = torch.linalg.solve(I9 - S, Vy)  # (9,)

    # 6. W[n] = −(y[n] + (1/N) A_inv[n] @ z)
    correction = torch.einsum('nij,j->ni', A_inv, z) / N  # (N, 9)
    W = -(y + correction)

    return W


def _compute_actual_elastic_tensor(bare_tensors, Ws):
    """Compute the actual (effective) elastic tensor per triangle.

    Uses the 4-index tensor contraction:
        C = A + A·W + A·W + A·W·W

    Full expression (Einstein summation, repeated indices summed):
        C_{manb} = A_{manb}
                 + A_{minj} W_{iajb}      (S2)
                 + A_{aibj} W_{imjn}      (S3)
                 + A_{kilj} W_{iajb} W_{kmln}  (S4)

    Args:
        bare_tensors: (N, 5)
        Ws:           (N, 9)

    Returns:
        (N, 6) — the 6 independent components of the symmetric elastic tensor:
        [C₁₁₁₁, C₁₁₁₂, C₁₁₂₂, C₂₁₁₂, C₂₁₂₂, C₂₂₂₂]
    """
    N = bare_tensors.shape[0]
    a = bare_tensors
    w = Ws

    # Build A_mat (N, 4, 4) from 5 components
    # M[r,c] = A_{m,a,n,b} where r = (m-1)*2+(a-1), c = (n-1)*2+(b-1)
    A_mat = torch.zeros(N, 4, 4, dtype=a.dtype, device=a.device)
    A_mat[:, 0, 0] = a[:, 0]
    A_mat[:, 0, 1] = a[:, 1]
    A_mat[:, 0, 2] = a[:, 1]
    A_mat[:, 0, 3] = a[:, 2]
    A_mat[:, 1, 0] = a[:, 1]
    A_mat[:, 1, 1] = a[:, 2]
    A_mat[:, 1, 2] = a[:, 2]
    A_mat[:, 1, 3] = a[:, 3]
    A_mat[:, 2, 0] = a[:, 1]
    A_mat[:, 2, 1] = a[:, 2]
    A_mat[:, 2, 2] = a[:, 2]
    A_mat[:, 2, 3] = a[:, 3]
    A_mat[:, 3, 0] = a[:, 2]
    A_mat[:, 3, 1] = a[:, 3]
    A_mat[:, 3, 2] = a[:, 3]
    A_mat[:, 3, 3] = a[:, 4]

    # Build W_mat (N, 4, 4) from 9 components
    W_mat = torch.zeros(N, 4, 4, dtype=w.dtype, device=w.device)
    W_mat[:, 0, 0] = w[:, 0]
    W_mat[:, 0, 1] = w[:, 1]
    W_mat[:, 0, 2] = w[:, 3]
    W_mat[:, 0, 3] = w[:, 4]
    W_mat[:, 1, 0] = w[:, 1]
    W_mat[:, 1, 1] = w[:, 2]
    W_mat[:, 1, 2] = w[:, 4]
    W_mat[:, 1, 3] = w[:, 5]
    W_mat[:, 2, 0] = w[:, 3]
    W_mat[:, 2, 1] = w[:, 4]
    W_mat[:, 2, 2] = w[:, 6]
    W_mat[:, 2, 3] = w[:, 7]
    W_mat[:, 3, 0] = w[:, 4]
    W_mat[:, 3, 1] = w[:, 5]
    W_mat[:, 3, 2] = w[:, 7]
    W_mat[:, 3, 3] = w[:, 8]

    # Reshape to rank-4 tensors: (N, 2, 2, 2, 2)
    A4 = A_mat.reshape(N, 2, 2, 2, 2)
    W4 = W_mat.reshape(N, 2, 2, 2, 2)

    # 4-term contraction
    S2 = torch.einsum('tminj,tiajb->tmanb', A4, W4)
    S3 = torch.einsum('taibj,timjn->tmanb', A4, W4)
    S4 = torch.einsum('tkilj,tiajb,tkmln->tmanb', A4, W4, W4)

    C4 = A4 + S2 + S3 + S4
    C_mat = C4.reshape(N, 4, 4)

    # Extract 6 independent components
    actual = torch.stack([
        C_mat[:, 0, 0],  # C₁₁₁₁
        C_mat[:, 0, 1],  # C₁₁₁₂
        C_mat[:, 1, 1],  # C₁₁₂₂
        C_mat[:, 1, 2],  # C₂₁₁₂
        C_mat[:, 2, 3],  # C₂₁₂₂
        C_mat[:, 3, 3],  # C₂₂₂₂
    ], dim=1)

    return actual


# ---------------------------------------------------------------------------
# Convenience: build solver from a scipy Delaunay triangulation
# ---------------------------------------------------------------------------

def from_triangulation(triangulation):
    """Create an ElasticSolver from a scipy.spatial.Delaunay object.

    This extracts positions, simplices, and computes edges, then builds the
    solver.  The triangulation must already have been filtered to "good"
    simplices (as generate_foam_points does).

    Args:
        triangulation: scipy.spatial.Delaunay with .points, .simplices

    Returns:
        (solver, default_rigidities, default_rest_lengths)
    """
    import numpy as np

    points = triangulation.points
    simplices = triangulation.simplices

    # Compute edges: for each triangle, the 3 edges are the 3 pairs
    edges = np.array([
        [(tri[i], tri[j]) for i in range(3) for j in range(i + 1, 3)]
        for tri in simplices
    ])  # (N, 3, 2)

    solver = ElasticSolver(points, simplices, edges)

    N = len(simplices)
    default_rigs = torch.ones(N, 3, dtype=torch.float64)
    default_rl = solver.actual_length2.sqrt()  # (N, 3)

    return solver, default_rigs, default_rl
