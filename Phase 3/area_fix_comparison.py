#!/usr/bin/env python3
"""
Three-way Poisson ratio comparison:
  1) Standard analytical D2C (1/16 normalization)
  2) Area-fixed D2C (1/Area_s normalization)
  3) KUBC mechanical simulation

For 4 network configurations (A-D).
"""

import sys, os, copy
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Phase 2'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Phase 3'))

import Disc_2_Cont_optimized as D2C
from sweep_utils import virtual_distortion_rigidities


# =========================================================================
# Area-fixed D2C: 3 modifications to the standard pipeline
# =========================================================================

def triangle_areas(points, simplices):
    """Compute area of each triangle."""
    p = points[simplices]  # (N, 3, 2)
    return 0.5 * np.abs(
        (p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1]) -
        (p[:, 2, 0] - p[:, 0, 0]) * (p[:, 1, 1] - p[:, 0, 1])
    )


def _woodbury_solve_area(A_blocks, B_blocks, dA_vecs, weights):
    """Woodbury solve with area weights instead of uniform 1/N.

    weights: (N,) array with w_s = Area_s / sum(Area).
    Changes vs standard: U = [w_1*I; w_2*I; ...] instead of (1/N)*[I;I;...].
    """
    N = A_blocks.shape[0]
    eps = 1e-14 * np.max(np.abs(A_blocks))
    A_blocks_reg = A_blocks + eps * np.eye(9)[None, :, :]
    A_inv = np.linalg.inv(A_blocks_reg)

    # y_i = A_i^{-1} @ dA_i
    y = np.einsum('nij,nj->ni', A_inv, dA_vecs)

    # V @ y = sum_j B_j @ y_j
    Vy = np.einsum('nij,nj->i', B_blocks, y)

    # S = sum_j w_j * B_j @ A_j^{-1}  (instead of 1/N * sum)
    S = np.einsum('n,nij,njk->ik', weights, B_blocks, A_inv)

    # Solve (I - S) z = Vy
    z = np.linalg.solve(np.eye(9) - S, Vy)

    # W_i = -(y_i + w_i * A_i^{-1} @ z)  (instead of 1/N)
    correction = np.einsum('nij,j->ni', A_inv, z) * weights[:, None]
    W = -(y + correction)

    return W


def analyze_area_fixed(triangulation, rigidities_per_tri=None):
    """D2C with 1/Area_s normalization (area-fixed variant).

    3 changes vs standard:
    1. Bare tensor: factor = k/l^2/Area_s  (instead of k/l^2/16)
    2. Woodbury coupling: area weights w_s = A_s/sum(A)  (instead of 1/N)
    3. Final averaging: area-weighted mean  (instead of simple mean)
    """
    # Step 1: Build edges (this resets rigidities to [])
    D2C.add_edges_to_triangulation(triangulation)
    # Restore rigidities if provided
    if rigidities_per_tri is not None:
        N = len(triangulation.simplices)
        triangulation.rigidities = [list(rigidities_per_tri[i]) for i in range(N)]

    N = len(triangulation.simplices)
    edges = triangulation.edges
    positions = triangulation.points

    # Compute triangle areas
    areas = triangle_areas(positions, triangulation.simplices)
    weights = areas / areas.sum()  # (N,) area weights

    # Edge vectors
    node_a = edges[:, :, 0]
    node_b = edges[:, :, 1]
    vecs = positions[node_a] - positions[node_b]
    vx = vecs[:, :, 0]
    vy = vecs[:, :, 1]

    # Rigidities
    rigs = np.ones((N, 3))
    for i, r in enumerate(triangulation.rigidities):
        if len(r) > 0:
            rigs[i] = r

    # Length squared
    length2 = np.sum(vecs ** 2, axis=2)
    for i, rl in enumerate(triangulation.rest_lenghts):
        if len(rl) > 0:
            length2[i] = np.array(rl) ** 2

    # CHANGE 1: use 1/Area_s instead of 1/16
    factor = rigs / length2 / areas[:, None]

    bare = np.column_stack([
        np.sum(factor * vx ** 4, axis=1),
        np.sum(factor * vx ** 3 * vy, axis=1),
        np.sum(factor * vx ** 2 * vy ** 2, axis=1),
        np.sum(factor * vx * vy ** 3, axis=1),
        np.sum(factor * vy ** 4, axis=1),
    ])

    # Delta tensor (area-weighted mean for the reference)
    mean_tensor = np.einsum('n,ni->i', weights, bare)  # area-weighted mean
    delta = bare - mean_tensor

    # Build 9x9 blocks
    A_blocks = D2C._batch_to_9x9(bare)
    B_blocks = D2C._batch_to_9x9(delta)
    dA_vecs = D2C._batch_to_9vec(delta)

    # CHANGE 2: Woodbury with area weights
    Ws = _woodbury_solve_area(A_blocks, B_blocks, dA_vecs, weights)

    # Actual elastic tensor
    actual = D2C._compute_actual_elastic_tensor_vectorized(bare, Ws)

    # CHANGE 3: area-weighted average
    C = np.einsum('n,ni->i', weights, actual)

    triangulation.totalElasticTensor = C
    triangulation.PoissonsRatio = (C[2] * C[3] - C[1] * C[4]) / (C[0] * C[3] - C[1] ** 2)
    triangulation.YoungsModulus = (
        (C[2] ** 2 * C[3] - 2 * C[1] * C[2] * C[4] + C[1] ** 2 * C[5] +
         C[0] * (C[4] ** 2 - C[3] * C[5])) /
        (C[1] ** 2 - C[0] * C[3])
    )


# =========================================================================
# Build the 4 configurations as scipy Delaunay-like objects
# =========================================================================

def make_triangulation(points, simplices, rigidities_per_tri=None):
    """Create a triangulation-like object for D2C.

    rigidities_per_tri: (N, 3) array or None (defaults to uniform k=1).
    """
    import scipy.spatial
    # Use existing Delaunay as container
    tri = scipy.spatial.Delaunay(points)
    tri.simplices = simplices.copy()

    # Pre-set rigidities and rest lengths as D2C expects
    N = len(simplices)
    if rigidities_per_tri is not None:
        tri.rigidities = [list(rigidities_per_tri[i]) for i in range(N)]
    else:
        tri.rigidities = [[] for _ in range(N)]
    tri.rest_lenghts = [[] for _ in range(N)]  # empty = use actual edge lengths
    return tri


def build_configs(size=(14, 14), eta=0.15, a=10, seed=42):
    """Return list of (label, points, simplices, rigidities_or_None)."""
    vd = virtual_distortion_rigidities(size=size, eta=eta, a=a, seed=seed)
    tri_reg = vd['tri']
    pts_reg = tri_reg.points.copy()
    simps = tri_reg.simplices.copy()
    pts_def = vd['deformed_points'].copy()
    rigs_vd = vd['rigidities_np']  # (N, 3)

    return [
        ('A) Regular, k=1', pts_reg, simps, None),
        ('B) Regular, VD rigs', pts_reg, simps, rigs_vd),
        ('C) Deformed, k=1', pts_def, simps, None),
        ('D) Deformed + VD rigs', pts_def, simps, rigs_vd),
    ]


# =========================================================================
# Main
# =========================================================================

if __name__ == '__main__':
    configs = build_configs()

    # KUBC simulation results (from mechanical_simulation.py runs)
    nu_kubc = {
        'A) Regular, k=1': 0.3303,
        'B) Regular, VD rigs': 0.2055,
        'C) Deformed, k=1': 0.3146,
        'D) Deformed + VD rigs': 0.1065,
    }

    print("=" * 80)
    print("  Three-way Poisson ratio comparison")
    print("  Standard D2C (1/16) vs Area-fixed D2C (1/Area) vs KUBC Simulation")
    print("=" * 80)
    print(f"  Mesh size: (14,14), eta=0.15, a=10, seed=42")
    print()

    results = []

    for label, pts, simps, rigs in configs:
        # Standard D2C (1/16)
        # Note: add_edges_to_triangulation resets rigidities, so we must
        # set them AFTER calling it. We call it manually, then set rigs.
        tri_std = make_triangulation(pts, simps)
        D2C.add_edges_to_triangulation(tri_std)
        if rigs is not None:
            for i in range(len(simps)):
                tri_std.rigidities[i] = list(rigs[i])
        tri_std.BareElasticTensor = D2C._compute_local_tensors_vectorized(tri_std)
        N = len(tri_std.simplices)
        mean_tensor = np.mean(tri_std.BareElasticTensor, 0)
        tri_std.delta_tensor = tri_std.BareElasticTensor - mean_tensor
        A_blocks = D2C._batch_to_9x9(tri_std.BareElasticTensor)
        B_blocks = D2C._batch_to_9x9(tri_std.delta_tensor)
        dA_vecs = D2C._batch_to_9vec(tri_std.delta_tensor)
        tri_std.Ws = D2C._woodbury_solve(A_blocks, B_blocks, dA_vecs)
        tri_std.ActualElasticTensor = D2C._compute_actual_elastic_tensor_vectorized(
            tri_std.BareElasticTensor, tri_std.Ws)
        C = np.mean(tri_std.ActualElasticTensor, 0)
        tri_std.totalElasticTensor = C
        nu_std = (C[2] * C[3] - C[1] * C[4]) / (C[0] * C[3] - C[1] ** 2)

        # Area-fixed D2C (1/Area_s)
        tri_af = make_triangulation(pts, simps)
        analyze_area_fixed(tri_af, rigidities_per_tri=rigs)
        nu_af = tri_af.PoissonsRatio

        nu_sim = nu_kubc[label]

        results.append((label, nu_std, nu_af, nu_sim))
        print(f"  {label:25s}  std={nu_std:+.4f}  area-fix={nu_af:+.4f}  sim={nu_sim:+.4f}")

    print()
    print("=" * 85)
    print(f"  {'Config':25s}  {'D2C (1/16)':>10s}  {'D2C (1/A)':>10s}  {'KUBC Sim':>10s}  "
          f"{'err(1/16)':>10s}  {'err(1/A)':>10s}")
    print("  " + "-" * 83)
    for label, nu_std, nu_af, nu_sim in results:
        if abs(nu_sim) > 1e-6:
            err_std = abs(nu_std - nu_sim) / abs(nu_sim) * 100
            err_af = abs(nu_af - nu_sim) / abs(nu_sim) * 100
            print(f"  {label:25s}  {nu_std:+10.4f}  {nu_af:+10.4f}  {nu_sim:+10.4f}  "
                  f"{err_std:9.1f}%  {err_af:9.1f}%")
        else:
            print(f"  {label:25s}  {nu_std:+10.4f}  {nu_af:+10.4f}  {nu_sim:+10.4f}")
    print("=" * 85)
    print()
    print("Notes:")
    print("  - err = |nu_method - nu_sim| / |nu_sim| * 100%")
    print("  - D2C (1/16) = standard analytical homogenization")
    print("  - D2C (1/A)  = area-fixed variant (1/Area_s normalization)")
    print("  - KUBC Sim   = spring energy minimization with kinematic boundary conditions")
