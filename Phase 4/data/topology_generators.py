"""Topology generators for Phase 4 GNN training.

Provides diverse mesh topologies across 4 categories:
  1. Hexagonal lattice family (wraps existing D2C functions)
  2. Fully random triangulations (Poisson, blue noise, clustered, gradient)
  3. Lattices with non-trivial basis (kagome, square, snub square, etc.)
  4. Non-triangulated meshes with soft-edge regularization (honeycomb, etc.)

All generators return a TriangulationResult dataclass that wraps scipy Delaunay
objects and adds metadata needed for GNN training (edge masks, rigidity defaults).
"""

import numpy as np
import scipy.spatial
from dataclasses import dataclass, field
from typing import Optional, Tuple
from pathlib import Path
import sys

# Import existing mesh generators
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "Phase 2"))
sys.path.insert(0, str(PROJECT_ROOT / "Phase 3"))

import Disc_2_Cont_optimized as D2C


# ─────────────────────────────────────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TriangulationResult:
    """Container for a triangulated mesh with metadata.

    Attributes:
        points:       (M, 2) node coordinates.
        simplices:    (N_tri, 3) triangle vertex indices.
        hard_edge_set: set of frozenset pairs — the original structural bonds.
                      None if all edges are structural (fully triangulated mesh).
        k_soft_ratio: ratio of soft to hard spring constants for regularization.
        topo_name:    human-readable topology name.
        topo_class:   category string for grouping.
    """
    points: np.ndarray
    simplices: np.ndarray
    hard_edge_set: Optional[set] = None
    k_soft_ratio: float = 1e-3
    topo_name: str = ""
    topo_class: str = ""

    def to_delaunay_compat(self):
        """Return an object compatible with from_triangulation().

        The forward solver expects tri.points, tri.simplices.
        """
        obj = _DelaunayCompat(self.points, self.simplices)
        return obj

    @property
    def n_tri(self):
        return len(self.simplices)

    @property
    def n_points(self):
        return len(self.points)

    def edge_is_hard(self, i, j):
        """Check if edge (i,j) is a hard (structural) edge."""
        if self.hard_edge_set is None:
            return True
        return frozenset((i, j)) in self.hard_edge_set

    def get_default_rigidities(self):
        """Return (N_tri, 3) default rigidities: 1.0 for hard, k_soft for soft."""
        rigs = np.ones((self.n_tri, 3))
        if self.hard_edge_set is None:
            return rigs
        for t, tri in enumerate(self.simplices):
            edges = [(tri[0], tri[1]), (tri[0], tri[2]), (tri[1], tri[2])]
            for e, (a, b) in enumerate(edges):
                if not self.edge_is_hard(a, b):
                    rigs[t, e] = self.k_soft_ratio
        return rigs

    def get_hard_edge_mask_per_triangle(self):
        """Return (N_tri, 3) bool mask: True for hard edges."""
        mask = np.ones((self.n_tri, 3), dtype=bool)
        if self.hard_edge_set is None:
            return mask
        for t, tri in enumerate(self.simplices):
            edges = [(tri[0], tri[1]), (tri[0], tri[2]), (tri[1], tri[2])]
            for e, (a, b) in enumerate(edges):
                if not self.edge_is_hard(a, b):
                    mask[t, e] = False
        return mask


class _DelaunayCompat:
    """Minimal object mimicking scipy.spatial.Delaunay for from_triangulation()."""
    def __init__(self, points, simplices):
        self.points = points
        self.simplices = simplices


# ─────────────────────────────────────────────────────────────────────────────
# Helper: filter triangles to interior of a bounding box
# ─────────────────────────────────────────────────────────────────────────────

def _filter_interior(points, simplices, size):
    """Keep only triangles whose centroid is inside [-size, size]."""
    centroids = np.mean(points[simplices], axis=1)
    mask = (np.abs(centroids[:, 0]) <= size[0]) & (np.abs(centroids[:, 1]) <= size[1])
    return simplices[mask]


# ═════════════════════════════════════════════════════════════════════════════
# CATEGORY 1: Hexagonal lattice family (wraps existing functions)
# ═════════════════════════════════════════════════════════════════════════════

def generate_iso_crystal(size):
    """Regular isotropic triangular lattice (z=6)."""
    tri = D2C.generate_cryratl_points(size=size, shape=(1, 1), orientation=0)
    return TriangulationResult(
        points=tri.points, simplices=tri.simplices,
        topo_name='iso_crystal', topo_class='crystal',
    )


def generate_aniso_crystal(size, shape=(1.5, 0.8), orientation=np.pi / 6):
    """Anisotropic (stretched + rotated) triangular lattice (z=6)."""
    tri = D2C.generate_cryratl_points(size=size, shape=shape, orientation=orientation)
    return TriangulationResult(
        points=tri.points, simplices=tri.simplices,
        topo_name='aniso_crystal', topo_class='crystal',
    )


def generate_rectangular_lattice(size, aspect=1.6):
    """Rectangular Bravais lattice (z=6 after Delaunay triangulation).

    Lattice vectors a1 = (a, 0), a2 = (0, b) with a != b.
    One atom per unit cell. Delaunay triangulation produces anisotropic
    triangles — longer in the stretched direction.

    This is a distinct 2D Bravais lattice from the square (a=b) and
    hexagonal families.

    Args:
        size: (sx, sy) half-extents.
        aspect: ratio a/b of lattice constants. Default 1.6.
    """
    # Use geometric mean = 1 so total density is comparable to iso_crystal
    b = 1.0 / np.sqrt(aspect)
    a = aspect * b

    buf = 3
    nx = int(np.ceil((size[0] + buf) / a)) + 1
    ny = int(np.ceil((size[1] + buf) / b)) + 1

    xs = np.arange(-nx, nx + 1) * a
    ys = np.arange(-ny, ny + 1) * b
    xx, yy = np.meshgrid(xs, ys)
    points = np.column_stack([xx.ravel(), yy.ravel()])

    DM = scipy.spatial.Delaunay(points)
    simplices = _filter_interior(DM.points, DM.simplices, size)

    return TriangulationResult(
        points=DM.points, simplices=simplices,
        topo_name='rectangular', topo_class='rectangular',
    )


def generate_oblique_lattice(size, a_len=1.0, b_len=1.2, angle_deg=70):
    """Oblique Bravais lattice (z=6 after Delaunay triangulation).

    The most general 2D Bravais lattice: a1 and a2 have different lengths
    and a non-special angle (neither 60°, 90°, nor 120°). One atom per
    unit cell.

    Args:
        size: (sx, sy) half-extents.
        a_len: length of first lattice vector.
        b_len: length of second lattice vector.
        angle_deg: angle between lattice vectors in degrees.
    """
    angle = np.radians(angle_deg)
    a1 = np.array([a_len, 0.0])
    a2 = np.array([b_len * np.cos(angle), b_len * np.sin(angle)])

    basis = np.array([[0.0, 0.0]])
    points, _ = _generate_lattice_points(size, a1, a2, basis)

    DM = scipy.spatial.Delaunay(points)
    simplices = _filter_interior(DM.points, DM.simplices, size)

    return TriangulationResult(
        points=DM.points, simplices=simplices,
        topo_name='oblique', topo_class='oblique',
    )


def generate_foam(size, eta=0.2):
    """Perturbed hexagonal lattice (z~6). Topology from crystal, positions perturbed."""
    tri = D2C.generate_foam_points(size=size, eta=eta)
    return TriangulationResult(
        points=tri.points, simplices=tri.simplices,
        topo_name=f'foam_eta{eta:.2f}'.replace('.', ''),
        topo_class=f'foam_{eta:.2f}'.replace('.', ''),
    )


def generate_foam_retriangulated(size, eta=0.2):
    """Perturbed hexagonal lattice with Delaunay re-triangulation.
    Unlike generate_foam, topology can differ from the crystal."""
    tri = D2C.generate_foam_points2(size=size, eta=eta)
    return TriangulationResult(
        points=tri.points, simplices=tri.simplices,
        topo_name=f'foam_retri_eta{eta:.2f}'.replace('.', ''),
        topo_class=f'foam_{eta:.2f}'.replace('.', ''),
    )


# ═════════════════════════════════════════════════════════════════════════════
# CATEGORY 2: Fully random triangulations
# ═════════════════════════════════════════════════════════════════════════════

def generate_poisson_delaunay(size):
    """Uniform random points -> Delaunay triangulation.

    Coordination varies widely (4-9+). No underlying lattice structure.
    Uses the same density as the hexagonal lattice for comparable triangle count.
    """
    density = 2.0 / np.sqrt(3)
    x_lo, x_hi = -(size[0] + 2), size[0] + 2
    y_lo, y_hi = -(size[1] + 2), size[1] + 2
    area = (x_hi - x_lo) * (y_hi - y_lo)
    n_points = int(round(density * area))
    points = np.column_stack([
        np.random.uniform(x_lo, x_hi, n_points),
        np.random.uniform(y_lo, y_hi, n_points),
    ])
    DM = scipy.spatial.Delaunay(points)
    simplices = _filter_interior(DM.points, DM.simplices, size)
    return TriangulationResult(
        points=DM.points, simplices=simplices,
        topo_name='poisson_delaunay', topo_class='poisson',
    )


def generate_blue_noise(size, min_dist=0.6):
    """Poisson disk sampling -> Delaunay. More uniform than pure random.

    Uses Bridson's fast algorithm: O(n) point generation with guaranteed
    minimum distance between points. Coordination numbers are narrower
    than pure Poisson (typically 5-7).

    Args:
        size: (sx, sy) half-extents of the interior region.
        min_dist: minimum distance between any two points.
    """
    x_lo, x_hi = -(size[0] + 2), size[0] + 2
    y_lo, y_hi = -(size[1] + 2), size[1] + 2
    width, height = x_hi - x_lo, y_hi - y_lo

    # Bridson's algorithm
    cell_size = min_dist / np.sqrt(2)
    cols = int(np.ceil(width / cell_size))
    rows = int(np.ceil(height / cell_size))
    grid = -np.ones((rows, cols), dtype=int)

    points = []
    active = []

    def _grid_idx(p):
        return (int((p[1] - y_lo) / cell_size),
                int((p[0] - x_lo) / cell_size))

    # Seed point
    p0 = np.array([np.random.uniform(x_lo, x_hi),
                    np.random.uniform(y_lo, y_hi)])
    points.append(p0)
    active.append(0)
    r, c = _grid_idx(p0)
    grid[r, c] = 0

    k_samples = 30  # candidates per active point

    while active:
        idx = np.random.randint(len(active))
        point = points[active[idx]]
        found = False

        for _ in range(k_samples):
            angle = np.random.uniform(0, 2 * np.pi)
            radius = np.random.uniform(min_dist, 2 * min_dist)
            candidate = point + radius * np.array([np.cos(angle), np.sin(angle)])

            if not (x_lo <= candidate[0] < x_hi and y_lo <= candidate[1] < y_hi):
                continue

            cr, cc = _grid_idx(candidate)
            ok = True
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] >= 0:
                        if np.linalg.norm(candidate - points[grid[nr, nc]]) < min_dist:
                            ok = False
                            break
                if not ok:
                    break

            if ok:
                new_idx = len(points)
                points.append(candidate)
                active.append(new_idx)
                grid[cr, cc] = new_idx
                found = True
                break

        if not found:
            active.pop(idx)

    points = np.array(points)
    DM = scipy.spatial.Delaunay(points)
    simplices = _filter_interior(DM.points, DM.simplices, size)
    return TriangulationResult(
        points=DM.points, simplices=simplices,
        topo_name='blue_noise', topo_class='blue_noise',
    )


def generate_clustered(size, cluster_spacing=2.0, cluster_std=0.4, pts_per_cluster=15):
    """Gaussian mixture model point process -> Delaunay.

    Dense cluster cores have high coordination (8-12+),
    sparse bridge regions have low coordination (3-4).
    Cluster centers are placed on a jittered grid so the overall
    structure remains spatially uniform.

    Args:
        size: (sx, sy) half-extents.
        cluster_spacing: approx distance between cluster centers.
        cluster_std: standard deviation of each cluster.
        pts_per_cluster: number of points per cluster.
    """
    x_lo, x_hi = -(size[0] + 2), size[0] + 2
    y_lo, y_hi = -(size[1] + 2), size[1] + 2

    # Place cluster centers on a grid, then jitter
    xs = np.arange(x_lo + cluster_spacing / 2, x_hi, cluster_spacing)
    ys = np.arange(y_lo + cluster_spacing / 2, y_hi, cluster_spacing)
    cx, cy = np.meshgrid(xs, ys)
    centers = np.column_stack([cx.ravel(), cy.ravel()])
    # Jitter centers by up to 30% of spacing
    centers += np.random.uniform(-0.3 * cluster_spacing, 0.3 * cluster_spacing,
                                 centers.shape)

    points = []
    for c in centers:
        cluster_pts = c + cluster_std * np.random.randn(pts_per_cluster, 2)
        cluster_pts[:, 0] = np.clip(cluster_pts[:, 0], x_lo, x_hi)
        cluster_pts[:, 1] = np.clip(cluster_pts[:, 1], y_lo, y_hi)
        points.append(cluster_pts)

    points = np.vstack(points)
    DM = scipy.spatial.Delaunay(points)
    simplices = _filter_interior(DM.points, DM.simplices, size)
    return TriangulationResult(
        points=DM.points, simplices=simplices,
        topo_name='clustered', topo_class='clustered',
    )


def generate_gradient_density(size, density_ratio=4.0):
    """Non-uniform density: dense on left, sparse on right -> Delaunay.

    Coordination varies smoothly across the mesh.

    Args:
        size: (sx, sy) half-extents.
        density_ratio: max/min density ratio from left to right.
    """
    base_density = 2.0 / np.sqrt(3)
    x_lo, x_hi = -(size[0] + 2), size[0] + 2
    y_lo, y_hi = -(size[1] + 2), size[1] + 2
    width = x_hi - x_lo

    # Rejection sampling: density(x) = base * (1 + (ratio-1) * (x - x_lo) / width)
    # normalized so average density = base_density
    avg_density = base_density
    area = width * (y_hi - y_lo)
    n_target = int(round(avg_density * area))

    # max density for rejection sampling
    max_density = base_density * density_ratio

    points = []
    while len(points) < n_target:
        batch_size = n_target * 3
        candidates = np.column_stack([
            np.random.uniform(x_lo, x_hi, batch_size),
            np.random.uniform(y_lo, y_hi, batch_size),
        ])
        # Local density: linear from 1 to density_ratio across x
        frac = (candidates[:, 0] - x_lo) / width
        local_density = base_density * (1 + (density_ratio - 1) * frac)
        accept_prob = local_density / max_density
        accepted = np.random.rand(batch_size) < accept_prob
        points.append(candidates[accepted])

    points = np.vstack(points)[:n_target]
    DM = scipy.spatial.Delaunay(points)
    simplices = _filter_interior(DM.points, DM.simplices, size)
    return TriangulationResult(
        points=DM.points, simplices=simplices,
        topo_name='gradient_density', topo_class='gradient',
    )


# ═════════════════════════════════════════════════════════════════════════════
# CATEGORY 3 & 4: Lattices with basis + non-triangulated with regularization
# ═════════════════════════════════════════════════════════════════════════════

# ── Shared helpers ────────────────────────────────────────────────────────────

def _generate_lattice_points(size, a1, a2, basis):
    """Generate points for a 2D Bravais lattice with given basis.

    Args:
        size: (sx, sy) half-extents for the interior.
        a1, a2: (2,) lattice vectors.
        basis: (n_basis, 2) positions within the unit cell.

    Returns:
        points: (M, 2) all generated lattice points (includes buffer zone).
        interior_mask: (M,) bool mask for points inside [-size, size].
    """
    buf = 2.0
    max_extent = max(size[0], size[1]) + buf
    # Determine range of lattice indices needed
    # Use the inverse of the lattice matrix to find bounds
    lat_mat = np.column_stack([a1, a2])  # (2, 2)
    lat_inv = np.linalg.inv(lat_mat)
    # Corner of the buffered box
    corners = np.array([
        [max_extent, max_extent],
        [max_extent, -max_extent],
        [-max_extent, max_extent],
        [-max_extent, -max_extent],
    ])
    frac_coords = corners @ lat_inv.T
    n_max = int(np.ceil(np.max(np.abs(frac_coords)))) + 1

    points = []
    for n in range(-n_max, n_max + 1):
        for m in range(-n_max, n_max + 1):
            origin = n * a1 + m * a2
            for b in basis:
                p = origin + b
                if abs(p[0]) <= size[0] + buf and abs(p[1]) <= size[1] + buf:
                    points.append(p)

    points = np.array(points)
    interior = ((np.abs(points[:, 0]) <= size[0] + 0.5) &
                (np.abs(points[:, 1]) <= size[1] + 0.5))
    return points, interior


def _build_edges_from_neighbor_distance(points, max_dist):
    """Find all pairs of points within max_dist (structural bonds).

    Returns set of frozenset pairs (i, j).
    """
    from scipy.spatial import cKDTree
    tree = cKDTree(points)
    pairs = tree.query_pairs(r=max_dist)
    return {frozenset(p) for p in pairs}


def _triangulate_with_soft_edges(points, hard_edges, size, k_soft_ratio=1e-3):
    """Triangulate a point set and mark hard vs soft edges.

    1. Delaunay triangulate all points.
    2. Filter to interior triangles.
    3. Mark edges: hard if in hard_edges set, soft otherwise.

    Returns:
        TriangulationResult with hard_edge_set populated.
    """
    DM = scipy.spatial.Delaunay(points)
    simplices = _filter_interior(DM.points, DM.simplices, size)
    return TriangulationResult(
        points=DM.points,
        simplices=simplices,
        hard_edge_set=hard_edges,
        k_soft_ratio=k_soft_ratio,
    )


def _add_face_centers(points, faces, hard_edges):
    """Add center vertex to each polygonal face and connect to corners.

    Args:
        points: (M, 2) existing points.
        faces: list of lists of vertex indices, each defining a polygon face.
        hard_edges: set of frozenset pairs (existing structural bonds).

    Returns:
        new_points: (M + len(faces), 2) points with centers appended.
        new_hard_edges: updated hard_edges (unchanged -- new edges are soft).
        face_simplices: list of (n_verts, 3) triangle arrays for each face.
    """
    new_points = list(points)
    all_new_simplices = []

    for face_verts in faces:
        # Compute centroid
        center = np.mean(points[face_verts], axis=0)
        center_idx = len(new_points)
        new_points.append(center)

        # Create triangles: (v_i, v_{i+1}, center) for each edge of the face
        n = len(face_verts)
        for i in range(n):
            v_a = face_verts[i]
            v_b = face_verts[(i + 1) % n]
            all_new_simplices.append([v_a, v_b, center_idx])

    new_points = np.array(new_points)
    return new_points, hard_edges, np.array(all_new_simplices)


# ── Honeycomb lattice ─────────────────────────────────────────────────────────

def generate_honeycomb(size, spacing=1.0, k_soft_ratio=1e-3):
    """True honeycomb lattice (z=3) with center-point regularization.

    The honeycomb has 2 atoms per hexagonal unit cell. Each vertex has exactly
    3 bonds. To make it compatible with the triangulation-based solver, we
    add a center vertex inside each hexagonal face and connect it to all 6
    corner vertices with soft springs.

    Args:
        size: (sx, sy) half-extents of the interior region.
        spacing: scale factor for lattice constant. Default 1.0.
        k_soft_ratio: rigidity of soft springs relative to hard (default 1e-3).
    """
    # Honeycomb lattice vectors
    a1 = np.array([1.0, 0.0]) * spacing
    a2 = np.array([0.5, np.sqrt(3) / 2]) * spacing

    # Two-atom basis
    # A-site at origin, B-site displaced by (0, 1/sqrt(3))
    d = spacing / np.sqrt(3)
    basis = np.array([
        [0.0, 0.0],
        [0.0, d],
    ])

    points, interior = _generate_lattice_points(size, a1, a2, basis)
    n_lattice = len(points)

    # Bond distance: nearest-neighbor in honeycomb = d
    bond_dist = d * 1.05  # small tolerance
    hard_edges = _build_edges_from_neighbor_distance(points, bond_dist)

    # Find hexagonal faces.
    # Each hexagonal face has 6 vertices. In the honeycomb, each hexagon
    # is centered at the centroid of a unit cell.
    # Strategy: find faces by walking around each vertex's neighbors.
    faces = _find_honeycomb_faces(points, hard_edges, size)

    if len(faces) > 0:
        # Add center vertices and triangulate faces
        new_points, hard_edges, face_simplices = _add_face_centers(
            points, faces, hard_edges
        )

        # Also triangulate the structural triangles (if any exist in the boundary)
        # Use full Delaunay on all points and merge
        DM = scipy.spatial.Delaunay(new_points)
        all_simplices = _filter_interior(DM.points, DM.simplices, size)

        return TriangulationResult(
            points=DM.points,
            simplices=all_simplices,
            hard_edge_set=hard_edges,
            k_soft_ratio=k_soft_ratio,
            topo_name='honeycomb',
            topo_class='honeycomb',
        )
    else:
        # Fallback: just Delaunay triangulate and mark honeycomb edges as hard
        return _triangulate_with_soft_edges(
            points, hard_edges, size, k_soft_ratio
        )


def _find_honeycomb_faces(points, hard_edges, size):
    """Find hexagonal faces of the honeycomb lattice.

    Uses the dual graph: each hexagonal face corresponds to a cycle of length 6
    in the bond graph. We find these by walking around each vertex.
    """
    from collections import defaultdict

    # Build adjacency from hard edges
    adj = defaultdict(set)
    for edge in hard_edges:
        i, j = tuple(edge)
        adj[i].add(j)
        adj[j].add(i)

    # For each vertex, sort neighbors by angle
    def _sorted_neighbors(v):
        nbrs = list(adj[v])
        if len(nbrs) == 0:
            return []
        angles = [np.arctan2(points[n][1] - points[v][1],
                             points[n][0] - points[v][0]) for n in nbrs]
        order = np.argsort(angles)
        return [nbrs[i] for i in order]

    sorted_adj = {v: _sorted_neighbors(v) for v in adj}

    # Walk around faces using the "next edge" rule:
    # From edge (u, v), the next edge is (v, w) where w is the neighbor of v
    # that comes right after u in the sorted neighbor list of v (clockwise).
    visited_halfedges = set()
    faces = []

    for u in sorted_adj:
        for v in sorted_adj[u]:
            if (u, v) in visited_halfedges:
                continue

            # Walk around the face
            face = []
            cu, cv = u, v
            for _ in range(20):  # safety limit
                if (cu, cv) in visited_halfedges:
                    break
                visited_halfedges.add((cu, cv))
                face.append(cu)

                # Find next: after cu in cv's sorted neighbors
                nbrs = sorted_adj.get(cv, [])
                if len(nbrs) < 2:
                    break
                try:
                    idx = nbrs.index(cu)
                except ValueError:
                    break
                # Previous neighbor in CW order = next face edge
                next_v = nbrs[(idx - 1) % len(nbrs)]
                cu, cv = cv, next_v

            # Keep only hexagonal faces (6 vertices)
            if len(face) == 6:
                # Check that the face center is inside the domain
                center = np.mean(points[face], axis=0)
                if abs(center[0]) <= size[0] and abs(center[1]) <= size[1]:
                    faces.append(face)

    return faces


# ── Kagome lattice ────────────────────────────────────────────────────────────

def generate_kagome(size, spacing=1.0, k_soft_ratio=1e-3):
    """Kagome lattice (z=4, 3 atoms per hexagonal unit cell).

    Corner-sharing triangles with hexagonal voids. The triangular faces
    are already triangulated (hard edges). The hexagonal voids are
    triangulated with center-point + soft radial edges.

    Args:
        size: (sx, sy) half-extents.
        spacing: scale factor for lattice constant. Default 1.0.
        k_soft_ratio: rigidity ratio for soft edges.
    """
    # Kagome on hexagonal Bravais lattice
    a1 = np.array([2.0, 0.0]) * spacing
    a2 = np.array([1.0, np.sqrt(3)]) * spacing

    # 3-atom basis: midpoints of the Bravais cell edges
    basis = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, np.sqrt(3) / 2],
    ]) * spacing

    points, interior = _generate_lattice_points(size, a1, a2, basis)

    # Kagome bonds: each site connects to 4 nearest neighbors
    # Bond length = spacing (distance between basis points)
    bond_dist = spacing * 1.05
    hard_edges = _build_edges_from_neighbor_distance(points, bond_dist)

    # Delaunay triangulate: this automatically fills hexagonal voids
    # The Delaunay edges inside hexagons become soft
    DM = scipy.spatial.Delaunay(points)
    simplices = _filter_interior(DM.points, DM.simplices, size)

    return TriangulationResult(
        points=DM.points,
        simplices=simplices,
        hard_edge_set=hard_edges,
        k_soft_ratio=k_soft_ratio,
        topo_name='kagome',
        topo_class='kagome',
    )


# ── Square lattice ────────────────────────────────────────────────────────────

def generate_square_lattice(size, spacing=1.0, k_soft_ratio=1e-3):
    """Square lattice (z=4) with soft diagonal regularization.

    Each square face gets one diagonal (alternating NE-SW and NW-SE)
    as a soft spring. This creates 2 triangles per square face.

    Args:
        size: (sx, sy) half-extents.
        spacing: distance between adjacent vertices.
        k_soft_ratio: rigidity of diagonal springs.
    """
    buf = 2
    nx = int(np.ceil((size[0] + buf) / spacing)) * 2
    ny = int(np.ceil((size[1] + buf) / spacing)) * 2

    # Generate grid points
    xs = np.arange(-nx, nx + 1) * spacing
    ys = np.arange(-ny, ny + 1) * spacing
    xx, yy = np.meshgrid(xs, ys)
    points = np.column_stack([xx.ravel(), yy.ravel()])

    # Build grid index: (ix, iy) -> point index
    n_cols = len(xs)
    n_rows = len(ys)

    def _idx(ix, iy):
        return iy * n_cols + ix

    # Hard edges: horizontal and vertical bonds
    hard_edges = set()
    for iy in range(n_rows):
        for ix in range(n_cols):
            if ix + 1 < n_cols:
                hard_edges.add(frozenset((_idx(ix, iy), _idx(ix + 1, iy))))
            if iy + 1 < n_rows:
                hard_edges.add(frozenset((_idx(ix, iy), _idx(ix, iy + 1))))

    # Triangulate: add one diagonal per square, alternating direction
    simplices = []
    for iy in range(n_rows - 1):
        for ix in range(n_cols - 1):
            v00 = _idx(ix, iy)
            v10 = _idx(ix + 1, iy)
            v01 = _idx(ix, iy + 1)
            v11 = _idx(ix + 1, iy + 1)

            if (ix + iy) % 2 == 0:
                # NE-SW diagonal: v00 -- v11
                simplices.append([v00, v10, v11])
                simplices.append([v00, v11, v01])
            else:
                # NW-SE diagonal: v10 -- v01
                simplices.append([v00, v10, v01])
                simplices.append([v10, v11, v01])

    simplices = np.array(simplices)
    simplices = _filter_interior(points, simplices, size)

    return TriangulationResult(
        points=points,
        simplices=simplices,
        hard_edge_set=hard_edges,
        k_soft_ratio=k_soft_ratio,
        topo_name='square_lattice',
        topo_class='square',
    )


# ── Penrose quasicrystal ─────────────────────────────────────────────────────

def generate_penrose(size, spacing=4.0):
    """Penrose quasicrystal point set -> Delaunay triangulation.

    Uses the cut-and-project method: project from a 5D hypercubic lattice
    onto a 2D plane with 5-fold symmetry. Only lattice points whose
    perpendicular-space projection falls within a decagonal acceptance
    window are kept.

    This produces a quasiperiodic point set with local 5-fold symmetry
    but no translational periodicity — the hallmark of quasicrystals.
    All edges are hard (fully triangulated Delaunay).

    Args:
        size: (sx, sy) half-extents of the interior region.
        spacing: scale factor for inter-point distance. Default 4.0 gives
                 ~700 points at size=(10,10), comparable to the triangular lattice.
    """
    # Projection matrices for Penrose (5-fold) quasicrystal
    # Parallel-space: project onto plane with 5-fold symmetry
    k = np.arange(5)
    angles_par = 2 * np.pi * k / 5
    angles_perp = 4 * np.pi * k / 5  # perpendicular space uses 2nd harmonic

    P_par = np.array([np.cos(angles_par), np.sin(angles_par)])      # (2, 5)
    P_perp = np.array([np.cos(angles_perp), np.sin(angles_perp)])    # (2, 5)

    # Normalization — spacing scales the physical-space lattice constant
    P_par *= np.sqrt(2.0 / 5) * spacing
    P_perp *= np.sqrt(2.0 / 5)

    # Acceptance window radius in perpendicular space (decagonal window)
    # For a Penrose tiling, the acceptance domain is a regular decagon.
    # We approximate with a circle of appropriate radius.
    window_radius = np.sqrt(2.0 / 5) * (1 + 2 * np.cos(np.pi / 5))

    # Scan lattice points in Z^5 that could project into our domain
    max_extent = max(size[0], size[1]) + 3
    # Estimate needed range in Z^5
    n_max = int(np.ceil(max_extent * np.sqrt(5) / np.sqrt(2))) + 2

    # Vectorized approach: loop over first 2 indices, vectorize last 3
    coords = np.arange(-n_max, n_max + 1)
    n_c = len(coords)

    # Pre-build the 3D grid for indices 2,3,4
    g2, g3, g4 = np.meshgrid(coords, coords, coords, indexing='ij')
    inner_grid = np.stack([g2.ravel(), g3.ravel(), g4.ravel()], axis=1)  # (n_c^3, 3)

    # Partial projections for inner indices
    P_par_inner = P_par[:, 2:5]      # (2, 3)
    P_perp_inner = P_perp[:, 2:5]    # (2, 3)
    inner_par = inner_grid @ P_par_inner.T      # (n_c^3, 2)
    inner_perp = inner_grid @ P_perp_inner.T    # (n_c^3, 2)

    points = []
    for n0 in coords:
        for n1 in coords:
            # Partial projection from outer two indices
            outer_par = P_par[:, 0] * n0 + P_par[:, 1] * n1      # (2,)
            outer_perp = P_perp[:, 0] * n0 + P_perp[:, 1] * n1   # (2,)

            # Full projections
            full_par = inner_par + outer_par       # (n_c^3, 2)
            full_perp = inner_perp + outer_perp    # (n_c^3, 2)

            # Filter: perpendicular space within window
            perp_r2 = full_perp[:, 0]**2 + full_perp[:, 1]**2
            mask = perp_r2 <= window_radius**2

            # Filter: parallel space within domain
            mask &= (np.abs(full_par[:, 0]) <= size[0] + 2)
            mask &= (np.abs(full_par[:, 1]) <= size[1] + 2)

            if mask.any():
                points.append(full_par[mask])

    if len(points) == 0:
        raise RuntimeError("Penrose generator produced 0 points.")

    points = np.vstack(points)

    if len(points) < 10:
        raise RuntimeError(
            f"Penrose generator produced only {len(points)} points. "
            f"Increase n_max or check window_radius."
        )

    # Remove near-duplicate points (projection can place points very close)
    from scipy.spatial import cKDTree
    tree = cKDTree(points)
    pairs = tree.query_pairs(r=1e-8)
    remove = set()
    for i, j in pairs:
        remove.add(max(i, j))
    if remove:
        keep = sorted(set(range(len(points))) - remove)
        points = points[keep]

    DM = scipy.spatial.Delaunay(points)
    simplices = _filter_interior(DM.points, DM.simplices, size)

    return TriangulationResult(
        points=DM.points, simplices=simplices,
        topo_name='penrose', topo_class='quasicrystal',
    )


# ── Lieb lattice (decorated square) ──────────────────────────────────────────

def generate_lieb_lattice(size, spacing=1.0, k_soft_ratio=1e-3):
    """Lieb lattice — square lattice with atoms on edge midpoints.

    The Lieb lattice has 3 atoms per unit cell: one at the corner and
    two at the midpoints of the horizontal and vertical edges. This
    creates a mix of z=4 (corner) and z=2 (edge-midpoint) sites.

    Known for its flat phonon band and interesting mechanical properties.
    The structure is not fully triangulated: we Delaunay-triangulate
    and mark the Lieb bonds (corner-to-midpoint) as hard.

    Args:
        size: (sx, sy) half-extents.
        spacing: lattice constant (distance between corner sites).
        k_soft_ratio: rigidity of soft (fill-in) springs.
    """
    a1 = np.array([spacing, 0.0])
    a2 = np.array([0.0, spacing])

    # 3-atom basis: corner + midpoints of two edges
    basis = np.array([
        [0.0, 0.0],               # corner (A site)
        [spacing / 2, 0.0],       # horizontal edge midpoint (B site)
        [0.0, spacing / 2],       # vertical edge midpoint (C site)
    ])

    points, interior = _generate_lattice_points(size, a1, a2, basis)

    # Lieb bonds: corner to its 4 nearest midpoint neighbors
    # Bond distance = spacing / 2
    bond_dist = spacing / 2 * 1.05
    hard_edges = _build_edges_from_neighbor_distance(points, bond_dist)

    DM = scipy.spatial.Delaunay(points)
    simplices = _filter_interior(DM.points, DM.simplices, size)

    return TriangulationResult(
        points=DM.points,
        simplices=simplices,
        hard_edge_set=hard_edges,
        k_soft_ratio=k_soft_ratio,
        topo_name='lieb',
        topo_class='lieb',
    )


# ── Diamond lattice (centered square, 2-atom basis) ─────────────────────────

def generate_diamond_lattice(size, spacing=1.0, k_soft_ratio=1e-3):
    """2D diamond lattice — square lattice with a 2-atom basis.

    Atom A sits at the corner of a square unit cell and atom B at the
    cell center, each A bonded to its 4 nearest B neighbors along the
    diagonals. Every site has z=4. This is the 2D analog of the diamond
    cubic structure.

    The A-B diagonal bonds are hard; Delaunay fill-in edges are soft.

    Args:
        size: (sx, sy) half-extents.
        spacing: lattice constant (side of the square unit cell).
        k_soft_ratio: rigidity of soft (fill-in) springs.
    """
    a1 = np.array([spacing, 0.0])
    a2 = np.array([0.0, spacing])

    # 2-atom basis: corner + body-center
    basis = np.array([
        [0.0, 0.0],                       # A site (corner)
        [spacing / 2, spacing / 2],       # B site (center)
    ])

    points, interior = _generate_lattice_points(size, a1, a2, basis)

    # Diamond bonds: A-B along diagonals, distance = spacing * sqrt(2) / 2
    bond_dist = spacing * np.sqrt(2) / 2 * 1.05
    hard_edges = _build_edges_from_neighbor_distance(points, bond_dist)

    DM = scipy.spatial.Delaunay(points)
    simplices = _filter_interior(DM.points, DM.simplices, size)

    return TriangulationResult(
        points=DM.points,
        simplices=simplices,
        hard_edge_set=hard_edges,
        k_soft_ratio=k_soft_ratio,
        topo_name='diamond',
        topo_class='diamond',
    )


# ── Bond-diluted triangular lattice ─────────────────────────────────────────

def generate_bond_diluted(size, p_remove=0.2):
    """Triangular lattice with randomly removed bonds (bond percolation).

    Start from a perfect z=6 triangular lattice and randomly remove a
    fraction of bonds. This models structural damage, porosity, or
    the effect of fabrication defects. The remaining bonds are hard;
    removed bonds become soft (fill-in) springs.

    Near the percolation threshold (p ~ 0.35 for triangular), the
    structure develops interesting mechanical properties: soft modes,
    floppy regions, and potentially auxetic behavior.

    Args:
        size: (sx, sy) half-extents.
        p_remove: fraction of bonds to remove (0-1). Default 0.2.
    """
    # Generate a perfect triangular lattice
    tri = D2C.generate_cryratl_points(size=size, shape=(1, 1), orientation=0)

    # Collect all unique edges
    all_edges = set()
    for simplex in tri.simplices:
        for a, b in [(simplex[0], simplex[1]),
                     (simplex[0], simplex[2]),
                     (simplex[1], simplex[2])]:
            all_edges.add(frozenset((a, b)))

    # Randomly remove p_remove fraction of edges
    all_edges_list = list(all_edges)
    n_remove = int(round(p_remove * len(all_edges_list)))
    remove_idx = np.random.choice(len(all_edges_list), size=n_remove, replace=False)
    removed = {all_edges_list[i] for i in remove_idx}
    hard_edges = all_edges - removed

    return TriangulationResult(
        points=tri.points,
        simplices=tri.simplices,
        hard_edge_set=hard_edges,
        k_soft_ratio=1e-3,
        topo_name='bond_diluted',
        topo_class='bond_diluted',
    )


# ── Re-entrant (auxetic) honeycomb ──────────────────────────────────────────

def generate_reentrant_honeycomb(size, theta=np.radians(30), H=1.0, L=1.0,
                                  k_soft_ratio=1e-3):
    """Re-entrant honeycomb — the classic auxetic metamaterial structure.

    Geometry (Gibson & Ashby convention):
        - Vertical ribs of length H.
        - Angled struts of length L at angle theta from vertical.
        - Alternating narrow (joint) and wide (rib-endpoint) rows create
          hourglass/bowtie cells.
        - z=3 at every interior vertex.

    Construction:
        4 rows repeat vertically with period 2*(H + h):

          Row A: joint vertices (narrow row) at x = l, 3l, 5l, ...
          Row B: rib endpoints (wide row) at x = 0, 2l, 4l, ...
          Row C: rib endpoints (wide row) — ribs connect B to C
          Row D: joint vertices (narrow row) — struts connect to C

        Angled struts connect each joint to two rib endpoints in the
        adjacent wide row. Vertical ribs connect pairs of wide rows.

    Note: the true auxetic (negative Poisson's ratio) behaviour of
    re-entrant honeycombs requires bending/angular spring stiffness.
    With central-force springs only, nu remains positive, but the
    topology is still structurally distinct and valuable for training.

    Args:
        size: (sx, sy) half-extents.
        theta: re-entrant angle from vertical (radians). Default pi/6.
        H: vertical rib length. Default 1.0.
        L: angled strut length. Default 1.0.
        k_soft_ratio: rigidity of soft (fill-in) springs.
    """
    h = L * np.cos(theta)   # vertical projection of angled strut
    l = L * np.sin(theta)   # horizontal projection of angled strut

    cell_w = 2 * l                # horizontal period
    cell_h = 2 * (H + h)          # vertical period

    buf = 3
    nx = int(np.ceil((size[0] + buf) / cell_w)) + 2
    ny = int(np.ceil((size[1] + buf) / (cell_h / 2))) + 2

    point_map = {}
    points = []

    def add_point(x, y):
        key = (round(x, 8), round(y, 8))
        if key not in point_map:
            point_map[key] = len(points)
            points.append([x, y])
        return point_map[key]

    edges = set()

    for iy in range(-ny, ny + 1):
        for ix in range(-nx, nx + 1):
            y_base = iy * cell_h
            x_base = ix * cell_w

            # Row A (y_base): narrow row — re-entrant vertex
            pA = add_point(x_base + l, y_base)

            # Row B (y_base + h): wide row — top of downward-V from A
            pB_left = add_point(x_base, y_base + h)
            pB_right = add_point(x_base + cell_w, y_base + h)

            # Angled struts: A connects UP-LEFT and UP-RIGHT (V opens upward)
            edges.add(frozenset((pA, pB_left)))
            edges.add(frozenset((pA, pB_right)))

            # Row C (y_base + h + H): wide row — connected to B by vertical rib
            pC_left = add_point(x_base, y_base + h + H)
            pC_right = add_point(x_base + cell_w, y_base + h + H)

            # Vertical ribs: B to C (same x)
            edges.add(frozenset((pB_left, pC_left)))
            edges.add(frozenset((pB_right, pC_right)))

            # Row D (y_base + 2h + H): narrow row — re-entrant vertex
            pD = add_point(x_base + l, y_base + 2 * h + H)

            # Angled struts: D connects DOWN-LEFT and DOWN-RIGHT (V opens downward)
            edges.add(frozenset((pD, pC_left)))
            edges.add(frozenset((pD, pC_right)))

            # Vertical rib from D to next A above:
            # Next A is at y_base + cell_h = y_base + 2*(H+h)
            # D is at y_base + 2h + H
            # distance = 2(H+h) - (2h+H) = H
            pA_next = add_point(x_base + l, y_base + cell_h)
            edges.add(frozenset((pD, pA_next)))

    points = np.array(points)
    if len(points) < 4:
        raise RuntimeError("Re-entrant honeycomb produced too few points.")

    # Filter edges to bounding box
    in_bounds = ((np.abs(points[:, 0]) <= size[0] + buf) &
                 (np.abs(points[:, 1]) <= size[1] + buf))
    keep_set = set(np.where(in_bounds)[0].tolist())
    hard_edges = {e for e in edges if all(v in keep_set for v in e)}

    DM = scipy.spatial.Delaunay(points)
    simplices = _filter_interior(DM.points, DM.simplices, size)

    return TriangulationResult(
        points=DM.points,
        simplices=simplices,
        hard_edge_set=hard_edges,
        k_soft_ratio=k_soft_ratio,
        topo_name='reentrant',
        topo_class='reentrant',
    )


# ── Cairo pentagonal tiling ────────────────────────────────────────────────

def generate_cairo_pentagonal(size, spacing=1.0, k_soft_ratio=1e-3):
    """Cairo pentagonal tiling — dual of the snub square tiling.

    The Cairo tiling covers the plane with congruent convex pentagons.
    Each pentagon has four equal sides and one shorter side, arranged
    so that exactly four pentagons meet at each vertex.

    Construction: place vertices on a square grid with a 4-atom basis.
    The tiling has p4g wallpaper symmetry. Each interior vertex has
    z=3 or z=4 bonds.

    The pentagonal faces are non-triangular, so we Delaunay-triangulate
    the vertex set and mark the pentagon edges as hard structural bonds.

    Args:
        size: (sx, sy) half-extents.
        spacing: overall scale factor. Default 1.0.
        k_soft_ratio: rigidity of soft (fill-in) springs.
    """
    # The Cairo tiling can be constructed from a square lattice with
    # lattice vectors a1 = (2, 0)*s, a2 = (0, 2)*s and a 4-atom basis.
    #
    # The key parameter is t = (sqrt(3) - 1) / 2 ≈ 0.366, which gives
    # the offset of the interior vertices.
    t = (np.sqrt(3) - 1) / 2
    s = spacing

    a1 = np.array([2.0, 0.0]) * s
    a2 = np.array([0.0, 2.0]) * s

    # 4-atom basis within the unit cell:
    #   Two "cross" vertices that sit on the midpoints of cell edges
    #   Two "interior" vertices offset by t
    basis = np.array([
        [0.5 + t, 0.5],        # vertex A
        [0.5 - t, 0.5],        # vertex B
        [0.0,     0.5 + t],    # vertex C (on cell edge)
        [0.0,     0.5 - t],    # vertex D (on cell edge)
        [1.0,     0.5 + t],    # vertex E
        [1.0,     0.5 - t],    # vertex F
        [0.5,     0.0],        # vertex G (on cell edge)
        [0.5,     1.0],        # vertex H (on cell edge)
    ]) * s

    points, interior = _generate_lattice_points(size, a1, a2, basis)

    # Cairo pentagonal bond distance: nearest neighbors are at distance
    # s * sqrt((2t)^2 + 0^2) = 2*t*s for horizontal pairs, and
    # s * sqrt(t^2 + (0.5-t)^2) for diagonal pairs.
    # The shortest bonds connect adjacent basis atoms.
    # Use a cutoff that captures all pentagon edges but not diagonals.
    d_short = 2 * t * s                                    # ~0.732 * s
    d_diag = s * np.sqrt((0.5 - t)**2 + (0.5)**2)         # ~0.541 * s
    bond_dist = max(d_short, d_diag) * 1.08

    hard_edges = _build_edges_from_neighbor_distance(points, bond_dist)

    DM = scipy.spatial.Delaunay(points)
    simplices = _filter_interior(DM.points, DM.simplices, size)

    return TriangulationResult(
        points=DM.points,
        simplices=simplices,
        hard_edge_set=hard_edges,
        k_soft_ratio=k_soft_ratio,
        topo_name='cairo_pentagonal',
        topo_class='cairo',
    )


# ── Ammann-Beenker quasicrystal ──────────────────────────────────────────

def generate_ammann_beenker(size, spacing=3.0):
    """Ammann-Beenker (octagonal) quasicrystal -> Delaunay triangulation.

    Uses the cut-and-project method from a 4D hypercubic lattice onto
    a 2D plane with 8-fold symmetry. Only lattice points whose
    perpendicular-space projection falls within an octagonal acceptance
    window are kept.

    This produces a quasiperiodic point set with local 8-fold symmetry
    (tiles are squares and 45-degree rhombi) — the 2D analog of the
    3D icosahedral quasicrystal. All edges are hard (fully triangulated).

    Args:
        size: (sx, sy) half-extents of the interior region.
        spacing: scale factor for inter-point distance. Default 3.0 gives
                 a comparable point count to other topologies at size=(10,10).
    """
    # Projection matrices for Ammann-Beenker (8-fold) quasicrystal
    # We project from Z^4 using basis vectors at multiples of pi/4
    k = np.arange(4)
    angles_par = np.pi * k / 4       # 0, pi/4, pi/2, 3pi/4
    angles_perp = np.pi * (k + 2) / 4  # pi/2, 3pi/4, pi, 5pi/4

    P_par = np.array([np.cos(angles_par), np.sin(angles_par)])    # (2, 4)
    P_perp = np.array([np.cos(angles_perp), np.sin(angles_perp)])  # (2, 4)

    # Normalization
    P_par *= spacing / np.sqrt(2)
    P_perp *= 1.0 / np.sqrt(2)

    # Acceptance window: regular octagon in perpendicular space
    # For the Ammann-Beenker tiling the window is a regular octagon
    # with circumradius R = 1 + sqrt(2) (in normalized units).
    # We approximate with a circle.
    window_radius = (1 + np.sqrt(2)) / np.sqrt(2)

    # Determine range of Z^4 indices to scan
    max_extent = max(size[0], size[1]) + 3
    n_max = int(np.ceil(max_extent * 2 / spacing)) + 3

    coords = np.arange(-n_max, n_max + 1)

    # Vectorize: build full 4D grid in batches to manage memory
    # Loop over first 2 indices, vectorize last 2
    g2, g3 = np.meshgrid(coords, coords, indexing='ij')
    inner_grid = np.stack([g2.ravel(), g3.ravel()], axis=1)  # (n_c^2, 2)

    P_par_inner = P_par[:, 2:4]      # (2, 2)
    P_perp_inner = P_perp[:, 2:4]    # (2, 2)
    inner_par = inner_grid @ P_par_inner.T     # (n_c^2, 2)
    inner_perp = inner_grid @ P_perp_inner.T   # (n_c^2, 2)

    points = []
    for n0 in coords:
        for n1 in coords:
            outer_par = P_par[:, 0] * n0 + P_par[:, 1] * n1
            outer_perp = P_perp[:, 0] * n0 + P_perp[:, 1] * n1

            full_par = inner_par + outer_par
            full_perp = inner_perp + outer_perp

            # Filter: perpendicular space within octagonal window (circle approx)
            perp_r2 = full_perp[:, 0]**2 + full_perp[:, 1]**2
            mask = perp_r2 <= window_radius**2

            # Filter: parallel space within domain
            mask &= (np.abs(full_par[:, 0]) <= size[0] + 2)
            mask &= (np.abs(full_par[:, 1]) <= size[1] + 2)

            if mask.any():
                points.append(full_par[mask])

    if len(points) == 0:
        raise RuntimeError("Ammann-Beenker generator produced 0 points.")

    points = np.vstack(points)

    if len(points) < 10:
        raise RuntimeError(
            f"Ammann-Beenker generator produced only {len(points)} points. "
            f"Try decreasing spacing."
        )

    # Remove near-duplicate points
    from scipy.spatial import cKDTree
    tree = cKDTree(points)
    pairs = tree.query_pairs(r=1e-8)
    remove = set()
    for i, j in pairs:
        remove.add(max(i, j))
    if remove:
        keep = sorted(set(range(len(points))) - remove)
        points = points[keep]

    DM = scipy.spatial.Delaunay(points)
    simplices = _filter_interior(DM.points, DM.simplices, size)

    return TriangulationResult(
        points=DM.points, simplices=simplices,
        topo_name='ammann_beenker', topo_class='quasicrystal',
    )


# ═════════════════════════════════════════════════════════════════════════════
# Master topology generator
# ═════════════════════════════════════════════════════════════════════════════

# Registry of all topology generators
TOPOLOGY_GENERATORS = {
    # Category 1: Bravais lattice family (all 5 types + aniso hexagonal)
    #   Target: ~950 triangles at size=(10,10)
    'iso_crystal':       lambda size: generate_iso_crystal(size),           # 944
    'aniso_crystal':     lambda size: generate_aniso_crystal(size),         # 1154
    'rectangular':       lambda size: generate_rectangular_lattice(size, aspect=1.4),  # ~950
    'oblique':           lambda size: generate_oblique_lattice(size, a_len=0.9, b_len=1.05, angle_deg=70),  # ~950
    'foam_eta02':        lambda size: generate_foam(size, eta=0.2),         # 944
    'foam_eta045':       lambda size: generate_foam(size, eta=0.45),        # 944

    # Category 2: Fully random
    'poisson_delaunay':  lambda size: generate_poisson_delaunay(size),      # 912
    'blue_noise':        lambda size: generate_blue_noise(size, min_dist=0.73),  # ~950
    'clustered':         lambda size: generate_clustered(size, cluster_spacing=2.5, cluster_std=0.5, pts_per_cluster=7),
    'gradient_density':  lambda size: generate_gradient_density(size),      # 897

    # Category 3 & 4: Lattices with basis / non-triangulated + regularization
    'honeycomb':         lambda size: generate_honeycomb(size, spacing=1.70),   # ~950
    'kagome':            lambda size: generate_kagome(size, spacing=0.85),      # ~950
    'square_lattice':    lambda size: generate_square_lattice(size, spacing=0.92),  # ~950

    # Category 5: Physically motivated non-trivial topologies
    'penrose':           lambda size: generate_penrose(size),               # 987
    'lieb':              lambda size: generate_lieb_lattice(size, spacing=1.59),   # ~950
    'diamond':           lambda size: generate_diamond_lattice(size, spacing=1.31),  # ~950
    'bond_diluted':      lambda size: generate_bond_diluted(size),          # 944

    # Category 6: Additional quasicrystals
    'ammann_beenker':    lambda size: generate_ammann_beenker(size, spacing=13.0),   # ~928
}

TOPO_CLASSES = {
    'iso_crystal':       'crystal',
    'aniso_crystal':     'crystal',
    'rectangular':       'rectangular',
    'oblique':           'oblique',
    'foam_eta02':        'foam_02',
    'foam_eta045':       'foam_045',
    'poisson_delaunay':  'poisson',
    'blue_noise':        'blue_noise',
    'clustered':         'clustered',
    'gradient_density':  'gradient',
    'honeycomb':         'honeycomb',
    'kagome':            'kagome',
    'square_lattice':    'square',
    'penrose':           'quasicrystal',
    'lieb':              'lieb',
    'diamond':           'diamond',
    'bond_diluted':      'bond_diluted',
    'ammann_beenker':    'quasicrystal',
}


def generate_topology(name, size, seed=None):
    """Generate a named topology with optional random seed.

    Args:
        name: topology name (key into TOPOLOGY_GENERATORS).
        size: (sx, sy) half-extents for the mesh.
        seed: optional random seed (set before generation).

    Returns:
        TriangulationResult with topo_name and topo_class set.
    """
    if name not in TOPOLOGY_GENERATORS:
        raise ValueError(f"Unknown topology '{name}'. "
                         f"Available: {list(TOPOLOGY_GENERATORS.keys())}")

    if seed is not None:
        np.random.seed(seed)

    result = TOPOLOGY_GENERATORS[name](size)
    if not result.topo_name:
        result.topo_name = name
    if not result.topo_class:
        result.topo_class = TOPO_CLASSES.get(name, name)

    return result


def generate_all_topologies(size, seeds_per_random=3):
    """Generate all topologies in the catalog.

    Deterministic topologies (crystal, honeycomb, kagome, square) are
    generated once. Random topologies (foam, poisson, etc.) are generated
    with multiple seeds.

    Args:
        size: (sx, sy) half-extents.
        seeds_per_random: how many random seeds per stochastic topology.

    Returns:
        list of TriangulationResult.
    """
    results = []

    # Deterministic topologies
    deterministic = [
        'iso_crystal', 'aniso_crystal', 'rectangular', 'oblique',
        'honeycomb', 'kagome', 'square_lattice',
        'penrose', 'lieb', 'diamond',
        'ammann_beenker',
    ]
    for name in deterministic:
        results.append(generate_topology(name, size, seed=0))

    # Stochastic topologies
    stochastic = [
        'foam_eta02', 'foam_eta045',
        'poisson_delaunay', 'blue_noise',
        'clustered', 'gradient_density',
        'bond_diluted',
    ]
    for name in stochastic:
        for s in range(seeds_per_random):
            seed = 42 + s * 137
            result = generate_topology(name, size, seed=seed)
            result.topo_name = f"{name}_s{seed}"
            results.append(result)

    return results
