"""Mesh / topology generators for 2D spring networks.

NOTE (2026 cleanup): this module used to also contain the original NumPy mean-field
D2C (disc-to-continuum) homogenisation solver (`analyze_elastic_struct` and its
Woodbury helpers). That solver was a single-site mean field and is now superseded by
the *intrinsic* metric solve in `Phase 2/forward_solver_torch.py`
(`forward(method='intrinsic')`), which reproduces the PBC simulation. The old solver
was removed to avoid confusion — recover it from git history if ever needed.

What remains here are the geometry generators only: they build the perturbed
triangular-lattice ("foam") and crystal meshes as `scipy.spatial.Delaunay` objects.
They are still imported by `Phase 4/data/topology_generators.py` (as `D2C`) as the
base for the topology catalogue, and are consumed by the forward solver's
`from_triangulation(...)`.
"""

import numpy as np
import scipy as sp


def cut_to_size(points, size):
    points = points[points[::, 0] <= size[0]]
    points = points[points[::, 0] >= -size[0]]
    points = points[points[::, 1] <= size[1]]
    points = points[points[::, 1] >= -size[1]]
    return points


def generate_cryratl_points(size, shape, orientation):
    v1 = np.array([1, 0])
    v2 = np.array([shape[0] / 2, np.sqrt(3) / 2 * shape[1]])
    MaxPos = int(round(2 * (max(size) / min([np.sqrt(3) / 2 * shape[1], 1]))))
    RotationMat = np.array([[np.cos(orientation), np.sin(orientation)], [-np.sin(orientation), np.cos(orientation)]])
    point_pos = np.zeros((((2 * MaxPos) ** 2), 2))
    index = 0
    for n in range(-MaxPos, MaxPos):
        for m in range(-MaxPos, MaxPos):
            tempv = np.dot(RotationMat, n * v1 + m * v2)
            point_pos[index] = tempv
            index += 1
    DM = sp.spatial.Delaunay(cut_to_size(point_pos, (size[0] + 2, size[1] + 2)))
    DM.centroids = np.array([np.mean(DM.points[tri], 0) for tri in DM.simplices])
    DM.goods_bool = np.array([((abs(cent[0]) <= size[0]) and (abs(cent[1]) <= size[1])) for cent in DM.centroids])
    DM.good_idxs = np.where(DM.goods_bool == True)
    DM.all_simplices = DM.simplices
    DM.simplices = DM.simplices[DM.good_idxs]
    return DM


def generate_foam_points2(size, eta):
    v1 = np.array([1, 0])
    v2 = np.array([1 / 2, np.sqrt(3) / 2])
    MaxPos = int(round(2 * (max(size) / min([np.sqrt(3) / 2, 1]))))
    point_pos = np.zeros((((2 * MaxPos) ** 2), 2))
    index = 0
    for n in range(-MaxPos, MaxPos):
        for m in range(-MaxPos, MaxPos):
            theta = 2 * np.pi + np.random.rand()
            tempv = n * v1 + m * v2 + np.array([np.cos(theta), np.sin(theta)])
            point_pos[index] = tempv
            index += 1
    DM = sp.spatial.Delaunay(cut_to_size(point_pos, (size[0] + 2, size[1] + 2)))
    DM.centroids = np.array([np.mean(DM.points[tri], 0) for tri in DM.simplices])
    DM.goods_bool = np.array([((abs(cent[0]) <= size[0]) and (abs(cent[1]) <= size[1])) for cent in DM.centroids])
    DM.good_idxs = np.where(DM.goods_bool == True)
    DM.all_simplices = DM.simplices
    DM.simplices = DM.simplices[DM.good_idxs]
    return DM


def generate_foam_points(size, eta):
    Max = max(size)
    DM = generate_cryratl_points(size, (1, 1), 0)
    counter = 0
    for point in DM.points:
        theta = 2 * np.pi * np.random.rand()
        DM.points[counter] = point + eta * np.array([np.cos(theta), np.sin(theta)])
        counter += 1
    return DM


def generate_foam_distort_first(size, eta, trim_frac=0.85):
    """Distort vertex positions first, then Delaunay-triangulate, then trim edges.

    Order:
      1. Place vertices on a regular hexagonal lattice (padded domain).
      2. Perturb every vertex by eta * [cos theta, sin theta], theta uniform in [0, 2pi).
      3. Delaunay-triangulate the perturbed positions.
      4. Keep triangles whose centroid lies within trim_frac of the bounding box.

    This differs from generate_foam_points (which triangulates first then distorts):
    here the topology itself is generated from the disordered point cloud.
    """
    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.5, np.sqrt(3) / 2])
    MaxPos = int(round(2 * (max(size) / min([np.sqrt(3) / 2, 1]))))

    point_pos = []
    for n in range(-MaxPos, MaxPos):
        for m in range(-MaxPos, MaxPos):
            theta = 2 * np.pi * np.random.rand()
            point_pos.append(n * v1 + m * v2 + eta * np.array([np.cos(theta), np.sin(theta)]))
    point_pos = np.array(point_pos)

    # Restrict to padded domain before triangulating
    point_pos = cut_to_size(point_pos, (size[0] + 2, size[1] + 2))

    DM = sp.spatial.Delaunay(point_pos)
    centroids = np.array([np.mean(DM.points[tri], 0) for tri in DM.simplices])

    # Trim edge triangles: keep only those within trim_frac of the centroid bbox
    cx = centroids[:, 0]
    cy = centroids[:, 1]
    xc, yc = (cx.max() + cx.min()) / 2, (cy.max() + cy.min()) / 2
    hw = (cx.max() - cx.min()) / 2
    hh = (cy.max() - cy.min()) / 2
    mask = (np.abs(cx - xc) <= trim_frac * hw) & (np.abs(cy - yc) <= trim_frac * hh)

    DM.centroids = centroids
    DM.goods_bool = mask
    DM.good_idxs = np.where(mask)
    DM.all_simplices = DM.simplices
    DM.simplices = DM.simplices[mask]
    return DM


def find_triangle_edges(triangle):
    return np.array([(a, b) for idx, a in enumerate(triangle) for b in triangle[idx + 1:]])


def add_edges_to_triangulation(triangulation):
    triangulation.edges = np.array([find_triangle_edges(tri) for tri in triangulation.simplices])
    triangulation.rigidities = [[] for tri in triangulation.simplices]
    triangulation.rest_lenghts = [[] for tri in triangulation.simplices]
    return 0
