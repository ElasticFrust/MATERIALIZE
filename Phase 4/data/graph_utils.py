"""Graph utilities for converting triangulations to PyG (PyTorch Geometric) Data objects.

This module bridges the gap between the forward solver's triangle-based representation
and the GNN's graph-based representation. The key transformation is:

    Solver world                          GNN world
    ───────────                           ─────────
    (N_tri, 3) simplices                  (2, 2*E_unique) edge_index (COO)
    (N_tri, 3) rigidities per triangle    (2*E_unique, 5) edge_attr per unique edge
    (N_tri, 3) rest_lengths per triangle  (M, 8) node features
    scalar: nu, Young's modulus           scalar labels: y_nu, y_young, y_tensor

The main challenge is **edge deduplication**: each interior edge is shared by two
triangles, so the solver stores it twice. The GNN needs each physical edge once
(or twice for bidirectional message passing, but with the SAME features).

Pipeline:
  1. deduplicate_edges(): simplices -> unique edge list + tri_edge_map
  2. solver_params_to_unique_edges(): per-triangle params -> per-edge params (averaging shared edges)
  3. compute_edge_attr(): edge params -> 5-dim feature vector [k, l0, l_actual, log(k/l0^2), is_real]
  4. init_node_features(): position + aggregated edge stats -> 8-dim node features
  5. triangulation_to_pyg_data(): full pipeline assembling a PyG Data object

Handles:
  - Edge deduplication (solver's per-triangle format -> unique undirected edges)
  - Node feature initialization from geometry + aggregated edge info
  - Hard/soft edge distinction via is_real flag and edge_mask
  - Geometric edge features for the CVAE decoder (k-free, l0-free)
  - Full triangulation-to-PyG conversion pipeline
"""

import numpy as np
import torch
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Edge deduplication
# ─────────────────────────────────────────────────────────────────────────────
#
# Why this is needed:
#   The forward solver works with TRIANGLES. Each triangle has 3 edges, stored
#   as (N_tri, 3) arrays. But interior edges are shared between two triangles,
#   meaning the same physical edge appears twice in the solver's representation.
#
#   The GNN works with GRAPHS (edges are unique). We need a canonical mapping
#   from the solver's duplicated edges to a unique edge list, and a way to
#   convert back (unique -> solver format) for CVAE output.

def deduplicate_edges(simplices):
    """Convert per-triangle edges to unique undirected edge list.

    The forward solver stores edges as (N_tri, 3, 2) -- per-triangle,
    per-edge, node-index pairs. Shared edges between triangles are duplicated.
    This function builds a unique edge list and a mapping from
    (triangle, local_edge_idx) -> unique_edge_idx.

    The local edge indexing convention matches the solver:
      edge 0: (tri[0], tri[1])
      edge 1: (tri[0], tri[2])
      edge 2: (tri[1], tri[2])

    Edges are canonicalized by sorting: (min(a,b), max(a,b)) to ensure
    the same physical edge maps to the same unique index regardless of
    which triangle references it.

    Args:
        simplices: (N_tri, 3) int array of triangle vertex indices.

    Returns:
        edge_index: (2, 2*E_unique) int64 tensor -- bidirectional COO edge list.
                    The first E_unique entries are (i->j), the next E_unique are (j->i).
                    PyG expects bidirectional edges for undirected message passing.
        tri_edge_map: (N_tri, 3) int array -- maps (triangle_idx, local_edge_idx) to
                      unique edge index (0..E_unique-1). Used to convert between
                      per-triangle and per-edge representations.
        unique_edges: (E_unique, 2) int array -- the canonical (i,j) pairs with i<j.
    """
    edge_to_idx = {}
    tri_edge_map = np.zeros((len(simplices), 3), dtype=np.int64)
    unique_edges = []

    for t, tri in enumerate(simplices):
        # Edges: (tri[0],tri[1]), (tri[0],tri[2]), (tri[1],tri[2])
        local_edges = [(tri[0], tri[1]), (tri[0], tri[2]), (tri[1], tri[2])]
        for e, (a, b) in enumerate(local_edges):
            key = (min(a, b), max(a, b))
            if key not in edge_to_idx:
                edge_to_idx[key] = len(unique_edges)
                unique_edges.append(key)
            tri_edge_map[t, e] = edge_to_idx[key]

    unique_edges = np.array(unique_edges)  # (E_unique, 2)

    # Build bidirectional edge_index for PyG (undirected: both i->j and j->i)
    src = np.concatenate([unique_edges[:, 0], unique_edges[:, 1]])
    dst = np.concatenate([unique_edges[:, 1], unique_edges[:, 0]])
    edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)

    return edge_index, tri_edge_map, unique_edges


# ─────────────────────────────────────────────────────────────────────────────
# Edge feature computation
# ─────────────────────────────────────────────────────────────────────────────
#
# Edge features are the GNN's primary source of physics information. The
# feature vector is designed to capture how each spring contributes to the
# elastic tensor:
#
#   [k, l0, l_actual, log(k/l0^2), is_real]
#    │   │      │         │            │
#    │   │      │         │            └─ 1.0 for structural bonds, 0.0 for soft regularization
#    │   │      │         └─ The key physics factor: each edge's contribution to the
#    │   │      │            elastic tensor scales as k/l0^2. Log-space is natural because
#    │   │      │            rigidities span orders of magnitude.
#    │   │      └─ Current geometric length (from node positions). Differs from l0 when
#    │   │         the mesh is pre-stressed (l_actual != l0).
#    │   └─ Rest length (natural spring length). When l_actual > l0, spring is stretched.
#    └─ Spring constant (rigidity). Higher k = stiffer spring.

def compute_edge_attr(points, unique_edges, rigidities_per_edge,
                      rest_lengths_per_edge, is_hard_per_edge):
    """Compute the 5-dimensional edge feature vectors for the GNN.

    The feature design mirrors the forward solver's physics: the elastic
    tensor contribution of each edge is proportional to k/l0^2 * (unit_vec ⊗ unit_vec).
    By including log(k/l0^2) as a feature, the GNN can directly learn this
    relationship without having to "discover" the k/l0^2 combination.

    Args:
        points: (M, 2) node positions.
        unique_edges: (E_unique, 2) node-index pairs.
        rigidities_per_edge: (E_unique,) spring constants k.
        rest_lengths_per_edge: (E_unique,) rest lengths l0.
        is_hard_per_edge: (E_unique,) bool -- True for structural bonds,
                          False for soft regularization edges.

    Returns:
        edge_attr: (2*E_unique, 5) features for bidirectional edges.
                   [k, l0, l_actual, log(k/l0^2), is_real]
                   First E_unique rows = forward direction (i->j).
                   Next E_unique rows = reverse direction (j->i), same features.
    """
    # Actual edge lengths from current node positions
    vecs = points[unique_edges[:, 0]] - points[unique_edges[:, 1]]
    l_actual = np.sqrt(np.sum(vecs ** 2, axis=1))

    k = rigidities_per_edge
    l0 = rest_lengths_per_edge

    # Physics factor: log(k / l0^2). This is the dominant term in the elastic tensor.
    # Clamped to avoid -inf for soft edges with very small k.
    with np.errstate(divide='ignore', invalid='ignore'):
        log_factor = np.log(np.maximum(k / (l0 ** 2), 1e-20))

    # Binary flag: 1.0 for hard (designable) edges, 0.0 for soft (frozen) edges.
    # The GNN uses this to learn different message-passing weights for real vs
    # regularization edges without needing separate code paths.
    is_real = is_hard_per_edge.astype(np.float32)

    # Stack features: [k, l0, l_actual, log(k/l0^2), is_real]
    attr = np.stack([k, l0, l_actual, log_factor, is_real], axis=1).astype(np.float32)

    # Duplicate for bidirectional edges (edge features are symmetric for undirected springs)
    edge_attr = np.concatenate([attr, attr], axis=0)

    return torch.tensor(edge_attr, dtype=torch.float32)


def compute_geometric_edge_attr(points, unique_edges, is_hard_per_edge):
    """Compute geometric-only edge features (for the CVAE decoder).

    The CVAE decoder generates k and l0 — so it must NOT see them as inputs.
    Instead, it receives only fixed geometric features of the mesh topology:
    edge length, direction vector, and angle. These are sufficient for the
    decoder to understand the spatial structure.

    Unlike edge_attr, these features ARE direction-dependent: the reverse
    edge (j->i) has flipped dx, dy and shifted angle. This lets the decoder
    distinguish "left neighbor" from "right neighbor".

    Returns:
        edge_geom: (2*E_unique, 4) [l_actual, dx, dy, angle]
                   Forward edges (i->j): [l, dx, dy, atan2(dy,dx)]
                   Reverse edges (j->i): [l, -dx, -dy, angle+pi]
    """
    vecs = points[unique_edges[:, 0]] - points[unique_edges[:, 1]]
    l_actual = np.sqrt(np.sum(vecs ** 2, axis=1))
    dx = vecs[:, 0]
    dy = vecs[:, 1]
    angle = np.arctan2(dy, dx)

    attr = np.stack([l_actual, dx, dy, angle], axis=1).astype(np.float32)

    # For reverse edges, flip dx and dy, shift angle by pi
    attr_rev = np.stack([l_actual, -dx, -dy, angle + np.pi], axis=1).astype(np.float32)

    edge_geom = np.concatenate([attr, attr_rev], axis=0)
    return torch.tensor(edge_geom, dtype=torch.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Node feature initialization
# ─────────────────────────────────────────────────────────────────────────────
#
# Node features summarize each vertex's local environment:
#   - Position (normalized): where is this node in the mesh?
#   - Degree: how many edges connect to this node (= coordination number)?
#   - Edge statistics (mean/std of k, l, log_factor): what do the local
#     springs look like on average?
#
# These features give the GNN a "warm start" — even before message passing,
# each node already knows its local environment. Without these, the first
# message-passing layer would have to aggregate edge info from scratch.

def init_node_features(pos, edge_index, edge_attr, n_nodes):
    """Build 8-dimensional node features from position + aggregated edge statistics.

    The feature vector for each node summarizes its geometric position and
    the statistics of its incident edges. This gives the GNN useful per-node
    information even at layer 0 (before any message passing).

    Feature breakdown:
      [0] x_norm:       normalized x-position (zero-mean, unit-std)
      [1] y_norm:       normalized y-position
      [2] degree:       number of incident edges (= coordination number z)
      [3] mean_k:       mean rigidity of incident edges
      [4] std_k:        std of rigidity (captures heterogeneity)
      [5] mean_l:       mean edge length (captures local mesh scale)
      [6] std_l:        std of edge length (captures local irregularity)
      [7] mean_log_f:   mean of log(k/l0^2) (dominant physics factor)

    Args:
        pos: (M, 2) tensor of node positions.
        edge_index: (2, 2*E_unique) bidirectional COO edges.
        edge_attr: (2*E_unique, 5) edge features [k, l0, l_actual, log_factor, is_real].
        n_nodes: total number of nodes M.

    Returns:
        x: (M, 8) node features.
    """
    # Normalize positions to zero mean, unit std across the entire mesh.
    # This makes the GNN position-invariant: shifting or scaling the mesh
    # doesn't change the node features.
    pos_mean = pos.mean(dim=0)
    pos_std = pos.std(dim=0).clamp(min=1e-8)
    pos_norm = (pos - pos_mean) / pos_std

    # Degree from edge_index: count incident edges per node using scatter_add.
    # Since edge_index is bidirectional, each undirected edge contributes 1
    # to each endpoint's degree.
    src = edge_index[0]
    degree = torch.zeros(n_nodes, dtype=torch.float32)
    degree.scatter_add_(0, src, torch.ones(src.shape[0], dtype=torch.float32))

    # Aggregate edge features to source nodes using scatter_add.
    # For each node, we compute mean and std of its incident edges' k and l values.
    k = edge_attr[:, 0]       # rigidity
    l = edge_attr[:, 2]       # l_actual (geometric length)
    f = edge_attr[:, 3]       # log(k/l0^2) (physics factor)

    # Mean of k at each node: sum_k / degree
    mean_k = torch.zeros(n_nodes, dtype=torch.float32)
    mean_k.scatter_add_(0, src, k)
    safe_degree = degree.clamp(min=1)  # avoid division by zero for isolated nodes
    mean_k = mean_k / safe_degree

    # Std of k at each node: sqrt(E[k^2] - E[k]^2), via the variance formula
    k_sq = torch.zeros(n_nodes, dtype=torch.float32)
    k_sq.scatter_add_(0, src, k ** 2)
    k_sq = k_sq / safe_degree
    std_k = (k_sq - mean_k ** 2).clamp(min=0).sqrt()  # clamp avoids sqrt of negative due to float error

    # Mean of l_actual at each node
    mean_l = torch.zeros(n_nodes, dtype=torch.float32)
    mean_l.scatter_add_(0, src, l)
    mean_l = mean_l / safe_degree

    # Std of l_actual
    l_sq = torch.zeros(n_nodes, dtype=torch.float32)
    l_sq.scatter_add_(0, src, l ** 2)
    l_sq = l_sq / safe_degree
    std_l = (l_sq - mean_l ** 2).clamp(min=0).sqrt()

    # Mean of log_factor
    mean_f = torch.zeros(n_nodes, dtype=torch.float32)
    mean_f.scatter_add_(0, src, f)
    mean_f = mean_f / safe_degree

    x = torch.stack([
        pos_norm[:, 0], pos_norm[:, 1],
        degree,
        mean_k, std_k,
        mean_l, std_l,
        mean_f,
    ], dim=-1)

    return x


# ─────────────────────────────────────────────────────────────────────────────
# Per-edge parameter extraction from solver format
# ─────────────────────────────────────────────────────────────────────────────
#
# The solver stores parameters per-triangle (N_tri, 3), but the GNN stores
# them per-unique-edge (E_unique,). These two functions convert between the
# representations using tri_edge_map as the bridge.
#
# When an edge is shared by two triangles, both triangles should have the
# same k and l0 for that edge (it's the same physical spring). We take the
# mean as a safety measure in case of slight numerical differences.

def solver_params_to_unique_edges(rigidities, rest_lengths, tri_edge_map):
    """Convert per-triangle (N_tri, 3) params to per-unique-edge (E_unique,) params.

    Interior edges appear in two triangles. We average the two values (which
    should be identical for well-formed inputs). Boundary edges appear once.

    Args:
        rigidities: (N_tri, 3) per-triangle per-edge rigidities.
        rest_lengths: (N_tri, 3) per-triangle per-edge rest lengths.
        tri_edge_map: (N_tri, 3) mapping from (triangle, local_edge) -> unique_edge_idx.

    Returns:
        k_per_edge: (E_unique,) rigidities per unique edge.
        l0_per_edge: (E_unique,) rest lengths per unique edge.
    """
    n_unique = tri_edge_map.max() + 1
    k_sum = np.zeros(n_unique)
    l0_sum = np.zeros(n_unique)
    count = np.zeros(n_unique)

    for t in range(len(rigidities)):
        for e in range(3):
            eidx = tri_edge_map[t, e]
            k_sum[eidx] += rigidities[t, e]
            l0_sum[eidx] += rest_lengths[t, e]
            count[eidx] += 1

    count = np.maximum(count, 1)
    return k_sum / count, l0_sum / count


def unique_edge_params_to_solver(k_per_edge, l0_per_edge, tri_edge_map):
    """Convert per-unique-edge (E_unique,) params back to solver format (N_tri, 3).

    This is the inverse of solver_params_to_unique_edges. Used when the CVAE
    decoder produces per-edge parameters and we need to run the forward solver
    to compute the resulting Poisson ratio (e.g., for the physics loss or for
    final validation).

    Args:
        k_per_edge: (E_unique,) rigidities per unique edge.
        l0_per_edge: (E_unique,) rest lengths per unique edge.
        tri_edge_map: (N_tri, 3) mapping from (triangle, local_edge) -> unique_edge_idx.

    Returns:
        rigidities: (N_tri, 3) per-triangle per-edge rigidities.
        rest_lengths: (N_tri, 3) per-triangle per-edge rest lengths.
    """
    n_tri = len(tri_edge_map)
    rigidities = np.zeros((n_tri, 3))
    rest_lengths = np.zeros((n_tri, 3))

    for t in range(n_tri):
        for e in range(3):
            eidx = tri_edge_map[t, e]
            rigidities[t, e] = k_per_edge[eidx]
            rest_lengths[t, e] = l0_per_edge[eidx]

    return rigidities, rest_lengths


# ─────────────────────────────────────────────────────────────────────────────
# Full conversion pipeline: TriangulationResult -> PyG Data
# ─────────────────────────────────────────────────────────────────────────────
#
# This is the main entry point called by generate_dataset.py for each sample.
# It orchestrates all the steps above into a single function call.

def triangulation_to_pyg_data(tri_result, rigidities=None, rest_lengths=None,
                               labels=None):
    """Convert a TriangulationResult to a PyG Data object ready for GNN training.

    This is the central conversion function that takes a mesh (from any topology
    generator) plus its edge parameters and labels, and produces a fully-featured
    PyG Data object. The resulting Data object contains everything needed for
    both GNN forward prediction and CVAE inverse design:

    Data fields:
      x:             (M, 8)           node features [pos, degree, edge_stats]
      pos:           (M, 2)           node positions
      edge_index:    (2, 2*E_unique)  bidirectional edge list (COO format)
      edge_attr:     (2*E_unique, 5)  edge features [k, l0, l_actual, log_factor, is_real]
      edge_geom:     (2*E_unique, 4)  geometric edge features [l, dx, dy, angle]
      edge_mask:     (2*E_unique,)    bool: True=designable, False=frozen/soft
      tri_edge_map:  (N_tri, 3)       maps back to solver format
      simplices:     (N_tri, 3)       triangle vertex indices
      y_nu:          scalar           Poisson's ratio (label)
      y_young:       scalar           Young's modulus (label)
      y_tensor:      (6,)             full elastic tensor (label)

    Args:
        tri_result: TriangulationResult from topology_generators.
        rigidities: (N_tri, 3) per-triangle rigidities. If None, uses defaults
                    (1.0 for hard edges, k_soft_ratio for soft edges).
        rest_lengths: (N_tri, 3) per-triangle rest lengths. If None, uses actual
                      edge lengths from the geometry (zero pre-stress).
        labels: optional dict with 'poisson', 'young', 'elastic_tensor' values
                computed by the forward solver.

    Returns:
        torch_geometric.data.Data object.
    """
    try:
        from torch_geometric.data import Data
    except ImportError:
        raise ImportError("torch_geometric is required. Install with: "
                          "pip install torch-geometric")

    points = tri_result.points
    simplices = tri_result.simplices

    # ── Step 1: Compute actual edge lengths from geometry ──
    # These serve as default rest lengths (l0 = l_actual means zero pre-stress).
    # Edge ordering matches the solver convention: edge0=(v0,v1), edge1=(v0,v2), edge2=(v1,v2).
    local_edges = np.array([
        [(tri[0], tri[1]), (tri[0], tri[2]), (tri[1], tri[2])]
        for tri in simplices
    ])  # (N_tri, 3, 2)

    vecs = points[local_edges[:, :, 0]] - points[local_edges[:, :, 1]]
    actual_lengths = np.sqrt(np.sum(vecs ** 2, axis=2))  # (N_tri, 3)

    if rigidities is None:
        rigidities = tri_result.get_default_rigidities()  # 1.0 hard, k_soft soft
    if rest_lengths is None:
        rest_lengths = actual_lengths.copy()  # zero pre-stress

    # ── Step 2: Deduplicate edges (per-triangle -> per-unique-edge) ──
    edge_index, tri_edge_map, unique_edges = deduplicate_edges(simplices)

    # ── Step 3: Convert solver params to per-unique-edge params ──
    # Shared edges get averaged (should be identical for well-formed inputs).
    k_per_edge, l0_per_edge = solver_params_to_unique_edges(
        rigidities, rest_lengths, tri_edge_map
    )

    # ── Step 4: Build hard/soft edge mask ──
    # For fully triangulated meshes (hard_edge_set is None), all edges are hard.
    # For regularized meshes, only original lattice bonds are hard.
    n_unique = len(unique_edges)
    is_hard = np.ones(n_unique, dtype=bool)
    if tri_result.hard_edge_set is not None:
        for eidx, (a, b) in enumerate(unique_edges):
            is_hard[eidx] = tri_result.edge_is_hard(a, b)

    # ── Step 5: Compute edge feature vectors ──
    edge_attr = compute_edge_attr(
        points, unique_edges, k_per_edge, l0_per_edge, is_hard
    )

    # Geometric edge features (for CVAE decoder — no k, l0 information)
    edge_geom = compute_geometric_edge_attr(points, unique_edges, is_hard)

    # ── Step 6: Compute node features ──
    pos = torch.tensor(points, dtype=torch.float32)
    n_nodes = len(points)
    x = init_node_features(pos, edge_index, edge_attr, n_nodes)

    # ── Step 7: Assemble the PyG Data object ──
    data = Data(
        x=x,                    # (M, 8) node features
        pos=pos,                # (M, 2) positions
        edge_index=edge_index,  # (2, 2*E) bidirectional COO
        edge_attr=edge_attr,    # (2*E, 5) full edge features
        edge_geom=edge_geom,    # (2*E, 4) geometric-only features
    )

    # Edge mask: True = designable (hard), False = frozen (soft).
    # Used by CVAE to know which edges to generate parameters for.
    # Bidirectional: duplicate mask for both directions.
    edge_mask = np.concatenate([is_hard, is_hard])
    data.edge_mask = torch.tensor(edge_mask, dtype=torch.bool)

    # Mapping back to solver format — needed to run the forward solver
    # on CVAE-generated edge parameters.
    data.tri_edge_map = torch.tensor(tri_edge_map, dtype=torch.long)
    data.simplices = torch.tensor(simplices, dtype=torch.long)
    data.n_unique_edges = n_unique

    # Topology metadata (for per-topology evaluation breakdown)
    data.topo_name = tri_result.topo_name
    data.topo_class = tri_result.topo_class
    data.k_soft_ratio = tri_result.k_soft_ratio

    # Labels from the forward solver
    if labels is not None:
        if 'poisson' in labels:
            data.y_nu = torch.tensor(labels['poisson'], dtype=torch.float32)
        if 'young' in labels:
            data.y_young = torch.tensor(labels['young'], dtype=torch.float32)
        if 'elastic_tensor' in labels:
            data.y_tensor = torch.tensor(labels['elastic_tensor'], dtype=torch.float32)

    return data
