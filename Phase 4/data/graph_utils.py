"""Graph utilities for converting triangulations to PyG Data objects.

Handles:
  - Edge deduplication (solver's (N_tri, 3, 2) -> unique undirected edges)
  - Node feature initialization from geometry + aggregated edge info
  - Full triangulation-to-PyG conversion pipeline
"""

import numpy as np
import torch
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Edge deduplication
# ─────────────────────────────────────────────────────────────────────────────

def deduplicate_edges(simplices):
    """Convert per-triangle edges to unique undirected edge list.

    The forward solver stores edges as (N_tri, 3, 2) -- per-triangle,
    per-edge, node-index pairs. Shared edges between triangles are duplicated.
    This function builds a unique edge list and a mapping from
    (triangle, local_edge_idx) -> unique_edge_idx.

    Args:
        simplices: (N_tri, 3) int array of triangle vertex indices.

    Returns:
        edge_index: (2, E_unique) int64 tensor -- unique undirected edges in COO.
                    Both directions included (i->j and j->i).
        tri_edge_map: (N_tri, 3) int array -- maps (triangle, local_edge) to
                      unique edge index (into the UNDIRECTED list, i.e. indices
                      into the first E_unique entries before bidirectional expansion).
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

def compute_edge_attr(points, unique_edges, rigidities_per_edge,
                      rest_lengths_per_edge, is_hard_per_edge):
    """Compute edge feature vectors for the unique edge list.

    Args:
        points: (M, 2) node positions.
        unique_edges: (E_unique, 2) node-index pairs.
        rigidities_per_edge: (E_unique,) spring constants.
        rest_lengths_per_edge: (E_unique,) rest lengths.
        is_hard_per_edge: (E_unique,) bool -- True for structural bonds.

    Returns:
        edge_attr: (2*E_unique, 5) features for bidirectional edges.
                   [k, l0, l_actual, log(k/l0^2), is_real]
    """
    # Actual edge lengths
    vecs = points[unique_edges[:, 0]] - points[unique_edges[:, 1]]
    l_actual = np.sqrt(np.sum(vecs ** 2, axis=1))

    k = rigidities_per_edge
    l0 = rest_lengths_per_edge

    # Physics factor: log(k / l0^2), clamped to avoid -inf
    with np.errstate(divide='ignore', invalid='ignore'):
        log_factor = np.log(np.maximum(k / (l0 ** 2), 1e-20))

    is_real = is_hard_per_edge.astype(np.float32)

    # Stack features: [k, l0, l_actual, log(k/l0^2), is_real]
    attr = np.stack([k, l0, l_actual, log_factor, is_real], axis=1).astype(np.float32)

    # Duplicate for bidirectional edges (same features for both directions)
    edge_attr = np.concatenate([attr, attr], axis=0)

    return torch.tensor(edge_attr, dtype=torch.float32)


def compute_geometric_edge_attr(points, unique_edges, is_hard_per_edge):
    """Compute geometric-only edge features (for CVAE decoder).

    These features don't include k or l0 -- only fixed geometry.

    Returns:
        edge_geom: (2*E_unique, 4) [l_actual, dx, dy, angle]
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

def init_node_features(pos, edge_index, edge_attr, n_nodes):
    """Build node features from position + aggregated edge information.

    Args:
        pos: (M, 2) tensor of node positions.
        edge_index: (2, 2*E_unique) bidirectional COO edges.
        edge_attr: (2*E_unique, 5) edge features [k, l0, l_actual, log_factor, is_real].
        n_nodes: total number of nodes M.

    Returns:
        x: (M, 8) node features:
           [x_norm, y_norm, degree, mean_k, std_k, mean_l, std_l, mean_log_factor]
    """
    # Normalize positions to zero mean, unit std
    pos_mean = pos.mean(dim=0)
    pos_std = pos.std(dim=0).clamp(min=1e-8)
    pos_norm = (pos - pos_mean) / pos_std

    # Degree from edge_index
    src = edge_index[0]
    degree = torch.zeros(n_nodes, dtype=torch.float32)
    degree.scatter_add_(0, src, torch.ones(src.shape[0], dtype=torch.float32))

    # Aggregate edge features to target nodes
    k = edge_attr[:, 0]
    l = edge_attr[:, 2]  # l_actual
    f = edge_attr[:, 3]  # log(k/l0^2)

    # Mean of k at each node
    mean_k = torch.zeros(n_nodes, dtype=torch.float32)
    mean_k.scatter_add_(0, src, k)
    safe_degree = degree.clamp(min=1)
    mean_k = mean_k / safe_degree

    # Std of k at each node
    k_sq = torch.zeros(n_nodes, dtype=torch.float32)
    k_sq.scatter_add_(0, src, k ** 2)
    k_sq = k_sq / safe_degree
    std_k = (k_sq - mean_k ** 2).clamp(min=0).sqrt()

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

def solver_params_to_unique_edges(rigidities, rest_lengths, tri_edge_map):
    """Convert per-triangle params to per-unique-edge params.

    When an edge is shared by multiple triangles, take the mean.

    Args:
        rigidities: (N_tri, 3) per-triangle per-edge rigidities.
        rest_lengths: (N_tri, 3) per-triangle per-edge rest lengths.
        tri_edge_map: (N_tri, 3) mapping to unique edge indices.

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
    """Convert per-unique-edge params back to solver format (N_tri, 3).

    Args:
        k_per_edge: (E_unique,) rigidities.
        l0_per_edge: (E_unique,) rest lengths.
        tri_edge_map: (N_tri, 3) mapping.

    Returns:
        rigidities: (N_tri, 3)
        rest_lengths: (N_tri, 3)
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
# Full conversion pipeline
# ─────────────────────────────────────────────────────────────────────────────

def triangulation_to_pyg_data(tri_result, rigidities=None, rest_lengths=None,
                               labels=None):
    """Convert a TriangulationResult to a PyG Data object.

    Args:
        tri_result: TriangulationResult from topology_generators.
        rigidities: (N_tri, 3) per-triangle rigidities. If None, uses defaults.
        rest_lengths: (N_tri, 3) per-triangle rest lengths. If None, uses actual.
        labels: optional dict with 'poisson', 'young', 'elastic_tensor' values.

    Returns:
        torch_geometric.data.Data object ready for GNN training.
    """
    try:
        from torch_geometric.data import Data
    except ImportError:
        raise ImportError("torch_geometric is required. Install with: "
                          "pip install torch-geometric")

    points = tri_result.points
    simplices = tri_result.simplices

    # Compute actual edge lengths for default rest lengths
    # Edges: (N_tri, 3, 2) node pairs
    local_edges = np.array([
        [(tri[0], tri[1]), (tri[0], tri[2]), (tri[1], tri[2])]
        for tri in simplices
    ])  # (N_tri, 3, 2)

    vecs = points[local_edges[:, :, 0]] - points[local_edges[:, :, 1]]
    actual_lengths = np.sqrt(np.sum(vecs ** 2, axis=2))  # (N_tri, 3)

    if rigidities is None:
        rigidities = tri_result.get_default_rigidities()
    if rest_lengths is None:
        rest_lengths = actual_lengths.copy()

    # Deduplicate edges
    edge_index, tri_edge_map, unique_edges = deduplicate_edges(simplices)

    # Per-unique-edge parameters
    k_per_edge, l0_per_edge = solver_params_to_unique_edges(
        rigidities, rest_lengths, tri_edge_map
    )

    # Hard/soft mask per unique edge
    n_unique = len(unique_edges)
    is_hard = np.ones(n_unique, dtype=bool)
    if tri_result.hard_edge_set is not None:
        for eidx, (a, b) in enumerate(unique_edges):
            is_hard[eidx] = tri_result.edge_is_hard(a, b)

    # Edge features
    edge_attr = compute_edge_attr(
        points, unique_edges, k_per_edge, l0_per_edge, is_hard
    )

    # Geometric edge features (for CVAE decoder)
    edge_geom = compute_geometric_edge_attr(points, unique_edges, is_hard)

    # Node features
    pos = torch.tensor(points, dtype=torch.float32)
    n_nodes = len(points)
    x = init_node_features(pos, edge_index, edge_attr, n_nodes)

    # Build Data object
    data = Data(
        x=x,
        pos=pos,
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_geom=edge_geom,
    )

    # Edge mask: True = designable (hard), False = frozen (soft)
    # Bidirectional: duplicate mask for both directions
    edge_mask = np.concatenate([is_hard, is_hard])
    data.edge_mask = torch.tensor(edge_mask, dtype=torch.bool)

    # Mapping back to solver format
    data.tri_edge_map = torch.tensor(tri_edge_map, dtype=torch.long)
    data.simplices = torch.tensor(simplices, dtype=torch.long)
    data.n_unique_edges = n_unique

    # Topology metadata
    data.topo_name = tri_result.topo_name
    data.topo_class = tri_result.topo_class
    data.k_soft_ratio = tri_result.k_soft_ratio

    # Labels
    if labels is not None:
        if 'poisson' in labels:
            data.y_nu = torch.tensor(labels['poisson'], dtype=torch.float32)
        if 'young' in labels:
            data.y_young = torch.tensor(labels['young'], dtype=torch.float32)
        if 'elastic_tensor' in labels:
            data.y_tensor = torch.tensor(labels['elastic_tensor'], dtype=torch.float32)

    return data
