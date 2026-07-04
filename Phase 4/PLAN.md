# Phase 4: GNN Surrogate + CVAE Inverse Design + Interpretability

> **⚠️ OUT OF DATE — UPDATE THIS FIRST (2026 cleanup).**
> This plan predates two major changes and must be revised before any Phase 4 work resumes:
> 1. The forward solver was replaced. The mean-field D2C model this plan assumes is now
>    the *legacy* path; the default is the **intrinsic** metric solve
>    (`Phase 2/forward_solver_torch.py`, `method='intrinsic'`), which reproduces the PBC
>    simulation. Any GNN/CVAE labels or physics-loss evaluations must use it — the old
>    labels are invalid. See `ANALYTICAL_MODEL_STATUS.md`.
> 2. The Phase 4 *implementation* (GNN, CVAE, interpretability, data pipeline, training
>    scripts) was removed in the cleanup and lives only in git history. What remains is
>    this plan plus the **topology generators** (`Phase 4/data/topology_generators.py`,
>    `rigidity_patterns.py`) — whose fate (which topologies to generate at large scale)
>    is still to be decided. Rescope this plan against the current solver before rebuilding.

## Table of Contents

1. [Expanded Topology Catalog](#a-expanded-topology-catalog)
2. [Non-Triangulated Meshes with Soft-Edge Regularization](#b-non-triangulated-meshes-with-soft-edge-regularization)
3. [Data Generation Pipeline](#c-data-generation-pipeline)
4. [GNN Forward Surrogate](#d-gnn-forward-surrogate)
5. [CVAE Inverse Design](#e-cvae-inverse-design)
6. [Interpretability Probes](#f-interpretability-probes)
7. [File Structure](#g-file-structure)
8. [Implementation Sequence](#h-implementation-sequence)

---

## A. Expanded Topology Catalog

The GNN must generalize across diverse graph structures, not just perturbed hexagonal
lattices. We train on **4 categories** of mesh topologies:

### Category 1: Hexagonal/Triangular Lattice Family (existing)

These are already implemented in `Disc_2_Cont_optimized.py`:

| Name | Generator | Coordination | Notes |
|------|-----------|--------------|-------|
| `iso_crystal` | `generate_cryratl_points(size, (1,1), 0)` | 6 (exact) | Regular triangular lattice |
| `aniso_crystal` | `generate_cryratl_points(size, (1.5,0.8), pi/6)` | 6 (exact) | Stretched + rotated |
| `foam_eta02` | `generate_foam_points(size, 0.2)` | ~6 | Mild perturbation of hex |
| `foam_eta045` | `generate_foam_points(size, 0.45)` | ~5.8 | Strong perturbation |

### Category 2: Fully Random Triangulations (new)

These have **no underlying lattice structure** and widely varying coordination numbers.

| Name | Method | Coordination | Notes |
|------|--------|--------------|-------|
| `poisson_delaunay` | Uniform random points -> Delaunay | 4-9, avg ~6 | Already exists in `sweep_utils.py` |
| `blue_noise_delaunay` | Poisson disk sampling -> Delaunay | 5-7, avg ~6 | More uniform spacing than pure random |
| `clustered_delaunay` | Gaussian mixture points -> Delaunay | 3-12+ | High coordination in dense regions, low in sparse |
| `gradient_density` | Non-uniform density field -> Delaunay | 3-10+ | Smoothly varying coordination across mesh |

**`poisson_delaunay`** (already exists as `generate_poisson_network`):
```python
# Uniform random points in [-size-2, size+2]^2
# Delaunay triangulation -> filter interior triangles
# Coordination varies: some vertices have 4-5, others 7-9
```

**`blue_noise_delaunay`** (new):
```python
def generate_blue_noise_network(size, min_dist=0.6):
    """Poisson disk sampling: random points with minimum distance constraint.
    More regular than pure random, but not lattice-periodic.
    Coordination numbers: typically 5-7 (narrower than pure Poisson)."""
    # Bridson's algorithm for Poisson disk sampling
    # Then Delaunay triangulate
```

**`clustered_delaunay`** (new):
```python
def generate_clustered_network(size, n_clusters=5, cluster_std=1.0):
    """Gaussian mixture model point process.
    Dense clusters connected by sparse bridges.
    Coordination: 3-4 in bridge regions, 8-12+ in cluster cores."""
    # Sample cluster centers uniformly
    # For each cluster, sample points from N(center, std^2)
    # Delaunay triangulate the union
```

**`gradient_density`** (new):
```python
def generate_gradient_density_network(size, density_ratio=4.0):
    """Non-uniform density: dense on left, sparse on right.
    Tests GNN's ability to handle varying local structure.
    Coordination: varies smoothly across mesh."""
    # Rejection sampling with density proportional to position
    # Delaunay triangulate
```

### Category 3: Lattices with Non-Trivial Basis (new, triangulated)

These are periodic lattices with >1 atom per unit cell. They produce fundamentally
different graph structures than the triangular family.

| Name | Basis | Coordination | Physical Relevance |
|------|-------|--------------|-------------------|
| **Kagome** | 3 per cell | 4 | Frustrated magnetism, floppy modes |
| **Square** | 1 per cell | 4 | Simplest non-triangular |
| **Snub square** | 4 per cell | 5 | Mix of triangles + squares |
| **Truncated hex** | 6 per cell | 3 | Related to honeycomb |
| **Cairo pentagonal** | 4 per cell | 3-4 | Exotic elastic behavior |

All of these must be **triangulated** before the solver can use them. See Section B for
how non-triangular faces are handled.

**Kagome lattice:**
```python
def generate_kagome_lattice(size):
    """Kagome = corner-sharing triangles on hexagonal Bravais lattice.
    Unit cell: 3 sites at (0,0), (1/2,0), (1/4, sqrt(3)/4).
    Naturally has triangular AND hexagonal faces.
    Hexagonal faces are triangulated with soft diagonal edges."""
    a1 = np.array([1, 0])
    a2 = np.array([0.5, np.sqrt(3)/2])
    basis = np.array([
        [0, 0],
        [0.5, 0],
        [0.25, np.sqrt(3)/4],
    ])
    # Generate lattice points, triangulate, add soft edges in hexagons
```

**Square lattice:**
```python
def generate_square_lattice(size, spacing=1.0):
    """Regular square grid. Each square face is triangulated by adding
    one diagonal with a soft spring. Alternating diagonal directions
    avoids systematic bias."""
    # Grid of points at (n*spacing, m*spacing)
    # Connect nearest neighbors (4-coordinated)
    # Add diagonal in each square face (soft spring)
    # -> becomes triangulated mesh
```

**Snub square lattice:**
```python
def generate_snub_square_lattice(size):
    """Archimedean tiling with vertex config (3.3.4.3.4).
    Mix of triangular and square faces.
    5-coordinated: every vertex touches 3 triangles + 2 squares.
    Squares are triangulated with soft diagonals."""
```

**Truncated hexagonal (trihexagonal/Kagome dual):**
```python
def generate_truncated_hex_lattice(size):
    """Hexagons + triangles in 3.6.3.6 pattern.
    3-coordinated vertices. All hexagonal faces triangulated
    with soft internal edges."""
```

**Cairo pentagonal:**
```python
def generate_cairo_lattice(size):
    """Cairo pentagonal tiling. Mix of 3- and 4-coordinated vertices.
    Pentagonal faces are triangulated (3 triangles per pentagon)
    with soft diagonals."""
```

### Category 4: Non-Triangulated Meshes with Soft-Edge Regularization (new)

These preserve the **mechanical identity** of non-triangulated lattices (honeycomb,
square, kagome) while making them compatible with the triangulation-based solver.
See detailed treatment in Section B.

| Name | Original Coordination | Face Types | Regularization Strategy |
|------|----------------------|------------|------------------------|
| **Honeycomb** | 3 | Hexagons only | Center vertex + 6 soft radial edges |
| **Square** (non-diag) | 4 | Squares only | Soft diagonal in each face |
| **Kagome** (pure) | 4 | Triangles + Hexagons | Soft diagonals in hexagonal faces |

---

## B. Non-Triangulated Meshes with Soft-Edge Regularization

### The Problem

The forward solver (`ElasticSolver`) requires every face to be a triangle:
```python
# solver input format: (N_triangles, 3) for simplices, rigidities, rest_lengths
solver = ElasticSolver(positions, simplices, edges)
result = solver(rigidities, rest_lengths)  # rigidities shape: (N_tri, 3)
```

Non-triangulated lattices (honeycomb, square grid) have non-triangular faces that
cause the solver to crash (degenerate elastic tensors, singular matrices).

### The Solution: Soft-Edge Regularization

**Strategy:** Triangulate every non-triangular face by adding internal edges with
very soft springs (k_soft << k_hard). The soft springs:
- Remove mechanical singularities (the mesh becomes fully triangulated)
- Contribute negligibly to the elastic response (k_soft/k_hard ~ 1e-3)
- Preserve the dominant physics of the original lattice connectivity

**Two triangulation methods for polygonal faces:**

#### Method 1: Fan Triangulation (no new vertices)
For an n-gon face with vertices [v0, v1, ..., v_{n-1}], add diagonals from v0
to all non-adjacent vertices: (v0,v2), (v0,v3), ..., (v0,v_{n-2}).
This creates n-2 triangles. All diagonals get soft springs.

```
Hexagonal face (6-gon):        Fan triangulation:
   v1 --- v2                      v1 --- v2
  /         \                    /|\       \
v0           v3       ->      v0  | \       v3
  \         /                    \|  \     /|
   v5 --- v4                      v5--\--v4
                                       \|
                                  4 diagonals added (soft)
                                  -> 4 triangles
```

#### Method 2: Center-Point Triangulation (adds a new vertex)
Add a vertex at the centroid of each face, connect to all face vertices.
Creates n triangles for an n-gon (vs n-2 for fan). All new edges are soft.

```
Hexagonal face:                Center-point:
   v1 --- v2                      v1 --- v2
  /         \                    /|\ c /|\
v0           v3       ->      v0  | \/  | v3
  \         /                    \| /\  |/
   v5 --- v4                      v5 --- v4
                                  6 radial edges (soft)
                                  -> 6 triangles
```

**Recommended:** **Center-point** for hexagonal faces (honeycomb, kagome hexagons).
It produces more uniform triangles and better numerical conditioning. Fan
triangulation creates very thin triangles at the far end of large polygons.

### Honeycomb Lattice Implementation

```python
def generate_honeycomb_lattice(size, k_soft_ratio=1e-3):
    """True honeycomb lattice (coordination 3) with center-point regularization.

    Returns:
        tri: Delaunay-like object with .points, .simplices, .edges
        hard_edge_mask: (N_tri, 3) bool -- True for original honeycomb edges
        k_soft_ratio: ratio of soft to hard spring constants
    """
    # 1. Generate honeycomb vertices
    #    Unit cell: 2 atoms at (0, 0) and (1/2, 1/(2*sqrt(3)))
    #    Bravais vectors: a1 = (1, 0), a2 = (1/2, sqrt(3)/2)
    a1 = np.array([1.0, 0.0])
    a2 = np.array([0.5, np.sqrt(3)/2])
    basis = np.array([
        [0.0, 0.0],
        [0.5, 1.0 / (2 * np.sqrt(3))],
    ])

    # 2. Generate all lattice points
    points = []
    for n in range(-max_n, max_n+1):
        for m in range(-max_m, max_m+1):
            origin = n * a1 + m * a2
            for b in basis:
                points.append(origin + b)
    points = np.array(points)

    # 3. Build honeycomb edges (nearest-neighbor only, coordination 3)
    # Each atom connects to 3 neighbors

    # 4. Identify hexagonal faces

    # 5. Add center vertices for each hexagonal face
    # 6. Connect center to all 6 face vertices (soft springs)
    # 7. Build triangle list from the now-triangulated mesh

    # 8. Mark edges:
    #    - hard_edge_mask[i,j] = True  if edge (i,j) is an original honeycomb bond
    #    - hard_edge_mask[i,j] = False if edge is a soft regularization edge
    #
    #    Default rigidities:
    #    k[hard] = 1.0
    #    k[soft] = k_soft_ratio = 0.001

    return tri_object, hard_edge_mask, default_rigidities
```

### Rigidity Assignment for Non-Triangulated Meshes

The key insight: in the CVAE and GNN, we need to distinguish between "real" edges
(the original lattice bonds, designable) and "regularization" edges (soft, fixed).

```python
# Per-edge attribute includes an "is_real" flag:
edge_attr = [k, l0, l_actual, log(k/l0^2), is_real]
#                                             ^^^^^
#                                   1.0 for lattice bonds
#                                   0.0 for soft regularization edges

# During inverse optimization / CVAE decoding:
# - Only "real" edges have their rigidities/rest_lengths varied
# - "Soft" edges keep k = k_soft_ratio, l0 = l_actual (frozen)
# This is implemented via the edge_mask in the Data object
```

### Regularization Parameter Selection

The ratio `k_soft / k_hard` must be:
- **Small enough** that soft edges don't dominate the elastic response
- **Large enough** that the solver doesn't encounter numerical singularities

From the forward solver, the per-edge contribution to the elastic tensor scales as
`k / l0^2`. So the effective contribution ratio is `(k_soft / l_soft^2) / (k_hard / l_hard^2)`.

**Recommended:** `k_soft_ratio = 1e-3` (soft springs are 1000x weaker than hard springs).
This gives a ~0.1% contribution to the elastic tensor from regularization edges.

**Verification:** Compute Poisson ratio of honeycomb with k_soft_ratio in {1e-2, 1e-3, 1e-4, 1e-5}.
Convergence as k_soft -> 0 confirms regularization doesn't alter the physics.

---

## Example: Uniformly Random Mesh (Not Perturbed Hexagonal)

A Poisson-Delaunay mesh has NO lattice structure. Compare:

```
PERTURBED HEXAGONAL (foam, eta=0.2):      UNIFORMLY RANDOM (Poisson-Delaunay):

    *---*---*---*---*                          *         *
   / \ / \ / \ / \ / \                       / \       / \
  *---*---*---*---*---*                   *--*   *---*   *
   \ / \ / \ / \ / \ /                    \ |\ /|  / \ / \
    *---*---*---*---*                       *| *-*--*   *  *
   / \ / \ / \ / \ / \                      |/ \ |  \ / \/
  *---*---*---*---*---*                   *--*   *|   *--*
   \ / \ / \ / \ / \ /                    \ / \/||  / \ |
    *---*---*---*---*                       *   *-*--*  *|
                                                   \ / \/
  -> Regular, all vertices ~6-coordinated           *   *
  -> Slight position noise, topology frozen
  -> All triangles similar size              -> Irregular, vertices 4-9 coordinated
                                             -> Some tiny triangles, some large
                                             -> Dense and sparse regions coexist
```

The Poisson-Delaunay mesh is generated by:
```python
def generate_poisson_network(size):
    density = 2.0 / np.sqrt(3)  # match hex lattice density
    area = (2*(size[0]+2)) * (2*(size[1]+2))
    n_points = int(round(density * area))
    points = np.column_stack([
        np.random.uniform(-(size[0]+2), size[0]+2, n_points),  # x ~ Uniform
        np.random.uniform(-(size[1]+2), size[1]+2, n_points),  # y ~ Uniform
    ])
    DM = scipy.spatial.Delaunay(points)  # Triangulate random points
    # Filter to interior...
```

Key difference: in the hexagonal foam, EVERY interior vertex has exactly 6 neighbors
because the topology is inherited from the perfect crystal (only positions are perturbed).
In Poisson-Delaunay, the triangulation itself is random, so connectivity varies per vertex.

We add more extreme random variations:
- **Clustered**: Gaussian mixture → some vertices have 10-12 neighbors in dense regions
- **Gradient density**: smooth density variation → coordination changes across the mesh
- **Blue noise**: minimum-distance constraint → more uniform than pure random, less than crystal

---

## C. Data Generation Pipeline

### Variation Axes

**Topology** (13+ types across 4 categories):

| Category | Topologies | Seeds/Variants |
|----------|-----------|----------------|
| Hex lattice family | iso_crystal, aniso_crystal, foam_eta02, foam_eta045 | 1, 1, 20, 20 |
| Fully random | poisson, blue_noise, clustered, gradient_density | 20 each |
| Lattices with basis | kagome, square, snub_square, truncated_hex, cairo | 1 each (periodic) |
| Non-triangulated | honeycomb_soft, square_soft, kagome_soft | 1 each (periodic) |

**Mesh size**: {(6,6), (8,8), (10,10)} for training; (12,12) for OOD test.

**Edge parameters** (3 modes):
1. **Rigidities only**: k ~ LogNormal(0, sigma) with sigma in {0.3, 0.5, 1.0, 2.0}
2. **Rest lengths only**: l0 ~ l_actual * exp(N(0, sigma)) with sigma in {0.05, 0.1, 0.2}
3. **Both**: independent sampling of k and l0

For non-triangulated meshes, only "real" edges have their parameters varied;
soft regularization edges are frozen at k_soft.

**Structured variation**: virtual distortion, Phase 3 optimized solutions.

### Dataset Size

| Split | Samples | Purpose |
|-------|---------|---------|
| Train | 80,000 | Main training (all topology categories represented) |
| Val   | 10,000 | Hyperparameter tuning, early stopping |
| Test  | 10,000 | Final evaluation |
| OOD   | 5,000  | Unseen topology seeds + unseen mesh sizes |

### Storage

PyG `InMemoryDataset`. Each `Data` object:
```python
Data(
    x         = (M, node_feat_dim),    # Node features
    pos       = (M, 2),               # Coordinates
    edge_index = (2, E),              # Unique undirected edges (COO)
    edge_attr  = (E, 5),             # [k, l0, l_actual, log(k/l0^2), is_real]
    edge_mask  = (E,),               # bool: True = designable, False = soft/frozen
    y_nu       = scalar,             # Poisson's ratio
    y_young    = scalar,             # Young's modulus
    y_tensor   = (6,),              # Full elastic tensor
    tri_edge_map = (N_tri, 3),      # Maps unique edge idx -> solver format
    simplices  = (N_tri, 3),        # Triangle vertex indices
    topo_class = int,               # Topology category ID
    mesh_size  = (2,),              # Grid size
)
```

---

## D. GNN Forward Surrogate

### Architecture: 4-layer NNConv MPNN

**Why NNConv:** The forward solver's physics is dominated by per-edge factors k/l0^2.
NNConv generates weight matrices from edge features, so each spring's physical
parameters directly modulate message flow. This mirrors the physics.

**Edge features (dim=5):** `[k, l0, l_actual, log(k/l0^2), is_real]`

The `is_real` flag (0/1) lets the GNN learn to weight real vs regularization edges
differently. For pure triangulated meshes, all edges have is_real=1.

**Node features (dim=8):** Computed from geometry + aggregated edge info:
`[x, y, degree, mean_k, std_k, mean_l, std_l, mean_log(k/l0^2)]`

```
Layer stack:
  [Input]  node_feat (M, 8)  +  edge_attr (E, 5)
      |
  Linear(8, 64)
      |
  4 x [NNConv(64->64, edge_nn: 5->128->64*64) + BatchNorm + ReLU + residual]
      |
  Set2Set pooling (steps=6)  ->  (batch, 128)
      |
  MLP: 128 -> 64 -> 32 -> 1
      |
  [Output]  nu_pred (scalar)
```

**Multi-task heads:** Shared backbone, separate MLPs for nu, Young's modulus,
full tensor. Loss: `L_nu + 0.1*L_young + 0.01*L_tensor`.

**Training:** AdamW lr=1e-3, cosine scheduler, batch=64, 300 epochs, early stop.
Target: MAE(nu) < 0.02.

---

## E. CVAE Inverse Design

### Encoder (training only)

GNN over (graph + true edge params + nu_target) -> (mu, logvar), latent dim=32.
Same NNConv backbone as forward GNN.

### Decoder (training + inference)

(graph topology + z + nu_target) -> per-edge (k, l0).
- Uses geometric edge features only (l_actual, dx, dy, angle) — NOT k, l0
- Node conditioning: geometry + broadcasted z + nu_embed (additive)
- Edge readout: MLP([h_i || h_j || edge_geom]) -> softplus -> (k, l0)

**Edge masking for non-triangulated meshes:** Only "real" edges are generated by
the decoder. Soft regularization edges keep their fixed k_soft values:
```python
k_pred[~edge_mask] = k_soft_ratio  # frozen
l0_pred[~edge_mask] = l_actual[~edge_mask]  # frozen
```

### Loss

```
L = L_recon + beta*L_KL + gamma*L_physics
```
- Reconstruction in log-space for k, normalized for l0
- Beta warmup: 0->1 over 50 epochs
- Physics loss (gamma=10): GNN surrogate for epochs 1-200, actual solver 201-300
- Inference: sample z ~ N(0,I), decode, optionally refine with L-BFGS (5-10 steps)

---

## F. Interpretability Probes

1. **Edge saliency:** d(nu)/d(k_e) per edge, via gradient + integrated gradients
2. **Node embedding clustering:** UMAP/t-SNE of hidden representations, K-means
3. **Subgraph activation:** Which local motifs maximally activate each hidden dimension
4. **GNNExplainer:** Minimal sufficient subgraph for each prediction
5. **Cross-topology comparison:** Do honeycomb and kagome share structural features?

All visualized via `LineCollection` on the lattice (matching sweep_utils.plot_mesh style).

---

## G. File Structure

```
Phase 4/
    PLAN.md                            # This file
    README.md                          # Phase 4 overview + results

    # ── Data ──
    data/
        __init__.py
        generate_dataset.py            # Data generation pipeline
        dataset.py                     # PyG InMemoryDataset class
        graph_utils.py                 # Edge dedup, node features, tri->PyG conversion
        topology_generators.py         # NEW: all lattice/mesh generators
                                       #   - generate_blue_noise_network()
                                       #   - generate_clustered_network()
                                       #   - generate_gradient_density_network()
                                       #   - generate_kagome_lattice()
                                       #   - generate_square_lattice()
                                       #   - generate_snub_square_lattice()
                                       #   - generate_truncated_hex_lattice()
                                       #   - generate_cairo_lattice()
                                       #   - generate_honeycomb_lattice()
                                       #   - triangulate_polygon_fan()
                                       #   - triangulate_polygon_center()
                                       #   - regularize_non_triangulated()
        configs/
            default_data_config.yaml

    # ── GNN Forward Surrogate ──
    gnn/
        __init__.py
        model.py                       # ForwardGNN class (NNConv MPNN)
        train.py                       # Training loop with logging
        evaluate.py                    # Test evaluation + error analysis
        configs/
            default_gnn_config.yaml

    # ── CVAE Inverse ──
    cvae/
        __init__.py
        encoder.py                     # GNN-based encoder -> (mu, logvar)
        decoder.py                     # GNN-based decoder -> edge params
        model.py                       # Full CVAE (encoder + decoder + loss)
        train.py                       # Training with beta warmup
        sample.py                      # Inference: sample designs from prior
        refine.py                      # Optional L-BFGS refinement
        configs/
            default_cvae_config.yaml

    # ── Interpretability ──
    interpret/
        __init__.py
        saliency.py                    # Gradient + integrated-gradient saliency
        embeddings.py                  # Node embedding extraction + clustering
        subgraph_analysis.py           # Subgraph activation (CNN analogy)
        gnn_explainer.py               # GNNExplainer wrapper
        visualize.py                   # Unified lattice visualization

    # ── Tests ──
    tests/
        test_topology_generators.py    # Test all new lattice generators
        test_data_pipeline.py          # Test data gen + edge deduplication
        test_gnn.py                    # Test forward GNN shapes + forward pass
        test_cvae.py                   # Test CVAE encode/decode/loss
        test_interpret.py              # Test saliency, embedding extraction

    # ── Shared ──
    utils.py                           # Logging, config, seeding

    # ── Outputs (gitignored) ──
    checkpoints/
    outputs/
```

---

## H. Implementation Sequence

### Phase 4a: Topology Generators + Data Pipeline

1. `data/topology_generators.py` -- All new lattice generators:
   - Blue noise, clustered, gradient density (random triangulations)
   - Kagome, square, snub square, truncated hex, Cairo (lattices with basis)
   - Honeycomb with center-point regularization
   - `triangulate_polygon_center()` and `triangulate_polygon_fan()` helpers
   - `regularize_non_triangulated()` -- adds soft edges to arbitrary polygon meshes
2. `data/graph_utils.py` -- Edge dedup, node feature init, tri->PyG conversion
3. `data/generate_dataset.py` -- Main data generation script
4. `data/dataset.py` -- PyG InMemoryDataset wrapper
5. `tests/test_topology_generators.py` -- Verify all generators produce valid meshes

### Phase 4b: GNN Forward Surrogate

6. `gnn/model.py` -- ForwardGNN architecture
7. `gnn/train.py` -- Training loop
8. `gnn/evaluate.py` -- Evaluation + error analysis
9. `tests/test_gnn.py`

### Phase 4c: CVAE Inverse

10. `cvae/encoder.py` + `cvae/decoder.py`
11. `cvae/model.py` + `cvae/train.py`
12. `cvae/sample.py` + `cvae/refine.py`
13. `tests/test_cvae.py`

### Phase 4d: Interpretability

14. `interpret/saliency.py`
15. `interpret/embeddings.py`
16. `interpret/subgraph_analysis.py`
17. `interpret/gnn_explainer.py`
18. `interpret/visualize.py`
19. `tests/test_interpret.py`

### Phase 4e: Documentation

20. `README.md` -- Phase 4 overview
21. `../Discussion.md` -- Our conversation
22. `../Tutorial.md` -- GNN + VAE tutorial

---

## Lattice Gallery: Suggested Non-Trivial Lattices with Basis

### 1. Honeycomb (coordination 3)

The simplest non-triangulated lattice. 2 atoms per hexagonal unit cell.
Each vertex has exactly 3 bonds. The dual of the triangular lattice.
Honeycomb networks are the natural model for graphene-like materials and
are expected to show strongly auxetic behavior (nu < 0) under certain
rigidity patterns.

**Regularization:** Center-point in each hexagonal face (adds M_hex vertices
and 6*M_hex soft edges, creating 6*M_hex triangles).

### 2. Kagome (coordination 4)

3 atoms per hexagonal unit cell forming corner-sharing triangles.
The hexagonal voids between the triangles make kagome special:
it sits right at the Maxwell isostaticity threshold (z=2d=4 in 2D).
This means it can host "zero-energy" floppy modes.

**Regularization:** Hexagonal faces get center-point triangulation (soft edges).
Triangular faces are already triangulated (hard edges).

### 3. Square Lattice (coordination 4)

1 atom per square unit cell. Also at the isostaticity threshold.
Square faces need one diagonal each (alternating NW-SE and NE-SW to
break directional bias). The choice of diagonal pattern can itself
affect elastic properties — an interesting variable.

**Regularization:** One soft diagonal per square face (2 triangles per face).

### 4. Snub Square (coordination 5)

Archimedean tiling with vertex configuration (3.3.4.3.4). Each vertex
touches 3 triangles and 2 squares. 4 atoms per unit cell. Above the
isostaticity threshold — intrinsically rigid.

**Regularization:** Square faces get soft diagonals. Triangular faces unchanged.

### 5. Truncated Hexagonal / Trihexagonal (coordination 3)

Archimedean tiling with alternating triangles and hexagons, vertex config (3.6.3.6).
The "Kagome dual" in some sense. 3-coordinated, strongly sub-isostatic.

**Regularization:** Hexagonal faces get center-point (6 soft edges each).

### 6. Cairo Pentagonal (coordination 3-4)

Dual of the snub square. Pentagonal faces with a mix of 3- and 4-coordinated
vertices. Interesting because it breaks the usual triangular/hexagonal symmetry.

**Regularization:** Each pentagon → 3 triangles via fan from one vertex (2 soft diags).

---

## Key Architectural Decisions

1. **NNConv over GAT/GCN** — edge features carry physics (k/l0^2); NNConv makes
   edge-dependent weight matrices; GCN ignores edges; GAT is less direct.

2. **is_real flag in edge features** — lets GNN learn to treat regularization edges
   differently from structural bonds without separate handling code.

3. **Center-point triangulation** for hexagonal faces — more uniform triangles than
   fan, better numerical conditioning, worth the extra vertices.

4. **k_soft_ratio = 1e-3** — 3 orders of magnitude separation. Verified by convergence
   test as k_soft -> 0.

5. **Log-space loss for rigidities** — spans 3-4 orders of magnitude; linear MSE
   would be dominated by large values.

6. **Edge masking in CVAE** — decoder outputs all edges, but gradients are masked
   for regularization edges. Same model handles all topology types.
