# Discussion: From Predictive Coding to GNN-Based Inverse Design

This document captures the design conversation that led to the Phase 4 architecture:
a GNN forward surrogate, a conditional VAE for inverse design, and interpretability
probes for elastic spring networks.

---

## 1. Starting Point: Why Move Beyond Direct Optimization?

Phase 3 established that gradient-based inverse optimization works: given a target
Poisson's ratio, L-BFGS finds spring rigidities that produce it. But the approach
has limitations:

- **Speed**: Each design requires 40-2000 L-BFGS steps through the differentiable solver
- **Single solutions**: Each optimization run converges to one local minimum. The inverse
  problem is massively underdetermined (N x 3 unknowns, 6 equations), so there are many
  valid solutions. Finding them all requires many random restarts.
- **No generalization**: Solving for nu = -0.5 teaches nothing about nu = -0.3. Each
  target requires optimization from scratch.

**The question**: Can we learn a model that (a) predicts elastic properties directly from
the spring network graph, and (b) generates valid designs for a given target in one shot?

---

## 2. Predictive Coding and the Brain Analogy

The conversation began with predictive coding -- the neuroscience theory that the brain
maintains a generative model of sensory input and learns by minimizing prediction errors.
The key insight that transfers to our problem:

**Predictive coding in the brain:**
- Top-down: generative model predicts sensory input
- Bottom-up: prediction errors propagate upward
- Learning: adjust the model to reduce prediction errors

**Predictive coding in elastic networks:**
- Forward model: spring parameters -> elastic properties (our forward solver)
- Inverse model: desired elastic properties -> spring parameters (what we want)
- Physics loss: "prediction error" = |nu_predicted - nu_target|^2

The CVAE architecture directly implements this: the decoder is the generative model
(top-down), the physics loss provides the error signal (bottom-up), and training
adjusts weights to minimize the error. The encoder adds an amortized inference
step -- instead of iteratively optimizing per-sample (as in traditional predictive
coding or our Phase 3 L-BFGS), it learns to map directly to the latent code.

---

## 3. Why GNN for the Forward Surrogate?

**Question**: Why not just use an MLP on flattened edge parameters?

**Answer**: The spring network is a graph with irregular topology. An MLP would need:
- Fixed input dimension (fails for different mesh sizes)
- Explicit encoding of which edges connect to which nodes
- No weight sharing across structurally equivalent subgraphs

A GNN naturally handles all three:
- **Variable-size graphs**: message passing works on any number of nodes/edges
- **Structural encoding**: the adjacency is the graph itself
- **Weight sharing**: the same message function applies to all edges

**The physics argument for GNN**: The forward solver computes per-triangle elastic
tensors, then solves a Woodbury system that couples all triangles. Each triangle's
contribution depends on its edge rigidities and geometry. This is naturally a
local computation (per-triangle) followed by global coupling -- exactly what
message-passing layers followed by global pooling compute.

### Choosing NNConv (Edge-Conditioned Convolution)

The critical design choice is how edge features enter the message function. Options:

| Method | Edge Feature Handling | Fit for Our Problem |
|--------|----------------------|---------------------|
| GCN | Ignores edge features entirely | Poor -- k and l0 are the inputs! |
| GAT | Attention weights from node features only | Mediocre -- edges are secondary |
| GATv2 + edge | Concat edge features into attention | OK but indirect |
| **NNConv** | **Edge features generate weight matrices** | **Best -- physics is edge-dominated** |

NNConv computes messages as:
```
m_{j->i} = MLP_edge(edge_attr_{ij}) @ h_j
```

The edge features `[k, l0, l_actual, log(k/l0^2)]` directly parameterize the weight
matrix that transforms each neighbor's hidden state. This mirrors the physics: in the
forward solver, the per-edge factor `k / l0^2 / 16` is literally a weight that
multiplies the geometric tensor of that edge.

### Set2Set Pooling

Poisson's ratio is a global property -- it describes how the entire material
responds to stress, not any local region. Simple mean pooling over node embeddings
loses distributional information. Set2Set (an LSTM-based attention pooling from
Vinyals et al.) computes:

```
q_t = LSTM(q_{t-1})
a_t = softmax(q_t^T h_i) for all nodes i
r_t = sum(a_t * h_i)
output = [q_T || r_T]
```

This captures which nodes are "important" for the prediction and how they relate to
the graph average -- essential for a quantity like nu that depends on correlations
across the entire structure.

---

## 4. Why CVAE for Inverse Design?

**Question**: Why not just train a regression model that maps nu_target -> edge parameters?

**Answer**: The inverse problem is one-to-many. For a given target nu, there are
exponentially many valid edge configurations (Phase 3 showed 5 random starts produce
5 distinct solution clusters). A regression model would average over these modes,
producing a "blurry" mean solution that isn't actually valid.

A CVAE handles multi-modality by learning a latent space where different regions
correspond to different solution families:

```
z ~ N(0, I)          # sample from latent space
k, l0 = decoder(graph, z, nu_target)  # different z -> different valid designs
```

### The Encoder-Decoder Asymmetry

A subtle but important design choice: the encoder and decoder see different information.

**Encoder** (training only): sees the full solution (graph + edge params + nu_target).
Its job is to compress a known solution into a compact latent code z that captures
what's "essential" about that particular solution (as opposed to the infinite other
solutions for the same target).

**Decoder** (training + inference): sees only the graph topology + z + nu_target.
It does NOT see edge parameters -- it must generate them. This forces the latent
code z to actually carry useful information.

### Physics Loss: The Key Ingredient

Standard VAE training uses reconstruction loss (make the output match the input).
For our problem, this is insufficient: we don't just want edge parameters that LOOK
like the training data, we want edge parameters that PRODUCE the target Poisson ratio.

The physics loss routes the decoder output through the forward model:
```
k, l0 = decoder(graph, z, nu_target)
nu_predicted = forward_model(graph, k, l0)   # GNN surrogate or actual solver
L_physics = (nu_predicted - nu_target)^2
```

This is differentiable end-to-end because both the GNN surrogate and the Phase 2
solver support backpropagation.

**Training strategy**: Use the fast GNN surrogate for physics loss during early
training (epochs 1-200), then switch to the exact differentiable solver for fine-tuning
(epochs 201-300). The surrogate has ~2% error, which becomes the noise floor if
used exclusively.

---

## 5. Edge Features, Not Node Features

**Question**: The lattice nodes have no intrinsic properties -- the physics is all
in the edges (springs). How do we handle "no node features"?

**Answer**: We construct node features from geometry and aggregated edge information:

```python
node_features = [
    x, y,                    # Position (normalized)
    degree,                  # Number of incident edges
    mean(k_incident),        # Mean rigidity of connected springs
    std(k_incident),         # Variability of connected springs
    mean(l_incident),        # Mean edge length
    std(l_incident),         # Variability of edge lengths
    mean(log(k/l0^2)),       # Mean of the physics-relevant factor
]
```

This initialization is sufficient because message passing propagates edge information
to nodes in the first layer. After 4 message-passing steps, every node's hidden
state encodes information from its 4-hop neighborhood, including all edge features
in that region.

**The is_real flag**: For non-triangulated meshes (honeycomb, kagome) that require
soft-edge regularization, we add a 5th edge feature: `is_real` (1.0 for structural
bonds, 0.0 for soft regularization edges). This lets the GNN learn to weight real
vs. regularization edges differently without any special-case code.

---

## 6. Interpretability: The CNN Analogy

**Question**: In CNNs, we can visualize what each filter detects (edges, textures,
objects). Can we do the same for GNNs on spring networks?

**Answer**: Yes, with adaptations. The key insight is that GNN "features" correspond
to local subgraph motifs, not pixel patterns.

### The Analogy

| CNN | GNN |
|-----|-----|
| Filter detects a local pixel pattern (edge, corner, texture) | Hidden dimension activated by a local subgraph motif |
| Feature map = where in the image the pattern appears | Node activation = where in the graph the motif appears |
| Saliency map = which pixels matter for the output | Edge saliency = which springs matter for nu |
| DeepDream = amplify a feature | Graph "dreaming" = find graph that maximally activates a feature |

### Four Interpretability Probes

**Probe 1: Edge Saliency**
Compute |d(nu)/d(k_e)| for each edge. High saliency = changing this spring's rigidity
strongly affects the Poisson ratio. For auxetic designs (nu < 0), we expect high saliency
on the "re-entrant hinge" edges identified in Phase 3.

Integrated gradients (interpolate from uniform k=1 to actual) are more robust than
simple gradients, which can be noisy at sharp loss landscape features.

**Probe 2: Node Embedding Clustering**
After message passing, each node's hidden state (64-dim vector) encodes its structural
role in the network. UMAP projection to 2D reveals functional clusters:
- Boundary nodes vs. interior nodes
- High-rigidity neighborhoods vs. low-rigidity neighborhoods
- Mechanically "important" nodes (high betweenness in force transmission)

**Probe 3: Subgraph Activation Analysis**
For each hidden dimension at each layer:
1. Run the model on many graphs
2. Find which nodes have the highest activation in that dimension
3. Extract the 2-hop subgraph around those nodes
4. Cluster the top-activating subgraphs

This reveals the "motifs" the GNN has learned -- for example, dimension 17 might
activate on star-like subgraphs where one rigid spring is surrounded by soft ones
(a hinge pattern).

**Probe 4: GNNExplainer**
PyTorch Geometric's GNNExplainer learns a soft mask over edges that identifies the
minimal subgraph sufficient to explain a prediction. This is complementary to
gradient-based saliency: GNNExplainer finds sufficient structure (which edges can
you keep and still get the same prediction?), while saliency finds necessary
structure (which edges most influence the prediction?).

---

## 7. Handling Diverse Topologies

**Question**: The original project only uses perturbed hexagonal lattices. How do we
generalize to arbitrary graph structures?

**Answer**: We expand the training data to include 4 categories of topologies:

### Category 1: Hexagonal Family (existing)
Regular and perturbed hexagonal lattices. All coordination-6. These are well-understood
and serve as the baseline.

### Category 2: Fully Random Triangulations
- **Poisson-Delaunay**: Uniform random points -> Delaunay triangulation. No lattice
  structure at all. Coordination varies from 4 to 9+ per vertex. Some triangles are
  tiny, others large. This tests the GNN's ability to handle truly irregular graphs.
- **Blue noise**: Poisson disk sampling (minimum distance constraint) -> Delaunay.
  More uniform than pure random but not periodic. Coordination 5-7.
- **Clustered**: Gaussian mixture model point process -> Delaunay. Dense regions have
  coordination 10-12+, sparse bridge regions have 3-4. Tests handling of multi-scale graphs.
- **Gradient density**: Spatially varying point density -> Delaunay. Coordination changes
  smoothly from one side of the mesh to the other.

### Category 3: Lattices with Non-Trivial Basis
Periodic lattices with multiple atoms per unit cell produce fundamentally different
graph structures:
- **Kagome** (z=4): corner-sharing triangles with hexagonal voids. At isostaticity.
- **Square** (z=4): simplest non-triangular lattice. At isostaticity.
- **Snub square** (z=5): Archimedean tiling (3.3.4.3.4). Above isostaticity.
- **Truncated hexagonal** (z=3): hexagons + triangles. Sub-isostatic.
- **Cairo pentagonal** (z=3-4): exotic pentagonal tiling.

### Category 4: Non-Triangulated with Soft-Edge Regularization
The forward solver requires triangulated meshes. For inherently non-triangulated lattices
(honeycomb, square grid), we add a center vertex inside each non-triangular face and
connect it to all face vertices with very soft springs (k_soft/k_hard = 1e-3).

The soft springs make the mesh triangulated and mechanically stable while contributing
<0.1% to the elastic response. In the CVAE, soft edges are frozen (not designable).

**Why honeycomb matters**: Honeycomb lattices (coordination 3) are strongly sub-isostatic
and can exhibit exotic elastic behavior including strongly negative Poisson's ratios.
They're the natural model for graphene and other 2D materials. By including them in
training, the GNN learns about a fundamentally different mechanical regime than the
coordination-6 triangular lattice.

---

## 8. The Full Pipeline

```
                    TRAINING
                    ========

    Topology    Edge Params     Forward Solver
    Generator   Sampler         (Phase 2)
       |            |              |
       v            v              v
    [graph]  +  [k, l0]  ----->  [nu, E, C]
                                    |
                                    v
              Training Data: {(graph, k, l0, nu)}
                     |
           +---------+---------+
           |                   |
    GNN Surrogate        CVAE Inverse
    (graph, k, l0)->nu   (graph, z, nu)->k, l0


                   INFERENCE
                   =========

    nu_target  +  graph_topology  +  z ~ N(0,I)
                        |
                        v
                   CVAE Decoder
                        |
                        v
                   k_pred, l0_pred
                        |
              +---------+---------+
              |                   |
     (optional)              Direct use
     L-BFGS refine
     via Phase 2 solver
              |
              v
     k_refined, l0_refined
```

---

## 9. Design Variable Masking

From Phase 3, three optimization modes exist:
1. **Rigidities only**: vary k, keep l0 = l_actual
2. **Rest lengths only**: vary l0, keep k = 1
3. **Both**: vary k and l0 independently

The CVAE handles all three through a single architecture with masking:
- Decoder always outputs 2 values per edge (raw_k, raw_l0)
- In "rigidities only" mode: l0 is fixed to l_actual, only k gradients flow
- In "rest lengths only" mode: k is fixed to 1.0, only l0 gradients flow
- In "both" mode: both channels are active

This avoids training separate models for each mode.

For non-triangulated meshes, there's a second level of masking:
- "Real" edges (original lattice bonds): designable
- "Soft" edges (regularization): frozen at k_soft, l0 = l_actual

---

## 10. Open Questions for Future Work

1. **Latent space structure**: What do different regions of z-space correspond to
   physically? Can we identify disentangled dimensions that control specific
   design aspects (e.g., one dimension controls the degree of re-entrant hinging,
   another controls the overall stiffness)?

2. **Transfer across topologies**: If we train the CVAE on one topology (e.g.,
   foam_eta02) and sample designs for another (e.g., honeycomb), do the latent
   codes transfer? This would suggest universal design principles.

3. **Active learning**: Can we identify which topology + edge parameter combinations
   are most informative for the GNN and preferentially generate those for training?

4. **Multi-objective design**: Extend the CVAE to condition on multiple targets
   simultaneously (e.g., nu = -0.5 AND E = 10.0).

5. **Scalability**: How does performance degrade as mesh size increases from
   10x10 (~944 triangles) to 50x50 (~25,000 triangles)?
