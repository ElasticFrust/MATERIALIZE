"""GNN forward surrogate: predicts elastic properties from spring network graphs.

This is the neural network that replaces the expensive numerical forward solver
(Phase 2's ElasticSolver). Given a spring network graph with per-edge parameters,
it predicts the macroscopic elastic properties (Poisson's ratio, Young's modulus,
full elastic tensor) in a single forward pass — roughly 1000x faster than the solver.

Architecture: NNConv (edge-conditioned convolution) MPNN with Set2Set pooling.

Why NNConv over GCN or GAT?
────────────────────────────
The forward solver's physics is dominated by the PER-EDGE factor k/l0^2. Each
edge contributes differently to the elastic tensor based on its spring constant
and rest length. NNConv generates weight matrices from edge features, so each
spring's physical parameters directly modulate the message flow:

    message_{j→i} = W(edge_attr_{ij}) · h_j

where W() is a learnable function (small MLP) of the edge features. This mirrors
how k/l0^2 enters the analytical elastic tensor formula:

    C_ij ~ Σ_edges (k_e / l0_e^2) * (n_e ⊗ n_e)_ij

In contrast, GCN ignores edge features entirely, and GAT uses scalar attention
weights that can't encode the rich edge physics (5-dim features).

The edge feature vector [k, l0, l_actual, log(k/l0^2), is_real] is designed to
give NNConv direct access to the physics-relevant quantities. The log(k/l0^2)
term is the dominant factor — including it as a feature means the edge network
doesn't need to "discover" this relationship from k and l0 separately.

Set2Set pooling (vs mean/max/sum):
──────────────────────────────────
Elastic properties depend on the GLOBAL distribution of edge stiffnesses, not just
local neighborhoods. Set2Set uses an attention mechanism over 6 processing steps
to produce a permutation-invariant graph-level representation that captures
higher-order statistics (variance, skewness) beyond what mean pooling can represent.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import NNConv, BatchNorm, Set2Set, global_mean_pool
except ImportError:
    raise ImportError("torch_geometric required: pip install torch-geometric")


class ForwardGNN(nn.Module):
    """Predicts Poisson's ratio (and optionally Young's modulus, full tensor)
    from a spring network graph.

    Network architecture (with default hyperparameters):

        Input:    node_feat (M, 8)  +  edge_attr (E, 5)
                           │
                  Linear(8 → 64)            ← input projection
                           │
          ┌─── 4× NNConv(64→64) + BN + ReLU + residual ───┐
          │    edge_nn: MLP(5→128→64×64)                    │
          │    Each layer generates per-edge weight matrices  │
          │    from the 5-dim edge features                   │
          └──────────────────────────────────────────────────┘
                           │
                  Set2Set pooling (6 steps) → (batch, 128)
                           │
                  MLP: 128 → 64 → 32 → 1    ← readout head
                           │
                  Output: ν_pred (scalar per graph)

    The residual connections in the message-passing layers help with gradient
    flow and allow the network to learn identity mappings for layers that
    aren't needed (e.g., shallow topologies may not require 4 layers).

    Args:
        node_in: input node feature dimension (default 8, from graph_utils.init_node_features).
        edge_in: input edge feature dimension (default 5: [k, l0, l_actual, log_factor, is_real]).
        hidden: hidden dimension for all message passing layers. Default 64.
        n_layers: number of NNConv message-passing layers. Default 4 (receptive field
                  covers ~4-hop neighborhoods, sufficient for most mesh sizes).
        pool_steps: Set2Set processing steps. More steps = richer global representation.
        dropout: dropout rate in the readout MLP (regularization).
        multitask: if True, adds auxiliary prediction heads for Young's modulus (scalar)
                   and the full elastic tensor (6 components). The shared backbone
                   learns features useful for all three tasks.
    """

    def __init__(self, node_in=8, edge_in=5, hidden=64, n_layers=4,
                 pool_steps=6, dropout=0.1, multitask=False):
        super().__init__()
        self.multitask = multitask

        # Input projection: map 8-dim node features to hidden dimension
        self.node_embed = nn.Linear(node_in, hidden)

        # Message-passing layers: NNConv with residual connections and batch normalization
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(n_layers):
            # Edge network: maps 5-dim edge features → hidden×hidden weight matrix.
            # This is the key to NNConv: each edge gets its OWN weight matrix
            # generated from its physical parameters (k, l0, etc.).
            # Output size = hidden * hidden because the weight matrix is (hidden, hidden).
            edge_nn = nn.Sequential(
                nn.Linear(edge_in, 128),
                nn.ReLU(),
                nn.Linear(128, hidden * hidden),
            )
            # aggr='mean': average messages from all neighbors (vs sum or max).
            # Mean is more stable for variable-degree nodes.
            self.convs.append(NNConv(hidden, hidden, edge_nn, aggr='mean'))
            self.bns.append(BatchNorm(hidden))

        # Global pooling: Set2Set (attention-based, outputs 2*hidden).
        # The factor of 2 comes from Set2Set's read-then-write mechanism
        # that produces [read_vector, write_vector] concatenated.
        self.pool = Set2Set(hidden, processing_steps=pool_steps)

        # Primary readout MLP: graph embedding → Poisson's ratio (scalar)
        self.readout_nu = nn.Sequential(
            nn.Linear(2 * hidden, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        # Auxiliary heads (optional): shared backbone, separate readout MLPs.
        # Multi-task training can improve the backbone's representations
        # since nu, Young's modulus, and the tensor are related but different.
        if multitask:
            self.readout_young = nn.Sequential(
                nn.Linear(2 * hidden, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )
            self.readout_tensor = nn.Sequential(
                nn.Linear(2 * hidden, 64),
                nn.ReLU(),
                nn.Linear(64, 6),  # 6 independent components of 2D elastic tensor
            )

    def forward(self, data):
        """Forward pass: graph → elastic property predictions.

        The forward pass has three stages:
          1. Node embedding: project 8-dim node features to hidden dim
          2. Message passing: 4 rounds of NNConv with residual connections.
             After 4 layers, each node's embedding incorporates information
             from its 4-hop neighborhood — sufficient for mesh sizes up to 12×12.
          3. Readout: Set2Set pools all node embeddings into a single graph
             vector, then MLP(s) predict the target properties.

        Args:
            data: PyG Data/Batch with:
                  x (M, 8): node features
                  edge_index (2, 2E): bidirectional edge list
                  edge_attr (2E, 5): edge features
                  batch (M,): graph membership for batched graphs

        Returns:
            dict with 'nu' (batch_size,) and optionally 'young', 'tensor'.
        """
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        batch = data.batch

        # Stage 1: Input projection (8-dim → hidden-dim)
        h = self.node_embed(x)

        # Stage 2: Message passing with residual connections.
        # Each NNConv layer: h_new = BN(ReLU(NNConv(h, edge_index, edge_attr)))
        # Residual: h = h + h_new (allows identity mapping, helps gradient flow)
        for conv, bn in zip(self.convs, self.bns):
            h_new = conv(h, edge_index, edge_attr)
            h_new = bn(h_new)
            h_new = F.relu(h_new)
            h = h + h_new  # residual connection

        # Stage 3: Global pooling → per-graph embedding
        h_graph = self.pool(h, batch)  # (batch_size, 2*hidden)

        # Readout: MLP maps graph embedding to predictions
        result = {'nu': self.readout_nu(h_graph).squeeze(-1)}

        if self.multitask:
            result['young'] = self.readout_young(h_graph).squeeze(-1)
            result['tensor'] = self.readout_tensor(h_graph)

        return result

    def predict_nu(self, data):
        """Convenience: returns just the Poisson ratio prediction."""
        return self.forward(data)['nu']


class ForwardGNNLoss(nn.Module):
    """Combined loss for ForwardGNN training.

    L = L_nu + alpha_young * L_young + alpha_tensor * L_tensor

    The primary target is Poisson's ratio (weight=1.0). Young's modulus and
    the full elastic tensor are auxiliary targets with smaller weights. The
    auxiliary losses act as regularizers: they encourage the backbone to learn
    richer representations that capture more of the elastic physics, even though
    Poisson's ratio is our main prediction target.

    The alpha weights are chosen so that all loss terms contribute roughly
    equally at initialization (before training changes the relative scales).
    """

    def __init__(self, alpha_young=0.1, alpha_tensor=0.01, multitask=False):
        super().__init__()
        self.alpha_young = alpha_young
        self.alpha_tensor = alpha_tensor
        self.multitask = multitask

    def forward(self, pred, data):
        # Primary loss: MSE on Poisson's ratio
        loss = F.mse_loss(pred['nu'], data.y_nu)

        # Auxiliary losses (only when multi-task is enabled)
        if self.multitask and 'young' in pred:
            loss = loss + self.alpha_young * F.mse_loss(pred['young'], data.y_young)
        if self.multitask and 'tensor' in pred:
            loss = loss + self.alpha_tensor * F.mse_loss(pred['tensor'], data.y_tensor)

        return loss
