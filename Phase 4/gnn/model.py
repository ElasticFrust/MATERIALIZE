"""GNN forward surrogate: predicts elastic properties from spring network graphs.

Architecture: NNConv (edge-conditioned convolution) MPNN with Set2Set pooling.
Edge features [k, l0, l_actual, log(k/l0^2), is_real] directly parameterize
the message weight matrices, mirroring how k/l0^2 enters the forward solver.
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

    Args:
        node_in: input node feature dimension (default 8).
        edge_in: input edge feature dimension (default 5).
        hidden: hidden dimension for message passing layers.
        n_layers: number of NNConv message-passing layers.
        pool_steps: Set2Set processing steps.
        dropout: dropout rate in readout MLP.
        multitask: if True, adds auxiliary heads for Young's modulus and tensor.
    """

    def __init__(self, node_in=8, edge_in=5, hidden=64, n_layers=4,
                 pool_steps=6, dropout=0.1, multitask=False):
        super().__init__()
        self.multitask = multitask

        # Input projection
        self.node_embed = nn.Linear(node_in, hidden)

        # Message-passing layers (NNConv with residual connections)
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(n_layers):
            # Edge network: maps edge features -> hidden*hidden weight matrix
            edge_nn = nn.Sequential(
                nn.Linear(edge_in, 128),
                nn.ReLU(),
                nn.Linear(128, hidden * hidden),
            )
            self.convs.append(NNConv(hidden, hidden, edge_nn, aggr='mean'))
            self.bns.append(BatchNorm(hidden))

        # Global pooling: Set2Set (attention-based, outputs 2*hidden)
        self.pool = Set2Set(hidden, processing_steps=pool_steps)

        # Primary readout: Poisson's ratio
        self.readout_nu = nn.Sequential(
            nn.Linear(2 * hidden, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        # Auxiliary heads (optional, shared backbone)
        if multitask:
            self.readout_young = nn.Sequential(
                nn.Linear(2 * hidden, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )
            self.readout_tensor = nn.Sequential(
                nn.Linear(2 * hidden, 64),
                nn.ReLU(),
                nn.Linear(64, 6),
            )

    def forward(self, data):
        """Forward pass.

        Args:
            data: PyG Data/Batch with x, edge_index, edge_attr, batch.

        Returns:
            dict with 'nu' (batch_size,) and optionally 'young', 'tensor'.
        """
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        batch = data.batch

        # Input projection
        h = self.node_embed(x)

        # Message passing with residual connections
        for conv, bn in zip(self.convs, self.bns):
            h_new = conv(h, edge_index, edge_attr)
            h_new = bn(h_new)
            h_new = F.relu(h_new)
            h = h + h_new  # residual

        # Global pooling
        h_graph = self.pool(h, batch)  # (batch_size, 2*hidden)

        # Predictions
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
    """

    def __init__(self, alpha_young=0.1, alpha_tensor=0.01, multitask=False):
        super().__init__()
        self.alpha_young = alpha_young
        self.alpha_tensor = alpha_tensor
        self.multitask = multitask

    def forward(self, pred, data):
        loss = F.mse_loss(pred['nu'], data.y_nu)

        if self.multitask and 'young' in pred:
            loss = loss + self.alpha_young * F.mse_loss(pred['young'], data.y_young)
        if self.multitask and 'tensor' in pred:
            loss = loss + self.alpha_tensor * F.mse_loss(pred['tensor'], data.y_tensor)

        return loss
