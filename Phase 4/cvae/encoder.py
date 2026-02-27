"""CVAE Encoder: compresses (graph + edge params + nu_target) -> latent (mu, logvar).

The encoder sees the full solution during training and must extract a compact
latent code that captures what's unique about this particular solution (vs the
many other valid solutions for the same target).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import NNConv, BatchNorm, Set2Set
except ImportError:
    raise ImportError("torch_geometric required: pip install torch-geometric")


class CVAEEncoder(nn.Module):
    """GNN-based encoder for the CVAE.

    Processes the full graph (with edge parameters as features) and the target
    condition, producing latent distribution parameters (mu, logvar).

    Args:
        node_in: input node feature dimension.
        edge_in: input edge feature dimension (includes k, l0, etc.).
        hidden: hidden dimension for NNConv layers.
        latent_dim: dimension of the latent code z.
        n_layers: number of message-passing layers.
        pool_steps: Set2Set processing steps.
    """

    def __init__(self, node_in=8, edge_in=5, hidden=64, latent_dim=32,
                 n_layers=4, pool_steps=6):
        super().__init__()

        self.node_embed = nn.Linear(node_in, hidden)

        # NNConv message-passing layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(n_layers):
            edge_nn = nn.Sequential(
                nn.Linear(edge_in, 128),
                nn.ReLU(),
                nn.Linear(128, hidden * hidden),
            )
            self.convs.append(NNConv(hidden, hidden, edge_nn, aggr='mean'))
            self.bns.append(BatchNorm(hidden))

        # Global pooling
        self.pool = Set2Set(hidden, processing_steps=pool_steps)

        # Condition embedding: nu_target -> 32-dim vector
        self.nu_embed = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
        )

        # Latent projection
        self.fc_mu = nn.Linear(2 * hidden + 32, latent_dim)
        self.fc_logvar = nn.Linear(2 * hidden + 32, latent_dim)

    def forward(self, data, nu_target):
        """Encode graph + condition -> latent distribution.

        Args:
            data: PyG Batch with x, edge_index, edge_attr, batch.
            nu_target: (batch_size,) target Poisson ratios.

        Returns:
            mu: (batch_size, latent_dim)
            logvar: (batch_size, latent_dim)
        """
        h = self.node_embed(data.x)

        for conv, bn in zip(self.convs, self.bns):
            h_new = conv(h, data.edge_index, data.edge_attr)
            h_new = bn(h_new)
            h_new = F.relu(h_new)
            h = h + h_new  # residual

        h_graph = self.pool(h, data.batch)  # (batch_size, 2*hidden)
        nu_emb = self.nu_embed(nu_target.unsqueeze(-1))  # (batch_size, 32)
        combined = torch.cat([h_graph, nu_emb], dim=-1)

        mu = self.fc_mu(combined)
        logvar = self.fc_logvar(combined)

        return mu, logvar
