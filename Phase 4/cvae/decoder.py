"""CVAE Decoder: generates per-edge parameters from (graph topology + z + nu_target).

The decoder is the creative half of the CVAE. Given:
  - A graph topology (positions, edges, connectivity — but NOT k or l0)
  - A latent code z (sampled from prior N(0,I) at inference time)
  - A target Poisson ratio nu_target

...it produces per-edge spring parameters (k, l0) that should yield the
target nu when plugged into the forward solver.

Key design decisions:
  1. Geometric features only: the decoder sees [l_actual, dx, dy, angle] for
     each edge, NOT k or l0 (those are outputs). This prevents trivial
     "copy-the-input" solutions during training.
  2. Additive conditioning: z and nu_target are projected to the hidden
     dimension and added to each node's initial embedding. This broadcasts
     global information (what nu to target, which "style" of solution) to
     every node in the graph.
  3. Edge readout: after message passing, per-edge parameters are predicted
     from [h_i || h_j || edge_geom] — the concatenation of both endpoint
     embeddings and the geometric edge features. This ensures the output
     respects the edge's spatial context.
  4. Softplus activation: k and l0 must be positive (physical constraint).
     softplus(x, beta=5) ≈ max(0, x) for large x, but is smooth at 0.

For non-triangulated meshes, the decoder outputs parameters for ALL edges,
but soft (regularization) edges are overwritten with their fixed values
after decoding. Only hard edges' gradients flow back to the decoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import NNConv, BatchNorm
except ImportError:
    raise ImportError("torch_geometric required: pip install torch-geometric")


class CVAEDecoder(nn.Module):
    """GNN-based decoder that generates per-edge parameters.

    Args:
        latent_dim: dimension of the latent code z.
        geom_node_in: geometric node feature dimension (x, y, degree, mean_length).
        geom_edge_in: geometric edge feature dimension (l_actual, dx, dy, angle).
        hidden: hidden dimension for NNConv layers.
        n_layers: number of message-passing layers.
        edge_out: output per edge (2 = k and l0).
    """

    def __init__(self, latent_dim=32, geom_node_in=4, geom_edge_in=4,
                 hidden=64, n_layers=4, edge_out=2):
        super().__init__()

        # Conditioning projections
        self.z_project = nn.Linear(latent_dim, hidden)
        self.nu_embed = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, hidden),
        )
        self.geom_embed = nn.Linear(geom_node_in, hidden)

        # Decoder GNN: uses geometric edge features only
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(n_layers):
            edge_nn = nn.Sequential(
                nn.Linear(geom_edge_in, 128),
                nn.ReLU(),
                nn.Linear(128, hidden * hidden),
            )
            self.convs.append(NNConv(hidden, hidden, edge_nn, aggr='mean'))
            self.bns.append(BatchNorm(hidden))

        # Edge readout: per-edge MLP from endpoint pair + geometric features
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden + geom_edge_in, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, edge_out),
        )

    def _init_geometric_node_features(self, data):
        """Extract geometric-only node features (no k or l0).

        Returns (total_nodes, 4): [x_norm, y_norm, degree, mean_edge_length]
        """
        pos = data.pos
        batch = data.batch
        edge_index = data.edge_index

        # Normalize positions per-graph
        # Compute per-graph mean/std using scatter
        n_nodes = pos.shape[0]
        batch_size = batch.max().item() + 1

        pos_mean = torch.zeros(batch_size, 2, device=pos.device)
        count = torch.zeros(batch_size, 1, device=pos.device)
        pos_mean.scatter_add_(0, batch.unsqueeze(1).expand(-1, 2), pos)
        count.scatter_add_(0, batch.unsqueeze(1), torch.ones(n_nodes, 1, device=pos.device))
        pos_mean = pos_mean / count.clamp(min=1)
        pos_norm = pos - pos_mean[batch]

        # Degree
        degree = torch.zeros(n_nodes, device=pos.device)
        degree.scatter_add_(0, edge_index[0],
                           torch.ones(edge_index.shape[1], device=pos.device))

        # Mean edge length per node
        if hasattr(data, 'edge_geom'):
            l_actual = data.edge_geom[:, 0]  # first col is l_actual
        else:
            src, dst = edge_index
            vecs = pos[src] - pos[dst]
            l_actual = vecs.norm(dim=-1)

        mean_l = torch.zeros(n_nodes, device=pos.device)
        mean_l.scatter_add_(0, edge_index[0], l_actual)
        safe_deg = degree.clamp(min=1)
        mean_l = mean_l / safe_deg

        return torch.stack([pos_norm[:, 0], pos_norm[:, 1], degree, mean_l], dim=-1)

    def forward(self, data, z, nu_target):
        """Decode latent code + condition -> per-edge parameters.

        Args:
            data: PyG Batch with pos, edge_index, edge_geom, batch.
            z: (batch_size, latent_dim) latent codes.
            nu_target: (batch_size,) target Poisson ratios.

        Returns:
            k: (total_edges,) predicted rigidities (positive, via softplus).
            l0: (total_edges,) predicted rest lengths (positive, via softplus).
        """
        batch = data.batch

        # Geometric node features
        x_geom = self._init_geometric_node_features(data)

        # Initialize node hidden states: geometry + z + nu (additive conditioning)
        h = self.geom_embed(x_geom)
        h = h + self.z_project(z)[batch]  # broadcast z to all nodes
        h = h + self.nu_embed(nu_target.unsqueeze(-1))[batch]  # broadcast nu

        # Geometric edge features
        edge_geom = data.edge_geom  # (total_edges, geom_edge_in)

        # GNN message passing
        for conv, bn in zip(self.convs, self.bns):
            h_new = conv(h, data.edge_index, edge_geom)
            h_new = bn(h_new)
            h_new = F.relu(h_new)
            h = h + h_new  # residual

        # Edge readout: combine endpoint embeddings + geometric features
        src, dst = data.edge_index
        edge_h = torch.cat([h[src], h[dst], edge_geom], dim=-1)
        raw_params = self.edge_mlp(edge_h)  # (total_edges, 2)

        # Softplus for positivity (beta=5 matches Phase 3 convention)
        k = F.softplus(raw_params[:, 0], beta=5.0)
        l0 = F.softplus(raw_params[:, 1], beta=5.0)

        return k, l0
