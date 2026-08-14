r"""Phase 5 / M2 — the GNN FORWARD SURROGATE (plain torch, no torch_geometric).

M2 v1 amortises the differentiable solver: input a periodic network graph
(node positions + per-bond stiffness k), output its homogenised elastic tensor
C6 (6 numbers).  nu(theta), E(theta) then derive from C6 exactly as the solver
does (c6_to_nuE / c6_to_nuE_theta).

The GNN is a lightweight message-passing network implemented in PLAIN torch
(index_add_ scatter) so it has NO dependency on torch_geometric:

  edge features per bond : [k, log(k+eps), |bond_R|, dir_x, dir_y]
      - bond_R is the TRUE periodic bond vector (already carries the wrap), so
        the geometry the net sees is the effective periodic connectivity.
      - dir = bond_R / |bond_R| (unit vector); NOT rotation-invariant (v1 ok).
  node features         : node degree (scalar), embedded to `hidden`.
  message passing       : a few rounds of  edge MLP([h_u,h_v,e]) -> scatter-mean
                          to nodes -> node update MLP (residual).
  pooling               : global mean + sum over each graph's nodes.
  head                  : MLP -> C6 (6,).

Batching is a plain "batch-of-graphs" index scheme: many graphs are concatenated
into one big graph and a `batch` vector maps each node to its graph id (see
collate() in train.py / the Batch dataclass below).
"""
import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# reuse the solver stack's C6 -> nu,E helpers (identical formula to the solver) -----------------
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'Phase 3', 'verifications'))
import _common as C                                    # noqa: F401  (sets up sys.path)
from inverse_design import c6_to_nuE, c6_to_nuE_theta, ANG
torch.set_default_dtype(torch.float64)

EPS = 1e-8


# ---- graph -> edge/node feature construction -------------------------------------------------
def build_edge_features(bond_R, k):
    """Directed edge features for an undirected periodic graph.

    Each bond (u,v) with periodic vector bond_R and stiffness k becomes TWO
    directed edges (u->v with +R, v->u with -R) so messages flow both ways.

    Returns (edge_index (2,2M) long, edge_attr (2M,5)):
        edge_attr = [k, log(k+eps), length, dir_x, dir_y]
    """
    bond_R = torch.as_tensor(bond_R, dtype=torch.float64)
    k = torch.as_tensor(k, dtype=torch.float64).reshape(-1)
    length = torch.linalg.norm(bond_R, dim=1).clamp_min(EPS)
    d = bond_R / length[:, None]
    logk = torch.log(k + EPS)
    fwd = torch.stack([k, logk, length, d[:, 0], d[:, 1]], dim=1)
    bwd = torch.stack([k, logk, length, -d[:, 0], -d[:, 1]], dim=1)
    edge_attr = torch.cat([fwd, bwd], dim=0)
    return edge_attr, length


def node_degree_feature(bond_u, bond_v, n_nodes):
    """Per-node degree (counting each undirected bond once at both ends), as (n_nodes,1)."""
    deg = torch.zeros(n_nodes, dtype=torch.float64)
    u = torch.as_tensor(bond_u, dtype=torch.long)
    v = torch.as_tensor(bond_v, dtype=torch.long)
    deg.index_add_(0, u, torch.ones_like(u, dtype=torch.float64))
    deg.index_add_(0, v, torch.ones_like(v, dtype=torch.float64))
    return deg[:, None]


# ---- the message-passing surrogate -----------------------------------------------------------
def _mlp(sizes, act=nn.SiLU):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act())
    return nn.Sequential(*layers)


class ForwardGNN(nn.Module):
    """Periodic message-passing GNN: graph (pts+k) -> C6 (6,).  Plain torch scatter."""

    def __init__(self, node_in=1, edge_in=5, hidden=64, n_layers=4, head_hidden=128, out_dim=6):
        super().__init__()
        self.hidden = hidden
        self.node_embed = _mlp([node_in, hidden, hidden])
        self.edge_mlps = nn.ModuleList(
            [_mlp([2 * hidden + edge_in, hidden, hidden]) for _ in range(n_layers)])
        self.node_mlps = nn.ModuleList(
            [_mlp([2 * hidden, hidden, hidden]) for _ in range(n_layers)])
        # pooled = mean + sum  ->  2*hidden
        self.head = _mlp([2 * hidden, head_hidden, head_hidden, out_dim])

    def forward(self, node_feat, edge_index, edge_attr, batch, num_graphs):
        """node_feat (N,node_in), edge_index (2,E) [src,dst], edge_attr (E,edge_in),
        batch (N,) graph id per node, num_graphs int  ->  C6 (num_graphs, out_dim)."""
        src, dst = edge_index[0], edge_index[1]
        h = self.node_embed(node_feat)                             # (N, hidden)
        n_nodes = h.shape[0]
        for edge_mlp, node_mlp in zip(self.edge_mlps, self.node_mlps):
            m = edge_mlp(torch.cat([h[src], h[dst], edge_attr], dim=1))   # (E, hidden) message src->dst
            agg = torch.zeros(n_nodes, self.hidden, dtype=h.dtype)
            agg.index_add_(0, dst, m)                              # sum messages into dst
            cnt = torch.zeros(n_nodes, 1, dtype=h.dtype)
            cnt.index_add_(0, dst, torch.ones(dst.shape[0], 1, dtype=h.dtype))
            agg = agg / cnt.clamp_min(1.0)                         # mean aggregation
            h = h + node_mlp(torch.cat([h, agg], dim=1))           # residual node update
        # global mean + sum pool per graph
        mean = torch.zeros(num_graphs, self.hidden, dtype=h.dtype)
        summ = torch.zeros(num_graphs, self.hidden, dtype=h.dtype)
        ncnt = torch.zeros(num_graphs, 1, dtype=h.dtype)
        summ.index_add_(0, batch, h)
        ncnt.index_add_(0, batch, torch.ones(n_nodes, 1, dtype=h.dtype))
        mean = summ / ncnt.clamp_min(1.0)
        pooled = torch.cat([mean, summ], dim=1)                    # (num_graphs, 2*hidden)
        return self.head(pooled)


# ---- C6 <-> nu,E reporting helpers -----------------------------------------------------------
def c6_pred_to_nuE(C6):
    """(scalar) nu, E from a predicted 6-vector (numpy or torch)  ->  (float, float)."""
    C6 = torch.as_tensor(np.asarray(C6, float))
    nu, E = c6_to_nuE(C6)
    return float(nu), float(E)


def c6_pred_to_nuE_theta(C6, thetas=ANG):
    """(directional) nu(theta), E(theta) from a predicted 6-vector  ->  (np(37,), np(37,))."""
    C6 = torch.as_tensor(np.asarray(C6, float))
    nu, E = c6_to_nuE_theta(C6, thetas)
    return nu.detach().numpy(), E.detach().numpy()
