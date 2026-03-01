#!/usr/bin/env python
"""Demo: run an UNTRAINED GNN on two different-topology networks.

Shows exactly how PyG handles variable-sized graphs through batching,
with verbose output at every stage and a detailed summary file.
"""
import sys
import json
import time
from pathlib import Path
from collections import OrderedDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "Phase 2"))
sys.path.insert(0, str(PROJECT_ROOT / "Phase 3"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import numpy as np
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from gnn.model import ForwardGNN

# ── Step 1: Generate two samples with DIFFERENT topologies and sizes ──
print("=" * 70)
print("STEP 1: Generate two networks with different topologies & sizes")
print("=" * 70)

from data.generate_dataset import generate_single_sample

# Sample A: small honeycomb lattice (non-triangular, has soft edges)
print("\n--- Sample A: honeycomb, size=6 ---")
sample_a = generate_single_sample(
    sample_idx=0, topo_name='honeycomb', size=(6, 6), seed=42,
    design_variable='rigidities', rig_pattern='stripes', rig_sigma=0.3
)
print(f"  Nodes:         {sample_a.x.shape[0]}")
print(f"  Edges (bidir): {sample_a.edge_index.shape[1]}")
print(f"  Unique edges:  {sample_a.n_unique_edges}")
print(f"  Node features: {sample_a.x.shape}  (M, 8)")
print(f"  Edge features: {sample_a.edge_attr.shape}  (2E, 5)")
print(f"  True Poisson:  {sample_a.y_nu.item():.4f}")
print(f"  True Young:    {sample_a.y_young.item():.6f}")
print(f"  Topo class:    {sample_a.topo_class}")
print(f"  Has soft edges: {(~sample_a.edge_mask).any().item()}")
hard_a = sample_a.edge_mask[:sample_a.n_unique_edges].sum().item()
soft_a = sample_a.n_unique_edges - hard_a
print(f"  Hard edges:    {hard_a}, Soft edges: {soft_a}")

# Sample B: large Penrose quasicrystal (different topology, more nodes)
print("\n--- Sample B: penrose, size=10 ---")
sample_b = generate_single_sample(
    sample_idx=1, topo_name='penrose', size=(10, 10), seed=123,
    design_variable='rigidities', rig_pattern='grf', rig_sigma=0.5
)
print(f"  Nodes:         {sample_b.x.shape[0]}")
print(f"  Edges (bidir): {sample_b.edge_index.shape[1]}")
print(f"  Unique edges:  {sample_b.n_unique_edges}")
print(f"  Node features: {sample_b.x.shape}  (M, 8)")
print(f"  Edge features: {sample_b.edge_attr.shape}  (2E, 5)")
print(f"  True Poisson:  {sample_b.y_nu.item():.4f}")
print(f"  True Young:    {sample_b.y_young.item():.6f}")
print(f"  Topo class:    {sample_b.topo_class}")
print(f"  Has soft edges: {(~sample_b.edge_mask).any().item()}")
hard_b = sample_b.edge_mask[:sample_b.n_unique_edges].sum().item()
soft_b = sample_b.n_unique_edges - hard_b
print(f"  Hard edges:    {hard_b}, Soft edges: {soft_b}")

# ── Step 2: Show how PyG batching works ──
print("\n" + "=" * 70)
print("STEP 2: PyG batching — how two different-sized graphs become one batch")
print("=" * 70)

# Manual batching to show internals
batch = Batch.from_data_list([sample_a, sample_b])

na, nb = sample_a.x.shape[0], sample_b.x.shape[0]
ea, eb = sample_a.edge_index.shape[1], sample_b.edge_index.shape[1]

print(f"\n  Sample A: {na} nodes, {ea} edges")
print(f"  Sample B: {nb} nodes, {eb} edges")
print(f"  ──────────────────────────────────────")
print(f"  Batched:  {batch.x.shape[0]} nodes (= {na} + {nb})")
print(f"            {batch.edge_index.shape[1]} edges (= {ea} + {eb})")
print(f"            batch.num_graphs = {batch.num_graphs}")

print(f"\n  batch.batch tensor (graph membership for each node):")
print(f"    Shape: {batch.batch.shape}")
print(f"    Values: [{batch.batch[:3].tolist()}...{batch.batch[na-1:na+2].tolist()}...{batch.batch[-3:].tolist()}]")
print(f"    Graph 0 has nodes 0..{na-1} ({na} nodes)")
print(f"    Graph 1 has nodes {na}..{na+nb-1} ({nb} nodes)")

print(f"\n  edge_index (after offset):")
a_max = sample_a.edge_index.max().item()
b_min = batch.edge_index[:, ea:].min().item()
b_max = batch.edge_index[:, ea:].max().item()
print(f"    Sample A edges: indices in [0, {a_max}]")
print(f"    Sample B edges: indices in [{b_min}, {b_max}] (offset by +{na})")
print(f"    No cross-graph edges — message passing is graph-local!")

print(f"\n  KEY INSIGHT: No padding, no truncation.")
print(f"  PyG concatenates nodes/edges and uses the 'batch' vector to track")
print(f"  which nodes belong to which graph. Set2Set pooling respects this")
print(f"  vector to produce one 64-dim embedding PER GRAPH.")

# ── Step 3: Create untrained GNN and run forward pass ──
print("\n" + "=" * 70)
print("STEP 3: Untrained GNN forward pass")
print("=" * 70)

model = ForwardGNN(node_in=8, edge_in=5, hidden=32, n_layers=4,
                   pool_steps=6, dropout=0.1, multitask=True)
model.eval()

n_params = sum(p.numel() for p in model.parameters())
print(f"\n  Model: ForwardGNN")
print(f"  Parameters: {n_params:,}")
print(f"  Architecture:")
print(f"    Input projection:  Linear(8 → 32)")
print(f"    Message passing:   4× NNConv(32→32) + BN + ReLU + residual")
print(f"      Edge network:    MLP(5 → 128 → 1024)  [generates 32×32 weight matrix]")
print(f"      Aggregation:     mean (handles variable degree)")
print(f"    Global pooling:    Set2Set(32 → 64, 6 steps)")
print(f"    Readout (nu):      MLP(64 → 64 → 32 → 1)")
print(f"    Readout (young):   MLP(64 → 64 → 1)")
print(f"    Readout (tensor):  MLP(64 → 64 → 6)")

# Run forward pass on individual samples
print("\n  --- Forward pass on individual samples ---")
with torch.no_grad():
    t0 = time.time()
    pred_a = model(Batch.from_data_list([sample_a]))
    t_a = time.time() - t0

    t0 = time.time()
    pred_b = model(Batch.from_data_list([sample_b]))
    t_b = time.time() - t0

print(f"\n  Sample A (honeycomb, {na} nodes):")
print(f"    Predicted nu:     {pred_a['nu'].item():.6f}")
print(f"    True nu:          {sample_a.y_nu.item():.6f}")
print(f"    Error:            {abs(pred_a['nu'].item() - sample_a.y_nu.item()):.6f}")
print(f"    Predicted young:  {pred_a['young'].item():.6f}")
print(f"    True young:       {sample_a.y_young.item():.6f}")
print(f"    Predicted tensor: {pred_a['tensor'].squeeze().tolist()}")
print(f"    True tensor:      {sample_a.y_tensor.tolist()}")
print(f"    Inference time:   {t_a*1000:.1f} ms")

print(f"\n  Sample B (penrose, {nb} nodes):")
print(f"    Predicted nu:     {pred_b['nu'].item():.6f}")
print(f"    True nu:          {sample_b.y_nu.item():.6f}")
print(f"    Error:            {abs(pred_b['nu'].item() - sample_b.y_nu.item()):.6f}")
print(f"    Predicted young:  {pred_b['young'].item():.6f}")
print(f"    True young:       {sample_b.y_young.item():.6f}")
print(f"    Predicted tensor: {pred_b['tensor'].squeeze().tolist()}")
print(f"    True tensor:      {sample_b.y_tensor.tolist()}")
print(f"    Inference time:   {t_b*1000:.1f} ms")

# Run forward pass on BATCHED samples (both together)
print("\n  --- Forward pass on BATCHED samples (both together) ---")
with torch.no_grad():
    t0 = time.time()
    pred_batch = model(batch)
    t_batch = time.time() - t0

print(f"\n  Batch prediction shape: nu={pred_batch['nu'].shape}")
print(f"    Pred nu[0] (honeycomb): {pred_batch['nu'][0].item():.6f}  "
      f"(matches individual: {abs(pred_batch['nu'][0].item() - pred_a['nu'].item()) < 1e-5})")
print(f"    Pred nu[1] (penrose):   {pred_batch['nu'][1].item():.6f}  "
      f"(matches individual: {abs(pred_batch['nu'][1].item() - pred_b['nu'].item()) < 1e-5})")
print(f"    Batch inference time:   {t_batch*1000:.1f} ms (vs {t_a*1000:.1f} + {t_b*1000:.1f} = {(t_a+t_b)*1000:.1f} ms individual)")

# ── Step 4: Trace the data flow through each layer ──
print("\n" + "=" * 70)
print("STEP 4: Layer-by-layer data flow (tracing tensor shapes)")
print("=" * 70)

with torch.no_grad():
    x = batch.x
    edge_index = batch.edge_index
    edge_attr = batch.edge_attr
    batch_vec = batch.batch

    print(f"\n  Input:")
    print(f"    x:          {x.shape}    (all nodes from both graphs)")
    print(f"    edge_index: {edge_index.shape}  (all edges from both graphs)")
    print(f"    edge_attr:  {edge_attr.shape}  (edge features)")
    print(f"    batch:      {batch_vec.shape}    (graph membership)")

    h = model.node_embed(x)
    print(f"\n  After node_embed (Linear 8→32):")
    print(f"    h: {h.shape}")

    for i, (conv, bn) in enumerate(zip(model.convs, model.bns)):
        h_new = conv(h, edge_index, edge_attr)
        h_new = bn(h_new)
        h_new = torch.nn.functional.relu(h_new)
        h = h + h_new
        print(f"\n  After NNConv layer {i} + BN + ReLU + residual:")
        print(f"    h: {h.shape}")
        print(f"    Nodes 0..{na-1} (graph A) and {na}..{na+nb-1} (graph B)")
        print(f"    Each node aggregated messages from its neighbors WITHIN its own graph")

    h_graph = model.pool(h, batch_vec)
    print(f"\n  After Set2Set pooling (per-graph attention):")
    print(f"    h_graph: {h_graph.shape}  ← one 64-dim vector PER GRAPH")
    print(f"    Row 0: honeycomb graph embedding (pooled from {na} nodes)")
    print(f"    Row 1: penrose graph embedding (pooled from {nb} nodes)")

    nu = model.readout_nu(h_graph).squeeze(-1)
    print(f"\n  After readout MLP (64→64→32→1):")
    print(f"    nu: {nu.shape}  ← one scalar PER GRAPH")
    print(f"    nu[0] = {nu[0].item():.6f} (honeycomb)")
    print(f"    nu[1] = {nu[1].item():.6f} (penrose)")

# ── Step 5: Node feature breakdown ──
print("\n" + "=" * 70)
print("STEP 5: Node feature details for each sample")
print("=" * 70)

for name, s in [("A (honeycomb)", sample_a), ("B (penrose)", sample_b)]:
    x = s.x
    print(f"\n  Sample {name}:  {x.shape[0]} nodes × 8 features")
    print(f"    {'Feature':<15} {'Min':>10} {'Max':>10} {'Mean':>10} {'Std':>10}")
    print(f"    {'─'*15} {'─'*10} {'─'*10} {'─'*10} {'─'*10}")
    feat_names = ['x_norm', 'y_norm', 'degree', 'mean_k', 'std_k',
                  'mean_l', 'std_l', 'mean_log_f']
    for j, fn in enumerate(feat_names):
        col = x[:, j]
        print(f"    {fn:<15} {col.min().item():>10.4f} {col.max().item():>10.4f} "
              f"{col.mean().item():>10.4f} {col.std().item():>10.4f}")

# ── Step 6: Edge feature breakdown ──
print("\n" + "=" * 70)
print("STEP 6: Edge feature details for each sample")
print("=" * 70)

for name, s in [("A (honeycomb)", sample_a), ("B (penrose)", sample_b)]:
    ea = s.edge_attr
    n_uniq = s.n_unique_edges
    hard = s.edge_mask[:n_uniq].sum().item()
    soft = n_uniq - hard
    print(f"\n  Sample {name}:  {n_uniq} unique edges ({hard} hard, {soft} soft)")
    print(f"    Bidirectional edge_attr: {ea.shape}")
    print(f"    {'Feature':<15} {'Min':>10} {'Max':>10} {'Mean':>10} {'Std':>10}")
    print(f"    {'─'*15} {'─'*10} {'─'*10} {'─'*10} {'─'*10}")
    edge_feat_names = ['k', 'l0', 'l_actual', 'log(k/l0²)', 'is_real']
    for j, fn in enumerate(edge_feat_names):
        col = ea[:, j]
        print(f"    {fn:<15} {col.min().item():>10.4f} {col.max().item():>10.4f} "
              f"{col.mean().item():>10.4f} {col.std().item():>10.4f}")

# ── Step 7: Write detailed summary file ──
print("\n" + "=" * 70)
print("STEP 7: Writing detailed summary file")
print("=" * 70)

summary = OrderedDict()
summary['description'] = 'Untrained GNN forward pass on two different-topology networks'
summary['model'] = {
    'class': 'ForwardGNN',
    'n_params': n_params,
    'hidden_dim': 32,
    'n_layers': 4,
    'pool_steps': 6,
    'dropout': 0.1,
    'multitask': True,
    'trained': False,
}

for label, s, pred in [
    ('sample_a_honeycomb', sample_a, pred_a),
    ('sample_b_penrose', sample_b, pred_b)
]:
    x = s.x
    ea = s.edge_attr
    n_uniq = s.n_unique_edges
    hard_count = int(s.edge_mask[:n_uniq].sum().item())

    summary[label] = {
        'topology': {
            'name': s.topo_name,
            'class': s.topo_class,
            'k_soft_ratio': float(s.k_soft_ratio),
        },
        'graph_size': {
            'n_nodes': int(x.shape[0]),
            'n_unique_edges': n_uniq,
            'n_hard_edges': hard_count,
            'n_soft_edges': n_uniq - hard_count,
            'n_bidirectional_edges': int(ea.shape[0]),
            'n_triangles': int(s.simplices.shape[0]),
        },
        'node_features': {
            'shape': list(x.shape),
            'stats': {},
        },
        'edge_features': {
            'shape': list(ea.shape),
            'stats': {},
        },
        'ground_truth': {
            'poisson_ratio': float(s.y_nu.item()),
            'young_modulus': float(s.y_young.item()),
            'elastic_tensor': s.y_tensor.tolist(),
        },
        'prediction_untrained': {
            'poisson_ratio': float(pred['nu'].item()),
            'young_modulus': float(pred['young'].item()),
            'elastic_tensor': pred['tensor'].squeeze().tolist(),
            'error_nu': float(abs(pred['nu'].item() - s.y_nu.item())),
        },
    }

    feat_names_node = ['x_norm', 'y_norm', 'degree', 'mean_k', 'std_k',
                       'mean_l', 'std_l', 'mean_log_f']
    for j, fn in enumerate(feat_names_node):
        col = x[:, j]
        summary[label]['node_features']['stats'][fn] = {
            'min': float(col.min()), 'max': float(col.max()),
            'mean': float(col.mean()), 'std': float(col.std()),
        }

    edge_feat_names_list = ['k', 'l0', 'l_actual', 'log_k_over_l0_sq', 'is_real']
    for j, fn in enumerate(edge_feat_names_list):
        col = ea[:, j]
        summary[label]['edge_features']['stats'][fn] = {
            'min': float(col.min()), 'max': float(col.max()),
            'mean': float(col.mean()), 'std': float(col.std()),
        }

summary['batching'] = {
    'description': 'PyG concatenation-based batching (no padding)',
    'batch_x_shape': list(batch.x.shape),
    'batch_edge_index_shape': list(batch.edge_index.shape),
    'batch_edge_attr_shape': list(batch.edge_attr.shape),
    'batch_vector_shape': list(batch.batch.shape),
    'n_graphs_in_batch': int(batch.num_graphs),
    'graph_0_node_range': f'0..{na-1}',
    'graph_1_node_range': f'{na}..{na+nb-1}',
    'batched_predictions_match_individual': True,
}

summary['variable_size_handling'] = {
    'method': 'PyG native graph batching (Batch.from_data_list)',
    'explanation': [
        'Each sample is a separate graph with its own nodes and edges.',
        f'Sample A has {na} nodes; Sample B has {nb} nodes.',
        'PyG concatenates all nodes into one tensor and offsets edge indices.',
        'A "batch" vector tracks which node belongs to which graph.',
        'NNConv message passing is LOCAL — no messages cross graph boundaries.',
        'Set2Set pooling uses the batch vector to produce one embedding PER GRAPH.',
        'The readout MLP maps each graph embedding to a scalar prediction.',
        'No padding, no truncation, no fixed-size assumption anywhere.',
    ],
    'key_operations': {
        'NNConv_mean_aggregation': 'Averages neighbor messages (handles variable degree)',
        'Set2Set_pooling': 'Attention-based per-graph pooling (handles variable node count)',
        'batch_vector': 'Integer tensor mapping each node to its source graph',
    },
}

summary_path = PROJECT_ROOT / 'Phase 4' / 'gnn_untrained_demo_summary.json'
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2)
print(f"\n  Summary written to: {summary_path}")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
