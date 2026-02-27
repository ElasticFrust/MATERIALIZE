"""Evaluation and error analysis for the GNN forward surrogate.

Provides per-topology breakdown, scatter plots, and error distribution analysis.
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from collections import defaultdict

try:
    from torch_geometric.loader import DataLoader
except ImportError:
    raise ImportError("torch_geometric required: pip install torch-geometric")

from gnn.model import ForwardGNN


@torch.no_grad()
def detailed_evaluate(model, loader, device='cpu'):
    """Run model on all data, collecting per-sample predictions and metadata.

    Returns:
        dict with arrays: nu_pred, nu_true, topo_class, etc.
    """
    model.eval()
    all_pred = []
    all_true = []
    all_topo = []

    for batch in loader:
        batch = batch.to(device)
        pred = model(batch)
        all_pred.append(pred['nu'].cpu().numpy())
        all_true.append(batch.y_nu.cpu().numpy())
        # Topology class is stored per-graph
        if hasattr(batch, 'topo_class'):
            all_topo.extend(batch.topo_class)

    return {
        'nu_pred': np.concatenate(all_pred),
        'nu_true': np.concatenate(all_true),
        'topo_class': all_topo if all_topo else None,
    }


def per_topology_metrics(results):
    """Compute MAE and RMSE broken down by topology class.

    Args:
        results: dict from detailed_evaluate.

    Returns:
        dict mapping topo_class -> {mae, rmse, n_samples}.
    """
    if results['topo_class'] is None:
        return {}

    metrics = defaultdict(lambda: {'errors': []})
    for i, topo in enumerate(results['topo_class']):
        err = results['nu_pred'][i] - results['nu_true'][i]
        metrics[topo]['errors'].append(err)

    summary = {}
    for topo, data in metrics.items():
        errors = np.array(data['errors'])
        summary[topo] = {
            'mae': np.abs(errors).mean(),
            'rmse': np.sqrt((errors ** 2).mean()),
            'n_samples': len(errors),
            'mean_error': errors.mean(),
        }

    return summary


def print_evaluation_report(model, data_path, device='cpu', batch_size=64):
    """Load test data, evaluate, and print a summary report."""
    data_path = Path(data_path)
    test_data = torch.load(data_path, weights_only=False)
    loader = DataLoader(test_data, batch_size=batch_size)

    results = detailed_evaluate(model, loader, device)

    errors = results['nu_pred'] - results['nu_true']
    abs_errors = np.abs(errors)

    print("=" * 60)
    print("GNN Forward Surrogate — Evaluation Report")
    print("=" * 60)
    print(f"  Samples:     {len(errors)}")
    print(f"  MAE:         {abs_errors.mean():.4f}")
    print(f"  RMSE:        {np.sqrt((errors**2).mean()):.4f}")
    print(f"  Max error:   {abs_errors.max():.4f}")
    print(f"  Median err:  {np.median(abs_errors):.4f}")
    print(f"  95th pctile: {np.percentile(abs_errors, 95):.4f}")
    print()

    topo_metrics = per_topology_metrics(results)
    if topo_metrics:
        print("Per-topology breakdown:")
        print(f"  {'Topology':20s} {'N':>6s} {'MAE':>8s} {'RMSE':>8s}")
        print("  " + "-" * 44)
        for topo in sorted(topo_metrics.keys()):
            m = topo_metrics[topo]
            print(f"  {topo:20s} {m['n_samples']:6d} "
                  f"{m['mae']:8.4f} {m['rmse']:8.4f}")

    return results
