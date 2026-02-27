"""Data generation pipeline for Phase 4 GNN training.

Generates (graph, edge_params, elastic_properties) tuples by:
1. Picking a topology from the catalog
2. Sampling random edge parameters (rigidities, rest lengths, or both)
3. Running the Phase 2 forward solver to get labels
4. Converting to PyG Data objects

Usage:
    python generate_dataset.py --output_dir ./processed --n_train 80000
"""

import argparse
import numpy as np
import torch
import sys
from pathlib import Path
from multiprocessing import Pool
from functools import partial

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "Phase 2"))
sys.path.insert(0, str(PROJECT_ROOT / "Phase 3"))

from data.topology_generators import generate_topology, TOPOLOGY_GENERATORS
from data.graph_utils import triangulation_to_pyg_data
from forward_solver_torch import from_triangulation


# ─────────────────────────────────────────────────────────────────────────────
# Edge parameter sampling
# ─────────────────────────────────────────────────────────────────────────────

def sample_rigidities(n_tri, mode='lognormal', sigma=1.0):
    """Sample per-triangle rigidities.

    Args:
        n_tri: number of triangles.
        mode: 'lognormal' or 'uniform'.
        sigma: std for lognormal, or range for uniform.

    Returns:
        (n_tri, 3) rigidities.
    """
    if mode == 'lognormal':
        return np.exp(np.random.randn(n_tri, 3) * sigma)
    elif mode == 'uniform':
        return np.random.uniform(0.01, 10.0, (n_tri, 3))
    else:
        return np.ones((n_tri, 3))


def sample_rest_lengths(actual_lengths, sigma=0.1):
    """Sample per-triangle rest lengths as perturbations of actual lengths.

    Args:
        actual_lengths: (n_tri, 3) actual edge lengths from geometry.
        sigma: std of log-perturbation.

    Returns:
        (n_tri, 3) rest lengths.
    """
    return actual_lengths * np.exp(np.random.randn(*actual_lengths.shape) * sigma)


# ─────────────────────────────────────────────────────────────────────────────
# Single sample generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_single_sample(sample_idx, topo_name, size, seed,
                           design_variable='rigidities',
                           rig_mode='lognormal', rig_sigma=1.0,
                           rl_sigma=0.1):
    """Generate one (graph, edge_params, labels) sample.

    Args:
        sample_idx: unique index for this sample.
        topo_name: topology name from catalog.
        size: (sx, sy) mesh size.
        seed: random seed for topology generation.
        design_variable: 'rigidities', 'rest_lengths', or 'both'.
        rig_mode: rigidity sampling mode.
        rig_sigma: rigidity sigma.
        rl_sigma: rest length sigma.

    Returns:
        PyG Data object with labels, or None if solver fails.
    """
    # Set seed for reproducibility of edge params (not topology)
    np.random.seed(sample_idx * 31337 + seed)

    try:
        tri_result = generate_topology(topo_name, size, seed=seed)
        compat = tri_result.to_delaunay_compat()
        solver, default_rigs, default_rl = from_triangulation(compat)

        n_tri = tri_result.n_tri
        default_rigs_np = default_rigs.numpy()
        actual_rl_np = default_rl.numpy()

        # Get hard/soft edge mask
        hard_mask = tri_result.get_hard_edge_mask_per_triangle()

        # Sample edge parameters
        if design_variable == 'rigidities':
            rigs = sample_rigidities(n_tri, mode=rig_mode, sigma=rig_sigma)
            rest_lengths = actual_rl_np.copy()
        elif design_variable == 'rest_lengths':
            rigs = np.ones((n_tri, 3))
            rest_lengths = sample_rest_lengths(actual_rl_np, sigma=rl_sigma)
        elif design_variable == 'both':
            rigs = sample_rigidities(n_tri, mode=rig_mode, sigma=rig_sigma)
            rest_lengths = sample_rest_lengths(actual_rl_np, sigma=rl_sigma)
        else:
            raise ValueError(f"Unknown design_variable: {design_variable}")

        # For soft (regularization) edges, force k = k_soft, l0 = l_actual
        if tri_result.hard_edge_set is not None:
            rigs[~hard_mask] = tri_result.k_soft_ratio
            rest_lengths[~hard_mask] = actual_rl_np[~hard_mask]

        # Run forward solver
        rigs_t = torch.tensor(rigs, dtype=torch.float64)
        rl_t = torch.tensor(rest_lengths, dtype=torch.float64)

        with torch.no_grad():
            result = solver(rigs_t, rl_t)

        nu = result['poisson'].item()
        young = result['young'].item()
        tensor = result['elastic_tensor'].numpy()

        # Skip invalid results (can happen with extreme parameters)
        if not np.isfinite(nu) or abs(nu) > 10:
            return None

        labels = {
            'poisson': nu,
            'young': young,
            'elastic_tensor': tensor,
        }

        data = triangulation_to_pyg_data(tri_result, rigs, rest_lengths, labels)
        data.sample_idx = sample_idx
        data.design_variable = ['rigidities', 'rest_lengths', 'both'].index(design_variable)

        return data

    except Exception as e:
        print(f"Sample {sample_idx} failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Batch generation
# ─────────────────────────────────────────────────────────────────────────────

def _generate_sample_config_list(n_samples, topologies=None, sizes=None,
                                  design_variables=None, rig_sigmas=None):
    """Build a list of sample configurations for parallel generation.

    Cycles through all combinations of topology, size, design_variable, sigma.
    """
    if topologies is None:
        topologies = list(TOPOLOGY_GENERATORS.keys())
    if sizes is None:
        sizes = [(6, 6), (8, 8), (10, 10)]
    if design_variables is None:
        design_variables = ['rigidities', 'rest_lengths', 'both']
    if rig_sigmas is None:
        rig_sigmas = [0.3, 0.5, 1.0, 2.0]

    configs = []
    random_topos = ['foam_eta02', 'foam_eta045', 'foam_retri_eta02',
                    'foam_retri_eta045', 'poisson_delaunay', 'blue_noise',
                    'clustered', 'gradient_density']
    deterministic_topos = [t for t in topologies if t not in random_topos]

    idx = 0
    while idx < n_samples:
        for topo in topologies:
            for size in sizes:
                for dv in design_variables:
                    for sigma in rig_sigmas:
                        if idx >= n_samples:
                            break
                        seed = 42 + (idx % 20) * 137 if topo in random_topos else 0
                        configs.append({
                            'sample_idx': idx,
                            'topo_name': topo,
                            'size': size,
                            'seed': seed,
                            'design_variable': dv,
                            'rig_mode': 'lognormal',
                            'rig_sigma': sigma,
                            'rl_sigma': 0.1,
                        })
                        idx += 1

    return configs[:n_samples]


def _worker(config):
    """Worker function for multiprocessing."""
    return generate_single_sample(**config)


def generate_dataset(n_samples, n_workers=1, **kwargs):
    """Generate a dataset of (graph, edge_params, labels) samples.

    Args:
        n_samples: total number of samples.
        n_workers: number of parallel workers (1 = serial).
        **kwargs: passed to _generate_sample_config_list.

    Returns:
        list of PyG Data objects (None entries filtered out).
    """
    configs = _generate_sample_config_list(n_samples, **kwargs)

    if n_workers <= 1:
        data_list = []
        for i, config in enumerate(configs):
            data = _worker(config)
            if data is not None:
                data_list.append(data)
            if (i + 1) % 100 == 0:
                print(f"  Generated {i+1}/{n_samples} samples "
                      f"({len(data_list)} valid)")
    else:
        with Pool(n_workers) as pool:
            results = pool.map(_worker, configs)
        data_list = [d for d in results if d is not None]

    print(f"Generated {len(data_list)}/{n_samples} valid samples")
    return data_list


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Phase 4 training data')
    parser.add_argument('--output_dir', type=str, default='./processed',
                        help='Directory to save processed data')
    parser.add_argument('--n_train', type=int, default=1000,
                        help='Number of training samples')
    parser.add_argument('--n_val', type=int, default=200,
                        help='Number of validation samples')
    parser.add_argument('--n_test', type=int, default=200,
                        help='Number of test samples')
    parser.add_argument('--n_workers', type=int, default=1,
                        help='Number of parallel workers')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split, n in [('train', args.n_train), ('val', args.n_val), ('test', args.n_test)]:
        print(f"\n=== Generating {split} split ({n} samples) ===")
        data_list = generate_dataset(n, n_workers=args.n_workers)
        save_path = output_dir / f'{split}.pt'
        torch.save(data_list, save_path)
        print(f"Saved {len(data_list)} samples to {save_path}")
