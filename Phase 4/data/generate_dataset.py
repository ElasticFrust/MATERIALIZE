"""Data generation pipeline for Phase 4 GNN training.

This script creates the training/validation/test datasets by:
  1. Picking a topology from the catalog (18 topologies across 6 categories)
  2. Sampling spatially structured edge parameters via rigidity patterns
  3. Running the Phase 2 forward solver to compute ground-truth labels
     (Poisson's ratio, Young's modulus, full elastic tensor)
  4. Converting (mesh + params + labels) to PyG Data objects via graph_utils

The key design choice is the **variation axes**: each sample varies along
multiple dimensions simultaneously:
  - Topology type (crystal, foam, random, honeycomb, kagome, etc.)
  - Mesh size (6x6, 8x8, 10x10) — for scale generalization
  - Edge parameter type (rigidities only, rest lengths only, or both)
  - Rigidity pattern (8 spatial patterns with varying correlation structures)
  - Parameter magnitude (sigma in {0.3, 0.5, 1.0, 2.0}) — for range coverage

The 8 rigidity patterns provide diverse spatial structure:
  - iid: independent per-edge (baseline)
  - grf: Gaussian random field (spatially correlated)
  - gradient: linear gradient across the mesh
  - radial: stiff core / soft boundary or vice versa
  - percolation: binary stiff/soft assignment at random probability
  - stripes: alternating bands of different stiffness
  - virtual_distortion: stiffness from virtual mesh deformation (Phase 3 method)
  - voronoi_clusters: piecewise-constant Voronoi patches

This creates a rich, high-dimensional training distribution that forces the
GNN to learn general structure-property relationships rather than memorizing
specific topology-parameter combinations.

For non-triangulated meshes (honeycomb, kagome, square), soft regularization
edges are frozen at k_soft=1e-3 regardless of the sampled parameters. Only
hard (structural) edges have their parameters varied.

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

# Add project roots to path so we can import the Phase 2 forward solver
# and the Phase 4 data modules (topology_generators, graph_utils).
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "Phase 2"))
sys.path.insert(0, str(PROJECT_ROOT / "Phase 3"))

from data.topology_generators import generate_topology, TOPOLOGY_GENERATORS
from data.graph_utils import triangulation_to_pyg_data
from data.rigidity_patterns import (
    sample_rigidity_pattern, random_pattern_name, random_pattern_params,
    RIGIDITY_PATTERNS,
)
# from_triangulation: given a Delaunay-like object, returns (solver_fn, default_rigs, default_rl)
from forward_solver_torch import from_triangulation


# ─────────────────────────────────────────────────────────────────────────────
# Edge parameter sampling
# ─────────────────────────────────────────────────────────────────────────────
#
# Rigidity sampling is handled by the rigidity_patterns module, which provides
# 8 spatially structured patterns (IID, GRF, gradient, radial, percolation,
# stripes, virtual distortion, Voronoi clusters). See rigidity_patterns.py.
#
# Rest length sampling remains here: multiplicative perturbation of actual
# edge lengths. l0 = l_actual * exp(N(0, sigma^2)) gives relative
# perturbations: at sigma=0.1, l0 varies by ~10%.

def sample_rest_lengths(actual_lengths, sigma=0.1):
    """Sample per-triangle rest lengths as multiplicative perturbations of actual lengths.

    l0 = l_actual * exp(N(0, sigma^2)). When l0 > l_actual, the spring is
    compressed (wants to be longer). When l0 < l_actual, it's stretched.
    This creates internal pre-stress that affects the elastic response.

    Args:
        actual_lengths: (n_tri, 3) actual edge lengths from mesh geometry.
        sigma: std of the log-perturbation. At sigma=0.1, ~68% of rest lengths
               fall within [0.9 * l_actual, 1.1 * l_actual].

    Returns:
        (n_tri, 3) rest lengths.
    """
    return actual_lengths * np.exp(np.random.randn(*actual_lengths.shape) * sigma)


# ─────────────────────────────────────────────────────────────────────────────
# Single sample generation
# ─────────────────────────────────────────────────────────────────────────────
#
# Each sample follows this pipeline:
#   1. Generate mesh topology (from the catalog)
#   2. Initialize the forward solver from the triangulation
#   3. Sample random edge parameters (k and/or l0)
#   4. Freeze soft edges (for non-triangulated meshes)
#   5. Run the forward solver to get Poisson's ratio, Young's modulus, tensor
#   6. Convert everything to a PyG Data object
#
# If the solver fails (e.g., singular matrix from extreme parameters), the
# sample is silently discarded (returns None). The outer loop generates
# extra samples to compensate for the ~5-10% failure rate.

def generate_single_sample(sample_idx, topo_name, size, seed,
                           design_variable='rigidities',
                           rig_pattern='iid', rig_pattern_params=None,
                           rig_sigma=1.0, rl_sigma=0.1):
    """Generate one complete (graph, edge_params, labels) training sample.

    This is the core function called once per sample. It:
      1. Generates the mesh topology
      2. Samples spatially structured edge parameters via rigidity patterns
      3. Runs the forward solver for ground-truth labels
      4. Packages everything as a PyG Data object

    The random seed for edge parameters is derived deterministically from
    sample_idx and seed, ensuring reproducibility even under parallel generation.
    The topology seed is set separately to allow the same mesh with different
    edge parameters.

    Args:
        sample_idx: unique index for this sample (determines edge param seed).
        topo_name: topology name from catalog (e.g., 'honeycomb', 'kagome').
        size: (sx, sy) mesh half-extents.
        seed: random seed for topology generation.
        design_variable: which edge parameters to vary:
            'rigidities': vary k, keep l0 = l_actual
            'rest_lengths': keep k = 1, vary l0
            'both': vary both k and l0
        rig_pattern: rigidity pattern name from RIGIDITY_PATTERNS.
        rig_pattern_params: dict of pattern-specific kwargs (correlation_length, etc.)
        rig_sigma: sigma for rigidity sampling (controls heterogeneity).
        rl_sigma: sigma for rest length perturbation.

    Returns:
        PyG Data object with labels, or None if the solver fails.
    """
    # Seed for edge parameter sampling — deterministic per (sample_idx, seed) pair.
    # Uses a large prime multiplier (31337) to decorrelate adjacent samples.
    np.random.seed(sample_idx * 31337 + seed)

    if rig_pattern_params is None:
        rig_pattern_params = {}

    try:
        # Step 1: Generate the mesh topology
        tri_result = generate_topology(topo_name, size, seed=seed)

        # Step 2: Initialize the forward solver from the triangulation.
        # from_triangulation returns:
        #   solver: callable(rigidities, rest_lengths) -> {poisson, young, elastic_tensor}
        #   default_rigs: (N_tri, 3) default rigidities (all 1.0)
        #   default_rl: (N_tri, 3) actual edge lengths from geometry
        compat = tri_result.to_delaunay_compat()
        solver, default_rigs, default_rl = from_triangulation(compat)

        n_tri = tri_result.n_tri
        actual_rl_np = default_rl.numpy()

        # Get hard/soft edge mask — True where edges are structural (designable)
        hard_mask = tri_result.get_hard_edge_mask_per_triangle()

        # Step 3: Sample spatially structured edge parameters
        if design_variable in ('rigidities', 'both'):
            rigs = sample_rigidity_pattern(
                tri_result.points, tri_result.simplices,
                pattern=rig_pattern, sigma=rig_sigma,
                **rig_pattern_params,
            )
        else:
            rigs = np.ones((n_tri, 3))

        if design_variable in ('rest_lengths', 'both'):
            rest_lengths = sample_rest_lengths(actual_rl_np, sigma=rl_sigma)
        else:
            rest_lengths = actual_rl_np.copy()

        # Step 4: Freeze soft (regularization) edges — they are NOT design variables.
        # Their k stays at k_soft_ratio (1e-3) and l0 stays at l_actual regardless
        # of what was sampled above.
        if tri_result.hard_edge_set is not None:
            rigs[~hard_mask] = tri_result.k_soft_ratio
            rest_lengths[~hard_mask] = actual_rl_np[~hard_mask]

        # Step 5: Run the Phase 2 forward solver to compute elastic properties.
        # The solver uses float64 for numerical precision (elastic tensor inversion).
        rigs_t = torch.tensor(rigs, dtype=torch.float64)
        rl_t = torch.tensor(rest_lengths, dtype=torch.float64)

        with torch.no_grad():
            result = solver(rigs_t, rl_t)

        nu = result['poisson'].item()       # Poisson's ratio (main prediction target)
        young = result['young'].item()       # Young's modulus
        tensor = result['elastic_tensor'].numpy()  # full 2D elastic tensor (6 components)

        # Skip invalid results — extreme parameters can cause singular matrices
        # or produce unphysical Poisson ratios
        if not np.isfinite(nu) or abs(nu) > 10:
            return None

        labels = {
            'poisson': nu,
            'young': young,
            'elastic_tensor': tensor,
        }

        # Step 6: Convert to PyG Data object (the full pipeline in graph_utils.py)
        data = triangulation_to_pyg_data(tri_result, rigs, rest_lengths, labels)
        data.sample_idx = sample_idx
        # Encode design_variable mode as an integer for potential downstream use
        data.design_variable = ['rigidities', 'rest_lengths', 'both'].index(design_variable)
        # Store pattern name for analysis/debugging
        data.rig_pattern = rig_pattern

        return data

    except Exception as e:
        print(f"Sample {sample_idx} failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Batch generation — configuration and orchestration
# ─────────────────────────────────────────────────────────────────────────────
#
# The batch generation system works in two stages:
#   1. _generate_sample_config_list(): creates a flat list of configurations,
#      each specifying (topology, size, design_variable, sigma, seed).
#      The configurations cycle through all combinations to ensure balanced
#      representation of each variation axis.
#   2. generate_dataset(): dispatches the configs to workers (serial or parallel)
#      and collects the results.
#
# This architecture allows easy parallelization: each config is independent
# and can be processed by a separate worker process.

def _generate_sample_config_list(n_samples, topologies=None, sizes=None,
                                  design_variables=None, rig_sigmas=None):
    """Build a flat list of sample configurations for generation.

    Cycles through all combinations of (topology, size, design_variable, sigma)
    in a round-robin fashion to ensure balanced coverage. Each sample also gets
    a randomly selected rigidity pattern (from PATTERN_WEIGHTS) with random
    pattern-specific hyperparameters.

    Stochastic topologies get different random seeds based on sample index;
    deterministic topologies always use seed=0.
    """
    # Default variation axes — these create the Cartesian product space that
    # each sample is drawn from. Total combinations per epoch:
    #   18 topos × 3 sizes × 3 design_vars × 4 sigmas = 648 unique configs
    #   Each config also gets a random rigidity pattern + params.
    if topologies is None:
        topologies = list(TOPOLOGY_GENERATORS.keys())
    if sizes is None:
        sizes = [(6, 6), (8, 8), (10, 10)]  # 3 sizes for scale generalization
    if design_variables is None:
        design_variables = ['rigidities', 'rest_lengths', 'both']
    if rig_sigmas is None:
        rig_sigmas = [0.3, 0.5, 1.0, 2.0]  # 4 levels of heterogeneity

    configs = []
    # Stochastic topologies need different seeds for structural variety.
    # Deterministic topologies (crystal, honeycomb, kagome) always produce
    # the same mesh regardless of seed.
    random_topos = ['foam_eta02', 'foam_eta045', 'foam_retri_eta02',
                    'foam_retri_eta045', 'poisson_delaunay', 'blue_noise',
                    'clustered', 'gradient_density']

    # Use a deterministic RNG for config generation so dataset is reproducible.
    config_rng = np.random.RandomState(12345)

    # Round-robin through all combinations until we have n_samples configs.
    # The modular seed (idx % 20) * 137 gives 20 distinct mesh realizations
    # per stochastic topology, cycling as needed.
    idx = 0
    while idx < n_samples:
        for topo in topologies:
            for size in sizes:
                for dv in design_variables:
                    for sigma in rig_sigmas:
                        if idx >= n_samples:
                            break
                        seed = 42 + (idx % 20) * 137 if topo in random_topos else 0

                        # Randomly select a rigidity pattern + its hyperparams.
                        # Use config_rng for reproducibility.
                        np.random.seed(config_rng.randint(0, 2**31))
                        pattern = random_pattern_name()
                        pattern_params = random_pattern_params(pattern)

                        configs.append({
                            'sample_idx': idx,
                            'topo_name': topo,
                            'size': size,
                            'seed': seed,
                            'design_variable': dv,
                            'rig_pattern': pattern,
                            'rig_pattern_params': pattern_params,
                            'rig_sigma': sigma,
                            'rl_sigma': 0.1,
                        })
                        idx += 1

    return configs[:n_samples]


def _worker(config):
    """Worker function for multiprocessing."""
    return generate_single_sample(**config)


def generate_chunk_live(n_samples, n_workers=4):
    """Generate a fresh chunk of training data on-the-fly (no disk I/O).

    This is the core of the *renewable* training mode: instead of loading
    the same pre-saved chunk files on every epoch, the training loop calls
    this function each chunk to obtain brand-new random samples. Because
    topology, rigidity pattern, sigma, and parameter type are all re-sampled,
    the model is exposed to an effectively infinite stream of training examples
    — preventing memorization and improving generalization.

    Uses torch.multiprocessing with the 'file_system' sharing strategy so
    that PyG Data objects can be passed back from worker processes safely.

    Args:
        n_samples: number of samples to attempt (actual count will be
                   ~90-95% of this after filtering failed simulations).
        n_workers: parallel worker processes. Set to 1 for debugging.

    Returns:
        list of valid PyG Data objects.
    """
    import torch.multiprocessing as tmp
    tmp.set_sharing_strategy('file_system')

    configs = _generate_sample_config_list(n_samples)
    if n_workers <= 1:
        results = [_worker(c) for c in configs]
    else:
        with tmp.Pool(n_workers) as pool:
            results = pool.map(_worker, configs)

    chunk = [d for d in results if d is not None]
    del results
    return chunk


def generate_dataset(n_samples, n_workers=1, **kwargs):
    """Generate a complete dataset of (graph, edge_params, labels) samples.

    This is the main entry point for dataset generation. It builds a config
    list, dispatches each config to generate_single_sample (either serially
    or via multiprocessing), and filters out failed samples (None returns).

    Typical failure rate is ~5-10% due to extreme parameter combinations
    causing singular solver matrices. The actual number of valid samples
    will be somewhat less than n_samples.

    Args:
        n_samples: total number of sample configs to attempt.
        n_workers: number of parallel workers. Use 1 for debugging (serial),
                   higher values for production data generation.
        **kwargs: passed to _generate_sample_config_list (topologies, sizes, etc.).

    Returns:
        list of PyG Data objects (failed samples filtered out).
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
# CLI — command-line entry point for dataset generation
# ─────────────────────────────────────────────────────────────────────────────
#
# Usage examples:
#   # Small test run (serial, fast):
#   python generate_dataset.py --n_train 100 --n_val 20 --n_test 20
#
#   # Production run (parallel):
#   python generate_dataset.py --n_train 80000 --n_val 10000 --n_test 10000 --n_workers 8
#
# Output: {output_dir}/train.pt, val.pt, test.pt
#   Each .pt file is a list of PyG Data objects, loadable with torch.load().

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Phase 4 training data')
    parser.add_argument('--output_dir', type=str, default='./processed',
                        help='Directory to save processed .pt files')
    parser.add_argument('--n_train', type=int, default=1000,
                        help='Number of training samples to generate')
    parser.add_argument('--n_val', type=int, default=200,
                        help='Number of validation samples')
    parser.add_argument('--n_test', type=int, default=200,
                        help='Number of test samples')
    parser.add_argument('--n_workers', type=int, default=1,
                        help='Number of parallel workers (1 = serial)')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate each split (train/val/test) independently.
    # Each split gets its own round-robin cycle through the variation axes.
    for split, n in [('train', args.n_train), ('val', args.n_val), ('test', args.n_test)]:
        print(f"\n=== Generating {split} split ({n} samples) ===")
        data_list = generate_dataset(n, n_workers=args.n_workers)
        save_path = output_dir / f'{split}.pt'
        torch.save(data_list, save_path)
        print(f"Saved {len(data_list)} samples to {save_path}")
