"""
Phase 3 — Direct Inverse Optimization

Given a target elastic tensor (or Poisson / Young targets), find spring
rigidities that reproduce it by gradient descent through the differentiable
forward solver from Phase 2.

===========================================================================
APPROACH
===========================================================================

1.  Generate a foam network (fixed geometry).
2.  Run the forward solver with unit rigidities → ground-truth elastic tensor T*.
3.  Starting from random rigidities, minimize:

        L = ||forward(k) - T*||²      (MSE on the 6-component tensor)

    Rigidities are parameterized as  k = softplus(raw)  to enforce positivity.

4.  Repeat from many random initializations to explore the solution landscape.
5.  Cluster converged solutions and report:
        - number of distinct solutions
        - per-edge variance across solutions
        - round-trip reconstruction error

===========================================================================
"""

import sys
import os
import time
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Resolve imports from project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "Phase 2"))

import Disc_2_Cont_optimized as D2C
from forward_solver_torch import ElasticSolver, from_triangulation


# ---------------------------------------------------------------------------
# Parameterization helpers
# ---------------------------------------------------------------------------

def raw_to_rigidities(raw):
    """Map unconstrained parameters → positive rigidities via softplus."""
    return F.softplus(raw, beta=5.0)


def rigidities_to_raw(rigidities):
    """Inverse of softplus: raw = log(exp(beta * k) - 1) / beta."""
    beta = 5.0
    return torch.log(torch.expm1(beta * rigidities)) / beta


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def tensor_mse_loss(pred_tensor, target_tensor):
    """MSE loss on the 6-component elastic tensor."""
    return ((pred_tensor - target_tensor) ** 2).mean()


def property_loss(pred_result, target_poisson, target_young,
                  weight_poisson=1.0, weight_young=1.0):
    """Loss on Poisson's ratio and Young's modulus directly."""
    loss = weight_poisson * (pred_result['poisson'] - target_poisson) ** 2
    loss = loss + weight_young * (pred_result['young'] - target_young) ** 2
    return loss


# ---------------------------------------------------------------------------
# Single optimization run
# ---------------------------------------------------------------------------

def run_single_optimization(
    solver,
    target_tensor,
    n_triangles,
    n_edges_per_tri=3,
    max_iter=500,
    lr=0.05,
    tol=1e-10,
    optimizer_type='lbfgs',
    seed=None,
    verbose=False,
):
    """Run one inverse optimization from a random initialization.

    Args:
        solver:         ElasticSolver instance (fixed geometry).
        target_tensor:  (6,) tensor — the target elastic tensor.
        n_triangles:    Number of triangles in the mesh.
        n_edges_per_tri: Edges per triangle (always 3).
        max_iter:       Maximum optimizer iterations.
        lr:             Learning rate.
        tol:            Convergence tolerance on loss.
        optimizer_type: 'lbfgs' or 'adam'.
        seed:           Random seed (None = random).
        verbose:        Print progress.

    Returns:
        dict with keys:
            'converged'      — bool
            'final_loss'     — float
            'iterations'     — int
            'rigidities'     — (N, 3) numpy array of converged rigidities
            'pred_tensor'    — (6,) numpy array
            'target_tensor'  — (6,) numpy array
            'rel_error'      — float, relative error ||pred - target|| / ||target||
            'poisson'        — float
            'young'          — float
            'time_seconds'   — float
    """
    if seed is not None:
        torch.manual_seed(seed)

    # Random initialization: log-uniform around 1.0
    init_rigs = torch.exp(0.5 * torch.randn(n_triangles, n_edges_per_tri,
                                              dtype=torch.float64))
    raw = rigidities_to_raw(init_rigs)
    raw = raw.clone().detach().requires_grad_(True)

    target = target_tensor.clone().detach()

    best_loss = float('inf')
    best_raw = raw.clone().detach()
    iterations = 0
    t0 = time.time()

    if optimizer_type == 'lbfgs':
        optimizer = torch.optim.LBFGS(
            [raw], lr=lr, max_iter=20, line_search_fn='strong_wolfe',
            tolerance_grad=1e-12, tolerance_change=1e-14,
        )

        for outer in range(max_iter // 20 + 1):
            def closure():
                optimizer.zero_grad()
                k = raw_to_rigidities(raw)
                result = solver(k, rest_lengths=None)
                loss = tensor_mse_loss(result['elastic_tensor'], target)
                loss.backward()
                return loss

            loss_val = optimizer.step(closure)
            iterations += 20

            current_loss = loss_val.item()
            if current_loss < best_loss:
                best_loss = current_loss
                best_raw = raw.clone().detach()

            if verbose and outer % 5 == 0:
                print(f"  iter {iterations:4d}  loss = {current_loss:.3e}")

            if current_loss < tol:
                break

    elif optimizer_type == 'adam':
        optimizer = torch.optim.Adam([raw], lr=lr)

        for it in range(max_iter):
            optimizer.zero_grad()
            k = raw_to_rigidities(raw)
            result = solver(k, rest_lengths=None)
            loss = tensor_mse_loss(result['elastic_tensor'], target)
            loss.backward()
            optimizer.step()
            iterations = it + 1

            current_loss = loss.item()
            if current_loss < best_loss:
                best_loss = current_loss
                best_raw = raw.clone().detach()

            if verbose and it % 50 == 0:
                print(f"  iter {it:4d}  loss = {current_loss:.3e}")

            if current_loss < tol:
                break
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")

    elapsed = time.time() - t0

    # Evaluate at best parameters
    with torch.no_grad():
        k_best = raw_to_rigidities(best_raw)
        result = solver(k_best, rest_lengths=None)
        pred = result['elastic_tensor']
        rel_error = torch.norm(pred - target) / (torch.norm(target) + 1e-30)

    return {
        'converged': best_loss < tol * 100,  # generous threshold
        'final_loss': best_loss,
        'iterations': iterations,
        'rigidities': k_best.numpy(),
        'pred_tensor': pred.numpy(),
        'target_tensor': target.numpy(),
        'rel_error': rel_error.item(),
        'poisson': result['poisson'].item(),
        'young': result['young'].item(),
        'time_seconds': elapsed,
    }


# ---------------------------------------------------------------------------
# Multi-start campaign
# ---------------------------------------------------------------------------

def run_campaign(
    solver,
    target_tensor,
    n_triangles,
    n_starts=50,
    max_iter=500,
    lr=0.05,
    tol=1e-10,
    optimizer_type='lbfgs',
    verbose=True,
):
    """Run multiple random-start optimizations and analyze the results.

    Returns:
        dict with keys:
            'runs'             — list of per-run result dicts
            'converged_runs'   — list of converged result dicts
            'n_converged'      — int
            'n_total'          — int
            'mean_rel_error'   — float (over converged runs)
            'clusters'         — list of cluster centroids (numpy arrays)
            'n_clusters'       — int
            'per_edge_std'     — (N, 3) numpy array, std of rigidities across converged runs
            'total_time'       — float
    """
    runs = []
    t0 = time.time()

    for i in range(n_starts):
        if verbose:
            print(f"\n--- Run {i+1}/{n_starts} ---")
        result = run_single_optimization(
            solver=solver,
            target_tensor=target_tensor,
            n_triangles=n_triangles,
            max_iter=max_iter,
            lr=lr,
            tol=tol,
            optimizer_type=optimizer_type,
            seed=i * 137 + 42,
            verbose=verbose,
        )
        if verbose:
            print(f"  loss = {result['final_loss']:.3e}  "
                  f"rel_err = {result['rel_error']:.3e}  "
                  f"converged = {result['converged']}  "
                  f"time = {result['time_seconds']:.2f}s")
        runs.append(result)

    total_time = time.time() - t0

    # Filter converged runs
    converged = [r for r in runs if r['converged']]
    n_converged = len(converged)

    if n_converged == 0:
        return {
            'runs': runs,
            'converged_runs': [],
            'n_converged': 0,
            'n_total': n_starts,
            'mean_rel_error': float('inf'),
            'clusters': [],
            'n_clusters': 0,
            'per_edge_std': None,
            'total_time': total_time,
        }

    # Stack converged rigidities: (n_converged, N, 3)
    all_rigs = np.stack([r['rigidities'] for r in converged])

    # Per-edge standard deviation across converged solutions
    per_edge_std = np.std(all_rigs, axis=0)  # (N, 3)

    # Cluster converged solutions by rigidity vector distance
    flat_rigs = all_rigs.reshape(n_converged, -1)  # (n_converged, N*3)
    clusters = _cluster_solutions(flat_rigs, threshold=0.05)

    mean_rel = np.mean([r['rel_error'] for r in converged])

    return {
        'runs': runs,
        'converged_runs': converged,
        'n_converged': n_converged,
        'n_total': n_starts,
        'mean_rel_error': mean_rel,
        'clusters': clusters,
        'n_clusters': len(clusters),
        'per_edge_std': per_edge_std,
        'total_time': total_time,
    }


def _cluster_solutions(flat_rigs, threshold=0.05):
    """Simple greedy clustering of rigidity vectors.

    Two solutions are in the same cluster if their normalized distance
    is below `threshold`.

    Args:
        flat_rigs: (K, D) array — K converged solutions, D = N*3.
        threshold: relative distance threshold for same-cluster.

    Returns:
        List of cluster dicts: {'centroid': array, 'size': int, 'indices': list}
    """
    K = flat_rigs.shape[0]
    if K == 0:
        return []

    norms = np.linalg.norm(flat_rigs, axis=1, keepdims=True) + 1e-30
    normed = flat_rigs / norms

    assigned = [False] * K
    clusters = []

    for i in range(K):
        if assigned[i]:
            continue
        # Start a new cluster with solution i
        members = [i]
        assigned[i] = True

        for j in range(i + 1, K):
            if assigned[j]:
                continue
            dist = np.linalg.norm(normed[i] - normed[j])
            if dist < threshold:
                members.append(j)
                assigned[j] = True

        centroid = flat_rigs[members].mean(axis=0)
        clusters.append({
            'centroid': centroid,
            'size': len(members),
            'indices': members,
        })

    return clusters


# ---------------------------------------------------------------------------
# Round-trip validation
# ---------------------------------------------------------------------------

def validate_round_trip(solver, rigidities, target_tensor):
    """Feed rigidities back through the forward solver and check agreement.

    Args:
        solver:        ElasticSolver
        rigidities:    (N, 3) numpy array
        target_tensor: (6,) numpy array

    Returns:
        dict with 'pred_tensor', 'rel_error', 'poisson', 'young'
    """
    with torch.no_grad():
        k = torch.as_tensor(rigidities, dtype=torch.float64)
        result = solver(k, rest_lengths=None)
        pred = result['elastic_tensor'].numpy()
        rel_error = np.linalg.norm(pred - target_tensor) / (np.linalg.norm(target_tensor) + 1e-30)

    return {
        'pred_tensor': pred,
        'rel_error': rel_error,
        'poisson': result['poisson'].item(),
        'young': result['young'].item(),
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Phase 3: Inverse optimization — find rigidities for a target elastic tensor"
    )
    parser.add_argument('--size', type=int, nargs=2, default=[4, 4],
                        help='Mesh size (default: 4 4)')
    parser.add_argument('--eta', type=float, default=0.2,
                        help='Foam disorder parameter (default: 0.2)')
    parser.add_argument('--n-starts', type=int, default=20,
                        help='Number of random initializations (default: 20)')
    parser.add_argument('--max-iter', type=int, default=500,
                        help='Max iterations per run (default: 500)')
    parser.add_argument('--lr', type=float, default=0.05,
                        help='Learning rate (default: 0.05)')
    parser.add_argument('--optimizer', choices=['lbfgs', 'adam'], default='lbfgs',
                        help='Optimizer type (default: lbfgs)')
    parser.add_argument('--tol', type=float, default=1e-10,
                        help='Convergence tolerance (default: 1e-10)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for mesh generation (default: 42)')
    parser.add_argument('--save', type=str, default=None,
                        help='Save results to JSON file')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='Suppress per-run output')
    args = parser.parse_args()

    np.random.seed(args.seed)

    # --- 1. Generate mesh ---
    print(f"Generating foam mesh: size={args.size}, eta={args.eta}")
    tri = D2C.generate_foam_points(size=tuple(args.size), eta=args.eta)
    solver, default_rigs, default_rl = from_triangulation(tri)
    n_tri = len(tri.simplices)
    print(f"  {n_tri} triangles, {n_tri * 3} edge-springs")

    # --- 2. Compute ground-truth target ---
    with torch.no_grad():
        gt = solver(default_rigs, rest_lengths=None)
    target = gt['elastic_tensor']
    print(f"\nGround-truth (unit rigidities):")
    print(f"  Poisson = {gt['poisson'].item():.6f}")
    print(f"  Young   = {gt['young'].item():.6f}")
    print(f"  Tensor  = {target.numpy()}")

    # --- 3. Run optimization campaign ---
    print(f"\nRunning {args.n_starts} random-start optimizations "
          f"(optimizer={args.optimizer}, max_iter={args.max_iter}, lr={args.lr})")
    campaign = run_campaign(
        solver=solver,
        target_tensor=target,
        n_triangles=n_tri,
        n_starts=args.n_starts,
        max_iter=args.max_iter,
        lr=args.lr,
        tol=args.tol,
        optimizer_type=args.optimizer,
        verbose=not args.quiet,
    )

    # --- 4. Report ---
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Converged: {campaign['n_converged']} / {campaign['n_total']}")
    print(f"Distinct clusters: {campaign['n_clusters']}")
    print(f"Mean relative error (converged): {campaign['mean_rel_error']:.3e}")
    print(f"Total time: {campaign['total_time']:.1f}s")

    if campaign['n_converged'] > 0:
        # Per-edge analysis
        std = campaign['per_edge_std']
        mean_rigs = np.mean(
            np.stack([r['rigidities'] for r in campaign['converged_runs']]),
            axis=0,
        )
        cv = std / (mean_rigs + 1e-30)  # coefficient of variation

        print(f"\nPer-edge rigidity variation (across converged solutions):")
        print(f"  Mean CV (coefficient of variation): {cv.mean():.4f}")
        print(f"  Max CV:  {cv.max():.4f}")
        print(f"  Edges with CV < 0.01 (locked):  {(cv < 0.01).sum()} / {cv.size}")
        print(f"  Edges with CV > 0.10 (free):    {(cv > 0.10).sum()} / {cv.size}")

        # Cluster details
        print(f"\nCluster details:")
        for ci, cl in enumerate(campaign['clusters']):
            print(f"  Cluster {ci+1}: {cl['size']} members")

        # Round-trip validation of best solution
        best_run = min(campaign['converged_runs'], key=lambda r: r['final_loss'])
        rt = validate_round_trip(solver, best_run['rigidities'], target.numpy())
        print(f"\nRound-trip validation (best solution):")
        print(f"  Relative error: {rt['rel_error']:.3e}")
        print(f"  Poisson: {rt['poisson']:.6f}  (target: {gt['poisson'].item():.6f})")
        print(f"  Young:   {rt['young']:.6f}  (target: {gt['young'].item():.6f})")

    # --- 5. Save ---
    if args.save:
        save_data = {
            'config': vars(args),
            'n_converged': campaign['n_converged'],
            'n_total': campaign['n_total'],
            'n_clusters': campaign['n_clusters'],
            'mean_rel_error': campaign['mean_rel_error'],
            'total_time': campaign['total_time'],
            'runs': [
                {
                    'converged': r['converged'],
                    'final_loss': r['final_loss'],
                    'iterations': r['iterations'],
                    'rel_error': r['rel_error'],
                    'poisson': r['poisson'],
                    'young': r['young'],
                    'time_seconds': r['time_seconds'],
                }
                for r in campaign['runs']
            ],
        }
        with open(args.save, 'w') as f:
            json.dump(save_data, f, indent=2)
        print(f"\nResults saved to {args.save}")


if __name__ == '__main__':
    main()
