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
# Property-targeting optimization (e.g. target Poisson = 0.1)
# ---------------------------------------------------------------------------

def run_property_optimization(
    solver,
    n_triangles,
    target_poisson=None,
    target_young=None,
    weight_poisson=1.0,
    weight_young=1.0,
    design_variable='rigidities',
    isotropic=False,
    weight_isotropy=10.0,
    n_edges_per_tri=3,
    max_iter=500,
    lr=0.05,
    tol=1e-12,
    optimizer_type='lbfgs',
    seed=None,
    verbose=False,
):
    """Optimize spring parameters to hit target Poisson ratio and/or Young's modulus.

    Unlike run_single_optimization (which matches all 6 tensor components),
    this targets scalar properties directly.

    Args:
        solver:          ElasticSolver instance.
        n_triangles:     Number of triangles.
        target_poisson:  Target Poisson's ratio (None = don't target).
        target_young:    Target Young's modulus (None = don't target).
        weight_poisson:  Weight for Poisson loss term.
        weight_young:    Weight for Young loss term.
        design_variable: What to optimize:
                         'rigidities'   — optimize k, keep rest lengths = actual (default)
                         'rest_lengths' — optimize l₀, keep rigidities = 1
                         'both'         — optimize both k and l₀ simultaneously
        isotropic:       If True, add penalty terms enforcing 2D isotropy:
                         C_xxxx = C_yyyy, C_xxxy = C_xyyy = 0,
                         C_xyxy = (C_xxxx - C_xxyy) / 2.
        weight_isotropy: Weight for isotropy penalty (relative to Poisson/Young).
        max_iter, lr, tol, optimizer_type, seed, verbose: same as run_single_optimization.

    Returns:
        dict with 'converged', 'final_loss', 'iterations', 'rigidities',
        'rest_lengths', 'poisson', 'young', 'pred_tensor', 'time_seconds'.
    """
    if target_poisson is None and target_young is None:
        raise ValueError("Must specify at least one of target_poisson or target_young")
    if design_variable not in ('rigidities', 'rest_lengths', 'both'):
        raise ValueError(f"design_variable must be 'rigidities', 'rest_lengths', or 'both'")

    if seed is not None:
        torch.manual_seed(seed)

    shape = (n_triangles, n_edges_per_tri)
    actual_lengths = solver.actual_length2.sqrt()  # (N, 3)

    # Set up optimizable parameters based on design_variable
    opt_params = []

    if design_variable in ('rigidities', 'both'):
        init_rigs = torch.exp(0.5 * torch.randn(*shape, dtype=torch.float64))
        raw_k = rigidities_to_raw(init_rigs).clone().detach().requires_grad_(True)
        opt_params.append(raw_k)
    else:
        # Fixed unit rigidities
        raw_k = None
        fixed_k = torch.ones(*shape, dtype=torch.float64)

    if design_variable in ('rest_lengths', 'both'):
        # Initialize near actual lengths (small perturbation)
        init_rl = actual_lengths * torch.exp(0.1 * torch.randn(*shape, dtype=torch.float64))
        raw_rl = rigidities_to_raw(init_rl).clone().detach().requires_grad_(True)
        opt_params.append(raw_rl)
    else:
        raw_rl = None

    best_loss = float('inf')
    best_params = [p.clone().detach() for p in opt_params]
    iterations = 0
    t0 = time.time()

    def compute_loss():
        optimizer.zero_grad()
        # Build rigidities
        if raw_k is not None:
            k = raw_to_rigidities(raw_k)
        else:
            k = fixed_k
        # Build rest lengths
        if raw_rl is not None:
            rl = raw_to_rigidities(raw_rl)
        else:
            rl = None
        result = solver(k, rest_lengths=rl)
        C = result['elastic_tensor']  # [C_xxxx, C_xxxy, C_xxyy, C_xyxy, C_xyyy, C_yyyy]

        loss = torch.tensor(0.0, dtype=torch.float64)
        if target_poisson is not None:
            if isotropic:
                # Isotropic 2D Poisson: ν = C_xxyy / C_xxxx
                # (For isotropic: C_voigt = E/(1-ν²) × [[1,ν,0],[ν,1,0],[0,0,(1-ν)/2]]
                #  so C_xxyy/C_xxxx = ν exactly.)
                nu_iso = C[2] / C[0]
                loss = loss + weight_poisson * (nu_iso - target_poisson) ** 2
            else:
                loss = loss + weight_poisson * (result['poisson'] - target_poisson) ** 2
        if target_young is not None:
            loss = loss + weight_young * (result['young'] - target_young) ** 2

        if isotropic:
            # Normalize penalties by tensor scale to keep them well-conditioned
            scale2 = (C[0] ** 2 + C[5] ** 2).detach().clamp(min=1e-30)
            # 1) C_xxxx = C_yyyy
            loss = loss + weight_isotropy * (C[0] - C[5]) ** 2 / scale2
            # 2) C_xxxy = 0
            loss = loss + weight_isotropy * C[1] ** 2 / scale2
            # 3) C_xyyy = 0
            loss = loss + weight_isotropy * C[4] ** 2 / scale2
            # 4) C_xyxy = (C_xxxx - C_xxyy) / 2
            loss = loss + weight_isotropy * (C[3] - (C[0] - C[2]) / 2) ** 2 / scale2

        loss.backward()
        return loss

    if optimizer_type == 'lbfgs':
        optimizer = torch.optim.LBFGS(
            opt_params, lr=lr, max_iter=20, line_search_fn='strong_wolfe',
            tolerance_grad=1e-14, tolerance_change=1e-16,
        )
        for outer in range(max_iter // 20 + 1):
            loss_val = optimizer.step(compute_loss)
            iterations += 20
            current_loss = loss_val.item()
            if current_loss < best_loss:
                best_loss = current_loss
                best_params = [p.clone().detach() for p in opt_params]
            if verbose and outer % 5 == 0:
                with torch.no_grad():
                    k = raw_to_rigidities(raw_k) if raw_k is not None else fixed_k
                    rl = raw_to_rigidities(raw_rl) if raw_rl is not None else None
                    r = solver(k, rest_lengths=rl)
                print(f"  iter {iterations:4d}  loss={current_loss:.3e}"
                      f"  ν={r['poisson'].item():.6f}  E={r['young'].item():.6f}")
            if current_loss < tol:
                break
    elif optimizer_type == 'adam':
        optimizer = torch.optim.Adam(opt_params, lr=lr)
        for it in range(max_iter):
            loss_val = compute_loss()
            optimizer.step()
            iterations = it + 1
            current_loss = loss_val.item()
            if current_loss < best_loss:
                best_loss = current_loss
                best_params = [p.clone().detach() for p in opt_params]
            if verbose and it % 100 == 0:
                with torch.no_grad():
                    k = raw_to_rigidities(raw_k) if raw_k is not None else fixed_k
                    rl = raw_to_rigidities(raw_rl) if raw_rl is not None else None
                    r = solver(k, rest_lengths=rl)
                print(f"  iter {it:4d}  loss={current_loss:.3e}"
                      f"  ν={r['poisson'].item():.6f}  E={r['young'].item():.6f}")
            if current_loss < tol:
                break

    elapsed = time.time() - t0

    # Recover best parameters
    with torch.no_grad():
        pi = 0
        if raw_k is not None:
            k_best = raw_to_rigidities(best_params[pi])
            pi += 1
        else:
            k_best = fixed_k
        if raw_rl is not None:
            rl_best = raw_to_rigidities(best_params[pi])
        else:
            rl_best = actual_lengths
        result = solver(k_best, rest_lengths=rl_best if raw_rl is not None else None)

    C_np = result['elastic_tensor'].numpy()
    out = {
        'converged': best_loss < tol * 1000,
        'final_loss': best_loss,
        'iterations': iterations,
        'rigidities': k_best.numpy(),
        'rest_lengths': rl_best.numpy(),
        'poisson': result['poisson'].item(),
        'young': result['young'].item(),
        'pred_tensor': C_np,
        'time_seconds': elapsed,
        'design_variable': design_variable,
        'isotropic': isotropic,
    }

    # Add full anisotropic properties via compliance matrix
    C_voigt = np.array([
        [C_np[0], C_np[2], C_np[1]],
        [C_np[2], C_np[5], C_np[4]],
        [C_np[1], C_np[4], C_np[3]],
    ])
    try:
        S = np.linalg.inv(C_voigt)
        out['nu_xy'] = -S[1, 0] / S[0, 0]
        out['nu_yx'] = -S[0, 1] / S[1, 1]
        out['E_x'] = 1.0 / S[0, 0]
        out['E_y'] = 1.0 / S[1, 1]
        out['G_xy'] = 1.0 / S[2, 2]
    except np.linalg.LinAlgError:
        out['nu_xy'] = out['nu_yx'] = float('nan')
        out['E_x'] = out['E_y'] = out['G_xy'] = float('nan')

    return out


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
# Visualization: compare multiple solutions on the mesh
# ---------------------------------------------------------------------------

def visualize_solutions(triangulation, solutions, title=None, save_path=None):
    """Plot the mesh with edges colored by rigidity for multiple solutions.

    Each subplot shows one converged solution. Edges are colored by their
    rigidity value (log scale), so you can see how different solutions
    distribute stiffness across the network to achieve the same tensor.

    Args:
        triangulation: scipy.spatial.Delaunay with .points, .simplices
        solutions:     list of dicts, each with 'rigidities' (N, 3) array
                       and optionally 'poisson', 'young', 'rel_error'.
        title:         Overall figure title.
        save_path:     If given, save the figure to this path.

    Returns:
        matplotlib Figure object.
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LogNorm
    import matplotlib.cm as cm

    points = triangulation.points
    simplices = triangulation.simplices
    n_sols = len(solutions)
    ncols = min(n_sols, 4)
    nrows = (n_sols + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows),
                             squeeze=False)
    axes = axes.ravel().tolist()

    # Collect all rigidities for consistent color scale
    all_rigs = np.concatenate([s['rigidities'].ravel() for s in solutions])
    vmin, vmax = all_rigs.min(), all_rigs.max()
    # Clamp for log scale
    vmin = max(vmin, 1e-6)
    norm = LogNorm(vmin=vmin, vmax=max(vmax, vmin * 10))

    for idx, (ax, sol) in enumerate(zip(axes, solutions)):
        rigs = sol['rigidities']  # (N, 3)

        # Build edge segments and colors
        segments = []
        colors = []
        for tri_idx, tri in enumerate(simplices):
            edge_pairs = [(tri[i], tri[j]) for i in range(3) for j in range(i+1, 3)]
            for e_idx, (a, b) in enumerate(edge_pairs):
                segments.append([points[a], points[b]])
                colors.append(rigs[tri_idx, e_idx])

        colors = np.array(colors)
        lc = LineCollection(segments, cmap='viridis', norm=norm, linewidths=0.8)
        lc.set_array(colors)
        ax.add_collection(lc)
        ax.set_xlim(points[:, 0].min() - 0.5, points[:, 0].max() + 0.5)
        ax.set_ylim(points[:, 1].min() - 0.5, points[:, 1].max() + 0.5)
        ax.set_aspect('equal')

        subtitle = f"Solution {idx + 1}"
        if 'poisson' in sol:
            subtitle += f"\nν={sol['poisson']:.4f}"
        if 'young' in sol:
            subtitle += f"  E={sol['young']:.4f}"
        if 'rel_error' in sol:
            subtitle += f"\nerr={sol['rel_error']:.1e}"
        ax.set_title(subtitle, fontsize=9)
        ax.tick_params(labelsize=7)

    # Hide unused axes
    for ax in axes[n_sols:]:
        ax.set_visible(False)

    fig.colorbar(cm.ScalarMappable(norm=norm, cmap='viridis'),
                 ax=axes[:n_sols], label='Edge rigidity (log scale)',
                 shrink=0.8)

    if title:
        fig.suptitle(title, fontsize=13, y=1.02)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    return fig


def visualize_rigidity_histograms(solutions, save_path=None):
    """Plot overlaid histograms of rigidity distributions for each solution.

    Shows how different solutions distribute edge stiffness differently.

    Args:
        solutions: list of dicts, each with 'rigidities' (N,3) array.
        save_path: If given, save figure.

    Returns:
        matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))

    for idx, sol in enumerate(solutions):
        rigs = sol['rigidities'].ravel()
        ax.hist(rigs, bins=50, alpha=0.4, label=f"Solution {idx+1}",
                density=True, edgecolor='none')

    ax.set_xlabel('Edge rigidity')
    ax.set_ylabel('Density')
    ax.set_title('Rigidity distributions across converged solutions\n'
                 '(all achieve the same elastic tensor)')
    ax.legend()

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    return fig


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
