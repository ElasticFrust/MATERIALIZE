"""Shared utilities for Phase 3 sweep scripts.

Contains topology generation, optimisation, and plotting functions
used across sweep runs and post-processing scripts.
"""

import os, time, warnings
import numpy as np
import scipy as sp
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import TwoSlopeNorm, LogNorm
import matplotlib.cm as cm

import Disc_2_Cont_optimized as D2C
from forward_solver_torch import from_triangulation

# ── Default configuration ─────────────────────────────────────────────────
K_CLAMP       = (1e-6, 1e6)
RL_CLAMP      = (1e-4, 1e4)
NU_INIT_LIMIT = 5.0

TOPO_CLASSES = {
    'iso_crystal':    'crystal',
    'aniso_crystal':  'crystal',
    'foam_eta02_42':  'foam_02',
    'foam_eta02_137': 'foam_02',
    'foam_eta045_256':'foam_045',
    'foam_eta045_314':'foam_045',
    'poisson_999':    'poisson',
    'poisson_1337':   'poisson',
}


# ── Parameterisation ──────────────────────────────────────────────────────
def raw_to_k(raw):
    return F.softplus(raw, beta=5.0)


def k_to_raw(k):
    return torch.log(torch.expm1(5.0 * k)) / 5.0


# ── Topology generators ──────────────────────────────────────────────────
def generate_poisson_network(size):
    """True Poisson point process: uniform random points + Delaunay."""
    density = 2.0 / np.sqrt(3)
    x_lo, x_hi = -(size[0] + 2), size[0] + 2
    y_lo, y_hi = -(size[1] + 2), size[1] + 2
    area = (x_hi - x_lo) * (y_hi - y_lo)
    n_points = int(round(density * area))
    points = np.column_stack([
        np.random.uniform(x_lo, x_hi, n_points),
        np.random.uniform(y_lo, y_hi, n_points),
    ])
    DM = sp.spatial.Delaunay(points)
    centroids = np.mean(DM.points[DM.simplices], axis=1)
    goods = (np.abs(centroids[:, 0]) <= size[0]) & (np.abs(centroids[:, 1]) <= size[1])
    DM.all_simplices = DM.simplices
    DM.simplices = DM.simplices[np.where(goods)]
    return DM


def generate_all_topologies(mesh_size):
    """Generate the standard set of 8 topologies for sweeps.

    Returns list of dicts with keys: name, tri, solver, n_tri,
    default_rigs, actual_rl, natural_poisson, natural_young, seed, topo_class.
    """
    topologies = []

    def _add(name, tri, seed_val):
        solver, default_rigs, _ = from_triangulation(tri)
        n_tri = len(tri.simplices)
        actual_rl = solver.actual_length2.sqrt().numpy()
        with torch.no_grad():
            gt = solver(default_rigs)
        topologies.append({
            'name': name, 'tri': tri, 'solver': solver,
            'n_tri': n_tri, 'default_rigs': default_rigs,
            'actual_rl': actual_rl,
            'natural_poisson': gt['poisson'].item(),
            'natural_young': gt['young'].item(),
            'seed': seed_val,
            'topo_class': TOPO_CLASSES[name],
        })

    # Crystals
    tri = D2C.generate_cryratl_points(size=mesh_size, shape=(1, 1), orientation=0)
    _add('iso_crystal', tri, 0)
    tri = D2C.generate_cryratl_points(size=mesh_size, shape=(1.5, 0.8),
                                      orientation=np.pi / 6)
    _add('aniso_crystal', tri, 0)

    # Foams eta=0.2
    for seed in [42, 137]:
        np.random.seed(seed)
        tri = D2C.generate_foam_points(size=mesh_size, eta=0.2)
        _add(f'foam_eta02_{seed}', tri, seed)

    # Foams eta=0.45
    for seed in [256, 314]:
        np.random.seed(seed)
        tri = D2C.generate_foam_points(size=mesh_size, eta=0.45)
        _add(f'foam_eta045_{seed}', tri, seed)

    # Poisson random
    for seed in [999, 1337]:
        np.random.seed(seed)
        tri = generate_poisson_network(size=mesh_size)
        _add(f'poisson_{seed}', tri, seed)

    return topologies


# ── Single optimisation run ──────────────────────────────────────────────
def run_optimisation(solver, n_tri, target_nu, dv, seed, init_width=0.5,
                     weight_isotropy=10.0, max_iter=1000, lr=0.05):
    """One L-BFGS run with clamped parameters. Returns dict or None."""
    torch.manual_seed(seed)
    shape = (n_tri, 3)
    actual_lengths = solver.actual_length2.sqrt()

    opt_params = []

    if dv in ('rigidities', 'both'):
        init_k = torch.exp(init_width * torch.randn(*shape, dtype=torch.float64)).clamp(*K_CLAMP)
        raw_k = k_to_raw(init_k).clone().detach().requires_grad_(True)
        opt_params.append(raw_k)
    else:
        raw_k = None
        fixed_k = torch.ones(*shape, dtype=torch.float64)

    if dv in ('rest_lengths', 'both'):
        init_rl = (actual_lengths * torch.exp(init_width * torch.randn(*shape, dtype=torch.float64))).clamp(*RL_CLAMP)
        raw_rl = k_to_raw(init_rl).clone().detach().requires_grad_(True)
        opt_params.append(raw_rl)
    else:
        raw_rl = None

    # Check initial nu — reject degenerate inits
    with torch.no_grad():
        k0 = raw_to_k(raw_k) if raw_k is not None else fixed_k
        rl0 = raw_to_k(raw_rl) if raw_rl is not None else None
        try:
            r0 = solver(k0, rest_lengths=rl0)
        except Exception:
            return None
        nu0 = r0['poisson'].item()
        if np.isnan(nu0) or np.isinf(nu0) or abs(nu0) > NU_INIT_LIMIT:
            return None

    best_loss = float('inf')
    best_params = [p.clone().detach() for p in opt_params]
    iterations = 0
    t0 = time.time()

    def compute_loss():
        optimizer.zero_grad()
        k = raw_to_k(raw_k).clamp(*K_CLAMP) if raw_k is not None else fixed_k
        rl = raw_to_k(raw_rl).clamp(*RL_CLAMP) if raw_rl is not None else None
        result = solver(k, rest_lengths=rl)
        C = result['elastic_tensor']

        if torch.isnan(C).any() or torch.isinf(C).any():
            return torch.tensor(float('inf'), dtype=torch.float64, requires_grad=True)

        nu_iso = C[2] / C[0]
        loss = (nu_iso - target_nu) ** 2

        if weight_isotropy > 0:
            scale2 = (C[0]**2 + C[5]**2).detach().clamp(min=1e-30)
            loss = loss + weight_isotropy * (C[0] - C[5])**2 / scale2
            loss = loss + weight_isotropy * C[1]**2 / scale2
            loss = loss + weight_isotropy * C[4]**2 / scale2
            loss = loss + weight_isotropy * (C[3] - (C[0] - C[2]) / 2)**2 / scale2

        if torch.isnan(loss) or torch.isinf(loss):
            return torch.tensor(float('inf'), dtype=torch.float64, requires_grad=True)

        loss.backward()

        for p in opt_params:
            if p.grad is not None:
                p.grad.clamp_(-1e3, 1e3)

        return loss

    optimizer = torch.optim.LBFGS(
        opt_params, lr=lr, max_iter=20, line_search_fn='strong_wolfe',
        tolerance_grad=1e-14, tolerance_change=1e-16)

    nan_streak = 0
    for outer in range(max_iter // 20 + 1):
        try:
            loss_val = optimizer.step(compute_loss)
        except Exception:
            nan_streak += 1
            if nan_streak > 3:
                break
            continue

        iterations += 20
        current = loss_val.item()

        if np.isnan(current) or np.isinf(current):
            nan_streak += 1
            if nan_streak > 3:
                break
            continue
        nan_streak = 0

        if current < best_loss:
            best_loss = current
            best_params = [p.clone().detach() for p in opt_params]
        if current < 1e-14:
            break

    elapsed = time.time() - t0

    if np.isnan(best_loss) or np.isinf(best_loss):
        return None

    # Recover best
    with torch.no_grad():
        pi = 0
        if raw_k is not None:
            k_best = raw_to_k(best_params[pi]).clamp(*K_CLAMP); pi += 1
        else:
            k_best = fixed_k
        if raw_rl is not None:
            rl_best = raw_to_k(best_params[pi]).clamp(*RL_CLAMP)
        else:
            rl_best = actual_lengths

        result = solver(k_best, rest_lengths=rl_best if raw_rl is not None else None)

    C_np = result['elastic_tensor'].numpy()
    C_voigt = np.array([
        [C_np[0], C_np[2], C_np[1]],
        [C_np[2], C_np[5], C_np[4]],
        [C_np[1], C_np[4], C_np[3]],
    ])
    try:
        S = np.linalg.inv(C_voigt)
        nu_xy = -S[1, 0] / S[0, 0]
        nu_yx = -S[0, 1] / S[1, 1]
        E_x = 1.0 / S[0, 0]
        E_y = 1.0 / S[1, 1]
    except np.linalg.LinAlgError:
        nu_xy = nu_yx = E_x = E_y = float('nan')

    return {
        'final_loss': best_loss,
        'iterations': iterations,
        'rigidities': k_best.numpy(),
        'rest_lengths': rl_best.numpy(),
        'poisson': result['poisson'].item(),
        'young': result['young'].item(),
        'nu_xy': nu_xy,
        'nu_yx': nu_yx,
        'E_x': E_x,
        'E_y': E_y,
        'pred_tensor': C_np,
        'time_seconds': elapsed,
        'init_width': init_width,
        'seed': seed,
    }


def run_case(solver, n_tri, target_nu, dv, n_restarts=5,
             init_widths=(0.5, 1.5, 2.5), **opt_kwargs):
    """Run multiple restarts with different init widths, return best result."""
    best = None
    for restart_idx in range(n_restarts):
        width = init_widths[restart_idx % len(init_widths)]
        seed = restart_idx * 137 + 42
        r = run_optimisation(solver, n_tri, target_nu, dv, seed=seed,
                             init_width=width, **opt_kwargs)
        if r is not None and (best is None or r['final_loss'] < best['final_loss']):
            best = r
    return best


# ── Plotting helpers ──────────────────────────────────────────────────────
def get_edge_angles(tri_obj):
    """Compute edge angles (radians) for each edge in each triangle."""
    points = tri_obj.points
    simplices = tri_obj.simplices
    angles = np.zeros((len(simplices), 3))
    for ti, sv in enumerate(simplices):
        for ei, (a, b) in enumerate([(sv[0],sv[1]),(sv[0],sv[2]),(sv[1],sv[2])]):
            dx = points[b, 0] - points[a, 0]
            dy = points[b, 1] - points[a, 1]
            angles[ti, ei] = np.arctan2(dy, dx)
    return angles


def compute_residual_energy(solver, rigs, rl):
    """Compute per-edge residual energy: 0.5 * k * (l - l0)^2."""
    actual_l = solver.actual_length2.sqrt().numpy()
    return 0.5 * rigs * (actual_l - rl) ** 2


def plot_mesh(tri_obj, rigs, rl, actual_rl, dv, ax, title_extra=''):
    """Plot mesh with edges coloured by rigidity or rest-length ratio."""
    points = tri_obj.points
    simplices = tri_obj.simplices
    segments, colors, lws = [], [], []

    if dv == 'rest_lengths':
        vals = (rl / actual_rl).ravel()
        label = 'l0/l_actual'
        p2, p98 = np.percentile(vals, [2, 98])
        if abs(p2 - p98) < 1e-10:
            p2, p98 = vals.min(), vals.max()
        if p2 == p98:
            p2, p98 = p2 - 0.1, p98 + 0.1
        norm = TwoSlopeNorm(vmin=min(p2, 0.5), vcenter=1.0, vmax=max(p98, 2.0))
        cmap = 'coolwarm'
        for ti, sv in enumerate(simplices):
            for ei, (a, b) in enumerate([(sv[0],sv[1]),(sv[0],sv[2]),(sv[1],sv[2])]):
                segments.append([points[a], points[b]])
                v = rl[ti, ei] / actual_rl[ti, ei]
                colors.append(v)
                lws.append(0.3 + 2.0 * abs(v - 1.0))
    else:
        vals = rigs.ravel()
        vals_pos = vals[vals > 0]
        if len(vals_pos) == 0:
            vals_pos = np.array([1.0])
        p2, p98 = np.percentile(vals_pos, [2, 98])
        if p98 / max(p2, 1e-15) < 100:
            p2 = p98 / 100
        norm = LogNorm(vmin=max(p2, 1e-10), vmax=max(p98, p2 * 100), clip=True)
        cmap = 'inferno'
        label = 'Rigidity k'
        for ti, sv in enumerate(simplices):
            for ei, (a, b) in enumerate([(sv[0],sv[1]),(sv[0],sv[2]),(sv[1],sv[2])]):
                segments.append([points[a], points[b]])
                c = max(rigs[ti, ei], 1e-10)
                colors.append(c)
                log_c = np.log10(c)
                log_lo = np.log10(max(p2, 1e-10))
                log_hi = np.log10(max(p98, p2 * 100))
                frac = (log_c - log_lo) / max(log_hi - log_lo, 1e-10)
                lws.append(0.15 + 2.35 * np.clip(frac, 0, 1))

    colors = np.array(colors)
    lc = LineCollection(segments, cmap=cmap, norm=norm, linewidths=lws)
    lc.set_array(colors)
    ax.add_collection(lc)
    ax.set_xlim(points[:, 0].min() - 0.5, points[:, 0].max() + 0.5)
    ax.set_ylim(points[:, 1].min() - 0.5, points[:, 1].max() + 0.5)
    ax.set_aspect('equal')
    ax.set_title(title_extra, fontsize=7)
    plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                 label=label, shrink=0.7, pad=0.02)


def plot_plain_mesh(tri_obj, ax, title=''):
    """Plot a plain gray mesh (for cases with no valid optimisation result)."""
    pts = tri_obj.points
    simps = tri_obj.simplices
    segs = []
    for sv in simps:
        for a, b in [(sv[0],sv[1]),(sv[0],sv[2]),(sv[1],sv[2])]:
            segs.append([pts[a], pts[b]])
    lc = LineCollection(segs, colors='gray', linewidths=0.3)
    ax.add_collection(lc)
    ax.set_xlim(pts[:, 0].min() - 0.5, pts[:, 0].max() + 0.5)
    ax.set_ylim(pts[:, 1].min() - 0.5, pts[:, 1].max() + 0.5)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=9)
