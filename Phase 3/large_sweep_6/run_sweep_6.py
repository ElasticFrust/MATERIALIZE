#!/usr/bin/env python
"""Large sweep 6 — Robust isotropic Poisson-ratio targeting (15×15 meshes).

Same as sweep 5 but on LARGER meshes (15×15) with 5 restarts to keep
runtime reasonable. Larger meshes have more edges → more design freedom
→ potentially better optimisation.

Total: 11 targets × 8 topos × 3 DVs = 264 cases
"""

import sys, os, json, time, glob, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'Phase 2'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'Phase 3'))
os.environ['MPLBACKEND'] = 'Agg'
warnings.filterwarnings('ignore', category=UserWarning)

import numpy as np
import scipy as sp
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import TwoSlopeNorm, LogNorm, Normalize
import matplotlib.cm as cm

import Disc_2_Cont_optimized as D2C
from forward_solver_torch import from_triangulation

# ── Configuration ─────────────────────────────────────────────────────────
POISSON_TARGETS = [-0.9, -0.7, -0.5, -0.3, -0.1, 0.0, 0.1, 0.3, 0.5, 0.7, 0.9]
DESIGN_VARS     = ['rigidities', 'rest_lengths', 'both']
MESH_SIZE       = (15, 15)
MAX_ITER        = 2000
LR              = 0.05
TOL             = 1e-14
ISOTROPIC       = True
WEIGHT_ISOTROPY = 10.0
SUCCESS_THRESH  = 0.01
K_CLAMP         = (1e-6, 1e6)        # clamp rigidities to prevent overflow
RL_CLAMP        = (1e-4, 1e4)        # clamp rest lengths
NU_INIT_LIMIT   = 5.0                # reject init if |nu| > this
N_RESTARTS      = 5                  # random restarts per case (fewer for larger mesh)
INIT_WIDTHS     = [0.5, 1.5, 2.5]   # spread for log-normal init
OUT             = os.path.dirname(os.path.abspath(__file__))

# Seeds for random restarts — spread across widths
OPT_SEEDS = list(range(N_RESTARTS))

# Topology class mapping
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


# ── Helper: Poisson random network ───────────────────────────────────────
def generate_poisson_network(size, eta):
    v1 = np.array([1, 0])
    v2 = np.array([0.5, np.sqrt(3)/2])
    MaxPos = int(round(2 * (max(size) / min([np.sqrt(3)/2, 1]))))
    points = []
    for n in range(-MaxPos, MaxPos):
        for m in range(-MaxPos, MaxPos):
            theta = 2 * np.pi * np.random.rand()
            pt = n*v1 + m*v2 + eta * np.array([np.cos(theta), np.sin(theta)])
            points.append(pt)
    points = np.array(points)
    mask = ((points[:, 0] <= size[0]+2) & (points[:, 0] >= -(size[0]+2)) &
            (points[:, 1] <= size[1]+2) & (points[:, 1] >= -(size[1]+2)))
    points = points[mask]
    DM = sp.spatial.Delaunay(points)
    centroids = np.mean(DM.points[DM.simplices], axis=1)
    goods = (np.abs(centroids[:, 0]) <= size[0]) & (np.abs(centroids[:, 1]) <= size[1])
    DM.all_simplices = DM.simplices
    DM.simplices = DM.simplices[np.where(goods)]
    return DM


# ── Single optimisation run ──────────────────────────────────────────────
def run_optimisation(solver, n_tri, target_nu, dv, seed, init_width=0.5):
    """One L-BFGS run with clamped parameters. Returns dict or None."""
    torch.manual_seed(seed)
    shape = (n_tri, 3)
    actual_lengths = solver.actual_length2.sqrt()

    opt_params = []

    # Rigidities
    if dv in ('rigidities', 'both'):
        init_k = torch.exp(init_width * torch.randn(*shape, dtype=torch.float64)).clamp(*K_CLAMP)
        raw_k = k_to_raw(init_k).clone().detach().requires_grad_(True)
        opt_params.append(raw_k)
    else:
        raw_k = None
        fixed_k = torch.ones(*shape, dtype=torch.float64)

    # Rest lengths
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

        # Check for NaN
        if torch.isnan(C).any() or torch.isinf(C).any():
            return torch.tensor(float('inf'), dtype=torch.float64, requires_grad=True)

        nu_iso = C[2] / C[0]
        loss = (nu_iso - target_nu) ** 2

        scale2 = (C[0]**2 + C[5]**2).detach().clamp(min=1e-30)
        loss = loss + WEIGHT_ISOTROPY * (C[0] - C[5])**2 / scale2
        loss = loss + WEIGHT_ISOTROPY * C[1]**2 / scale2
        loss = loss + WEIGHT_ISOTROPY * C[4]**2 / scale2
        loss = loss + WEIGHT_ISOTROPY * (C[3] - (C[0] - C[2]) / 2)**2 / scale2

        if torch.isnan(loss) or torch.isinf(loss):
            return torch.tensor(float('inf'), dtype=torch.float64, requires_grad=True)

        loss.backward()

        # Clamp gradients to prevent explosion
        for p in opt_params:
            if p.grad is not None:
                p.grad.clamp_(-1e3, 1e3)

        return loss

    optimizer = torch.optim.LBFGS(
        opt_params, lr=LR, max_iter=20, line_search_fn='strong_wolfe',
        tolerance_grad=1e-14, tolerance_change=1e-16)

    nan_streak = 0
    for outer in range(MAX_ITER // 20 + 1):
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
        if current < TOL:
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
    # Compliance matrix for nu_xy, nu_yx
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


def run_case(solver, n_tri, target_nu, dv):
    """Run multiple restarts with different init widths, return best result."""
    best = None
    for restart_idx in range(N_RESTARTS):
        width = INIT_WIDTHS[restart_idx % len(INIT_WIDTHS)]
        seed = restart_idx * 137 + 42
        r = run_optimisation(solver, n_tri, target_nu, dv, seed=seed, init_width=width)
        if r is not None and (best is None or r['final_loss'] < best['final_loss']):
            best = r
    return best


# ── Mesh plot ─────────────────────────────────────────────────────────────
def plot_mesh(tri_obj, rigs, rl, actual_rl, dv, ax, title_extra=''):
    points = tri_obj.points
    simplices = tri_obj.simplices
    segments, colors, lws = [], [], []

    if dv == 'rest_lengths':
        vals = (rl / actual_rl).ravel()
        label = 'l₀/l_actual'
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
    ax.set_xlim(points[:, 0].min()-0.5, points[:, 0].max()+0.5)
    ax.set_ylim(points[:, 1].min()-0.5, points[:, 1].max()+0.5)
    ax.set_aspect('equal')
    ax.set_title(title_extra, fontsize=8)
    plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                 label=label, shrink=0.7, pad=0.02)


# ── Edge angles helper ────────────────────────────────────────────────────
def get_edge_angles(tri_obj):
    points = tri_obj.points
    simplices = tri_obj.simplices
    angles = np.zeros((len(simplices), 3))
    for ti, sv in enumerate(simplices):
        for ei, (a, b) in enumerate([(sv[0],sv[1]),(sv[0],sv[2]),(sv[1],sv[2])]):
            dx = points[b, 0] - points[a, 0]
            dy = points[b, 1] - points[a, 1]
            angles[ti, ei] = np.arctan2(dy, dx)
    return angles


# ── Residual energy ───────────────────────────────────────────────────────
def compute_residual_energy(solver, rigs, rl):
    actual_l = solver.actual_length2.sqrt().numpy()
    return 0.5 * rigs * (actual_l - rl) ** 2


# ── Generate topologies ──────────────────────────────────────────────────
print("=" * 70)
print("Generating topologies ...")
print("=" * 70)

topologies = []

def add_topo(name, tri, seed_val):
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

tri = D2C.generate_cryratl_points(size=MESH_SIZE, shape=(1, 1), orientation=0)
add_topo('iso_crystal', tri, 0)
tri = D2C.generate_cryratl_points(size=MESH_SIZE, shape=(1.5, 0.8), orientation=np.pi/6)
add_topo('aniso_crystal', tri, 0)
for seed in [42, 137]:
    np.random.seed(seed)
    tri = D2C.generate_foam_points(size=MESH_SIZE, eta=0.2)
    add_topo(f'foam_eta02_{seed}', tri, seed)
for seed in [256, 314]:
    np.random.seed(seed)
    tri = D2C.generate_foam_points(size=MESH_SIZE, eta=0.45)
    add_topo(f'foam_eta045_{seed}', tri, seed)
for seed in [999, 1337]:
    np.random.seed(seed)
    tri = generate_poisson_network(size=MESH_SIZE, eta=0.3)
    add_topo(f'poisson_{seed}', tri, seed)

N_TOPOS = len(topologies)
for t in topologies:
    print(f"  {t['name']:25s}  {t['n_tri']:4d} tri  natural nu={t['natural_poisson']:+.4f}"
          f"  class={t['topo_class']}")
print(f"\nTotal: {N_TOPOS} topologies")


# ── Main optimisation loop ───────────────────────────────────────────────
N_CASES = len(POISSON_TARGETS) * N_TOPOS * len(DESIGN_VARS)
print(f"\n{'='*70}")
print(f"Running ISOTROPIC sweep: {len(POISSON_TARGETS)} targets × {N_TOPOS} topos"
      f" × {len(DESIGN_VARS)} dvs = {N_CASES} cases")
print(f"  restarts per case = {N_RESTARTS}, init widths = {INIT_WIDTHS}")
print(f"{'='*70}")

results = {}
t_start = time.time()
case_num = 0

for target_nu in POISSON_TARGETS:
    key = f"{target_nu:+.1f}"
    results[key] = {}
    for topo in topologies:
        results[key][topo['name']] = {}
        for dv in DESIGN_VARS:
            case_num += 1
            elapsed_total = time.time() - t_start
            if case_num > 1:
                eta_s = elapsed_total / (case_num - 1) * (N_CASES - case_num + 1)
            else:
                eta_s = 0

            r = run_case(topo['solver'], topo['n_tri'], target_nu, dv)

            if r is None:
                results[key][topo['name']][dv] = {
                    'final_loss': float('inf'), 'nu_xy': float('nan'),
                    'nu_yx': float('nan'), 'poisson': float('nan'),
                    'young': float('nan'), 'iterations': 0,
                    'time_seconds': 0, 'converged': False,
                }
                tag = 'FAIL'
                nu_str = '   nan'
            else:
                ok = r['final_loss'] < SUCCESS_THRESH
                results[key][topo['name']][dv] = {
                    'final_loss': r['final_loss'],
                    'nu_xy': float(r['nu_xy']),
                    'nu_yx': float(r['nu_yx']),
                    'poisson': r['poisson'],
                    'young': r['young'],
                    'E_x': float(r.get('E_x', float('nan'))),
                    'E_y': float(r.get('E_y', float('nan'))),
                    'iterations': r['iterations'],
                    'time_seconds': r['time_seconds'],
                    'converged': ok,
                    'init_width': r['init_width'],
                    'seed': r['seed'],
                    'pred_tensor': r['pred_tensor'].tolist(),
                }
                tag = '[OK]' if ok else '[  ]'
                nu_str = f'{r["nu_xy"]:+.4f}'

            print(f"  [{case_num:3d}/{N_CASES}] nu*={target_nu:+.1f}  "
                  f"topo={topo['name']:25s}  dv={dv:13s}  "
                  f"nu_xy={nu_str}  "
                  f"loss={results[key][topo['name']][dv]['final_loss']:.2e}  "
                  f"{tag}  [ETA {eta_s:.0f}s]")

total_time = time.time() - t_start
print(f"\nSweep complete in {total_time:.1f}s")

# Save JSON
json_path = os.path.join(OUT, 'sweep_results.json')
with open(json_path, 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"Saved {json_path}")


# ── Per-target visualisations ─────────────────────────────────────────────
print(f"\n{'='*70}")
print("Generating per-target visualisations + markers ...")
print(f"{'='*70}")

# We need to re-run best cases to get per-edge arrays for plots
# (JSON only stores scalars)
edge_data = {}  # [key][topo_name][dv] -> full result dict with arrays

for target_nu in POISSON_TARGETS:
    key = f"{target_nu:+.1f}"
    edge_data[key] = {}

    if target_nu < 0:
        tag = f"num{abs(target_nu):.1f}".replace('.', '')
    elif target_nu == 0:
        tag = "nup00"
    else:
        tag = f"nup{target_nu:.1f}".replace('.', '')
    subdir = os.path.join(OUT, tag)
    os.makedirs(subdir, exist_ok=True)

    # Count converged
    n_conv = sum(1 for tn in results[key] for dv in results[key][tn]
                 if results[key][tn][dv].get('converged', False))
    n_total = N_TOPOS * len(DESIGN_VARS)

    # Success/fail marker
    for old in glob.glob(os.path.join(subdir, 'aaa--*')):
        os.remove(old)
    marker = 'aaa--success' if n_conv > 0 else 'aaa--fail'
    open(os.path.join(subdir, marker), 'w').close()

    # Re-run best cases to get arrays
    for topo in topologies:
        edge_data[key][topo['name']] = {}
        for dv in DESIGN_VARS:
            rd = results[key][topo['name']][dv]
            if rd['final_loss'] < 0.1 and rd.get('seed') is not None:
                # Re-run this exact case
                r = run_optimisation(
                    topo['solver'], topo['n_tri'], target_nu, dv,
                    seed=rd['seed'], init_width=rd.get('init_width', 0.5))
                edge_data[key][topo['name']][dv] = r
            else:
                edge_data[key][topo['name']][dv] = None

    # ── Mesh plots ────────────────────────────────────────────────────
    for topo in topologies:
        has_any = any(edge_data[key][topo['name']][dv] is not None for dv in DESIGN_VARS)
        if not has_any:
            # Still plot default mesh for context
            fig, ax = plt.subplots(1, 1, figsize=(6, 5.5))
            pts = topo['tri'].points
            simps = topo['tri'].simplices
            segs = []
            for sv in simps:
                for a, b in [(sv[0],sv[1]),(sv[0],sv[2]),(sv[1],sv[2])]:
                    segs.append([pts[a], pts[b]])
            lc = LineCollection(segs, colors='gray', linewidths=0.3)
            ax.add_collection(lc)
            ax.set_xlim(pts[:,0].min()-0.5, pts[:,0].max()+0.5)
            ax.set_ylim(pts[:,1].min()-0.5, pts[:,1].max()+0.5)
            ax.set_aspect('equal')
            ax.set_title(f'No convergence (all loss > 0.1)\n{topo["name"]}', fontsize=9)
            fig.savefig(os.path.join(subdir, f'mesh_topo_{topo["name"]}.png'),
                        dpi=100, bbox_inches='tight')
            plt.close(fig)
            continue

        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
        for col, dv in enumerate(DESIGN_VARS):
            r = edge_data[key][topo['name']][dv]
            if r is None:
                axes[col].text(0.5, 0.5, 'N/A', transform=axes[col].transAxes,
                               ha='center', va='center', fontsize=14, color='gray')
                axes[col].set_title(f'{dv}', fontsize=9)
                continue
            plot_mesh(topo['tri'], r['rigidities'], r['rest_lengths'],
                      topo['actual_rl'], dv, axes[col],
                      f'{dv}\nnu_xy={r["nu_xy"]:+.4f}  nu_yx={r["nu_yx"]:+.4f}\nloss={r["final_loss"]:.2e}')
        fig.suptitle(f'ISOTROPIC | Target nu={target_nu:+.1f} | {topo["name"]}', fontsize=12, y=1.02)
        fig.tight_layout()
        fig.savefig(os.path.join(subdir, f'mesh_topo_{topo["name"]}.png'),
                    dpi=100, bbox_inches='tight')
        plt.close(fig)

    # ── Histogram + Polar plots ───────────────────────────────────────
    for topo in topologies:
        has_any = any(edge_data[key][topo['name']][dv] is not None for dv in DESIGN_VARS)
        if not has_any:
            continue

        angles = get_edge_angles(topo['tri'])
        actual_rl = topo['actual_rl']

        fig, axes = plt.subplots(2, 3, figsize=(18, 9))

        for col, dv in enumerate(DESIGN_VARS):
            r = edge_data[key][topo['name']][dv]
            if r is None:
                axes[0, col].text(0.5, 0.5, 'N/A', transform=axes[0, col].transAxes,
                                  ha='center', va='center', fontsize=14, color='gray')
                axes[1, col].text(0.5, 0.5, 'N/A', transform=axes[1, col].transAxes,
                                  ha='center', va='center', fontsize=14, color='gray')
                continue

            # Top: histograms
            ax_h = axes[0, col]
            rl_ratio = (r['rest_lengths'] / actual_rl).ravel()
            rigs_flat = r['rigidities'].ravel()
            ax_h.hist(rl_ratio, bins=50, alpha=0.6, color='steelblue',
                      edgecolor='none', density=True, label='l₀/l_actual')
            ax_h.axvline(1.0, color='k', ls='--', lw=0.8)
            ax_h.set_xlabel('l₀ / l_actual')
            ax_h.set_ylabel('Density')
            ax_h.set_title(f'{dv}\nnu_xy={r["nu_xy"]:+.4f}  nu_yx={r["nu_yx"]:+.4f}\nloss={r["final_loss"]:.2e}', fontsize=8)
            ax_h.set_xlim(0, max(4, np.percentile(rl_ratio, 99) * 1.1))
            ax_h2 = ax_h.twinx()
            ax_h2.hist(np.log10(np.clip(rigs_flat, 1e-10, None)), bins=50,
                       alpha=0.4, color='orange', edgecolor='none', density=True, label='log₁₀(k)')
            ax_h2.set_ylabel('Density (log₁₀ k)', color='orange')
            ax_h2.tick_params(axis='y', labelcolor='orange')
            lines1, labels1 = ax_h.get_legend_handles_labels()
            lines2, labels2 = ax_h2.get_legend_handles_labels()
            ax_h.legend(lines1 + lines2, labels1 + labels2, fontsize=7)

            # Bottom: polar
            axes[1, col].remove()
            ax_p = fig.add_subplot(2, 3, 4 + col, projection='polar')
            ratios_flat = (r['rest_lengths'] / actual_rl).ravel()
            angles_flat = angles.ravel()
            vmin_p = min(ratios_flat.min(), 0.3)
            vmax_p = max(ratios_flat.max(), 3.0)
            rl_norm = TwoSlopeNorm(vmin=vmin_p, vcenter=1.0, vmax=vmax_p)
            ax_p.scatter(angles_flat, ratios_flat, c=ratios_flat, cmap='coolwarm',
                         norm=rl_norm, s=1, alpha=0.3)
            ax_p.set_ylim(0, min(5, np.percentile(ratios_flat, 99) * 1.2))
            ax_p.set_title(f'{dv}', fontsize=9, pad=12)

        fig.suptitle(f'ISOTROPIC | Target nu={target_nu:+.1f} | {topo["name"]}\n'
                     f'Top: histograms  |  Bottom: polar (l₀/l_actual vs angle)',
                     fontsize=11, y=1.03)
        fig.tight_layout()
        fig.savefig(os.path.join(subdir, f'hist_polar_{topo["name"]}.png'),
                    dpi=100, bbox_inches='tight')
        plt.close(fig)

    # ── Residual energy mesh ──────────────────────────────────────────
    for topo in topologies:
        has_any = any(edge_data[key][topo['name']][dv] is not None for dv in DESIGN_VARS)
        if not has_any:
            continue

        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
        pts = topo['tri'].points
        simps = topo['tri'].simplices

        for col, dv in enumerate(DESIGN_VARS):
            r = edge_data[key][topo['name']][dv]
            if r is None:
                axes[col].text(0.5, 0.5, 'N/A', transform=axes[col].transAxes,
                               ha='center', va='center', fontsize=14, color='gray')
                axes[col].set_title(f'{dv}', fontsize=9)
                continue

            re = compute_residual_energy(topo['solver'], r['rigidities'], r['rest_lengths'])
            re_flat = re.ravel()
            re_pos = re_flat[re_flat > 0]
            if len(re_pos) > 0 and re_pos.max() > 0:
                p1 = max(np.percentile(re_pos, 1), 1e-15)
                p99 = np.percentile(re_pos, 99)
                re_norm = LogNorm(vmin=p1, vmax=max(p99, p1 * 100), clip=True)
            else:
                re_norm = LogNorm(vmin=1e-15, vmax=1e-5, clip=True)

            segments, colors = [], []
            for ti, sv in enumerate(simps):
                for ei, (a, b) in enumerate([(sv[0],sv[1]),(sv[0],sv[2]),(sv[1],sv[2])]):
                    segments.append([pts[a], pts[b]])
                    colors.append(re[ti, ei])
            colors = np.array(colors)
            log_c = np.log10(np.clip(colors, re_norm.vmin, re_norm.vmax))
            log_range = np.log10(re_norm.vmax) - np.log10(re_norm.vmin)
            if log_range > 0:
                lw = 0.15 + 2.85 * (log_c - np.log10(re_norm.vmin)) / log_range
            else:
                lw = np.full_like(colors, 0.8)
            lc = LineCollection(segments, cmap='hot', norm=re_norm, linewidths=lw)
            lc.set_array(colors)
            axes[col].add_collection(lc)
            axes[col].set_xlim(pts[:,0].min()-0.5, pts[:,0].max()+0.5)
            axes[col].set_ylim(pts[:,1].min()-0.5, pts[:,1].max()+0.5)
            axes[col].set_aspect('equal')
            total_E = np.sum(re)
            axes[col].set_title(f'{dv}\nTotal E = {total_E:.3e}', fontsize=9)
            plt.colorbar(cm.ScalarMappable(norm=re_norm, cmap='hot'),
                         ax=axes[col], label='Residual energy', shrink=0.7, pad=0.02)

        fig.suptitle(f'ISOTROPIC | Residual Energy: 0.5·k·(l−l₀)²\n'
                     f'Target nu={target_nu:+.1f} | {topo["name"]}', fontsize=12, y=1.02)
        fig.tight_layout()
        fig.savefig(os.path.join(subdir, f'residual_energy_{topo["name"]}.png'),
                    dpi=100, bbox_inches='tight')
        plt.close(fig)

    # ── README ────────────────────────────────────────────────────────
    readme_lines = [f"# Target Poisson ratio = {target_nu:+.1f}  (ISOTROPIC)",
                    f"", f"Converged: {n_conv}/{n_total}", f"",
                    f"| Topology | Class | DV | nu_xy | nu_yx | Loss | OK? |",
                    f"|----------|-------|----|-------|-------|------|-----|"]
    for topo in topologies:
        for dv in DESIGN_VARS:
            rd = results[key][topo['name']][dv]
            ok = 'Y' if rd.get('converged') else ''
            readme_lines.append(
                f"| {topo['name']:25s} | {topo['topo_class']:8s} | {dv:13s} | "
                f"{rd['nu_xy']:+.4f} | {rd['nu_yx']:+.4f} | "
                f"{rd['final_loss']:.2e} | {ok} |")
    with open(os.path.join(subdir, 'README.md'), 'w') as f:
        f.write('\n'.join(readme_lines) + '\n')

    print(f"  {tag}: {marker.split('--')[1]} ({n_conv}/{n_total} converged)")


# ── Aggregate statistics by topology class ────────────────────────────────
print(f"\n{'='*70}")
print("Generating aggregate statistics by topology class ...")
print(f"{'='*70}")

classes = ['crystal', 'foam_02', 'foam_045', 'poisson']
class_colors = {'crystal': '#e74c3c', 'foam_02': '#3498db',
                'foam_045': '#2ecc71', 'poisson': '#9b59b6'}

# 1) Success rate per class per target
fig_agg, axes_agg = plt.subplots(2, 2, figsize=(14, 10))
for ci, cls in enumerate(classes):
    ax = axes_agg[ci // 2, ci % 2]
    class_topos = [t for t in topologies if t['topo_class'] == cls]
    n_dvs = len(DESIGN_VARS)

    targets_f = []
    success_rates = []
    best_losses = []
    best_nu_xys = []

    for target_nu in POISSON_TARGETS:
        key = f"{target_nu:+.1f}"
        n_ok = 0
        n_tot = 0
        best_l = float('inf')
        best_nu = float('nan')
        for topo in class_topos:
            for dv in DESIGN_VARS:
                rd = results[key][topo['name']][dv]
                n_tot += 1
                if rd.get('converged'):
                    n_ok += 1
                if rd['final_loss'] < best_l:
                    best_l = rd['final_loss']
                    best_nu = rd['nu_xy']
        targets_f.append(target_nu)
        success_rates.append(n_ok / n_tot if n_tot > 0 else 0)
        best_losses.append(best_l)
        best_nu_xys.append(best_nu)

    ax.bar(targets_f, success_rates, width=0.15, color=class_colors[cls], alpha=0.7)
    ax.set_xlabel('Target ν')
    ax.set_ylabel('Success rate')
    ax.set_title(f'{cls} ({len(class_topos)} topologies)', fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.axhline(1.0, color='gray', ls='--', lw=0.5)

    # Overlay best achieved nu as line
    ax2 = ax.twinx()
    ax2.plot(targets_f, best_nu_xys, 'ko-', markersize=4, label='Best nu_xy')
    ax2.plot(targets_f, targets_f, 'r--', lw=0.8, label='Target')
    ax2.set_ylabel('Best achieved nu_xy')
    ax2.legend(fontsize=7, loc='upper left')

fig_agg.suptitle('Success rate & best achieved ν by topology class (ISOTROPIC)', fontsize=13, y=1.02)
fig_agg.tight_layout()
fig_agg.savefig(os.path.join(OUT, 'aggregate_by_class.png'), dpi=120, bbox_inches='tight')
plt.close(fig_agg)
print("  Saved aggregate_by_class.png")

# 2) Loss heatmap per class
fig_heat, axes_heat = plt.subplots(2, 2, figsize=(14, 10))
for ci, cls in enumerate(classes):
    ax = axes_heat[ci // 2, ci % 2]
    class_topos = [t for t in topologies if t['topo_class'] == cls]
    # Rows: topo×dv, Cols: targets
    row_labels = []
    data_rows = []
    for topo in class_topos:
        for dv in DESIGN_VARS:
            row_labels.append(f"{topo['name'][:15]}\n{dv[:4]}")
            row = []
            for target_nu in POISSON_TARGETS:
                key = f"{target_nu:+.1f}"
                loss = results[key][topo['name']][dv]['final_loss']
                row.append(np.log10(max(loss, 1e-16)))
            data_rows.append(row)

    data = np.array(data_rows)
    im = ax.imshow(data, aspect='auto', cmap='RdYlGn_r', vmin=-14, vmax=1)
    ax.set_xticks(range(len(POISSON_TARGETS)))
    ax.set_xticklabels([f"{t:+.1f}" for t in POISSON_TARGETS], fontsize=7)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=6)
    ax.set_xlabel('Target ν')
    ax.set_title(f'{cls}', fontsize=11)
    plt.colorbar(im, ax=ax, label='log₁₀(loss)', shrink=0.8)

fig_heat.suptitle('Loss heatmap by topology class (ISOTROPIC)', fontsize=13, y=1.02)
fig_heat.tight_layout()
fig_heat.savefig(os.path.join(OUT, 'aggregate_loss_heatmap.png'), dpi=120, bbox_inches='tight')
plt.close(fig_heat)
print("  Saved aggregate_loss_heatmap.png")

# 3) Achieved nu_xy vs target — one line per class (best across topos & DVs)
fig_track, ax_track = plt.subplots(figsize=(8, 6))
for cls in classes:
    class_topos = [t for t in topologies if t['topo_class'] == cls]
    best_nus = []
    for target_nu in POISSON_TARGETS:
        key = f"{target_nu:+.1f}"
        best_nu = float('nan')
        best_l = float('inf')
        for topo in class_topos:
            for dv in DESIGN_VARS:
                rd = results[key][topo['name']][dv]
                if rd['final_loss'] < best_l:
                    best_l = rd['final_loss']
                    best_nu = rd['nu_xy']
        best_nus.append(best_nu)
    ax_track.plot(POISSON_TARGETS, best_nus, 'o-', color=class_colors[cls],
                  label=cls, markersize=5)

ax_track.plot(POISSON_TARGETS, POISSON_TARGETS, 'k--', lw=1, label='Perfect')
ax_track.set_xlabel('Target ν')
ax_track.set_ylabel('Best achieved nu_xy')
ax_track.set_title('Best achieved isotropic ν per topology class')
ax_track.legend()
ax_track.grid(True, alpha=0.3)
fig_track.tight_layout()
fig_track.savefig(os.path.join(OUT, 'aggregate_tracking.png'), dpi=120, bbox_inches='tight')
plt.close(fig_track)
print("  Saved aggregate_tracking.png")

# 4) Summary performance plot (all topos on one axis)
fig_perf, ax_perf = plt.subplots(figsize=(12, 6))
markers = {'rigidities': 'o', 'rest_lengths': 's', 'both': '^'}
for ti, topo in enumerate(topologies):
    cls = topo['topo_class']
    for dv in DESIGN_VARS:
        achieved = []
        for target_nu in POISSON_TARGETS:
            key = f"{target_nu:+.1f}"
            achieved.append(results[key][topo['name']][dv]['nu_xy'])
        ax_perf.plot(POISSON_TARGETS, achieved, marker=markers[dv],
                     color=class_colors[cls], alpha=0.4, markersize=3,
                     linewidth=0.5)
ax_perf.plot(POISSON_TARGETS, POISSON_TARGETS, 'k--', lw=2, label='Perfect')
# Legend for classes
for cls in classes:
    ax_perf.plot([], [], '-', color=class_colors[cls], label=cls, lw=2)
for dv in DESIGN_VARS:
    ax_perf.plot([], [], 'k', marker=markers[dv], label=dv, lw=0, markersize=5)
ax_perf.legend(ncol=2, fontsize=8)
ax_perf.set_xlabel('Target ν')
ax_perf.set_ylabel('Achieved nu_xy')
ax_perf.set_title('All topologies: achieved vs target ν (ISOTROPIC)')
ax_perf.grid(True, alpha=0.3)
fig_perf.tight_layout()
fig_perf.savefig(os.path.join(OUT, 'summary_all_topos.png'), dpi=120, bbox_inches='tight')
plt.close(fig_perf)
print("  Saved summary_all_topos.png")


print(f"\n{'='*70}")
print("ALL DONE")
print(f"{'='*70}")
