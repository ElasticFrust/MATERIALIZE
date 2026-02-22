#!/usr/bin/env python
"""Large sweep 3: 4 topology types × multiple seeds × 11 Poisson targets × 3 design vars.

Topology types:
  0) iso_crystal     — perfect isotropic triangular lattice (shape=(1,1), orientation=0)
  1) aniso_crystal   — anisotropic crystal (shape=(1.5, 0.8), orientation=pi/6)
  2) foam_eta02      — Voronoi foam with eta=0.2  (Delaunay BEFORE perturbation)
  3) foam_eta045     — Voronoi foam with eta=0.45 (Delaunay BEFORE perturbation)
  4) poisson         — Poisson random network     (Delaunay AFTER perturbation, eta=0.3)

Each type gets 2 random seeds (except crystals which are deterministic → 1 seed).
Total topologies: 1 + 1 + 2 + 2 + 2 = 8

For each (target, topology, design_var), we run 3 random restarts and keep the best.
Total cases: 11 targets × 8 topos × 3 dvs = 264
"""

import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'Phase 2'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'Phase 3'))
os.environ['MPLBACKEND'] = 'Agg'

import numpy as np
import scipy as sp
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import TwoSlopeNorm, LogNorm
import matplotlib.cm as cm

import Disc_2_Cont_optimized as D2C
from forward_solver_torch import from_triangulation
from inverse_optimize import run_property_optimization

# ── Configuration ──────────────────────────────────────────────────────────
POISSON_TARGETS = [-0.9, -0.7, -0.5, -0.3, -0.1, 0.0, 0.1, 0.3, 0.5, 0.7, 0.9]
DESIGN_VARS = ['rigidities', 'rest_lengths', 'both']
OPT_SEEDS = [7, 144, 281]       # 3 random restarts
MESH_SIZE = (10, 10)
MAX_ITER = 500
LR = 0.05
TOL = 1e-14
SUCCESS_THRESHOLD = 0.01         # loss < this → plausibly successful
OUT = os.path.dirname(os.path.abspath(__file__))


# ── Helper: Poisson random network (Delaunay AFTER perturbation) ──────────
def generate_poisson_network(size, eta):
    """Perturb points FIRST, then Delaunay triangulation."""
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
    goods = ((np.abs(centroids[:, 0]) <= size[0]) &
             (np.abs(centroids[:, 1]) <= size[1]))
    DM.all_simplices = DM.simplices
    DM.simplices = DM.simplices[np.where(goods)]
    return DM


# ── Generate all topologies ───────────────────────────────────────────────
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
    info = {
        'name': name,
        'tri': tri,
        'solver': solver,
        'n_tri': n_tri,
        'default_rigs': default_rigs,
        'actual_rl': actual_rl,
        'natural_poisson': gt['poisson'].item(),
        'natural_young': gt['young'].item(),
        'seed': seed_val,
    }
    topologies.append(info)
    idx = len(topologies) - 1
    print(f"  Topo {idx}: {name:25s} seed={seed_val:4d}  "
          f"{n_tri:3d} triangles  natural ν={info['natural_poisson']:+.4f}")
    return info

# 0) Isotropic crystal — deterministic, no seed needed
tri = D2C.generate_cryratl_points(size=MESH_SIZE, shape=(1, 1), orientation=0)
add_topo('iso_crystal', tri, 0)

# 1) Anisotropic crystal — stretched lattice
tri = D2C.generate_cryratl_points(size=MESH_SIZE, shape=(1.5, 0.8), orientation=np.pi/6)
add_topo('aniso_crystal', tri, 0)

# 2-3) Foam eta=0.2, two seeds
for seed in [42, 137]:
    np.random.seed(seed)
    tri = D2C.generate_foam_points(size=MESH_SIZE, eta=0.2)
    add_topo(f'foam_eta02_{seed}', tri, seed)

# 4-5) Foam eta=0.45, two seeds
for seed in [256, 314]:
    np.random.seed(seed)
    tri = D2C.generate_foam_points(size=MESH_SIZE, eta=0.45)
    add_topo(f'foam_eta045_{seed}', tri, seed)

# 6-7) Poisson random network (Delaunay AFTER perturbation), two seeds
for seed in [999, 1337]:
    np.random.seed(seed)
    tri = generate_poisson_network(size=MESH_SIZE, eta=0.3)
    add_topo(f'poisson_{seed}', tri, seed)

N_TOPOLOGIES = len(topologies)
print(f"\nTotal: {N_TOPOLOGIES} topologies")

# ── Run the sweep ─────────────────────────────────────────────────────────
print("\n" + "=" * 70)
total = len(POISSON_TARGETS) * N_TOPOLOGIES * len(DESIGN_VARS)
print(f"Running sweep: {len(POISSON_TARGETS)} targets × {N_TOPOLOGIES} topos × "
      f"{len(DESIGN_VARS)} dvs × {len(OPT_SEEDS)} restarts = {total} cases")
print("=" * 70)

all_results = {}
done = 0
t_start = time.time()

for target_nu in POISSON_TARGETS:
    all_results[target_nu] = {}
    for topo_idx, topo in enumerate(topologies):
        all_results[target_nu][topo_idx] = {}
        for dv in DESIGN_VARS:
            best = None
            for opt_seed in OPT_SEEDS:
                try:
                    r = run_property_optimization(
                        solver=topo['solver'],
                        n_triangles=topo['n_tri'],
                        target_poisson=target_nu,
                        weight_poisson=1.0,
                        design_variable=dv,
                        max_iter=MAX_ITER,
                        lr=LR,
                        tol=TOL,
                        optimizer_type='lbfgs',
                        seed=opt_seed,
                        verbose=False,
                    )
                    if best is None or r['final_loss'] < best['final_loss']:
                        best = r
                except Exception as e:
                    print(f"  FAILED: ν*={target_nu}, topo={topo_idx}({topo['name']}), "
                          f"dv={dv}, seed={opt_seed}: {e}")
            all_results[target_nu][topo_idx][dv] = best
            done += 1
            elapsed = time.time() - t_start
            eta_s = (elapsed / done) * (total - done) if done > 0 else 0
            tag = "OK" if best is not None else "FAIL"
            if best is not None:
                print(f"  [{done:3d}/{total}] ν*={target_nu:+.1f}  "
                      f"topo={topo_idx}({topo['name']:25s})  dv={dv:13s}  "
                      f"nu_xy={best['poisson']:+.4f}  loss={best['final_loss']:.2e}  "
                      f"[{tag}]  [ETA {eta_s:.0f}s]")
            else:
                print(f"  [{done:3d}/{total}] ν*={target_nu:+.1f}  "
                      f"topo={topo_idx}({topo['name']:25s})  dv={dv:13s}  "
                      f"→ FAILED ALL SEEDS  [ETA {eta_s:.0f}s]")

print(f"\nSweep complete in {time.time() - t_start:.1f}s")

# ── Save JSON results ─────────────────────────────────────────────────────
summary_data = {}
for target_nu in POISSON_TARGETS:
    key = f"{target_nu:+.1f}"
    summary_data[key] = {}
    for topo_idx in range(N_TOPOLOGIES):
        topo_key = topologies[topo_idx]['name']
        summary_data[key][topo_key] = {}
        for dv in DESIGN_VARS:
            r = all_results[target_nu][topo_idx][dv]
            if r is not None:
                summary_data[key][topo_key][dv] = {
                    'poisson': r['poisson'],
                    'young': r['young'],
                    'final_loss': r['final_loss'],
                    'converged': r['converged'],
                    'iterations': r['iterations'],
                    'time_seconds': r.get('time_seconds', 0),
                }

with open(os.path.join(OUT, 'sweep_results.json'), 'w') as f:
    json.dump(summary_data, f, indent=2)
print("Saved sweep_results.json")

# ── Create per-target folders with plots and success/fail markers ─────────
print("\n" + "=" * 70)
print("Generating per-target visualizations + success/fail markers ...")
print("=" * 70)

folder_map = {}
for target_nu in POISSON_TARGETS:
    if target_nu < 0:
        tag = f"num{abs(target_nu):.1f}".replace('.', '')
    elif target_nu == 0:
        tag = "nup00"
    else:
        tag = f"nup{target_nu:.1f}".replace('.', '')

    subdir = os.path.join(OUT, tag)
    os.makedirs(subdir, exist_ok=True)
    folder_map[target_nu] = (tag, subdir)

    # Check if any (topo, dv) combo is plausibly successful
    any_success = False
    for topo_idx in range(N_TOPOLOGIES):
        for dv in DESIGN_VARS:
            r = all_results[target_nu][topo_idx][dv]
            if r is not None and r['final_loss'] < SUCCESS_THRESHOLD:
                any_success = True
                break
        if any_success:
            break

    marker = 'aaa--success' if any_success else 'aaa--fail'
    open(os.path.join(subdir, marker), 'w').close()
    print(f"  {tag}: {marker}")

    # ── Mesh plots per topology ──
    for topo_idx, topo in enumerate(topologies):
        tri_obj = topo['tri']
        actual_rl = topo['actual_rl']
        fig, axes = plt.subplots(2, 3, figsize=(18, 11))

        for col, dv in enumerate(DESIGN_VARS):
            r = all_results[target_nu][topo_idx][dv]
            if r is None:
                axes[0, col].text(0.5, 0.5, 'FAILED', transform=axes[0, col].transAxes,
                                  ha='center', va='center', fontsize=14, color='red')
                axes[1, col].set_visible(False)
                continue

            # Row 0: rigidities
            rigs = r['rigidities']
            points = tri_obj.points
            simplices = tri_obj.simplices
            rmin = max(rigs.ravel().min(), 1e-6)
            rmax = max(rigs.ravel().max(), rmin * 10)
            k_norm = LogNorm(vmin=rmin, vmax=rmax)

            segments, colors = [], []
            for ti, sv in enumerate(simplices):
                for e_idx, (a, b) in enumerate([(sv[0], sv[1]), (sv[0], sv[2]), (sv[1], sv[2])]):
                    segments.append([points[a], points[b]])
                    colors.append(rigs[ti, e_idx])
            colors = np.array(colors)
            lc = LineCollection(segments, cmap='viridis', norm=k_norm, linewidths=0.6)
            lc.set_array(colors)
            axes[0, col].add_collection(lc)
            axes[0, col].set_xlim(points[:, 0].min()-0.5, points[:, 0].max()+0.5)
            axes[0, col].set_ylim(points[:, 1].min()-0.5, points[:, 1].max()+0.5)
            axes[0, col].set_aspect('equal')
            axes[0, col].set_title(f'{dv}\nν={r["poisson"]:+.4f}  E={r["young"]:.3e}', fontsize=9)
            fig.colorbar(cm.ScalarMappable(norm=k_norm, cmap='viridis'),
                         ax=axes[0, col], label='k', shrink=0.7, pad=0.02)

            # Row 1: rest-length ratio
            rl_ratio = r['rest_lengths'] / actual_rl
            rl_min, rl_max = rl_ratio.min(), rl_ratio.max()
            rl_norm = TwoSlopeNorm(vmin=min(rl_min, 0.3), vcenter=1.0, vmax=max(rl_max, 3.0))
            segments2, colors2 = [], []
            for ti, sv in enumerate(simplices):
                for e_idx, (a, b) in enumerate([(sv[0], sv[1]), (sv[0], sv[2]), (sv[1], sv[2])]):
                    segments2.append([points[a], points[b]])
                    colors2.append(rl_ratio[ti, e_idx])
            colors2 = np.array(colors2)
            lc2 = LineCollection(segments2, cmap='coolwarm', norm=rl_norm, linewidths=0.6)
            lc2.set_array(colors2)
            axes[1, col].add_collection(lc2)
            axes[1, col].set_xlim(points[:, 0].min()-0.5, points[:, 0].max()+0.5)
            axes[1, col].set_ylim(points[:, 1].min()-0.5, points[:, 1].max()+0.5)
            axes[1, col].set_aspect('equal')
            axes[1, col].set_title('l_0/l_actual', fontsize=9)
            fig.colorbar(cm.ScalarMappable(norm=rl_norm, cmap='coolwarm'),
                         ax=axes[1, col], label='l_0/l_actual', shrink=0.7, pad=0.02)

        fig.suptitle(f'Target ν={target_nu:+.1f}  |  Topo {topo_idx}: {topo["name"]} '
                     f'(seed={topo["seed"]}, natural ν={topo["natural_poisson"]:.3f})',
                     fontsize=12, y=1.01)
        axes[0, 0].set_ylabel('Rigidities (log)', fontsize=10)
        axes[1, 0].set_ylabel('Rest-length ratio', fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(subdir, f'mesh_topo{topo_idx}_{topo["name"]}.png'),
                    dpi=100, bbox_inches='tight')
        plt.close(fig)

    print(f"  {tag}: saved mesh plots for {N_TOPOLOGIES} topos")


# ── Summary plots ─────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("Generating summary plots ...")
print("=" * 70)

targets = np.array(POISSON_TARGETS)

def collect(key):
    out = {}
    for dv in DESIGN_VARS:
        arr = np.full((len(POISSON_TARGETS), N_TOPOLOGIES), np.nan)
        for i, nu in enumerate(POISSON_TARGETS):
            for j in range(N_TOPOLOGIES):
                r = all_results[nu][j][dv]
                if r is not None:
                    arr[i, j] = r[key]
        out[dv] = arr
    return out

achieved_nu = collect('poisson')
achieved_E = collect('young')
loss_arr = collect('final_loss')

dv_colors = {'rigidities': 'C0', 'rest_lengths': 'C1', 'both': 'C2'}
dv_labels = {'rigidities': 'Rigidities', 'rest_lengths': 'Rest lengths', 'both': 'Both'}

def plot_mean_std(ax, x, data_dict, ylabel, title):
    for dv in DESIGN_VARS:
        arr = data_dict[dv]
        mean = np.nanmean(arr, axis=1)
        std = np.nanstd(arr, axis=1)
        ax.plot(x, mean, 'o-', color=dv_colors[dv], label=dv_labels[dv], markersize=4)
        ax.fill_between(x, mean - std, mean + std, alpha=0.15, color=dv_colors[dv])
    ax.set_xlabel('Target ν')
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

# Figure A: Performance summary
fig_a, axs = plt.subplots(2, 2, figsize=(14, 10))

ax = axs[0, 0]
for dv in DESIGN_VARS:
    arr = achieved_nu[dv]
    mean = np.nanmean(arr, axis=1)
    std = np.nanstd(arr, axis=1)
    ax.plot(targets, mean, 'o-', color=dv_colors[dv], label=dv_labels[dv], markersize=4)
    ax.fill_between(targets, mean - std, mean + std, alpha=0.15, color=dv_colors[dv])
ax.plot([-1, 1], [-1, 1], 'k--', lw=0.8, label='Perfect')
ax.set_xlabel('Target ν')
ax.set_ylabel('Achieved ν')
ax.set_title('Achieved vs Target Poisson ratio')
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

plot_mean_std(axs[0, 1], targets, achieved_E, "Young's modulus E",
              "Young's modulus vs target ν")
axs[0, 1].set_yscale('log')

plot_mean_std(axs[1, 0], targets, loss_arr, 'Final loss',
              'Optimization loss vs target ν')
axs[1, 0].set_yscale('log')

# Per-topology type comparison
ax = axs[1, 1]
topo_type_colors = {
    'iso_crystal': 'red', 'aniso_crystal': 'darkred',
    'foam_eta02': 'blue', 'foam_eta045': 'green',
    'poisson': 'purple'
}
for j, topo in enumerate(topologies):
    base_name = topo['name'].rsplit('_', 1)[0] if any(c.isdigit() for c in topo['name'].split('_')[-1]) else topo['name']
    color = topo_type_colors.get(base_name, f'C{j}')
    vals = [all_results[nu][j]['rest_lengths']['poisson']
            if all_results[nu][j]['rest_lengths'] is not None else np.nan
            for nu in POISSON_TARGETS]
    ax.plot(targets, vals, 'o-', color=color, label=topo['name'], markersize=3, alpha=0.7)
ax.plot([-1, 1], [-1, 1], 'k--', lw=0.8)
ax.set_xlabel('Target ν')
ax.set_ylabel('Achieved ν (rest_lengths)')
ax.set_title('Per-topology: achieved ν (rest_lengths)')
ax.legend(fontsize=6, ncol=2)
ax.grid(True, alpha=0.3)

fig_a.suptitle(f'Sweep 3: Performance Summary\n'
               f'({len(POISSON_TARGETS)} targets × {N_TOPOLOGIES} topologies: '
               f'crystal, foam, Poisson)',
               fontsize=13, y=1.02)
fig_a.tight_layout()
fig_a.savefig(os.path.join(OUT, 'summary_performance.png'), dpi=150, bbox_inches='tight')
plt.close(fig_a)
print("  Saved summary_performance.png")

# Figure B: Convergence heatmap
fig_b, axs = plt.subplots(1, 3, figsize=(20, 6))
topo_labels = [t['name'] for t in topologies]

for col, dv in enumerate(DESIGN_VARS):
    converged = np.zeros((N_TOPOLOGIES, len(POISSON_TARGETS)))
    for i, nu in enumerate(POISSON_TARGETS):
        for j in range(N_TOPOLOGIES):
            r = all_results[nu][j][dv]
            if r is not None and r['final_loss'] < 1e-6:
                converged[j, i] = 1.0
            elif r is not None and r['final_loss'] < SUCCESS_THRESHOLD:
                converged[j, i] = 0.5

    im = axs[col].imshow(converged, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
    axs[col].set_xticks(range(len(POISSON_TARGETS)))
    axs[col].set_xticklabels([f'{t:+.1f}' for t in POISSON_TARGETS], fontsize=7, rotation=45)
    axs[col].set_yticks(range(N_TOPOLOGIES))
    axs[col].set_yticklabels(topo_labels, fontsize=7)
    axs[col].set_xlabel('Target ν')
    axs[col].set_title(f'{dv_labels[dv]}', fontsize=11)

fig_b.colorbar(im, ax=axs.tolist(),
               label='Convergence (green=converged, yellow=partial, red=failed)',
               shrink=0.8)
fig_b.suptitle('Convergence Heatmap by Topology Type', fontsize=13, y=1.03)
fig_b.tight_layout()
fig_b.savefig(os.path.join(OUT, 'summary_convergence.png'), dpi=150, bbox_inches='tight')
plt.close(fig_b)
print("  Saved summary_convergence.png")

# Figure C: Loss heatmap (log scale)
fig_c, axs = plt.subplots(1, 3, figsize=(20, 6))
for col, dv in enumerate(DESIGN_VARS):
    loss_grid = np.full((N_TOPOLOGIES, len(POISSON_TARGETS)), np.nan)
    for i, nu in enumerate(POISSON_TARGETS):
        for j in range(N_TOPOLOGIES):
            r = all_results[nu][j][dv]
            if r is not None:
                loss_grid[j, i] = np.log10(max(r['final_loss'], 1e-17))

    im = axs[col].imshow(loss_grid, aspect='auto', cmap='RdYlGn_r', vmin=-16, vmax=0)
    axs[col].set_xticks(range(len(POISSON_TARGETS)))
    axs[col].set_xticklabels([f'{t:+.1f}' for t in POISSON_TARGETS], fontsize=7, rotation=45)
    axs[col].set_yticks(range(N_TOPOLOGIES))
    axs[col].set_yticklabels(topo_labels, fontsize=7)
    axs[col].set_xlabel('Target ν')
    axs[col].set_title(f'{dv_labels[dv]}', fontsize=11)

fig_c.colorbar(im, ax=axs.tolist(), label='log10(loss)', shrink=0.8)
fig_c.suptitle('Loss Heatmap (log10) by Topology Type', fontsize=13, y=1.03)
fig_c.tight_layout()
fig_c.savefig(os.path.join(OUT, 'summary_loss_heatmap.png'), dpi=150, bbox_inches='tight')
plt.close(fig_c)
print("  Saved summary_loss_heatmap.png")

print("\n" + "=" * 70)
print("ALL DONE")
print("=" * 70)
