#!/usr/bin/env python
"""Full parameter sweep: Poisson-ratio targets × network topologies × design variables.

Generates per-case visualizations and summary plots.
"""

import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'Phase 2'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'Phase 3'))
os.environ['MPLBACKEND'] = 'Agg'

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import TwoSlopeNorm, LogNorm
import matplotlib.cm as cm

import Disc_2_Cont_optimized as D2C
from forward_solver_torch import from_triangulation
from inverse_optimize import run_property_optimization

# ── Configuration ──────────────────────────────────────────────────────────
POISSON_TARGETS = [-0.9, -0.7, -0.5, -0.3, -0.1, 0.0, 0.1, 0.3, 0.5, 0.7, 0.9]
N_TOPOLOGIES = 5
TOPO_SEEDS = [42, 137, 256, 314, 999]
DESIGN_VARS = ['rigidities', 'rest_lengths', 'both']
OPT_SEEDS = [7, 144, 281]  # 3 random restarts per case for robustness
MESH_SIZE = (10, 10)
ETA = 0.2
MAX_ITER = 2000
LR = 0.05
TOL = 1e-14
OUT = os.path.dirname(os.path.abspath(__file__))

# ── Generate the 5 fixed topologies ───────────────────────────────────────
print("=" * 70)
print("Generating 5 fixed network topologies ...")
print("=" * 70)

topologies = []
for i, seed in enumerate(TOPO_SEEDS):
    np.random.seed(seed)
    tri = D2C.generate_foam_points(size=MESH_SIZE, eta=ETA)
    solver, default_rigs, _ = from_triangulation(tri)
    n_tri = len(tri.simplices)
    actual_rl = solver.actual_length2.sqrt().numpy()
    with torch.no_grad():
        gt = solver(default_rigs)
    info = {
        'tri': tri,
        'solver': solver,
        'n_tri': n_tri,
        'default_rigs': default_rigs,
        'actual_rl': actual_rl,
        'natural_poisson': gt['poisson'].item(),
        'natural_young': gt['young'].item(),
        'seed': seed,
    }
    topologies.append(info)
    print(f"  Topo {i}: seed={seed}, {n_tri} triangles, "
          f"natural ν={info['natural_poisson']:.4f}, E={info['natural_young']:.4f}")

# ── Run the sweep ─────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print(f"Running sweep: {len(POISSON_TARGETS)} targets × {N_TOPOLOGIES} topos × "
      f"{len(DESIGN_VARS)} design vars × {len(OPT_SEEDS)} restarts")
print("=" * 70)

# results[target_nu][topo_idx][dv] = best result dict
all_results = {}
total = len(POISSON_TARGETS) * N_TOPOLOGIES * len(DESIGN_VARS)
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
                    print(f"  FAILED: target={target_nu}, topo={topo_idx}, "
                          f"dv={dv}, seed={opt_seed}: {e}")
            all_results[target_nu][topo_idx][dv] = best
            done += 1
            elapsed = time.time() - t_start
            eta_s = (elapsed / done) * (total - done) if done > 0 else 0
            if best is not None:
                print(f"  [{done:3d}/{total}] ν*={target_nu:+.1f}  topo={topo_idx}  "
                      f"dv={dv:13s}  → ν={best['poisson']:+.6f}  "
                      f"E={best['young']:.4e}  loss={best['final_loss']:.2e}  "
                      f"[ETA {eta_s:.0f}s]")
            else:
                print(f"  [{done:3d}/{total}] ν*={target_nu:+.1f}  topo={topo_idx}  "
                      f"dv={dv:13s}  → FAILED ALL SEEDS  [ETA {eta_s:.0f}s]")

print(f"\nSweep complete in {time.time() - t_start:.1f}s")

# ── Save raw results as JSON ──────────────────────────────────────────────
summary_data = {}
for target_nu in POISSON_TARGETS:
    key = f"{target_nu:+.1f}"
    summary_data[key] = {}
    for topo_idx in range(N_TOPOLOGIES):
        summary_data[key][str(topo_idx)] = {}
        for dv in DESIGN_VARS:
            r = all_results[target_nu][topo_idx][dv]
            if r is not None:
                summary_data[key][str(topo_idx)][dv] = {
                    'poisson': r['poisson'],
                    'young': r['young'],
                    'final_loss': r['final_loss'],
                    'converged': r['converged'],
                    'iterations': r['iterations'],
                    'time_seconds': r['time_seconds'],
                }

with open(os.path.join(OUT, 'sweep_results.json'), 'w') as f:
    json.dump(summary_data, f, indent=2)
print("Saved sweep_results.json")


# ══════════════════════════════════════════════════════════════════════════
# Per-case visualizations
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Generating per-case visualizations ...")
print("=" * 70)

def plot_mesh(ax, tri, result, actual_rl, mode='rigidities', norm=None, cmap='viridis'):
    """Draw edge-colored mesh on an axes."""
    points = tri.points
    simplices = tri.simplices

    if mode == 'rigidities':
        vals_per_tri = result['rigidities']
    elif mode == 'rest_lengths':
        vals_per_tri = result['rest_lengths'] / actual_rl
    elif mode == 'both_k':
        vals_per_tri = result['rigidities']
    elif mode == 'both_rl':
        vals_per_tri = result['rest_lengths'] / actual_rl

    segments, colors = [], []
    for ti, sv in enumerate(simplices):
        for e_idx, (i, j) in enumerate([(sv[0], sv[1]), (sv[0], sv[2]), (sv[1], sv[2])]):
            segments.append([points[i], points[j]])
            colors.append(vals_per_tri[ti, e_idx])

    colors = np.array(colors)
    lc = LineCollection(segments, cmap=cmap, norm=norm, linewidths=0.6)
    lc.set_array(colors)
    ax.add_collection(lc)
    ax.set_xlim(points[:, 0].min() - 0.5, points[:, 0].max() + 0.5)
    ax.set_ylim(points[:, 1].min() - 0.5, points[:, 1].max() + 0.5)
    ax.set_aspect('equal')
    return lc


for target_nu in POISSON_TARGETS:
    tag = f"nu{target_nu:+.1f}".replace('+', 'p').replace('-', 'm').replace('.', '')
    subdir = os.path.join(OUT, tag)
    os.makedirs(subdir, exist_ok=True)

    for topo_idx, topo in enumerate(topologies):
        tri_obj = topo['tri']
        actual_rl = topo['actual_rl']

        # ── Mesh plots (one figure per topology, columns = design vars) ──
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
            rmin = max(rigs.ravel().min(), 1e-6)
            rmax = max(rigs.ravel().max(), rmin * 10)
            k_norm = LogNorm(vmin=rmin, vmax=rmax)
            lc = plot_mesh(axes[0, col], tri_obj, r, actual_rl,
                           mode='rigidities', norm=k_norm, cmap='viridis')
            axes[0, col].set_title(f'{dv}\nν={r["poisson"]:+.4f}  E={r["young"]:.3e}',
                                   fontsize=9)
            fig.colorbar(cm.ScalarMappable(norm=k_norm, cmap='viridis'),
                         ax=axes[0, col], label='k', shrink=0.7, pad=0.02)

            # Row 1: rest-length ratio
            rl_ratio = r['rest_lengths'] / actual_rl
            rl_min = max(rl_ratio.min(), 0.1)
            rl_max = max(rl_ratio.max(), 1.5)
            rl_norm = TwoSlopeNorm(vmin=min(rl_min, 0.3), vcenter=1.0,
                                   vmax=max(rl_max, 3.0))
            lc2 = plot_mesh(axes[1, col], tri_obj, r, actual_rl,
                            mode='rest_lengths', norm=rl_norm, cmap='coolwarm')
            axes[1, col].set_title(f'l₀/l_actual', fontsize=9)
            fig.colorbar(cm.ScalarMappable(norm=rl_norm, cmap='coolwarm'),
                         ax=axes[1, col], label='l₀/l_actual', shrink=0.7, pad=0.02)

        fig.suptitle(f'Target ν = {target_nu:+.1f}  |  Topology {topo_idx} '
                     f'(seed={topo["seed"]}, natural ν={topo["natural_poisson"]:.3f})',
                     fontsize=13, y=1.01)
        axes[0, 0].set_ylabel('Rigidities (log)', fontsize=10)
        axes[1, 0].set_ylabel('Rest-length ratio', fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(subdir, f'mesh_topo{topo_idx}.png'),
                    dpi=120, bbox_inches='tight')
        plt.close(fig)

        # ── Histogram + Polar (combined) ──
        fig2, axes2 = plt.subplots(2, 3, figsize=(18, 9),
                                    subplot_kw={'projection': None})
        # Top row: histograms of rest-length ratio
        # Bottom row: polar plots
        for col, dv in enumerate(DESIGN_VARS):
            r = all_results[target_nu][topo_idx][dv]
            if r is None:
                continue

            # Histogram
            ax_h = axes2[0, col]
            rl_ratio = (r['rest_lengths'] / actual_rl).ravel()
            rigs_flat = r['rigidities'].ravel()
            ax_h.hist(rl_ratio, bins=50, alpha=0.6, color='steelblue',
                      edgecolor='none', density=True, label='l₀/l_actual')
            ax_h.axvline(1.0, color='k', ls='--', lw=0.8)
            ax_h.set_xlabel('l₀ / l_actual')
            ax_h.set_ylabel('Density')
            ax_h.set_title(f'{dv}', fontsize=9)
            ax_h.set_xlim(0, max(4, np.percentile(rl_ratio, 99) * 1.1))

            ax_h2 = ax_h.twinx()
            ax_h2.hist(rigs_flat, bins=50, alpha=0.4, color='orange',
                       edgecolor='none', density=True, label='k')
            ax_h2.set_ylabel('Density (k)', color='orange')
            ax_h2.tick_params(axis='y', labelcolor='orange')

            lines1, labels1 = ax_h.get_legend_handles_labels()
            lines2, labels2 = ax_h2.get_legend_handles_labels()
            ax_h.legend(lines1 + lines2, labels1 + labels2, fontsize=7)

        # Bottom row: polar plots of rl ratio vs angle
        for col, dv in enumerate(DESIGN_VARS):
            # Remove the rectangular axes and add polar
            axes2[1, col].remove()
            ax_p = fig2.add_subplot(2, 3, 4 + col, projection='polar')
            r = all_results[target_nu][topo_idx][dv]
            if r is None:
                continue

            rl_ratio = r['rest_lengths'] / actual_rl
            angles, ratios_flat = [], []
            points = tri_obj.points
            for ti, sv in enumerate(tri_obj.simplices):
                for e_idx, (a, b) in enumerate([(sv[0], sv[1]), (sv[0], sv[2]),
                                                 (sv[1], sv[2])]):
                    dx = points[b, 0] - points[a, 0]
                    dy = points[b, 1] - points[a, 1]
                    angles.append(np.arctan2(dy, dx))
                    ratios_flat.append(rl_ratio[ti, e_idx])
            angles = np.array(angles)
            ratios_flat = np.array(ratios_flat)

            rl_norm = TwoSlopeNorm(vmin=min(ratios_flat.min(), 0.3), vcenter=1.0,
                                   vmax=max(ratios_flat.max(), 3.0))
            ax_p.scatter(angles, ratios_flat, c=ratios_flat, cmap='coolwarm',
                         norm=rl_norm, s=1, alpha=0.3)
            ax_p.set_ylim(0, min(5, np.percentile(ratios_flat, 99) * 1.2))
            ax_p.set_title(f'{dv}', fontsize=9, pad=12)

        fig2.suptitle(f'Target ν = {target_nu:+.1f}  |  Topology {topo_idx}\n'
                      f'Top: histograms (blue=l₀/l_actual, orange=k)  '
                      f'Bottom: polar (l₀/l_actual vs edge angle)',
                      fontsize=11, y=1.03)
        fig2.tight_layout()
        fig2.savefig(os.path.join(subdir, f'hist_polar_topo{topo_idx}.png'),
                     dpi=120, bbox_inches='tight')
        plt.close(fig2)

    print(f"  ν*={target_nu:+.1f}: saved mesh/hist/polar for {N_TOPOLOGIES} topos")


# ══════════════════════════════════════════════════════════════════════════
# Summary plots
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Generating summary plots ...")
print("=" * 70)

# Collect arrays: [target_nu][dv] -> list of values across topologies
targets = np.array(POISSON_TARGETS)

def collect(key):
    """Return dict[dv] -> (n_targets, n_topos) array."""
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
iters_arr = collect('iterations')
time_arr = collect('time_seconds')

# Rest-length and rigidity statistics
rl_median = {dv: np.full((len(POISSON_TARGETS), N_TOPOLOGIES), np.nan) for dv in DESIGN_VARS}
rl_std = {dv: np.full((len(POISSON_TARGETS), N_TOPOLOGIES), np.nan) for dv in DESIGN_VARS}
rl_pct_compressed = {dv: np.full((len(POISSON_TARGETS), N_TOPOLOGIES), np.nan) for dv in DESIGN_VARS}
rl_pct_stretched = {dv: np.full((len(POISSON_TARGETS), N_TOPOLOGIES), np.nan) for dv in DESIGN_VARS}
k_median = {dv: np.full((len(POISSON_TARGETS), N_TOPOLOGIES), np.nan) for dv in DESIGN_VARS}
k_std = {dv: np.full((len(POISSON_TARGETS), N_TOPOLOGIES), np.nan) for dv in DESIGN_VARS}
k_pct_dead = {dv: np.full((len(POISSON_TARGETS), N_TOPOLOGIES), np.nan) for dv in DESIGN_VARS}

for i, nu in enumerate(POISSON_TARGETS):
    for j in range(N_TOPOLOGIES):
        actual_rl = topologies[j]['actual_rl']
        for dv in DESIGN_VARS:
            r = all_results[nu][j][dv]
            if r is None:
                continue
            rl_ratio = r['rest_lengths'] / actual_rl
            rl_median[dv][i, j] = np.median(rl_ratio)
            rl_std[dv][i, j] = np.std(rl_ratio)
            rl_pct_compressed[dv][i, j] = (rl_ratio < 0.8).mean() * 100
            rl_pct_stretched[dv][i, j] = (rl_ratio > 1.2).mean() * 100
            rigs = r['rigidities']
            k_median[dv][i, j] = np.median(rigs)
            k_std[dv][i, j] = np.std(rigs)
            k_pct_dead[dv][i, j] = (rigs < 0.01).mean() * 100

dv_colors = {'rigidities': 'C0', 'rest_lengths': 'C1', 'both': 'C2'}
dv_labels = {'rigidities': 'Rigidities', 'rest_lengths': 'Rest lengths', 'both': 'Both'}

def plot_mean_std(ax, x, data_dict, ylabel, title):
    for dv in DESIGN_VARS:
        arr = data_dict[dv]
        mean = np.nanmean(arr, axis=1)
        std = np.nanstd(arr, axis=1)
        ax.plot(x, mean, 'o-', color=dv_colors[dv], label=dv_labels[dv], markersize=4)
        ax.fill_between(x, mean - std, mean + std, alpha=0.15, color=dv_colors[dv])
    ax.set_xlabel('Target Poisson ratio')
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

# ── Figure A: Achieved ν, E, loss, iterations ────────────────────────────
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

plot_mean_std(axs[1, 1], targets, iters_arr, 'Iterations',
              'Iterations to converge vs target ν')

fig_a.suptitle('Optimization Performance Summary\n'
               f'({len(POISSON_TARGETS)} targets × {N_TOPOLOGIES} topologies, '
               f'mean ± std across topologies)', fontsize=13, y=1.02)
fig_a.tight_layout()
fig_a.savefig(os.path.join(OUT, 'summary_performance.png'), dpi=150, bbox_inches='tight')
plt.close(fig_a)
print("  Saved summary_performance.png")

# ── Figure B: Rest-length statistics ──────────────────────────────────────
fig_b, axs = plt.subplots(2, 2, figsize=(14, 10))

plot_mean_std(axs[0, 0], targets, rl_median, 'Median l₀/l_actual',
              'Median rest-length ratio vs target ν')
axs[0, 0].axhline(1.0, color='k', ls=':', lw=0.8)

plot_mean_std(axs[0, 1], targets, rl_std, 'Std(l₀/l_actual)',
              'Spread of rest-length ratios vs target ν')

plot_mean_std(axs[1, 0], targets, rl_pct_compressed, '% edges with l₀/l < 0.8',
              'Fraction of compressed edges vs target ν')

plot_mean_std(axs[1, 1], targets, rl_pct_stretched, '% edges with l₀/l > 1.2',
              'Fraction of stretched edges vs target ν')

fig_b.suptitle('Rest-Length Statistics vs Target Poisson Ratio\n'
               f'(mean ± std across {N_TOPOLOGIES} topologies)',
               fontsize=13, y=1.02)
fig_b.tight_layout()
fig_b.savefig(os.path.join(OUT, 'summary_rest_lengths.png'), dpi=150, bbox_inches='tight')
plt.close(fig_b)
print("  Saved summary_rest_lengths.png")

# ── Figure C: Rigidity statistics ─────────────────────────────────────────
fig_c, axs = plt.subplots(1, 3, figsize=(18, 5))

plot_mean_std(axs[0], targets, k_median, 'Median k',
              'Median rigidity vs target ν')

plot_mean_std(axs[1], targets, k_std, 'Std(k)',
              'Spread of rigidities vs target ν')

plot_mean_std(axs[2], targets, k_pct_dead, '% edges with k < 0.01',
              'Fraction of dead springs vs target ν')

fig_c.suptitle('Rigidity Statistics vs Target Poisson Ratio\n'
               f'(mean ± std across {N_TOPOLOGIES} topologies)',
               fontsize=13, y=1.04)
fig_c.tight_layout()
fig_c.savefig(os.path.join(OUT, 'summary_rigidities.png'), dpi=150, bbox_inches='tight')
plt.close(fig_c)
print("  Saved summary_rigidities.png")

# ── Figure D: Per-topology comparison ─────────────────────────────────────
fig_d, axs = plt.subplots(2, 3, figsize=(18, 10))

for j in range(N_TOPOLOGIES):
    for col, dv in enumerate(DESIGN_VARS):
        axs[0, col].plot(targets, achieved_nu[dv][:, j], 'o-', markersize=3,
                         label=f'Topo {j} (ν₀={topologies[j]["natural_poisson"]:.3f})',
                         alpha=0.7)
        axs[1, col].plot(targets, achieved_E[dv][:, j], 'o-', markersize=3,
                         label=f'Topo {j}', alpha=0.7)

for col, dv in enumerate(DESIGN_VARS):
    axs[0, col].plot([-1, 1], [-1, 1], 'k--', lw=0.8)
    axs[0, col].set_title(f'{dv_labels[dv]}', fontsize=11)
    axs[0, col].set_xlabel('Target ν')
    axs[0, col].set_ylabel('Achieved ν')
    axs[0, col].legend(fontsize=6)
    axs[0, col].grid(True, alpha=0.3)

    axs[1, col].set_title(f'{dv_labels[dv]}', fontsize=11)
    axs[1, col].set_xlabel('Target ν')
    axs[1, col].set_ylabel("Young's modulus E")
    axs[1, col].set_yscale('log')
    axs[1, col].legend(fontsize=6)
    axs[1, col].grid(True, alpha=0.3)

fig_d.suptitle('Per-Topology Achieved Properties\n'
               f'Top: achieved ν  |  Bottom: Young\'s modulus',
               fontsize=13, y=1.02)
fig_d.tight_layout()
fig_d.savefig(os.path.join(OUT, 'summary_per_topology.png'), dpi=150, bbox_inches='tight')
plt.close(fig_d)
print("  Saved summary_per_topology.png")

# ── Figure E: Convergence rate heatmap ────────────────────────────────────
fig_e, axs = plt.subplots(1, 3, figsize=(18, 5))

for col, dv in enumerate(DESIGN_VARS):
    converged = np.zeros((len(POISSON_TARGETS), N_TOPOLOGIES))
    for i, nu in enumerate(POISSON_TARGETS):
        for j in range(N_TOPOLOGIES):
            r = all_results[nu][j][dv]
            if r is not None and r['final_loss'] < 1e-6:
                converged[i, j] = 1.0
            elif r is not None and r['final_loss'] < 1e-3:
                converged[i, j] = 0.5

    im = axs[col].imshow(converged.T, aspect='auto', cmap='RdYlGn',
                          vmin=0, vmax=1,
                          extent=[targets[0] - 0.1, targets[-1] + 0.1,
                                  N_TOPOLOGIES - 0.5, -0.5])
    axs[col].set_xticks(targets)
    axs[col].set_xticklabels([f'{t:+.1f}' for t in targets], fontsize=7, rotation=45)
    axs[col].set_yticks(range(N_TOPOLOGIES))
    axs[col].set_yticklabels([f'Topo {j}' for j in range(N_TOPOLOGIES)], fontsize=8)
    axs[col].set_xlabel('Target ν')
    axs[col].set_title(f'{dv_labels[dv]}', fontsize=11)

fig_e.colorbar(im, ax=axs.tolist(), label='Convergence (green=converged, yellow=partial, red=failed)',
               shrink=0.8)
fig_e.suptitle('Convergence Heatmap', fontsize=13, y=1.03)
fig_e.tight_layout()
fig_e.savefig(os.path.join(OUT, 'summary_convergence.png'), dpi=150, bbox_inches='tight')
plt.close(fig_e)
print("  Saved summary_convergence.png")

print("\n" + "=" * 70)
print("ALL DONE")
print("=" * 70)
