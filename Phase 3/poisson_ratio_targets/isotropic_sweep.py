#!/usr/bin/env python
"""Isotropic sweep: same as the original sweep but with isotropy enforced.

Compares isotropic=True vs isotropic=False for all 3 design variables.
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

OUT = os.path.dirname(os.path.abspath(__file__))

# ── Configuration ─────────────────────────────────────────────────────────
POISSON_TARGETS = [-0.9, -0.7, -0.5, -0.3, -0.1, 0.0, 0.1, 0.3, 0.5, 0.7, 0.9]
N_TOPOLOGIES = 5
TOPO_SEEDS = [42, 137, 256, 314, 999]
DESIGN_VARS = ['rigidities', 'rest_lengths', 'both']
OPT_SEEDS = [7, 144, 281]  # 3 restarts
MESH_SIZE = (10, 10)
ETA = 0.2

# ── Generate topologies ───────────────────────────────────────────────────
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
    topologies.append({
        'tri': tri, 'solver': solver, 'n_tri': n_tri,
        'actual_rl': actual_rl, 'seed': seed,
    })
    print(f"  Topo {i}: seed={seed}, {n_tri} triangles")

# ── Run isotropic sweep ──────────────────────────────────────────────────
print("\n" + "=" * 70)
print("Running ISOTROPIC sweep")
print("=" * 70)

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
                        solver=topo['solver'], n_triangles=topo['n_tri'],
                        target_poisson=target_nu, weight_poisson=1.0,
                        design_variable=dv,
                        isotropic=True, weight_isotropy=10.0,
                        max_iter=1000, lr=0.05, tol=1e-14,
                        optimizer_type='lbfgs', seed=opt_seed, verbose=False,
                    )
                    if best is None or r['final_loss'] < best['final_loss']:
                        best = r
                except Exception as e:
                    print(f"  FAILED: ν*={target_nu}, topo={topo_idx}, dv={dv}, "
                          f"seed={opt_seed}: {e}")
            all_results[target_nu][topo_idx][dv] = best
            done += 1
            elapsed = time.time() - t_start
            eta_s = (elapsed / done) * (total - done) if done > 0 else 0
            if best is not None:
                print(f"  [{done:3d}/{total}] ν*={target_nu:+.1f}  topo={topo_idx}  "
                      f"dv={dv:13s}  → ν_xy={best['nu_xy']:+.4f}  ν_yx={best['nu_yx']:+.4f}  "
                      f"E_x={best['E_x']:.3e}  E_y={best['E_y']:.3e}  "
                      f"loss={best['final_loss']:.2e}  [ETA {eta_s:.0f}s]")

print(f"\nSweep complete in {time.time() - t_start:.1f}s")

# ── Save raw results ──────────────────────────────────────────────────────
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
                    'nu_xy': r['nu_xy'], 'nu_yx': r['nu_yx'],
                    'E_x': r['E_x'], 'E_y': r['E_y'],
                    'G_xy': r['G_xy'],
                    'poisson': r['poisson'], 'young': r['young'],
                    'final_loss': r['final_loss'],
                    'converged': r['converged'],
                    'iterations': r['iterations'],
                }

with open(os.path.join(OUT, 'isotropic_sweep_results.json'), 'w') as f:
    json.dump(summary_data, f, indent=2)
print("Saved isotropic_sweep_results.json")

# ══════════════════════════════════════════════════════════════════════════
# Per-case visualizations (mesh + hist/polar for each target × topo)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Generating per-case visualizations ...")
print("=" * 70)

for target_nu in POISSON_TARGETS:
    tag = f"iso_nu{target_nu:+.1f}".replace('+', 'p').replace('-', 'm').replace('.', '')
    subdir = os.path.join(OUT, tag)
    os.makedirs(subdir, exist_ok=True)

    for topo_idx, topo in enumerate(topologies):
        tri_obj = topo['tri']
        actual_rl = topo['actual_rl']
        points = tri_obj.points
        simplices = tri_obj.simplices

        # ── Mesh plots ────────────────────────────────────────────────
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
            segments, colors = [], []
            for ti, sv in enumerate(simplices):
                for e_idx, (a, b) in enumerate([(sv[0], sv[1]), (sv[0], sv[2]), (sv[1], sv[2])]):
                    segments.append([points[a], points[b]])
                    colors.append(rigs[ti, e_idx])
            lc = LineCollection(segments, cmap='viridis', norm=k_norm, linewidths=0.6)
            lc.set_array(np.array(colors))
            axes[0, col].add_collection(lc)
            axes[0, col].set_xlim(points[:, 0].min()-0.5, points[:, 0].max()+0.5)
            axes[0, col].set_ylim(points[:, 1].min()-0.5, points[:, 1].max()+0.5)
            axes[0, col].set_aspect('equal')
            axes[0, col].set_title(
                f'{dv}\n'
                f'ν_xy={r["nu_xy"]:+.4f}  ν_yx={r["nu_yx"]:+.4f}\n'
                f'E_x={r["E_x"]:.3e}  E_y={r["E_y"]:.3e}', fontsize=8)
            fig.colorbar(cm.ScalarMappable(norm=k_norm, cmap='viridis'),
                         ax=axes[0, col], label='k', shrink=0.7, pad=0.02)

            # Row 1: rest-length ratio
            rl_ratio = r['rest_lengths'] / actual_rl
            rl_norm = TwoSlopeNorm(vmin=min(rl_ratio.min(), 0.3), vcenter=1.0,
                                   vmax=max(rl_ratio.max(), 3.0))
            segments2, colors2 = [], []
            for ti, sv in enumerate(simplices):
                for e_idx, (a, b) in enumerate([(sv[0], sv[1]), (sv[0], sv[2]), (sv[1], sv[2])]):
                    segments2.append([points[a], points[b]])
                    colors2.append(rl_ratio[ti, e_idx])
            lc2 = LineCollection(segments2, cmap='coolwarm', norm=rl_norm, linewidths=0.6)
            lc2.set_array(np.array(colors2))
            axes[1, col].add_collection(lc2)
            axes[1, col].set_xlim(points[:, 0].min()-0.5, points[:, 0].max()+0.5)
            axes[1, col].set_ylim(points[:, 1].min()-0.5, points[:, 1].max()+0.5)
            axes[1, col].set_aspect('equal')
            axes[1, col].set_title('l₀/l_actual', fontsize=9)
            fig.colorbar(cm.ScalarMappable(norm=rl_norm, cmap='coolwarm'),
                         ax=axes[1, col], label='l₀/l_actual', shrink=0.7, pad=0.02)

        fig.suptitle(f'ISOTROPIC  |  Target ν = {target_nu:+.1f}  |  Topology {topo_idx}',
                     fontsize=13, y=1.01)
        axes[0, 0].set_ylabel('Rigidities (log)', fontsize=10)
        axes[1, 0].set_ylabel('Rest-length ratio', fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(subdir, f'mesh_topo{topo_idx}.png'),
                    dpi=120, bbox_inches='tight')
        plt.close(fig)

        # ── Histogram + Polar ─────────────────────────────────────────
        fig2, axes2 = plt.subplots(2, 3, figsize=(18, 9))
        for col, dv in enumerate(DESIGN_VARS):
            r = all_results[target_nu][topo_idx][dv]
            if r is None:
                continue

            # Histogram
            rl_ratio = (r['rest_lengths'] / actual_rl).ravel()
            rigs_flat = r['rigidities'].ravel()
            ax_h = axes2[0, col]
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
            lines1, labels1 = ax_h.get_legend_handles_labels()
            lines2, labels2 = ax_h2.get_legend_handles_labels()
            ax_h.legend(lines1 + lines2, labels1 + labels2, fontsize=7)

        # Polar row
        for col, dv in enumerate(DESIGN_VARS):
            axes2[1, col].remove()
            ax_p = fig2.add_subplot(2, 3, 4 + col, projection='polar')
            r = all_results[target_nu][topo_idx][dv]
            if r is None:
                continue
            rl_ratio = r['rest_lengths'] / actual_rl
            angles, ratios_flat = [], []
            for ti, sv in enumerate(simplices):
                for e_idx, (a, b) in enumerate([(sv[0], sv[1]), (sv[0], sv[2]), (sv[1], sv[2])]):
                    dx = points[b, 0] - points[a, 0]
                    dy = points[b, 1] - points[a, 1]
                    angles.append(np.arctan2(dy, dx))
                    ratios_flat.append(rl_ratio[ti, e_idx])
            angles = np.array(angles)
            ratios_flat = np.array(ratios_flat)
            rl_norm_p = TwoSlopeNorm(vmin=min(ratios_flat.min(), 0.3), vcenter=1.0,
                                     vmax=max(ratios_flat.max(), 3.0))
            ax_p.scatter(angles, ratios_flat, c=ratios_flat, cmap='coolwarm',
                         norm=rl_norm_p, s=1, alpha=0.3)
            ax_p.set_ylim(0, min(5, np.percentile(ratios_flat, 99) * 1.2))
            ax_p.set_title(f'{dv}', fontsize=9, pad=12)

        fig2.suptitle(f'ISOTROPIC  |  Target ν = {target_nu:+.1f}  |  Topology {topo_idx}',
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

targets = np.array(POISSON_TARGETS)
dv_colors = {'rigidities': 'C0', 'rest_lengths': 'C1', 'both': 'C2'}
dv_labels = {'rigidities': 'Rigidities', 'rest_lengths': 'Rest lengths', 'both': 'Both'}


def collect_field(field):
    out = {}
    for dv in DESIGN_VARS:
        arr = np.full((len(POISSON_TARGETS), N_TOPOLOGIES), np.nan)
        for i, nu in enumerate(POISSON_TARGETS):
            for j in range(N_TOPOLOGIES):
                r = all_results[nu][j][dv]
                if r is not None and field in r:
                    arr[i, j] = r[field]
        out[dv] = arr
    return out


nu_xy = collect_field('nu_xy')
nu_yx = collect_field('nu_yx')
E_x = collect_field('E_x')
E_y = collect_field('E_y')
loss_arr = collect_field('final_loss')


def plot_mean_std(ax, x, data, ylabel, title):
    for dv in DESIGN_VARS:
        m = np.nanmean(data[dv], axis=1)
        s = np.nanstd(data[dv], axis=1)
        ax.plot(x, m, 'o-', color=dv_colors[dv], label=dv_labels[dv], markersize=4)
        ax.fill_between(x, m - s, m + s, alpha=0.15, color=dv_colors[dv])
    ax.set_xlabel('Target ν')
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)


# ── Figure 1: Isotropy quality ───────────────────────────────────────────
fig1, axes = plt.subplots(2, 3, figsize=(18, 11))

# (0,0) ν_xy vs ν_yx scatter
ax = axes[0, 0]
for dv in DESIGN_VARS:
    ax.scatter(nu_xy[dv].ravel(), nu_yx[dv].ravel(),
               c=dv_colors[dv], label=dv_labels[dv], s=15, alpha=0.6)
ax.plot([-1, 1], [-1, 1], 'k--', lw=0.8, label='Isotropic')
ax.set_xlabel(r'$\nu_{xy}$')
ax.set_ylabel(r'$\nu_{yx}$')
ax.set_title(r'$\nu_{xy}$ vs $\nu_{yx}$ — isotropy check')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

# (0,1) ν_xy and ν_yx vs target
ax = axes[0, 1]
for dv in DESIGN_VARS:
    m_xy = np.nanmean(nu_xy[dv], axis=1)
    m_yx = np.nanmean(nu_yx[dv], axis=1)
    s_xy = np.nanstd(nu_xy[dv], axis=1)
    s_yx = np.nanstd(nu_yx[dv], axis=1)
    ax.plot(targets, m_xy, 'o-', color=dv_colors[dv], markersize=4,
            label=f'{dv_labels[dv]} ν_xy')
    ax.plot(targets, m_yx, 's--', color=dv_colors[dv], markersize=4,
            label=f'{dv_labels[dv]} ν_yx', alpha=0.6)
    ax.fill_between(targets, m_xy - s_xy, m_xy + s_xy, alpha=0.08, color=dv_colors[dv])
    ax.fill_between(targets, m_yx - s_yx, m_yx + s_yx, alpha=0.08, color=dv_colors[dv])
ax.plot([-1, 1], [-1, 1], 'k:', lw=0.8, label='Perfect')
ax.set_xlabel('Target ν')
ax.set_ylabel('Achieved ν')
ax.set_title(r'$\nu_{xy}$ (solid) vs $\nu_{yx}$ (dashed)')
ax.legend(fontsize=5.5, ncol=2)
ax.grid(True, alpha=0.3)

# (0,2) |ν_xy - ν_yx| gap
ax = axes[0, 2]
for dv in DESIGN_VARS:
    gap = np.abs(nu_xy[dv] - nu_yx[dv])
    m = np.nanmean(gap, axis=1)
    s = np.nanstd(gap, axis=1)
    ax.plot(targets, m, 'o-', color=dv_colors[dv], label=dv_labels[dv], markersize=4)
    ax.fill_between(targets, np.maximum(m - s, 1e-16), m + s, alpha=0.15, color=dv_colors[dv])
ax.set_xlabel('Target ν')
ax.set_ylabel(r'$|\nu_{xy} - \nu_{yx}|$')
ax.set_title('Poisson anisotropy gap (should be ~0)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_yscale('log')

# (1,0) E_x vs E_y scatter
ax = axes[1, 0]
for dv in DESIGN_VARS:
    ax.scatter(np.abs(E_x[dv].ravel()), np.abs(E_y[dv].ravel()),
               c=dv_colors[dv], label=dv_labels[dv], s=15, alpha=0.6)
ax.plot([1e-12, 1], [1e-12, 1], 'k--', lw=0.8)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'$E_x$')
ax.set_ylabel(r'$E_y$')
ax.set_title(r'$E_x$ vs $E_y$ (should be on diagonal)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# (1,1) E_x/E_y ratio
ax = axes[1, 1]
for dv in DESIGN_VARS:
    ratio = np.abs(E_x[dv]) / np.maximum(np.abs(E_y[dv]), 1e-30)
    m = np.nanmean(ratio, axis=1)
    s = np.nanstd(ratio, axis=1)
    ax.plot(targets, m, 'o-', color=dv_colors[dv], label=dv_labels[dv], markersize=4)
    ax.fill_between(targets, np.maximum(m - s, 0.01), m + s, alpha=0.15, color=dv_colors[dv])
ax.axhline(1.0, color='k', ls='--', lw=0.8)
ax.set_xlabel('Target ν')
ax.set_ylabel(r'$E_x / E_y$')
ax.set_title('Stiffness ratio (= 1 for isotropic)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# (1,2) loss
ax = axes[1, 2]
plot_mean_std(ax, targets, loss_arr, 'Final loss', 'Optimization loss')
ax.set_yscale('log')

fig1.suptitle('ISOTROPIC Optimization: Anisotropy Quality Check\n'
              '(isotropy=True, weight_isotropy=10)',
              fontsize=14, y=1.02)
fig1.tight_layout()
fig1.savefig(os.path.join(OUT, 'isotropic_summary_anisotropy.png'),
             dpi=150, bbox_inches='tight')
plt.close(fig1)
print("  Saved isotropic_summary_anisotropy.png")

# ── Figure 2: Performance comparison ─────────────────────────────────────
fig2, axes = plt.subplots(2, 2, figsize=(14, 10))

# Achieved ν (use mean of nu_xy and nu_yx as the isotropic ν)
ax = axes[0, 0]
for dv in DESIGN_VARS:
    nu_mean = (nu_xy[dv] + nu_yx[dv]) / 2
    m = np.nanmean(nu_mean, axis=1)
    s = np.nanstd(nu_mean, axis=1)
    ax.plot(targets, m, 'o-', color=dv_colors[dv], label=dv_labels[dv], markersize=4)
    ax.fill_between(targets, m - s, m + s, alpha=0.15, color=dv_colors[dv])
ax.plot([-1, 1], [-1, 1], 'k--', lw=0.8, label='Perfect')
ax.set_xlabel('Target ν')
ax.set_ylabel('Achieved ν (avg of ν_xy, ν_yx)')
ax.set_title('Achieved Poisson ratio')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Young's modulus
ax = axes[0, 1]
for dv in DESIGN_VARS:
    E_mean = (np.abs(E_x[dv]) + np.abs(E_y[dv])) / 2
    m = np.nanmean(E_mean, axis=1)
    s = np.nanstd(E_mean, axis=1)
    ax.plot(targets, m, 'o-', color=dv_colors[dv], label=dv_labels[dv], markersize=4)
    ax.fill_between(targets, np.maximum(m - s, 1e-12), m + s, alpha=0.15, color=dv_colors[dv])
ax.set_xlabel('Target ν')
ax.set_ylabel("Young's modulus (avg E_x, E_y)")
ax.set_title("Young's modulus vs target ν")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_yscale('log')

# Rest-length stats
ax = axes[1, 0]
for dv in DESIGN_VARS:
    arr = np.full((len(POISSON_TARGETS), N_TOPOLOGIES), np.nan)
    for i, nu in enumerate(POISSON_TARGETS):
        for j in range(N_TOPOLOGIES):
            r = all_results[nu][j][dv]
            if r is not None:
                rl_ratio = r['rest_lengths'] / topologies[j]['actual_rl']
                arr[i, j] = np.median(rl_ratio)
    m = np.nanmean(arr, axis=1)
    s = np.nanstd(arr, axis=1)
    ax.plot(targets, m, 'o-', color=dv_colors[dv], label=dv_labels[dv], markersize=4)
    ax.fill_between(targets, m - s, m + s, alpha=0.15, color=dv_colors[dv])
ax.axhline(1.0, color='k', ls=':', lw=0.8)
ax.set_xlabel('Target ν')
ax.set_ylabel('Median l₀/l_actual')
ax.set_title('Median rest-length ratio')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Off-diagonal C components
ax = axes[1, 1]
for dv in DESIGN_VARS:
    c1 = np.full((len(POISSON_TARGETS), N_TOPOLOGIES), np.nan)
    c4 = np.full((len(POISSON_TARGETS), N_TOPOLOGIES), np.nan)
    for i, nu in enumerate(POISSON_TARGETS):
        for j in range(N_TOPOLOGIES):
            r = all_results[nu][j][dv]
            if r is not None:
                c1[i, j] = abs(r['pred_tensor'][1])
                c4[i, j] = abs(r['pred_tensor'][4])
    m1 = np.nanmean(c1, axis=1)
    m4 = np.nanmean(c4, axis=1)
    ax.plot(targets, m1, 'o-', color=dv_colors[dv], markersize=4,
            label=f'{dv_labels[dv]} |C_xxxy|')
    ax.plot(targets, m4, 's--', color=dv_colors[dv], markersize=4,
            label=f'{dv_labels[dv]} |C_xyyy|', alpha=0.6)
ax.set_xlabel('Target ν')
ax.set_ylabel('|Off-diagonal C|')
ax.set_title('Off-diagonal components (should be ~0)')
ax.legend(fontsize=6, ncol=2)
ax.grid(True, alpha=0.3)
ax.set_yscale('log')

fig2.suptitle('ISOTROPIC Optimization: Performance Summary\n'
              f'({len(POISSON_TARGETS)} targets × {N_TOPOLOGIES} topologies)',
              fontsize=13, y=1.02)
fig2.tight_layout()
fig2.savefig(os.path.join(OUT, 'isotropic_summary_performance.png'),
             dpi=150, bbox_inches='tight')
plt.close(fig2)
print("  Saved isotropic_summary_performance.png")

print("\n" + "=" * 70)
print("ALL DONE")
print("=" * 70)
