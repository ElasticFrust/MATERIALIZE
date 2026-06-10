#!/usr/bin/env python
"""Large sweep 6 — Robust isotropic Poisson-ratio targeting (15x15 meshes).

Uses shared utilities from sweep_utils.py.

Topology classes:
  crystal   — iso_crystal, aniso_crystal
  foam_02   — foam_eta02_{42,137}
  foam_045  — foam_eta045_{256,314}
  poisson   — poisson_{999,1337}

Total: 11 targets x 8 topos x 3 DVs = 264 cases
"""

import sys, os, json, time, glob, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'Phase 2'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'Phase 3'))
os.environ['MPLBACKEND'] = 'Agg'
warnings.filterwarnings('ignore', category=UserWarning)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LogNorm
import matplotlib.cm as cm

from sweep_utils import (
    generate_all_topologies, run_optimisation, run_case,
    plot_mesh, plot_plain_mesh, get_edge_angles, compute_residual_energy,
    TOPO_CLASSES,
)

# ── Configuration ─────────────────────────────────────────────────────────
POISSON_TARGETS = [-0.9, -0.7, -0.5, -0.3, -0.1, 0.0, 0.1, 0.3, 0.5, 0.7, 0.9]
DESIGN_VARS     = ['rigidities', 'rest_lengths', 'both']
MESH_SIZE       = (15, 15)
MAX_ITER        = 1000
LR              = 0.05
SUCCESS_THRESH  = 0.01
N_RESTARTS      = 5
INIT_WIDTHS     = [0.5, 1.5, 2.5]
OUT             = os.path.dirname(os.path.abspath(__file__))

# ── Generate topologies ──────────────────────────────────────────────────
print("=" * 70)
print("Generating topologies ...")
print("=" * 70)

topologies = generate_all_topologies(MESH_SIZE)
N_TOPOS = len(topologies)

for t in topologies:
    print(f"  {t['name']:25s}  {t['n_tri']:4d} tri  natural nu={t['natural_poisson']:+.4f}"
          f"  class={t['topo_class']}")
print(f"\nTotal: {N_TOPOS} topologies")

# ── Main optimisation loop (with checkpointing) ─────────────────────────
N_CASES = len(POISSON_TARGETS) * N_TOPOS * len(DESIGN_VARS)
json_path = os.path.join(OUT, 'sweep_results.json')

if os.path.exists(json_path):
    with open(json_path) as f:
        results = json.load(f)
    n_existing = sum(1 for k in results for tn in results[k] for dv in results[k][tn])
    print(f"\nResuming from checkpoint: {n_existing} cases already done")
else:
    results = {}

print(f"\n{'='*70}")
print(f"Running ISOTROPIC sweep: {len(POISSON_TARGETS)} targets x {N_TOPOS} topos"
      f" x {len(DESIGN_VARS)} dvs = {N_CASES} cases")
print(f"  restarts per case = {N_RESTARTS}, init widths = {INIT_WIDTHS}")
print(f"{'='*70}")

t_start = time.time()
case_num = 0
n_skipped = 0

for target_nu in POISSON_TARGETS:
    key = f"{target_nu:+.1f}"
    if key not in results:
        results[key] = {}
    for topo in topologies:
        if topo['name'] not in results[key]:
            results[key][topo['name']] = {}
        for dv in DESIGN_VARS:
            case_num += 1

            if dv in results[key][topo['name']]:
                n_skipped += 1
                continue

            n_done = case_num - n_skipped
            elapsed_total = time.time() - t_start
            if n_done > 1:
                eta_s = elapsed_total / (n_done - 1) * (N_CASES - case_num)
            else:
                eta_s = 0

            r = run_case(topo['solver'], topo['n_tri'], target_nu, dv,
                         n_restarts=N_RESTARTS, init_widths=INIT_WIDTHS,
                         max_iter=MAX_ITER, lr=LR)

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
                  f"{tag}  [ETA {eta_s:.0f}s]", flush=True)

            with open(json_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)

total_time = time.time() - t_start
print(f"\nSweep complete in {total_time:.1f}s (skipped {n_skipped} cached)")
print(f"Saved {json_path}")


# ── Per-target visualisations ─────────────────────────────────────────────
print(f"\n{'='*70}")
print("Generating per-target visualisations + markers ...")
print(f"{'='*70}")

edge_data = {}

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

    n_conv = sum(1 for tn in results[key] for dv in results[key][tn]
                 if results[key][tn][dv].get('converged', False))
    n_total = N_TOPOS * len(DESIGN_VARS)

    for old in glob.glob(os.path.join(subdir, 'aaa--*')):
        os.remove(old)
    marker = 'aaa--success' if n_conv > 0 else 'aaa--fail'
    open(os.path.join(subdir, marker), 'w').close()

    # Re-run best cases to get per-edge arrays
    for topo in topologies:
        edge_data[key][topo['name']] = {}
        for dv in DESIGN_VARS:
            rd = results[key][topo['name']][dv]
            if rd['final_loss'] < 0.1 and rd.get('seed') is not None:
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
            fig, ax = plt.subplots(1, 1, figsize=(6, 5.5))
            plot_plain_mesh(topo['tri'], ax,
                            f'No convergence (all loss > 0.1)\n{topo["name"]}')
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

            ax_h = axes[0, col]
            rl_ratio = (r['rest_lengths'] / actual_rl).ravel()
            rigs_flat = r['rigidities'].ravel()
            ax_h.hist(rl_ratio, bins=50, alpha=0.6, color='steelblue',
                      edgecolor='none', density=True, label='l0/l_actual')
            ax_h.axvline(1.0, color='k', ls='--', lw=0.8)
            ax_h.set_xlabel('l0 / l_actual')
            ax_h.set_ylabel('Density')
            ax_h.set_title(f'{dv}\nnu_xy={r["nu_xy"]:+.4f}  nu_yx={r["nu_yx"]:+.4f}\nloss={r["final_loss"]:.2e}', fontsize=8)
            ax_h.set_xlim(0, max(4, np.percentile(rl_ratio, 99) * 1.1))
            ax_h2 = ax_h.twinx()
            ax_h2.hist(np.log10(np.clip(rigs_flat, 1e-10, None)), bins=50,
                       alpha=0.4, color='orange', edgecolor='none', density=True, label='log10(k)')
            ax_h2.set_ylabel('Density (log10 k)', color='orange')
            ax_h2.tick_params(axis='y', labelcolor='orange')
            lines1, labels1 = ax_h.get_legend_handles_labels()
            lines2, labels2 = ax_h2.get_legend_handles_labels()
            ax_h.legend(lines1 + lines2, labels1 + labels2, fontsize=7)

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
                     f'Top: histograms  |  Bottom: polar (l0/l_actual vs angle)',
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

            from matplotlib.collections import LineCollection
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

        fig.suptitle(f'ISOTROPIC | Residual Energy: 0.5*k*(l-l0)^2\n'
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

    targets_f, success_rates, best_nu_xys = [], [], []
    for target_nu in POISSON_TARGETS:
        key = f"{target_nu:+.1f}"
        n_ok = n_tot = 0
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
        best_nu_xys.append(best_nu)

    ax.bar(targets_f, success_rates, width=0.15, color=class_colors[cls], alpha=0.7)
    ax.set_xlabel('Target nu')
    ax.set_ylabel('Success rate')
    ax.set_title(f'{cls} ({len(class_topos)} topologies)', fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.axhline(1.0, color='gray', ls='--', lw=0.5)
    ax2 = ax.twinx()
    ax2.plot(targets_f, best_nu_xys, 'ko-', markersize=4, label='Best nu_xy')
    ax2.plot(targets_f, targets_f, 'r--', lw=0.8, label='Target')
    ax2.set_ylabel('Best achieved nu_xy')
    ax2.legend(fontsize=7, loc='upper left')

fig_agg.suptitle('Success rate & best achieved nu by topology class (ISOTROPIC)', fontsize=13, y=1.02)
fig_agg.tight_layout()
fig_agg.savefig(os.path.join(OUT, 'aggregate_by_class.png'), dpi=120, bbox_inches='tight')
plt.close(fig_agg)
print("  Saved aggregate_by_class.png")

# 2) Loss heatmap per class
fig_heat, axes_heat = plt.subplots(2, 2, figsize=(14, 10))
for ci, cls in enumerate(classes):
    ax = axes_heat[ci // 2, ci % 2]
    class_topos = [t for t in topologies if t['topo_class'] == cls]
    row_labels, data_rows = [], []
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
    ax.set_xlabel('Target nu')
    ax.set_title(f'{cls}', fontsize=11)
    plt.colorbar(im, ax=ax, label='log10(loss)', shrink=0.8)

fig_heat.suptitle('Loss heatmap by topology class (ISOTROPIC)', fontsize=13, y=1.02)
fig_heat.tight_layout()
fig_heat.savefig(os.path.join(OUT, 'aggregate_loss_heatmap.png'), dpi=120, bbox_inches='tight')
plt.close(fig_heat)
print("  Saved aggregate_loss_heatmap.png")

# 3) Achieved nu_xy vs target — one line per class
fig_track, ax_track = plt.subplots(figsize=(8, 6))
for cls in classes:
    class_topos = [t for t in topologies if t['topo_class'] == cls]
    best_nus = []
    for target_nu in POISSON_TARGETS:
        key = f"{target_nu:+.1f}"
        best_nu, best_l = float('nan'), float('inf')
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
ax_track.set_xlabel('Target nu')
ax_track.set_ylabel('Best achieved nu_xy')
ax_track.set_title('Best achieved isotropic nu per topology class')
ax_track.legend()
ax_track.grid(True, alpha=0.3)
fig_track.tight_layout()
fig_track.savefig(os.path.join(OUT, 'aggregate_tracking.png'), dpi=120, bbox_inches='tight')
plt.close(fig_track)
print("  Saved aggregate_tracking.png")

# 4) Summary performance plot
fig_perf, ax_perf = plt.subplots(figsize=(12, 6))
markers = {'rigidities': 'o', 'rest_lengths': 's', 'both': '^'}
for topo in topologies:
    cls = topo['topo_class']
    for dv in DESIGN_VARS:
        achieved = []
        for target_nu in POISSON_TARGETS:
            key = f"{target_nu:+.1f}"
            achieved.append(results[key][topo['name']][dv]['nu_xy'])
        ax_perf.plot(POISSON_TARGETS, achieved, marker=markers[dv],
                     color=class_colors[cls], alpha=0.4, markersize=3, linewidth=0.5)
ax_perf.plot(POISSON_TARGETS, POISSON_TARGETS, 'k--', lw=2, label='Perfect')
for cls in classes:
    ax_perf.plot([], [], '-', color=class_colors[cls], label=cls, lw=2)
for dv in DESIGN_VARS:
    ax_perf.plot([], [], 'k', marker=markers[dv], label=dv, lw=0, markersize=5)
ax_perf.legend(ncol=2, fontsize=8)
ax_perf.set_xlabel('Target nu')
ax_perf.set_ylabel('Achieved nu_xy')
ax_perf.set_title('All topologies: achieved vs target nu (ISOTROPIC)')
ax_perf.grid(True, alpha=0.3)
fig_perf.tight_layout()
fig_perf.savefig(os.path.join(OUT, 'summary_all_topos.png'), dpi=120, bbox_inches='tight')
plt.close(fig_perf)
print("  Saved summary_all_topos.png")

print(f"\n{'='*70}")
print("ALL DONE")
print(f"{'='*70}")
