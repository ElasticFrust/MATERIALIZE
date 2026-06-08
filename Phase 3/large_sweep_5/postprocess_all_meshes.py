#!/usr/bin/env python
"""Post-processing: generate mesh/polar/histogram plots for ALL results,
including non-converged cases (crystals etc).

For cases where isotropic convergence failed:
  - Show the best isotropic attempt anyway
  - Also run a NON-isotropic optimisation (weight_isotropy=0) to show what
    the topology CAN achieve when freed from isotropy constraints.
  - Show both side-by-side in the output.

Usage:
    python postprocess_all_meshes.py           # uses this folder
    python postprocess_all_meshes.py /path/to  # uses given folder
"""

import sys, os, json, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'Phase 2'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'Phase 3'))
os.environ['MPLBACKEND'] = 'Agg'
warnings.filterwarnings('ignore', category=UserWarning)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from sweep_utils import (
    generate_all_topologies, run_optimisation, run_case,
    plot_mesh, plot_plain_mesh, get_edge_angles, compute_residual_energy,
    TOPO_CLASSES,
)

# ── Configuration ─────────────────────────────────────────────────────────
POISSON_TARGETS = [-0.9, -0.7, -0.5, -0.3, -0.1, 0.0, 0.1, 0.3, 0.5, 0.7, 0.9]
DESIGN_VARS     = ['rigidities', 'rest_lengths', 'both']
SUCCESS_THRESH  = 0.01
N_RESTARTS_ANISO = 5
INIT_WIDTHS     = [0.5, 1.5, 2.5]

# ── Determine paths ───────────────────────────────────────────────────────
if len(sys.argv) > 1:
    OUT = os.path.abspath(sys.argv[1])
else:
    OUT = os.path.dirname(os.path.abspath(__file__))

json_path = os.path.join(OUT, 'sweep_results.json')
assert os.path.exists(json_path), f"No sweep_results.json in {OUT}"

with open(json_path) as f:
    results = json.load(f)

# Detect mesh size from the run_sweep script
MESH_SIZE = (10, 10)
for fn in ['run_sweep_5.py', 'run_sweep_6.py', 'run_sweep_7.py']:
    spath = os.path.join(OUT, fn)
    if os.path.exists(spath):
        with open(spath) as f:
            for line in f:
                if 'MESH_SIZE' in line and '=' in line:
                    try:
                        val = line.split('=', 1)[1].strip()
                        MESH_SIZE = eval(val)
                        print(f"Detected MESH_SIZE = {MESH_SIZE} from {fn}")
                    except:
                        pass
                    break
        break
else:
    print(f"Using default MESH_SIZE = {MESH_SIZE}")


def run_case_aniso(solver, n_tri, target_nu, dv):
    """Run non-isotropic optimisation (just target nu, no isotropy penalty)."""
    return run_case(solver, n_tri, target_nu, dv,
                    n_restarts=N_RESTARTS_ANISO, init_widths=INIT_WIDTHS,
                    weight_isotropy=0.0)


# ── Generate topologies ─────────────────────────────────────────────────
print("=" * 70)
print("Generating topologies ...")
print("=" * 70)

topologies = generate_all_topologies(MESH_SIZE)
for t in topologies:
    print(f"  {t['name']:25s}  {t['n_tri']:4d} tri  class={t['topo_class']}")

topo_by_name = {t['name']: t for t in topologies}

# ── Output directory ──────────────────────────────────────────────────────
all_meshes_dir = os.path.join(OUT, 'all_meshes')
os.makedirs(all_meshes_dir, exist_ok=True)

# ── Process every target x topology ──────────────────────────────────────
print(f"\n{'='*70}")
print("Generating plots for ALL cases (including non-converged) ...")
print(f"{'='*70}")

aniso_results = {}

for target_nu in POISSON_TARGETS:
    key = f"{target_nu:+.1f}"
    if key not in results:
        print(f"  Skipping {key} (not in results)")
        continue

    if target_nu < 0:
        tag = f"num{abs(target_nu):.1f}".replace('.', '')
    elif target_nu == 0:
        tag = "nup00"
    else:
        tag = f"nup{target_nu:.1f}".replace('.', '')
    subdir = os.path.join(all_meshes_dir, tag)
    os.makedirs(subdir, exist_ok=True)

    for topo in topologies:
        tname = topo['name']
        if tname not in results[key]:
            continue

        # Find best DV (lowest isotropic loss)
        best_dv = None
        best_loss = float('inf')
        for dv in DESIGN_VARS:
            if dv not in results[key][tname]:
                continue
            rd = results[key][tname][dv]
            l = rd['final_loss']
            if isinstance(l, str):
                l = float(l) if l != 'Infinity' else float('inf')
            if l < best_loss:
                best_loss = l
                best_dv = dv

        if best_dv is None:
            continue

        rd_best = results[key][tname][best_dv]
        converged = rd_best.get('converged', False)

        # Re-run best isotropic case to get per-edge arrays
        iso_edge = None
        if rd_best.get('seed') is not None:
            iso_edge = run_optimisation(
                topo['solver'], topo['n_tri'], target_nu, best_dv,
                seed=rd_best['seed'],
                init_width=rd_best.get('init_width', 0.5),
                weight_isotropy=10.0)

        # For non-converged: also run non-isotropic
        aniso_edge = None
        aniso_dv = None
        if not converged:
            cache_key = f"{key}|{tname}"
            if cache_key not in aniso_results:
                print(f"    Running non-isotropic opt: nu*={target_nu:+.1f}  {tname:25s} ...",
                      end='', flush=True)
                best_aniso = None
                best_aniso_dv = None
                for adv in DESIGN_VARS:
                    r = run_case_aniso(topo['solver'], topo['n_tri'], target_nu, adv)
                    if r is not None and (best_aniso is None or r['final_loss'] < best_aniso['final_loss']):
                        best_aniso = r
                        best_aniso_dv = adv
                aniso_results[cache_key] = (best_aniso, best_aniso_dv)
                if best_aniso is not None:
                    print(f"  nu_xy={best_aniso['nu_xy']:+.4f}  loss={best_aniso['final_loss']:.2e}")
                else:
                    print("  FAILED")
            aniso_edge, aniso_dv = aniso_results[cache_key]

        has_aniso = aniso_edge is not None and not converged
        ncols = 2 if has_aniso else 1

        # ============================================================
        # MESH PLOT
        # ============================================================
        fig, axes = plt.subplots(1, ncols, figsize=(8*ncols, 6))
        if ncols == 1:
            axes = [axes]

        if iso_edge is not None:
            plot_mesh(topo['tri'], iso_edge['rigidities'], iso_edge['rest_lengths'],
                      topo['actual_rl'], best_dv, axes[0],
                      f'ISOTROPIC (w=10)\ndv={best_dv}\n'
                      f'nu_xy={iso_edge["nu_xy"]:+.4f}  nu_yx={iso_edge["nu_yx"]:+.4f}\n'
                      f'loss={iso_edge["final_loss"]:.2e}  {"CONVERGED" if converged else "NOT converged"}')
        else:
            plot_plain_mesh(topo['tri'], axes[0],
                            f'ISOTROPIC (w=10)\nNo valid optimisation result\nloss={best_loss:.2e}')

        if has_aniso:
            plot_mesh(topo['tri'], aniso_edge['rigidities'], aniso_edge['rest_lengths'],
                      topo['actual_rl'], aniso_dv, axes[1],
                      f'ANISOTROPIC (w=0, no isotropy constraint)\ndv={aniso_dv}\n'
                      f'nu_xy={aniso_edge["nu_xy"]:+.4f}  nu_yx={aniso_edge["nu_yx"]:+.4f}\n'
                      f'loss={aniso_edge["final_loss"]:.2e}')

        status = "OK" if converged else "FAIL"
        fig.suptitle(f'Target nu={target_nu:+.1f} | {tname} | [{status}]',
                     fontsize=12, y=1.02)
        fig.tight_layout()
        fig.savefig(os.path.join(subdir, f'mesh_{tname}.png'),
                    dpi=120, bbox_inches='tight')
        plt.close(fig)

        # ============================================================
        # POLAR + HISTOGRAM PLOT
        # ============================================================
        angles = get_edge_angles(topo['tri'])

        fig, axes = plt.subplots(2, ncols, figsize=(8*ncols, 10))
        if ncols == 1:
            axes = axes.reshape(2, 1)

        for col_idx, (edge_r, dv_label, w_label) in enumerate([
            (iso_edge, best_dv, 'Isotropic (w=10)'),
            (aniso_edge, aniso_dv if has_aniso else '', 'Anisotropic (w=0)'),
        ]):
            if col_idx >= ncols or edge_r is None:
                continue

            ax_h = axes[0, col_idx]
            rl_ratio = (edge_r['rest_lengths'] / topo['actual_rl']).ravel()
            rigs_flat = edge_r['rigidities'].ravel()
            ax_h.hist(rl_ratio, bins=50, alpha=0.6, color='steelblue',
                      edgecolor='none', density=True, label='l0/l_actual')
            ax_h.axvline(1.0, color='k', ls='--', lw=0.8)
            ax_h.set_xlabel('l0 / l_actual')
            ax_h.set_ylabel('Density')
            ax_h.set_title(f'{w_label} | dv={dv_label}\n'
                           f'nu_xy={edge_r["nu_xy"]:+.4f}  nu_yx={edge_r["nu_yx"]:+.4f}\n'
                           f'loss={edge_r["final_loss"]:.2e}', fontsize=8)
            ax_h.set_xlim(0, max(4, np.percentile(rl_ratio, 99)*1.1))
            ax_h2 = ax_h.twinx()
            ax_h2.hist(np.log10(np.clip(rigs_flat, 1e-10, None)), bins=50,
                       alpha=0.4, color='orange', edgecolor='none', density=True,
                       label='log10(k)')
            ax_h2.set_ylabel('Density (log10 k)', color='orange')
            ax_h2.tick_params(axis='y', labelcolor='orange')

            axes[1, col_idx].remove()
            ax_p = fig.add_subplot(2, ncols, ncols + col_idx + 1, projection='polar')
            ratios_flat = (edge_r['rest_lengths'] / topo['actual_rl']).ravel()
            angles_flat = angles.ravel()
            vmin_p = min(ratios_flat.min(), 0.3)
            vmax_p = max(ratios_flat.max(), 3.0)
            rl_norm = TwoSlopeNorm(vmin=vmin_p, vcenter=1.0, vmax=vmax_p)
            ax_p.scatter(angles_flat, ratios_flat, c=ratios_flat, cmap='coolwarm',
                         norm=rl_norm, s=1, alpha=0.3)
            ax_p.set_ylim(0, min(5, np.percentile(ratios_flat, 99)*1.2))
            ax_p.set_title(f'{w_label}', fontsize=9, pad=12)

        fig.suptitle(f'Target nu={target_nu:+.1f} | {tname}\n'
                     f'Top: histograms | Bottom: polar (l0/l_actual vs angle)',
                     fontsize=11, y=1.03)
        fig.tight_layout()
        fig.savefig(os.path.join(subdir, f'hist_polar_{tname}.png'),
                    dpi=100, bbox_inches='tight')
        plt.close(fig)

        # ============================================================
        # POLAR RIGIDITY MAP
        # ============================================================
        fig, axes_pr = plt.subplots(1, ncols, figsize=(8*ncols, 7),
                                    subplot_kw={'projection': 'polar'})
        if ncols == 1:
            axes_pr = [axes_pr]

        for col_idx, (edge_r, dv_label, w_label) in enumerate([
            (iso_edge, best_dv, 'Isotropic (w=10)'),
            (aniso_edge, aniso_dv if has_aniso else '', 'Anisotropic (w=0)'),
        ]):
            if col_idx >= ncols or edge_r is None:
                continue

            rigs_flat = edge_r['rigidities'].ravel()
            angles_flat = angles.ravel()
            log_k = np.log10(np.clip(rigs_flat, 1e-10, None))
            p5, p95 = np.percentile(log_k, [5, 95])
            if abs(p95 - p5) < 0.1:
                p5, p95 = p5 - 1, p95 + 1
            k_norm = plt.Normalize(vmin=p5, vmax=p95)

            ax_pr = axes_pr[col_idx]
            sc = ax_pr.scatter(angles_flat, rigs_flat, c=log_k, cmap='inferno',
                               norm=k_norm, s=1, alpha=0.3)
            ax_pr.set_ylim(0, min(np.percentile(rigs_flat, 99)*1.2,
                                  np.percentile(rigs_flat, 99)*2))
            ax_pr.set_title(f'{w_label} | dv={dv_label}\n'
                            f'nu_xy={edge_r["nu_xy"]:+.4f}', fontsize=9, pad=15)
            plt.colorbar(sc, ax=ax_pr, label='log10(k)', shrink=0.8, pad=0.1)

        fig.suptitle(f'Polar Rigidity Map: k vs edge angle\n'
                     f'Target nu={target_nu:+.1f} | {tname}', fontsize=12, y=1.05)
        fig.tight_layout()
        fig.savefig(os.path.join(subdir, f'polar_rigidity_{tname}.png'),
                    dpi=100, bbox_inches='tight')
        plt.close(fig)

        # ============================================================
        # RESIDUAL ENERGY
        # ============================================================
        from matplotlib.collections import LineCollection
        from matplotlib.colors import LogNorm
        import matplotlib.cm as cm

        fig, axes = plt.subplots(1, ncols, figsize=(8*ncols, 6))
        if ncols == 1:
            axes = [axes]

        for col_idx, (edge_r, dv_label, w_label) in enumerate([
            (iso_edge, best_dv, 'Isotropic (w=10)'),
            (aniso_edge, aniso_dv if has_aniso else '', 'Anisotropic (w=0)'),
        ]):
            if col_idx >= ncols or edge_r is None:
                continue

            re = compute_residual_energy(topo['solver'],
                                         edge_r['rigidities'], edge_r['rest_lengths'])
            re_flat = re.ravel()
            re_pos = re_flat[re_flat > 0]
            pts = topo['tri'].points
            simps = topo['tri'].simplices

            if len(re_pos) > 0 and re_pos.max() > 0:
                p1 = max(np.percentile(re_pos, 1), 1e-15)
                p99 = np.percentile(re_pos, 99)
                re_norm = LogNorm(vmin=p1, vmax=max(p99, p1*100), clip=True)
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
            lc_art = LineCollection(segments, cmap='hot', norm=re_norm, linewidths=lw)
            lc_art.set_array(colors)
            axes[col_idx].add_collection(lc_art)
            axes[col_idx].set_xlim(pts[:,0].min()-0.5, pts[:,0].max()+0.5)
            axes[col_idx].set_ylim(pts[:,1].min()-0.5, pts[:,1].max()+0.5)
            axes[col_idx].set_aspect('equal')
            total_E = np.sum(re)
            axes[col_idx].set_title(f'{w_label} | dv={dv_label}\nTotal E = {total_E:.3e}',
                                    fontsize=9)
            plt.colorbar(cm.ScalarMappable(norm=re_norm, cmap='hot'),
                         ax=axes[col_idx], label='Residual energy', shrink=0.7, pad=0.02)

        fig.suptitle(f'Residual Energy: 0.5*k*(l-l0)^2\n'
                     f'Target nu={target_nu:+.1f} | {tname}', fontsize=12, y=1.02)
        fig.tight_layout()
        fig.savefig(os.path.join(subdir, f'residual_{tname}.png'),
                    dpi=100, bbox_inches='tight')
        plt.close(fig)

    print(f"  {tag}: done ({len([t for t in topologies if t['name'] in results[key]])} topos)")


# ── Summary table ─────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("Summary: best isotropic vs best anisotropic for non-converged cases")
print(f"{'='*70}")
print(f"{'Topo':25s} {'Target':>7s} {'ISO nu_xy':>10s} {'ISO loss':>10s} "
      f"{'ANISO nu_xy':>12s} {'ANISO loss':>12s}")
print("-" * 80)

for target_nu in POISSON_TARGETS:
    key = f"{target_nu:+.1f}"
    if key not in results:
        continue
    for topo in topologies:
        tname = topo['name']
        if tname not in results[key]:
            continue
        best_loss = float('inf')
        best_nu = float('nan')
        for dv in DESIGN_VARS:
            if dv not in results[key][tname]:
                continue
            rd = results[key][tname][dv]
            l = rd['final_loss']
            if isinstance(l, str):
                l = float(l) if l != 'Infinity' else float('inf')
            if l < best_loss:
                best_loss = l
                best_nu = rd['nu_xy']
        converged = best_loss < SUCCESS_THRESH
        if not converged:
            cache_key = f"{key}|{tname}"
            if cache_key in aniso_results and aniso_results[cache_key][0] is not None:
                ar = aniso_results[cache_key][0]
                print(f"{tname:25s} {target_nu:+7.1f} {best_nu:+10.4f} {best_loss:10.2e} "
                      f"{ar['nu_xy']:+12.4f} {ar['final_loss']:12.2e}")
            else:
                print(f"{tname:25s} {target_nu:+7.1f} {best_nu:+10.4f} {best_loss:10.2e} "
                      f"{'N/A':>12s} {'N/A':>12s}")

print(f"\n{'='*70}")
print(f"All plots saved to: {all_meshes_dir}")
print(f"{'='*70}")
