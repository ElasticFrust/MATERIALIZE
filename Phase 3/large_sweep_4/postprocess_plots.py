#!/usr/bin/env python
"""Post-processing: generate polar plots, residual energy, and histograms
for large_sweep_4 results.

Run AFTER run_sweep_4.py completes. Reads sweep_results.json and regenerates
the topologies to access per-edge data.
"""

import sys, os, json
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
from matplotlib.colors import TwoSlopeNorm, LogNorm, Normalize
import matplotlib.cm as cm

import Disc_2_Cont_optimized as D2C
from forward_solver_torch import from_triangulation
from inverse_optimize import run_property_optimization, raw_to_rigidities

# ── Configuration ──────────────────────────────────────────────────────────
POISSON_TARGETS = [-0.9, -0.7, -0.5, -0.3, -0.1, 0.0, 0.1, 0.3, 0.5, 0.7, 0.9]
DESIGN_VARS = ['rigidities', 'rest_lengths', 'both']
OPT_SEEDS = [7, 144, 281]
MESH_SIZE = (10, 10)
MAX_ITER = 2000
LR = 0.05
TOL = 1e-14
ISOTROPIC = True
WEIGHT_ISOTROPY = 10.0
SUCCESS_THRESHOLD = 0.01
OUT = os.path.dirname(os.path.abspath(__file__))


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
    goods = ((np.abs(centroids[:, 0]) <= size[0]) &
             (np.abs(centroids[:, 1]) <= size[1]))
    DM.all_simplices = DM.simplices
    DM.simplices = DM.simplices[np.where(goods)]
    return DM


# ── Generate topologies ───────────────────────────────────────────────────
print("Regenerating topologies ...")
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
    tri = generate_poisson_network(size=MESH_SIZE)
    add_topo(f'poisson_{seed}', tri, seed)

N_TOPOLOGIES = len(topologies)
topo_name_to_idx = {t['name']: i for i, t in enumerate(topologies)}
print(f"  {N_TOPOLOGIES} topologies ready")


def get_edge_angles(tri_obj):
    """Compute edge angle for each (triangle, edge) pair."""
    points = tri_obj.points
    simplices = tri_obj.simplices
    angles = np.zeros((len(simplices), 3))
    for ti, sv in enumerate(simplices):
        for e_idx, (a, b) in enumerate([(sv[0], sv[1]), (sv[0], sv[2]), (sv[1], sv[2])]):
            dx = points[b, 0] - points[a, 0]
            dy = points[b, 1] - points[a, 1]
            angles[ti, e_idx] = np.arctan2(dy, dx)
    return angles


def compute_residual_energy(solver, rigs_np, rl_np):
    """Compute per-edge residual energy = 0.5 * k * (l - l0)^2."""
    # actual_length2 is the squared actual edge length from the geometry
    actual_l = solver.actual_length2.sqrt().numpy()
    return 0.5 * rigs_np * (actual_l - rl_np) ** 2


# ── Re-run optimization to get full parameter arrays ─────────────────────
# (JSON only stores scalars, not the per-edge arrays)
# We re-run each case with the same seeds to reproduce
print("\nRe-running optimizations to get per-edge data ...")
print("(Only cases that succeeded, loss < 0.1)")

json_path = os.path.join(OUT, 'sweep_results.json')
with open(json_path) as f:
    summary_data = json.load(f)

all_results = {}  # [target_nu][topo_idx][dv] -> full result dict

for target_nu in POISSON_TARGETS:
    key = f"{target_nu:+.1f}"
    all_results[target_nu] = {}
    for topo_idx, topo in enumerate(topologies):
        all_results[target_nu][topo_idx] = {}
        topo_name = topo['name']
        for dv in DESIGN_VARS:
            # Check if this case is worth re-running
            if (key in summary_data and topo_name in summary_data[key]
                    and dv in summary_data[key][topo_name]):
                r_json = summary_data[key][topo_name][dv]
                if r_json['final_loss'] < 0.1:  # only re-run decent cases
                    best = None
                    for opt_seed in OPT_SEEDS:
                        try:
                            r = run_property_optimization(
                                solver=topo['solver'],
                                n_triangles=topo['n_tri'],
                                target_poisson=target_nu,
                                weight_poisson=1.0,
                                design_variable=dv,
                                isotropic=ISOTROPIC,
                                weight_isotropy=WEIGHT_ISOTROPY,
                                max_iter=MAX_ITER,
                                lr=LR,
                                tol=TOL,
                                optimizer_type='lbfgs',
                                seed=opt_seed,
                                verbose=False,
                            )
                            if best is None or r['final_loss'] < best['final_loss']:
                                best = r
                        except Exception:
                            pass
                    all_results[target_nu][topo_idx][dv] = best
                    if best is not None:
                        print(f"  nu*={target_nu:+.1f} {topo_name:25s} {dv:13s} "
                              f"loss={best['final_loss']:.2e}")
                    continue
            all_results[target_nu][topo_idx][dv] = None


# ── Generate polar + histogram + residual energy plots ───────────────────
print("\n" + "=" * 70)
print("Generating polar, histogram, and residual energy plots ...")
print("=" * 70)

for target_nu in POISSON_TARGETS:
    if target_nu < 0:
        tag = f"num{abs(target_nu):.1f}".replace('.', '')
    elif target_nu == 0:
        tag = "nup00"
    else:
        tag = f"nup{target_nu:.1f}".replace('.', '')
    subdir = os.path.join(OUT, tag)
    os.makedirs(subdir, exist_ok=True)

    for topo_idx, topo in enumerate(topologies):
        tri_obj = topo['tri']
        actual_rl = topo['actual_rl']
        edge_angles = get_edge_angles(tri_obj)

        # Check if we have any data for this topology
        has_any = any(all_results[target_nu][topo_idx][dv] is not None
                      for dv in DESIGN_VARS)
        if not has_any:
            continue

        # ── Figure 1: Histogram + Polar (combined) ──
        fig2, axes2 = plt.subplots(2, 3, figsize=(18, 9),
                                    subplot_kw={'projection': None})

        for col, dv in enumerate(DESIGN_VARS):
            r = all_results[target_nu][topo_idx][dv]
            if r is None:
                axes2[0, col].text(0.5, 0.5, 'N/A', transform=axes2[0, col].transAxes,
                                   ha='center', va='center', fontsize=14, color='gray')
                continue

            # Top row: histograms
            ax_h = axes2[0, col]
            rl_ratio = (r['rest_lengths'] / actual_rl).ravel()
            rigs_flat = r['rigidities'].ravel()
            ax_h.hist(rl_ratio, bins=50, alpha=0.6, color='steelblue',
                      edgecolor='none', density=True, label='l_0/l_actual')
            ax_h.axvline(1.0, color='k', ls='--', lw=0.8)
            ax_h.set_xlabel('l_0 / l_actual')
            ax_h.set_ylabel('Density')
            ax_h.set_title(f'{dv}\nloss={r["final_loss"]:.2e}', fontsize=9)
            ax_h.set_xlim(0, max(4, np.percentile(rl_ratio, 99) * 1.1))

            ax_h2 = ax_h.twinx()
            ax_h2.hist(np.log10(np.clip(rigs_flat, 1e-10, None)), bins=50,
                       alpha=0.4, color='orange', edgecolor='none', density=True,
                       label='log10(k)')
            ax_h2.set_ylabel('Density (log10 k)', color='orange')
            ax_h2.tick_params(axis='y', labelcolor='orange')

            lines1, labels1 = ax_h.get_legend_handles_labels()
            lines2, labels2 = ax_h2.get_legend_handles_labels()
            ax_h.legend(lines1 + lines2, labels1 + labels2, fontsize=7)

        # Bottom row: polar plots
        for col, dv in enumerate(DESIGN_VARS):
            axes2[1, col].remove()
            ax_p = fig2.add_subplot(2, 3, 4 + col, projection='polar')
            r = all_results[target_nu][topo_idx][dv]
            if r is None:
                continue

            rl_ratio = r['rest_lengths'] / actual_rl
            angles_flat = edge_angles.ravel()
            ratios_flat = rl_ratio.ravel()

            rl_norm = TwoSlopeNorm(vmin=min(ratios_flat.min(), 0.3), vcenter=1.0,
                                   vmax=max(ratios_flat.max(), 3.0))
            ax_p.scatter(angles_flat, ratios_flat, c=ratios_flat, cmap='coolwarm',
                         norm=rl_norm, s=1, alpha=0.3)
            ax_p.set_ylim(0, min(5, np.percentile(ratios_flat, 99) * 1.2))
            ax_p.set_title(f'{dv}', fontsize=9, pad=12)

        fig2.suptitle(f'ISOTROPIC | Target nu = {target_nu:+.1f}  |  Topo {topo_idx}: {topo["name"]}\n'
                      f'Top: histograms (blue=l_0/l_actual, orange=log10(k))  '
                      f'Bottom: polar (l_0/l_actual vs edge angle)',
                      fontsize=11, y=1.03)
        fig2.tight_layout()
        fig2.savefig(os.path.join(subdir, f'hist_polar_topo{topo_idx}_{topo["name"]}.png'),
                     dpi=120, bbox_inches='tight')
        plt.close(fig2)

        # ── Figure 2: Residual energy mesh ──
        fig3, axes3 = plt.subplots(1, 3, figsize=(18, 5.5))
        points = tri_obj.points
        simplices = tri_obj.simplices

        for col, dv in enumerate(DESIGN_VARS):
            r = all_results[target_nu][topo_idx][dv]
            if r is None:
                axes3[col].text(0.5, 0.5, 'N/A', transform=axes3[col].transAxes,
                                ha='center', va='center', fontsize=14, color='gray')
                axes3[col].set_title(f'{dv}', fontsize=9)
                continue

            # Residual energy per edge
            res_energy = compute_residual_energy(topo['solver'],
                                                  r['rigidities'],
                                                  r['rest_lengths'])
            re_flat = res_energy.ravel()
            re_pos = re_flat[re_flat > 0]

            if len(re_pos) > 0 and re_pos.max() > 0:
                p1 = max(np.percentile(re_pos, 1), 1e-15)
                p99 = np.percentile(re_pos, 99)
                re_norm = LogNorm(vmin=p1, vmax=max(p99, p1 * 100), clip=True)
            else:
                re_norm = LogNorm(vmin=1e-15, vmax=1e-5, clip=True)

            segments, colors = [], []
            for ti, sv in enumerate(simplices):
                for e_idx, (a, b) in enumerate([(sv[0], sv[1]), (sv[0], sv[2]),
                                                 (sv[1], sv[2])]):
                    segments.append([points[a], points[b]])
                    colors.append(res_energy[ti, e_idx])
            colors = np.array(colors)

            # Linewidth proportional to energy
            lw_min, lw_max = 0.15, 3.0
            log_c = np.log10(np.clip(colors, re_norm.vmin, re_norm.vmax))
            log_range = np.log10(re_norm.vmax) - np.log10(re_norm.vmin)
            if log_range > 0:
                lw = lw_min + (lw_max - lw_min) * (log_c - np.log10(re_norm.vmin)) / log_range
            else:
                lw = np.full_like(colors, 0.8)

            lc = LineCollection(segments, cmap='hot', norm=re_norm, linewidths=lw)
            lc.set_array(colors)
            axes3[col].add_collection(lc)
            axes3[col].set_xlim(points[:, 0].min()-0.5, points[:, 0].max()+0.5)
            axes3[col].set_ylim(points[:, 1].min()-0.5, points[:, 1].max()+0.5)
            axes3[col].set_aspect('equal')
            total_E = np.sum(res_energy)
            axes3[col].set_title(f'{dv}\nTotal residual E = {total_E:.3e}', fontsize=9)
            fig3.colorbar(cm.ScalarMappable(norm=re_norm, cmap='hot'),
                         ax=axes3[col], label='Residual energy', shrink=0.7, pad=0.02)

        fig3.suptitle(f'ISOTROPIC | Residual Energy: 0.5*k*(l-l0)^2\n'
                      f'Target nu={target_nu:+.1f}  |  Topo {topo_idx}: {topo["name"]}',
                      fontsize=12, y=1.02)
        fig3.tight_layout()
        fig3.savefig(os.path.join(subdir, f'residual_energy_topo{topo_idx}_{topo["name"]}.png'),
                     dpi=120, bbox_inches='tight')
        plt.close(fig3)

    print(f"  {tag}: polar/histogram/residual plots done")

print("\n" + "=" * 70)
print("POST-PROCESSING DONE")
print("=" * 70)
