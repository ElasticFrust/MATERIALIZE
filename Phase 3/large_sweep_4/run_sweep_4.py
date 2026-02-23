#!/usr/bin/env python
"""Large sweep 4: ISOTROPIC constraint.

Same 8 topologies as sweep 3, but with isotropic=True enforced during
optimization. This penalizes anisotropy in the elastic tensor so that
the resulting material has C_xxxx=C_yyyy, C_xxxy=C_xyyy=0, and
C_xyxy=(C_xxxx-C_xxyy)/2.

Topology types:
  0) iso_crystal     — perfect isotropic triangular lattice
  1) aniso_crystal   — anisotropic crystal (shape=(1.5, 0.8), orientation=pi/6)
  2) foam_eta02_42   — Voronoi foam eta=0.2 (seed 42)
  3) foam_eta02_137  — Voronoi foam eta=0.2 (seed 137)
  4) foam_eta045_256 — Voronoi foam eta=0.45 (seed 256)
  5) foam_eta045_314 — Voronoi foam eta=0.45 (seed 314)
  6) poisson_999     — Poisson random network (seed 999)
  7) poisson_1337    — Poisson random network (seed 1337)

Total: 11 targets x 8 topos x 3 dvs = 264 cases
"""

import sys, os, json, time, glob
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
from inverse_optimize import run_property_optimization

# ── Configuration ──────────────────────────────────────────────────────────
POISSON_TARGETS = [-0.9, -0.7, -0.5, -0.3, -0.1, 0.0, 0.1, 0.3, 0.5, 0.7, 0.9]
DESIGN_VARS = ['rigidities', 'rest_lengths', 'both']
OPT_SEEDS = [7, 144, 281]       # 3 random restarts
MESH_SIZE = (10, 10)
MAX_ITER = 2000
LR = 0.05
TOL = 1e-14
ISOTROPIC = True                 # <-- KEY DIFFERENCE from sweep 3
WEIGHT_ISOTROPY = 10.0
SUCCESS_THRESHOLD = 0.01
OUT = os.path.dirname(os.path.abspath(__file__))


# ── Helper: Poisson random network ───────────────────────────────────────
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
          f"{n_tri:3d} triangles  natural nu={info['natural_poisson']:+.4f}")
    return info

# 0) Isotropic crystal
tri = D2C.generate_cryratl_points(size=MESH_SIZE, shape=(1, 1), orientation=0)
add_topo('iso_crystal', tri, 0)

# 1) Anisotropic crystal
tri = D2C.generate_cryratl_points(size=MESH_SIZE, shape=(1.5, 0.8), orientation=np.pi/6)
add_topo('aniso_crystal', tri, 0)

# 2-3) Foam eta=0.2
for seed in [42, 137]:
    np.random.seed(seed)
    tri = D2C.generate_foam_points(size=MESH_SIZE, eta=0.2)
    add_topo(f'foam_eta02_{seed}', tri, seed)

# 4-5) Foam eta=0.45
for seed in [256, 314]:
    np.random.seed(seed)
    tri = D2C.generate_foam_points(size=MESH_SIZE, eta=0.45)
    add_topo(f'foam_eta045_{seed}', tri, seed)

# 6-7) Poisson random network
for seed in [999, 1337]:
    np.random.seed(seed)
    tri = generate_poisson_network(size=MESH_SIZE, eta=0.3)
    add_topo(f'poisson_{seed}', tri, seed)

N_TOPOLOGIES = len(topologies)
print(f"\nTotal: {N_TOPOLOGIES} topologies")

# ── Run the sweep ─────────────────────────────────────────────────────────
print("\n" + "=" * 70)
total = len(POISSON_TARGETS) * N_TOPOLOGIES * len(DESIGN_VARS)
print(f"Running ISOTROPIC sweep: {len(POISSON_TARGETS)} targets x {N_TOPOLOGIES} topos x "
      f"{len(DESIGN_VARS)} dvs x {len(OPT_SEEDS)} restarts = {total} cases")
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
                except Exception as e:
                    print(f"  FAILED: nu*={target_nu}, topo={topo_idx}({topo['name']}), "
                          f"dv={dv}, seed={opt_seed}: {e}")
            all_results[target_nu][topo_idx][dv] = best
            done += 1
            elapsed = time.time() - t_start
            eta_s = (elapsed / done) * (total - done) if done > 0 else 0
            tag = "OK" if best is not None else "FAIL"
            if best is not None:
                print(f"  [{done:3d}/{total}] nu*={target_nu:+.1f}  "
                      f"topo={topo_idx}({topo['name']:25s})  dv={dv:13s}  "
                      f"nu_xy={best['poisson']:+.4f}  loss={best['final_loss']:.2e}  "
                      f"[{tag}]  [ETA {eta_s:.0f}s]")
            else:
                print(f"  [{done:3d}/{total}] nu*={target_nu:+.1f}  "
                      f"topo={topo_idx}({topo['name']:25s})  dv={dv:13s}  "
                      f"-> FAILED ALL SEEDS  [ETA {eta_s:.0f}s]")

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
                    'nu_xy': r.get('nu_xy', float('nan')),
                    'nu_yx': r.get('nu_yx', float('nan')),
                    'E_x': r.get('E_x', float('nan')),
                    'E_y': r.get('E_y', float('nan')),
                    'G_xy': r.get('G_xy', float('nan')),
                }

with open(os.path.join(OUT, 'sweep_results.json'), 'w') as f:
    json.dump(summary_data, f, indent=2)
print("Saved sweep_results.json")

# ── Create per-target folders with plots and success/fail markers ─────────
print("\n" + "=" * 70)
print("Generating per-target visualizations + success/fail markers ...")
print("=" * 70)

TOPO_DESCRIPTIONS = {
    'iso_crystal': 'Isotropic crystal -- regular triangular lattice, shape=(1,1), orientation=0',
    'aniso_crystal': 'Anisotropic crystal -- stretched lattice, shape=(1.5, 0.8), orientation=pi/6',
    'foam_eta02_42': 'Foam eta=0.2 (seed 42) -- Delaunay BEFORE perturbation',
    'foam_eta02_137': 'Foam eta=0.2 (seed 137) -- Delaunay BEFORE perturbation',
    'foam_eta045_256': 'Foam eta=0.45 (seed 256) -- Delaunay BEFORE perturbation',
    'foam_eta045_314': 'Foam eta=0.45 (seed 314) -- Delaunay BEFORE perturbation',
    'poisson_999': 'Poisson random network (seed 999) -- Delaunay AFTER perturbation, eta=0.3',
    'poisson_1337': 'Poisson random network (seed 1337) -- Delaunay AFTER perturbation, eta=0.3',
}

topo_names = [t['name'] for t in topologies]

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

    # Remove old markers
    for old_marker in glob.glob(os.path.join(subdir, 'aaa--*')):
        os.remove(old_marker)

    # Check success and build README data
    any_success = False
    n_converged = 0
    best_loss = float('inf')
    best_label = ""
    total_cases = 0

    for topo_idx in range(N_TOPOLOGIES):
        for dv in DESIGN_VARS:
            r = all_results[target_nu][topo_idx][dv]
            if r is not None:
                total_cases += 1
                if r['final_loss'] < SUCCESS_THRESHOLD:
                    any_success = True
                    n_converged += 1
                if r['final_loss'] < best_loss:
                    best_loss = r['final_loss']
                    best_label = f"{topologies[topo_idx]['name']} / {dv}"

    marker = 'aaa--success' if any_success else 'aaa--fail'
    open(os.path.join(subdir, marker), 'w').close()

    # ── README ──
    lines = []
    lines.append(f"# Target Poisson Ratio: nu = {target_nu} (ISOTROPIC)")
    lines.append("")
    lines.append("## Run Configuration")
    lines.append("")
    lines.append(f"- **Target nu**: {target_nu}")
    lines.append(f"- **Constraint**: ISOTROPIC (weight_isotropy={WEIGHT_ISOTROPY})")
    lines.append(f"- **Mesh size**: {MESH_SIZE}")
    lines.append(f"- **Design variables**: {', '.join(DESIGN_VARS)}")
    lines.append(f"- **Optimizer**: L-BFGS, lr={LR}, tol={TOL}, max_iter={MAX_ITER}")
    lines.append(f"- **Random restarts per case**: {len(OPT_SEEDS)} (seeds: {', '.join(map(str, OPT_SEEDS))})")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- **Total cases**: {total_cases} ({N_TOPOLOGIES} topologies x {len(DESIGN_VARS)} design variables)")
    lines.append(f"- **Converged (loss < {SUCCESS_THRESHOLD})**: {n_converged}/{total_cases}")
    lines.append(f"- **Best result**: {best_label} (loss = {best_loss:.2e})")
    lines.append("")
    lines.append("## Results by Topology")

    for topo_idx in range(N_TOPOLOGIES):
        topo_name = topologies[topo_idx]['name']
        desc = TOPO_DESCRIPTIONS.get(topo_name, topo_name)
        lines.append("")
        lines.append(f"### {topo_name}")
        lines.append(f"_{desc}_")
        lines.append("")
        lines.append("| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |")
        lines.append("|---|---|---|---|---|---|")

        for dv in DESIGN_VARS:
            r = all_results[target_nu][topo_idx][dv]
            if r is None:
                lines.append(f"| {dv} | N/A | N/A | No | N/A | N/A |")
                continue
            nu_val = r['poisson']
            nu_str = f"{nu_val:+f}" if nu_val == nu_val else "NaN"
            loss_str = f"{r['final_loss']:.2e}"
            conv_str = "Yes" if r['converged'] else "No"
            iter_str = str(r['iterations'])
            time_str = f"{r.get('time_seconds', 0):.1f}"
            lines.append(f"| {dv} | {nu_str} | {loss_str} | {conv_str} | {iter_str} | {time_str} |")

    readme_path = os.path.join(subdir, 'README.md')
    with open(readme_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    print(f"  {tag}: {marker} ({n_converged}/{total_cases} converged)")

    # ── Mesh plots per topology (improved contrast) ──
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

            # Row 0: rigidities — improved contrast
            rigs = r['rigidities']
            points = tri_obj.points
            simplices = tri_obj.simplices

            # Use percentile-based log normalization for better contrast
            rigs_flat = rigs.ravel()
            rigs_pos = rigs_flat[rigs_flat > 0]
            if len(rigs_pos) > 0:
                p2 = np.percentile(rigs_pos, 2)
                p98 = np.percentile(rigs_pos, 98)
                vmin = max(p2, 1e-8)
                vmax = max(p98, vmin * 10)
                # Ensure at least 2 orders of magnitude spread for readability
                if vmax / vmin < 100:
                    mid = np.sqrt(vmin * vmax)
                    vmin = mid / 50
                    vmax = mid * 50
            else:
                vmin, vmax = 1e-6, 1.0
            k_norm = LogNorm(vmin=vmin, vmax=vmax, clip=True)

            segments, colors = [], []
            for ti, sv in enumerate(simplices):
                for e_idx, (a, b) in enumerate([(sv[0], sv[1]), (sv[0], sv[2]), (sv[1], sv[2])]):
                    segments.append([points[a], points[b]])
                    colors.append(rigs[ti, e_idx])
            colors = np.array(colors)

            # Edge linewidth proportional to rigidity for extra contrast
            lw_min, lw_max = 0.15, 2.5
            log_colors = np.log10(np.clip(colors, vmin, vmax))
            log_range = np.log10(vmax) - np.log10(vmin)
            if log_range > 0:
                lw = lw_min + (lw_max - lw_min) * (log_colors - np.log10(vmin)) / log_range
            else:
                lw = np.full_like(colors, 0.8)

            lc = LineCollection(segments, cmap='inferno', norm=k_norm, linewidths=lw)
            lc.set_array(colors)
            axes[0, col].add_collection(lc)
            axes[0, col].set_xlim(points[:, 0].min()-0.5, points[:, 0].max()+0.5)
            axes[0, col].set_ylim(points[:, 1].min()-0.5, points[:, 1].max()+0.5)
            axes[0, col].set_aspect('equal')
            nu_disp = f"{r['poisson']:+.4f}" if r['poisson'] == r['poisson'] else "NaN"
            axes[0, col].set_title(f'{dv}\nnu={nu_disp}  E={r["young"]:.3e}\n'
                                   f'loss={r["final_loss"]:.2e}', fontsize=9)
            fig.colorbar(cm.ScalarMappable(norm=k_norm, cmap='inferno'),
                         ax=axes[0, col], label='k (rigidity)', shrink=0.7, pad=0.02)

            # Row 1: rest-length ratio — improved contrast
            rl_ratio = r['rest_lengths'] / actual_rl
            rl_flat = rl_ratio.ravel()
            rl_p2 = np.nanpercentile(rl_flat, 2)
            rl_p98 = np.nanpercentile(rl_flat, 98)
            rl_lo = min(rl_p2, 0.5)
            rl_hi = max(rl_p98, 2.0)
            rl_norm = TwoSlopeNorm(vmin=rl_lo, vcenter=1.0, vmax=rl_hi)

            segments2, colors2 = [], []
            for ti, sv in enumerate(simplices):
                for e_idx, (a, b) in enumerate([(sv[0], sv[1]), (sv[0], sv[2]), (sv[1], sv[2])]):
                    segments2.append([points[a], points[b]])
                    colors2.append(rl_ratio[ti, e_idx])
            colors2 = np.array(colors2)

            # Linewidth: thicker for more deviation from 1.0
            deviation = np.abs(colors2 - 1.0)
            dev_max = max(deviation.max(), 0.1)
            lw2 = 0.3 + 2.0 * (deviation / dev_max)

            lc2 = LineCollection(segments2, cmap='coolwarm', norm=rl_norm, linewidths=lw2)
            lc2.set_array(colors2)
            axes[1, col].add_collection(lc2)
            axes[1, col].set_xlim(points[:, 0].min()-0.5, points[:, 0].max()+0.5)
            axes[1, col].set_ylim(points[:, 1].min()-0.5, points[:, 1].max()+0.5)
            axes[1, col].set_aspect('equal')
            axes[1, col].set_title('l_0 / l_actual', fontsize=9)
            fig.colorbar(cm.ScalarMappable(norm=rl_norm, cmap='coolwarm'),
                         ax=axes[1, col], label='l_0 / l_actual', shrink=0.7, pad=0.02)

        fig.suptitle(f'ISOTROPIC | Target nu={target_nu:+.1f}  |  Topo {topo_idx}: {topo["name"]} '
                     f'(seed={topo["seed"]}, natural nu={topo["natural_poisson"]:.3f})',
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
                    val = r[key]
                    if val == val:  # not NaN
                        arr[i, j] = val
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
    ax.set_xlabel('Target nu')
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
ax.set_xlabel('Target nu')
ax.set_ylabel('Achieved nu')
ax.set_title('Achieved vs Target Poisson ratio (ISOTROPIC)')
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

plot_mean_std(axs[0, 1], targets, achieved_E, "Young's modulus E",
              "Young's modulus vs target nu")
axs[0, 1].set_yscale('log')

plot_mean_std(axs[1, 0], targets, loss_arr, 'Final loss',
              'Optimization loss vs target nu')
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
ax.set_xlabel('Target nu')
ax.set_ylabel('Achieved nu (rest_lengths)')
ax.set_title('Per-topology: achieved nu (rest_lengths, ISOTROPIC)')
ax.legend(fontsize=6, ncol=2)
ax.grid(True, alpha=0.3)

fig_a.suptitle(f'Sweep 4 (ISOTROPIC): Performance Summary\n'
               f'({len(POISSON_TARGETS)} targets x {N_TOPOLOGIES} topologies: '
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
    axs[col].set_xlabel('Target nu')
    axs[col].set_title(f'{dv_labels[dv]}', fontsize=11)

fig_b.colorbar(im, ax=axs.tolist(),
               label='Convergence (green=converged, yellow=partial, red=failed)',
               shrink=0.8)
fig_b.suptitle('Convergence Heatmap (ISOTROPIC) by Topology Type', fontsize=13, y=1.03)
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
    axs[col].set_xlabel('Target nu')
    axs[col].set_title(f'{dv_labels[dv]}', fontsize=11)

fig_c.colorbar(im, ax=axs.tolist(), label='log10(loss)', shrink=0.8)
fig_c.suptitle('Loss Heatmap log10 (ISOTROPIC) by Topology Type', fontsize=13, y=1.03)
fig_c.tight_layout()
fig_c.savefig(os.path.join(OUT, 'summary_loss_heatmap.png'), dpi=150, bbox_inches='tight')
plt.close(fig_c)
print("  Saved summary_loss_heatmap.png")

print("\n" + "=" * 70)
print("ALL DONE")
print("=" * 70)
