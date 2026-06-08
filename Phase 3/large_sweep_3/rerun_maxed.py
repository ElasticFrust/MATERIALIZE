#!/usr/bin/env python
"""Re-run only the cases from sweep 3 that hit the old MAX_ITER=500 limit.

Loads existing sweep_results.json, identifies cases with iterations >= 500,
re-runs them with MAX_ITER=2000, merges improved results back, and regenerates
READMEs + success/fail markers.
"""

import sys, os, json, time, glob
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'Phase 2'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'Phase 3'))
os.environ['MPLBACKEND'] = 'Agg'

import numpy as np
import scipy as sp
import torch

import Disc_2_Cont_optimized as D2C
from forward_solver_torch import from_triangulation
from inverse_optimize import run_property_optimization

# ── Configuration (must match run_sweep_3.py) ─────────────────────────────
POISSON_TARGETS = [-0.9, -0.7, -0.5, -0.3, -0.1, 0.0, 0.1, 0.3, 0.5, 0.7, 0.9]
DESIGN_VARS = ['rigidities', 'rest_lengths', 'both']
OPT_SEEDS = [7, 144, 281]
MESH_SIZE = (10, 10)
MAX_ITER = 2000
LR = 0.05
TOL = 1e-14
SUCCESS_THRESHOLD = 0.01
OLD_ITER_LIMIT = 500  # cases with iterations >= this are re-run
OUT = os.path.dirname(os.path.abspath(__file__))


# ── Helper: Poisson random network ───────────────────────────────────────
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


# ── Generate topologies (same seeds as run_sweep_3.py) ───────────────────
print("=" * 70)
print("Regenerating topologies ...")
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
    tri = generate_poisson_network(size=MESH_SIZE)
    add_topo(f'poisson_{seed}', tri, seed)

N_TOPOLOGIES = len(topologies)
topo_name_to_idx = {t['name']: i for i, t in enumerate(topologies)}

# ── Load existing results ────────────────────────────────────────────────
json_path = os.path.join(OUT, 'sweep_results.json')
with open(json_path) as f:
    summary_data = json.load(f)
print(f"\nLoaded {json_path}")

# ── Identify maxed-out cases ─────────────────────────────────────────────
rerun_cases = []
for target_nu in POISSON_TARGETS:
    key = f"{target_nu:+.1f}"
    if key not in summary_data:
        continue
    for topo_name, topo_results in summary_data[key].items():
        topo_idx = topo_name_to_idx.get(topo_name)
        if topo_idx is None:
            continue
        for dv in DESIGN_VARS:
            if dv not in topo_results:
                continue
            r = topo_results[dv]
            if r['iterations'] >= OLD_ITER_LIMIT:
                rerun_cases.append((target_nu, key, topo_name, topo_idx, dv))

print(f"\nFound {len(rerun_cases)} cases that hit the old iteration limit (>={OLD_ITER_LIMIT})")
if not rerun_cases:
    print("Nothing to re-run!")
    sys.exit(0)

# ── Re-run maxed-out cases ───────────────────────────────────────────────
print("\n" + "=" * 70)
print(f"Re-running {len(rerun_cases)} cases with MAX_ITER={MAX_ITER} ...")
print("=" * 70)

done = 0
t_start = time.time()

for target_nu, key, topo_name, topo_idx, dv in rerun_cases:
    topo = topologies[topo_idx]
    old_r = summary_data[key][topo_name][dv]
    old_loss = old_r['final_loss']

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
            print(f"  FAILED: nu*={target_nu}, topo={topo_name}, dv={dv}, seed={opt_seed}: {e}")

    done += 1
    elapsed = time.time() - t_start
    eta_s = (elapsed / done) * (len(rerun_cases) - done) if done > 0 else 0

    if best is not None:
        new_loss = best['final_loss']
        improved = new_loss < old_loss
        # Always update with the re-run result (more iterations)
        summary_data[key][topo_name][dv] = {
            'poisson': best['poisson'],
            'young': best['young'],
            'final_loss': best['final_loss'],
            'converged': best['converged'],
            'iterations': best['iterations'],
            'time_seconds': best.get('time_seconds', 0),
        }
        tag = "IMPROVED" if improved else "same"
        print(f"  [{done:3d}/{len(rerun_cases)}] nu*={target_nu:+.1f}  "
              f"topo={topo_name:25s}  dv={dv:13s}  "
              f"loss: {old_loss:.2e} -> {new_loss:.2e}  [{tag}]  "
              f"[ETA {eta_s:.0f}s]")
    else:
        print(f"  [{done:3d}/{len(rerun_cases)}] nu*={target_nu:+.1f}  "
              f"topo={topo_name:25s}  dv={dv:13s}  "
              f"-> FAILED ALL SEEDS  [ETA {eta_s:.0f}s]")

print(f"\nRe-runs complete in {time.time() - t_start:.1f}s")

# ── Save updated JSON ────────────────────────────────────────────────────
with open(json_path, 'w') as f:
    json.dump(summary_data, f, indent=2)
print(f"Saved updated {json_path}")

# ── Regenerate READMEs and success/fail markers ──────────────────────────
print("\n" + "=" * 70)
print("Regenerating READMEs and success/fail markers ...")
print("=" * 70)

TOPO_DESCRIPTIONS = {
    'iso_crystal': 'Isotropic crystal \u2014 regular triangular lattice, shape=(1,1), orientation=0',
    'aniso_crystal': 'Anisotropic crystal \u2014 stretched lattice, shape=(1.5, 0.8), orientation=pi/6',
    'foam_eta02_42': 'Foam eta=0.2 (seed 42) \u2014 Delaunay BEFORE perturbation',
    'foam_eta02_137': 'Foam eta=0.2 (seed 137) \u2014 Delaunay BEFORE perturbation',
    'foam_eta045_256': 'Foam eta=0.45 (seed 256) \u2014 Delaunay BEFORE perturbation',
    'foam_eta045_314': 'Foam eta=0.45 (seed 314) \u2014 Delaunay BEFORE perturbation',
    'poisson_999': 'Poisson random network (seed 999) \u2014 Delaunay AFTER perturbation, eta=0.3',
    'poisson_1337': 'Poisson random network (seed 1337) \u2014 Delaunay AFTER perturbation, eta=0.3',
}

topo_names = [t['name'] for t in topologies]

for target_nu in POISSON_TARGETS:
    key = f"{target_nu:+.1f}"
    if target_nu < 0:
        tag = f"num{abs(target_nu):.1f}".replace('.', '')
    elif target_nu == 0:
        tag = "nup00"
    else:
        tag = f"nup{target_nu:.1f}".replace('.', '')

    subdir = os.path.join(OUT, tag)
    os.makedirs(subdir, exist_ok=True)

    # Remove old markers
    for old_marker in glob.glob(os.path.join(subdir, 'aaa--*')):
        os.remove(old_marker)

    # Check success
    any_success = False
    n_converged = 0
    best_loss = float('inf')
    best_label = ""
    total_cases = 0

    for topo_name in topo_names:
        if topo_name not in summary_data.get(key, {}):
            continue
        for dv in DESIGN_VARS:
            if dv not in summary_data[key][topo_name]:
                continue
            total_cases += 1
            r = summary_data[key][topo_name][dv]
            loss = r['final_loss']
            if loss < SUCCESS_THRESHOLD:
                any_success = True
                n_converged += 1
            if loss < best_loss:
                best_loss = loss
                best_label = f"{topo_name} / {dv}"

    marker = 'aaa--success' if any_success else 'aaa--fail'
    open(os.path.join(subdir, marker), 'w').close()

    # Generate README
    lines = []
    lines.append(f"# Target Poisson Ratio: nu = {target_nu}")
    lines.append("")
    lines.append("## Run Configuration")
    lines.append("")
    lines.append(f"- **Target nu**: {target_nu}")
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

    for topo_name in topo_names:
        if topo_name not in summary_data.get(key, {}):
            continue
        desc = TOPO_DESCRIPTIONS.get(topo_name, topo_name)
        lines.append("")
        lines.append(f"### {topo_name}")
        lines.append(f"_{desc}_")
        lines.append("")
        lines.append("| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |")
        lines.append("|---|---|---|---|---|---|")

        for dv in DESIGN_VARS:
            if dv not in summary_data[key][topo_name]:
                lines.append(f"| {dv} | N/A | N/A | No | N/A | N/A |")
                continue
            r = summary_data[key][topo_name][dv]
            nu_str = f"{r['poisson']:+f}" if not (r['poisson'] != r['poisson']) else "NaN"
            loss_str = f"{r['final_loss']:.2e}"
            conv_str = "Yes" if r['converged'] else "No"
            iter_str = str(r['iterations'])
            time_str = f"{r.get('time_seconds', 0):.1f}"
            lines.append(f"| {dv} | {nu_str} | {loss_str} | {conv_str} | {iter_str} | {time_str} |")

    readme_path = os.path.join(subdir, 'README.md')
    with open(readme_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    print(f"  {tag}: {marker}, README updated ({n_converged}/{total_cases} converged)")

print("\n" + "=" * 70)
print("ALL DONE")
print("=" * 70)
