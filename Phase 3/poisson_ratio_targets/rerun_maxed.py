#!/usr/bin/env python
"""Re-run only the cases from poisson_ratio_targets that hit the old MAX_ITER=500 limit.

Loads existing sweep_results.json, identifies cases with iterations >= 500,
re-runs them with MAX_ITER=2000, and merges improved results back.
"""

import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'Phase 2'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'Phase 3'))
os.environ['MPLBACKEND'] = 'Agg'

import numpy as np
import torch

import Disc_2_Cont_optimized as D2C
from forward_solver_torch import from_triangulation
from inverse_optimize import run_property_optimization

# ── Configuration (must match run_sweep.py) ───────────────────────────────
POISSON_TARGETS = [-0.9, -0.7, -0.5, -0.3, -0.1, 0.0, 0.1, 0.3, 0.5, 0.7, 0.9]
N_TOPOLOGIES = 5
TOPO_SEEDS = [42, 137, 256, 314, 999]
DESIGN_VARS = ['rigidities', 'rest_lengths', 'both']
OPT_SEEDS = [7, 144, 281]
MESH_SIZE = (10, 10)
ETA = 0.2
MAX_ITER = 2000
LR = 0.05
TOL = 1e-14
OLD_ITER_LIMIT = 500
OUT = os.path.dirname(os.path.abspath(__file__))

# ── Generate the 5 fixed topologies (same seeds) ─────────────────────────
print("=" * 70)
print("Regenerating 5 fixed network topologies ...")
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
          f"natural nu={info['natural_poisson']:.4f}")

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
    for topo_idx in range(N_TOPOLOGIES):
        topo_key = str(topo_idx)
        if topo_key not in summary_data[key]:
            continue
        for dv in DESIGN_VARS:
            if dv not in summary_data[key][topo_key]:
                continue
            r = summary_data[key][topo_key][dv]
            if r['iterations'] >= OLD_ITER_LIMIT:
                rerun_cases.append((target_nu, key, topo_key, topo_idx, dv))

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

for target_nu, key, topo_key, topo_idx, dv in rerun_cases:
    topo = topologies[topo_idx]
    old_r = summary_data[key][topo_key][dv]
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
            print(f"  FAILED: target={target_nu}, topo={topo_idx}, dv={dv}, seed={opt_seed}: {e}")

    done += 1
    elapsed = time.time() - t_start
    eta_s = (elapsed / done) * (len(rerun_cases) - done) if done > 0 else 0

    if best is not None:
        new_loss = best['final_loss']
        improved = new_loss < old_loss
        summary_data[key][topo_key][dv] = {
            'poisson': best['poisson'],
            'young': best['young'],
            'final_loss': best['final_loss'],
            'converged': best['converged'],
            'iterations': best['iterations'],
            'time_seconds': best.get('time_seconds', 0),
        }
        tag = "IMPROVED" if improved else "same"
        print(f"  [{done:3d}/{len(rerun_cases)}] nu*={target_nu:+.1f}  "
              f"topo={topo_idx}  dv={dv:13s}  "
              f"loss: {old_loss:.2e} -> {new_loss:.2e}  [{tag}]  "
              f"[ETA {eta_s:.0f}s]")
    else:
        print(f"  [{done:3d}/{len(rerun_cases)}] nu*={target_nu:+.1f}  "
              f"topo={topo_idx}  dv={dv:13s}  "
              f"-> FAILED ALL SEEDS  [ETA {eta_s:.0f}s]")

print(f"\nRe-runs complete in {time.time() - t_start:.1f}s")

# ── Save updated JSON ────────────────────────────────────────────────────
with open(json_path, 'w') as f:
    json.dump(summary_data, f, indent=2)
print(f"Saved updated {json_path}")

print("\n" + "=" * 70)
print("ALL DONE")
print("=" * 70)
