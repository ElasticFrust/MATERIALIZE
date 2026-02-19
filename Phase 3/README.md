# Phase 3 — Direct Inverse Optimization

Given a **target** elastic tensor, find spring rigidities that produce it — using gradient descent through the differentiable forward solver from Phase 2.

## What's in this folder

| File | What it does |
|------|-------------|
| `inverse_optimize.py` | Inverse optimization engine: target tensor → spring rigidities. Supports L-BFGS and Adam, multi-start campaigns, clustering of converged solutions, and round-trip validation. |
| `test_inverse.py` | Test suite: parameterization round-trip, single optimization, Adam/L-BFGS convergence, round-trip validation, gradient flow, mini campaign. |

## How it works

### Problem setup

We fix the mesh geometry (node positions, triangulation) and ask: **what spring rigidities reproduce a given macroscopic elastic tensor?**

```
Target elastic tensor T*  (6 components)
         ↓
    Optimize k such that  forward(k) ≈ T*
         ↓
    Loss = MSE( forward(k), T* )
         ↓
    Gradient ∂L/∂k  via PyTorch autograd
         ↓
    L-BFGS / Adam update
```

### Key design choices

1. **Softplus parameterization** — rigidities must be positive (springs can't have negative stiffness). We optimize unconstrained `raw` parameters and map them to positive rigidities via `k = softplus(raw, β=5)`.

2. **L-BFGS optimizer** (default) — second-order method, ideal for smooth low-dimensional problems. Falls back to Adam if needed.

3. **Multi-start campaign** — runs 20+ random initializations to explore the solution landscape. Are there many distinct solutions or just one?

4. **Greedy clustering** — groups converged rigidity vectors by normalized distance to identify distinct solution families.

5. **Round-trip validation** — every converged solution is fed back through the forward solver to verify reconstruction error < 1e-3.

## Quick start

### Run the tests

```bash
python "Phase 3/test_inverse.py"
```

### Run a full campaign

```bash
# Default: 4×4 mesh, 20 random starts, L-BFGS
python "Phase 3/inverse_optimize.py"

# Custom settings
python "Phase 3/inverse_optimize.py" \
    --size 6 6 \
    --eta 0.3 \
    --n-starts 50 \
    --optimizer lbfgs \
    --save results.json

# Quick test (quiet mode)
python "Phase 3/inverse_optimize.py" --n-starts 5 -q
```

### Use as a library

```python
import sys; sys.path.insert(0, '..')
sys.path.insert(0, '../Phase 2')

import Disc_2_Cont_optimized as D2C
from forward_solver_torch import from_triangulation
from inverse_optimize import run_single_optimization, validate_round_trip

# Build solver
tri = D2C.generate_foam_points(size=(4, 4), eta=0.2)
solver, default_rigs, _ = from_triangulation(tri)

# Get target tensor
import torch
with torch.no_grad():
    gt = solver(default_rigs)
target = gt['elastic_tensor']

# Inverse optimize
result = run_single_optimization(
    solver=solver,
    target_tensor=target,
    n_triangles=len(tri.simplices),
    optimizer_type='lbfgs',
)

print(f"Relative error: {result['rel_error']:.3e}")
print(f"Poisson: {result['poisson']:.6f} (target: {gt['poisson'].item():.6f})")
```

## CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `--size W H` | `4 4` | Mesh size |
| `--eta` | `0.2` | Foam disorder |
| `--n-starts` | `20` | Random initializations |
| `--max-iter` | `500` | Max iterations per run |
| `--lr` | `0.05` | Learning rate |
| `--optimizer` | `lbfgs` | `lbfgs` or `adam` |
| `--tol` | `1e-10` | Convergence tolerance |
| `--seed` | `42` | Mesh generation seed |
| `--save FILE` | — | Save results to JSON |
| `-q` | — | Quiet mode |

## What the output tells you

- **Converged N/M** — how many random starts found a valid solution
- **Distinct clusters** — how many qualitatively different solutions exist
- **Per-edge CV** — coefficient of variation across solutions. Low CV = "locked" edge (same across all solutions). High CV = "free" edge (varies widely)
- **Round-trip error** — forward(optimized_k) vs target, should be < 1e-3
