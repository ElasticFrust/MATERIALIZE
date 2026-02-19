# Phase 3 — Direct Inverse Optimization

Given a **target** elastic tensor, find spring rigidities that produce it — using gradient descent through the differentiable forward solver from Phase 2.

## What's in this folder

| File | What it does |
|------|-------------|
| `inverse_optimize.py` | Inverse optimization engine: target tensor → spring rigidities. Supports L-BFGS and Adam, multi-start campaigns, clustering of converged solutions, and round-trip validation. |
| `test_inverse.py` | Test suite: parameterization round-trip, single optimization, Adam/L-BFGS convergence, round-trip validation, gradient flow, mini campaign. |

## The inverse problem

Phase 2 computes `T(k)` — the macroscopic elastic tensor as a function of spring rigidities. Phase 3 inverts this:

> **Given T\*, find k such that T(k) ≈ T\*.**

This is hard because:
- The mapping `k → T` is **many-to-one** — there are N×3 spring rigidities but only 6 tensor components, so the system is heavily underdetermined.
- Rigidities must be **positive** (springs can't have negative stiffness).
- We need gradients to flow backward through the entire 7-step physics pipeline (edge vectors → bare tensor → Woodbury solve → 4-index contraction → homogenization).

## How each piece works

### 1. Softplus Parameterization

We can't optimize rigidities directly — gradient descent might push them negative. Instead we optimize unconstrained parameters `raw ∈ ℝ` and map:

```
k = softplus(raw, β=5) = (1/β) · log(1 + exp(β · raw))
```

- Always positive for any input
- Smooth, with well-behaved gradients everywhere
- β=5 keeps the mapping steep: k ≈ raw for moderate values, floors at 0
- Inverse: `raw = log(exp(β·k) - 1) / β`

### 2. Loss function

MSE on the 6 independent components of the symmetric elastic tensor:

```
L = (1/6) Σᵢ₌₁⁶ (Cᵢ(k) - Cᵢ*)²
```

where `C(k) = forward_solver(k)` runs the full Phase 2 pipeline. There's also a `property_loss` variant that targets Poisson's ratio and Young's modulus directly:

```
L = w_ν · (ν(k) - ν*)² + w_E · (E(k) - E*)²
```

### 3. Single optimization run

One complete optimization from a random starting point:

```
1. Initialize: raw ~ 0.5 · N(0,1)
   → k = softplus(raw) ~ log-uniform around 1.0

2. Optimization loop:
     k = softplus(raw)                    ← enforce positivity
     result = forward_solver(k)           ← Phase 2's full 7-step pipeline
     loss = MSE(result.tensor, target)    ← scalar loss
     loss.backward()                      ← autograd: ∂L/∂raw through all 7 steps
     optimizer.step(raw)                  ← L-BFGS or Adam update

3. Return best-loss rigidities and diagnostics
```

**L-BFGS** (default): Quasi-Newton method that approximates the Hessian from gradient history. Uses strong Wolfe line search. Converges in ~60 iterations (~1 second) for typical meshes. Ideal because the loss landscape is smooth.

**Adam**: First-order momentum optimizer. Needs ~1000 iterations but is more robust to noisy/flat regions. Used as fallback.

### 4. Multi-start campaign

Since the inverse problem may have multiple solutions, we run N random initializations and analyze the results:

```
For each of N random seeds:
    result = run_single_optimization(seed=i)

Post-processing:
    1. Filter to converged runs (loss < tolerance)
    2. Cluster converged solutions by rigidity similarity
    3. Compute per-edge statistics (mean, std, coefficient of variation)
    4. Round-trip validate the best solution
```

### 5. Greedy clustering

Groups converged rigidity vectors into solution families:

```
1. Normalize: v̂ᵢ = vᵢ / ||vᵢ||
2. For each unassigned solution i:
     Create cluster = {i}
     For each unassigned j > i:
       if ||v̂ᵢ - v̂ⱼ|| < 0.05:
         Add j to cluster
3. Report cluster centroids and sizes
```

This reveals the **solution landscape**: is there one unique answer or many qualitatively different rigidity patterns that produce the same elastic behavior?

### 6. Round-trip validation

Every converged solution is verified by feeding it back through the forward solver:

```
k_optimized → forward_solver(k_optimized) → T_pred
relative_error = ||T_pred - T*|| / ||T*||
```

Confirms the optimization actually found a physically valid solution (not just a local minimum of the numerical loss).

## Test results

All 6 tests pass:

| Test | What it verifies | Key metric |
|------|-----------------|------------|
| Softplus roundtrip | `softplus(inv_softplus(k)) == k` and positivity | max error = 1.7e-18 |
| Single L-BFGS optimization | Recovers target tensor from random init | rel_error = 8.0e-06, 60 iters |
| Adam optimization | Adam also converges (slower) | rel_error = 3.1e-03, 1000 iters |
| Round-trip validation | `forward(optimized_k) ≈ target` | rel_error = 2.9e-06 |
| Mini campaign (5 starts) | Full pipeline works end-to-end | 5/5 converged, 5 clusters |
| Gradient flow | ∂L/∂raw is nonzero and correct shape | grad norm = 2.0e-07 |

### Key findings

- **L-BFGS converges in ~60 iterations** (< 1 second) to relative error ~10⁻⁵
- **The inverse problem has many solutions**: 5 random starts → 5 distinct clusters. This is expected — N×3 unknowns vs 6 equations means the solution manifold has dimension ~(3N - 6).
- **Per-edge coefficient of variation** identifies structurally important springs ("locked" edges with low CV across all solutions) vs redundant springs ("free" edges with high CV).

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
- **Distinct clusters** — how many qualitatively different solutions exist (reveals the dimensionality of the solution manifold)
- **Per-edge CV** — coefficient of variation of rigidities across converged solutions. Low CV = "locked" edge (structurally essential, same across all solutions). High CV = "free" edge (can vary without changing macroscopic behavior)
- **Round-trip error** — `forward(optimized_k)` vs target, should be < 1e-3

## Dependencies

- Phase 2's `forward_solver_torch.py` (differentiable forward solver)
- `Disc_2_Cont_optimized.py` (mesh generation)
- PyTorch (autograd, optimizers)
- NumPy, SciPy (mesh geometry)
