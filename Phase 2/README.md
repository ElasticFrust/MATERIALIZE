# Phase 2 — Differentiable Forward Solver (PyTorch)

## What's in this folder

| File | What it does |
|------|-------------|
| `forward_solver_torch.py` | PyTorch port of the Woodbury-optimized forward solver. Computes the effective elastic tensor, Poisson's ratio, and Young's modulus — and is **fully differentiable** with respect to per-edge spring rigidities and rest lengths. |
| `test_torch_vs_numpy.py` | Cross-validation suite: compares PyTorch vs NumPy outputs on 10+ random networks, verifies gradient correctness via `torch.autograd.gradcheck`, and benchmarks performance. |

## The forward solver, explained

We model a 2D material as a Delaunay triangulation of perturbed hexagonal lattice points. Each triangle has 3 edges, each edge a spring. The solver computes the **macroscopic elastic properties** of the bulk material from the **microscopic spring parameters**.

### Pipeline (7 steps)

```
Inputs: node positions (fixed), mesh topology (fixed),
        spring rigidities k[n,i], rest lengths l₀[n,i]  ← differentiable
                                                ↓
┌─────────────────────────────────────────────────────────────────┐
│ Step 1: Edge vectors                                            │
│   v[n,i] = pos[node_a] − pos[node_b]                           │
│                                                                 │
│ Step 2: Bare elastic tensor  A[n]  (5 components per triangle)  │
│   factor = k / l₀² / 16                                        │
│   a = [Σ f·vx⁴,  Σ f·vx³vy,  Σ f·vx²vy²,  Σ f·vxvy³,  Σ f·vy⁴] │
│                                                                 │
│ Step 3: Deviation from the mean                                 │
│   δA[n] = A[n] − mean(A)                                       │
│                                                                 │
│ Step 4: Build 9×9 block matrices via kron(M, I₃)               │
│   A_blocks[n], B_blocks[n] (from δA), dA_vecs[n]               │
│                                                                 │
│ Step 5: Woodbury solve  — O(N) instead of O(N³)                 │
│   Inverts N batched 9×9 blocks + one 9×9 Schur complement      │
│   Exploits: A is block-diagonal, perturbation B is rank-9       │
│                                                                 │
│ Step 6: 4-index tensor contraction                              │
│   C = A + A·W + A·W + A·W·W   (einsum over 2×2×2×2 tensors)   │
│                                                                 │
│ Step 7: Homogenize → material properties                        │
│   C_eff = mean(C)  →  Poisson's ratio ν,  Young's modulus E    │
└─────────────────────────────────────────────────────────────────┘
                                                ↓
Outputs: ν (Poisson's ratio), E (Young's modulus),
         C_eff (6-component elastic tensor)
         — all differentiable w.r.t. k and l₀
```

### Why this matters

Because the solver is differentiable, we can:
- **Optimize** spring parameters to achieve a target elastic tensor (Phase 3)
- **Learn** the manifold of solutions via generative models (Phase 4)
- **Extract** physical design principles about which microstructures produce which bulk behavior

## Quick start

```python
import sys; sys.path.insert(0, '..')
import Disc_2_Cont_optimized as D2C
from forward_solver_torch import from_triangulation

# Generate a foam network
tri = D2C.generate_foam_points(size=(4, 4), eta=0.2)

# Build differentiable solver
solver, rigidities, rest_lengths = from_triangulation(tri)

# Forward pass (no gradients needed)
result = solver(rigidities, rest_lengths)
print(f"Poisson = {result['poisson'].item():.6f}")
print(f"Young   = {result['young'].item():.6f}")

# Forward + backward (for optimization)
rigidities.requires_grad_(True)
result = solver(rigidities, rest_lengths)
result['poisson'].backward()
print(f"d(Poisson)/d(rigidities) shape: {rigidities.grad.shape}")
```

## Running the tests

```bash
cd /path/to/MATERIALIZE
python "Phase 2/test_torch_vs_numpy.py"
```

This runs 4 test suites:
1. **Output comparison** — 10 random networks, all eta values, checks every intermediate
2. **Custom parameters** — random rigidities and rest lengths, verifies agreement
3. **Gradient correctness** — `torch.autograd.gradcheck` (finite differences vs analytic)
4. **Performance** — timing for various network sizes, forward and forward+backward
