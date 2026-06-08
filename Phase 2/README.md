# Phase 2 — Differentiable Forward Solver (PyTorch)

## What's in this folder

| File | What it does |
|------|-------------|
| `forward_solver_torch.py` | PyTorch port of the Woodbury-optimized forward solver. Computes the effective elastic tensor, Poisson's ratio, and Young's modulus — and is **fully differentiable** with respect to per-edge spring rigidities and rest lengths. Includes the KKT edge-compatibility correction (see below). |
| `test_torch_vs_numpy.py` | Cross-validation suite: compares PyTorch vs NumPy outputs on 10+ random networks, verifies gradient correctness via `torch.autograd.gradcheck`, and benchmarks performance. |
| `test_kkt_correction.py` | Six-test verification suite for the KKT correction: uniform-mesh W=0 check, constraint satisfaction ‖JW‖≈0, ν(η) comparison vs mean-field, hexagonal crystal regression, and gradcheck. |

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

---

## KKT Edge-Compatibility Correction

### The problem with mean-field alone

The Woodbury solve (Step 5 above, paper Eq. 19–20) treats each triangle as if it lives in the global mean field. This ignores spatial correlations between neighbouring triangles. The consequence is that the non-affine response tensor W can be locally *incompatible*: two triangles sharing an edge may disagree on that edge's metric contribution, violating the physical requirement that edge lengths are shared.

### The constraint

For every interior edge $e$ shared by triangles $s_1$ and $s_2$, with edge vector $\Delta x$:

$$\bigl[W(s_1) - W(s_2)\bigr]^{\alpha\beta}_{\mu\nu}\, \Delta x^\mu \Delta x^\nu = 0 \quad \forall\,\alpha\beta$$

In matrix form, $J W = 0$, where $J \in \mathbb{R}^{3 E_\text{int} \times 9N}$ encodes all edge constraints (one scalar equation per interior edge per loading mode).

### The KKT system

The augmented (KKT) system that enforces both stationarity and compatibility is:

$$\begin{pmatrix} P & J^\top \\ J & 0 \end{pmatrix} \begin{pmatrix} W \\ \Lambda \end{pmatrix} = \begin{pmatrix} -\delta A \\ 0 \end{pmatrix}$$

where $P = A - B$ is the mean-field operator from the Woodbury solve and $\Lambda \in \mathbb{R}^{3 E_\text{int}}$ are Lagrange multipliers (one per edge-loading pair). Geometrically, $W$ is the **projection of $W_0$ onto $\ker J$** in the $P$-metric — the closest compatible field to the unconstrained mean-field solution.

### Closed-form solution via two Woodbury steps

1. **Unconstrained solve** (existing Woodbury code): $W_0 = -P^{-1} \delta A$
2. **Constraint residual**: $r = J W_0$
3. **Gram matrix**: $G = J P^{-1} J^\top$  (how incompatible directions interact under $P$)
4. **Multipliers**: $G \Lambda = r$
5. **Correction**: $W = W_0 - P^{-1} J^\top \Lambda$

Step 3 applies $P^{-1}$ to each row of $J$, reusing the same Woodbury factors $(A^{-1}, S, z)$ already computed in step 1.

### Dense vs sparse path

| Network size | Path | Notes |
|---|---|---|
| $N \leq 500$ triangles | **Dense differentiable** (`_woodbury_solve` with `J`) | Supports `torch.autograd.gradcheck`; $J$ is $3 E_\text{int} \times 9N$ ($\lesssim 200$ MB) |
| $N > 500$ triangles | **Sparse (scipy)** (`_woodbury_kkt_sparse`) | $J$ never stored explicitly; uses $G = G_\text{local} + \text{rank-9}$ decomposition; no gradient |

The sparse path stores only the arrays `(s1, s2, q)` (O(N) scalars) and assembles $G_\text{local} = J A_\text{diag}^{-1} J^\top$ as a sparse CSC matrix with $\sim 9$ non-zeros per row. The rank-9 correction from the Woodbury term is handled via a dense $9 \times 9$ system. Memory is $O(N)$ rather than $O(N^2)$.

> **Important**: The $J^\top \Lambda$ accumulation uses `np.add.at` rather than fancy-index `+=`. When multiple interior edges share the same triangle, the same `(triangle, column)` index appears multiple times in the scatter; numpy's `+=` with fancy indexing silently drops all but the last update, while `np.add.at` correctly accumulates all contributions.

### Empirical results (20×20 networks, 10 trials per η)

| η | ν mean-field | ν KKT | Δν |
|---|---|---|---|
| 0.00 | +0.333 | +0.333 | 0.000 |
| 0.10 | +0.327 | +0.326 | −0.001 |
| 0.20 | +0.305 | +0.303 | −0.002 |
| 0.30 | +0.225 | +0.197 | −0.028 |
| 0.40 | −0.145 | +0.017 | +0.162 |

At $\eta \leq 0.30$, the KKT correction consistently shifts $\nu$ more negative (more auxetic). At $\eta \geq 0.35$, near-degenerate sliver triangles make both solvers numerically unstable; the KKT correction partially stabilizes the result (median closer to zero, smaller variance).

Full benchmarking script and cached data: `benchmarking/plot_poisson_vs_eta_kkt.py`.

---

## Quick start

```python
import sys; sys.path.insert(0, '..')
import Disc_2_Cont_optimized as D2C
from forward_solver_torch import from_triangulation

# Generate a foam network
tri = D2C.generate_foam_points(size=(4, 4), eta=0.2)

# Build differentiable solver (KKT correction automatic; dense path for N≤500)
solver, rigidities, rest_lengths = from_triangulation(tri)

# Forward pass (no gradients needed)
result = solver(rigidities, rest_lengths)
print(f"Poisson = {result['poisson'].item():.6f}")
print(f"Young   = {result['young'].item():.6f}")

# Forward + backward (for optimization — only on N≤500 networks)
rigidities.requires_grad_(True)
result = solver(rigidities, rest_lengths)
result['poisson'].backward()
print(f"d(Poisson)/d(rigidities) shape: {rigidities.grad.shape}")

# Large network (N>500): KKT correction via scipy, no gradient
tri_large = D2C.generate_foam_points(size=(20, 20), eta=0.2)
solver_large, rigs_large, rl_large = from_triangulation(tri_large)
import torch
with torch.no_grad():
    result_large = solver_large(rigs_large, rl_large)
print(f"Large network Poisson = {result_large['poisson'].item():.6f}")
```

## Running the tests

```bash
cd /path/to/MATERIALIZE
python "Phase 2/test_torch_vs_numpy.py"
python "Phase 2/test_kkt_correction.py"
```

`test_torch_vs_numpy.py` runs 4 test suites:
1. **Output comparison** — 10 random networks, all eta values, checks every intermediate
2. **Custom parameters** — random rigidities and rest lengths, verifies agreement
3. **Gradient correctness** — `torch.autograd.gradcheck` (finite differences vs analytic)
4. **Performance** — timing for various network sizes, forward and forward+backward

`test_kkt_correction.py` runs 6 targeted tests for the KKT correction:
1. **Backup files exist** — confirms `.bckp` copies are present
2. **Uniform mesh** — $\delta A = 0 \Rightarrow W = 0$, $\nu = 1/3$ exactly
3. **Constraint satisfaction** — $\|J W\| \approx 0$ after correction (vs $\|J W_0\| \gg 0$ before)
4. **Auxetic response** — $\nu(\eta)$ table for $\eta \in [0, 0.5]$, old vs new
5. **Hexagonal crystal** — both solvers agree to $< 10^{-4}$ at $\eta = 0$
6. **Gradcheck** — end-to-end differentiability through the KKT solve (small mesh, double precision)
