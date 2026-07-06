# Phase 3 — Inverse Design

Find per-bond rigidities `k` (and, optionally, rest lengths `l0`) that realise a **target elastic
response**, by gradient descent through the physical intrinsic forward solver
(`forward(method='intrinsic', physical_units=True)` — see `../Phase 2/SOLVER_GUIDE.md`).

Targets can be **global**, **local** (a sub-region), or **mixed** (several at once, e.g. a global
average with a specific local patch). Works at any mesh size — gradients flow through the dense
path (≤600 tri) or the large-N adjoint path (>600) automatically — on both periodic and open
networks.

Files:
| file | what |
|---|---|
| `inverse_design.py` | the design engine: `DesignProblem`, `Objective`, `optimize`, `validate` |
| `test_inverse_design.py` | 7-test suite (round-trip, property, tensor, local, mixed, large-N, open) |

---

## Concepts

**`DesignProblem`** wraps a network (periodic or open) + the solver + the per-bond→per-triangle
map. Design variables are **per-bond** (shared edges get one consistent `k`).
- `DesignProblem.periodic(N, eta, seed)` — a periodic perturbed-lattice unit cell.
- `DesignProblem.open(tri)` — an open mesh from a scipy triangulation (`Disc_2_Cont_optimized`).

**`Objective(kind, target, region=None, weight=1.0)`** — one target over a region.
- `kind`: `'nu'` (Poisson ratio, scale-invariant), `'E'` (Young's modulus, physical units), or
  `'tensor'` (the full physical 6-vector).
- `region`: `None` → whole network (**global**); an array of triangle indices → **local**.
  Build regions with `prob.region_in_circle(center, radius)` or `prob.region_where(predicate)`.
- Multiple objectives in one `optimize` call → **mixed** design.

**`optimize(prob, objectives, mode='k', optimizer='lbfgs', n_iter, n_restarts, seed)`** — runs the
design. `mode ∈ {'k','l0','both'}`. Returns `dict(k, l0, loss, history)`.

**`validate(prob, k, l0, objectives)`** — re-evaluates each objective at the designed params and
returns achieved-vs-target.

How it works: variables are softplus-positive (`k = softplus(raw)`); the loss is a weighted sum of
per-objective distances, where each region's response is the physical homogenised tensor over that
region (unweighted per-triangle mean × `8·N_reg/A_reg`); optimised with L-BFGS (strong-Wolfe) or
Adam. Because `k→C_eff` is ~`3·N_bond → 6`, the problem is heavily underdetermined — L-BFGS hits
targets in tens of iterations.

---

## Examples

**Global auxetic target:**
```python
from inverse_design import DesignProblem, Objective, optimize, validate
prob = DesignProblem.periodic(N=14, eta=0.3, seed=0)
res = optimize(prob, [Objective('nu', target=-0.2)], mode='k', n_iter=80)
print(validate(prob, res['k'], res['l0'], [Objective('nu', -0.2)]))
# -> achieved nu ≈ -0.200
```

**Match a full elastic tensor (round-trip):**
```python
import torch
k_true = 0.3 + 1.5 * torch.rand(prob.n_bond)
target = prob.region_tensor(prob.forward(k_true)['per_triangle'], None).detach()  # (6,) physical
res = optimize(prob, [Objective('tensor', target)], mode='k', n_iter=120)
```

**Local region — a specific response in a patch:**
```python
prob = DesignProblem.periodic(N=16, eta=0.2, seed=4)
patch = prob.region_in_circle(prob.centroids.mean(0), radius=1.5)
res = optimize(prob, [Objective('nu', target=-0.15, region=patch)], mode='k', n_iter=120)
```

**Mixed — global average with a local auxetic patch (the headline capability):**
```python
prob = DesignProblem.periodic(N=18, eta=0.2, seed=5)
patch = prob.region_in_circle(prob.centroids.mean(0), radius=1.4)
objs = [Objective('nu', target=+0.25, region=None,  weight=1.0),   # whole network stays positive
        Objective('nu', target=-0.20, region=patch, weight=2.0)]   # the patch is auxetic
res = optimize(prob, objs, mode='k', n_iter=150)
# validate -> global nu ≈ +0.25 AND patch nu ≈ -0.20, simultaneously
```

**Open mesh:**
```python
import Disc_2_Cont_optimized as D2C
prob = DesignProblem.open(D2C.generate_foam_points((5, 5), 0.2))
res = optimize(prob, [Objective('nu', target=0.1)], mode='k', n_iter=100)
```

**Large network (>600 triangles) — gradients via the adjoint, no code change:**
```python
prob = DesignProblem.periodic(N=20, eta=0.3, seed=6)   # 800 triangles
res = optimize(prob, [Objective('nu', target=0.0)], mode='k', n_iter=60)
```

**Design rest lengths instead of / with k** (`mode='l0'` or `'both'`) is supported and plumbed, but
note: in the current solver `l0` enters only as `k/l0²`, so `l0`-design is mathematically
degenerate with `k`-design until reference-metric / residual-stress physics is added
(`../Phase 2/SOLVER_GUIDE.md` §7).

---

## Verify

```
python "Phase 3/test_inverse_design.py"
```
Covers round-trip (tensor err ~1e-5), auxetic ν targeting, E targeting, a local patch, a mixed
global+local design, a large-N (800-tri, adjoint) design, and an open mesh — all recovering their
targets. For a designed **periodic** cell you can independently confirm the physical ν/E against
the PBC simulation via `verification_tools/physical_homog.sim_nuE`.

---

## Scope (v1) / not yet

- **Connectivity** is fixed (given network). Later: sparsity-penalised `k` soft-removal on a dense
  base graph.
- **Multi-start clustering / per-edge CV** solution-manifold analysis is deferred (`optimize` has
  `n_restarts` but no clustering yet).
- **`l0` independent physics** (incompatible reference metric → residual stress) waits on the
  reference-metric work; the variable is exposed but degenerate today.
