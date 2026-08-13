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

**`Objective(kind, target, region=None, weight=1.0, thetas=None)`** — one target over a region.
- `kind`:
  - `'nu'` / `'E'` — a **scalar target that means ISOTROPIC ν / E**: that value in *every* direction
    (ν(θ)/E(θ) held flat). **This is the default meaning of "ν = v"** — asking for a scalar means you
    want it isotropic. `validate` reports the achieved mean plus the angular `spread`.
  - `'nu_dir'` / `'E_dir'` — the **legacy single-direction** scalar (one contraction of the tensor);
    it pins only one orientation, so the tensor can still be strongly anisotropic. Kept for when you
    deliberately want just one axis.
  - `'tensor'` — the full physical 6-vector;
  - `'nu_theta'` / `'E_theta'` — a **directional profile** ν(θ)/E(θ) over `thetas` (default
    `ANG=linspace(0,π,37)`); scalar target broadcasts to flat. Autograd-safe (`c6_to_nu_theta`/
    `c6_to_E_theta`), match `_common.nu_E_theta` exactly.
  - `'isotropy'` — penalise the tensor's anisotropic part (`_anisotropy(C6)`); force
    direction-independence with the level free.
- `region`: `None` → whole network (**global**); an array of triangle indices → **local**.
  Build regions with `prob.region_in_circle(center, radius)` or `prob.region_where(predicate)`
  (the verifications harness adds disc/rect/ring/polygon shapes via `_common.region_shape`).
- Multiple objectives in one `optimize` call → **mixed** design (e.g. a flat `nu` **and** a shaped
  `E_theta` → isotropic ν with directional E).

**`constrain(...)` — fix some quantities, free the rest.** A convenience wrapper that assembles the
objectives for "these fixed, everything else free":
- `constrain(region=R, nu=v)` → isotropic ν=v, E free · `constrain(region=R, E=v)` → isotropic E, ν free.
- `constrain(region=R, isotropic=True)` → direction-independent, level(s) free.
- `constrain(region=R, tensor=isotropic_c6(v_nu, v_E))` → **exact isotropic** (ν,E), nothing free.
  `isotropic_c6(nu,E)` returns the isotropic 2D 6-vector.
- legacy single-direction knobs via `nu_scalar`/`E_scalar` (→ `'nu_dir'`/`'E_dir'`).

Why scalar = isotropic matters (regular lattice, central auxetic patch, ν(θ) measured *inside*):
the legacy `Objective('nu_dir', −0.3)` gives ν(θ) range **1.69** (wildly anisotropic — and its scalar
doesn't even read the isotropic value), while `Objective('nu', −0.3)` gives range **0.011**, and
`constrain(tensor=isotropic_c6(−0.3,·))` → **0.001**.

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

**Directional response ν(θ) / E(θ) — program or isotropise anisotropy:**
```python
import numpy as np
from inverse_design import ANG
prob = DesignProblem.periodic(N=20, eta=0.3, seed=0)
# program a 4-fold Poisson profile (4θ is flat-E-compatible; a 2θ ν would need matching E-anisotropy — reciprocity):
res = optimize(prob, [Objective('nu_theta', 0.2 + 0.35*np.cos(4*ANG))], mode='k', n_iter=110)
# isotropise an anisotropic base to a chosen flat level (scalar broadcasts):
res = optimize(prob, [Objective('nu_theta', -0.2)], mode='k', n_iter=150)
# independent anisotropy: flat ν, directional E (mixed directional objectives):
res = optimize(prob, [Objective('nu_theta', 0.2,               weight=4.0),
                      Objective('E_theta', 1.0*(1+0.4*np.cos(2*ANG)), weight=1.0)], mode='k', n_iter=150)
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
note: in the current flat solver `l0` enters only as `k/l0²`, so `l0`-design *appears* degenerate
with `k`-design (a flat-gauge artefact — `ḡ=ḡ(l0)` in general) until reference-metric / residual-stress physics is added
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
  reference-metric work; the variable is exposed but only *apparently* degenerate (flat-gauge artefact) today.
