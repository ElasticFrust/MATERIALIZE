# Forward solver — usage & components guide

`Phase 2/forward_solver_torch.py`. The canonical, up-to-date reference for the differentiable
forward solver: what it computes, its public API, the physical-units convention, and the
large-mesh differentiable (adjoint) path. (The older `README.md` in this folder documents the
legacy mean-field/Woodbury internals and KKT derivation — background only; the **intrinsic**
method below is the verified default.)

---

## 1. What it computes

Given a 2D spring network (a triangulation + per-bond stiffness `k` and rest length `l₀`), the
solver returns the **homogenised effective elastic tensor** and, from it, the Poisson ratio `ν`
and Young's modulus `E` — plus the **per-triangle non-affine response** `W`. It is **fully
differentiable** w.r.t. `k` and `l₀`.

Pipeline (metric-space intrinsic solve, no node displacements):
1. Per-triangle metric Hessian `A(s) = Σ_e (k_e / 4ℓ_e²) q_e q_eᵀ` (from `k`, `l₀`, geometry).
2. Solve the constrained saddle for the strain-concentration `W(s)` — energy minimised subject to
   **edge compatibility + curvature (zero discrete Gaussian curvature) + area-weighted mean**.
3. Per-triangle actual tensor `C(s) = (I+W)ᵀ A(s) (I+W)`; homogenise → `C_eff → ν, E`.

**Homogenisation is the UNWEIGHTED mean of `C(s)`** — this is the true physical (energy = virial)
effective tensor. (Historically an area-weighted mean was used; it biased `ν` on
disordered/anisotropic meshes and over-stated auxeticity. The area weight is still used where it
is correct — the compatibility normalisation `Σ_s S_s δg=0` — just not in the final average.)

**One solve gives the COMPLETE tensor, direction-agnostic.** The saddle solve for `W` handles the
3 independent 2D strain modes (e_xx, e_yy, e_xy) together, so a single `forward(...)` returns the
**full** elastic tensor — all independent components, per-triangle and homogenised — *not* the
response to one imagined load. From that one `C` you read off the entire directional response
(`ν(θ)`, `E(θ)`, shear) at every angle; there is no separate solve per direction. Consequently `ν`
and `E` are **coupled** — both are contractions of the same `C`, so they cannot be prescribed
independently (see `../Phase 3/README.md` on realizability). Caveat: `C` is the **linear (tangent)**
tensor about the given **reference** (geometry + `l₀`) — complete, but small-strain; it does not
capture nonlinear large-deformation response.

---

## 2. Constructing a solver

**Open mesh** (self-contained in Phase 2):
```python
import Disc_2_Cont_optimized as D2C
from forward_solver_torch import from_triangulation

tri = D2C.generate_foam_points(size=(6, 6), eta=0.2)     # scipy Delaunay-like object
solver, k, l0 = from_triangulation(tri)                  # k=(N,3) ones, l0=reference lengths
```

**Periodic unit cell** (via the verification helper, which mounts periodic-correct geometry):
```python
import sys; sys.path.insert(0, 'verification_tools')
import test_cluster_VD as VD
from test_intrinsic_VD import kkt_from_tri_bond
from verify_solver_sweep import make_solver
import numpy as np, torch

geo = VD.build_geometry(N=14, eta=0.3, seed=0); VD.set_VD(geo, 0)   # periodic lattice, k=1
kkt = kkt_from_tri_bond(geo['tri_bond'], geo['edge_vecs'])
solver = make_solver(geo, kkt)
k  = torch.as_tensor(geo['tri_k'])                       # (N,3) per-triangle-edge stiffness
l0 = torch.as_tensor(np.sqrt(geo['actual_len2']))       # reference edge lengths
```

---

## 3. `forward(...)` — the call

```python
out = solver.forward(rigidities, rest_lengths=l0,
                     method='intrinsic',        # default; 'woodbury' = legacy mean field
                     physical_units=False)      # default; True -> E in true physical units
# convenience: solver(k, l0) == solver.forward(k, l0)
```

Arguments:
| arg | meaning |
|---|---|
| `rigidities` | `(N,3)` per-triangle-edge stiffness `k` (differentiable) |
| `rest_lengths` | `(N,3)` rest lengths `l₀` (differentiable); `None` → actual edge length (zero prestress) |
| `method` | `'intrinsic'` (verified default) or `'woodbury'` (legacy mean field) |
| `physical_units` | `False` → internal scale; `True` → rescale `elastic_tensor`/`young` to physical units |
| `use_kkt` / `use_angle_kkt` / `area_weighted` | constraint toggles; default ON for intrinsic |

Returns a dict:
| key | shape | meaning |
|---|---|---|
| `poisson` | scalar | Poisson ratio `ν` (physical, scale-invariant) |
| `young` | scalar | Young's modulus `E` (internal scale, or physical if `physical_units=True`) |
| `elastic_tensor` | `(6,)` | homogenised `C_eff` (6 independent comps) |
| `per_triangle` | `(N,6)` | per-triangle actual tensor `C(s)` — **use for local/regional objectives** |
| `W` | `(N,9)` | per-triangle strain-concentration response |
| `bare` | `(N,5)` | per-triangle bare tensor |

---

## 4. Units — `ν` vs `E`

- **`ν` (poisson) is always physical** — it is dimensionless / scale-invariant, so it is correct
  regardless of `physical_units`.
- **`E` (young) and `elastic_tensor`** are in an internal scale by default (the bare-tensor `/16`
  convention). Pass **`physical_units=True`** to rescale by `8·N/A_total` to true physical units
  (e.g. the unit triangular lattice → `E = 2/√3 ≈ 1.155`). Exact for periodic meshes.

```python
E_internal = solver.forward(k, l0)['young']                          # e.g. 0.0536
E_physical = solver.forward(k, l0, physical_units=True)['young']     # e.g. 0.992
nu         = solver.forward(k, l0)['poisson']                        # same either way
```

---

## 5. Differentiability & mesh size (Component A — the adjoint)

The intrinsic solve has two implementations, chosen automatically by triangle count
`INTRINSIC_DENSE_MAX = 600`:

| Ntri | path | gradients |
|---|---|---|
| ≤ 600 | dense torch Woodbury (`_woodbury_solve_aw`) | ✅ autograd |
| > 600, no grad needed | sparse scipy `splu` (`_intrinsic_solve_W`) | forward-only |
| > 600, **grad needed** | **adjoint** (`_IntrinsicSparseWFn`) | ✅ autograd |

The large-mesh **adjoint** makes `forward(method='intrinsic')` differentiable above the dense cap:
- *forward* is identical to the sparse solve (bit-for-bit same `ν/E`);
- *backward* is the adjoint of the linear KKT solve. The saddle is **symmetric**, so it **reuses
  the forward's `splu` factorisation** — one extra triangular solve + O(N) assembly.
- **Cost:** forward-only is unchanged; gradients add ~**1.07×** (measured, 800 tri). Scales as
  sparse O(N^1.5), not the dense path's O(N³) — so gradient-based design works into the thousands
  of triangles.
- It triggers automatically when the input requires grad (`torch.is_grad_enabled() and
  A3.requires_grad`); forward-only inference keeps the original non-differentiable fast path.

```python
# gradient-based use (e.g. inverse design) at ANY mesh size:
k = solver_k.clone().requires_grad_(True)
nu = solver.forward(k, l0, method='intrinsic', physical_units=True)['poisson']
nu.backward()              # d(nu)/dk in k.grad — dense (<=600) or adjoint (>600), automatically
```

---

## 6. Verification

- **Unit regression:** `python "Phase 2/test_forward_solver.py"` — checks ν=1/3 crystal, foam
  finiteness, dense gradients, autograd-vs-finite-diff, the **large-N adjoint** ([6]), and legacy
  woodbury.
- **Physical ground truth for sim-vs-solver:** `verification_tools/physical_homog.py`
  (`sim_nuE` = virial, `energy_nuE`) — the relaxed network's physical ν/E, used by the
  `verify_*` scripts.
- **Do not** use the legacy area-weighted metric average — `test_cluster_Ceff.Ceff_nuE` has been
  **removed/tombstoned** (superseded; it biased ν on unequal-area meshes); use the physical
  (virial/energy) ground truth.

---

## 7. Notes / caveats

- **Reference lengths `l₀`** are a differentiable input and set the reference metric `ḡ(l₀)` — they
  are **not** a free knob independent of the geometry. In the current flat solver `l₀` enters only as
  `k/l₀²`, so it *appears* degenerate with `k` — but that is a **flat-gauge (`ḡ=I`, geometry-pinned)
  artefact, not a true degeneracy** (moving `l₀` at fixed `ḡ=I` forces the geometry to change). True
  reference-metric / residual-stress physics (incompatible `l₀`) is not yet implemented.
- `physical_units` E-scale is exact for periodic meshes; on open finite samples the boundary edges
  make it slightly approximate (`ν` is unaffected).
- Protected core: changes to `forward_solver_torch.py` are gated by `test_forward_solver.py` and
  the physical `verify_*` suite.
