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

**Periodic unit cell** (`mesh_build` + `solver_build` mount periodic-correct geometry — both live
here in Phase 2, alongside the solver):
```python
from mesh_build import build_geometry, set_VD, kkt_from_tri_bond
from solver_build import make_solver
import numpy as np, torch

geo = build_geometry(N=14, eta=0.3, seed=0); set_VD(geo, 0)   # periodic lattice, k=1
kkt = kkt_from_tri_bond(geo['tri_bond'], geo['edge_vecs'])
solver = make_solver(geo, kkt)
k  = torch.as_tensor(geo['tri_k'])                       # (N,3) per-triangle-edge stiffness
l0 = torch.as_tensor(np.sqrt(geo['actual_len2']))       # reference edge lengths
```
*(Until the A-7b re-layering these came from `verification_tools/test_cluster_VD.py`,
`test_intrinsic_VD.py` and `verify_solver_sweep.py` — i.e. the design layer built its solver out of
the temporary, retireable oracle layer. See `documentation/AUDIT_2026-08.md` A-7b.)*

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
  finiteness, dense gradients, autograd-vs-finite-diff, the **large-N adjoint** ([6]), legacy
  woodbury, and **[7] `C_eff` vs the energy Hessian COMPONENT BY COMPONENT on disordered meshes**.
- **Physical ground truth for sim-vs-solver:** `verification_tools/physical_homog.py`
  (`sim_nuE` = virial, `energy_nuE`, and `energy_C` for the full **tensor**) — the relaxed network's
  physical response, used by the `verify_*` scripts.
- **CLOSED-FORM gate** *(added 2026-08-17)*: `Phase 5/verifications/test_hex_closed_form.py`. The
  hexagon diameter family — 7 nodes, 6 triangles, no periodicity, milliseconds per point — has an
  analytic answer for a rigid perimeter with free hinges,

  ```
  ν(r) = (4r² − 1) / (3 + 4r − 4r²),        r = d/2
  ```

  reproduced to **4.4e-06** over d ∈ [0.05, 2], with landmarks ν(½) = −0.2, ν(1) = 0, ν(2) = +1 (the
  textbook honeycomb value). This is the **strongest** rung of the verification ladder — a reference
  independent of *both* the solver and the sim, where everything else compares two codes. It is
  sensitive to the **shear channel** (ν=1 at the regular hexagon depends on `C_xyxy`), so it would
  have caught **A-0** immediately. The test also asserts the finite-hinge residual is **first order in
  `k_spoke`**, which is what proves the ~0.3–2.7 % offset at the designer's default is the model
  difference and not solver error.
- **Mesh preconditions** *(A-17, added 2026-08-17)*: `Phase 2/mesh_build.check_mesh_preconditions`
  — (1) combinatorially closed (every bond in exactly 2 triangles, V−E+F=0; **periodic meshes only**)
  and (2) no inverted/negative-signed-area triangles. Nothing checked either before; the solver is
  **wrong**, not merely inaccurate, on a mesh that fails them. Gated by
  `Phase 5/verifications/test_designer_surface.py` [5].
- **A crystal-anchored check cannot gate the homogenisation.** On the regular uniform-k lattice
  `W ≡ 0` identically, so `C(s) = A(s)` and *any* error in how `W` is contracted is invisible —
  ν=1/3, E=2/√3 passes regardless. Scalar ν,E hides it too (they are weakly sensitive to the
  shear-shear entry). Only a **component-wise tensor** check on a mesh with `W ≠ 0` gates it: that is
  test [7], and it is why the 2026-08 shear-channel defect survived every earlier check.
- **The tensor oracle must be `physical_homog.energy_C` (or the virial), never
  `_common.sim_region_C6`** — the latter routes the sim's relaxation through the solver's own
  `_compute_actual_elastic_tensor`, so comparing against it is self-verification.
- **Do not** use the legacy area-weighted metric average — `test_cluster_Ceff.Ceff_nuE` is
  **tombstoned and raises `NotImplementedError`** (superseded; it biased ν on unequal-area meshes);
  use the physical (virial/energy) ground truth. Call sites survive only in the superseded legacy
  island (`verification_tools/README.md` §3), which consequently does not run — audit C-5 flagged
  the earlier "removed" wording as overstating a cleanup that was not finished.

---

## 7. Notes / caveats

- **Reference lengths `l₀`** are a differentiable input and set the reference metric `ḡ(l₀)` — they
  are **not** a free knob independent of the geometry. In the current flat solver `l₀` enters only as
  `k/l₀²`, so it *appears* degenerate with `k` — but that is a **flat-gauge (`ḡ=I`, geometry-pinned)
  artefact, not a true degeneracy** (moving `l₀` at fixed `ḡ=I` forces the geometry to change). True
  reference-metric / residual-stress physics (incompatible `l₀`) is not yet implemented.
- **vec3 → 4-index lift of `W` (implementation convention).** `W` is stored as a vec3 operator,
  `w[3·loc + k] = ∂δg_loc/∂Δg_k`, basis `[xx,xy,yy]` with **no factor-2 on shear**. Lifting it into
  the 4-index `W_{ijkl}` of `C = (1+W)ᵀA(1+W)` carries a **½ on a shear INPUT pair** (because
  `δg_ij = W_{ijkl} Δg_kl` sums over both `k` and `l`), and the identity in `(1+W)` must be the
  **symmetrised** delta ½(δ_ik δ_jl + δ_il δ_jk). The bare tensor `A` is fully symmetric and so is
  immune to both. Getting this wrong over-stiffens `C_xyxy` only — which is exactly the 2026-08 bug.
- `physical_units` E-scale is exact for periodic meshes; on open finite samples the boundary edges
  make it slightly approximate (`ν` is unaffected).
- Protected core: changes to `forward_solver_torch.py` are gated by `test_forward_solver.py` and
  the physical `verify_*` suite.
