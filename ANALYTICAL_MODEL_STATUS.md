# The Forward Solver — What It Does and How It Works

**Status: current.** This document describes the canonical forward solver,
`Phase 2/forward_solver_torch.py`. It supersedes the earlier status notes that
described the single-site mean field as the default; that mean field is now the
**legacy** `method='woodbury'` path, kept only for reference.

Companion math:
- `INTRINSIC_METRIC_SOLVE.md` — the full derivation of the default solver
  (energy, all three constraints incl. the curvature/angle term, the KKT system,
  and the proof that the metric solve equals the configuration/simulation solve).
- `THEORY_NOTES.md` — why the single-site mean field fails and why the added
  constraints fix it.
- `derivation_edge_compatibility.pdf` — the edge-compatibility (length) derivation
  in Grossman & Boudaoud (PRR 2026, arXiv:2309.07844) notation. The angle/curvature
  and area-weighted-mean corrections are written up in `INTRINSIC_METRIC_SOLVE.md`.

---

## 1. What the solver computes

Given a 2D spring network — a triangulated mesh (fixed node positions + topology)
where every triangle edge is a spring with rigidity `k_e` and rest length `ℓ0_e` —
the solver computes the **macroscopic (homogenised) linear-elastic response**:

- the effective elastic tensor `C_eff` (6 independent components),
- Poisson's ratio `ν`,
- Young's modulus `E`.

It is **differentiable** with respect to the per-edge `k` and `ℓ0`, which is what
makes gradient-based inverse design and ML-label generation possible.

Entry point:

```python
import sys; sys.path.insert(0, 'Phase 2')
import Disc_2_Cont_optimized as D2C
from forward_solver_torch import from_triangulation

tri = D2C.generate_foam_points(size=(4, 4), eta=0.2)
solver, rigidities, rest_lengths = from_triangulation(tri)   # defaults: k=1, ℓ0=actual length
result = solver(rigidities, rest_lengths)                    # method='intrinsic' by default
print(result['poisson'].item(), result['young'].item())
```

---

## 2. The physics (metric / incompatible-elasticity picture)

Each triangle `s` carries a symmetric 2×2 metric `g(s)`, decomposed as

```
g(s) = ḡ  +  Δg  +  δg(s)
```

- `ḡ` = reference (rest) metric (here `ḡ = I`),
- `Δg` = the uniform **affine** macroscopic loading (the applied strain),
- `δg(s)` = the **non-affine fluctuation** — the unknown we solve for.

The linear response is `δg(s) = W(s) Δg`, where `W(s)` is the per-triangle
strain-concentration tensor. The per-triangle stiffness is the spring-energy
metric Hessian (no area prefactor):

```
A(s)  ↔  H_s = Σ_{e∈s} (k_e / 4 ℓ_e²) q_e q_eᵀ ,     q_e^{μν} = Δx_e^μ Δx_e^ν
```

with `Δx_e` the reference edge vector of length `ℓ_e`. We minimise the elastic
energy over the fluctuation field:

```
E[δg] = ½ Σ_s A(s) (Δg + δg(s))²
```

subject to the three constraints below.

---

## 3. The three constraints (this is the core of the solver)

The default `intrinsic` solve enforces **all three**, all ON by default. Turn any
off via `forward(use_kkt=..., use_angle_kkt=..., area_weighted=...)`.

### 3.1 Edge compatibility — "lengths" (multiplier λ)
Adjacent triangles sharing reference edge `Δx_e` must agree on that edge's length:

```
[δg(s1) − δg(s2)]_{μν} Δx_e^μ Δx_e^ν = 0        ⇔   (J δg)_e = 0
```

one scalar per interior edge. `J` is the `E_int × 3N` edge-compatibility matrix.
This is the Grossman & Boudaoud edge constraint (`derivation_edge_compatibility.pdf`).
Built in `_build_intrinsic_constraints` → `_J_edge_sp`; toggle `use_kkt` (default True).

### 3.2 Curvature / vertex-angle — "angles" (multiplier κ) — **added; absent from G&B**
Edge-length agreement alone is *not* full compatibility: a field can match every
shared edge length yet fail to close into a flat triangulation (the vertex angles
don't sum correctly). The missing condition is **zero discrete Gaussian curvature**
(= zero disclination density = discrete St-Venant `inc(δg)=0`). For each interior
vertex `v`:

```
(C δg)_v := Σ_{s∋v} a^{(s)}_v{}^{μν} δg(s)_{μν} = 0 ,   a^{(s)}_v = ∂θ^{(s)}_v / ∂g(s)
```

one scalar per interior vertex; `θ^{(s)}_v` is the interior angle triangle `s`
contributes at `v`. `κ_v` is the discrete analogue of an Airy stress potential.
Without this term the solve minimises over a space ~`n_int` dimensions too large
(spurious disclinations), producing the over-compliant overshoot that the old
mean field showed. **The size of that overshoot is NOT a constant** (re-derived
2026-08-17): it is strongly mesh-dependent — measured 1.015 (frozen η=0.15), 1.038
(η=0.25), 1.120 (VD a=−2), 1.166 (η=0.35), 1.642 (VD a=+5), 228 (VD a=+10, near a
mechanism); median 1.14. Only the DIRECTION is universal (omitting C2 is always
over-compliant). The “≈1.4×” once quoted here came through the tombstoned
area-weighted `Ceff_nuE` (audit A-7) — quote a range, never that constant. Built in `_build_intrinsic_constraints` → `_C_curv_sp` (tensor
convention, **no** engineering-Voigt `/2`); toggle `use_angle_kkt` (default True).

> Note: the legacy `woodbury` path implemented the same physics as a fragile
> angle-Gram correction that **diverged** at high η on near-degenerate triangles.
> The `intrinsic` path's sparse curvature operator is the stable replacement.

### 3.3 Normalisation — the area-weighted mean (multiplier χ) — **corrected**
G&B impose the *unweighted* `Σ_s δg(s) = 0`. The correct condition — the one the
simulation satisfies, and the one *implied by compatibility* — is the
**area-weighted** mean:

```
Σ_s S_s δg(s)_{μν} = 0          (3 scalar conditions),   S_s = area of triangle s
```

Written `M_S δg = 0`. Built → `_M_S_sp`; toggle `area_weighted` (default True).

---

## 4. The system solved

Stacking `δg ∈ ℝ^{3N}` with `A = blkdiag(A(s))` **block-diagonal** (the global
coupling lives entirely in the constraints, not in an `A−B` mean-field operator),
the loading-independent response `W` solves the KKT saddle system:

```
[ A    Jᵀ   Cᵀ   M_Sᵀ ] [ W ]   [ −A ]
[ J    0    0    0    ] [ Λ ] = [  0 ]
[ C    0    0    0    ] [ K ]   [  0 ]
[ M_S  0    0    0    ] [ X ]   [  0 ]          δg(s) = W(s) Δg
```

The homogenised tensor is the **area-weighted** average

```
C_eff = (1/Σ_s S_s) Σ_s S_s (1 + W(s))ᵀ A(s) (1 + W(s))
```

from which `ν` and `E` follow. Full derivation and the proof that this equals the
node-space equilibrium `K u = −f_aff` (the simulation) are in
`INTRINSIC_METRIC_SOLVE.md` §5–6.

---

## 5. Methods, differentiability, and sizes

`forward(rigidities, rest_lengths=None, method='intrinsic', area_weighted=None,
use_kkt=None, use_angle_kkt=None)`

| `method` | What | Defaults | Use |
|---|---|---|---|
| `'intrinsic'` (default) | block-diagonal `A` + edge + curvature + area-weighted mean; the metric solve that reproduces the simulation | `area_weighted=True, use_kkt=True, use_angle_kkt=True` | production |
| `'woodbury'` (legacy) | original G&B `(A−B)` single-site mean field, optional edge/angle KKT projection | `area_weighted=False, use_kkt=True, use_angle_kkt=False` | reference / regression only |

Dense vs sparse (constant `INTRINSIC_DENSE_MAX = 600`):

- **≤ 600 triangles** → differentiable dense torch Woodbury (`_woodbury_solve_aw`
  with the area-weighted χ-elimination `δA(s)=A_s−[A]·S_s/[S_s]`). Carries
  autograd gradients — usable for inverse design and ML labelling.
- **> 600 triangles** → NumPy sparse saddle solve (`_intrinsic_solve_W`).
  Forward-only (no gradients). Both reproduce the simulation.

---

## 6. Validation status

The intrinsic solve reproduces the PBC (periodic) spring-network simulation:

- **Per-triangle response `δg(s)`**: correlation ≈ 1.0 with the simulation across
  geometric disorder **through η = 0.5**, and across rigidity contrast (incl. signed
  VD patterns), soft inclusions, and a 50×50 two-soft-circles test (corr ≥ 0.999 in
  all three macro-strain directions, incl. irregular Poisson–Delaunay meshes).
- **Homogenised `ν`, `E`**: match the simulation to 3–4 digits over the same range,
  where the single-site mean field collapses (ν/E off by tens of percent or wrong
  sign at high η).

(Numerical tables are in `INTRINSIC_METRIC_SOLVE.md` §8. The verification scripts
that produce them live in `verification_tools/` — see `verification_tools/VERIFICATION_SUMMARY.md`
and `verify_solver_sweep.py`.)

---

## 7. Known limitations / open work

1. **Extreme disorder beyond the validated range.** The area-weighted normalisation
   and the curvature constraint are the first two orders of the exact area law
   `Σ_s S_s det(F_s) = det(F) Σ_s S_s`. The next (O(δ²)) term is not yet carried, so
   very large deformations past the validated regime will eventually need it (or the
   cluster solve, which never linearises).
2. **Spatially-correlated / anisotropic rigidity.** All three constraints are
   strictly local (per-edge, per-vertex). Rigidity fields whose correlation length
   exceeds that locality (e.g. bimodal-checkerboard clusters, orientation-dependent
   `k(θ)`) are the known failure cases and need a cluster-radius correction not yet
   in the solver.
3. **No gradients above 600 triangles.** Large-mesh inverse design / ML labelling
   currently needs the dense path or a future differentiable sparse route.
4. **Residual pre-stress / non-trivial reference metric** (`ℓ0 ≠ actual length`)
   is implemented but not yet verified against the simulation.
5. **Test coverage.** A minimal regression test lives in `Phase 2/test_forward_solver.py`
   (ν(η=0)=1/3, autograd-vs-FD gradcheck, legacy path). The fuller sim-parity suite is
   in `verification_tools/` (`verify_solver_sweep.py` and friends).
