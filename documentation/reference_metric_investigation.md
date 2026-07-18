# Reference metric (ḡ) and residual stress — investigation notes

Status: **exploratory / work in progress.** This records what we built and learned about carrying a
per-triangle **reference metric** `ḡ_s` through the D2C forward solve, and where the current
(linear) formulation succeeds and where it falls short. It complements the theory note
`residual_stress_note.pdf/.tex`; here the emphasis is the numerical findings.

All code lives in `Phase 3/verifications/reference_metric/`. The protected Phase 2 core
(`Phase 2/forward_solver_torch.py`) was **not** modified — the solver here reuses its operators.

---

## 1. The solver — `forward_solver_dgbar.py`

Clean, documented reference-metric forward-like solver (written in the style of the Phase 2 core,
marked as a TODO scaffold). Each triangle carries its own reference metric `ḡ_s` (rest state).

Decomposition: `g_s = g + δg_s`, `ḡ_s = ḡ + δḡ_s`, `Δg = g − ḡ`, mismatch `m_s = Δg + δg_s − δḡ_s`,
ansatz `δg_s = W_s Δg`, energy `E = ½ Σ_s m_s : A_s : m_s`,
`A_s = Σ_e (k_e/4ℓ0,e²) q_e q_eᵀ`, `ℓ0,e² = ḡ_s : q_e`.

`ḡ` enters **two ways**:

1. **Operator** — the rest length `ℓ0,e² = ḡ_s:q_e` sets `A_s = A(ḡ)` (longer rest → softer bond,
   `A ∝ k/ℓ0²`). Implemented by feeding `rest_lengths = √(ḡ_s:q_e)` to the Phase 2 core, which then
   builds `A(ḡ)`, solves `W`, and homogenises — bit-identical to the validated solver; `ḡ=I`
   reproduces the ordinary solve exactly.
2. **Source** — `δḡ_s` enters the mismatch as a `Δg`-independent term → a **residual stress**. We
   solve the relaxed prestress fluctuation `δg⁰` (energy minimum at `Δg=0` over the compatible
   field) and report `σ⁰_s = A_s(δg⁰_s − δḡ_s)`.

---

## 2. ν and E are not covariant — the covariant readout

`ν` and `E` are **not** coordinate-invariant scalars. When `ḡ ≠ I`, reading them with ordinary
(Euclidean) directions gives a frame-dependent, wrong answer. The covariant moduli contract the
compliance with **ḡ-orthonormal** directions and carry a `√det ḡ` volume factor:

```
ν(m) = − S(n,n,m,m) / S(m,m,m,m),        ḡ(m,m)=ḡ(n,n)=1, ḡ(m,n)=0
E(m) = 1 / [ √det ḡ · S(m,m,m,m) ],      S = C_eff⁻¹
```

**Result (`cov_poisson.py`, fig `cov_poisson.png`).** The two descriptions of the *same* material —
(a) a network physically elongated (ψ=1.3), `ḡ=I`; (b) a regular network carrying the full
`ḡ = diag(1, 1.69)` — give ν(θ), E(θ) that **disagree** under the flat readout but **collapse onto a
single curve** under the covariant readout (max difference ~`1e-15`). Poisson is invariant under
conformal (isotropic) maps but not anisotropic ones, which is exactly why the flat readout fails
off the identity.

---

## 3. Residual stress from an incompatible (spherical) reference

A **constant** `ḡ` is compatible (flat): it is equivalent to a coordinate change and its prestress
relaxes away. The interesting case is an **incompatible** `ḡ` — nonzero Gaussian curvature — which
cannot be laid flat and stores a non-relaxable residual stress.

Test (`sphere_reference.py`, fig `sphere_reference.png`): conformal spherical reference
`ḡ = λ(r) I`, `λ = 1/(1 + K r²/4)²`, `K = 1/8²`. The solver returns a spatially structured residual
stress with a nonzero **deviatoric** part (rms ≈ 0.033) — the signature of frustration that a
compatible (uniform) reference does not have.

**Validation against simulation (`sphere_sim_compare.py`, fig `sphere_sim_compare.png`).** We relax
a *real* prestressed spring network (rest lengths from the same `ḡ`) and read its residual pressure.
The intrinsic and simulated fields agree at **correlation 0.993**, with a best-fit scale of exactly
**½** — the known metric-vs-engineering strain convention (`Δg = δ(L²) = 2·ε·L²`). So the residual
**stress** is correct.

---

## 4. The response is NOT captured — geometric stiffening (the gap)

Do the intrinsic solver and a prestressed simulation give the same **elastic response**?
(`sphere_response_compare.py`, fig `sphere_response_compare.png`.) **No.**

| | ν | E |
|---|---|---|
| intrinsic (linear D2C) | +0.33 | 1.38 |
| simulation (full tangent) | +0.03 | 1.53 |

The simulation carries an isotropic residual tension; that prestress **geometrically stiffens** the
network (a real bond's tangent stiffness is `k n̂n̂ + (t/L)(𝟙−n̂n̂)`; the `(t/L)` prestress term is
absent from a constant `A`), collapsing ν from ~1/3 to ~0.03. The linear D2C has a fixed `A` and
cannot see it.

### Why the linear equation cannot produce it (settled numerically)

- **The complete equation is one KKT solve** (`Δg` and `δḡ` together). Solving it gives *exactly*
  `δg = W·Δg + δg⁰` (max residual ~`8e-9`) — response plus prestress, superposed.
- **The response `W` is `δḡ`-independent to `1.2e-14`**: solving for `∂δg/∂Δg` with vs without the
  reference gives the identical operator. So `C_eff` is blind to the prestress by construction.
- **Ansatz choice does not help.** Rewriting `δg = W(Δg − δḡ)` (the eigenstrain form) leaves the
  `W`-determining part `A(𝟙+W)+CᵀΛ=0` unchanged — the extra `δḡ` term is `Δg`-independent, so it
  moves only the prestress, not `W`. It also makes the prestress *worse* (a local `−Wδḡ` is a crude
  stand-in for the non-local relaxed source). The genuine difference between the two ansätze is only
  at `Δg=0`: the previous one forces `δg⁰=0` (raw stress), the new one relaxes it.

**Conclusion.** The stiffening is a genuinely *non-linear* effect. It lives either in keeping the
constitutive law beyond quadratic in the metric strain (the cubic `−k/16ℓ0⁴ e³` term from
`L=√(g:q)`), or — as incompatible elasticity does with a *quadratic* energy — in the **non-linear
compatibility** constraint (`g` must be realizable/flat; Gauss / Föppl–von Kármán `[w,w]`). Our
solver linearizes that compatibility, which is precisely why the prestress decouples from the
response.

---

## 5. TODO

- Add the **geometric-stiffness operator** `G(σ⁰)` so the tangent is `C_tangent = A + G(σ⁰)` — i.e.
  the `σ⁰:∇∇` / second-order-compatibility term. This is the one piece needed for the *response* of
  a residually-stressed network; the residual *stress* itself is already correct.
- Decide the intended bond model (fixed spring constant `k` → 2D modulus scale-invariant; vs
  material rods `k ∝ 1/L` → modulus changes with reference length).
- Self-consistent reference: renormalize the effective medium by its own (coarse-scale, curvature-
  set) prestress.

---

## 6. Files

| file | what |
|---|---|
| `forward_solver_dgbar.py` | the reference-metric solver (operator + source, covariant readout) |
| `cov_poisson.py` / `.png` | covariant ν(θ), E(θ): two descriptions collapse |
| `sphere_reference.py` / `.png` | spherical incompatible reference → residual stress field |
| `sphere_sim_compare.py` / `.png` | residual stress vs direct simulation (corr 0.993) |
| `sphere_response_compare.py` / `.png` | response vs simulation — the geometric-stiffening gap |
