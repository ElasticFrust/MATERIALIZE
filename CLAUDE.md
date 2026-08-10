# MATERIALIZE — project guide for Claude

> Project-specific companion to the global working charter (`~/.claude/CLAUDE.md`, auto-loaded).
> Covers only what is specific to MATERIALIZE: scope, architecture, and — once agreed — the
> maths/physics conventions and working habits. It does **not** restate the charter.
>
> **Session start — don't start cold.** Before acting each session, read: this file (§1–2), the
> relevant `Phase */PLAN.md`, `Phase 2/SOLVER_GUIDE.md`, and — for theory — `documentation/MATERIALIZE.md`.
> Keep code, docs, and this file mutually consistent; drift is a bug (there is a standing
> code-vs-docs consistency-sweep task).
>
> Precedence (from the charter): live instruction > this file > global charter > memory.

---

## 1. What the project is (scope)

MATERIALIZE does **inverse design of mechanical metamaterials**: planar triangulated spring
networks whose *effective* elastic response — Poisson ratio ν, modulus E, the full tensor C, and
*spatial patterns* of them — is prescribed and realised by choosing topology, node positions, and
per-bond stiffness k.

The forward map (network → response) is **many-to-one**: a target is realised by many different
networks. So the endpoint is **not a single optimised network** but a **learned model of the
solution space** — a generative / edit **neural network** (GNN edit-policy, VAE, with an
interpreter front-end mapping user intent → targets). Differentiable gradient-descent design
through the homogeniser (L-BFGS + adjoint) is the current *engine* — it produces solutions and
the labelled data the learned models train on — **not the end goal**. Of these, the **GNN
edit-policy is already prototyped** (`Phase 5/m2/`: model + trained checkpoint); the **VAE and
interpreter remain envisioned**.

Formalism: **incompatible / reference-metric elasticity** (the geometric "D2C" homogenisation).
The elastic strain is the metric change

```
    Δg = g − ḡ ,    g = FᵀF
```

with ḡ the **reference metric** encoded by the reference lengths ℓ₀. ḡ is *not* a "rest"
(zero-energy) state: when ḡ is incompatible it cannot be embedded, so the assembly cannot put
every bond at its reference length at once and the ground state carries **residual stress**. The
energy is quadratic in Δg (linear *in energy*, geometrically *exact* — not a linearisation).
Notation follows Grossman & Boudaoud, PRR 2026 (arXiv:2309.07844).

**The complete formalism is the frame; today's code is its first special case.** Implemented &
verified now: 2D, flat & compatible (ḡ = I), design of k — **fully** on periodic (PBC) unit cells,
and only **partially** on open / finite samples (Phase 5 is PBC-only; open-domain support in
Phase 2/3 is implemented and verified only in part). The program extends to non-flat geometry,
incompatible ḡ / residual stress (ℓ₀ becomes an *independent* design channel), and higher
dimensions — **order of these TBD** — all under the learned-model goal above.

**Verification (for now):** designs are cross-checked against an **independent physical
simulation**. This is a *temporary* oracle used *while the framework is being validated* — retired
once the solver is trusted, not a permanent per-design requirement.

---

## 2. Architecture & general plan

**Stability layering — dependencies point inward to the core; the core never references the layers
above it.** Phase numbering is historical, *not* a clean ladder: the live arc is
**Phase 2 (core) → Phase 3 (design) → Phase 5 (designer)**; Phase 4 is a retained remnant.

| layer | dir | role | stability |
|---|---|---|---|
| **Core (protected)** | `Phase 2/forward_solver_torch.py` | differentiable geometric homogeniser: (k, ℓ₀, geometry) → C(s), C_eff, ν, E, W; adjoint above 600 tri | **never modify without explicit approval**; gated by `test_forward_solver.py` + the physical `verify_*` suite |
| Inverse design | `Phase 3/` | objectives, `optimize` (L-BFGS), differentiable/adjoint path | verified truth; change with care |
| Designer (current) | `Phase 5/` | M1 search-based designer (topology + positions + k), **built on Phase 3's `inverse_design`**; M2 **GNN edit-policy prototyped** in `Phase 5/m2/` (model + trained checkpoint); VAE / interpreter still roadmap | active work |
| **Independent oracle** | `verification_tools/`, `Phase */verifications/` | full-PBC relaxation sim (`_common.sim_per_triangle_C6`, `physical_homog`) — a *different code path* from the solver | temporary validation oracle (see §1); retireable once the solver is trusted |
| Topology remnant | `Phase 4/` | **legacy, not a development stage**; its topology / point-cloud generators are reused (e.g. by `Phase 5/seeds.py: seed_from_phase4`). Directory name kept by deliberate decision | frozen remnant |
| Docs | `documentation/` | canonical `MATERIALIZE.md` (+pdf), `FUTURE_DIRECTIONS`, residual-stress & reference-metric notes | lockstep with code |
| Foundations (repo root) | `/` | theory backbone (`THEORY_NOTES.md`, `INTRINSIC_METRIC_SOLVE.md`, `derivation_edge_compatibility.pdf`, `ANALYTICAL_MODEL_STATUS.md`, `Tutorial.md`) and the D2C origin / legacy numpy path (`Disc_2_Cont_optimized.py`, also foam-mesh helpers) the Phase 2 core was ported from | foundational; legacy code frozen |

Principles specific here:

- **Protected core + independent oracle** are the two load-bearing invariants *for now*: the solver
  is the fast path, the simulation the truth it is checked against while being validated.
- **Compute separated from render:** designed networks are **saved**; figures *load* them and never
  re-optimise (random restarts ⇒ non-reproducible otherwise).
- **Artifacts live per-phase** with provenance (seed, config, commit); scratch stays disposable.
- **Root `README.md` is a deliberate stub — do not edit.** Project docs go in `documentation/`.

---

## 3. Conventions & working habits

> Maths & physics conventions below are **settled**. The remaining subsections (verification,
> environment, plotting, saving) are **under active development — not yet authoritative**: being
> co-developed section by section, in the complete formalism. Until a subsection is marked settled,
> **do not treat it as law and do not invent its content** — raise it and agree it first.

### Maths & physics conventions  *(settled 2026-08-09)*

Operative distillation; full statements in `documentation/MATERIALIZE.md §3–4, §10`,
`Phase 2/SOLVER_GUIDE.md §1,§4`; derivations `INTRINSIC_METRIC_SOLVE.md`, `THEORY_NOTES.md`.
Notation follows Grossman & Boudaoud, PRR 2026 (arXiv:2309.07844).

**Objects.**
- `q_e = Δx_e Δx_eᵀ` — rank-1 **edge carrier** (symmetric 2×2 ≡ vec3); Δx_e the reference edge
  vector, ℓ_e = |Δx_e|.
- `A(s) = Σ_{e∈s} (k_e/4ℓ_e²) q_e q_eᵀ` — per-triangle **bare tensor**, closed-form in geometry & k.
- **load ≡ Δg** — the applied **macroscopic (affine) strain**, a vec3; the input driving the
  response. Local strain under a load = (1+W)·Δg; local stress = A:(strain).
- `W(s)` — **strain-concentration operator**: δg(s) = W(s) Δg.

**Strain = metric change (general).** Δg = g − ḡ, g = FᵀF the actual metric, ḡ = ḡ(ℓ₀) the
**reference metric** encoded by reference lengths ℓ₀ (+ rest angles). ḡ need not be flat or
embeddable: a curved reference (e.g. a spherical ḡ in flat 2D) carries nonzero discrete Gaussian
curvature → geometric frustration → **residual stress**. Choosing coordinates so ḡ = I is a *gauge
choice*, not the definition. Voigt **vec3 = [xx,xy,yy], no factor-2 on shear**.

**Linear in ENERGY, exact in GEOMETRY.** The strain *measure* Δg = FᵀF − ḡ is geometrically
**exact**. The only linearisation is the **constitutive law** (energy quadratic in Δg), which makes
the response W linear. **Never** call the framework/W a "geometric linearisation."

**ℓ₀ defines ḡ — NOT a free knob, NOT degenerate with k.** ℓ₀ sets ḡ(ℓ₀); it is not independent of
the network geometry (moving ℓ₀ at fixed ḡ=I forces the edge vectors Δx to change). The solver's
apparent `k/ℓ₀²` degeneracy is an **artefact of the ḡ=I gauge with geometry pinned**, not physics.
Treat ℓ₀ as ḡ(ℓ₀) throughout.

**Reference split ḡ = ḡ_bg + δḡ(s) — [OPEN].** Residual stress needs the incompatible fluctuation
δḡ(s), entering the *energy* as an added RHS source A(s)·δḡ(s) (bare operators A, q_e untouched).
The decomposition has **two independent axes**: (i) a **scale filter** — macroscopic Δg vs sub-patch
reference; (ii) a **compatible/incompatible (St-Venant, `inc`) projection** — only the incompatible
part `inc(ḡ)≠0` sources stress. **In the current FLAT setting axis (ii) collapses to ḡ constant vs
non-constant** (any *compatible* ḡ is flattenable to constant by a coordinate choice); the full
compatible/incompatible distinction matters only for **non-flat / higher-D embeddings**. **Working
default = A** (ḡ_bg = uniform patch-mean; δḡ = systematic + disorder) — a perturbative finite-size
stand-in; the energy-optimal/ensemble background is the sharper target. **The correct finite-size
theory is an open research task** (see the reference-metric-split todo; MATERIALIZE.md §4;
FUTURE_DIRECTIONS #1). Note (C1)/(C2) below assume a **flat actual embedding**; a curved actual
space changes them.

**Intrinsic solve, full constraints (verified default).** Minimise energy over δg s.t. (C1) edge
compatibility Jδg=0; (C2) zero discrete Gaussian curvature 𝒞δg=0 (**omitting it is the classic
single-site mean-field ≈1.4× over-compliance error**); (C3) area-weighted normalisation M_S δg=0.
`method='intrinsic'` is the verified default; **legacy single-site mean-field / Woodbury and the
plain unweighted-mean normalisation are superseded — never in verification or plots.**

**Homogenisation = UNWEIGHTED mean of C(s)** = (1/N)Σ_s (1+W)ᵀA(s)(1+W), the physical
(energy=virial) modulus. Area weight is used *only* in C3, never in the final average (it would bias
ν on disordered/anisotropic meshes).

**One solve → the whole response operator ("direct").** W(s) is the complete linear map; from it
fall C_eff (all components), ν(θ),E(θ) at every angle, per-triangle strain/stress under any load —
no per-direction solve, no global-displacement adjoint.

**Units.** ν always physical/scale-invariant; E, C_eff internal-scale unless `physical_units=True`
(rescale 8N/Σ_s S_s → regular triangular lattice ν=1/3, E=2/√3).

**ν–E coupling (nontrivial, not absolute).** At **fixed topology** ν and E are contractions of the
same C_eff and generically coupled — but nontrivially, not a rigid lock. **Changing topology can
(at least partly) decouple them.** Do not assert they are never independently controllable.

**Realizability & Poisson bounds.**
- A single homogeneous 2D tile = **≤6-parameter tensor** (fewer under symmetry). A target ν(θ),E(θ)
  is realizable by one tile iff it is the response of some positive-definite C in that family;
  **spatial patterning C*(x) enlarges the set.**
- **1/E(θ) is a quartic form in n → bounded at 4θ. ν(θ), E(θ) are rational (ratio/reciprocal of
  quartics) → NOT 4θ-bounded**; they can be sharp / high-harmonic (any anisotropic crystal shows this).
- **Poisson bounds:** isotropic 2D −1<ν<1; isotropic 3D −1<ν<½; **anisotropic — no bound**, ν(θ)
  arbitrarily large ± (Ting & Chen 2005), limited only by positive-definiteness of C.
- *Empirical (flag as such):* cos4θ amplitude ceilings, "hard" profiles — network-family/size
  properties, not fundamental limits.

**Honest limits.** Linear constitutive (no strain-stiffening); C is the tangent about the reference.
**Mechanism** = a zero-/near-zero-energy floppy deformation mode (energy Hessian near-singular
beyond rigid modes); near one, W is ill-conditioned and the linear read-back diverges from the true
nonlinear relaxation — a soft-eigenvalue issue, **not** a linearisation artefact — hence every
design is checked against the independent simulation. Bulk/periodic by construction; open-boundary
questions use a separate nodal solve, for verification only.

### Verification discipline  *(settled 2026-08-10)*

Full suite: `documentation/MATERIALIZE.md §9`, `Phase 3/verifications/README.md`,
`Phase 2/SOLVER_GUIDE.md §6`.

Three DISTINCT quantities — keep them separate:

- **Two different code paths — never self-verify.** Design path = the differentiable *linear* metric
  solve `forward()`; verification path = an **independent** NumPy periodic *nonlinear* relaxation →
  virial/energy homogenisation (`verification_tools/physical_homog.py`: `virial_nuE`, `energy_nuE`,
  which must agree).
- **Define the ensemble** (usually natural). Verification quantities are computed over an *explicit*
  ensemble: a **crystal = the single structure** (or a parametric family, e.g. the Bravais sweep) —
  no randomness ⇒ nothing to average; a disordered family = fixed parameters (η, N) over seeds.
- **(A) Solver accuracy vs physics — matched realizations, averaged.** The honest measure is
  solver−sim on the **same network** (same geo + k), averaged over the ensemble — bulk ν(θ),E(θ) and
  **per-triangle δg/C(s)** (correlation ≳0.99 bulk / ≳0.97 per-triangle). When same-realization
  matching is unachievable, aggregate by shared parameters: compare ensemble-**averaged** responses
  (⟨solver⟩ vs ⟨sim⟩ by network kind) — equivalently a difference of averages.
- **Per-design relative gap** (the single-realization instance of A): `gap =
  max_θ|ν_sim−ν_solver|/(|ν_sim|+ε_ν) + max_θ|E_sim−E_solver|/|E_sim|`, ε_ν=0.05 (at ν→0 the absolute
  floor is `gap_tol·ε_ν`, so ε_ν and the tolerance are coupled); flag `gap>gap_tol≈0.05`. Least
  validated on **topologies far from those already checked** — watch the gap there, never silently
  keep a large-gap design. (Non-Delaunay is *not* itself unvalidated — the η-disorder family reaches
  non-Delaunay connectivity and is verified; *novelty* is the flag.)
- **(B) Design gap — achieved vs target, on BOTH solver and sim.** Report achieved−target against the
  **solver** (the optimiser's own residual — always available) *and* the **independent sim** of the
  design (`target_err_sim` — the stronger *physical* check, but only when a correct sim exists; e.g.
  not yet for incompatible-ḡ / curved designs). Per-triangle for local objectives, bulk for global.
- **Per-triangle checks by default, but watch for redundancy** — in both A and B, resolve the
  per-triangle field where it adds information beyond bulk; drop it where it proves redundant.
- **The oracle is TEMPORARY & LIMITED.** Retireable once the solver is trusted; **flat-compatible
  only** (sim assembles from stress-free ḡ=I, no δḡ — *not* a ground truth for residual-stress /
  incompatible-ḡ / curved designs, which need a generalised prestressed virial, FUTURE_DIRECTIONS #1,
  not yet available); near a mechanism it is today's check, expected *resolved* later.
- **Open-boundary** designs (cut-and-stretch, finite specimens: `two_region`, `_common.open_stretch`)
  are verified by a **separate classical nodal solve** — distinct from the PBC virial oracle, for
  verification only, never in the design loop. *(The open-boundary verification suite is incomplete —
  to be completed; tracked todo.)*
- **Sanity gate first.** Regular triangular lattice → **ν=1/3, E=2/√3** on *both* solver and sim
  (`Phase 5/verifications/sanity.py`); run before trusting any run.
- **Physical ground truth = virial/energy (unweighted mean of C(s)).** The legacy **area-weighted
  metric average** (`Ceff_nuE`) is physically wrong (biases ν on unequal-area meshes) and has been
  **removed** (tombstoned 2026-08-10; git-restorable).
- **Protected core is gated.** `Phase 2/forward_solver_torch.py` changes require passing
  `test_forward_solver.py` **and** the physical `verify_*` suite; not done until dependents are
  re-verified and blast radius checked.
- **"Verified" (charter ladder), strongest first:** independent code path (the sim) → prior detailed
  results → a known value (ν=1/3, E=2/√3) → an analytical argument. Passing tests alone ≠ verified.
- **Regressions:** `test_forward_solver.py` (crystal ν, **gradients + large-N adjoint**);
  `test_inverse_design.py` (16 tests). Use the anaconda python (see environment subsection).

### Remaining subsections — under active development
- **Environment & reproducibility** *(pending)* — anaconda python (Store stubs broken); float64
  throughout; PBC; `ANG` grid; seeds always injected.
- **Plotting & output conventions** *(pending)* — the canonical network draw is the
  **tiled-continuous, cropped view** (bonds crossing the periodic boundary render continuously, no
  non-physical gaps; cf. `Phase 5/gallery.py: draw_one_tiled`), to *replace* the stub-producing
  variants; square plot regions; **means-together / spread-separate** (all group means in one plot,
  each group's mean±σ in its own subplot) — standing, with a **formalized "combine-all" exception**
  still to be specified; ν(θ), E(θ) panels per designed network; per-panel high-DPI reusable
  elements; load-don't-reoptimise. Requires a **shared `plotting.py`** (single source of truth)
  that unifies all plot families and retires the current several-variants state; concrete style
  values catalogued in `documentation/MATERIALIZE.md §10`.
- **Saving / docs habits** *(pending)* — experiment persistence (script + outputs + plot in-repo);
  module `.md` lockstep with code.
