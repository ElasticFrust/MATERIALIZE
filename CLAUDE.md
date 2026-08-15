# MATERIALIZE — project guide for Claude

> Project-specific companion to the global working charter (`~/.claude/CLAUDE.md`, auto-loaded).
> Covers only what is specific to MATERIALIZE: scope (§1), architecture (§2), and the settled
> maths/physics conventions & working habits (§3). It does **not** restate the charter.
>
> **Session start — don't start cold.** Before acting each session, read: this file (§1–3), the
> relevant `Phase */PLAN.md`, `Phase 2/SOLVER_GUIDE.md`, and — for theory — `documentation/MATERIALIZE.md`.
> Keep code, docs, and this file mutually consistent; drift is a bug (there is a standing
> code-vs-docs consistency-sweep task).
>
> **Entry points:** to run a design, `Phase 5/designer.py` → `design(nu_target, E_target, tag)`;
> reusable API in `Phase 5/PLAN.md §1` and `MATERIALIZE.md §11`. M2 (GNN edit-policy): `Phase 5/m2/M2.md`.
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
| Core-adjacent | `Phase 2/metric_ops.py`, `mesh_build.py`, `solver_build.py` | the NumPy side of the core: metric/tensor ops in the solver's conventions (`vec3`, `tri_metric_change`, `bare_tensor`); periodic + open mesh construction (`build_geometry`, `set_VD`, `kkt_from_tri_bond`, `clean_tri`, `build_open_mesh`); `make_solver`/`_mount` | same layer & same gate as the core, **not** the protected file; `DesignProblem`'s constructors are built on them, so treat as verified truth |
| Inverse design | `Phase 3/` | objectives, `optimize` (L-BFGS), differentiable/adjoint path | verified truth; change with care |
| Designer (current) | `Phase 5/` | M1 search-based designer (topology + positions + k), **built on Phase 3's `inverse_design`**; M2 **GNN edit-policy prototyped** in `Phase 5/m2/` (model + trained checkpoint); VAE / interpreter still roadmap | active work |
| **Independent oracle** | `verification_tools/physical_homog.py` + `sim_assembly.py`; `Phase */verifications/` | full-PBC relaxation sim (`physical_homog`, fed by `sim_assembly.assemble_K_faff`) — a *different code path* from the solver. The rest of `verification_tools/` is **experiment scripts, not a library** | temporary validation oracle (see §1); retireable once the solver is trusted |
| Topology remnant | `Phase 4/` | **legacy, not a development stage**; its topology / point-cloud generators are reused (e.g. by `Phase 5/seeds.py: seed_from_phase4`). Directory name kept by deliberate decision | frozen remnant |
| Docs | `documentation/` | canonical `MATERIALIZE.md` (+pdf), `FUTURE_DIRECTIONS`, residual-stress & reference-metric notes | lockstep with code |
| Foundations (repo root) | `/` | theory backbone (`THEORY_NOTES.md`, `INTRINSIC_METRIC_SOLVE.md`, `derivation_edge_compatibility.pdf`, `ANALYTICAL_MODEL_STATUS.md`, `Tutorial.md`) and the D2C origin / legacy numpy path (`Disc_2_Cont_optimized.py`, also foam-mesh helpers) the Phase 2 core was ported from | foundational; legacy code frozen |

Principles specific here:

- **Protected core + independent oracle** are the two load-bearing invariants *for now*: the solver
  is the fast path, the simulation the truth it is checked against while being validated.
- **Dependencies point inward, and the oracle is off to the side.** `Phase 5 → Phase 3 → Phase 2`;
  nothing in Phase 2/3 may import from `verification_tools/`, which is *temporary and retireable* —
  a design layer rooted in it could not survive its retirement. `verification_tools/` is
  deliberately **not** on `Phase 3/inverse_design.py`'s `sys.path`, so the inversion cannot return
  silently. *(A-7b, fixed 2026-08-15: `inverse_design.py` had been importing its mesh construction,
  its constraint topology and its **solver construction** from four scripts in there.)*
  Corollary, and the deeper reason: **the oracle must share NO code with the design path** —
  otherwise "checked against the sim" degrades into checking the code against itself (cf. the shear
  defect, §3 verification discipline). `physical_homog` therefore keeps its own `DELTA`/`MODES`
  rather than importing them; that duplication is **deliberate — do not "fix" it**.
- **Compute separated from render:** designed networks are **saved**; figures *load* them and never
  re-optimise (random restarts ⇒ non-reproducible otherwise).
- **Artifacts live per-phase** with provenance (seed, config, commit); scratch stays disposable.
- **Root `README.md` is a deliberate stub — do not edit.** Project docs go in `documentation/`.

---

## 3. Conventions & working habits

> All subsections below are **settled** (see the date on each) and authoritative — co-developed in
> the complete formalism. **NB:** some canonical docs referenced for "full statements" still carry
> pre-correction wording (e.g. the ℓ₀-"degeneracy" and cos4θ framings) pending the code-vs-docs
> consistency sweep; **on any conflict, this file's settled statement wins.**

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
no per-direction solve, and no adjoint of a global displacement equilibrium solve (the classical
inverse-design adjoint). *(Distinct from the sparse-KKT **gradient** adjoint that backprops through
this solve above 600 tri — §2.)*

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
- **BUT the crystal gate is structurally blind to the homogenisation.** On the regular uniform-k
  lattice **W ≡ 0 identically**, so C(s)=A(s) and *any* error in how W is contracted passes it — as
  does any scalar-ν,E check, since ν,E are weakly sensitive to the **shear-shear** entry. A defect
  can therefore live in C_xyxy alone, growing with |W|, and show up only in ν(θ) off-axis and in
  anisotropic designs. **A homogenisation claim needs a COMPONENT-WISE tensor check on a mesh with
  W≠0** (`Phase 2/test_forward_solver.py` [7]). *(2026-08: exactly this went unseen for months —
  the shear entry was over-stiff by 29–90% on the anisotropic designs, and reported ν(θ) dips to
  −0.30 where the truth was +0.13.)*
- **The tensor oracle is `physical_homog.energy_C` / the virial — NEVER `_common.sim_region_C6`.**
  `sim_region_C6` takes the sim's relaxation but pushes it through the solver's own
  `_compute_actual_elastic_tensor`, so it shares the contraction: comparing against it is
  **self-verification** and hides precisely the defects a tensor check exists to catch.
- **The sim self-screens near-singular geometry.** The independent sim is dense scipy/LAPACK — a
  near-singular geometry (sliver / near-zero-area triangles) would **hard-crash it (native segfault,
  UNCATCHABLE in Python)**. So the sim entry (`physical_homog.relax` / `energy_nuE`, hence
  `_common.sim_per_triangle_C6` / `virial_nuE` / `sim_region_*`) calls `require_healthy_mesh` and
  **raises a catchable `UnhealthyGeometryError`** instead — callers just `try/except` it, no
  pre-screening needed. (The crash is **sim-only**; the torch solver degrades gracefully and needs no
  guard.) The finer "is the RESPONSE physical" check (solver ν,E finite) stays **caller-side** (e.g.
  the designer) — it needs the solver, on which the sim must not depend. *(TODO #2.6:
  `_common.open_stretch` open-boundary solve not yet guarded.)*
- **Physical ground truth = virial/energy (unweighted mean of C(s)).** The legacy **area-weighted
  metric average** (`Ceff_nuE`) is physically wrong (biases ν on unequal-area meshes) and has been
  **removed** (tombstoned 2026-08-10; git-restorable).
- **Protected core is gated.** `Phase 2/forward_solver_torch.py` changes require passing
  `test_forward_solver.py` **and** the physical `verify_*` suite; not done until dependents are
  re-verified and blast radius checked.
- **"Verified" (charter ladder), strongest first:** independent code path (the sim) → prior detailed
  results → a known value (ν=1/3, E=2/√3) → an analytical argument. Passing tests alone ≠ verified.
- **New functionality gets a test.** Core and any new capability get a structured test — compared vs
  an alternative code path or a known value — alongside the existing regressions.
- **Regressions:** `test_forward_solver.py` (crystal ν, **gradients + large-N adjoint**);
  `test_inverse_design.py` (16 tests). Use the anaconda python (see environment subsection).

### Environment & reproducibility  *(settled 2026-08-10)*

- **Python:** run everything with `C:\Users\doron\anaconda3\python.exe` (verified stack: numpy 2.3.5,
  scipy 1.16.3, torch 2.12.1+cpu). **`python`/`python3` on PATH are broken Windows Store stubs — never
  use them.**
- **float64 everywhere.** `torch.set_default_dtype(torch.float64)` is REQUIRED — the whole solver
  stack is float64 (the constrained saddle/KKT solves need the precision).
- **Periodic (PBC) by default in Phase 5** — ν(θ),E(θ) is then a clean bulk property. Canonical angle
  grid `ANG = np.linspace(0, np.pi, 37)`; directional targets are length-37 arrays (scalar → flat).
- **Phase 5 import preamble (verbatim):** set `REPO`, `sys.path.insert` Phase 3/verifications, then
  `import _common as C` (this wires the rest of sys.path + the solver stack) **before**
  `from inverse_design import …`. Path-depth: `REPO=join(dirname,'..')` for scripts directly in
  `Phase 5/`; `'..','..'` for `Phase 5/verifications/`.
- **Seeds always.** A seed is an explicit input everywhere; if none given, auto-generate, **output
  it**, and note it was auto-generated. Isolate randomness at explicit injection points (a seeded
  generator passed down — no global RNG).
- **Save-then-load, never re-optimise at plot time.** Every `optimize()` saves the network once
  (`save_network`: k + C6 + meta); figures **load** it and never re-optimise (random restarts ⇒
  non-reproducible). *(Full persistence conventions in the saving subsection.)*
- **Docs build:** `python documentation/_build_pdf.py [NAME…]` regenerates the doc PDFs
  (`MATERIALIZE`, `FUTURE_DIRECTIONS`) from their `.md` (markdown → self-contained HTML with local
  MathJax → headless Edge `--print-to-pdf`; needs the anaconda `markdown` pkg + Edge). Docs ship as
  **both `.md` and `.pdf`** — rebuild the PDF after editing a `.md` (lockstep). Doc *conventions* →
  saving/docs subsection.
- **Traceability:** every artifact traceable to (code commit, config, seed).

### Plotting & output conventions  *(settled 2026-08-13)*

Policy here; **implemented in the root `plotting.py` module (the single source of truth).** **Every
plotting task MUST start from `plotting.py`** — import it and use its primitives (`draw_network`,
`draw_field`, `plot_directional`, `plot_means_spread`, `save_element`/`save_fig`, `montage`,
`STYLE`); never roll a one-off `plot_*`/`draw_*`. Missing a primitive ⇒ add it *there* and update
this policy. It is pure-render (numpy + matplotlib; callers pass already-loaded data).

- **Square plot regions always** for spatial/network panels.
- **Canonical network draw = tiled-continuous, cropped.** Tile the periodic cell (reps×reps) and crop
  to the central cell so boundary-crossing bonds render continuously (no non-physical gaps);
  `Phase 5/gallery.py: draw_one_tiled` is the reference. REPLACES the stub variants (`draw_network`,
  `draw_one`).
- **Bond styling.** Colour by k with **viridis** (sequential, k≥0); **constant medium line width**
  (do NOT encode k by width). Bonds **very close to k=0** are drawn **dashed** (solid otherwise) — no
  faintness/alpha.
- **Field maps (per-triangle ν or E): fill the whole triangle** (filled polygons). **ν → diverging
  colormap centered at 0**; **E → sequential**. Colorbar each.
- **Directional response ν(θ), E(θ).** Cartesian is the MAIN plot (ν and E vs θ∈[0,π], target dashed).
  ADD a polar plot (esp. E). **Polar ν scheme:** radius = **|ν(θ)|**, coloured **blue where ν>0, red
  where ν<0**; E polar is direct (E>0).
- **Means & spread.** Default: all group means together in ONE clean plot (no bands); each group's
  mean±σ in its OWN subplot. **Combine-all** (means+σ overlaid) only when ≤3 groups AND the
  comparison IS the overlap.
- **Resolution.** Standalone reusable elements **≥300 DPI** (higher for publication); montages 200
  (working; bump for production). Render each element/panel as its own high-DPI image, then compose.
- **Load, don't re-optimise** at plot time (`load_network`; random restarts ⇒ non-reproducible).

### Design-workflow conventions  *(settled 2026-08-13)*

- **Regularization: penalise k-VARIANCE, not deviation from 1.** `optimize(reg>0)` adds
  `reg·mean((k−mean k)²)` — keeps k near a *constant level* (the level floats freely, e.g. for an E
  target), discouraging floppy/near-mechanism designs. Use `reg ≈ 0.01–0.05`. **The default is
  `reg=0.0` (no regularization) — you MUST pass reg;** reg=0 lets k drift to soft channels →
  near-mechanism designs the linear solver mispredicts. *(Any further reg feature: discuss first.)*
- **Multi-region = GLUE, never joint-optimize.** Design each patch independently on its own cell,
  then `C.glue()` (retriangulate the seam, keep each side's k; `glue_square_hole` for inclusions). A
  joint region-scoped `optimize()` over one connected lattice drives the *interface* bonds to k≈0 —
  slitting the sheet into a mechanism (looks solved, physically broken).
- **Position optimization (default ON) — ALTERNATING polish, not simultaneous.** `design()` searches k
  over the topology pool, then polishes the top design by alternating `[gradient k-design] ↔ [SPSA
  position nudge]`. **Why alternating:** k has cheap solver gradients; **positions do not** — they
  enter via the geometry (`q_e`, triangle areas, the constraint operators), off the autograd path — so
  they use derivative-free SPSA. Simultaneous joint k+position would need differentiable positions (a
  Phase 2 core change) or SPSA-on-everything (loses k's gradient speed); logged as a future direction
  (FUTURE_DIRECTIONS #13). Rules:
  - **Weight-matching (default):** pass the k-design's `nu_weight`/`E_weight` to the polish, else it
    silently optimises a different loss and destroys anisotropy. Opt-out only to change weights on purpose.
  - **Never re-triangulate** (`redelaunay_every=0`): distortion only, connectivity frozen.
  - **Safety gate:** the polished design replaces the top ONLY IF it both lowers the design loss AND
    passes the independent-sim honesty check (`solver_sim_gap < gap_tol`).
  - Positions help isotropic targets / loss but NOT the **anisotropy-amplitude ceiling** — an
    *empirical* topology/size bound (not a harmonic limit); to push further, change topology or grow
    the cell, don't burn budget on positions.
- **Triangulation vs re-triangulation.** Triangulating a point cloud to CREATE a topology is the
  normal generation step (`seeds`) — fine, the only time triangulation is needed. **Re-triangulation**
  (re-Delaunaying an existing network after moving its points, discarding the topology) is AVOIDED
  unless specifically required.
- **Disorder — two intents, don't conflate.** (a) *Frozen-connectivity magnitude-η* (Phase 2
  `build_periodic_tf_mesh`, η<0.5): perturb node positions on a FIXED topology, no re-triangulation, no
  `uniform(−η,η)` — this yields the auxetic band. (b) *Topology scan*: triangulate perturbed point
  clouds (`seeds`) to reach different topologies — here triangulation is the point.
- **Always save the designed network once** (`save_network`: k + C6_per + meta incl. target, gap,
  `is_fictional` mask); figures LOAD it, never re-optimise (cf. plotting/env).

### Project coding conventions  *(settled 2026-08-13)*

Complements the global charter's coding style; only the project-specific bits here.

- **Match the base-paper notation** (Grossman & Boudaoud, PRR 2026) in code and comments: ν, E, k, ℓ₀,
  Voigt vec3 `[xx,xy,yy]`, `q_e`, `A(s)`, `W`, `C_eff`, Δg, ḡ — names track the maths.
- **Conform to the core.** `Phase 2/forward_solver_torch.py` is authoritative for solver conventions
  and API; new code adapts to it (never the reverse). Reuse `_common` / `inverse_design`; if a helper
  must be factored out, **copy it into the consuming phase** rather than editing the protected core.
- **File-header provenance:** each file states its purpose + which paper §/equation it implements.
- **DRY, minimal diffs.** Least code that does the job; a second consumer of a routine ⇒ factor it into
  one shared module; code reads like its surroundings. Naming: clarity over brevity.
- **Fail-fast** errors (project default); annotate types where they sharpen intent.

### Saving / docs habits  *(settled 2026-08-13)*

- **Keep every experiment in-repo** (script + outputs + plot), add-only during active work — never
  only in scratch/tmp. Convention: the **script** lives in `Phase 5/verifications/` (or the relevant
  phase's `verifications/`), **outputs + a results doc** in `Phase 5/results/<exp>/<EXP>.md`. **End
  every experiment with that doc** — *what · method · key numbers · limitations · figure links* — and
  carry provenance (commit / config / seed).
- **Designs: save-then-load** — `save_network` once; figures load, never re-optimise (cf. env/plotting).
- **Docs in lockstep with code.** Module `.md` updated at the end of every task; project docs
  (`documentation/`) on big changes; papers only after a wide range of experiments. **`.md` + `.pdf`
  rebuilt together** (`_build_pdf.py` — cf. env). Root `README.md` stays a stub — do not edit.
- **Cleanup only proposed-and-approved, never mid-work or autonomous.** Keep what backs a
  result/figure/oracle; drop failed/superseded/redundant; ask when in doubt.
- **Memory vs repo docs:** memory = context for continuing development; repo docs = every completed
  achievement / interface. Keep resumable at every checkpoint (docs + memory + committed/noted work).
