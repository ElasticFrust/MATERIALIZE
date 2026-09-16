# PLAN A — the science

> **Split out of the approved plan of 2026-09-15**, which was written as one file only because plan
> mode permits editing one. **Plan A and Plan B are standalone and independent.** Nothing in A depends
> on B; only **B2** depends on A, and says so where it bites. Plan B is `documentation/PLAN_B.md`.
>
> **Purpose of this file:** the live plan for the A-track, carrying its own status. `M2_V2_PLAN.md`
> remains the stage plan (S0…S5) for the M2 arc; this file is the current stage's working plan.

**Status as of 2026-09-16:** **A0 is COMPLETE** (A0.1–A0.4, all gated, all committed). **A1 is
PARTIAL** — A1.a audited but not re-split, A1.b measured, A1.c not started, A1.d generated but not
sim-checked. Per-item status is in each section.

---

## Context

**Why this stage exists.** The M2 forward surrogate sits at per-triangle MAE/σ **0.0925** (`bravais`
holdout) and every identified lever on that number is measured and closed: data volume (+9.7 % gap),
capacity (§9g), the schedule on the full arm (§9i), `--bulk_weight`, `--graph_balance`, the
near-mechanism tail (§9j — error redistributed, not reduced) and `cond(G)` (§9k). The model is
representation-limited; the deficit is the per-triangle spatial structure (bulk 0.031 vs 0.0925).

**But the scalar was never the point.** The surrogate's purpose is that geometry is differentiable
through it, which the exact solver structurally is not (`ElasticSolver`: positions are "fixed inputs,
set at construction, not differentiated" — hence SPSA; `CLAUDE.md` §3, FD #13).

**Intended outcome.** Turn "is the surrogate good enough?" from an unanswerable question about a
scalar into a measured statement about the quantity a designer actually consumes.

### Scope decisions taken 2026-09-15

- **Trajectory work is SPLIT.** **A1.1a — the trace hook: KEPT, folded into A0** (§A0.4); recording
  costs nothing, while *not* recording means every design run from here discards its trajectory and
  recovering it later costs a full re-run. **A1.1b — the corpus campaign: DEFERRED** (§A1.e).
  *Why the policy question waits:* if A0 shows the GNN gradient is usable, **descent through the GNN
  is itself the design capability**, and a policy trained to imitate Δk is amortisation, not
  capability — it inherits the surrogate's error as a floor. The policy's non-substitutable value is
  **discrete topology edits** (no gradient exists for adding/removing/flipping a bond) and solution
  diversity.
- **`auxetic` is an EFFECT, not a family** (user's correction). η is an ADDED axis, not a replacement.
  Label every sample by **measured response** (sign of ν, anisotropy, mechanism proximity) as well as
  by generator, so the sharp holdout is *a region of response space*, not a generator.
- **`angle_gradient_vec` stays at TWO implementations, not three** — `model_v3`'s copy was replaced by
  a torch version that serves NumPy callers too, not supplemented by a third.
- **Position gradients in the exact solver (FD #13) are a FORK, not a cost question.** The asymptotic
  cost is fine (implicit differentiation of the constrained solve ≈ one extra solve per backward, the
  same order as the existing k-adjoint); the cost is rebuilding operator construction in torch
  *inside the protected core* and re-gating everything. A0.2's finite-difference reference is a cheap
  prototype of exactly that information.

**Not in this plan:** the weight-tied iteration arm (parked — §9k left the hypothesis unsupported and
the arm carries an lr confound making a negative uninterpretable); the remaining S3 folds; the web app
(`documentation/PLAN_B.md`).

---

## A0.1 — port the input pipeline to torch ✅ DONE (`4fa81d4`)

**Files:** `Phase 5/m2/train_v3.py` (`prepare`, `collate`, new `bond_vectors`),
`Phase 5/m2/model_v3.py` (`angle_gradient_vec`, `corner_index`, `vertex_stars`, `triangle_areas`).
**`Phase 2/` untouched.**

Everything inside `ForwardGNNv3.forward()` was already torch; the break was entirely upstream —
`prepare()` was pure NumPy, so the model was differentiable **only in its own weights**, despite
`model_v3.py:36` asserting the chain rule flows.

**The hinge — positions reach the carriers through the periodic offset:**

```
shift   = bond_R_stored - (pts_stored[v] - pts_stored[u])     # once, NumPy, constant
bond_R  = pts[v] - pts[u] + shift                              # torch, differentiable
```

Verified on 12 random dataset networks: `shift / (Lx, Ly)` is an integer to ≤2.2e-16, image indices
only in {−1, 0, +1}. `shift` is a property of the *bond list* — "u connects to the image of v offset
by (nx, ny) boxes" — i.e. part of the frozen connectivity, not of the coordinates. Holding it fixed is
required, not merely convenient.

**The real hazard is triangle INVERSION, not a changing shift.** We never re-derive offsets by minimum
image. Every perturbed geometry gets a signed-area check.

**Convention, settled consistently on both sides:** `DesignProblem.from_geo` takes rest lengths from
`geo['actual_len2']` and `prepare()` sets `l0 = ell` — both mean **ℓ₀ = ℓ, geometry and reference move
together, no prestress**. This is also what SPSA has always done.

**Per-channel switch (D6):** `k` and positions each independently `requires_grad`, so the design
surface can be stiffness-only, geometry-only, or both.

**Gates, all passed:** forward answer unchanged (`C6_per` ≤1e-12 relative on 50 samples) · recomputed
areas ≤1e-13 · torch `angle_gradient_vec` matches the core elementwise · **autograd vs central FD
≤1e-6** on `∂C6/∂pts` and `∂C6/∂k` across three families · `test_m2_head_v3.py`,
`test_m2_constraints.py`, `test_m2_train_surface.py` green.

**Also fixed here, and it was the session's biggest bug (`5625254`):** `evaluate_v2.geo_of` rebuilt a
solver from a stored sample with the **wrong edge orientation**, silently, for months.
`Phase 2/mesh_build.edge_vec_orientation` now returns (edge_idx, sign) with **sign 0 = unknown for
self-loops**, `check_edge_vecs` gates it, and `Phase 3/inverse_design.from_geo` raises on
inconsistency.

## A0.2 — gradient fidelity against the solver ✅ DONE (`10659e1`)

**Producer:** `Phase 5/verifications/m2_gradient_fidelity.py` · **Results:**
`Phase 5/results/m2_s1/GRADIENT_FIDELITY.md`

Objective = the designer's own weighted ν(θ)/E(θ) residual, reused not reimplemented, so the measured
gradient is the one descent would use. `∂L/∂k` against the solver's exact adjoint; `∂L/∂pts` against
**full 2N central finite differences** (the exact cosine, not a random-direction estimator),
affordable at `n_node ≤ 64`.

| channel | n | median cos | IQR | cos > 0 | \|∇GNN\|/\|∇solver\| |
|---|---|---|---|---|---|
| `∂L/∂k` | 300 | **0.9499** | [0.794, 0.986] | 90.7 % | 1.014 |
| `∂L/∂pts` | 100 | **0.9148** | [0.623, 0.965] | 89.0 % | 0.937 |

Pre-registered rule (median > 0.9 on `k`, > 0.7 on positions) **met on both**; magnitudes essentially
unbiased. 0 of 6282 health probes rejected — and rejections would have been counted, not dropped.

**The tail is the finding.** The cosine collapses monotonically with `max|W|` on both channels —
`∂L/∂k` 0.9986 (`max|W|` < 1) → 0.0424 (> 100); `∂L/∂pts` 0.9868 → −0.16. The gradient is near-perfect
in the bulk and fails precisely in the **near-mechanism tail**, the same place the value error
concentrates. Coherent rather than surprising: where the response is near-singular, so is its
derivative. **This is the evidence base for the trust-region gating rule.**

## A0.3 — end-to-end descent through the frozen surrogate ✅ DONE (`7db476a`)

**Producer:** `Phase 5/verifications/m2_designer_probe.py` · **Results:**
`Phase 5/results/m2_s1/DESIGNER_PROBE.md`

Adversarial by construction: descent walks toward wherever the surrogate claims the loss is low, which
is exactly where a surrogate is most likely to be wrong. Both arms run the **identical** optimiser
(L-BFGS + strong Wolfe on `raw`, same initialisation, same `reg = 0.02`); **only the forward model
differs.** 20 targets inside the measured reach envelope, so every request is reachable.

| | median loss (solver) | vs INDEPENDENT SIM | solver calls |
|---|---|---|---|
| start, no design | 4.526e-01 | 4.526e-01 | 0 |
| **GNN, free** | **6.412e-02** | **6.047e-02** | **0** |
| GNN + 4 refine rounds | 6.371e-02 | 6.027e-02 | 4 |
| M1, full run | 2.779e-02 | 2.779e-02 | **86** (range 34–101) |

> **The S4 number: M1 needs a median of 9 exact-solver calls to match the design the surrogate
> produces for free** — matched on **20 of 20** targets (min 1, max 32), against the 86 M1 spends in a
> full run.

Improves on the starting network on **90 %** (18/20); **beats M1 on 0 %**. Floored ratio 1.82, but
**p90 = 20.2** — heavy-tailed, so the median is the honest summary. Degrades with `max|W|` exactly as
A0.2 predicts. **Verdict: a strong initialiser, not a replacement.**

## A0.4 — the trace hook ✅ DONE (`52eec59`)

**File:** `Phase 3/inverse_design.py: optimize()` — Phase 3 is verified truth, so: a new `trace=None`
argument, and with `trace=None` the current path is byte-identical (asserted by a test comparing final
loss and `k` against the prior implementation).

**Why it belonged in A0.** A0.2 samples random networks from the dataset families, but the deployment
distribution is *networks a designer visits en route to a target* — the heavy-tailed `k` (max/median
17 healthy, 147 degenerate). **The solver gradient is already computed at every closure call and
discarded.** Free data on exactly the question A0.2 exists to answer.

**The line-search trap, handled.** `strong_wolfe` calls `closure()` several times per accepted step, so
`history` is per *evaluation*; recording every call stores near-copies of one iterate — the worst
possible leakage, and precisely what splitting on `traj_id` exists to prevent. When tracing, `optimize`
loops `opt.step(closure)` with `max_iter=1` and records **one row per accepted iterate**. Torch's
L-BFGS keeps its curvature pairs in `self.state` across `.step()` calls, so the traced path reaches the
same optimum — asserted as a test, not assumed.

**Each record:** `raw_k` / `raw_l0` and **`grad_k` / `grad_l0` — the solver gradient at that iterate**,
never the L-BFGS step (which is preconditioned by the inverse-Hessian approximation and is not the
gradient) — plus restart, step, loss, seed, mode, reg and thread count. The response (`k`, `C6`,
`C6_per`, `w_max`, `k_min`) is filled **afterwards** by `_trace_fill_response`, one solver forward per
row recomputed from that row's stored `raw`.

**Why the response is filled afterwards, and not in the closure.** An earlier version called
`prob.forward` inside the closure, which inserted an extra solver call *into the optimisation's call
sequence* — and the solver is measurably history-sensitive (`max|Δk|` 1.7e-08 between traced and
untraced runs, plus a 1.3e-03 first-call effect within a process). Filling afterwards leaves
`trace=[]` bit-identical to `trace=None`.

**⚠ POSITIONS ARE NOT RECORDED — the plan said they would be, and they are not.** `optimize()`'s modes
are `k` / `l0` / `both`; positions never enter its `raw` dict, because the SPSA position polish lives
outside it in `Phase 5/positions.py`, which keeps only an `spsa_history` of accepted *losses*, not the
coordinates. So A1.e **does** need a second hook, in `positions.py`, before a trajectory corpus can
carry the geometry channel. Recorded here rather than discovered later.

---

## A0 decision rule — pre-registered, and its verdict

| outcome | reading | next |
|---|---|---|
| median cos > 0.9 on `k` **and** > 0.7 on positions; A0.3 matches M1 within its scatter | usable gradient source | stop tuning accuracy; re-pose the edit-policy around topology |
| **cos > 0 in ≥80 % of cases, low median** | **usable WITH correction** | **proceed; record the correction rate as the cost** |
| cos ≤ 0 non-negligibly, or A0.3 diverges | the gradient is the defect, localised by family and `max|W|` | that localisation drives the choice between an architecture change and FD #13 |

**VERDICT (2026-09-15): row 1 on the cosines, row 2 on the end-to-end.** The gradient thresholds were
met, but A0.3 beats M1 on 0 of 20 targets, so the surrogate is **not** a replacement for the solver —
it is a free initialiser worth ~9 exact solves, and a usable descent direction **inside a trust region
gated on `max|W|`**. The user's reading stands: *"we know the surrogate cannot be a replacement."*

**Consequence for the edit-policy:** an edit policy via the surrogate with **the solver as verifier**
is the live route (the user, 2026-09-16), and **FD #13** — exact position gradients in the solver —
remains the fork that would largely remove the surrogate's reason to exist for geometry design.

---

## A1 — the dataset

### The finding that reshaped this stage *(measured 2026-09-15)*

**"Auxetic" is an EFFECT, not a family — and as an effect it is already well populated.** The
`auxetic` *generator* is 299 samples (0.7 %); networks with **ν < 0 number 8 003 (19.3 %)**, with
4 334 below −0.2 and 1 871 below −0.5. Only 2.9 % of them come from the `auxetic` generator;
**`disordered` supplies 64 %.** An earlier version of this plan read the generator share as the physics
coverage and badly understated it.

**So the real imbalance is in RESPONSE space, not in family shares:**

| ν bin | n | share | dominant generators |
|---|---|---|---|
| < −1 | 573 | 1.4 % | `disordered` 50 %, `auxetic` 16 % |
| −1…−0.5 | 1298 | 3.1 % | `disordered` 61 % |
| −0.5…−0.2 | 2463 | 5.9 % | `disordered` 69 % |
| −0.2…0 | 3669 | 8.9 % | `disordered` 65 % |
| 0…0.2 | 7241 | 17.5 % | `random` 35 % |
| **0.2…1/3** | **17697** | **42.7 %** | **`random` 74 %** |
| 1/3…0.5 | 6033 | 14.6 % | `random` 64 % |
| 0.5…1 | 2183 | 5.3 % | `random` 52 % |
| > 1 | 274 | 0.7 % | `cells` 38 % |

Anisotropy is well covered (10.9 % below 1.2× through 9.9 % above 20×); 4.0 % of samples have
E < 0.01. **Consequence: the corrective is to SUBSAMPLE `random`, not to generate more of anything.**
Zero compute. Generation is reserved for regions the audit proves empty.

### A1.a — response-space audit and re-split ⏳ AUDITED, NOT RE-SPLIT (`52eec59`)

The descriptors already exist per sample — `nu`, `E`, `anisotropy`, `w_max`, `min_eig`, `contrast`,
`min_quality`, `spd`, `sim_ok`, `sim_gap` are all stored in `dataset_v2_s0.npz`. Nothing needed
rebuilding; what was missing is that **nothing used them for stratification or splitting.** The audit
is done and is the table above.

**OPEN — the user's decision.** Holding no response bin above ~15 % costs **59 % of the data**; a
looser cap keeps more and leaves the ν ≈ 1/3 mass dominant. **The cap has not been chosen.** Once it
is: re-split on a *region of response space* (e.g. ν < −0.2, which removes 4 334 samples across four
generators) rather than a generator — that is the test the tool actually has to pass, and
leave-one-family-out on `auxetic`'s 299 samples never was. Keep the generator label as provenance; it
stays useful for diagnosis, it just stops being the holdout axis.

**Gate:** no response bin above the chosen cap; the held-out response region absent from training under
the new split; the re-split set reproduces the existing model's score on the *old* split, proving the
re-split is a relabelling and not a silent data change.

### A1.b — the `cells` label floor ✅ MEASURED

"Floor" = solver-vs-sim disagreement on the same network, i.e. the best score achievable from these
labels; the GNN is not involved (`evaluate_v2.py:233`, `nu_s` from the stored solver label vs `nu_p`
from the independent virial). ~0 on `bravais`, `random`, `disordered`, `longrange` (mean ≤0.0013), but
**`cells` mean 0.0130 with 9.3 % above 0.02** — so for a tenth of that family no model trained on
solver labels can meet the must-tier.

*Provenance:* the stored per-family JSONs are the **s777 n=2000** fresh draw (`disordered` is s4321).
Draw-to-draw the MEAN moves a lot and the COVERAGE fraction barely does — bootstrap, 2000 resamples,
same n: the mean's 95 % CI spans **27–89 %** of its own value while the fraction within 0.05 spans
**3.2–8.1 %**. Recorded as data; the mean-based criterion stays by the user's standing call.

Candidates for the cause: minimal √N averaging at 6–24 triangles; a high share of near-degenerate
configurations where the solver's regulariser rather than physics sets the answer; something specific
to periodic wrapping at tiny cell size.

**OPEN:** whether `cells` stays in training or is scored as a known-degraded regime.

### A1.c — ingest the 660 designed networks ⛔ NOT STARTED

**660 `.npz` files** under `Phase 5/networks/**`, all carrying `C6_per`. They matter for the
**heavy-tailed `k`** an optimiser actually produces (max/median 17 healthy, 147 degenerate) — the
deployment distribution, which lognormal sampling does not reach.

Two mandatory filters: **provenance** (unstamped ⇒ pre-August ⇒ suspect;
`validation_2026-08/scan_provenance.py`; the earlier sweep kept 77 of 520, and every `tiling` result
predates the crossing-chord fix) and **source attribution**, so an ingested design is held out with
whatever region/generator it came from.

### A1.d — generation from arbitrary bases ✅ DONE (`650fa91`, `03c464a`)

**Producer:** `Phase 5/verifications/m2_auxetic_generator.py` · **Results:**
`Phase 5/results/auxetic_generation/AUXETIC_GENERATION.md`

**2832 networks, 0 inverted, 0 solver failures**, through the user's three mechanisms — **VD**
(geometry untouched, η only decides rigidities, `k` from `atanh(a(l−l0))`), **η** (geometry really
distorted), and **η+α** (distorted, with `k` set from the distorted lengths) — swept across *ranges* of
η and α (α ∈ [−10, +10]), never at fixed values, on 10 Bravais bases and 4 random point-cloud bases.

Underneath it, `Phase 5/m2/fields.py` gained a perturbation that **cannot break a triangle**: the
**exact first-inversion scale** (a quadratic in the displacement scale, because signed area is), a
per-triangle limit, and a **local fixed point** (`local_scale_field`) recovering the factor ~2.3 in
median amplitude that a single global scale throws away. `displace_safe(..., scale=)` picks between two
**mutually exclusive** guarantees — `'global'` reproduces the requested amplitude shape exactly,
`'local'` gives every vertex its own neighbourhood's maximum. Gated by 12 tests in
`Phase 5/verifications/test_displace_safe.py`, including the √3/4 regular-lattice floor.

**OPEN — the tier-(A) sim cross-check.** These carry solver labels only (`sim_status='not_run'`), and
30 % of them sit above `max|W|` = 10, where the folding measurement showed solver-sim agreement can
degrade. **This must precede training on them.**

**OPEN — the targeted isotropic run.** Zero isotropic-auxetic samples came from the 9 intrinsically
anisotropic bases across 1890 rows, while the capable bases went 0.26 % → 6.8 % with cell size (the
1/√N anisotropy self-averaging). `--bases p1.0_1.0_a2-a1,random --reps 16` spends the same budget with
every sample capable of landing in the cell — roughly 4× the yield. Not launched; it is compute.

### A1.e — the trajectory corpus (A1.1b) ⛔ DEFERRED

Re-run M1 over new targets spanning the reach envelope. The `k`/`l0` hook (A0.4) exists, **but the
position hook does not** — see the warning in A0.4 — so this is a little code and then compute.
**What the demonstrations should contain depends on what the policy is for** — and A0's verdict
(surrogate = initialiser and critic, solver = verifier) is the input to that, so this waits on the
edit-policy design.

### A1 verification gate — before anything trains on the result

- response-bin occupancy printed, no bin above the chosen cap;
- held-out response region absent from training, including through the ingest;
- **`traj_id` count strictly < sample count** if A1.e has run — the leakage guard has never done any
  work in its life and that is the first build where it can be tested at all;
- label audit on a random 200, **reporting the solver-vs-sim floor per response bin** (not per family —
  the floor is a property of the networks, and `cells` showed it is not uniform);
- `train_v2.oracle_check` passes; threads and tiling method recorded (D3/D4).

---

## Cost

**A0** — spent: ~2 h of compute, no retraining. **A1** — A1.a is pure re-splitting with **zero
generation**; A1.c is the ingest; A1.d's two open items are about one arm of compute each.
For scale: a full retrain is ~85 h.

## Deferred beyond this plan

1. Re-pose the edit-policy around **topology edits and solution diversity**, the parts descent cannot
   provide — with the surrogate as critic and the solver as verifier.
2. **FD #13** — exact position gradients in the solver — if the position channel proves the binding
   constraint.
3. The remaining S3 folds, on the re-split data.
4. **Supercell re-representation for self-loops** (user-approved, not started).
5. Rename the two Bravais diagonals in `seeds.py` (`a1+a2` at φ=ψ=1 is *not* the triangular lattice —
   sides 1,1,√3, angles 30-30-120, quality 0.600 — while `a2−a1` is, at quality 1.000; calling both
   `p1.0_1.0` caused three wrong readings in one session). Generalise or document
   `mesh_build.set_VD`'s hardcoded `l₀ = 1`.
