# NEXT SESSION — start here

**Rewritten 2026-08-22 (end of session).** Working tree clean, everything pushed on
`claude/funny-davinci-H4pdS`. Read `CLAUDE.md` §1–3, this file, and
**`documentation/VERIFICATION_CAMPAIGN.md` — the index of what has already been measured. Read it
before proposing any new measurement.** `AUDIT_2026-08.md` §6 STATUS holds the older backlog.

> **The goal is the GNN.** Everything below is ordered by what has to be true before training M2 is
> worth doing. Item 1 is the only real blocker; items 2–3 are decisions only the user can make.

---

## THE TODO, in order

### 1. B-1 — the overnight run. THE BLOCKER.
Solver nondeterminism. **Why it gates the GNN:** if the solver can differ from the independent sim
by 1.18e-02 on ~1 run in 21, a training set labelled by that solver carries unlabelled noise of the
same size, and no amount of model capacity fixes that. M2's labels are only as good as this.

**Caught again 2026-08-22** during a routine gate run: `test_homogenization`, regular lattice
(φ=ψ=1, η=0, seed 0), **1.18e-02 against a 1.6e-12 baseline**, then 8/8 PASS on repeat in isolation.
Evidence in `Phase 3/verifications/b1_dumps/` (harness auto-dump
`b1_anomaly_20260822T142234Z_eta0.0_s0.json` + the suite log).

**Ruled out today:**
- **The cheap harness does not reproduce it.** New `b1_reproduce.py threads N` mode: suite context
  built once, then 360 probes across 1- and 4-thread arms → **0 hits in both**, every probe at
  exactly 1.00× baseline. ~17 hits were expected at the nominal rate. So replaying the 13 preceding
  tests is **not sufficient context** — consistent with "clusters in time with a persistent state
  transition", and it means the "~1 in 21" refers to something other than these probes.
- **Thread count is not the direct cause.** Threading *does* make the forward path nondeterministic
  (40 calls on the regular lattice: 2 distinct bit patterns at 4 threads, 1 bit-exact at 1 thread —
  isolated for the first time), but at **1.5e-33**, thirty orders below the failure, consistent with
  the 1–2 ulp envelope already on record.

**The plan (agreed):** an overnight loop of the **FULL suite** (`Phase 3/test_inverse_design.py`,
~12 min/run), **paired 1-thread vs default** so the thread question is answered at the same time.
~40 runs/night, ~2 expected events. **Write each run's verdict incrementally** — a suspend at hour
six must not cost the night (one g1_2 design already recorded 18266 s from a machine suspend despite
`_keep_awake`).

Useful control that fell out: `torch.set_num_threads(1)` makes the forward path bit-reproducible, so
"noise or real state change?" becomes decidable.

### 2. M2 — two decisions, then it is unblocked
- **What M2 IS.** `CLAUDE.md` calls it a GNN **edit-policy** in four places (§1 twice, the
  entry-point line, §2's table); `Phase 5/m2/model.py` says **forward surrogate**. A straight
  docs-vs-code contradiction, deliberately left for the user. It decides the dataset, the loss and
  the success criterion, so it comes first.
- **What it trains against.** Graph → C6 (6 numbers) as now, or graph → ν(θ),E(θ) on the 37-angle
  grid; loss on raw C6 / normalised components / derived ν,E; success measured against the
  **independent sim on held-out topology FAMILIES**, which `M2.md` records the July validation got
  wrong (random split against the solver labels the model trained on).

### 3. M2 — rebuild, retrain, validate
Labels are **verified stale** (37/41 drift, worst |Δν| = 1.73), so `checkpoint.pt` is unverified.
`Phase 5/dataset/` and `m2/data/` are deliberately ABSENT so nothing retrains on the old labels; the
July set is archived at `validation_2026-08/attic/m2_dataset_2026-07_pre-A0/`. Record the smoke-train
metrics this time (`M2.md` still has `<FILL>`).

**Today's work substantially improves what a dataset must cover** — see
`Phase 5/results/reach_summary/REACH_SUMMARY.md` for the reachable envelope on one axis.

### 4. Deferred, deliberately
- **The tiling discrepancy — LEFT AS IS, no run-time guard** (user's call, 2026-08-22). See below.
- goal2 over its full target range — its target set has never been scoped (directional/full-tensor).
- `A-8` open-boundary suite · `A-17` tail (centre-vertex re-representation) · `A-18` analytic-oracle
  generalisation · TODO 2.6 `open_stretch` guard.
- **FD #2 needs redesign, not re-aiming** — both proposed targets measured insufficient (below).

---

## What changed on 2026-08-21/22

**1. "Untrustworthy" is NOT "unreal".** `trustworthy` is
`gap = max_θ|Δν(θ)|/(|ν_sim(θ)|+0.05) + max_θ|ΔE(θ)|/|E_sim(θ)| < 0.05` — a **relative, PER-ANGLE
agreement test between two code paths**, not a physicality verdict. The +0.05 floor makes it a ~5%
relative test where |ν| ≫ 0.05 but an absolute |Δν| ≤ 0.0025 where |ν| ≪ 0.05, ~20× stricter near
zero. Tabulating only gap-passing rows **censors the extremes**.

> **Not a new observation.** `AUDIT_2026-08.md` already recorded for `g1_2_freed` that the optimiser
> "claims ν down to −0.761 over all 110 runs" while the trustworthy subset gives [+0.100, +0.340].
> Today added the *interpretation*: those designs are **49/49 sign-agreeing** between the two codes
> (median bulk |Δν| = 0.073), so they are not "a near-mechanism regime the sim rejects" as the audit
> concluded — they are designs both codes broadly agree about that fail a strict per-angle
> tolerance. And the fix below.

**2. Selection is SCORED, not vetoed** — in **both** `run_g1_2.design_one` and `designer.design()`:
`err + 0.5·gap`, with PHYSICALITY still a veto (non-SPD / non-finite / |ν|≥ν_max is a statement about
the network, audit A-5). `gap_tol` now only sets the reported `trustworthy` flag.
Measured cause of the old rule's damage: the **SPSA step size**. At `a = 0.15` the search barely
moves and returns a *target-independent* endpoint; at `a = 0.25` it reaches ν = −0.134 with a larger
gap, and the veto discarded it.

**3. g1_2 re-run: ALL TEN topologies reach auxetic ν.** `triangular` +0.038 → **−0.217**,
`flipped_tri_f8` −0.057 → −0.272, `tetrakis` −0.067 → −0.226, `rotating_squares` −0.259 → −0.398.
Deepest −0.436 (honeycomb, solver −0.488) at k ≡ 1. Trustworthy unchanged at 45/110 — the fix
recovered *reach*, not agreement.

**4. goal1: symmetric grid + repaired position budget → a conclusion overturned.** `NU_GRID` is now
13 points over [−0.95, +0.95]; `design_iso` now passes `spsa_a = 0.25` instead of falling through to
the library default 0.02 (travel per coordinate **0.202 → 2.53** lattice spacings, matching g1_2, at
the same step count and runtime). Result: **every contrast band reaches ν < 0**, including f = 0.99
at −0.336. So "forbid contrast and the design space collapses onto +1/3" was a **search artefact**.

> **Corrected physics:** the lever is **asymmetric**. **Contrast raises the CEILING** (+0.34 at
> f = 0.99 vs +0.90 at f = 0 — geometry cannot push ν above the uniform-lattice value). **Positions
> supply the FLOOR** (−0.29 … −0.34 at every band; −0.436 in g1_2 at k ≡ 1). At f = 0.99 the window
> is [−0.336, +0.338] — near-symmetric, ≈ ±1/3, *not* a point near +1/3.

Median |err| **0.198 → 0.0063**, success 33 % → 55 %. Cost: trustworthy 114 → 67 (bigger moves, more
disagreement). `quality_floor` also enabled at **1e-3** from its A/B — a degeneracy guard only;
**never raise it to buy agreement** (0.03+ multiplies median error ×17 and would destroy the −0.436
design).

**5. A(s) conditioning is not the whole story.** `rcond(A(s))` alone does **not** predict solver
error: on the hexagon closed form, driving `A(s)` to numerical rank-1 with soft spokes *improves*
accuracy by six orders (corr **+0.73**, best accuracy at worst conditioning). SOFT k is benign;
**DEAD** k and **slivers** are not — `CLAUDE.md` §3 now separates them. In `g1_2` (k ≡ 1, purely
geometric) the sliver route is confirmed hard (`quality_p05` corr −0.86, and the **tail** predicts
better than the worst triangle).

**6. A THIRD verification path already exists** — `test_hex_closed_form.py` is analytic and
independent of both solver and sim (4.4e-06). **A-18 generalises it; it is not the first.**

**7. New instruments:** `plotting.plot_ranges` (reach intervals; pale = full reach, solid = agreeing
sub-range, filled/hollow markers), `Phase 5/results/reach_summary/` (every ν achieved, on one axis),
`conditioning_probe.py`, `hex_conditioning_check.py`, `g1_2_solver_recheck.py`,
`g1_2_triangular_start_probe.py`, `b1_reproduce.py threads`.

---

## The discrepancy we are LEAVING OPEN (user's call)

`tiling_honeycomb_r3` — a 36-node chord-triangulated honeycomb — is the worst solver-vs-sim
disagreement in every goal1 run: **0.311 → 0.323 → 0.708** as the position budget grew. It is **one
deterministic network recurring**, not three independent samples (the tiling pool is fixed and each
topology appears once per sweep) — an earlier version of this file overstated it.

The 08-18 instance was **healthy on every axis** (`rcond_min` 5e-03, shape quality 0.475 ≈ 20°,
`k_min/mean` 0.65), so neither §3 route explains it. The current 0.708 instance has **not** been
re-measured. The live hypothesis (the user's) is that **position optimisation degrades the
geometry**, hardest where k is constrained but the target still looks chaseable — which fits the
disagreement more than doubling when travel went 13×, and fits the worst case sitting in `medium`
rather than at extreme ν.

**Decisions taken:** the tilings are **legitimate networks** — chord triangulation with all edges at
designed k is a real spring network, just not the tiling it is named after. **Honouring `k0` is NOT
the fix** (soft chords carry the face's shear, so it would only make everything near-mechanism). A
centre-vertex fan would be a better *representation of a honeycomb* (what A-17's tail and the hexagon
gate use) but is not automatically a better network. **No run-time geometry guard is being added.**
Note `run_goal1`'s CSV records `kmin_avg` and `solver_sim_gap` but **no geometry health**, so this is
not answerable from the results without recomputing.

---

## Things not to re-learn the hard way

- **The project keeps re-deriving what it already knows**, because findings live in prose spread over
  a 1300-line audit and a dozen results docs rather than in the data. Three times in one session I
  proposed analyses that already existed (per-triangle localisation *with a ‖W‖ field*; the
  sliver-vs-gap correlations in `positions.tri_shape_quality`'s docstring; the `ab_quality_floor`
  trade-off). **Check `VERIFICATION_CAMPAIGN.md` and `verification_tools/README.md` first, and say
  what you found.** That rule is in `CLAUDE.md` §3 — but it indexes *scripts*, not findings.
- **Read reach over ALL runs, never the trustworthy subset.** This concealed a real result four
  separate times, including in `run_goal1`'s own printed summary, which would have reported
  "medium: [+0.15, +0.38]" and hidden the −0.377.
- **`ab_quality_floor` is a SHAPE floor, not a stiffness floor.** The mislabel was in three places
  (its results doc, FD #2, and the audit) — all now corrected.
- **A re-run must be closed under producer→consumer.** The 08-18 re-run ran the producers, never the
  plotters, so figures and docs quoted July numbers over August data.
- **Look at the rendered figure.** Six plotter defects this session were found by eye, none by an
  exit code — a dropped topology, missing legend entries, a legend covering data, a `$\nu$` title
  rendering as a newline, filled/hollow markers indistinguishable against their own bar, and bars
  1.7 slots wide after a scale went stale.
- **An edit whose assertion fails can still get committed** (`eb0b8b8` → corrected by `612e8cf`).
- **Wall-clock timings in logs may be fiction** — one design recorded 18266 s from a machine suspend.
- **ν < 0 is reached and sim-confirmed** in `auxetic_sweep` (to −0.6009), `g1_2` (49 designs, 49/49
  sign-agreeing, to −0.436) and goal1 (to −0.82 at f = 0, and ν < 0 at *every* contrast band).
