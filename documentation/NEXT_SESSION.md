# NEXT SESSION — start here

**Rewritten 2026-08-22.** Repo state: working tree clean, everything pushed through `612e8cf` on
`claude/funny-davinci-H4pdS`, all five gates green. Read `CLAUDE.md` §1–3, this file,
`documentation/VERIFICATION_CAMPAIGN.md` (**the index of what has already been measured — read it
before proposing any new measurement**), then `AUDIT_2026-08.md` §6 STATUS for the audit backlog.

---

## What changed on 2026-08-21/22, and why it matters

The session began as a conformance sweep and turned into a correction, after the user objected on
physical grounds that η-disorder alone reaches ν < 0, so a directed optimiser must at least match it.
He was right, and three layers of reporting said otherwise.

**1. "Untrustworthy" is NOT "unreal" — the single most important thing to carry forward.**
`trustworthy` is `gap = max_θ|Δν(θ)|/(|ν_sim(θ)|+0.05) + max_θ|ΔE(θ)|/|E_sim(θ)| < 0.05`: a
**relative, PER-ANGLE agreement test between two code paths**, not a physicality verdict. Because the
denominator carries a 0.05 floor it silently changes meaning — a ~5% relative test where |ν| ≫ 0.05,
but an absolute |Δν| ≤ 0.0025 test where |ν| ≪ 0.05, i.e. ~20× stricter near ν = 0. Tabulating only
gap-passing rows therefore **censors the extremes**, which is exactly what an inverse-design
experiment exists to map. It made `g1_2` read as "distortion never reaches auxetic ν" when 49 designs
reach ν < 0 with **49/49 sign agreement between the two codes**.

**2. The selection rule was discarding good designs.** `run_g1_2.design_one` kept the
closest-to-target *trustworthy* candidate and fell back to the lowest-gap one — which systematically
preferred timid designs. It now scores `err + 0.5·gap` (`SCORE_LAMBDA`), the gap a **cost, not a
veto**. Measured cause (`g1_2_triangular_start_probe.py`, 36 runs): the controlling variable is the
SPSA **step size**. At `a = 0.15` the search barely moves and returns a *target-independent* endpoint
— identical for ν* = −0.10 and −0.30, which is exactly the saved-data signature of one geometry
written for all five negative targets; at `a = 0.25` it reaches ν = −0.134 with a larger gap, and the
old rule threw that away.

**3. g1_2 re-run under the new rule: ALL TEN topologies reach auxetic ν.** `triangular` went
**+0.038 → −0.217**, `flipped_tri_f8` −0.057 → −0.272, `tetrakis` −0.067 → −0.226,
`rotating_squares` −0.259 → −0.398. Deepest overall −0.436 (honeycomb, solver −0.488) at k ≡ 1.
Trustworthy count unchanged at 45/110 — the fix recovered *reach*, not agreement.

**4. `positions.quality_floor` is ON at 1e-3**, set from the A/B its old default deferred to
(`ab_quality_floor.py`, 40 runs). It is a **degeneracy guard only**: 1e-3 is free, 0.03+ doubles
trustworthiness while multiplying median error ×17 and would destroy the ν = −0.436 design.
**Do not raise it to buy agreement — that is the veto mistake one level down.**

**5. A(s) conditioning is not the whole story.** `rcond(A(s))` alone does **not** predict solver
error: on the hexagon closed form, driving `A(s)` to numerical rank-1 with soft spokes *improves*
accuracy by six orders (corr **+0.73**, best accuracy at worst conditioning). Soft k is benign;
**dead** k and **slivers** are not. In `g1_2` (k ≡ 1, purely geometric) the sliver route is confirmed
hard (`quality_p05` corr −0.86), but **goal1's worst disagreement, |Δν| = 0.311, is healthy on every
axis** — conditioning, shape quality and k-contrast. At least two failure modes; FD #2 addresses one.

**5b. `designer.design()` is now consistent with `run_g1_2`** (2026-08-22): PHYSICALITY vetoes,
the gap is a **cost** in the ranking score, and the position polish accepts on the score. Physicality
stays a veto because non-SPD / non-finite / |ν|≥ν_max is a statement about the network (audit A-5).

**6. A THIRD verification path already exists** — `test_hex_closed_form.py` is analytic and
independent of both solver and sim (4.4e-06). A-18 *generalises* it; it is not the first.

---

## What to do next

### Highest value
1. **goal2 over its full target range — goal1 is DONE.** goal1 was re-run 2026-08-22 on a symmetric
   13-point grid over [−0.95, +0.95] (130 runs, 114 trustworthy): at f=0 it now reaches
   **[−0.825, +0.901]**, most of the physical range, and `goal1_frontier` is subsumed (it survives as
   an independent cross-check: +0.63 vs +0.658 for the f=0.1 frontier). **goal2's target set has NOT
   been scoped** — it is directional/full-tensor, so "the full range" means something different there
   and needs a look before a grid can be proposed.
   `Phase 5/results/reach_summary/REACH_SUMMARY.md` has the whole picture on one axis.
   *(Note `run_goal1` never vetoed on the gap, so unlike g1_2 its reach was never censored — an
   earlier claim here said otherwise and was wrong.)*
3. **B-1** — solver nondeterminism, ~1 run in 21, up to 6 %. Still the only correctness blocker.
   **CAUGHT AGAIN 2026-08-22** during a routine gate run: `test_homogenization`, regular lattice
   (φ=ψ=1, η=0, seed 0), **1.18e-02 against a 1.6e-12 baseline** — then 8/8 PASS on repeat in
   isolation. Evidence: `Phase 3/verifications/b1_dumps/` (the harness auto-dumped
   `b1_anomaly_20260822T142234Z_eta0.0_s0.json`, plus the suite log).

   **Two things measured while chasing it, both new:**
   - **Thread count causes run-to-run nondeterminism — isolated for the first time.** 40 identical
     `forward()` calls on the regular lattice: at `torch.set_num_threads(4)` → **2 distinct bit
     patterns**; at 1 thread → **1, bit-exact**. `b1_reproduce.py` *records* the thread count
     (`fingerprint`) but no mode ever *varies* it, so this was untested.
   - **But it cannot be the direct cause:** the variation is **1.5e-33** against a 1.18e-02 failure —
     thirty orders apart, consistent with the 1–2 ulp envelope already on record. Reduction order
     alone does not do it; whether it can *seed* a divergence through an unstable branch is untested
     speculation.

   **Useful control that falls out of this:** `torch.set_num_threads(1)` makes the forward path
   bit-reproducible, which turns "is this run-to-run noise or a real state change?" into a decidable
   question.

   **`threads` mode added and RUN — came back EMPTY, which is itself informative.**
   `b1_reproduce.py threads 60`: suite context built once, then 60 probes x 3 cases per arm at 1 and
   4 threads. **0/180 hits in BOTH arms**, every probe at exactly 1.00x baseline
   (`b1_dumps/2026-08-22_threads_mode_360probes_empty.log`).
   - Threading is not implicated *at this sample size* — but that is weak, because
   - **the cheap harness does not reproduce the phenomenon at all.** ~360 samples against a nominal
     1-in-21 rate should have produced ~17 hits. Zero. So **rebuilding context by replaying the 13
     preceding tests is NOT sufficient** to produce the excursion — which fits the documented
     "clusters in time with a persistent state transition": the transition has to *happen*, and a
     replay does not trigger it. It also suggests the "~1 in 21" on record refers to something other
     than these probes.
   - Incidental: in suite context 1 thread gave MORE distinct bit patterns (3) than 4 threads (2),
     contradicting the isolated 40-run test where 1 thread was bit-exact. Do not over-read either.

   **So the route is the expensive one:** an overnight loop of the FULL suite
   (`Phase 3/test_inverse_design.py`, ~12 min/run), ideally paired 1-thread vs default so the thread
   question is answered at the same time. ~40 runs a night, ~2 expected events at the observed rate.
4. **M2 rebuild + retrain** — parked by request. Labels verified stale (37/41 drift, worst |Δν| 1.73),
   so `checkpoint.pt` is unverified. Two questions before any training: **what M2 is meant to be**
   (CLAUDE.md calls it a GNN *edit-policy* in four places while `m2/model.py` says *forward
   surrogate* — an unresolved docs-vs-code contradiction, deliberately left for the user to settle),
   and what the training target and success criterion are.

2. ~~goal1's position budget is effectively zero~~ **FIXED AND RE-RUN 2026-08-22.**
   `design_iso` had called `spsa_positions` without passing `a`/`c`, so positions ran at the library
   defaults: **0.202** lattice spacings of travel against g1_2's 2.68. Now `spsa_a = 0.25` (travel
   **2.53**, same step count, same runtime — travel scales linearly in `a`).
   **The result overturned a conclusion.** Every contrast band now reaches ν < 0, including f = 0.99
   at **−0.336**, where k is effectively frozen. So "forbid contrast and the design space collapses
   onto +1/3" was a *search* artefact. The corrected physics is asymmetric:
   **contrast raises the CEILING (+0.34 at f=0.99 vs +0.90 at f=0); positions supply the FLOOR
   (−0.29 … −0.34 at every band, −0.436 in g1_2 at k ≡ 1).**
   Target accuracy improved with it: median |err| **0.198 → 0.0063**, success 33 % → 55 %.
   Cost: trustworthy 114 → 67 — bigger position moves, more solver-sim disagreement.

### Also open
- ~~`designer.design()`'s safety gate still vetoes~~ **DONE 2026-08-22** — it now filters on
  PHYSICALITY only and ranks by `target_err_sim + 0.5·solver_sim_gap`; the position polish accepts on
  the score rather than on `gap_tol`. `gap_tol` sets only the reported `trustworthy` flag. Verified:
  designer gate 5/5 plus an end-to-end smoke keeping two `trustworthy=False` designs ranked by score.
- **goal1's unexplained disagreement — THREE independent reproductions; the best open lead.**
  |Δν| = 0.311 at ν=+0.286 (08-18, asymmetric grid) → **0.323 at ν=+0.296** (08-22, symmetric grid)
  → **0.708 at ν=+0.306** (08-22, symmetric grid + 13× position budget). **`tiling`, `medium` band
  (f=0.5), ν ≈ +0.30 every time**, across different grids, independently drawn topologies and a large
  change in search budget. Meanwhile the `soft` band spans [−0.82, +0.90] with a worst case of **0.0068** —
  **50× better while reaching 10× further**, so extreme ν is NOT where the codes disagree. The old
  instance was healthy on every conditioning axis (rcond 5e-03, quality 0.475, k_min/mean 0.65), so
  neither §3 route applies. Suspect |W| — but note the prior art:
  `verification_tools/plots/per_triangle_C/PER_TRIANGLE_C.md` already measures corr(|ΔC|, ‖W‖) = +0.39
  for the *contraction-isolation* residual, which is not the same quantity as end-to-end disagreement.
  **Start here: a `tiling` cell at f=0.5 targeting ν ≈ +0.29 is a reproducible instance to dissect.**
- **FD #2 needs redesign, not re-aiming** — both proposed targets are now measured insufficient.
- A-8 open-boundary suite · A-17 tail · A-18 generalisation · TODO 2.6 `open_stretch` guard.

---

## Things not to re-learn the hard way

- **Check `VERIFICATION_CAMPAIGN.md` and `verification_tools/README.md` before proposing a new
  measurement.** Twice this session I proposed analyses that already existed — the per-triangle
  localisation *with a ‖W‖ field*, and the sliver-vs-gap correlations (already sitting in
  `positions.tri_shape_quality`'s docstring). Both were indexed. CLAUDE.md §3 now carries this rule.
- **A re-run must be closed under producer→consumer.** The 08-18 re-run re-ran the producers and never
  the plotters, so goal1/g1_2 figures were July while their data was August, and both results docs
  quoted July numbers (100/110 vs 93/110; 52 vs 45).
- **Look at the rendered figure.** Five plotter defects this session were found by eye and none by an
  exit code: a dropped topology (`TOPO_ORDER` still naming a removed tiling, hiding `flipped_tri_f14`
  from every g1_2 figure), missing legend entries, a legend covering data, a `$\nu$` title that
  rendered as a literal newline, and a marker encoding in which filled and hollow were
  indistinguishable because the fill matched the bar beneath it.
- **An edit whose assertion fails can still get committed.** `eb0b8b8` claimed a CLAUDE.md change that
  had not landed; `612e8cf` corrected it. Verify the edit took before describing it.
- **Wall-clock timings in logs may be fiction.** One g1_2 design recorded 18266 s because the machine
  suspended mid-run despite `_keep_awake`; that run's "431.9 min" total is meaningless (real ≈ 2.4 h).
- **ν < 0 is reached and sim-confirmed** in `auxetic_sweep` (36/70 rows below −0.05, to −0.6009),
  `g1_2` (49 designs, 49/49 sign-agreeing, to −0.436) and goal1 f=0 (−0.825). Any future claim that
  some setup "cannot reach negative ν" should be checked against these before it is written down.
