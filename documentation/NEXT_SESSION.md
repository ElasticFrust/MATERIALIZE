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

**6. A THIRD verification path already exists** — `test_hex_closed_form.py` is analytic and
independent of both solver and sim (4.4e-06). A-18 *generalises* it; it is not the first.

---

## What to do next

### Highest value
1. **Re-run goal1 (and scope goal2) over a symmetric −1 < ν < 1 grid, under the new selection.**
   goal1's grid stops at +0.45 — precisely the censoring `goal1_frontier` had to be built to expose —
   and both predate the scored selection that moved g1_2's floors substantially. ν = **+0.906** is
   already reached trustworthily; ν = 1 is attainable in principle (hexagon closed form at d = 2).
   `Phase 5/results/reach_summary/REACH_SUMMARY.md` has the whole picture on one axis: **contrast is
   the lever on both ends, and positions alone (k ≡ 1) cannot exceed the uniform-lattice +1/3.**
   goal2's target set has **not** been scoped — it is directional/full-tensor, so "the full range"
   means something different there.
2. **B-1** — solver nondeterminism localised to `W`, ~1 run in 21, up to 6 %. Still the only
   correctness blocker. Harness `Phase 3/verifications/b1_reproduce.py`. It has to be caught.
3. **M2 rebuild + retrain** — parked by request. Labels verified stale (37/41 drift, worst |Δν| 1.73),
   so `checkpoint.pt` is unverified. Two questions before any training: **what M2 is meant to be**
   (CLAUDE.md calls it a GNN *edit-policy* in four places while `m2/model.py` says *forward
   surrogate* — an unresolved docs-vs-code contradiction, deliberately left for the user to settle),
   and what the training target and success criterion are.

### Also open
- **`designer.design()`'s safety gate still VETOES** on `solver_sim_gap < gap_tol`. Only
  `run_g1_2.design_one` was changed. Deliberate: it reaches into the design path and needs its own
  blast-radius check.
- **goal1's |Δν| = 0.311 is unexplained.** Suspect |W| — but note the prior art:
  `verification_tools/plots/per_triangle_C/PER_TRIANGLE_C.md` already measures corr(|ΔC|, ‖W‖) = +0.39
  for the *contraction-isolation* residual, which is not the same quantity as end-to-end disagreement.
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
  `g1_2` (49 designs, 49/49 sign-agreeing, to −0.436) and goal1 f=0 (−0.789). Any future claim that
  some setup "cannot reach negative ν" should be checked against these before it is written down.
