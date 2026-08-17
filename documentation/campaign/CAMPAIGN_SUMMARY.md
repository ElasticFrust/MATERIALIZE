# The post-audit verification campaign — synthesis

**What.** Re-run every Phase 5 experiment under instruments repaired by the 2026-08 audit (A-0 shear
channel, A-10 ν,E convention, A-9/A-1 independent oracle, A-2 ranking metric, A-11/A-12 honest
drivers, B-3 provenance). Plan: `VERIFICATION_CAMPAIGN.md`. Per-stage docs under
`documentation/campaign/stage*/`.

**One-line answer:** *the designer's capability was always real; the instruments measuring it were
not.* Every campaign that could be re-run recovered to at or better than its originally claimed
performance — and the re-run surfaced **seven defects that reading the saved outputs could never have
shown**.

---

## 1. Results by stage

| stage | outcome |
|---|---|
| **0** Phase 2+3 re-verify | Phase 2 8/8, Phase 3 16/16, sanity exact (ν=1/3, E=2/√3 both paths), designer surface 4/4; the 9-family disorder sweep **reproduced bit-for-bit** across a day and several commits |
| **1** shakedown | caught two traceability gaps (`seed` never recorded, `trustworthy` never saved) *before* the 3 h run |
| **2** `goal1` | auxetic reach **retreated in all five bands** — the A-0 signature, as predicted. Untrustworthy 10 → 21: the *gate got sharper*, not the designer worse |
| **2b** ν frontier | the reported "+0.45 ceiling" was **the grid edge**. True frontier **+0.91** at f=0; **+0.63** at f=0.1 is genuine (measured, saturating) |
| **3/3b/3c** `g1_2` | physics **withheld** — the experiment braced its own auxetic motifs, and four topologies fail A-17's mesh preconditions. Freeing the bracing (3c) changed their physics completely and their trustworthiness not at all, which is what identified malformation as the governing defect |
| **—** hexagon suite | three **analytic anchors** hit unprompted; single hexagon = lattice = sim to **1e-11** |
| **4** `goal2` | **RECOVERED.** Mean error **0.041** vs **0.48** for the old designs re-measured — and better than the original *claim* of 0.0298 |
| **4b** `goal2_attempts` | selection story holds **13/13**; kept 42 / dropped 52 / out-ranked 65 |
| **5** A-7 figure | the "≈1.4× mean-field over-compliance" is **not a constant** — 1.015 … 228, median 1.14 |

## 2. Is the solver trustworthy?

**Yes on well-formed meshes, and the boundary is now known and cheap to test.**

Exact (gap 0.00000 – 1e-11) across: the crystal (ν=1/3, E=2/√3); nine disorder families to η=0.42;
**eight orders** of stiffness contrast; deep re-entrant geometry (ν = −9.99); one cell or a lattice
alike. Three analytic anchors on the hexagon family — ν=1 at the regular hexagon, ν=−0.20 re-entrant,
ν=(3−d)/2 as it flattens — all reproduced without tuning.

**It fails only when the mesh violates one of two preconditions (A-17):**
1. **combinatorially closed** — every bond in exactly 2 triangles, V−E+F = 0
2. **geometrically consistent** — no inverted (negative signed-area) triangles

Either alone breaks it; they are independent. Mechanism: the metric formulation uses `q_e = Δx Δxᵀ`
and triangle *areas*, both orientation-blind. **Neither condition is checked anywhere** — and both are
a few lines. That is the single highest-value outstanding fix.

## 3. What re-running found that re-analysis could not

Seven defects, none visible in the saved outputs:

1. `g1_2` set k=1 on **fictional bracing edges**, welding its auxetic motifs shut (rotating squares
   +0.289 braced vs **−1.000** freed; reentrant honeycomb +0.303 vs **−1.083**)
2. `goal1`'s ν grid stopped at +0.45, so its reported *maximum* measured **where the search stopped**
3. `goal2`'s pool appended edge-flipped topologies **without screening**; one degenerate entry
   (min area 1.1e-16) killed **10 of 13 cases**
4. **A-17** — the solver's two mesh preconditions, unchecked
5. `goal2_attempts` was **re-analysing, not re-running** — it reloaded 115 pre-fix artifacts. Had they
   not happened to contain the degenerate geometry, it would have reported them as a fresh run
6. `run_goal1` recorded **neither seed nor trustworthiness** in its artifacts
7. **A-4**'s guard covered `designer.design()` but not the drivers' own sim calls

**Generalisable:** "add-only / resume if outputs exist" is a convenience in incremental work and a
**correctness hazard** in a re-run campaign.

## 4. What the independent oracle earned

Every defect above was caught by **solver-vs-sim disagreement** or by the sim hitting a known value the
solver missed. In two days it caught three distinct classes: the A-0 shear channel, A-17's mesh
preconditions, and **B-1 in the wild** (a routine call returning ν = +2.769 where the truth is
+2.950 — a 6 % error).

The register plans to retire the oracle "once the solver is trusted". On this evidence that is
backwards: **the oracle is the reason we know what to trust.** Retiring it should require an
argument stronger than "the solver looks fine now" — it looked fine before A-0, too.

## 5. Open

| item | status |
|---|---|
| **B-1** | **open blocker.** ~1 in 21 measurements wrong; caught at **6 %** in ordinary use, above design tolerance. Cause unknown after ~10 eliminated hypotheses. **No solver number is trustworthy without repetition or a sim cross-check.** |
| **A-17 gate** | not implemented — the highest-value cheap fix |
| **A-18** | analytic basis-cell oracle; two decisions pending (braced central-force vs a bending term; standing oracle or one-off) |
| `g1_2` physics | withheld until the four malformed topologies are fixed or excluded |
| A-8 open-boundary suite; A-14 (parked); B-4; B-5; C-6 | unchanged |

## 6. Honest limitations

- One run per case throughout; only the disorder sweep was repeated. Under B-1 that is the weakest
  part of the campaign, mitigated only by every design being sim-cross-checked.
- `random SPD` targets may simply not be realizable by these networks; their larger errors are not
  necessarily a designer limit.
- Old-vs-new comparisons are by **re-derivation**; stale `target_err_sim` values used a different
  metric definition (A-2) and are not directly comparable.
- The campaign did not touch residual stress / incompatible ḡ / curved geometry — the sim is not a
  ground truth there (FUTURE_DIRECTIONS #1).
