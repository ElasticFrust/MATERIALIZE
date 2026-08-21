# ab_quality_floor — what does a SHAPE floor cost in reach?

**Script** `Phase 5/verifications/ab_quality_floor.py`

> ### ⚠ CORRECTION 2026-08-22 — this document previously described the wrong experiment
> The earlier version of this file called it *"an A/B over a **minimum-stiffness floor** applied to
> the designed k"*. **It is not.** The script patches `positions.spsa_positions` to inject
> **`quality_floor`** — a per-triangle SHAPE-QUALITY floor that rejects sliver-producing SPSA steps
> inside the optimizer — and then runs the real `run_g1_2.design_one`. No k floor is involved
> anywhere. `FUTURE_DIRECTIONS` #2 repeated the same mislabel ("a `min(k)` floor, this entry's cheap
> S option") and has been corrected too.

## What

`quality_floor` rejects any SPSA step that drives a triangle's shape quality
(`q = 4√3·Area/Σedge²`, 1 = equilateral, 0 = degenerate) below the floor. It exists because the
in-loop **area** guard does not work: on the saved `g1_2` designs the untrusted ones have median
min/mean area 0.047, i.e. **47× above the 1e-3 area floor**, so it essentially never fires. A sliver
is an *angle* failure, and area is a poor proxy — measured against log10(gap): shape quality −0.523,
min angle −0.505, area −0.418, min edge length −0.107 (useless).

**A floor RESTRICTS THE DESIGN SPACE**, so the honest question is not "does trustworthiness rise" but
**what does it cost in reach**. If trustworthiness and target error move together, the floor is
forbidding the designs that were the point.

## Result — 2026-08-22, under the SCORED selection

40 runs: 4 topologies × 2 targets (ν* = −0.30, +0.10) × 5 floors. Matched pairs — `design_one` seeds
itself identically per topology/target, so the floor is the only difference.

| floor | ≈ min angle | trustworthy | median gap | median err | median minq |
|---|---|---|---|---|---|
| 0.000 | — | 38% | 0.1034 | **0.0061** | 0.0223 |
| **0.001** | ~0.03° | 38% | 0.1020 | **0.0043** | 0.0247 |
| 0.010 | ~0.3° | 38% | 0.0840 | 0.0490 | 0.0319 |
| 0.030 | ~1.0° | **75%** | 0.0000 | 0.1028 | 0.0648 |
| 0.050 | ~1.7° | **75%** | 0.0000 | 0.1236 | 0.0730 |

**Decision: default `quality_floor = 1e-3`** (set in `positions.spsa_positions` and
`design_with_positions`, 2026-08-22).

- **1e-3 is free.** Trustworthy fraction unchanged, gap unchanged, median error within noise. It is a
  DEGENERACY guard, not an accuracy gate: only **2 of the 110** g1_2 designs sit below it
  (`q_min` = 4.3e-04 kagome and 9.2e-04 square_octagon — both with |Δν| > 0.1), and neither is its
  topology's extreme, so the reach table is not expected to move.
- **0.01 is a bad trade**: gap −19%, error ×8.
- **0.03 and above buy agreement by abandoning the objective**: trustworthiness doubles and the gap
  goes to zero, but median error rises **×17**. Both move together — the stated failure signature.
  It would also destroy the deepest auxetic result in the project (ν = −0.436 at `q_min` = 0.0090,
  ≈ 0.3°). **Do not raise the floor to buy agreement** — that is the veto mistake one level down.

## Caveats

- **n = 8 per floor.** The medians are thin and the 0.001 arm's apparent improvement is noise.
  The claim is "costs nothing", not "helps".
- `minq` does not move monotonically with the floor in the old run, which suggests the floor changes
  *which restart wins* rather than only what is permitted.
- `min` is the right statistic for a degeneracy floor but the **wrong** one for an accuracy criterion:
  on the full 110-design set the 5th-percentile quality predicts the error far better than the worst
  triangle (`quality_p05` corr −0.86 vs `quality_min` −0.69). The damage is done by a *tail* of bad
  triangles, not by one.

## The earlier run (old VETO selection) — kept for comparison

Floors 0.00 / 0.02 / 0.05 / 0.10 over 6 topologies, before `design_one` was changed to scored
selection: median gap 0.6595 / 0.4193 / 0.3672 / 0.3806, median err 0.1886 / 0.2541 / 0.2740 / 0.2877.
Same qualitative verdict (gap and error move together), at roughly an order of magnitude worse on
both axes. The comparison is confounded — different topology count and floor grid — so read the
direction, not the ratio.

## Outputs

`ab_results.json` — 40 records with `floor`, `nu_target`, `topo`, `gap`, `nu`, `err`, `minq`, `secs`.
The pre-2026-08-22 (48-record, veto-selection) version is recoverable from git at `aff69d9`.
