# ab_quality_floor — does flooring k buy honesty, and at what cost?

**Script** `Phase 5/verifications/ab_quality_floor.py`

## What

An A/B over a **minimum-stiffness floor** applied to the designed k. The hypothesis: designs that
reach extreme ν do so by driving bonds to ~0, entering the near-mechanism regime where the solver's
read-back diverges from the independent sim (`CLAUDE.md` §3, `A(s)` rank loss). Flooring k should
prevent that — at some cost in target accuracy. 48 runs: 4 floors × 6 topologies × ν targets.

## Key numbers

| floor | n | median solver-sim gap | median \|err\| | median min triangle quality |
|---|---|---|---|---|
| 0.00 | 12 | 0.6595 | 0.1886 | 0.0868 |
| 0.02 | 12 | 0.4193 | 0.2541 | 0.0536 |
| 0.05 | 12 | 0.3672 | 0.2740 | 0.0863 |
| 0.10 | 12 | 0.3806 | 0.2877 | 0.1149 |

**The trade-off is real and monotone in the interesting direction.** Raising the floor from 0 to
0.05 nearly halves the median honesty gap (0.66 → 0.37) while the median target error rises
0.19 → 0.29. Beyond 0.05 the gap stops improving (0.367 → 0.381) but the error keeps rising, so
**0.05 is the knee**.

Topologies: triangular, honeycomb, kagome, rotating_squares, square_octagon, tetrakis.

## Limitations

- **Even at the best floor the median gap is 0.37 — far above the 0.05 tolerance.** Flooring k
  reduces the dishonesty but does not remove it; it is a mitigation, not a fix.
- 12 runs per floor is thin for a median, and the six topologies are not equally represented across
  targets.
- `minq` (minimum triangle quality) does not move monotonically with the floor, so the mechanism is
  not simply "floor k ⇒ better-shaped triangles".

## Outputs

`ab_results.json` — 48 records with `floor`, `nu_target`, `topo`, `gap`, `nu`, `err`, `minq`, `secs`.
