# Does A(s) conditioning explain where the solver and the sim disagree?

**Script** `Phase 5/verifications/conditioning_probe.py` · **Data** `g1_2_conditioning.csv`,
`goal1_conditioning.csv` · run 2026-08-22 (commit `a031675`, dirty)

## What

`CLAUDE.md` §3 names two routes to per-triangle `A(s)` rank loss — **dead k** and **degenerate
geometry** — and states that where `A(s)` loses rank the solver's inverse is set by its regulariser
rather than by physics. `FUTURE_DIRECTIONS` #2's re-aim proposes gating on that conditioning. Nothing
had ever measured whether conditioning actually predicts solver-vs-sim error.

This does. Per design it assembles `A(s)` **exactly as the solver does**
(`q_geom = [vx², 2vxvy, vy²]`, `A3 = Σ_e (k_e/4ℓ_e²) q_e q_eᵀ` — reproducing the solver's own packing
is the point; a re-derived A(s) would measure a different matrix from the one that gets inverted),
takes `rcond = λ_min/λ_max` per triangle, and correlates design-level tail statistics against that
design's |Δν| between solver and sim.

**Both experiments, deliberately.** In `g1_2`, k ≡ 1 exactly, so `A(s)` is a purely GEOMETRIC quantity
and the dead-k route cannot arise. Only `goal1` designs k, across five contrast bands, so only goal1
can separate the two routes — and it holds the largest disagreement on record.

## Key numbers — correlation against log10|Δν|

**g1_2** (110 designs, positions-only at k ≡ 1; 43 agree to <1e-6 and sit at the floor)

| predictor | pearson | spearman |
|---|---|---|
| **`quality_p05`** (5th-pct shape quality) | **−0.862** | **−0.859** |
| `frac_rcond < 1e-3` | +0.797 | +0.874 |
| `frac_rcond < 1e-4` | +0.782 | +0.850 |
| `area_min_over_mean` | −0.721 | −0.755 |
| `quality_min` | −0.692 | −0.790 |
| `rcond_p05` | −0.509 | −0.864 |
| `rcond_min` | −0.359 | −0.789 |
| `k_min_over_mean` | — | −0.049 (k ≡ 1: no variance) |

Worst disagreements, all at k ≡ 1 — i.e. **purely geometric**:

| \|Δν\| | ν | topology | rcond_min | q_min |
|---|---|---|---|---|
| 0.1948 | −0.215 | flipped_tri_f14 | 4.67e-10 | 0.0080 |
| 0.1890 | −0.197 | flipped_tri_f14 | 2.52e-08 | 0.0135 |
| 0.1718 | −0.310 | square_octagon | 1.16e-08 | 0.0179 |
| 0.1657 | −0.294 | square_octagon | 1.07e-13 | 0.0009 |

**goal1** (110 designs, k designed over five contrast bands; **71** agree to <1e-6)

| predictor | pearson | spearman |
|---|---|---|
| `quality_p05` | −0.363 | −0.294 |
| `quality_min` | −0.322 | −0.292 |
| `rcond_p05` | −0.297 | −0.305 |
| `rcond_min` | −0.243 | −0.313 |
| `k_min_over_mean` | +0.081 | +0.011 |

Worst disagreement **in the entire project**:

> |Δν| = **0.311** at ν = +0.286 (`tiling`) — `rcond_min` = **4.96e-03**, `q_min` = **0.475**,
> `k_min/mean` = **0.65**. Healthy on *every* axis.

## Conclusions

1. **The geometric route is confirmed, strongly — in `g1_2`.** `frac_rcond<1e-3` correlates +0.87 and
   `quality_p05` −0.86. The worst designs are slivers with sub-degree angles at uniform k.
2. **The TAIL predicts better than the worst triangle** (`quality_p05` −0.86 vs `quality_min` −0.69).
   The damage is done by *how many* bad triangles there are, not by the single worst one. So `min` is
   the right statistic for a **degeneracy floor** but the wrong one for an **accuracy criterion**.
3. **Conditioning does NOT explain goal1's worst case.** Neither route in §3 applies to it, and a
   conditioning gate would pass it cleanly. **There are at least two distinct failure modes and
   FD #2's re-aim addresses one of them.**
4. **`rcond` alone is not a valid predictor at all** — see the companion measurement in
   `../hex_validation/HEX_VALIDATION.md` §"A(s) conditioning vs accuracy": on the hexagon closed-form
   family, driving `A(s)` to numerical rank-1 with soft spokes *improves* accuracy by six orders
   (corr **+0.73**). Soft k is a benign, well-posed limit; dead k and slivers are not.

## Limitations

- **`goal1`'s correlations are unreliable**: 71 of 110 designs sit at the <1e-6 floor, so the
  response is mostly constant with a few outliers. Read the outlier table, not the coefficients.
- `goal1`'s |Δν| is a scalar difference (`nu_sim_full` vs `nu_solver_full`), not angle-resolved;
  `g1_2`'s is angle-averaged from `solver_recheck.npz`. They are not identically defined.
- Correlation is evidence for a mechanism, not proof. Both codes are linear formulations, and for
  these disordered cells no third path exists to say which is right where they differ — the hexagon
  analytic oracle covers only that family until A-18 generalises it.
- The `g1_2` half was first run against a directory being actively rewritten by the re-run and
  collected only 20 designs; those numbers were void and are not reported here. This is the clean
  re-run on the finished design set.
- No suspect identified for goal1's mode. `‖W(s)‖` was the obvious candidate and is **already
  measured elsewhere**: `verification_tools/plots/per_triangle_C/PER_TRIANGLE_C.md` records
  corr(|ΔC|, ‖W‖) = **+0.39**, with an explicit note that an earlier draft overstating it as "tracks
  ‖W‖" was corrected. That residual is the contraction-isolation one, not end-to-end disagreement, so
  it does not transfer directly — but it is the prior art to start from.
