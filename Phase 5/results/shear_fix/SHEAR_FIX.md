# Blast radius of the shear-channel core fix, over every saved Phase 5 design

**What.** After the 2026-08-14 fix to `_compute_actual_elastic_tensor` (full account:
`documentation/shear_channel_defect.md`), re-evaluate all saved designs in `Phase 5/networks/` with
the corrected code and ask two questions: does the *honesty check* change, and do the designs still
meet the targets recorded with them?

> **Scope correction (2026-08-14).** The first run of this assessment used a non-recursive glob and
> covered only the 50 designs in the top level of `networks/`, missing the 313 in
> `networks/{goal1,g1_2,goal2,goal2_attempts,reentrant}/` — i.e. exactly the ones backing the
> results docs. Commit 318b4a2 and the first version of this doc claimed "all 50 saved designs";
> that was wrong. The numbers below are the full recursive run (363 designs assessed).
> **The corrected picture is substantially worse than the sample suggested** (mean target error
> 0.18 → 0.31 on the sample; 0.16 → 0.56 over everything).

**Method.** `Phase 5/verifications/blast_radius_shear_fix.py`. For each `*.npz`: reload
(geo, k, stored meta) via `_common.load_network`, then recompute ν(θ), E(θ) on three routes —

| route | what it is | role |
|---|---|---|
| `solver` | `DesignProblem.forward` | the design path |
| `sim` | `_common.sim_per_triangle_C6` (PBC relaxation) | the designer's honesty check |
| `physical` | `physical_homog.energy_C` | **independent** energy-Hessian tensor, ground truth |

Gaps use `designer.verify()`'s relative θ-resolved form (ε_ν = 0.05, `gap_tol` = 0.05). Target error
is `max(|ν_physical − ν_target|, |E_physical − E_target|)` against the target stored in each design's
own metadata (185 of 363 designs carry one). No optimisation is re-run and no network is rewritten.
Deterministic: nothing is re-optimised, so no seeds are involved.

## Per experiment

| experiment | n | gap stored | gap now (vs sim) | gap vs **physical** | over tol (physical) | target err stored → corrected | worsened |
|---|---:|---:|---:|---:|---:|---:|---:|
| `goal1` | 110 | 0.0121 | 0.0221 | **0.0167** | 6/110 | — | — |
| `g1_2` | 53 | 0.0151 | 0.0920 | **0.0906** | **21/53** | — | — |
| `goal2` | 39 | 0.0088 | 0.0149 | **0.0009** | 0/39 | 0.050 → **0.483** | **39/39** |
| `goal2_attempts` | 102 | 0.0792 | 0.0321 | **0.0056** | 2/102 | 0.199 → **0.706** | 98/102 |
| `reentrant` | 9 | — | 0.1134 | **0.0000** | 0/9 | — | — |
| (top level) | 50 | 0.0036 | 0.0604 | 0.0477 | 3/50 | 0.184 → 0.308 | 39/44 |
| **all** | **363** | 0.0308 | 0.0419 | **0.0266** | **32/363** | **0.164 → 0.564** | **176/185** |

Worst corrected target errors, all of them **random-SPD (strongly anisotropic) targets** — precisely
where the shear-shear component dominates:

| design | stored | corrected |
|---|---:|---:|
| `goal2_attempts/design_g2att_random_spd_s0_3_2` | 0.509 | **4.143** |
| `goal2_attempts/design_g2att_random_spd_s0_3_8` | 0.230 | 2.091 |
| `goal2_attempts/design_g2att_random_spd_s0_3_1` | 0.354 | 1.905 |
| `goal2/design_g2_random_spd_s0_3_2` | 0.101 | 1.598 |

## Reading

**The solver is right; the designs moved.** Against the independent oracle the mean gap is 0.0266
and `goal2`/`goal2_attempts`/`reentrant` sit at 0.0009–0.0056 — the forward map now agrees with
physics. What changed is that designs produced *against the old map* do not do what was recorded:
the optimiser had been exploiting the defect, driving k so the *reported* response hit the target.

**`goal2` is the most affected result: 39/39 designs worsened, mean target error ~10× (0.050 →
0.483).** It targets full anisotropic tensors, so it leaned hardest on the component that was wrong.
`goal2_attempts` is the same story at 98/102. **Its achieved numbers should be treated as
invalidated, not merely uncertain.**

**`g1_2` is the least trustworthy set: 21 of 53 saved designs now exceed `gap_tol` against the
physical oracle** (40%), versus 6/110 for `goal1`. `goal1` largely survives on the gap metric.

**A data-loss consequence.** `run_g1_2.py:330` saves **only** designs that passed the old
trustworthiness check (`if trust:`), and 58 of its 110 runs were discarded. That filter was applied
with an instrument that was ~200× too lenient in θ-resolved terms, so the rejected set cannot be
re-examined — it is gone. The same is true wherever a run saved only its survivors. This is a
concrete argument for re-running rather than re-analysing.

## Limitations

- **Bulk only.** The energy-Hessian oracle is a bulk quantity, so designs with *local/regional*
  objectives are assessed on their global response alone. Per-triangle C(s) has no independent
  oracle yet (audit finding A-9; queued).
- `target_err_sim` as stored mixes ν and E in one absolute max, so "stored" and "corrected" share
  that scale-mixing. Like-for-like, but a crude metric (audit finding A-2).
- 178 of 363 designs carry no stored target, so the target-error columns cover 185.
- Says nothing about whether the targets are *reachable* — only that these saved networks no longer
  reach them.

## Consequence

`results/goal2/GOAL2.md` and the `goal2_attempts` figures report achieved responses that the
corrected solver does not reproduce; `results/g1_2/G1_2.md`'s trustworthy set is 40% smaller against
the true oracle. `results/goal1/GOAL1.md` is the most robust of the three. All three need re-running
before their numbers are cited again — a separate, deferred decision.

**Provenance.** Fix commit 65734b0; full recursive re-run after feb77e0. Gate at time of run:
`test_forward_solver.py` 7/7, `test_inverse_design.py` 16/16, `Phase 5/verifications/sanity.py`
ν=1/3, E=2/√3 exact on solver and sim. Python: anaconda (numpy 2.3.5, scipy 1.16.3, torch 2.12.1+cpu).
