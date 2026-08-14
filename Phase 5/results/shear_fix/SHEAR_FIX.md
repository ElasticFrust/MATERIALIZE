# Blast radius of the shear-channel core fix, over every saved Phase 5 design

**What.** After the 2026-08-14 fix to `_compute_actual_elastic_tensor` (full account:
`documentation/shear_channel_defect.md`), re-evaluate all 50 saved designs in `Phase 5/networks/`
with the corrected code and ask two questions: does the *honesty check* change, and do the designs
still meet the targets recorded with them?

**Method.** `Phase 5/verifications/blast_radius_shear_fix.py`. For each `*.npz`: reload
(geo, k, stored meta) via `_common.load_network`, then recompute ν(θ), E(θ) on three routes —

| route | what it is | role |
|---|---|---|
| `solver` | `DesignProblem.forward` | the design path |
| `sim` | `_common.sim_per_triangle_C6` (PBC relaxation) | the designer's honesty check |
| `physical` | `physical_homog.energy_C` | **independent** energy-Hessian tensor, ground truth |

Gaps use `designer.verify()`'s relative θ-resolved form (ε_ν = 0.05, `gap_tol` = 0.05). Target error
is `max(|ν_physical − ν_target|, |E_physical − E_target|)` against the target stored in the design's
own metadata. No optimisation is re-run and no network is rewritten — the saved designs are left
exactly as they are. Deterministic: no seeds involved, since nothing is re-optimised.

**Key numbers** (full table: `blast_radius.csv`, 50 designs, 0 unsimulable).

- **Solver now agrees with the independent physical oracle: gap = 0.0000 on 47/50 designs.**
- 3 designs show a real gap and are correctly flagged untrustworthy — they passed before:
  `seed_square_octagon` 0.574, `seed_honeycomb` 0.367, `design_auxetic_4` 0.147. All are
  soft/near-mechanism networks (the tilings carry fictional edges). Designs over `gap_tol = 0.05`:
  **stored 0/50 → now 3/50**, matching the independent oracle's 3/50.
- **Target error worsened on 39 of 44 designs** carrying a target; mean 0.18 → 0.31.

| design | target err stored | corrected |
|---|---|---|
| `design_auxetic_0` | 0.027 | 0.251 |
| `design_auxetic_3` | 0.030 | 0.490 |
| `design_ds_num030_E100_2` | 0.044 | 0.543 |
| `design_demo_0` | 0.010 | 0.264 |
| `design_aniso4_2` | 0.272 | 0.431 |
| `design_ds_nup030_E100_0..2` (ν = +0.30) | 0.003–0.006 | 0.022–0.053 |

**Reading.** The forward map is now correct; what moved is the *designs*. The optimiser had been
exploiting the defect — driving k so the reported response hit the target, which the physical
network does not deliver. Designs targeting ordinary **positive ν** largely survive; **auxetic and
anisotropic** ones do not. The recorded auxetic dips in the anisotropic family were substantially
artefacts: on `design_aniso4_2` the old path reported ν(θ) reaching −0.30 where the corrected tensor
never goes below +0.13.

**Limitations.**

- Bulk only. Per-triangle C(s) is not checked here (the energy-Hessian oracle is a bulk quantity),
  so designs with *local* objectives are assessed on their global response alone.
- `target_err_sim` as stored mixes ν and E in one absolute max, so "stored" and "corrected" share
  that scale-mixing; the comparison is like-for-like but the metric itself is crude (a separate
  audit finding).
- Says nothing about whether the targets are *reachable* — only that these particular saved networks
  no longer reach them. Re-running the designs is a separate decision.

**Consequence.** The three results docs that rest on these designs — `results/goal1/GOAL1.md`,
`results/goal2/GOAL2.md`, `results/g1_2/G1_2.md` — report achieved responses computed with the old
contraction. Their anisotropic and auxetic numbers should be treated as unverified until the
experiments are re-run. Not yet done.

**Provenance.** Fix commit: see `documentation/shear_channel_defect.md`. Gate at time of run:
`test_forward_solver.py` 7/7, `test_inverse_design.py` 16/16, `Phase 5/verifications/sanity.py`
ν=1/3, E=2/√3 exact on solver and sim. Python: anaconda (numpy 2.3.5, scipy 1.16.3, torch 2.12.1+cpu).
