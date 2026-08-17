# Campaign Stage 4 — `goal2` re-run (anisotropic tensor targets)

**Provenance.** Commit recorded in `COMMIT.txt`, clean tree. Log: `run_goal2.log`.
`goal2` targets FULL anisotropic tensors, not scalar ν — so it leaned hardest on `C_xyxy`, the
component **A-0** corrupted by 29–90 %. `Phase 5/PLAN.md` marks its saved results **INVALIDATED**.

## 1. Result — the capability was real, the artifacts were broken

| | old *claimed* (broken instrument) | old designs *re-measured* | **new (re-designed)** |
|---|---|---|---|
| physical cases | 13, all trustworthy | — | **13, all trustworthy** |
| median target err | 0.0298 | — | **0.0153** |
| mean target err | — | **0.48** | **0.0411** |
| worst | 0.1028 | (worst case 4.14) | 0.1840 |
| max solver-vs-sim gap | 0.015 | — | 0.043 |

The saved designs genuinely were invalid (0.48 when measured correctly). **Re-designing with working
instruments reaches 0.041 mean / 0.0153 median — better than the original *claim* of 0.0298.**
`goal2`'s scientific conclusion (full anisotropic tensor targets are achievable) **survives**, now
resting on a correct shear channel and an independent oracle rather than on a broken one.

By provenance, in a physically sensible order:

| provenance | mean err | reading |
|---|---|---|
| crystal | **0.0075** | targets read off real lattices — certainly realizable |
| hand orthotropic | 0.0342 | plausible designed tensors |
| random SPD | 0.1084 | hardest; some may not be realizable by these networks at all |

Positions helped in **10/13** cases (mean +0.0038) — same picture as `goal1`: k does the bulk,
positions polish.

## 2. A DRIVER defect found and fixed (not a result)

The first attempt designed only **3 of 13** cases; ten died with a byte-identical error:

```
UnhealthyGeometryError: min triangle area 1.110e-16, mean 4.330e-01
```

Diagnostic numbers: 1.110e-16 is machine epsilon (collapsed), 0.4330 = √3/4 = the area of a UNIT
EQUILATERAL triangle — so the surrounding mesh was pristine and exactly one triangle had collapsed.
A *design outcome* would vary case to case; a constant means it is baked into the inputs.

Root cause: `build_pool` appended `random_flipped_geo` results **without screening them**, and the
pool is SHARED across all cases, so one bad entry (`flipped_reg_s50`, min/mean area 2.6e-16) poisoned
every case that reached it. `run_goal1` and `run_g1_2` both search for a healthy seed before adding a
flipped variant; `goal2` did not. Fixed with the same criterion (`min > 1e-3 · mean`); s50/s52 are
degenerate, s51/s53 kept, pool still 12 topologies, worst min/mean area now 6.9e-03.

**Target generation was unaffected** — 13 physical / 6 rejected, identical to the historical run, so
the A-10 convention change did not move the physicality filter.

**Follow-up left open:** audit **A-4**'s coverage is incomplete. `verify()` is guarded inside
`designer.design()`, but `goal2`'s `design_case` reaches the sim by another path, so the error killed a
whole CASE rather than one candidate. The pool fix makes it moot here; the guard is still missing.

## 3. Limitations

- One run per case; per the B-1 constraint no single number here should be quoted without repetition
  or a sim cross-check (every case above *is* sim-cross-checked — that is the `gap` column).
- `random SPD` targets are not known to be realizable; their 0.108 mean error may be a property of the
  targets rather than of the designer.
- Positions "helping" is measured as k-only minus k+pos on the same case, one restart budget.
