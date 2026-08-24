# fig1 — the frozen showcase network was STALE

**Date** 2026-08-17 · **Script** `fig1_redesign_check.py` · **Case** `disorder_hi`, N=16
(3456 bonds / 2304 triangles), `reg=5e-3`, target ν(θ) copied from the φ=4, ψ=1 crystal
(`optimize` seed 0, 5 restarts × 500 L-BFGS iterations).

## What

`fig1_recreate_pointy.py` caches its design in `fig1_pointy_network.npz`, **frozen 2026-07-12**, and
renders whatever that file contains — by design (`design_or_load`, the save-then-load policy). The
August audit changed the forward map (**A-0** shear contraction, **A-10** ν convention), so the
committed figure shows a *July design read by August instruments*.

This was noticed from the plot, not from the process: the regenerated `fig1_recreate_pointy.png`
showed ν(θ) swinging to **[−18.1, +19.1]** against a target of **[−3.63, +7.18]**.

It is **not** an optimiser failure and **not** a rendering failure. It is `Phase 5/PLAN.md`'s
"saved designs are stale — re-run, don't re-analyse", applying to *Phase 3's* frozen showcase
networks, where it had not been noticed. Three such files exist here: `fig1_pointy_network.npz`,
`fig1c_rot90_regular_network.npz`, `fig1d_rot90_disorder_network.npz` — all July.

## Method

1. Evaluate the **frozen July k** with today's solver and compare against the C6 July recorded.
2. Redesign from scratch at HEAD, running fig1's five restarts **one at a time** (`seed=0+r`,
   `n_restarts=1`, equivalent to `optimize`'s own loop, which calls `_init_raw(prob, mode, seed+r)`)
   so each is timed and scored separately.
3. Check the redesign against the **independent sim** (`sim_bulk_C6` → virial), not the solver alone.

## Key numbers

**The forward map moved for a fixed k.** Same frozen k, July's record vs today:

| | xx,xx | xx,xy | xx,yy | xy,xy | xy,yy | yy,yy |
|---|---|---|---|---|---|---|
| stored (July) | 0.00996 | 0.00228 | 0.00143 | 0.00453 | 0.00359 | 0.00326 |
| today, same k | 0.17135 | 0.04097 | 0.02333 | 0.07704 | 0.06542 | 0.05662 |
| ratio | 17.2 | 18.0 | 16.3 | 17.0 | 18.2 | 17.4 |

The **~10 % spread between components** is what matters. A uniform factor would leave ν untouched;
the spread is what moves ν(θ) from [−2.42, +2.80] to [−18.1, +19.1].

**July never met its target either** — its own stored C6 gives ν peaking at **+2.80** against a
target of **+7.18**. The legacy figure looked plausible only because its y-axis was ±4.

**The redesign meets it, and all five restarts agree** (so it is not a lucky basin):

| restart | seconds | loss | rms(ν−target) | k median | k min | dashed (k<0.02·med) |
|---|---|---|---|---|---|---|
| 0 | 473.6 | 4.906e-04 | 1.6e-03 | 0.249 | 2.3e-40 | 925/3456 |
| 1 | 474.2 | 5.628e-04 | 1.4e-03 | 0.402 | 2.6e-43 | 1095/3456 |
| 2 | 494.4 | 7.503e-04 | 1.1e-03 | 0.690 | 2.6e-55 | 1046/3456 |
| 3 | 448.3 | 6.003e-04 | 1.8e-03 | 0.676 | 2.9e-40 | 1022/3456 |
| **4** | 400.9 | **1.409e-04** | **1.7e-04** | 0.179 | 6.3e-40 | 950/3456 |

Best (restart 4), against the **independent sim**:

```
rms(sim − target)        = 1.745e-04      ν_sim ∈ [−3.6333, +7.1798]   (target [−3.6334, +7.1802])
max |ν_solver − ν_sim|   = 4.786e-09
relative max |ΔE|        = 8.966e-10
```

## Limitations

- **The design is floppy.** ~27 % of bonds end below 0.02 × median k, and k_min ≈ 6e-40. It hits the
  target exactly and the independent sim agrees to 5e-09, so it is sound *as a solution* — but
  `reg=5e-3` is below the **0.01–0.05** `CLAUDE.md` §3 recommends, and that is the knob for a less
  degenerate design at the same target. Not changed here: matching fig1's published configuration
  was the point.
- **The July artifacts are left untouched** (`fig1_pointy_network.npz`, `fig1_recreate_pointy.png`).
  Replacing fig1's cache is a separate decision. *(The `*_legacy_prePlotPolicy.png` tombstones this
  line also used to name were deleted 2026-08-24 — see `AUDIT_2026-08.md` B-4; git history keeps them.)*
- **`fig1c` / `fig1d` are equally stale** and not addressed here.
- The E(θ) of this design is tiny (peak ≈ 0.0065) — expected, since only ν was targeted; fig1's own
  title records the design as "~50× softer".

## The plot-policy defect this exposed (audit B-4) — now FIXED in `plotting.py`

`draw_network` coloured bonds with a linear norm over **[k_min, k_max]**. Designed k is heavy-tailed
— k_max/median ≈ **17** here and ≈ **147** for the July network — so essentially every bond landed in
the bottom few percent of viridis and a 3456-bond panel rendered as a flat dark mass, in which the
~30 % of bonds *correctly* drawn dashed read as a torn mesh. That is what made the B-4 regenerated
figures look like they had "missing parts".

**The data was never missing and `draw_network` was never wrong** — verified: zoomed to a 6×6-unit
window it gives a clean fully-connected triangulation, and bond lengths match `actual_len2` exactly.
**The norm was the defect, not the dashes**; once the scale is fixed the dash rule reads as intended,
marking dead bonds distinctly.

Settled 2026-08-17 (`CLAUDE.md` §3): the colour scale is cut at `STYLE.K_HI_PCT = 90`, **a percentile,
never the max**; bonds above it saturate, so the true max goes in the panel title and the colorbar
carries `extend='max'`. `draw_network(..., scale='log')` is available for large contrast — at
max/median ≈ 147 even the 90th-percentile *linear* cut still leaves the bulk dark — and its range is
taken over **live bonds only**, since the dead population (~1e-40) would otherwise span ~18 decades.
The figures here use the policy default; no local deviation.

A second, unrelated rendering bug was found and fixed while checking these figures: `_polar_nu`
NaN-masked the ν>0 and ν<0 branches apart, so neither reached r = |ν| = 0 and the polar curve **tore
open at every zero crossing**. `_insert_zero_crossings` now adds the interpolated crossing angle;
verified against `cos(2θ)` (exactly 4 insertions, at 45/135/225/315°). Every previously-generated
polar ν figure whose ν changes sign is cosmetically affected.

## Figures

- [`fig1_stale_vs_fresh.png`](fig1_stale_vs_fresh.png) — July-read-today vs July-as-recorded vs the redesign, against the target
- [`fig1_redesigned_response.png`](fig1_redesigned_response.png) — ν(θ), E(θ) cartesian + polar; solver and sim share each panel (they coincide, the solver line sits under the sim)
- [`fig1_redesigned_network.png`](fig1_redesigned_network.png) — the designed network

## Artifacts

- `fig1_pointy_network_redesigned.npz` — the redesign (k + per-triangle C6 + provenance: commit, dirty, saved_utc, seed)
- `fig1_redesign_restarts.npz` — the restart table above
