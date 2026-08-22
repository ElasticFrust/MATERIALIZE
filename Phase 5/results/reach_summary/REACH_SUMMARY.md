# What Poisson ratio has this project actually achieved?

**Script** `Phase 5/verifications/plot_reach_summary.py` · **Figure**
[`reach_summary.png`](reach_summary.png) · rendered 2026-08-22 (commit `62b69f7`, dirty)

## What

The reachable-ν evidence was scattered across four experiments with different setups, and the one
holding the **highest ν on record** — `goal1_frontier`, at **+0.906** — had no figure at all; its
result existed only as a table. This puts all of them on one ν axis against the physical bound.

Loads and renders only — nothing is re-optimised. Rendering goes through the root `plotting.py`
primitive **`plot_ranges`**, added for this figure rather than rolled locally (three call sites
already wanted an interval chart; that is what the single-source-of-truth policy exists to prevent).

## Key numbers

| group | design freedom | n | agree | reach (independent sim) |
|---|---|---:|---:|---|
| goal1 | k-design, contrast f=0 | 22 | 20 | [−0.789, +0.450] |
| goal1 | k-design, f=0.1 | 22 | 16 | [−0.179, +0.450] |
| goal1 | k-design, f=0.5 | 22 | 20 | [+0.000, +0.400] |
| goal1 | k-design, f=0.9 | 22 | 19 | [+0.156, +0.345] |
| goal1 | k-design, f=0.99 | 22 | 18 | [+0.196, +0.333] |
| **goal1_frontier** | k-design, high-ν probe | 24 | 17 | **[+0.490, +0.906]** |
| g1_2 | **positions only, k ≡ 1** | 110 | 45 | [−0.436, +0.342] |
| auxetic_sweep | k-design, deep auxetic | 70 | 68 | [−0.601, +0.301] |

Landmarks drawn: the isotropic 2D bound (−1, +1), ν = 0, the uniform triangular lattice +1/3, the
η-disorder edge (−0.115), and the hexagon closed form's ν = 1 at d = 2.

## What it shows

1. **Stiffness contrast is the lever, on BOTH ends.** Reach collapses monotonically as the contrast
   floor `f` rises: [−0.789, +0.450] at f=0 → [+0.196, +0.333] at f=0.99, i.e. onto the uniform
   lattice value. Forbid contrast and the design space shrinks to a point near +1/3.
2. **ν → +1 is approachable: +0.906, trustworthy.** Several of the frontier runs have a solver-sim
   gap of *exactly* zero to four decimals. The old goal1 ceiling of +0.45 was purely its top grid
   point. ν = 1 exactly is attainable in principle — the hexagon closed form gives it at d = 2.
3. **Positions alone cannot exceed +1/3.** `g1_2` (k ≡ 1) tops out at +0.342, the uniform-lattice
   value: geometry redistributes ν downward but cannot raise it. High ν needs *k contrast*.
4. **The auxetic side is the mirror image.** −0.601 (`auxetic_sweep`, k-design) and −0.789 (goal1
   f=0) with contrast; −0.436 with positions alone.
5. **The white markers are where the two codes disagree**, and they cluster on the auxetic side of
   `g1_2` — reach that is real (49/49 sign-agree) but not gate-passing. See
   [`../g1_2/G1_2.md`](../g1_2/G1_2.md).

## Limitations

- **Not a frontier.** Each interval is where *these runs* landed, bounded by the ν grid each
  experiment used — goal1's grid stops at +0.45, which is exactly the censoring `goal1_frontier`
  was built to expose. Read the bars as "reached", never as "reachable".
- **The groups are not comparable like-for-like**: different topology pools, cell sizes, budgets and
  ν grids. The contrast-band rows *are* mutually comparable (same run, same pool).
- `auxetic_sweep` has no solver-sim gap column; its `stable` flag is used for the solid arm, which is
  a different criterion from `gap < 0.05` used by the others.
- goal1 and goal1_frontier predate the **scored selection** (2026-08-22) and may understate reach the
  way g1_2 did — g1_2's floors moved substantially once the veto was removed. Re-running both under
  the new rule, over a symmetric −1 < ν < 1 grid, is the obvious next step.
