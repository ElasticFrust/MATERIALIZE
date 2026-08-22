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
| goal1 | k-design, contrast f=0 | 26 | 24 | **[−0.825, +0.901]** |
| goal1 | k-design, f=0.1 | 26 | 23 | [−0.296, +0.658] |
| goal1 | k-design, f=0.5 | 26 | 22 | [+0.040, +0.385] |
| goal1 | k-design, f=0.9 | 26 | 21 | [+0.130, +0.343] |
| goal1 | k-design, f=0.99 | 26 | 24 | [+0.179, +0.339] |
| **goal1_frontier** | k-design, high-ν probe | 24 | 17 | **[+0.490, +0.906]** |
| g1_2 | **positions only, k ≡ 1** | 110 | 45 | [−0.436, +0.342] |
| auxetic_sweep | k-design, deep auxetic | 70 | 68 | [−0.601, +0.301] |

Landmarks drawn: the isotropic 2D bound (−1, +1), ν = 0, the uniform triangular lattice +1/3, the
η-disorder edge (−0.115), and the hexagon closed form's ν = 1 at d = 2.

## What it shows

1. **Stiffness contrast is the lever FOR k-DESIGN.** goal1's reach narrows monotonically as the
   contrast floor rises: **[−0.825, +0.901] at f=0 → [+0.179, +0.339] at f=0.99**. The grid is
   symmetric at every band, so this is not a grid artefact — **but it IS partly a search artefact.**
   `design_iso` runs positions at the library defaults (a=0.02), giving 0.202 lattice spacings of
   travel against g1_2's 2.68 — **13× less**, effectively off. At high f, k is pinned and positions
   cannot move either, so those rows report little more than the seed's ν. **Do not conclude that the
   reachable set collapses without contrast**: the g1_2 row below reaches −0.436 at k ≡ 1 *exactly*,
   which is more restrictive than f = 0.99. Positions are a second lever goal1 barely used.
2. **ν → +1 is approachable: +0.901 in the main sweep, +0.906 in the probe, both trustworthy.**
   Several runs have a solver-sim gap of *exactly* zero to four decimals. The old goal1 ceiling of
   +0.45 was purely its top grid point. ν = 1 exactly is attainable in principle — the hexagon closed
   form gives it at d = 2. **At f=0 the reachable window is now [−0.825, +0.901]: near-symmetric, and
   most of the physical range.**
3. **Positions alone cannot exceed +1/3, but reach far below it.** `g1_2` (k ≡ 1) tops out at +0.342,
   the uniform-lattice value — geometry moves ν *down*, not up, so high ν does need k contrast. On the
   auxetic side positions alone reach **−0.436**, deeper than goal1's f=0.9/0.99 rows manage with k,
   which is the clearest evidence that those rows are search-limited rather than physics-limited.
4. **The auxetic side is the mirror image.** −0.825 (goal1 f=0) and −0.601 (`auxetic_sweep`) with
   contrast; −0.436 with positions alone. At f=0 the window [−0.825, +0.901] is near-symmetric.
5. **The white markers are where the two codes disagree**, and they cluster on the auxetic side of
   `g1_2` — reach that is real (49/49 sign-agree) but not gate-passing. See
   [`../g1_2/G1_2.md`](../g1_2/G1_2.md).

## Limitations

- **Not a frontier.** Each interval is where *these runs* landed, bounded by the ν grid each
  experiment used. goal1's grid is now symmetric over [−0.95, +0.95] so its bars are no longer
  clipped by the search edge, but `g1_2` and `auxetic_sweep` still carry their own grids. Read the
  bars as "reached", never as "reachable".
- **The groups are not comparable like-for-like**: different topology pools, cell sizes, budgets and
  ν grids. The contrast-band rows *are* mutually comparable (same run, same pool).
- `auxetic_sweep` has no solver-sim gap column; its `stable` flag is used for the solid arm, which is
  a different criterion from `gap < 0.05` used by the others.
- **CORRECTED 2026-08-22:** an earlier version of this file said goal1 "predates the scored selection
  and may understate reach the way g1_2 did". That was wrong — `run_goal1` never vetoed on the gap; it
  records the gap as a number and discards nothing, so its reach was never censored. Its only
  limitation was the **grid**, and goal1 has now been re-run on a symmetric 13-point grid over
  [−0.95, +0.95] (130 runs), which is what the table above reflects. `goal1_frontier` is now
  **subsumed** by that sweep and is retained only as the independent cross-check of the f=0.1
  frontier (+0.63 there vs +0.658 in the symmetric run, different grids).
