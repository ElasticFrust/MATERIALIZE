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
| goal1 | k-design, contrast f=0 | 26 | 22 | **[−0.823, +0.899]** |
| goal1 | k-design, f=0.1 | 26 | 15 | [−0.573, +0.650] |
| goal1 | k-design, f=0.5 | 26 | 12 | [−0.377, +0.378] |
| goal1 | k-design, f=0.9 | 26 | 10 | [−0.287, +0.343] |
| goal1 | k-design, f=0.99 | 26 | 8 | [−0.336, +0.338] |
| **goal1_frontier** | k-design, high-ν probe | 24 | 17 | **[+0.490, +0.906]** |
| g1_2 | **positions only, k ≡ 1** | 110 | 45 | [−0.436, +0.342] |
| auxetic_sweep | k-design, deep auxetic | 70 | 68 | [−0.601, +0.301] |

Landmarks drawn: the isotropic 2D bound (−1, +1), ν = 0, the uniform triangular lattice +1/3, the
η-disorder edge (−0.115), and the hexagon closed form's ν = 1 at d = 2.

## What it shows

1. **The lever is ASYMMETRIC: contrast raises the CEILING, positions supply the FLOOR.**
   *(Corrected 2026-08-22 after repairing goal1's position budget — the previous version of this
   entry said contrast was the lever at both ends, which was a search artefact.)*
   - **High ν needs contrast:** +0.899 at f = 0 against **+0.338** at f = 0.99. Geometry cannot push
     ν above the uniform-lattice value; only stiffness contrast can.
   - **Auxetic ν does not:** **every** band reaches ν < 0, including f = 0.99 at **−0.336**, where k
     is effectively frozen. Positions alone deliver it — consistent with `g1_2`'s −0.436 at k ≡ 1
     *exactly*, a stricter constraint than f = 0.99.
   - So at f = 0.99 the window is **[−0.336, +0.338]** — near-symmetric, roughly ±1/3, *not* a point
     near +1/3. Contrast widens that to [−0.823, +0.899].
2. **ν → +1 is approachable: +0.901 in the main sweep, +0.906 in the probe, both trustworthy.**
   Several runs have a solver-sim gap of *exactly* zero to four decimals. The old goal1 ceiling of
   +0.45 was purely its top grid point. ν = 1 exactly is attainable in principle — the hexagon closed
   form gives it at d = 2. **At f=0 the reachable window is now [−0.825, +0.901]: near-symmetric, and
   most of the physical range.**
3. **Positions alone cannot exceed +1/3, but reach far below it.** `g1_2` (k ≡ 1) tops out at +0.342,
   the uniform-lattice value — geometry moves ν *down*, not up. Its −0.436 is now corroborated by
   goal1's own high-f bands (−0.29 … −0.34 at f = 0.9/0.99), which are the same physical situation
   reached by a different route.
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
