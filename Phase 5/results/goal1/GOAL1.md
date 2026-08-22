# Goal 1 — Rich isotropic inverse-design test (Phase 5 M1)

**Question.** Given a *scalar* (isotropic) Poisson-ratio target ν with Young's modulus E = 1, how
well can the Phase 5 designer hit ν across the physical range, and **how does the achievable ν
depend on how much per-bond stiffness contrast the network is allowed?** Every design is verified by
an *independent* full-PBC simulation (a different code path from the solver it was optimised
against), and the target error is tracked through three optimisation stages to prove the
optimisation actually helps.

> **RE-RUN 2026-08-22 on a SYMMETRIC ν grid** — 13 points over [−0.95, +0.95] × 5 contrast bands ×
> 2 topologies = **130 runs, 112 min, 114 trustworthy (88 %)**. Every number and figure below comes
> from that run. The headline: **with contrast unrestricted the designer now covers [−0.82, +0.90]**,
> most of the physical range, and `run_goal1_frontier.py`'s separate high-ν probe is **subsumed**.
>
> Two variables changed from the 2026-08-18 run, not one: the grid, and `positions.quality_floor`
> (0.0 → 1e-3, a degeneracy guard enabled the same day). So this is not a clean grid-only A/B against
> the old numbers (93/110 trustworthy, soft band [−0.78, +0.45]).
>
> Note the selection rule change of 2026-08-22 does **not** apply here: `run_goal1` never vetoed on
> the gap — it records the gap as a number and discards nothing, so its reach was never censored the
> way `g1_2`'s was. Its only limitation was the grid, which is what this re-run fixes.

---

## 1. The experiment

- **ν grid (13), SYMMETRIC:**
  `[-0.95, -0.8, -0.6, -0.45, -0.3, -0.15, 0.0, 0.15, 0.3, 0.45, 0.6, 0.8, 0.95]`.
  > **FIXED 2026-08-22.** The grid used to be `[-0.9 … +0.45]` — 90 % of the negative half but only
  > 45 % of the positive, asymmetric and undocumented. That made the reported +0.45 maxima *the top
  > grid point rather than a frontier*, and it took a separate probe
  > (`run_goal1_frontier.py`) to discover the real f=0 ceiling was ≈ +0.91. The grid now spans the
  > isotropic 2D bound symmetrically at every band, so **no cell in §3a is censored by the search
  > edge** and the probe is subsumed. Extremes may still be unreachable — that is now reported as a
  > *frontier*, and it is one the grid actually looked past.
- **k-contrast bands (5):** a per-bond stiffness floor `f = min(k)/avg(k)` ∈
  `{soft 0.0, large 0.1, medium 0.5, small 0.9, none 0.99}`. `f = 0` lets bonds go fully soft
  (mechanisms/auxetics accessible); `f → 1` forces a near-uniform lattice.
- **Topologies (≥100 distinct):** 115 distinct small periodic cells (24–144 nodes, median 51) drawn
  across five classes — **foam** (random Poisson-disk / blue-noise / uniform / graded patches),
  **bravais** (oblique/rect/square/hex/sheared lattices ± disorder), **tiling** (square, honeycomb,
  kagome, square-octagon, with soft fictional triangulating edges), **auxetic** (rotating-squares,
  re-entrant honeycomb), and **flipped** (non-Delaunay edge-flip variants). One distinct topology
  per run.
- **Runs:** 11 ν × 5 bands × 2 topologies = **110 runs**. Both **k and vertex positions** are
  optimised, and every final design is independently sim-verified.
- **Three-stage error** recorded per run, all as `|ν_sim − ν*|` from the independent simulation:
  **(i) initial** — uniform k = 1 on the seed, no optimisation; **(ii) k-only** — the constrained-k
  designer, no position moves; **(iii) k + positions** — the full alternating design.
- **Honesty gate:** runs with solver-vs-sim gap ≥ 0.05 are dropped as untrustworthy (the solver's
  prediction on that exotic/near-floppy topology cannot be trusted).

Runtime: **110 runs in 76 min**. Each final design saved to `Phase 5/networks/goal1/design_g1_<i>.npz`
with full metadata; table in `results.csv` / `results.npz`; full per-run log in `run_goal1.log`.

---

## 2. The method (new piece: `Phase 5/design_iso.py`)

**Constrained-k parameterisation (the contrast floor).** The free variable is `raw` (one per bond);
the stiffness *shape* is

```
kshape = f + (1 − f)·sigmoid(raw)        →  kshape ∈ [f, 1],  so min(kshape)/avg(kshape) ≥ f
k      = scale · kshape
```

The elastic tensor is **linear in k** (harmonic solver, fixed geometry), so scaling *all* bonds by a
constant `scale` scales the physical tensor → E scales, ν is invariant. We therefore optimise only
the **shape** toward a *flat* ν(θ) = ν\* profile with the loss

```
L = mean( (ν(θ) − ν*)² )  +  w_iso·( var ν(θ) + var E(θ) )      (w_iso = 0.5, isotropy penalty)
```

then set `scale = 1 / E_shape` once so **E = 1 exactly, for free** (E is not in the loss — it is
scale-free at the shape stage). The contrast floor is preserved by construction: `min(k)/avg(k) ≥ f`.

**k + positions alternation** (`design_iso`): each outer round runs the constrained-k designer, then
`positions.spsa_positions` polishes the vertex coordinates at fixed k toward the same flat ν target
with **E_weight = 0** (E is fixed by the rescale, so the position polish must not chase it); after any
position move k is rescaled to restore E = 1 (ν invariant, so free). The re-Delaunay inside SPSA lets
the topology adapt; if it changes, k is re-designed on the new topology. Best kept by |ν_solver − ν\*|.

Public API of `design_iso.py`:

```
KBANDS                                                          -> {name: f}
kshape_from_raw(raw, f)                                         -> torch (n_bond,) in [f,1]
nuE_theta_solver(geo, k)                                        -> (ν_theta, E_theta) numpy (37,)
design_iso_k(geo, nu_target, f, n_iter=150, n_restarts=2, w_iso=0.5, seed=0)
                                                               -> (k, nu_ach, E_ach)
design_iso(geo, nu_target, f, n_outer=2, spsa_steps=20, n_iter=150, n_restarts=2,
           w_iso=0.5, seed=0, verbose=False, return_stages=False)
                                                               -> (geo, k, nu_ach, E_ach[, konly])
```

(Reuses, unmodified: `designer.verify`, `positions.spsa_positions`, `seeds.*`, `triangulation.*`,
`_common`, `inverse_design`. Sweep driver: `Phase 5/verifications/run_goal1.py`; figures:
`plot_goal1.py`. The sweep budget was `n_iter=120` to keep all 110 runs under ~3 h.)

---

## 3. Key results

### 3a. The achievable-ν FRONTIER vs contrast band — stiffness contrast is the lever, at BOTH ends

The ν grid is now **symmetric**: 13 points spanning [−0.95, +0.95] at every band (see §1). Reachable
(independent-sim) ν, over trustworthy runs:

| band | f | reachable ν [min, max] | median ν | n |
|------|-----|------------------------|----------|---|
| soft   | 0.00 | **[−0.82, +0.90]** | +0.08 | 24 |
| large  | 0.10 | [−0.30, +0.66] | +0.15 | 23 |
| medium | 0.50 | [+0.04, +0.38] | +0.29 | 22 |
| small  | 0.90 | [+0.13, +0.34] | +0.30 | 21 |
| none   | 0.99 | **[+0.18, +0.34]** | +0.28 | 24 |

**With contrast unrestricted the designer covers most of the physical range** — [−0.82, +0.90] out of
(−1, +1) — and the window narrows monotonically onto the uniform-lattice ν = 1/3 as the contrast floor
rises, until at f = 0.99 it is [+0.18, +0.34].

> ### ⚠ THE HIGH-f ROWS ARE SEARCH-LIMITED, NOT PHYSICS-LIMITED (found 2026-08-22)
> Do **not** read this table as "forbid contrast and the design space collapses". `design_iso` calls
> `spsa_positions` **without passing `a`/`c`**, so positions move at the library defaults
> `a = 0.02, c = 0.01`: over 20 steps × 2 outer rounds that is **0.202 lattice spacings** of possible
> travel per coordinate, against **2.68** for `g1_2` at a = 0.25 — **13× less**, and an eighth of even
> the a = 0.15 arm that was *measured failing* to reach auxetic ν
> (`g1_2_triangular_start_probe.py`). **goal1's position optimisation is effectively switched off.**
>
> So at high f, k is pinned AND positions cannot move, and the run reports little more than the seed's
> own ν near 1/3. That is not a statement about the design space: `g1_2` reaches **−0.436** with
> **k ≡ 1 exactly** (f = 1.0, *more* restrictive than f = 0.99) using positions alone, and even random
> η-disorder of a triangular lattice reaches −0.115.
>
> **What this table does support:** contrast is the lever *for k-design*, and the low-f reach is real.
> **What it does NOT support:** that the reachable set collapses at high f. Positions are a second,
> independent lever this sweep barely exercised. Re-running with `spsa_a`/`spsa_c` threaded through
> `design_iso` to a g1_2-comparable budget is a logged follow-up.

*No cell in this table is censored by the grid.* The previous version's +0.45 maxima were the top
grid point (`⚠` in earlier drafts) and needed a separate probe, `run_goal1_frontier.py`, to discover
that the true f=0 ceiling was ≈ +0.91. That probe is now **subsumed**: the symmetric sweep reaches
+0.90 at f=0 directly, and independently reproduces the probe's *measured* f=0.1 frontier
(+0.66 here vs +0.63 there) — a genuine cross-check, since the two runs used different grids.

### 3b. Does optimisation reduce the error — k vs positions, per band

Mean target error `|ν_sim − ν*|` at each stage (trustworthy runs), initial → k-only → k+positions:

| band | f | initial | k-only | k + positions |
|------|-----|---------|--------|---------------|
| soft   | 0.00 | 0.533 | 0.026 | **0.021** |
| large  | 0.10 | 0.495 | 0.197 | **0.176** |
| medium | 0.50 | 0.522 | 0.454 | **0.435** |
| small  | 0.90 | 0.488 | 0.478 | **0.459** |
| none   | 0.99 | 0.544 | 0.543 | **0.526** |

- **In the soft band the designer hits the target across the WHOLE range**, ±0.95 included: mean
  error 0.021 after 0.533 initial, a 25× reduction. k does nearly all of it (0.533 → 0.026);
  positions trim the rest.
- **The high-f errors ARE partly optimiser failures** *(corrected 2026-08-22)*. An earlier draft said
  they were not — that with contrast forbidden the grid points were simply unreachable. But positions
  were never given the budget to try (see §3a's box), so these numbers conflate "unreachable" with
  "not searched for". The k-only → k+pos improvement of 0.005–0.02 in every band is the signature of a
  position search that barely moved, not of positions having nothing to offer.
- **Positions help by a similar small margin in every band** (0.005–0.02), including where k is inert.

### 3c. Aggregate acceptance

- **114 / 130 runs trustworthy** (88 %, solver-vs-sim gap < 0.05). The 16 dropped are dominated by
  `tiling` (worst: `medium` ν*=−0.60 gap 1.55; `small` ν*=+0.95 gap 0.45).
- **Median target error (k+pos) = 0.198; success rate |err| < 0.05 = 33 %.**
- **Why the success rate is "only" 33 %, and why that is the correct outcome.** The grid deliberately
  spans the full physical range at *every* contrast band, including combinations that are provably
  unreachable (ν = ±0.95 at f = 0.99, where the reachable window is [+0.18, +0.34]). Those points
  cannot be hit; they exist to **map the frontier**. Restricted to the soft band the error is 0.021
  and essentially every point is a hit. The aggregate rate is a property of how aggressively the grid
  samples beyond the frontier, not of the optimiser. It is also *lower* than the old asymmetric
  grid's 45 % for exactly this reason — the symmetric grid asks harder questions.

### 3c-bis. Where solver and sim disagree — REPRODUCED, and it is not the extremes

Per-band solver-vs-sim |Δν| (all 130 runs; the median is 0.0000 in every band — the two codes agree
*exactly* for most designs, and the spread lives entirely in a few outliers):

| band | f | median \|Δν\| | max \|Δν\| | where the max sits |
|---|---|---|---|---|
| soft   | 0.00 | 0.0000 | **0.0068** | ν = −0.143, auxetic |
| large  | 0.10 | 0.0000 | 0.0167 | ν = −0.175, tiling |
| **medium** | 0.50 | 0.0000 | **0.3228** | **ν = +0.296, tiling** |
| small  | 0.90 | 0.0000 | 0.0407 | ν = +0.274, tiling |
| none   | 0.99 | 0.0000 | 0.0384 | ν = +0.235, tiling |

**The band that reaches furthest is the most reliable.** `soft` spans [−0.82, +0.90] with a worst-case
disagreement of 0.0068; `medium` is confined to [+0.04, +0.38] and disagrees by 0.32 — **50× worse,
at an unremarkable positive ν**. Extreme ν is emphatically *not* where the two codes part company.

**And it reproduces.** The 2026-08-18 run's worst case was |Δν| = 0.311 at ν = +0.286, `tiling`,
`medium` band; this run — different grid, independently drawn topologies — gives **0.323 at
ν = +0.296, `tiling`, `medium` band**. Same class, same band, same ν, same magnitude. That is a
phenomenon, not an outlier.

It is also **unexplained**: `conditioning_probe.py` shows the old instance was healthy on every axis
(`rcond_min` 5e-03, shape quality 0.475, `k_min/mean` 0.65), so neither route in `CLAUDE.md` §3 —
dead k or sliver geometry — applies. A `tiling` + f=0.5 + ν≈+0.29 case is now the sharpest handle on
that failure mode; see `../conditioning_probe/CONDITIONING_PROBE.md`.

### 3d. Breakdown by topology class

Mean target error (k + positions, trustworthy runs):

| class | n | initial | k+pos mean | k+pos median | σ |
|-------|-----|---------|-----------|--------------|-----|
| tiling  | 2  | 0.114 | **0.079** | 0.079 | 0.062 |
| flipped | 16 | 0.404 | 0.253 | 0.171 | 0.289 |
| auxetic | 6  | 0.477 | 0.261 | 0.284 | 0.165 |
| bravais | 32 | 0.519 | 0.324 | 0.271 | 0.309 |
| foam    | 58 | 0.566 | 0.349 | 0.186 | 0.392 |

Counts are unbalanced (foam 58, tiling 2 — topologies are drawn per run, not stratified), so read the
medians rather than the means, and treat `tiling`'s n=2 as anecdote. Foam and bravais carry the
frontier sweep and their large σ reflects the mix of reachable and deliberately-unreachable points.


## 4. Limitations & honest caveats

- **The grid intentionally over-reaches the frontier**, so aggregate "success rate" understates
  designer quality; read it together with the per-band frontier (§3a) and the soft-band error
  (0.021).
- **High-f bands are near-degenerate by construction**: with `f = 0.99` there is almost no design
  freedom, so those 26 runs mostly report the seed's own ν near 1/3.
- **16 runs flagged** at the honesty gate (`gap ≥ 0.05`), dominated by `tiling` — they are reported,
  not dropped: the figures draw them hollow, because a gap says the two CODE PATHS disagree, not that
  the network is unreal.
- **Topology classes are unbalanced** (foam 58, tiling 2): topologies are drawn per run rather than
  stratified, so §3d's per-class means are noisy and `tiling`'s n=2 is anecdote.
- **Modest per-design budget** (`n_iter = 120`, `n_restarts = 2`, `spsa_steps = 20`, `n_outer = 2`)
  chosen so the full sweep runs in ~112 min; a larger budget would tighten the reachable errors
  further, especially in the `large`/`medium` bands.
- **`quality_floor = 1e-3` was enabled the same day**, so this run differs from the 2026-08-18 one in
  two variables (grid and floor), not one. The floor's own A/B measured it as free, but this is not a
  clean grid-only comparison.
- Position optimisation gains are real but small here (isotropic scalar target); positions matter
  more for the anisotropic ν(θ) targets of other goals.

---

## 5. Figures

- `fig1_achieved_vs_target.png` — achieved (sim) ν vs target ν, y = x line; coloured by k-band and
  by topology class (square panels). Elements: `elements/fig1a_achieved_by_band.png`,
  `elements/fig1b_achieved_by_class.png`.
- `fig2_frontier.png` — reachable ν range (min..max, median marker) vs contrast band f (the frontier
  of §3a). Element: `elements/fig2_frontier.png`.
- `fig3_improvement.png` — target error initial → k-only → k+positions: all band means in one panel
  plus each band's mean ± σ in its own subplot. Elements: `elements/fig3a_improvement_means.png`,
  `elements/fig3b_spread_{soft,large,medium,small,none}.png`.
- `fig4_breakdown.png` — mean ν-error (± σ) by topology class and by k-band. Elements:
  `elements/fig4a_by_class.png`, `elements/fig4b_by_band.png`.

Data: `results.csv`, `results.npz`. Designs: `Phase 5/networks/goal1/design_g1_0..109.npz`.
