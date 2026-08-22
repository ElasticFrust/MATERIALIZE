# Goal 1 — Rich isotropic inverse-design test (Phase 5 M1)

**Question.** Given a *scalar* (isotropic) Poisson-ratio target ν with Young's modulus E = 1, how
well can the Phase 5 designer hit ν across the physical range, and **how does the achievable ν
depend on how much per-bond stiffness contrast the network is allowed?** Every design is verified by
an *independent* full-PBC simulation (a different code path from the solver it was optimised
against), and the target error is tracked through three optimisation stages to prove the
optimisation actually helps.

> **RE-RUN 2026-08-22, twice.** (i) a SYMMETRIC ν grid — 13 points over [−0.95, +0.95] × 5 contrast
> bands × 2 topologies = 130 runs; then (ii) the same grid with the **position budget repaired**
> (`spsa_a` 0.02 → 0.25; the old value gave positions 13× too little travel — §3a's box).
> **All numbers and figures below are from run (ii): 130 runs, 97 min, 67 trustworthy.**
>
> Two headlines. **With contrast unrestricted the designer covers [−0.823, +0.899]**, most of the
> physical range, and `run_goal1_frontier.py`'s separate high-ν probe is **subsumed**. And **every
> contrast band reaches negative ν**, including f = 0.99 — so the earlier reading that the design
> space collapses onto +1/3 without contrast was a search artefact, not physics.
>
> Three variables have changed since the 2026-08-18 run (93/110 trustworthy, soft band
> [−0.78, +0.45]): the grid, `positions.quality_floor` (0.0 → 1e-3), and the position budget. This is
> **not** a clean A/B against those numbers on any single axis.
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

### 3a. Reachable ν vs contrast band — contrast raises the CEILING, positions supply the FLOOR

Symmetric 13-point ν grid over [−0.95, +0.95] at every band, **and positions optimised at a real
budget** (`spsa_a = 0.25`; see the box below for why the previous run's was not). Reach is quoted over
**ALL runs**, with the gap-passing sub-range beside it — they answer different questions, and quoting
only the second is what hid this result the first time:

| band | f | reachable ν, ALL runs | sub-range where solver = sim | n trust |
|------|-----|------------------------|------------------------------|---|
| soft   | 0.00 | **[−0.823, +0.899]** | [−0.823, +0.899] | 22 |
| large  | 0.10 | **[−0.573, +0.650]** | [−0.300, +0.650] | 15 |
| medium | 0.50 | **[−0.377, +0.378]** | [+0.150, +0.378] | 12 |
| small  | 0.90 | **[−0.287, +0.343]** | [+0.145, +0.343] | 10 |
| none   | 0.99 | **[−0.336, +0.338]** | [+0.257, +0.338] | 8 |

**Every band reaches negative ν — including f = 0.99, where k is effectively frozen.** The window at
f = 0.99 is [−0.336, +0.338]: **near-symmetric about zero, spanning roughly ±1/3**, not a point near
+1/3. Consistent with `g1_2`, which reaches −0.436 at k ≡ 1 *exactly*.

So the lever is **asymmetric**, and the earlier "contrast is the lever at both ends" was wrong:

- **High ν REQUIRES stiffness contrast.** +0.34 at f = 0.99 against +0.90 at f = 0. Geometry cannot
  push ν above the uniform-lattice value; only contrast can.
- **Auxetic ν does NOT require contrast.** Positions alone deliver −0.29 … −0.34 at every band, and
  −0.436 in `g1_2`. Contrast deepens it (−0.82 at f = 0) but is not necessary for it.

> ### ⚠ WHY THE PREVIOUS TABLE SAID OTHERWISE — a search artefact, now fixed
> Until 2026-08-22 `design_iso` called `spsa_positions` **without passing `a`/`c`**, so positions ran
> at the library defaults `a = 0.02, c = 0.01`: **0.202 lattice spacings** of possible travel per
> coordinate over 20 steps × 2 outer, against **2.68** for `g1_2` at a = 0.25 — 13× less, and an
> eighth of even the a = 0.15 arm measured *failing* to reach auxetic ν. **goal1's position
> optimisation was effectively off**, so the high-f rows reported little more than the seed's own ν
> and the table appeared to show the design space collapsing onto +1/3.
>
> With `a = 0.25` (travel 2.53, matching g1_2, at the *same* step count and runtime) the high-f
> floors move from +0.04 / +0.13 / +0.18 to **−0.377 / −0.287 / −0.336**. Target accuracy improves
> too: median |err| **0.198 → 0.0063**, success rate 33 % → 55 %.
>
> **Cost:** trustworthy runs fall **114 → 67**. Larger position moves produce more solver-sim
> disagreement (median |Δν| 0.0022, max 0.708) — bigger distortions, more near-slivers. That is a
> statement about where the two codes stop agreeing, not about where the material stops.

### 3b. Does optimisation reduce the error — k vs positions, per band

Mean target error `|ν_sim − ν*|` at each stage (trustworthy runs), initial → k-only → k+positions:

| band | f | initial | k-only | k + positions |
|------|-----|---------|--------|---------------|
| soft   | 0.00 | 0.486 | 0.021 | **0.015** |
| large  | 0.10 | 0.344 | 0.092 | **0.068** |
| medium | 0.50 | 0.290 | 0.230 | **0.220** |
| small  | 0.90 | 0.266 | 0.257 | **0.243** |
| none   | 0.99 | 0.359 | 0.358 | **0.353** |

k does the heavy lifting where it is free (soft: 0.486 → 0.021), and positions then trim 20–30 % off
what k leaves. In the high-f bands k is inert by construction and the residual error is dominated by
grid points outside the reachable window — but note §3a: those bands *do* now reach ν < 0, so the
residual is the distance to unreachable *targets*, not evidence that the band cannot move.

### 3c. Aggregate acceptance

- **67 / 130 runs pass the agreement gate** (`solver_sim_gap < 0.05`), down from 114 before the
  position budget was fixed — the price of letting the optimiser actually move.
- **Median target error (k+pos) = 0.0063 over trustworthy runs; success rate |err| < 0.05 = 55 %**
  (was 0.198 and 33 %).
- Solver-vs-sim over all 130: **median |Δν| = 0.0022, max 0.708**.


### 3c-bis. Where solver and sim disagree — THREE independent reproductions of the same corner

Per-band solver-vs-sim |Δν| (all 130 runs):

| band | f | median \|Δν\| | max \|Δν\| | where the max sits |
|---|---|---|---|---|
| soft   | 0.00 | 0.0000 | **0.0535** | ν = −0.772, foam |
| large  | 0.10 | 0.0003 | 0.2518 | ν = −0.403, tiling |
| **medium** | 0.50 | 0.0241 | **0.7082** | **ν = +0.306, tiling** |
| small  | 0.90 | 0.0205 | 0.1396 | ν = −0.134, flipped |
| none   | 0.99 | 0.0143 | 0.1486 | ν = +0.048, foam |

**The same corner, three times, across independent runs:**

| run | grid | position budget | worst \|Δν\| | where |
|---|---|---|---|---|
| 2026-08-18 | asymmetric | travel 0.202 | 0.311 | ν = +0.286, tiling, **medium** |
| 2026-08-22 a | symmetric | travel 0.202 | 0.323 | ν = +0.296, tiling, **medium** |
| 2026-08-22 b | symmetric | travel 2.53 | **0.708** | ν = +0.306, tiling, **medium** |

Different grids, independently drawn topologies, and a 13× change in position budget — and the worst
disagreement lands on `tiling` + f = 0.5 + ν ≈ +0.30 every time. **This is a reproducible
phenomenon, and it is the sharpest open lead in the project.**

Two things it is *not*: it is not at extreme ν (the `soft` band spans [−0.823, +0.899] with a worst
case of 0.0535, **13× better while reaching 3× further**), and it is not explained by conditioning —
`conditioning_probe.py` found the 08-18 instance healthy on every axis (`rcond_min` 5e-03, shape
quality 0.475, `k_min/mean` 0.65), so neither route in `CLAUDE.md` §3 applies. A `tiling` cell at
f = 0.5 targeting ν ≈ +0.30 is a ready-made reproducer for whoever picks this up.

*(The medians in the three high-f bands are no longer ~0 — 0.024 / 0.021 / 0.014 against 0.0000
before. Larger position moves raise typical disagreement there, not just the worst case.)*


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
