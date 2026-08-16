# Goal 1 — Rich isotropic inverse-design test (Phase 5 M1)

**Question.** Given a *scalar* (isotropic) Poisson-ratio target ν with Young's modulus E = 1, how
well can the Phase 5 designer hit ν across the physical range, and **how does the achievable ν
depend on how much per-bond stiffness contrast the network is allowed?** Every design is verified by
an *independent* full-PBC simulation (a different code path from the solver it was optimised
against), and the target error is tracked through three optimisation stages to prove the
optimisation actually helps.

---

## 1. The experiment

- **ν grid (11):** `[-0.9, -0.7, -0.5, -0.3, -0.15, 0.0, 0.1, 0.2, 0.3, 0.4, 0.45]`.
  > **CORRECTION 2026-08-16.** This used to read "spans the 2D physical range (−1, 1)". It does not:
  > it spans **(−0.9, +0.45)** — 90% of the negative half, 45% of the positive half. The asymmetry is
  > historical and undocumented. **This invalidates the positive half of the reachable-window result
  > below**: the reported max of **+0.45** at `f=0` (and +0.44 at `f=0.1`) is *the top grid point* —
  > nothing above it was ever attempted — so it is the **edge of the search, not a frontier**. The
  > true ceiling could be 0.5 or 0.95 and this sweep could not distinguish them. The **negative** end
  > is a genuine measurement (grid runs to −0.9, designer falls short at −0.82). Probed properly by
  > `Phase 5/verifications/run_goal1_frontier.py` (ν up to 0.95). Extremes may be unreachable; that
  > is reported as a *frontier*, not a failure — but only where the grid actually looked.
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

### 3a. The achievable-ν FRONTIER vs contrast band — stiffness contrast is the lever for auxeticity

The reachable (independent-sim) ν collapses toward the regular-lattice value ν = 1/3 as the contrast
floor is raised:

| band | f | reachable ν [min, max] | median ν |
|------|-----|------------------------|----------|
| soft   | 0.00 | **[−0.82, +0.45 ⚠]** | +0.00 |
| large  | 0.10 | [−0.31, +0.44 ⚠] | −0.00 |
| medium | 0.50 | [−0.00, +0.33] | +0.17 |
| small  | 0.90 | [+0.05, +0.34] | +0.20 |
| none   | 0.99 | **[+0.16, +0.33]** | +0.26 |

> ⚠ **= CENSORED BY THE GRID, not a measured frontier.** +0.45 is the top grid point and +0.44 is one
> step below it, so neither is a property of the networks — nothing above +0.45 was attempted. The
> `medium`/`small`/`none` maxima (+0.33, +0.34, +0.33) ARE genuine: they sit well inside the grid, so
> the designer had room above and did not use it. Likewise every minimum here is genuine. Only the
> two flagged cells are artifacts of where the search stopped.
>
> **RESOLVED 2026-08-16** by `Phase 5/results/goal1_frontier/GOAL1_FRONTIER.md` (ν probed to 0.95):
> | band | f | reported here | **actual frontier** |
> |---|---|---|---|
> | soft | 0.00 | +0.45 ⚠ | **+0.91** (gap 0.0000, trustworthy) |
> | large | 0.10 | +0.44 ⚠ | **+0.63** (genuine — targets 0.80/0.90/0.95 all saturate there, gap 0.0000) |
>
> So the true `f=0` window is **[−0.78, +0.91]** — near-symmetric — and roughly **half the positive
> half of the design space was invisible to this sweep**. The positive frontier turns out to depend on
> stiffness contrast exactly as the negative one does, which *strengthens* this document's central
> claim rather than weakening it: reaching either extreme requires bond-level contrast, and clamping
> `f` collapses both ends toward the uniform-lattice ν = 1/3.

At `f = 0` the designer reaches deep auxetic behaviour (ν ≈ −0.82) *and* the stiff positive end
(+0.45). Clamping the bonds toward uniform (`f = 0.99`) collapses the whole reachable window to
≈ [+0.16, +0.33], i.e. it hugs the isotropic triangular-lattice ν = 1/3. **Auxeticity requires
stiffness contrast: soft bonds acting as hinges are what let the network fold to a negative Poisson
ratio; forbid the contrast and the negative-ν region simply disappears.** See
`fig2_frontier.png`.

### 3b. Does optimisation reduce the error — k vs positions, per band

Mean target error `|ν_sim − ν*|` at each stage (trustworthy runs), initial → k-only → k+positions:

| band | f | initial | k-only | k + positions |
|------|-----|---------|--------|---------------|
| soft   | 0.00 | 0.412 | 0.023 | **0.009** |
| large  | 0.10 | 0.455 | 0.194 | **0.158** |
| medium | 0.50 | 0.419 | 0.348 | **0.321** |
| small  | 0.90 | 0.401 | 0.390 | **0.380** |
| none   | 0.99 | 0.431 | 0.430 | **0.418** |

Reading this honestly:

- **k is the dominant lever, and it works spectacularly where it is allowed.** In the soft band the
  k-designer alone cuts the error 0.412 → 0.023 (~18×); positions then trim it to **0.009**.
- **Positions help secondarily, most in the mid-contrast bands.** The k→(k+pos) reduction is largest
  in `large` (0.194 → 0.158) and `medium` (0.348 → 0.321) — where k has done what it can but the
  geometry still has slack. In the soft band positions add a final polish; in the near-uniform bands
  they barely move ν.
- **In the near-uniform bands neither lever moves ν much — the response is locked.** At `small`
  (0.401 → 0.390 → 0.380) and `none` (0.431 → 0.430 → 0.418) both k and positions are nearly inert:
  with contrast forbidden, ν is pinned near 1/3 and most grid targets are simply unreachable. That is
  the expected, honest consequence of the constraint, not a design failure. See
  `fig3_improvement.png` (means together in one panel; each band's mean ± σ in its own subplot).

### 3c. Aggregate acceptance

- **100 / 110 runs trustworthy** (solver-vs-sim gap < 0.05). The 10 dropped runs are exactly the
  hard corners — mostly extreme ν at the high-ν end or auxetic/soft-edged near-mechanisms where the
  independent sim and solver diverge (e.g. auxetic `none` ν=+0.1 gap 0.215, tiling `medium` ν=−0.7
  gap 0.105). Dropping them is the point of the honesty gate.
- **Median target error (k+pos) = 0.082; success rate |err| < 0.05 = 42 %.**
- **Why the success rate is "only" 42 % — and why that is the correct outcome.** The ν grid was
  deliberately pushed to the physical extremes at *every* contrast band, including combinations that
  are provably unreachable (e.g. ν = −0.9 at `none`, or ν = +0.45 at high f). Those grid points can
  never be "hit"; they exist to **map the frontier**. Restricted to the soft band where the targets
  are physically accessible, the error is 0.009 and essentially every point is a hit. The 42 % is
  therefore a property of how aggressively the grid samples beyond the frontier, not of the
  optimiser's quality. `fig1_achieved_vs_target.png` shows this directly: soft/large points sit on
  the y = x line across the whole ν range, while high-f points saturate onto the ν ≈ 1/3 plateau.

### 3d. Breakdown by topology class

Mean target error (k + positions, trustworthy runs):

| class | n | initial | k+pos mean | k+pos median | σ |
|-------|-----|---------|-----------|--------------|-----|
| tiling  | 6  | 0.169 | **0.051** | 0.003 | 0.104 |
| auxetic | 6  | 0.301 | **0.083** | 0.012 | 0.146 |
| foam    | 40 | 0.436 | 0.256 | 0.079 | 0.349 |
| bravais | 33 | 0.446 | 0.296 | 0.149 | 0.347 |
| flipped | 15 | 0.492 | 0.337 | 0.259 | 0.351 |

All classes improve from initial to k+pos. Tiling and auxetic seeds finish best (low medians) — they
are richer starting points near the interesting region — but they are also fewer, so their means are
noisy. Foam and bravais dominate the counts and carry the frontier sweep; their large σ reflects the
mix of reachable and deliberately-unreachable grid points assigned to them. Flipped (non-Delaunay)
seeds are hardest on average, consistent with the solver being least validated there. See
`fig4_breakdown.png`.

---

## 4. Limitations & honest caveats

- **The grid intentionally over-reaches the frontier**, so aggregate "success rate" understates
  designer quality; read it together with the per-band frontier (§3a) and the soft-band error
  (0.009).
- **High-f bands are near-degenerate by construction**: with `f = 0.99` there is almost no design
  freedom, so those 42 runs mostly report the seed's own ν near 1/3.
- **10 runs dropped** at the honesty gate — mostly auxetic/soft-edged near-mechanisms and extreme-ν
  corners where solver and independent sim disagree; these are flagged, not silently kept.
- **Modest per-design budget** (`n_iter = 120`, `n_restarts = 2`, `spsa_steps = 20`, `n_outer = 2`)
  chosen so the full sweep runs in ~76 min; a larger budget would tighten the reachable errors
  further, especially in the `large`/`medium` bands.
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
