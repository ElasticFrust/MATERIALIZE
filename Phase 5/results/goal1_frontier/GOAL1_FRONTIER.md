# GOAL 1b — the positive-ν frontier

**What.** `goal1` sweeps ν over `[-0.9 … +0.45]` and its results doc reported the reachable window at
`f=0` as **[−0.82, +0.45]**. The upper number is **the top grid point** — nothing above it was ever
attempted — so it measured where the search stopped, not what the networks can do. This probe extends
only the positive side and asks what the real ceiling is.

**Method.** `Phase 5/verifications/run_goal1_frontier.py` — the *same* driver, budget and verification
as `goal1`, with only the grid monkey-patched (so `goal1` itself stays byte-comparable with its
historical run). ν ∈ {0.5, 0.6, 0.7, 0.8, 0.9, 0.95}, bands `f` ∈ {0.0, 0.1} (the two that reached the
old ceiling), 2 reps → **24 runs, 15.1 min**.

**Provenance.** Commit `9b7c7eb`, clean tree. Seeds `topo=0`, `shuffle=12345`. Log:
`documentation/campaign/stage2b_frontier/run_frontier.log`.

**Reading pre-specified before the run** (so the conclusion was not chosen after seeing the data):
reach ~0.9 trustworthily ⇒ +0.45 was a pure grid artifact; saturate at ν\* with *small* gaps ⇒ ν\* is a
real frontier; saturate with *large* gaps ⇒ the limit is the solver's `A(s)` validity condition
(`CLAUDE.md` §3), not the physics.

---

## 1. Result

**The +0.45 ceiling was a grid artifact. At `f=0` the designer reaches ν ≈ +0.91.**

| ν target | band | achieved ν | err | solver-sim gap | trustworthy |
|---|---|---|---|---|---|
| 0.80 | soft | **+0.8007** | 0.0007 | **0.0000** | ✓ |
| 0.90 | soft | **+0.8915** | 0.0085 | **0.0000** | ✓ |
| 0.95 | soft | **+0.9024** | 0.0476 | **0.0000** | ✓ |
| 0.95 | soft | **+0.9061** | 0.0439 | 0.0416 | ✓ |

Max over trustworthy runs: **ν = +0.906**. Several of these have a solver-sim gap of *exactly zero* to
four decimals, so this is not a marginal or ill-conditioned reading — the independent simulation
confirms it.

**Corrected reachable window at `f = 0`: [−0.78, +0.91]** — near-symmetric, against the
[−0.82, +0.45] previously reported. Roughly **half the positive half of the design space was
invisible** to the old sweep.

## 2. `f = 0.1` has a REAL frontier at ≈ +0.63

Unlike the `f=0` case, this one is measured, not censored:

| ν target | achieved (band `large`) | gap |
|---|---|---|
| 0.80 | +0.6197, +0.6091 | 0.0000 |
| 0.90 | +0.6124, +0.5936 | 0.0000 |
| 0.95 | +0.6280, +0.6244 | 0.0000, 0.0126 |

Targets of 0.80, 0.90 and 0.95 all land at **+0.59 … +0.63**, with the errors growing (0.18 → 0.29 →
0.33) precisely because the achieved value stops moving. **Gaps are 0.0000** — the designer stops
there and the independent sim agrees it stopped there. That is a genuine physical frontier, not an
instrument limit.

**So the positive frontier depends on stiffness contrast, exactly as the negative one does:**

| band | f | positive frontier |
|---|---|---|
| soft | 0.00 | **≈ +0.91** |
| large | 0.10 | **≈ +0.63** |
| (medium/small/none, from `goal1`) | 0.5–0.99 | +0.39, +0.34, +0.33 — genuine, well inside the grid |

This mirrors `goal1`'s central claim on the auxetic side. Reaching *either* extreme requires bond-level
stiffness contrast; clamp `f` toward uniform and both ends collapse toward the uniform triangular
lattice value ν = 1/3.

**Why this is physically sensible.** In 2D isotropic elasticity ν = (K − G)/(K + G), so ν → +1 needs
the shear modulus G → 0 and ν → −1 needs the area modulus K → 0. Both extremes require a *vanishing
modulus*, achievable only by making some bonds much softer than others. A stiffness floor `f` forbids
exactly that, so it truncates both frontiers — which is what both experiments now show.

## 3. Why this end matters especially

High ν is the **shear-soft** channel — precisely the channel the **A-0** defect corrupted (`C_xyxy`
over-stiff by 29–90% wherever `W ≠ 0`). It is therefore the region most likely to have been
mis-measured before the fix, and the one where a post-fix measurement carries the most information.
That it now comes back with *zero* solver-sim gap at ν ≈ 0.9 is a strong end-to-end check on the shear
channel repair.

## 4. Limitations

- **24 runs.** One topology per run, 2 reps per (ν, band). Enough to demolish a censored ceiling; not
  enough to pin the frontier to better than ~±0.03.
- **7 of 24 untrustworthy** (gap ≥ 0.05), concentrated at ν ≥ 0.70 in the `soft` band — e.g. two runs
  with gaps 0.90 and 0.93. Consistent with the `A(s)` validity limit: reaching extreme ν at `f=0`
  drives bonds toward zero, which is what breaks per-triangle invertibility (`CLAUDE.md` §3). The
  frontier quoted here uses **only trustworthy runs**.
- Only `f ∈ {0.0, 0.1}` probed above +0.45. The higher-`f` frontiers are taken from `goal1`, where
  they are genuine (well inside that grid).
- Isotropic scalar ν only, E = 1. Nothing here speaks to directional targets.
- The optimiser is nondeterministic (audit B-1 stage 1), so single-run achieved values carry that
  spread; the frontier claim rests on several runs agreeing, not on one.

## 5. Consequence for `goal1`

`Phase 5/results/goal1/GOAL1.md`'s reachable-window table has been corrected: the `f=0` and `f=0.1`
maxima are flagged as censored, with a pointer here. **The paper-level claim of `goal1` — auxeticity
requires stiffness contrast — is unaffected**; it rests on the negative end and on the `f`-dependence,
neither of which was censored. What changes is that the *positive* half of the story was never
measured, and now is.
