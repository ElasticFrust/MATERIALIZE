# Clean validation re-run — 2026-08-17/18

Every Phase 2/3/5 validation in scope, re-run under the post-audit code, with outputs, logs and
provenance collected in one tree. Driver: [`run_validation.py`](run_validation.py).

**27 scripts · 11.1 h · 701 output files · 2 non-zero exits** (both genuine, both explained below).

Scope, agreed with the user: the four gate suites plus the headline validations `MATERIALIZE.md` §9
names. The one-off audit diagnostics and superseded probes in `Phase 3/verifications` were **not**
run — 112 scripts exist, 42 of which optimise; running all of them would take days and would fill a
*clean* tree with output from scripts already superseded, making retirement harder rather than
easier.

## Why this re-run existed

The August audit changed the forward map (**A-0** shear contraction, **A-10** ν convention) and fixed
two rendering defects (**B-4b** colour norm, **B-4c** polar zero crossings). Everything produced
before that is suspect — audit **A-19**. The trap is that stale artifacts *look* fine:
`fig1_recreate_pointy.png` "regenerated" in 41 s and appeared to be a completed re-run while
re-rendering a design frozen 2026-07-12.

So the runner **stows the design caches** before starting (moved to `attic/`, never deleted — the
July files are the evidence for A-19).

## Results

| | |
|---|---|
| gates | `test_forward_solver` 8/8 · `test_inverse_design` 16/16 · `test_designer_surface` 5/5 · `test_hex_closed_form` 3/3 · `sanity` PASS |
| `auxetic_sweep` | solver=sim to 3 decimals across the grid; ν=−0.60 hit exactly at N=12 |
| `anisotropy` | solver and sim identical on every row; `four_fold` err 0.047, flat-ν series 0.500 at flatness 0.000 |
| `fig1` | **redesigned**: ν [−3.63,+7.18] = target, sim-confirmed (was [−18.1,+19.1] from the July cache) |
| `fig1c` / `fig1d` | the pair now makes its point — see below |
| `run_goal1` | 110 runs, 93 trustworthy, median ν-err 0.098; reachable ν collapses [−0.78,+0.45] → [+0.20,+0.33] as the stiffness band tightens |
| `run_g1_2` | 110 runs, **45** trustworthy; every topology reaches only **positive** ν ≈ [+0.10,+0.34] |
| `run_goal2` | **13/13 trustworthy**, all gaps 0.000; positions helped in 10/13 |
| `run_goal2_attempts` | redesigned: attempts cluster away from target in **13/13** cases |

### The two non-zero exits are both real

- **`verify_lattice` (exit 1)** — the regular lattice driven to ν=−0.2: solver **−0.112** vs sim
  **+0.137**, gap 0.248, flagged `UNTRUSTWORTHY`. This reproduces the documented 2026-08-16
  measurement (−0.110 / +0.136) to ~0.002, so the `A(s)` rank-loss regime survives the instrument
  repairs. Everything else in that script passed, including all five geometries agreeing to three
  decimals. **The script is doing its job**; "all green" is not the success criterion for this sweep.
- **`verify_positions` (exit 1)** — **an open failure, not yet diagnosed.** (a) position polish
  *raised* the design loss 1.8090e-03 → 1.8217e-03, and (b) `solver_sim_gap` 0.0646 exceeds the 0.05
  tolerance. `CLAUDE.md` §3 says the polished design replaces the top only if it **both** lowers the
  loss **and** passes the honesty check — so either that gate is not applied on this path, or the
  test probes the raw polish before it. Until then, treat position-polish claims as unverified.

### fig1c / fig1d — the pair, finally complete

| | k floor | achieved ν | target |
|---|---|---|---|
| fig1c (rotated **regular**) | 1e-2 on 1167 bonds | [−0.37, +0.83] | [−3.63, +7.18] |
| fig1d (rotated **disordered**) | none, 0 bonds floored | **[−3.63, +7.18]** | [−3.63, +7.18] |

The response orientation is set by the **design**, not the substrate's orientation: a 90°-rotated
disordered network reproduces the unrotated lab-frame ν(θ) exactly, while the 90°-rotated regular
lattice provably cannot. *fig1d had never been run* — the script takes a substrate argument and the
first pass invoked it with none, so it silently produced only the negative half of the pair.

### g1_2 — read this one carefully

Not one topology reaches negative ν trustworthily; every auxetic target comes back with a solver-sim
gap of 0.475–1.004. The optimiser does emit sub-zero numbers (ν=−0.294 for a −0.95 target) but the
independent sim disagrees. **Within g1_2's setup, ν<0 is reachable only by entering the
near-mechanism regime where the read-back cannot be trusted.** Two caveats: `run_goal2` *did* land a
negative-ν full-tensor target trustworthily, so "we cannot reach ν<0" would be the wrong general
conclusion; and every reachable range bottoms out at ≈+0.10, exactly the smallest positive target on
the grid, so the lower edge is set by **what was sampled**, not by what is achievable.

The mesh repair worked partially: `honeycomb` and `rotating_squares` now give six trustworthy runs
each. **`square_octagon` gives none despite its mesh being repaired to gap 0.0000000** — repair was
necessary, not sufficient. `reentrant_honeycomb` (unrepairable, `mesh_ok=False`) also gives none.

## Provenance — A-19 progress

| | before | after |
|---|---|---|
| `.npz` in repo | 1171 | 1186 |
| stamped | 651 | **783** |
| unstamped (suspect) | 520 | **403** |

`Phase 5/networks/goal2_attempts` went from 115 unstamped to **117 stamped**; `networks/g1_2` gained
110 fresh. The remaining 403 are the retirement target — see [`provenance_scan.csv`](provenance_scan.csv),
regenerate with `scan_provenance.py`.

## What this sweep exposed about the method

Four defects, all found by *checking outputs*, never by an exit code:

1. **B-6** — `dhex_family.py` ran a design sweep **on import**; two gates silently rewrote 8 files.
2. **Runner, output collision** — folders keyed on basename, and four scripts are called
   `design_and_verify.py`. Cost: one log.
3. **Runner, missing arguments** — fig1d never ran (above).
4. **`verify_positions` exit 0 on failure** — a failing validation recorded as a success.

Plus a **second A-19 class the cache-stowing cannot catch**: scripts that purely *load and plot*.
Stowing their input only makes them crash; their **producer** must run first. `fig5_bullseye_strain`
rendered a 9-July network in 7.5 s and exited 0. **A re-run set must be closed under
producer→consumer**, which this one was not.

fig1 (41 s), fig5 (7.5 s) and goal2_attempts (42 s) were each caught because they finished
suspiciously fast. Three lucky catches is not a method, so the runner now carries `verdict()` (fails
on `FAILED`/`UNTRUSTWORTHY`/traceback in stdout, not just the exit code) and `stale_inputs()`.

## Known-incomplete

- `auxetic_sweep`'s log was lost to defect 2 above. Its 78 outputs and its CSV survive, so the
  numbers are intact; only the console transcript is gone.
- Logs written before commit `0c235f4` carry cp1252 mojibake (`Î½` for `ν`). Numbers are ASCII and
  unaffected.
- `fig5_bullseye_strain` still renders a July network; its producer `strain_stress/large16k_rings.py`
  is outside the agreed scope.
