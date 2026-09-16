# A1/d — what deformation + stiffness contrast actually reach, and the "hole" that wasn't

**Producer:** `Phase 5/verifications/m2_coverage_sweep.py` (+ the focused probe in §3)
**Data:** `coverage_sweep_pilot.json`, `coverage_sweep_iso.json` (this directory)
**Run:** 2026-09-16, float64, `OMP_NUM_THREADS=1`

---

## The claim under test

`m2_response_audit.py` found five empty (ν × anisotropy) cells in `dataset_v2_s0`, among them
**isotropic auxetic** — ν ∈ [−0.5, −0.2] with anisotropy < 1.2. I reported that as a coverage hole.

**The user's objection:** η plus VD / α contrast should reach a wide range of *isotropic* negative
Poisson ratios, and deforming other crystal topologies with and without stiffness contrast should
fill most of the rest. If so, the hole is a **sampling choice of the existing builder**, not a reach
limit — and the fix is to sample differently, not to build a new generator.

**Verdict: the objection is correct.** The cell is reachable, reliably, with machinery that already
exists.

## 1. VD contrast alone is a powerful ν lever

First sweep (5 Bravais types × both diagonals × `frac` × VD, 99 samples):

| VD `a` | ν median | ν range |
|---|---|---|
| −5 | +0.629 | [+0.333, +0.993] |
| 0 | +0.223 | [−0.232, +0.333] |
| +5 | −0.083 | [−1.193, +0.333] |

`k = 1 + tanh(a(|R| − 1))` spans **ν from +0.993 to −1.193** — wider than the designed-`k` reach
envelope in `results/reach_summary/`. But anisotropy stayed at median 5.5–12.8 and the `iso<1.2`
column was empty except at ν ≈ +1/3.

## 2. That was MY sampling, not the physics

Two defects in the first sweep, both mine:

- **Four of five (φ, ψ) choices are intrinsically anisotropic lattices**, and length-keyed VD
  *amplifies* that: in a non-equilateral cell, bond length correlates with orientation, so VD
  becomes an orientation field.
- **`bravais_lattice(1, 1)` has two diagonals and only one is the regular triangular lattice.**
  `a2−a1` has bond length 1; `a1+a2` has length **√3**. Sweeping both put a mixed-length lattice
  under the label "triangular", which is why anisotropy read a median of **17** at `frac = 0` where
  an undeformed triangular lattice must be exactly isotropic.

## 3. The focused probe — and it self-averages, as it should

Triangular lattice, diagonal `a2−a1` only, white disorder via `displace_safe`, 8 seeds:

| reps | n_tri | frac | VD | ν median [min, max] | anisotropy median (p90) |
|---|---|---|---|---|---|
| 6 | 96 | 0.0 | +5 | **+0.333** [+0.333, +0.333] | **1.00** (1.00) |
| 6 | 96 | 0.9 | +2 | −0.251 [−0.304, −0.184] | 1.24 (1.35) |
| 6 | 96 | 0.9 | +5 | −0.482 [−0.537, −0.358] | 1.44 (1.94) |
| 10 | 240 | 0.9 | +2 | −0.203 [−0.263, −0.145] | 1.16 (1.28) |
| 10 | 240 | 0.9 | +5 | −0.407 [−0.508, −0.309] | 1.39 (1.57) |
| **16** | **576** | **0.9** | **+2** | **−0.197** [−0.230, −0.163] | **1.08 (1.10)** |
| **16** | **576** | **0.9** | **+5** | **−0.421** [−0.474, −0.333] | **1.19 (1.23)** |

**Anisotropy falls monotonically toward 1 as the cell grows** — 1.44 → 1.39 → 1.19 at VD = +5, and
1.24 → 1.16 → 1.08 at VD = +2. That is disorder self-averaging, exactly as it should.

At **reps = 16** the isotropic-auxetic cell is populated *routinely*, not by luck: ν = −0.42 at
anisotropy 1.19, and ν = −0.20 at anisotropy 1.08 with p90 only 1.10.

**Sanity check passes exactly:** `frac = 0` with VD = +5 gives ν = **+0.333** and anisotropy **1.00**
on every seed — because every bond has length 1, so `tanh(a·0) = 0` and VD produces uniform `k`. The
lattice is untouched, as it must be.

## 4. The mechanism, stated plainly

VD is **length-keyed**. On an undeformed lattice every bond has the same length, so VD does nothing.
Disorder creates the length variation; VD converts it into stiffness contrast; and because the
disorder is statistically isotropic, so is the contrast — at a cell large enough to self-average.
η and VD are therefore not two independent knobs but a **pair**: neither reaches isotropic auxetic
alone, and together they reach it easily.

## 5. Consequences

- **The audit's "hole" is withdrawn as a coverage limit.** It is a sampling gap: `dataset_v2_s0`
  never crossed η-disorder with VD contrast at a large enough cell. `build_dataset` should sample
  that cross, and A1 needs no new generator to fill the cell.
- **Two of the five empty cells look like genuine sampling gaps of the same kind** (`<−1` and `>1` at
  low anisotropy); they were not targeted here and remain open.
- **Cell size is a sampling parameter with physics in it, not a cost knob.** Anisotropy at fixed
  disorder is a finite-size effect, so a sweep that varies cell size without saying so conflates
  "anisotropic material" with "small sample".
- **My framing was wrong in a second way, recorded in `GRADIENT_FIDELITY.md`'s discussion:** a
  bulk-response hole matters for the INVERSE model, much less for the surrogate, which predicts
  per-triangle `C(s)` and averages. What the surrogate needs coverage of is local environments.

## Limitations

- One disorder structure (`white`) in the focused probe; `correlated` was swept only in §1.
- `displace_safe` guarantees no inversion, and 0 of 350 samples were rejected — but the amplitude it
  reaches is mesh-dependent (`geom_eta_equiv` ≈ 0.42 at `frac = 0.9` here), so "frac" and the
  classical η are related, not identical.
- The ν values are the **solver's**; none of this run was cross-checked against the independent sim.
  That is the standing tier-(A) check and should be run before these numbers are used as labels.
- 8 seeds per cell: enough to see a monotone trend, not enough for a tight distribution.
