# A1/d — auxetic networks from arbitrary base topologies, via η / VD / η+α

**Producer:** `Phase 5/verifications/m2_auxetic_generator.py`
**Data:** `auxetic_generation_{pilot,breadth,large}.json` (this directory)
**Datasets:** `Phase 5/m2/data/auxgen_breadth.npz` (24.9 MB), `auxgen_large.npz` (25.3 MB) —
gitignored by repo policy (`.gitignore:17`), regenerable from the config recorded in each JSON
**Run:** 2026-09-16, float64, `OMP_NUM_THREADS=1`, seed 0, commits `650fa91` → `03c464a`

---

## What and why

The A1.a response audit found the auxetic corner of `dataset_v2_s0` thinly and *unevenly* sampled:
ν < 0 is 19.3 % overall but concentrated in two generator families, and the **isotropic-auxetic cell
was empty**. The prior reading was that this is a reach limit. It is not — A1/d established it is a
**sampling gap**, and this run is the crossing of the axes that fills it.

The user's instruction set the design: *"generate more networks with auxetic behaviour based on other
networks through the η, VD, and η+α mechanisms, where the maximal amplitudes may be position dependent
but use a simple parameter indicating a portion of it, uniform everywhere"*, sweeping **ranges** of η
and α rather than fixed operating points, with α capped at |α| ≤ 10.

## Method

### The three mechanisms, in the user's definitions

| mechanism | geometry | stiffness |
|---|---|---|
| **η** | really distorted | uniform `k` |
| **VD** | **NOT distorted** — virtual displacements only | an η variation decides the rigidities, `k` from `atanh(a(l − l₀))` |
| **η+α** | really distorted | `k = 1 + tanh(α(l − l₀))` off the **distorted** lengths |

VD is the one that was previously got wrong in this repo, and the correction is recorded in
`Phase 5/results/coverage_sweep/COVERAGE_SWEEP.md`: what had been called VD moved the vertices. VD
works on irregular lattices too, which is why it is swept on every base here and not only on crystals.

### Amplitude: position-dependent maximum, one uniform knob

`fields.displace_safe(scale='local')` gives every vertex **its own maximum** — the exact distance to
the first inversion in its own neighbourhood, by local fixed point, nothing global — and `frac` is the
single scalar saying what portion of that maximum to take, uniform everywhere. So the amplitude adapts
to the mesh while the caller turns one dial.

Worth **2.3× median amplitude** over a single global scale on a disordered mesh, and up to **12×** on
individual vertices (`m2_local_scale_probe.py`). On a regular mesh it is worth nothing, which is the
correct behaviour — there is nothing to localise.

The two modes are **mutually exclusive guarantees**, documented in `fields.py` and gated in
`test_displace_safe.py` [2g]: `'global'` reproduces the requested amplitude *shape* exactly, up to one
multiplier; `'local'` gives every vertex its neighbourhood's maximum and therefore modulates that
shape. Generation uses `'local'`.

### Why not just a fixed η

At η ≳ 0.44 classical fixed-amplitude disorder **folds** meshes (10/10 seeds at η = 0.48), and on a
folded mesh solver-vs-sim agreement degrades from **1e-13 to 1.6e-2** — fine for a qualitative claim,
not fine for labels whose must-tier is 0.02. `displace_safe` cannot fold a mesh by construction. That
is the whole reason for using it.

### Coverage

Three arms; the **pilot** fixed the sweep, the two production arms are the dataset.

| arm | bases | mechanisms | `frac` | α | reps | nodes | n |
|---|---|---|---|---|---|---|---|
| pilot | 1 random | η, VD, η+α | 0.4, 0.8 | ±5, ±10 | 5 | 60 | 216 |
| **breadth** | 14 | η, VD, η+α | 0.2…0.8 | ±2, ±5, ±10 | 8 | 120 | **2184** |
| **large** | 13 | VD, η+α | 0.4…0.8 | +2, +5, +10 | 16 | 400 | **648** |

---

## Result

### The headline

**2832 networks, 0 inverted, 0 solver failures, 2832/2832 SPD.** The safety construction held across
every mechanism, base, amplitude and cell size.

| arm | n | auxetic (ν<0) | median ν | min ν |
|---|---|---|---|---|
| breadth | 2184 | 599 (**27.4 %**) | +0.174 | −1.048 |
| large | 648 | 341 (**52.6 %**) | −0.020 | −1.095 |

### By mechanism — η+α is the workhorse, and the effect is not small

| arm | mechanism | n | auxetic yield | median ν | min ν |
|---|---|---|---|---|---|
| breadth | η | 168 | 14.9 % | +0.208 | −0.593 |
| breadth | VD | 1008 | 22.1 % | +0.163 | −0.859 |
| breadth | η+α | 1008 | **34.8 %** | +0.183 | −1.048 |
| large | VD | 324 | 22.8 % | +0.124 | −0.532 |
| large | **η+α** | 324 | **82.4 %** | **−0.236** | −1.095 |

η+α at large cell size turns the distribution over: the *median* network is auxetic, not the tail.

### Response coverage of the combined 2832

| ν bin | n |
|---|---|
| < −1 | 7 |
| −1 … −0.5 | 118 |
| −0.5 … −0.2 | 375 |
| −0.2 … 0 | 440 |
| 0 … 1/3 | 1607 |
| > 1/3 | 285 |

`max|W|` median 5.67, p90 24.2, max 438; **30.4 % above 10** and 0.60 % above 100. E median 0.551,
min 4.7e-05. `eta_equiv` (largest node displacement in mean-bond-length units) median 0.611, max 1.47.

---

## The decisive finding: 9 of 14 bases cannot reach the isotropic cell at all

The `large` arm was launched to fill the **isotropic-auxetic** cell, on the 1/√N argument that
anisotropy self-averages with cell size. It did — but only on the bases that were capable in the first
place, and three quarters of its budget went to bases that structurally could not contribute.

Isotropy has no canonical threshold, so the claim is given at three cuts. The conclusion does not
depend on which is chosen.

| `aniso` cut | breadth, capable bases (n=780) | breadth, anisotropic bases (n=1404) | large, capable (n=162) | large, anisotropic (n=486) |
|---|---|---|---|---|
| < 1.05 | 0 (0.00 %) | **0** | 4 (2.47 %) | **0** |
| < 1.10 | 2 (0.26 %) | **0** | 11 (6.79 %) | **0** |
| < 1.20 | 19 (2.44 %) | **0** | 38 (23.5 %) | **0** |

**Zero isotropic-auxetic samples from the intrinsically anisotropic bases, across 1890 rows, at every
threshold.** On the capable bases the yield rises by roughly 10–25× with cell size, which is the 1/√N
self-averaging doing exactly what it should.

Measured base anisotropy (median over the arm's samples), which is what predicts capability:

| base | median `aniso` | iso-auxetic (cut 1.10) |
|---|---|---|
| `random_poisson_disk_0` | 1.17 | **7** |
| `bravais_p1.0_1.0_a2-a1` | 1.22 | **4** |
| `random_blue_noise_0` | 1.24 | **2** |
| `random_blue_noise_1` | 1.25 | 0 |
| `random_poisson_disk_1` | 1.34 | 0 |
| `bravais_p1.2_0.8_a2-a1` | 2.08 | 0 |
| `bravais_p0.6_1.3_a2-a1` | 2.54 | 0 |
| `bravais_p0.0_1.2_*` | 4.9–5.0 | 0 |
| `bravais_p0.0_1.0_*` | 5.5 | 0 |
| `bravais_p0.6_1.3_a1+a2` | 12.5 | 0 |
| `bravais_p1.0_1.0_a1+a2` | 32.4 | 0 |
| `bravais_p1.2_0.8_a1+a2` | 79.2 | 0 |

**Consequence, implemented:** `m2_auxetic_generator.py` gained a `--bases` filter with this table in
its docstring, so the next targeted arm spends its budget only where the cell is reachable.

### The `a1+a2` / `a2−a1` trap, worth stating once

The two Bravais diagonals are *not* interchangeable, and the naming hides it. At φ = ψ = 1:

| basis | sides | angles | shape quality | median `aniso` |
|---|---|---|---|---|
| `a2 − a1` | 1, 1, 1 | 60-60-60 | **1.000** — the true triangular lattice | 1.22 |
| `a1 + a2` | 1, 1, 1.7321 | 30-30-120 | 0.600 — flat, obtuse, non-Delaunay | 32.4 |

The user's instinct — *"you have to bond the difference"* — was right. Both are currently called
`p1.0_1.0`, which caused three wrong readings in one session; renaming them is logged in
`Phase 5/m2/PLAN_A.md` (deferred item 5) because it touches stored `topology_id` strings.

---

## Limitations — read before training on this data

1. **These carry SOLVER labels only.** Every record has `sim_status='not_run'`. **A tier-(A) sim
   cross-check must precede training on them**, weighted toward the deep-ν samples: 30 % of the set
   sits above `max|W|` = 10, and the folding measurement showed solver-sim agreement degrading in
   exactly that regime.
2. **`eta_equiv` exceeds the classical band** (median 0.611, max 1.47 vs η < 0.5). That is intended —
   area-positivity, not the collision bound, is the real constraint, and the regular-lattice floor is
   √3/4 = 0.433, not 1/2 (`a7192cb`). But it does mean these geometries sit **outside the validated
   η-disorder family**, which is a second reason the sim check is not optional.
3. **The isotropy yield is still low in absolute terms** — 13 samples at the 1.10 cut across 2832. The
   targeted arm (`--bases p1.0_1.0_a2-a1,random --reps 16`) is the fix, at roughly 4× the yield per
   unit compute. Not yet run.
4. **`auxgen_large.npz` was generated under the tag `isotropic`** and the JSON's internal `config`
   still records that name and the old `save_npz` path. The file was renamed after the measurement
   above showed it contains mostly large-cell *anisotropic* samples. Provenance is intact; the name in
   the config is historical.
5. **No figures.** This is a generation run scored by tables; the networks themselves are drawn
   downstream if they enter training.

## Verification

- **0 inverted / 0 failed** is the primary gate and is reported by the generator on every arm.
- The perturbation itself is gated by **12 tests** in `Phase 5/verifications/test_displace_safe.py`,
  including [2b] exactness of the requested amplitude profile under `scale='global'`, [2d]
  size-independence, [2g] local-vs-global (local strictly better on a disordered mesh, not worse on a
  regular one, no inversions in either mode), and [2h] the √3/4 regular-lattice floor.
- The dataset records are built through **`build_dataset.graph_of` / `BD.save`** — the builder's own
  packer, reused rather than reimplemented — and both `.npz` load through `train_v2.load` with the
  builder's invariant at **1.7e-16**.
- Bases, mechanism, `frac`, α and seed are stored per record (`family`, `topology_id`,
  `geom_variant`, `k_pattern`, `k_source`, `traj_id`, `seed`, `n_threads`, `commit`), so a holdout can
  be taken on any of those axes without parsing strings.
