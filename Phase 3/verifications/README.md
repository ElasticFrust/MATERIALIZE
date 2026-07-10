# Phase 3 — inverse-design verifications

Each case **designs** k with the inverse designer (`../inverse_design.py`), then **independently
simulates** the designed network (PBC relaxation → physical per-triangle tensor) and checks it
does as prescribed. Every case runs a matrix of **topologies × sizes**, and saves in its own folder:
a **`<case>.csv`** table (printed too) plus four kinds of figure — **`_summary.png`** (all
topologies overlaid, sim vs solver vs target), **`_bytopo.png`** (per-topology small multiples),
**`_detail.png`** (5-topology grid: designed rigidity k · local ν · local E, filled triangles, target
region marked in lime), and one **`<case>_<topology>.png`** detail file per topology.

Run a case:
```
python "Phase 3/verifications/<case>/design_and_verify.py"
```

## Shared harness — `_common.py`
- **Lattice constructor (`make_lattice`):** base vectors `v1=(1,0)`, `v2=(φ/2, ψ·√3/2)`
  (`φ=ψ=1` → regular triangular); keep a SQUARE real-space region as an axis-aligned PERIODIC box;
  PBC via periodic Delaunay. `verify_lattice.py` is the geometry/physics/design gate (regular →
  ν=1/3, E=2/√3; solver = sim exactly; boxes square; valid PBC).
- **Topologies (periodic):** `regular` (φ=ψ=1) · `aniso_str` (ψ=0.6, compressed rows) ·
  `aniso_shr` (φ=1.5, sheared) · `disorder_lo` (η=0.20) · `disorder_hi` (η=0.35). See
  `topologies_overview.png` (real geometry — the anisotropy is visible).
- **Sizes:** `SIZES = [8, 12]` (square half-size) → ~600 / ~1300–2200 triangles (both adjoint path).
- **Plots:** REAL geometry in a SQUARE axes frame + periodic-box outline (`draw_network` with line
  color/width ∝ k; `local_field_smooth` + `fill_local_map` → local ν/E as **filled triangles**;
  `square_frame`/`draw_box`; `mark_region` outlines the target region in lime). The detail figures
  (`design_detail_figure` grid + `design_detail_per_topology` files) show, per design, the designed
  rigidity network alongside its local ν and local E maps.
- **Independent simulation:** `sim_per_triangle_C6` relaxes the designed network once and returns
  the physical per-triangle tensors; `region_phys_C6` averages any region; `nu_E_theta` gives the
  directional ν(θ)/E(θ); `solver_region_nuE` is the solver's own prediction (plotted beside the sim
  on every summary). A light uniformity regulariser (`reg` in `optimize`) discourages the optimiser
  from exploiting floppy/unstable configurations.

**Verification strength (stated on each plot):**
- **Global ν/E** — the sim ground truth (energy = virial) is *independent* of the solver's
  homogenisation → a genuine end-to-end check.
- **Local/regional ν** — defined as the region-averaged per-triangle physical tensor; the sim uses
  the same definition, so it validates the design→realise→simulate loop and the *spatial pattern*
  (not a fully independent sub-region modulus measurement).

## Cases
| case | what it designs | verified against sim |
|---|---|---|
| `auxetic_sweep` | global ν over a sweep (+0.3 … −0.6) | simulated global ν vs target; **achievable range** per topology; unstable designs flagged; per-topology detail across **all** targets (`_alltargets_<topo>.png`) |
| `auxetic_patch` | **local/spatial control** — G1 shape gallery (disc/square/triangle/ring), G2 contrast variety (auxetic-in-normal, normal-in-auxetic, stiff-/soft-E), G3 **decoupled** (E differs only in R_E, ν only in a different R_ν), + a regional ν(θ) isotropy check | simulated regional ν/E per region; local ν/E maps with the region marked |
| `graded_nu` | ν varying across x (+0.3 → −0.3) | simulated ν(x) profile; spatial local-ν map |
| `anisotropy` | directional ν(θ)/E(θ): **A** program any ν(θ) profile · **B** isotropise a base to a chosen flat ν0∈{0.8..−0.8} · **C** independent E/ν anisotropy (one modulus flat, the other directional) | simulated ν(θ)/E(θ) vs target curves |

Directional design uses the `nu_theta`/`E_theta` objectives (see `../README.md`); the harness adds
shaped regions (`region_shape`) and multi-region marking (`mark_region`).

## Findings (summary)
- **auxetic_sweep:** designs hit their global ν target and the *independent* simulation confirms it,
  up to a **topology-dependent limit** — disordered η=0.35 reaches ν=−0.6, disordered η=0.20 ≈ −0.45,
  the regular lattice ≈ −0.3, while the anisotropic lattices barely go auxetic. Beyond each limit the
  ν-only objective produces floppy networks that the simulation flags as **unstable** (6 cases, all at
  the extremes).
- **auxetic_patch:** the prescribed *spatial* pattern holds under real physics for every region shape
  (disc/square/triangle/ring) and position. Contrast is general — auxetic-in-normal, stiff-/soft-E, and
  (on a base that reaches global auxetic, e.g. η=0.35) normal-in-auxetic. **Decoupled control works:** E
  is stiffened only in R_E and ν made auxetic only in a *different* R_ν, simultaneously, with the
  background normal in both. **Caveat surfaced by the regional ν(θ) check:** a scalar regional ν hides
  strong angular variation — inside an auxetic patch ν(θ) is far from flat, so the patch is auxetic *on
  average* but **not isotropically**.
- **graded_nu:** the simulated ν(x) follows the prescribed +0.3→−0.3 left-to-right gradient for four of
  five topologies; the strongly stretched lattice saturates on the auxetic end and disordered η=0.35 is
  an outlier (its strips don't track the gradient).
- **anisotropy (3 tests):**
  - **A/program:** k-design realises a prescribed ν(θ) profile; the **disordered** lattices are the most
    designable (a 4-fold profile to max|err|≈0.05), while the regular lattice saturates on the
    sign-flipping directional-auxetic profile.
  - **B/isotropise:** the disordered base can be flattened to a chosen isotropic ν0 across the whole
    range **−0.8…+0.5** (near-zero angular spread); the crystals reach the mid-range levels cleanly but
    **fail at the extremes ±0.8**, and the strongly compressed lattice resists the high-positive levels.
  - **C/independent E,ν:** E can be made directional while ν stays ~flat (on the regular lattice); the
    reverse only partly succeeds — strong ν-anisotropy generically forces E-anisotropy (a coupling
    asymmetry, not a solver limit).

## Extended studies (maps, all-topology sweeps, regimes, large-N)
Everything below **saves every designed network** to `<case>/networks/*.npz`; the map scripts *load*
them (no re-optimising).
- **`anisotropy/make_maps.py`** — local ν & E maps for **every** anisotropy design (per-test grids
  `anisotropy_maps_{A,B,C}_{nu,E}.png`). The B/ν grid shows disorder_hi isotropising *uniformly* at
  every ν0 while the crystals go patchy at the hard targets.
- **`auxetic_patch/design_all.py` + `make_maps.py`** — all 8 patch cases (disc/square/triangle/ring
  auxetic, normal-in-auxetic, stiff-/soft-E, decoupled) × all 5 topologies, with **per-topology**
  (`_bytopo_<topo>.png`) and **per-case** (`_bycase_<case>.png`) local maps. Decoupled E/ν holds on
  every topology (ν auxetic only in R_ν, E stiff only in R_E).
- **`regimes/working_regimes.py`** — good working regimes: the **regular** lattice isotropises to any
  ν0∈[−0.8,+0.5] (flatness ~0 → ordered *isotropic-auxetic*); a *realizable* anisotropic target is hit
  to <0.01 on **every** topology; independent E/ν decouples cleanly in the E-directional/ν-flat
  direction (the reverse trades off).
- **`large16k/large_designs.py`** — the standout of each type (iso / aniso / indep / decoupled patch)
  designed at **~16 000 triangles** on regular vs disordered, with local maps.
- **`vd_demo/`** — the VD (virtual-distortion) rigidity study on a regular lattice (see its README):
  rigidity contrast alone is auxetic above a contrast threshold; solver = simulation at 16k triangles.
- **`two_region/`** — does a design's prescribed behaviour survive a REAL open cut-and-stretch (clamp
  x only on the two ends, everything else incl. clamp-y free), and does gluing two differently-behaved
  regions together actually produce a differential mechanical response? All designs saved as `*.npz`.
  - **`_common.glue(pieces, Lx, Ly)`** (in `../_common.py`) — the key technique: design each region as
    its OWN independent whole-domain periodic material (no shared design variables), then retriangulate
    the union of the regions' point clouds and transfer each region's own designed bond stiffnesses by
    matching real endpoint positions; any brand-new interface bond (the physical seam) defaults to a
    plain undesigned k=1. **Do not jointly optimise one connected lattice with region-scoped
    objectives** for a multi-region design — the optimiser exploits shared interface bonds, driving
    them to exactly k=0 (a hinge/mechanism), which showed up as a chaotic, unphysically "soft" zone
    right at the boundary under a real stretch test even though the region-averaged ν still hit target.
  - **`ribbon.py`** — an auxetic (ν=−0.5) and a regular (ν=+0.5) 30×30 patch glued side by side.
    `ribbon_fields.py` adds local ν/E/√‖stress‖ maps. Local ν(x) from the actual open-stretch
    simulation transitions smoothly from ≈−0.5 to ≈+0.5 right at the glued seam.
  - **`inclusion_square.py`** — a 30×30 regular matrix (ν=+0.5) with a 10×10 auxetic (ν=−0.5) square
    inclusion, glued, at three inclusion rigidities (stiff/same/soft E vs the matrix).
    `inclusion_square_fields.py` adds local ν/E/√‖stress‖ maps. Unlike the ribbon (each region spans
    the full specimen height), the small inclusion's embedded lateral strain stays near-zero/wrong-
    signed under edge stretch regardless of rigidity contrast — it's mechanically dominated by the
    much larger surrounding matrix (a composite/Eshelby-type inclusion effect), not a design failure.
  - **`inclusion_rigidity_only.py`** — control/sanity check: matrix and inclusion both at the plain
    ν=1/3 (no design needed — nu is scale-invariant under uniform k-rescaling), only rigidity contrast
    (stiff ×6 / soft ×1/6). Confirms the glue+cut-stretch pipeline reproduces ordinary composite
    behaviour with no sign anomalies: the stiff region strains less and carries more stress, the soft
    region strains more and carries less.
  - **`demo.py`/`response_fields.py`** — an earlier pair of two-region demos (a disorder_hi bar,
    auxetic-vs-regular top/bottom; a circular auxetic disc in a soft matrix at two rigidities) with
    coarse-grained strain/stress response fields, predating the `glue()` technique above.

_(See each case's `_summary.png`/`_bytopo.png`/`_detail.png` for the result and its `<case>.csv` for
the raw numbers.)_

## `strain_stress/` — designing the ACTUAL per-triangle response, not just derived ν/E
`inverse_design.py` gained two new `Objective` kinds, `'strain'` and `'stress'`, that fit the actual
per-triangle metric-change response (vec3=[xx,xy,yy]) under an explicit applied macro load, plus an
opt-in `homogeneity` penalty (within-region response variance) for any kind. Both reuse the existing
differentiable `forward()` outputs (`bare`, `W`) — no solver change — via the new
`per_triangle_strain_stress(bare, W, load)` helper. The convention (Voigt basis, the `(I+W)@load`
strain-concentration formula, no extra factor-of-2 on shear) was pinned *empirically*: a unit test
checks it against `_common.unit_mode_response`'s independent non-autograd NumPy PBC simulation.
Whole-cell mean strain is degenerate (it always equals the applied load exactly, since the
fluctuation field has zero cell-mean) — `'strain'` objectives only make sense on a sub-region;
`'stress'` is designable everywhere.

`strain_stress/design_and_verify.py` has four demos:
- **stress concentrator** (uniaxial pull) — a patch designed to carry amplified σ_xx.
- **strain shield** (uniaxial pull) — a (different) patch designed for near-zero local strain (rigid
  inclusion).
- **bulge — auxetic inclusion** (uniaxial pull) — an off-centre top rectangle of a regular cell
  designed as a strongly AUXETIC material (ν<0) in a flat ν=0 background. An auxetic inclusion expands
  laterally under an x-pull, so it bulges where the ν=0 surround stays neutral; and because ν is a
  clean *stable* material property (not a raw strain), the bulge appears in the real open cut-and-
  stretch test WITHOUT a near-mechanism. On a regular lattice ν cleanly reaches only ~-0.4 (the
  auxetic_sweep limit) and a small inclusion is **Eshelby-dominated** by the surrounding matrix (see
  `two_region/inclusion_square.py`), so the open-stretch signature is moderate (the patch contracts
  *less* than its surroundings) — the honest physical result; the clean local-ν map is the primary
  design-intent figure. *History kept as a lesson:* a low-σ_xx `'stress'` patch target
  under-constrained the deformation SIGN; a raw `'strain'` eyy target pinned the sign but a STRONG one
  (eyy~2.5, a 250 % response) physically requires a near-mechanism and rendered as floppy strain
  streaks — designing the patch as an auxetic *material* is the clean, mechanism-free route.
- **concentric rings** (isotropic stretch) — a stress "bullseye": three contiguous rings jointly
  designed for alternating-**sign** mean stress p=(σ_xx+σ_yy)/2 (+A/−A/+A) in one optimize() call, at
  two magnitudes (0.35, 1.0) on regular and disorder_hi (4 designs). **Negative result, kept documented
  rather than hidden:** the middle ring (sandwiched between two same-sign neighbours) fails — under
  uniaxial pull it underachieves and goes wrong-sign as magnitude/disorder grow; under an isotropic
  load (matching the rings' symmetry) it is *worse*, blowing the independent check up numerically. The
  reason is physical: under an imposed global dilation a passive *stable* sub-region's mean stress must
  share the dilation's sign (the opposite needs a locally negative bulk modulus, forbidden for a stable
  material) — the optimizer can only fake it with a near-mechanism, which is exactly why the independent
  nonlinear relaxation becomes ill-conditioned. The achievable, *cool* bullseye is the same-SIGN
  alternating-MAGNITUDE version at 16k triangles below.

**`strain_stress/large16k_rings.py`** — the achievable "bullseye" scaled to ~16k triangles (HALF=42),
using alternating **magnitude** (STRONG/WEAK/STRONG/WEAK, all same positive sign — nothing forbidden)
under the isotropic load, with a low-weight per-band `'isotropy'` objective. The STRONG/WEAK levels are
both genuinely **stress-bearing** (a ~2× contrast, 0.8/0.4) — an earlier 10× contrast (1.0/0.1) drove
the weak bands so soft they became near-mechanisms at 16k and the independent nonlinear relaxation
diverged (stress ~1e19). Plots BOTH the mean-stress and mean-strain response fields, and each band's
directional ν(θ)/E(θ) average (polar), on regular and disorder_hi. Outputs `large16k_rings.csv`,
`strain_stress_large16k_rings_*.png`.

**`strain_stress/mode_selective.py`** — a single ~16k-triangle network whose stress pattern **depends
on the load direction**: stretch +x lights up a filled DISC of high stress; compress −y lights up a
separate TRIANGLE; each shape stays quiet under the other load, and the background is suppressed for
definition. On regular and disorder_hi. This is the cleanest demonstration of the **"full response
operator"** character of the formulation: because the solver returns the whole strain-concentration
operator W (not one load's answer), **all objectives across BOTH loads** (disc-loud/triangle-quiet
under +x, triangle-loud/disc-quiet under −y, background suppressed under both) live in **one**
`optimize()` call, querying the same W at two different loads — a displacement/FEM design loop would
need a separate solve + adjoint per load case. Mechanism: directional stiffness contrast (disc
stiff-x/soft-y, triangle stiff-y/soft-x). Achieved: disc ~6–10× own-load selectivity, triangle
~2.5–3.4× (the disc is consistently sharper — "stiff-x/soft-y" is an easier anisotropy to build than
a triangle forced "stiff-y/soft-x" across its own edges). Outputs `mode_selective.csv`,
`strain_stress_mode_selective_<topo>.png`.

**`strain_stress/mode_selective_colocated.py`** — the harder variant: the SAME spot shows a DIFFERENT
shape by load direction. Two shape-sets (× regular/disorder_hi):
- **`cross_bars`** — a full-width HORIZONTAL bar (lights under +x) crossing a full-height VERTICAL bar
  (lights under −y): a "+" whose arms are direction-selective, ~9–10× both, **cleanly rendered**.
- **`disc_triangle`** — an inner DISC (under +x) nested in a TRIANGULAR frame (under −y): striking
  (disc ~6–7×, frame ~2.7–3.3×) but the disc keeps a horizontal feeder streak.

The fidelity gap is an honest **physical** point, not a tuning failure: stress under a uniform load is
not free-form — equilibrium (div σ = 0) conserves the load flux across every cut, so high stress forms
a connected PATH spanning the cell in the load direction. A bar aligned with the load **is** such a
path (renders crisply); an isolated blob (a disc) cannot carry the flux alone, so the horizontal strip
at its height necessarily lights up too (Eshelby gives a uniform disc *interior*, but the flux still
has to enter and leave). Takeaway: **load-aligned shapes render cleanly; shapes across the load always
grow a feeder streak.** Outputs `mode_selective_colocated.csv`,
`strain_stress_mode_selective_colocated_<kind>_<topo>.png`. `replot_clean.py` re-renders any saved
mode-selective network *without* the guiding region outlines (`*_clean.png`) — the shapes on their own.

**`strain_stress/mode_selective_strain.py`** — the STRAIN analogue: pull +x and a DISC undergoes total
DILATION (fills red) while a separate TRIANGLE blends into the background; pull +y and they SWAP. On
regular + disorder_hi. This renders as **clean FILLED shapes** (unlike the stress load-paths) because
strain concentrates *locally* in a soft/auxetic inclusion (Eshelby: ~uniform strain *inside* it) rather
than percolating, and a `homogeneity` penalty keeps each shape uniform. Physical constraints pinned
here: (a) the **cell-mean dilation is fixed by the pull** (≈1 in the metric convention), so a region can
only stand out by dilating ABOVE the mean — you cannot suppress the surround toward zero (an early
attempt was infeasible and pushed the compensating strain into floppy streaks); the active shape targets
dilation > mean, and the **inactive shape targets the background mean so it vanishes** (a rigid=0 target
made it a visible white hole). (b) A strong dilation needs a very SOFT shape (E→0, near-mechanism), so
the target is kept moderate (DIL=1.25) — the shapes sit at E≈0.25–0.30 vs the ≈0.94 background (a real
stable soft inclusion, not a mechanism), the tradeoff being gentler contrast. Each design also saves a
**material COMPOSITION** map (`_nuE.png`) — the local angle-averaged ν and E (`_common.local_nuE_angleavg`)
— showing the shapes are soft inclusions. Outputs `mode_selective_strain.csv`,
`strain_stress_mode_selective_strain_<topo>{,_clean,_nuE}.png`.

**`strain_stress/mode_selective_colocated_stress.py`** — the STRESS analogue of the co-located idea,
with two PULLS (εxx and εyy) and a FREE non-zero background: a filled DISC (responds to εxx) nested in a
filled TRIANGLE (responds to εyy). Under εyy the whole triangle fills; under εxx only the disc reads. The
honest caveat (documented in the script): because stress follows equilibrium load PATHS, the disc under
εxx is a horizontal load-path streak and its surrounding frame can't go fully quiet (disc/frame contrast
caps ~1.3×) — the εyy triangle fills fine, the εxx disc does not cleanly isolate. Contrast with the
strain version above, which renders both as clean filled shapes.

**Re-plot utilities (no re-optimising, load saved `.npz`):** `replot_clean.py` re-renders any saved
mode-selective network *without* the guiding region outlines (`*_clean.png`). `replot_bullseye.py` adds
two views to the `large16k_rings` bullseye: its **strain response** (local dilation under the isotropic
stretch — complementary to the stress bands) and its **composition** (local angle-averaged E and ν maps,
which reveal the alternating stiff/soft rigidity rings) — `strain_stress_large16k_rings_<topo>_{strain,composition}.png`.

All designs are checked two independent ways per the convention above (differentiable-path readback
+ `_common.unit_mode_response`'s separate NumPy simulation). Outputs: `strain_stress.csv`, `bulge.csv`,
`rings.csv`, `large16k_rings.csv`, `mode_selective.csv`, `mode_selective_colocated.csv`,
`mode_selective_strain.csv`, `mode_selective_colocated_stress.csv`, `strain_stress_*.png`, saved networks.

## Homogeneity regularizer
`Objective(..., homogeneity=w)` adds `w * var(local_field[region])` to the loss for any `nu`/`E`/
`strain`/`stress` kind — the loss-level analogue of the `glue()` workaround above: it discourages the
optimizer from hitting a region-mean target via a few floppy (k→0) hinge triangles instead of a
uniform response. `test_inverse_design.py::test_homogeneity_regularizer` confirms lower within-region
variance with the penalty on, at comparable achieved error, using the codebase's own `n_restarts`
robustness mechanism against LBFGS run-to-run nondeterminism.
