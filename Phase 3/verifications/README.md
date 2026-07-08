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

_(See each case's `_summary.png`/`_bytopo.png`/`_detail.png` for the result and its `<case>.csv` for
the raw numbers.)_
