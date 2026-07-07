# Phase 3 — inverse-design verifications

Each case **designs** k with the inverse designer (`../inverse_design.py`), then **independently
simulates** the designed network (PBC relaxation → physical per-triangle tensor) and checks it
does as prescribed. Every case runs a matrix of **topologies × sizes**, and saves its figure +
`results.npz` in its own folder.

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
- **Sizes:** `SIZES = [8, 12]` (square half-size) → ~600 / ~1300 triangles (both adjoint path).
- **Plots:** REAL geometry in a SQUARE axes frame + periodic-box outline (`draw_network` with line
  color/width ∝ k; `local_scalar_field` → local ν/E maps; `square_frame`/`draw_box`). Each case also
  saves a `<case>_detail.png`: designed rigidity network + local ν map + local E map per
  representative design.
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
| `auxetic_sweep` | global ν over a sweep (+0.3 … −0.6) | simulated global ν vs target; **achievable range** per topology; unstable designs flagged |
| `auxetic_patch` | normal matrix (ν≈+0.3) with a central **auxetic patch** (ν≈−0.3) | simulated patch ν vs surroundings; spatial local-ν map |
| `graded_nu` | ν varying across x (+0.3 → −0.3) | simulated ν(x) profile; spatial local-ν map |
| `anisotropy` | directional ν(θ): **isotropize** the anisotropic bases, and **anisotropize** the isotropic ones | simulated ν(θ) vs target curves |

## Findings (summary)
- **auxetic_sweep:** designs hit their global ν target and the *independent* simulation confirms it,
  up to a **topology-dependent limit** — disordered η=0.35 reaches ν=−0.6, disordered η=0.20 ≈ −0.45,
  the regular lattice ≈ −0.3, while the anisotropic lattices barely go auxetic. Beyond each limit the
  ν-only objective produces floppy networks that the simulation flags as **unstable** (6 cases, all at
  the extremes).
- **auxetic_patch:** the simulated network is auxetic in the central patch and normal outside — the
  prescribed *spatial* pattern holds under real physics across all topologies; the local-ν map shows a
  clean auxetic disc in a positive matrix.
- **graded_nu:** the simulated ν(x) follows the prescribed +0.3→−0.3 left-to-right gradient for four of
  five topologies; the strongly stretched lattice saturates on the auxetic end and disordered η=0.35 is
  an outlier (its strips don't track the gradient).
- **anisotropy:** rigidity design **cancels geometric anisotropy** — ν(θ) flattens to the isotropic
  target 1/3 **fully** for the sheared and disordered lattices, **partially** for the strongly stretched
  lattice (a physical limit on how much stretch rigidity can offset); and conversely **induces** a
  prescribed anisotropic ν(θ) on **every** topology (all curves overlay the anisotropic target).

_(See each case's PNG for the quantitative result and its `results.npz` for the raw numbers.)_
