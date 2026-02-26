# Phase 3 — Session Summary

## What we did today

This session focused on **validating the analytical D2C (Disc-to-Continuum) homogenisation model** against direct mechanical simulation of spring networks. The goal was to confirm that the Woodbury-based self-consistent effective-medium theory faithfully predicts the macroscopic elastic response (Poisson ratio, Young's modulus) when the network has heterogeneous rigidities and/or geometric deformation.

### 1. Built a KUBC mechanical simulator (`mechanical_simulation.py`)

We wrote a direct spring-energy minimisation code using Kinematic Uniform Boundary Conditions (KUBC):
- Boundary nodes (outermost 10% of the bounding box) are displaced affinely according to the applied macroscopic strain.
- Interior nodes relax via L-BFGS-B to minimise total spring energy E = Sum 1/2 k (|r_i - r_j| - l0)^2.
- Six independent strain tests (exx, eyy, exy, and three combinations) yield the full 3x3 Voigt stiffness matrix.
- Poisson ratio is extracted from the compliance matrix: S = inv(C), nu = -S12/S11.

### 2. Tested 4 network configurations (A-D)

All built from the same 14x14 mesh (~1200 nodes, ~1800 triangles, ~2800 edges):

| Config | Geometry | Rigidities | Description |
|--------|----------|------------|-------------|
| A | Regular triangular lattice | Uniform k=1 | Baseline |
| B | Regular triangular lattice | VD heterogeneous | Rigidity disorder only |
| C | Deformed (eta=0.15) | Uniform k=1 | Geometric disorder only |
| D | Deformed (eta=0.15) | VD heterogeneous | Both effects combined |

The VD (Virtual Distortion) rigidity rule is: k = 1 + tanh(a * (l_deformed - l0)) with a=10. This creates a nearly binary distribution: edges that lengthen under virtual perturbation get k near 2, edges that shorten get k near 0.

### 3. Compared analytical vs simulation results

| Config | nu (analytical) | nu (KUBC sim) | Error |
|--------|----------------|---------------|-------|
| A) Regular, k=1 | 0.333 | 0.330 | 0.9% |
| B) Regular, VD rigs | 0.349 | 0.206 | 69.7% |
| C) Deformed, k=1 | 0.299 | 0.315 | 4.9% |
| D) Deformed + VD | 0.049 | 0.107 | 54.5% |

**Key finding**: The analytical model agrees well for uniform rigidities (A: 0.9%, C: 4.9%) but has massive errors for heterogeneous rigidities (B: 69.7%, D: 54.5%).

### 4. Tested the 1/Area normalisation fix (`area_fix_comparison.py`)

Investigated whether replacing the hardcoded 1/16 factor in the bare tensor with 1/Area_s (each triangle's actual area) improves accuracy:
- For configs A-C: negligible difference (the fix is a near-identity for regular/near-regular meshes).
- For config D: **catastrophically wrong** — nu goes to 0.506 (375% error vs simulation's 0.107). The 1/Area fix breaks the self-consistent coupling for heterogeneous networks.

**Conclusion**: The 1/Area normalisation is rejected.

### 5. Built a ribbon (strip) simulation (`ribbon_simulation.py`)

An alternative validation approach: 4:1 aspect ratio strip, fix left/right edges in x, apply 1% uniaxial strain, measure transverse contraction in the middle 20% of the strip.

This avoids the KUBC bias (38% of nodes affinely constrained) and provides a more direct measurement of the Poisson ratio.

### 6. Diagnosed the discrepancy

The error pattern reveals that **rigidity heterogeneity**, not geometric deformation, is the dominant source of error:

**Why the analytical model fails for heterogeneous rigidities:**
- The D2C method is a **mean-field** (self-consistent) effective medium theory. Each triangle is treated as if embedded in the average effective medium.
- This ignores **spatial correlations**: a soft edge next to another soft edge behaves very differently than a soft edge surrounded by stiff ones.
- With a=10, the tanh gives nearly binary rigidities (k near 0 or 2), creating percolation-like "soft channels" that the mean-field approximation misses entirely.

**Why the KUBC simulation is also imperfect:**
- 38% of nodes are boundary-constrained, artificially stiffening the response.
- Finite size (14x14) introduces additional error.
- The analytical model assumes an infinite periodic medium — the simulation doesn't match this assumption.

**The truth is likely between the two**: the analytical model over-predicts Poisson for some configs and under-predicts for others; the KUBC simulation has its own biases.

---

## Conclusions and path forward

1. **The D2C analytical model works well for geometric disorder** (config C: 5% error) but **fails for rigidity disorder** (configs B, D: 50-70% error).

2. **The 1/16 normalisation should be kept** — the 1/Area alternative is worse.

3. **The inverse optimisation pipeline (Phases 2-3) is self-consistent**: it correctly inverts the forward solver to machine precision. The problem is not the inverse solver — it is the forward solver's mean-field approximation.

4. **Next steps require either**:
   - A better analytical theory (beyond mean-field: cluster methods, differential effective medium, etc.)
   - A better simulation (periodic boundary conditions) for ground truth
   - Acceptance of the mean-field limitations and use of the pipeline for design within its validity range (geometric disorder, moderate rigidity contrast)

---

## Files in this folder

### Core modules

| File | Description |
|------|-------------|
| `inverse_optimize.py` | Inverse optimisation engine: target tensor -> spring rigidities via L-BFGS through the differentiable forward solver. Supports tensor-matching and property-targeting modes, multi-start campaigns, clustering, round-trip validation. |
| `sweep_utils.py` | Shared utilities: topology generators (crystal, foam, Poisson random), VD rigidity assignment, single optimisation run, mesh plotting helpers. |
| `mechanical_simulation.py` | KUBC mechanical simulation: spring energy minimisation with affine boundary conditions, 6-strain-test stiffness extraction, compliance-based Poisson measurement. |
| `ribbon_simulation.py` | Ribbon (strip) simulation: 4:1 aspect ratio strip, uniaxial elongation, transverse contraction measurement in the middle. Alternative to KUBC. |
| `area_fix_comparison.py` | Three-way comparison: D2C with 1/16 factor vs D2C with 1/Area vs KUBC simulation. Shows 1/Area normalisation fails for heterogeneous networks. |

### Test suites

| File | Description |
|------|-------------|
| `test_inverse.py` | 6 tests: softplus roundtrip, single L-BFGS optimisation, Adam fallback, round-trip validation, gradient flow, mini campaign. |
| `test_virtual_distortion.py` | Tests for VD rigidity assignment: symmetry, tanh bounds, edge-length sensitivity. |

### Visualisation scripts

| File | Description |
|------|-------------|
| `plot_virtual_distortion.py` | Visualises VD rigidity patterns on the regular lattice. |
| `plot_virtual_deformed.py` | Visualises the deformed mesh with VD rigidities. |
| `crystal_diagnostic2.py` | Crystal lattice diagnostic: checks isotropy, edge angles, tensor symmetry. |

### Documentation

| File | Description |
|------|-------------|
| `README.md` | Phase 3 technical documentation: inverse problem formulation, algorithm details, test results, large-mesh benchmarks, code examples. |
| `SUMMARY.md` | Comprehensive sweep results: key findings from sweeps 2-6 on topology effects, isotropic constraints, design variable choice. |
| `results_summary.txt` | Detailed numerical results table: analytical vs KUBC simulation for all 4 configurations, area-fix comparison, three-way analysis. |
| `SESSION_SUMMARY.md` | This file. Summary of the validation session and analytical model diagnosis. |

### Sweep result directories

| Directory | Description |
|-----------|-------------|
| `large_sweep_2/` | 5 foam topologies, isotropic constraint, 3 design variables x 11 Poisson targets x 3 restarts. 48% success rate — isotropic constraint is very restrictive. |
| `large_sweep_3/` | 8 mixed topologies (crystal, foam, Poisson), anisotropic, 3 design variables x 11 targets x 3 restarts. 81% success. Key finding: topology type dominates achievability. |
| `large_sweep_4/` | Follow-up sweep with refined parameters. Similar structure to sweep 3. |
| `large_sweep_5/` | Most comprehensive sweep with aggregate analysis, residual energy plots, and per-material README files. |
| `large_sweep_6/` | Latest sweep iteration. |
| `poisson_ratio_targets/` | Systematic Poisson ratio targeting: 5 foam topologies, 11 targets, 3 design variables. 96% success rate for anisotropic case. |

### Output images

| File | Description |
|------|-------------|
| `mechanical_simulation.png` | 4x3 panel: reference meshes, uniaxial strain deformation, shear energy maps for all 4 configs. |
| `solution_comparison.png` | 4 different rigidity solutions that produce the same elastic tensor (solution manifold). |
| `rigidity_histograms.png` | Overlaid rigidity distributions for multiple solutions. |
| `poisson_01_solutions.png` | Mesh plots for Poisson=0.1 target solutions. |
| `poisson_01_histograms.png` | Rigidity distributions for Poisson=0.1 solutions. |
| `poisson_neg03_*.png` | Various plots for Poisson=-0.3 target (solutions, histograms, rigidities, rest lengths, combined). |
| `restlength_*.png` | Rest-length optimisation visualisations (histogram, mesh, polar). |
| `virtual_distortion_mesh.png` | VD rigidity pattern on regular lattice. |
| `virtual_deformed_mesh.png` | Deformed mesh with VD rigidities. |
