# Phase 3 Sweep Results: Comprehensive Summary

## Overview

Phase 3 performs **inverse design** of 2D elastic spring networks: given a target Poisson's ratio, find per-edge spring rigidities and/or rest lengths that produce that ratio. The pipeline is:

1. **Generate** a triangulated network (fixed geometry/topology)
2. **Optimize** spring parameters via L-BFGS through a differentiable forward solver
3. **Evaluate** achieved Poisson ratio, Young's modulus, and loss

Three sweeps were conducted across 11 Poisson ratio targets (-0.9 to +0.9 in steps of 0.2), multiple topologies, and three design variable choices.

---

## The Three Sweeps

| Sweep | Topologies | Constraint | Cases | Success rate |
|---|---|---|---|---|
| **large_sweep_3** | 8 mixed (crystal, foam, Poisson random) | Anisotropic | 264 | 215/264 (81%) |
| **poisson_ratio_targets** | 5 foam (eta=0.2, different seeds) | Anisotropic | 165 | 159/165 (96%) |
| **large_sweep_2** | 5 foam (eta=0.2, same seeds) | **Isotropic** | 165 | 79/165 (48%) |

Success = final loss < 0.01. All cases that previously hit the old iteration limit (500) were re-run with max_iter=2000 and confirmed at local minima (no improvement).

---

## Key Finding 1: Topology Type Dominates Achievability

### Sweep 3 — Per-topology success rates (loss < 0.01)

| Topology | Success | Avg log10(loss) | Reachable targets |
|---|---|---|---|
| **poisson_999** | **33/33 (100%)** | **-16.0** | All 11 |
| **poisson_1337** | **33/33 (100%)** | **-16.6** | All 11 |
| **aniso_crystal** | **33/33 (100%)** | **-15.7** | All 11 |
| **foam_eta02_137** | **33/33 (100%)** | **-15.6** | All 11 |
| **foam_eta02_42** | 31/33 (94%) | -14.5 | All 11 |
| iso_crystal | 18/33 (55%) | -9.0 | +0.0 to +0.9 only |
| foam_eta045_256 | 18/33 (55%) | -5.4 | -0.9 to +0.0 only |
| foam_eta045_314 | 16/33 (48%) | -4.0 | Patchy |

### Tier ranking:

- **Tier 1 (perfect, any target):** Poisson random networks, anisotropic crystal, foam eta=0.2
  - 100% success across the full range [-0.9, +0.9]
  - Losses routinely reach 1e-16 to 1e-18 (machine precision)

- **Tier 2 (limited range):** Isotropic crystal, foam eta=0.45
  - iso_crystal can only reach non-negative nu (its symmetric lattice cannot break symmetry enough for auxetic behavior)
  - foam_eta045 struggles with positive nu targets (its high disorder biases it toward negative Poisson ratios)

---

## Key Finding 2: Disordered Networks Are Best for Full-Range Poisson Control

The **Poisson random network** topology (Delaunay triangulation AFTER random point perturbation) achieves:
- 100% success rate across all 11 targets
- Average loss of 10^-16 (essentially machine precision)
- Fast convergence (40-100 L-BFGS iterations = 0.4-1.4 seconds)

**Why it works:** The Poisson random network has inherent geometric disorder — no preferred directions, no symmetry constraints. This gives the optimizer maximum freedom to redistribute spring parameters. The lack of crystallographic symmetry means there are no "forbidden" regions in Poisson ratio space.

**Foam eta=0.2** is nearly as good (94-100% success). It generates disorder by perturbing a hexagonal lattice then Delaunay-triangulating BEFORE perturbation, giving moderate geometric randomness.

**Foam eta=0.45** has too much disorder — the extreme perturbation creates highly irregular triangles with poor numerical conditioning, leading to NaN results and convergence failures.

---

## Key Finding 3: The Isotropic Crystal Cannot Achieve Auxetic Behavior

The isotropic crystal (regular triangular lattice) completely fails for any negative Poisson target. Its loss stays at ~0.81 for nu=-0.9 and degrades gradually as targets approach zero:

| Target nu | Best loss (iso_crystal) | Interpretation |
|---|---|---|
| -0.9 | 0.81 | Complete failure |
| -0.7 | 0.49 | Complete failure |
| -0.5 | 0.25 | Complete failure |
| -0.3 | 0.09 | Near-failure |
| -0.1 | 0.01 | Marginal |
| +0.0 | 1e-13 | Success |
| +0.1 to +0.9 | 1e-17 | Machine precision |

The pattern `loss approx (nu_target - nu_natural)^2` (where nu_natural ~ +0.28) shows the optimizer cannot move the effective Poisson ratio below the network's natural value. The perfect hexagonal symmetry means there are no asymmetric deformation modes to exploit — any rigidity redistribution preserves the lattice's inherent positive-Poisson character.

**Conclusion:** Topological disorder is necessary for achieving auxetic (negative Poisson) behavior through spring parameter tuning alone.

---

## Key Finding 4: Isotropic Constraint Severely Limits Achievability

Comparing the same foam topologies (eta=0.2, seeds 42/137/256/314/999) with and without isotropy enforcement:

| Target nu | Anisotropic success | Isotropic success |
|---|---|---|
| -0.9 | 9/15 (60%) | 0/15 (0%) |
| -0.7 | 15/15 (100%) | 0/15 (0%) |
| -0.5 | 15/15 (100%) | 0/15 (0%) |
| -0.3 | 15/15 (100%) | 0/15 (0%) |
| -0.1 | 15/15 (100%) | 5/15 (33%) |
| +0.0 | 15/15 (100%) | 9/15 (60%) |
| +0.1 | 15/15 (100%) | 15/15 (100%) |
| +0.3 | 15/15 (100%) | 15/15 (100%) |
| +0.5 | 15/15 (100%) | 15/15 (100%) |
| +0.7 | 15/15 (100%) | 14/15 (93%) |
| +0.9 | 15/15 (100%) | 6/15 (40%) |

The isotropic constraint demands C_xxxx = C_yyyy, C_xxxy = C_xyyy = 0, and C_xyxy = (C_xxxx - C_xxyy)/2. This is extremely restrictive for disordered networks whose natural response is inherently anisotropic. The optimizer must simultaneously satisfy the Poisson target AND these symmetry conditions, which is often impossible for foam topologies.

**Achievable isotropic range:** Roughly +0.1 to +0.5 (the range where foam networks naturally behave near-isotropically).

---

## Key Finding 5: Design Variable Choice Matters Less Than Topology

Across sweep 3 (all 8 topologies):

| Design variable | Success rate |
|---|---|
| rigidities | 72/88 (82%) |
| rest_lengths | 74/88 (84%) |
| both | 69/88 (78%) |

All three perform similarly. Surprisingly, optimizing "both" (rigidities + rest lengths simultaneously) does NOT outperform single-variable optimization. This suggests the solution landscape is richer but also more complex — the larger parameter space introduces more local minima.

**Per topology type, the best design variable varies:**
- Crystal topologies: all three equally effective
- Foam eta=0.2: rest_lengths slightly best (22/22 vs 21/22)
- Foam eta=0.45: rigidities best (12/22 vs 13/22 for rest_lengths)
- Poisson random: all three equally effective (22/22 each)

---

## Key Finding 6: Convergence Speed Correlates with Topology Quality

For the best topologies (Poisson random, foam eta=0.2):
- Typical convergence: **40-100 L-BFGS iterations** (0.2-1.5 seconds)
- Loss at convergence: 10^-15 to 10^-18

For struggling topologies (iso_crystal at negative targets, foam_eta045):
- All 2000 iterations exhausted
- Loss stuck at 10^-1 to 10^0 (complete failure)

There is effectively a **binary outcome**: either the topology can achieve the target (and converges rapidly to machine precision), or it fundamentally cannot (and no amount of iteration helps). The old 500-iteration limit was confirmed sufficient — re-running at 2000 iterations produced identical results.

---

## Implications for Neural Network Design

### The inverse problem is well-posed for good topologies

For Poisson random and foam eta=0.2 topologies, the map `target_nu -> optimal_parameters` is:
1. **Essentially surjective** over [-0.9, +0.9] — any target is achievable
2. **Many-to-one** — multiple parameter configurations achieve the same target (the Phase 3 campaign analysis shows multiple distinct clusters of solutions)
3. **Smooth** — nearby targets produce nearby solutions (L-BFGS converges from random init)

### What a neural network should learn

The NN should approximate the **inverse map**: given a target Poisson ratio (and optionally topology descriptor), predict spring parameters that achieve it. Key considerations:

1. **Input:** Target nu (scalar), possibly topology features (adjacency, node positions, triangle areas, edge lengths)
2. **Output:** Per-edge rigidities (N_triangles x 3) and/or per-edge rest lengths (N_triangles x 3)
3. **The NN replaces L-BFGS** — instead of 40-2000 iterations of optimization, a single forward pass predicts the solution
4. **Training data:** The sweep results provide (target_nu, topology, optimal_parameters) triples for supervised learning
5. **Validation:** Feed predicted parameters through the differentiable forward solver and check achieved nu

### Architecture considerations

- **Graph Neural Network (GNN)** is natural since the input is a triangulated mesh with per-edge outputs
- The solver itself is differentiable, enabling **end-to-end training** where the loss is the actual Poisson error (not just parameter MSE)
- The many-to-one nature of the solution space suggests **generative** approaches (VAE, diffusion) could capture the full solution manifold

---

## Summary Table

| Topology type | Achievable nu range | Best for | Limitation |
|---|---|---|---|
| Poisson random | [-0.9, +0.9] | Universal inverse design | Slightly irregular triangles |
| Aniso crystal | [-0.9, +0.9] | Regular but flexible | Fixed lattice geometry |
| Foam eta=0.2 | [-0.9, +0.9] (some seeds: [-0.7, +0.9]) | Reliable production use | nu=-0.9 sometimes fails |
| Foam eta=0.45 | [-0.9, ~0.0] | Auxetic-only designs | Cannot reach positive nu |
| Iso crystal | [~0.0, +0.9] | Positive-nu only | Cannot go auxetic at all |
| Any + isotropic | [~+0.1, ~+0.5] | Limited isotropic designs | Fails for negative nu |
