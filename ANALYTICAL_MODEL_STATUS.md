# Analytical Model Status — Known Limitations

## The D2C (Disc-to-Continuum) forward solver has a mean-field limitation

The Woodbury-based self-consistent homogenisation (Phase 2's `forward_solver_torch.py`) computes the effective elastic tensor of a spring network by treating each triangle as if embedded in the average effective medium. This is a **mean-field approximation**.

### Where it works well

- **Uniform rigidities** on any geometry: <1% error for regular lattice, ~5% for deformed lattice.
- **Moderate rigidity contrast**: when all springs have similar stiffness, the mean-field assumption is accurate.
- **Geometric disorder alone**: deformed meshes with uniform k are well-captured.

### Where it fails

- **Highly heterogeneous rigidities** (VD with a=10 produces binary k near 0 or 2):
  - Regular lattice + VD rigidities: **70% error** on Poisson ratio
  - Deformed lattice + VD rigidities: **55% error** on Poisson ratio

### Root cause

The mean-field self-consistent scheme ignores **spatial correlations** between neighbouring triangles. When rigidities are nearly binary (stiff/soft), the network develops percolation-like "soft channels" and "stiff backbones" whose cooperative behaviour cannot be captured by averaging each triangle independently into the effective medium.

### What this means for Phase 3 inverse optimisation

The inverse pipeline (Phase 3) correctly inverts the forward solver to machine precision — the issue is not the inversion but the forward model itself. Solutions found by the optimiser satisfy T(k_opt) = T_target within the D2C model, but **the D2C model does not accurately represent the true mechanical response** when rigidity heterogeneity is strong.

This matters because:
1. Optimised rigidity patterns with high contrast may not produce the predicted Poisson ratio in a real (simulated or physical) network.
2. Designs that rely on subtle near-auxetic behaviour (nu near 0 or negative) through rigidity patterning may be unreliable.

### Validation evidence

Full numerical comparison in `Phase 3/results_summary.txt` and `Phase 3/SESSION_SUMMARY.md`.

| Config | nu (D2C) | nu (simulation) | Error |
|--------|----------|-----------------|-------|
| Regular, k=1 (baseline) | 0.333 | 0.330 | 0.9% |
| Regular + VD rigidities | 0.349 | 0.206 | 69.7% |
| Deformed, k=1 | 0.299 | 0.315 | 4.9% |
| Deformed + VD rigidities | 0.049 | 0.107 | 54.5% |

Note: the simulation (KUBC) also has known biases (38% boundary-constrained nodes). A periodic boundary condition (PBC) simulation is the next step for a fairer comparison (see `Phase 4 - PBC simulation/`).

### Possible fixes (to be explored)

1. **Periodic boundary condition simulation** — better ground truth, matches the analytical infinite-medium assumption.
2. **Cluster self-consistent method** — embed clusters of neighbouring triangles rather than single ones.
3. **Differential effective medium** — add heterogeneity incrementally.
4. **Reduce rigidity contrast** — use smaller `a` parameter so the mean-field remains valid.
5. **Accept the limitation** — use the pipeline for designs with moderate rigidity contrast, where it is accurate.
