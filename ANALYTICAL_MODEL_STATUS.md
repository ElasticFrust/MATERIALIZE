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

### KKT edge-compatibility correction (implemented — June 2026)

The mean-field limitation was partially addressed by adding a KKT projection step after the Woodbury solve. For every interior edge $e$ shared by triangles $s_1$ and $s_2$ with edge vector $\Delta x$, the corrected $W$ satisfies:

$$\bigl[W(s_1) - W(s_2)\bigr]^{\alpha\beta}_{\mu\nu}\, \Delta x^\mu \Delta x^\nu = 0$$

The correction is geometrically equivalent to projecting $W_0$ (the unconstrained mean-field solution) onto the subspace of compatible metric fields, using the KKT system:

$$\begin{pmatrix} P & J^\top \\ J & 0 \end{pmatrix} \begin{pmatrix} W \\ \Lambda \end{pmatrix} = \begin{pmatrix} -\delta A \\ 0 \end{pmatrix}$$

**Effect on geometric disorder (uniform k):** For networks with $\eta \leq 0.30$, the correction consistently shifts $\nu$ more negative by $0.001$–$0.028$. At $\eta \geq 0.35$, both solvers become unstable (near-degenerate triangles), but the KKT correction partially stabilizes the median.

**Effect on rigidity heterogeneity (the known failure mode):** The KKT correction enforces *edge-length* compatibility between neighbouring triangles, not full strain-field compatibility. It does not capture the long-range cooperative effects (percolation-like soft channels, stiff backbones) responsible for the 70% error on VD rigidity patterns. The correction reduces the error but does not eliminate it.

**Implementation:** See `Phase 2/forward_solver_torch.py` — `_woodbury_kkt_sparse()` for networks with $N > 500$ triangles; `_woodbury_solve(J=...)` for smaller differentiable networks. Full algorithm documented in `Phase 2/README.md`.

### Remaining fixes (to be explored)

1. **Periodic boundary condition simulation** — better ground truth, matches the analytical infinite-medium assumption.
2. **Cluster self-consistent method** — embed clusters of neighbouring triangles rather than single ones.
3. **Differential effective medium** — add heterogeneity incrementally.
4. **Reduce rigidity contrast** — use smaller `a` parameter so the mean-field remains valid.
5. **Accept the limitation** — use the pipeline for designs with moderate rigidity contrast, where it is accurate.
