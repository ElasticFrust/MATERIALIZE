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

---

## Bug fixes and benchmarking — June 2026

### Bugs fixed in `_woodbury_kkt_sparse_combined`

Three bugs were found and fixed in the refactored unified KKT function:

**1. Spurious /N in non-AW Woodbury branch**

The rank-correction `Vy = B·y` is a sum over triangles (no 1/N). Two lines divided by `N` erroneously:
```python
# BEFORE (buggy):
Vy3 = np.einsum('nij,njk->ik', dM_np, y3) / N
gs3 = np.einsum('nij,njk->ik', dM_np, AinvCt) / N
# AFTER (fixed):
Vy3 = np.einsum('nij,njk->ik', dM_np, y3)
gs3 = np.einsum('nij,njk->ik', dM_np, AinvCt)
```
Effect: the Woodbury correction was ~N=800× too small, effectively disabling it.

**2. K0 einsum transposition — edge part**

`BMA_inv[s,i,k] = Σ_j δM[s,i,j]·M_inv[s,j,k]` is not symmetric. The K0 matrix
requires contracting `q[e,l]` against row `m`, column `l` of `BMA_inv`:
```python
# BEFORE (buggy): contracts q[e,l] against BMA_inv[s1,l,m] (wrong axis order)
qBM1 = np.einsum('el,elm->em', q_arr, BMA_inv[s1_arr])
# AFTER (fixed):
qBM1 = np.einsum('el,eml->em', q_arr, BMA_inv[s1_arr])
```

**3. K0 einsum transposition — angle part**

Same transposition error in the angle-constraint branch:
```python
# BEFORE (buggy):
aBM = np.einsum('pl,plm->pm', a_arr, BMA_inv[sv_arr])
# AFTER (fixed):
aBM = np.einsum('pl,pml->pm', a_arr, BMA_inv[sv_arr])
```

**4. Working precision**

All inputs to `forward()` are now explicitly cast to `float64` at entry, preventing
silent float32 degradation when called from PyTorch training loops:
```python
rigidities = rigidities.double()
if rest_lengths is not None:
    rest_lengths = rest_lengths.double()
```

### 20×20 benchmark: all six methods vs simulation (DF and TF meshes)

Comparison of six MF variants against a direct spring-network simulation (KUBC with
unit springs, rest lengths = equilibrium) on 20×20 foam meshes, 10 random trials per η,
η ∈ {0.0, 0.05, …, 0.50}. Results normalized by each method's own η=0 value.

**Mesh types:**
- **DF** (distort-first): Delaunay triangulation after vertex distortion — well-shaped triangles
- **TF** (triangulate-first): fixed topology, positions distorted after — can produce poor triangles at high η

**Methods compared:** Std, AW (area-weighted), Std+edge, AW+edge, Std+full (edge+angle), AW+full

**E/E₀ at η=0.5 (DF mesh):**

| Method | E/E₀ | vs. Simulation (0.874) |
|--------|-------|------------------------|
| Simulation | 0.874 | — |
| Std | 0.011 | −98% |
| AW | 0.272 | −69% |
| Std+edge | 0.023 | −97% |
| AW+edge | 1.590 | diverges upward |
| Std+full | 0.907 | unstable (diverged at η=0.4–0.45) |
| AW+full | 164 | wildly divergent |

**ν at η=0.5 (DF mesh):**

| Method | ν | vs. Simulation (+0.281) |
|--------|---|------------------------|
| Simulation | +0.281 | — |
| Std | −0.094 | wrong sign |
| AW | +0.040 | low |
| Std+edge | +0.084 | low |
| AW+edge | −0.607 | wrong sign, large |
| Std+full | −0.532 | unstable |
| AW+full | +0.218 | unstable |

**Key finding:** Even for the DF mesh with uniform k=1 springs (geometric disorder only),
all MF methods dramatically underestimate E at high η. The simulation shows only ~13% drop
in E/E₀ from η=0 to η=0.5; MF methods predict 70–99% drops or divergence.
This contradicts the earlier assumption (recorded above) that "geometric disorder alone:
deformed meshes with uniform k are well-captured." That finding held only for small η (≤0.20).

Note: the reference geometry has **no frustration** — interior vertex angle sums equal 2π
to machine precision at all η (confirmed numerically). The failures below are not
caused by an inconsistent rest state.

**Why the MF fails for E:**

The MF framework assigns the same macroscopic strain Δg to every triangle. In a
disordered network the real strain is spatially inhomogeneous: stiff regions deform
less, soft ones more. The Woodbury non-affine correction W accounts for this
perturbatively around the mean field, but for large disorder the perturbation is large
and the correction overshoots — driving C_eff = ⟨A(I+W)⟩ toward zero. No backbone
mechanism exists in the single-site MF to prevent this.

**Why KKT edge constraints do not rescue E:**

The edge-KKT correction projects W onto the subspace of compatible metric fields.
When W_0 (the unconstrained MF solution) is already strongly negative (overshooting),
the projection can push W further in the wrong direction. The constraints are physically
correct; the problem is that they are applied on top of an already-wrong mean field.

**Why angle constraints (Std+full, AW+full) diverge:**

The rest configuration has angle sums exactly 2π — there is no inconsistency. The
divergence is a numerical/geometric issue: as η grows, some triangles become elongated
(minimum angles reach ~14° at η=0.35 for DF; smaller for TF). For a near-degenerate
triangle, the angle gradient ∂θ/∂g is large in one direction, making the corresponding
angle-constraint row nearly a linear combination of the edge-constraint rows already in
the system. This near-redundancy drives eigenvalues of the Gram matrix G toward zero,
causing the KKT solve to produce large multipliers Λ and a large (positive) correction
to C_eff — hence the upward divergence rather than collapse to zero.

**Plot:** `benchmarking/new/all6_comparison.png`

### Remaining fixes (to be explored)

1. **Periodic boundary condition simulation** — better ground truth, matches the analytical infinite-medium assumption.
2. **Cluster self-consistent method** — embed clusters of neighbouring triangles rather than single ones.
3. **Differential effective medium** — add heterogeneity incrementally.
4. **Reduce rigidity contrast** — use smaller `a` parameter so the mean-field remains valid.
5. **Accept the limitation** — use the pipeline for designs with moderate rigidity contrast (η ≤ 0.20), where it is accurate.
6. **GNN / ML forward model (Phase 4)** — learns the microstructure-to-property map directly from simulation data, bypassing the MF approximation entirely.
