# Forward-solver investigation — session summary (June 2026)

Goal: understand why the D2C mean-field (MF) forward solver disagrees with direct simulation,
and find a principled fix. All work is in `breakdown/`; see `THEORY_NOTES.md` for the theory
and `TODO.md` for the roadmap. Companion derivation: `derivation_edge_compatibility.pdf`.

## Setup
Periodic "triangulate-first-then-deform" meshes; per-triangle **non-affine metric change**
`δg` (the framework strain `Δg = FᵀF − I`, nonlinear, ≈2ε), lab-frame, compared against a
direct PBC spring-network simulation. Quantities: per-triangle `δg` (principal values +
axis angle) and homogenised `C_eff → ν, E`.

## Findings (in order)
1. **Definition pitfalls fixed.** `δg` must be the lab-frame per-triangle strain
   `εₛ = sym(Fₛ−I)` / metric change `FₛᵀFₛ−I` (from the deformation gradient), NOT the
   edge-basis Gram matrix (which mixes strain with edge orientation and gave a spurious
   "spread" even for a perfect crystal). And it is the **non-affine** part `gₛ − g`,
   referenced to the single global affine `Δg`.
2. **Solver `W` convention.** The strain concentration is the 3×3 `W₃` (`W₃[loc,k]=W[3loc+k]`),
   not the 4×4 `W_mat` (which is for the elastic tensor). With `W₃`, edge-KKT is verified
   edge-compatible (residual→0) and `⟨δg_MF⟩≈0`.
3. **MF is predictive at low η, breaks at high η** (per-triangle corr 0.9→0.3; ν/E collapse).
4. **Edge-KKT correct but marginal; angle-KKT** (vertex/St-Venant) implemented — found a
   shear-factor (`/2`) convention bug in the solver's `_build_vertex_angle_constraints`;
   even fixed it only marginally helps and diverges at high η.
5. **The overshoot.** MF over-predicts `δg` by ~1.4× (stable across η, survives edge+angle
   KKT). Traced to the **single-site uniform-stress (Reuss-type) mean field**: the solver's
   `W` equals `−A⁻¹δA·Δg` (corr 0.98), i.e. each triangle carries the mean stress, embedded
   in the *average* medium.
6. **Where the theory is approximate** (`THEORY_NOTES.md`): the MF minimises energy over
   per-triangle metric fields constrained only by edge agreement + zero-mean — a space
   `≈n` DOF **larger** than physically realisable (node displacements). The extra DOF are
   **incompatible** fields (nonzero discrete Gaussian curvature / disclinations). Minimising
   over too-large a set ⇒ over-compliant ⇒ the overshoot.
7. **Cluster (local-patch) response — the fix.** Relax a radius-d node patch (boundary
   affine, real spring physics); read the central triangle's `δg`. Works in node space ⇒
   automatically compatible, and uses the *actual* neighbours. Recovers the simulation's
   per-triangle `δg`:
   - geometric disorder: **d=1** gives corr 0.97–0.98 at *all* η (incl. η=0.5, where MF→−0.07);
   - rigidity contrast: fixes the overshoot immediately, direction converges with radius
     (longer correlation length → larger d).
8. **Homogenised properties (task 2).** Cluster `C_eff → ν, E` tracks the simulation where
   single-site MF is catastrophic (E→0, ν→−0.9 and worse). Confirmed across η, signed VD
   rigidity contrast, and combined disorder.
9. **Formalisation.** It's a real-space truncation of the lattice Green's function
   `W = L·K⁻¹·∂f/∂Δg`, with error `~e^{−d/ξ}`; cluster radius ≈ disorder correlation length
   `ξ` (short for geometry, long for rigidity contrast). The MF is the `ξ→0`, average-medium
   limit. Compatibility (node space) and Green's-function locality are dual views.
10. **Extracting `W` from the cluster.** Solve the patch for the 3 macro modes →
    `W(s) = [δg responses]·[Δg modes]⁻¹` — a loading-independent, per-triangle, non-iterative,
    differentiable response; a drop-in replacement for the MF `W` in the homogenisation /
    inverse-design / GNN pipeline.

## Geometric reading (defect theory)
Compatibility ⟺ zero discrete Gaussian curvature (vertex angle-deficit) = no disclinations =
linearised St-Venant `inc(δg)=0`. Torsion (dislocations) needs a frame lift / is not an
independent 2D condition (Gauss only; Codazzi trivial for 2D-in-2D). The MF generates a
spurious disclination density; the cluster solution is defect-free by construction.

## Key scripts (breakdown/)
`pbc_dg_analysis.py` (core sim+MF+KKT, --regen), `compat_projection.py`,
`analyze_overshoot.py`, `test_angle_response.py`, `test_renormalize.py`,
`test_cluster_response.py`, `test_cluster_rigidity.py`, `test_cluster_Ceff*.py`,
`test_cluster_combined.py`, `test_cluster_VD.py`.

## Open / next
Differentiable batched cluster solver (task 3); measure ξ directly; metric-space
full-compatibility (curvature-operator) condition on `W` as the "one global solve"
alternative to the cluster (see curvature/torsion operators).
