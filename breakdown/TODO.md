# breakdown/ — forward-solver investigation TODO

## Done
- [x] PBC sim vs MF per-triangle δg on periodic triangulate-first meshes; lab-frame
      metric-change δg = FᵀF − I (`pbc_dg_analysis.py`).
- [x] MF predictive at low η, breaks at high η; quantified corr / |Δθ| / overshoot.
- [x] Edge-KKT correct (drives edge residual→0) but marginal gain; ~1.4 overshoot persists.
- [x] Compatibility-projection test: overshoot is NOT a compatibility artefact (`compat_projection.py`).
- [x] Overshoot traced to the single-site **uniform-stress** mean field (`analyze_overshoot.py`);
      source explained in `THEORY_NOTES.md`.
- [x] Angle/vertex (full St-Venant) KKT implemented + tested; found the shear-factor (/2)
      convention bug; marginal gain, diverges at high η (`test_angle_response.py`).
- [x] **Cluster (local-patch) response** recovers the simulation's per-triangle δg —
      d=1 for geometric disorder, larger radius for rigidity contrast
      (`test_cluster_response.py`, `test_cluster_rigidity.py`).
- [x] **Mean constraint isolated as the metric-solve failure**: the plain ⟨δg⟩=0 is wrong;
      the correct normalisation is the **S_triangle-weighted** mean (implied by compatibility,
      `M_S·Π ≡ 0`), and the curvature/incompatibility constraint `C δg=0` was missing
      (`test_mean_isolation.py`, `test_curvature_operator.py`).
- [x] **Intrinsic (configuration-free) metric solve** `[edge ; curvature ; area-weighted mean]`
      reproduces the simulation's per-triangle δg (corr ≈1.0, η≤0.4) and homogenised C_eff/ν/E
      to 3–4 digits, incl. signed VD rigidity contrast; full derivation in G&B notation
      (`test_intrinsic_metric.py`, `test_intrinsic_VD.py`, `INTRINSIC_METRIC_SOLVE.md`).
- [x] **Verified G&B `(A−B)` Woodbury + area + edge + angle does NOT reproduce sim**
      (corr 0.93→collapse at η≥0.4); the working route is the intrinsic block-diagonal solve
      (`verify_gb_formulation.py`).
- [x] **Intrinsic solver shipped in `forward_solver_torch.py`** — `forward(method='intrinsic')`
      is the new default (explicit-multiplier, loading-independent W-saddle; block-diagonal A
      + edge + curvature + `S_s`-mean), toggles `use_kkt`/`use_angle_kkt`/`area_weighted` ON
      by default; old `(A−B)` reachable via `method='woodbury'`; verified to reproduce sim
      ν/E across η (`verify_intrinsic_solver.py`). Woodbury regression tests pinned to
      `method='woodbury'`.

## Next
- [x] **Sub-choice 2 (area-weighted χ-elimination) SHIPPED + differentiable**
      `δA(s) = A_s − [A]·S_s/[S_s]`, coupling `(Bδg)(s)=(S_s/[S_s])Σ_{s'}δA(s')δg(s')`, on the
      SYMMETRIC metric Hessian `A3=Σ_e(k/4l²)q qᵀ`. Implemented as `_woodbury_solve_aw` (pure
      torch) and made the engine of `method='intrinsic'` for ≤`INTRINSIC_DENSE_MAX` triangles
      (sparse saddle fallback above that). Verified: reproduces sim ν/E across η (geometric +
      VD), gradients correct (autograd vs finite-diff agree), `method='woodbury'` backward-
      compatible (`verify_intrinsic_solver.py`, `verify_subchoice2*.py`).

## In progress
- [ ] **(2) Batched cluster forward solver → homogenised C_eff / ν / E**, compared to the
      PBC simulation across η and rigidity contrast. Validate ν(η=0)=1/3.
      (`test_cluster_Ceff.py`)

## Next
- [x] **Code cleanup / readability pass on `Phase 2/forward_solver_torch.py`** (conservative,
      forward-solver-only; −141 lines, public/semi-public API unchanged so breakdown/
      benchmarking/Phase 3 still import every name). Verification-gated, each step diffed against
      a frozen single-thread baseline (`verify_intrinsic_solver.py`, `verify_gb_formulation.py`,
      autograd-vs-FD gradcheck — all identical; ν/E match sim as before, grads match FD to ~1e-7):
      - Unified the per-triangle metric Hessian `A(s)=Σ_e(k_e/4ℓ_e²)q qᵀ` build in `forward()`
        (was recomputed in both the dense and the fallback branch).
      - Collapsed `_woodbury_solve`'s weighted/unweighted branches: unweighted **is** the
        area-weighted solve with `w_n=1/N` (the constant commutes through the solve) — verified
        bit-identical W, base + KKT branch.
      - Deleted the unused `_woodbury_kkt_sparse_aw` (benchmarking keeps its own local copy).
      - Rewrote the stale module docstring (now describes the `intrinsic` default + `woodbury`
        legacy methods) and documented the `M` (asymmetric mixed-Voigt) vs `A3` (symmetric
        Hessian) distinction at `_batch_to_9x9` to fence off that bug class.
      - *Deliberately NOT merged:* `_woodbury_kkt_sparse_combined`'s weighted/unweighted branches
        (the `N·(I−S)` vs `(I−S)` placement conditions the rank-3 Woodbury differently → ~1e-6
        drift, a load-bearing numerical difference) and `_woodbury_solve` vs `_woodbury_solve_aw`
        (9×9 asymmetric `M`-space vs 3×3 symmetric metric space — conflating them caused the
        ν=−0.001 bug this session).

- [ ] **Remove unused / deprecated methods & code versions (repo-wide).** Now that the
      `intrinsic` solve is the verified default, prune dead/superseded code so only one
      canonical version of each routine remains:
      - In `Phase 2/forward_solver_torch.py`: `_woodbury_kkt_sparse` is never reached by
        `forward()` (the woodbury+KKT path uses `_woodbury_kkt_sparse_combined`); it survives
        only because `benchmarking/` imports it. Decide: keep as a documented public helper or
        retire it (and the `method='woodbury'` legacy path) once benchmarking is migrated.
      - Stale duplicates elsewhere: `Phase 2/forward_solver_torch.py.bckp`,
        `benchmarking/` local re-defs (`_woodbury_kkt_sparse_aw`, `run_all_models_comp.py`),
        `Phase 3/area_fix_comparison.py::_woodbury_solve_area`, and any `*_vectorized` /
        pre-refactor variants in `Disc_2_Cont_optimized` referenced by the Phase 2 tests.
      - Audit `breakdown/` for superseded scripts (old MF/KKT experiments now subsumed by the
        intrinsic solve) and mark/retire them; keep the verification suite.
      - Each removal must be import-checked across `breakdown/`, `benchmarking/`, `Phase 3/4`
        before deleting (the helpers are a de-facto cross-repo API).

- [ ] **Second-order (O(δ²)) term of the intrinsic metric solve** — the area-weighted
      normalisation `Σ_s S_s δg(s)=0` and the identity `M_S·Π ≡ 0` are only the **first
      order** of the exact area law `Σ_s S_s det(F_s) = det(F) Σ_s S_s`. This is why the
      intrinsic solve matches the simulation to η≈0.4 but slips at η=0.5 (where the `M_S·Π`
      residual reaches ~3e-3 and ⟨tr δg⟩_S ~ δ²). Add the next order — i.e. carry the
      nonlinear `det(F_s)` constraint (the quadratic-in-δg correction to the normalisation /
      compatibility) — to make the metric solve exact at large η without resorting to the
      cluster. Verify against sim ν/E at η=0.5 and across VD contrasts.
- [ ] **(3) Differentiable cluster solver** — each triangle's response is a small local
      linear solve (3 RHS for the 3 macro modes); make it autograd-friendly so Phase 3
      inverse design and Phase 4 GNN labelling can use it. No new hyperparameters beyond
      the cluster radius.

- [ ] **Implement and verify varying reference metric and residual stresses.** All tests so
      far drive the solver with `rest_lengths` = the actual (current-configuration) edge
      length, i.e. zero intrinsic residual stress (bare = actual reference metric). Extend
      to per-edge `rest_lengths` != actual edge lengths (a non-trivial reference metric
      g0(s) varying per triangle/bond), which induces residual (pre-)stress in the
      unstrained network. Verify the forward solver's per-triangle response W3(s) and
      homogenised ν/E against the PBC simulation for: (a) a spatially uniform rest-length
      mismatch (uniform residual stress/pressure), and (b) a spatially varying reference
      metric (e.g. localized inclusions or a smooth field of rest-length mismatches),
      including combinations with rigidity disorder.

- [ ] **Periodic complex structures.** Extend the sim-vs-solver verification beyond
      triangulated lattices/Poisson-Delaunay to other periodic network topologies relevant
      to metamaterial design (e.g. honeycomb / kagome-derived triangulations, multi-motif
      unit cells, networks with engineered hierarchical or composite unit cells). Confirm
      the periodic bookkeeping (`kkt_from_tri_bond`, `_build_intrinsic_constraints`,
      `INTRINSIC_DENSE_MAX` fallback) generalises correctly and that per-triangle response
      and homogenised ν/E still match the PBC simulation.

- [x] **Final test: 10 networks, varied structure + rigidity distributions** (not eta-disorder
      sweep, not tanh-VD) — for each, own (N, eta, seed) periodic lattice + a distinct per-bond
      rigidity distribution (uniform random, lognormal, bimodal, bimodal-checkerboard clusters,
      x-graded, power-law, orientation-dependent k(θ), sparse soft/stiff inclusions, smooth
      random field), comparing PBC sim vs `forward(method='intrinsic')` (`verify_solver_final.py`,
      `plots/dg_solver_final_10networks.{png,npz}`). 8/10 cases agree to a few % in ν and E.
      Two outliers with **spatially-correlated / long-range-structured** rigidity:
      - bimodal checkerboard clusters: Δν≈+0.045, ΔE/E≈+14%
      - orientation-dependent k(θ) (anisotropic, near-regular lattice): Δν≈-0.106, ΔE/E≈-47%
      Both are patterns whose correlation length exceeds the per-triangle/edge/curvature
      locality of the intrinsic solve — likely the same regime flagged in "cluster-radius
      selection" below for strong rigidity contrast; needs a larger-radius/cluster correction
      for spatially-correlated or anisotropic rigidity fields.

- [x] **Soft-region per-triangle response test**: a small cluster (6 bonds around one vertex,
      k=0.05 vs k=1 elsewhere) embedded in a perfect lattice (eta=0) and a disordered lattice
      (eta=0.15); compared the per-triangle response W3(s) (delta_g(s)=W3(s)@Delta_g, all 3
      macro modes) between PBC sim and `forward(method='intrinsic')`
      (`verify_soft_region.py`, `plots/dg_soft_region_per_triangle.png`). Strong agreement:
      corr=0.994 (perfect lattice), corr=0.998 (disordered); the soft-region triangles'
      distinct (larger) response is captured correctly by the solver in both cases.

- [x] **50x50, two soft circles, three network structures**: a fixed pair of circular
      soft-bond regions (k=0.05 vs 1, r=0.05*Lx, centred in the box) embedded in (a) a
      perfect periodic triangular lattice (eta=0), (b) a disordered one (eta=0.3), and (c) a
      periodic Delaunay triangulation of N*N=2500 Poisson points (same density, built via
      3x3-tile periodic Delaunay). Compared per-triangle response W3(s) sim vs
      `forward(method='intrinsic')` (5000 triangles -> exercises the sparse-saddle
      `INTRINSIC_DENSE_MAX` fallback) (`verify_soft_circles_50x50.py`,
      `plots/dg_soft_circles_50x50.png`). corr = 1.0000 (perfect lattice), 1.0000
      (eta=0.3), 0.9987 (Poisson-Delaunay) -- all three structures agree closely, including
      the irregular non-lattice triangulation.
      (Fixed a latent indexing bug in the soft-region triangle lookup, shared with
      `verify_soft_region.py`: `tri_bond` values were used directly as triangle indices.)
      Extended (`verify_soft_circles_50x50_directions.py`,
      `plots/dg_soft_circles_50x50_directions.png`) with the other two response directions
      (dilation resp. to e_yy, shear resp. to e_xy), spatial maps + value distributions.
      All corr >= 0.999 across all three networks and all three macro directions.

- [ ] **Standardize visualizations.** The verification plots (`verify_solver_sweep.py`,
      `verify_solver_final.py`, `verify_soft_region.py`, `verify_soft_circles_50x50*.py`,
      etc.) have each grown their own ad-hoc layout, color scales, naming, and figure sizes.
      Factor out a shared plotting module (e.g. `breakdown/plot_utils.py`) for the recurring
      panel types -- sim-vs-solver parity scatter, per-triangle spatial maps (shared
      colorbar/colormap conventions), response-distribution histograms, k/edge-length
      diagnostic panels -- so new sim-vs-solver tests reuse one consistent visual language
      instead of copy-pasted plotting code.

## Open questions / later
- [ ] Cluster-radius selection: tie radius to a measured local correlation length
      (cheap for geometric, ~4–6 for strong rigidity contrast); or adaptive per-triangle.
- [ ] Embedding choice: affine-clamped patch (used here; converges from below) vs
      self-consistent / effective-medium boundary (may converge faster at small radius).
- [ ] Overlapping-patch efficiency (avoid recomputing shared neighbourhoods); batch the
      local solves.
- [ ] Push the `/2` shear-convention fix into the Phase 2 solver's
      `_build_vertex_angle_constraints` (if the angle-KKT path is kept).
- [ ] Re-run with full geometric+rigidity disorder combined; confirm radius ≈ correlation length.
