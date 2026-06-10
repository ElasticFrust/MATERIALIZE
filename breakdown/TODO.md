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
