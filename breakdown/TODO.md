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

## In progress
- [ ] **(2) Batched cluster forward solver → homogenised C_eff / ν / E**, compared to the
      PBC simulation across η and rigidity contrast. Validate ν(η=0)=1/3.
      (`test_cluster_Ceff.py`)

## Next
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
