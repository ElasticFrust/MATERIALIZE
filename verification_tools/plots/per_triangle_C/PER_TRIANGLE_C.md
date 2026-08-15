# Per-triangle C(s): solver contraction vs the independent oracle — visualised

**What.** `test_forward_solver` [8] reduces the per-triangle comparison to one number per mesh. That
tells you whether the gate passes; it does not show where the two differ, how the difference is
distributed, or how far it sits from what a real defect looks like. This is that picture.

**Method.** `verification_tools/per_triangle_C_comparison.py`, over the same five mesh cases [8]
uses. Solver side = `_compute_actual_elastic_tensor` fed the SIM's measured W (so the comparison
isolates the *contraction*, as in [7]). Oracle side = `physical_homog.energy_C_per_triangle`, which
never forms W and never does the 4-index contraction. Commit `4c3c584`+.

---

> **What this residual is NOT.** It is *not* solver-vs-sim disagreement.
>
> All comparisons here compute the SAME object two ways: C(s) = (1+W)ᵀA(s)(1+W), via the SAME
> `_compute_actual_elastic_tensor`, with the SAME A(s). They differ only in **where W comes from**,
> and are scored against the same reference — `energy_C_per_triangle`, the Hessian ∂²U_s/∂g² of
> triangle s's own relaxed energy. Measured N=10, η=0.30, VD a=+5:
>
> | source of W | per-triangle | bulk |
> |---|---|---|
> | sim's relaxed **u** → per-triangle F_s → Δg = FᵀF − I (**exact** strain) — *this is what [7]/[8] use* | 1.05e-02 | 5.37e-03 |
> | the same **u**, same F_s, but Δg = (F−I)+(F−I)ᵀ (**linearised** strain) | **5.47e-13** | **2.33e-13** |
> | the **solver's own** W — the intrinsic constrained solve over δg. **No displacement field is involved at all** | **7.54e-10** | 5.45e-12 |
>
> Rows 1–2 share a displacement field; **row 3 does not have one** — the solver works in metric
> space (W is (N,9), per-triangle, no nodal DOF). The two strain measures in rows 1–2 differ by
> 5.84e-03 at DELTA=1e-3, which accounts for row 1 exactly and scales linearly with DELTA. So the
> ~1e-2 figures below measure a strain-measure mismatch at finite probe amplitude — a property of
> the isolation construction, chosen to make the check sensitive to the CONTRACTION. Row 3 is the
> genuine cross-formulation check (metric-space constrained solve vs nodal relaxation) and is ~8
> orders tighter. See audit A-15.


## 1. Numbers

| case | triangles | per-triangle rel. error | **with the A-0 shear defect** | ratio |
|---|---|---|---|---|
| crystal η=0 (W≡0) | 128 | 2.22e-13 | **4.44e-13** | **1.0** |
| η=0.20 | 128 | 2.13e-03 | 2.56e-01 | 120× |
| η=0.20, VD a=+5 | 128 | 2.14e-03 | 1.12e+00 | 523× |
| η=0.35 | 128 | 3.95e-03 | 3.49e-01 | 88× |
| η=0.30, VD a=+10 | 288 | 7.74e-03 | 2.40e+00 | 310× |

Gate tolerance is 2e-2.

**The first row is the important one.** With W ≡ 0 the defective contraction and the correct one are
*indistinguishable* — 4.44e-13 vs 2.22e-13, both at round-off. This is the "crystal gate is blind"
principle (`CLAUDE.md` §3) reproduced directly: no crystal-anchored check, at any precision, could
ever have caught A-0. On meshes with W ≠ 0 the same defect stands 88–523× above the correct
residual and 1–2 orders above the gate tolerance.

## 2. Figures

- **`error_by_case.png`** — the three curves per case on one panel: per-triangle (correct), bulk
  (correct, = test [7]), and per-triangle with the A-0 defect, against the gate tolerance. Shows
  both that per-triangle is 2–5× the bulk error (fluctuations do not average out) and that a real
  defect is nowhere near the floor.
- **`per_triangle_residual.png`** — component-wise RESIDUAL against the tolerance band, all cases
  overlaid. *(A plain solver-vs-oracle scatter was produced first and discarded: at ~2e-3 relative
  error every point lies on the diagonal and the figure carries no information. The residual form
  is what shows structure — a slight positive bias on the normal components C_xxxx / C_yyyy growing
  with the component's value, and tighter shear-coupling entries.)*
- **`residual_field.png`** — the per-triangle residual next to ‖W(s)‖ on a disordered cell.

## 3. Where the residual comes from

It is O(DELTA), from the strain measure: `tri_metric_change` is geometrically exact (Δg = FᵀF − I)
while the energy uses the linearised bond extension. Verified by *scaling*, not asserted — the
residual falls linearly with DELTA:

| DELTA | 1e-3 | 3e-4 | 1e-4 | 3e-5 |
|---|---|---|---|---|
| per-triangle rel. error | 7.74e-3 | 2.32e-3 | 7.73e-4 | 2.32e-4 |

**Honest caveat on the spatial picture:** corr(|ΔC|, ‖W‖) = **+0.39** on the disordered cell. The
residual is *related* to the non-affine response but is **not** simply proportional to it, and the
two field maps do not look alike by eye. An earlier draft of this figure claimed the residual
"tracks ‖W‖"; that was overstated and has been corrected. What is established is the DELTA scaling
above, not a pointwise proportionality to ‖W‖.

## 4. Limitations

- Five meshes, two sizes (128 and 288 triangles); no size convergence study.
- Periodic cells only — the open-boundary per-triangle case is untouched (audit **A-8**).
- The A-0 curve is a *local* reimplementation of the contraction with the ½ dropped, asserted to
  reproduce the core bit-for-bit when correct. The protected core was never modified to produce
  these figures.
- The comparison validates the CONTRACTION given the energy-partition convention (each bond counted
  fully in both its triangles, matching `A(s)`). The partition itself is a convention, not something
  this check tests.
- Everything here is at the tensor level, so it is unaffected by audit **A-10** (the design path
  reports directional ν,E while the oracle averages) — which remains open.
