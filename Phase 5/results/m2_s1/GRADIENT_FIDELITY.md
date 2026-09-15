# A0.2 — does the surrogate's GRADIENT point the way the solver's does?

**Producer:** `Phase 5/verifications/m2_gradient_fidelity.py`
**Data:** `gradient_fidelity_full.json` (this directory; the run log is gitignored by repo policy)
**Model:** `checkpoint_v3_res_bravais_w10_h1_L5_ns48_h64_e400_star_ms.pt`, frozen (per-triangle MAE/σ 0.0925)
**Population:** `dataset_fresh_s4321.npz` — freshly generated, UNSEEN meshes
**Run:** 2026-09-15, float64, `OMP_NUM_THREADS=1`, seed 0, commit `5625254`+

---

## What and why

Every M2 measurement before this one scores the surrogate's **value**. None scored the quantity a
designer consumes. The surrogate exists because geometry is differentiable through it while the exact
solver's positions are *"fixed inputs, set at construction, not differentiated"* (`ElasticSolver`;
hence SPSA, `CLAUDE.md` §3, FD #13). So the operative question is not "is the prediction accurate" but
**"does descending the surrogate move a design the way descending the truth would"**.

## Method

Two losses — the **same** weighted ν(θ)/E(θ) residual the designer minimises (`inverse_design._loss`'s
`nu_theta` + `E_theta` arithmetic, reused, not reimplemented) — evaluated at the same point against the
same target, then `cos(∇L_solver, ∇L_gnn)`:

```
L_solver(x) = loss( C6 from the exact solver at x )
L_gnn(x)    = loss( C6 from the frozen surrogate at x )
```

They are gradients of *different functions*: that is the point. A surrogate can be biased and still be
a perfect descent direction, and accurate while pointing sideways.

**Target** = a design specification, so anchored on the SOLVER's response, never the model's:
`ν_target = ν_solver − 0.2`, `E_target = 0.8·E_solver`. Both channels displaced so both contribute.

**References differ by channel** — the asymmetry the project rests on:
- `∂L/∂k` — the solver's own **adjoint**. Exact, one backward.
- `∂L/∂pts` — the solver has none, so **full 2N central finite differences** (the exact cosine, not a
  random-direction estimator), affordable only at `n_node ≤ 64`.

**Health:** every perturbed geometry checked for triangle INVERSION via the signed area and against
`check_mesh_preconditions`. **0 of 6282 probes rejected**; rejections would have been counted, not
dropped.

## Result — the pre-registered thresholds are MET

| channel | n | median cos | IQR | cos > 0 | cos > 0.9 | \|∇GNN\|/\|∇solver\| |
|---|---|---|---|---|---|---|
| `∂L/∂k` | 300 | **0.9499** | [0.794, 0.986] | 90.7 % | 65.7 % | 1.014 |
| `∂L/∂pts` | 100 | **0.9148** | [0.623, 0.965] | 89.0 % | 54.0 % | 0.937 |

Rule was *median > 0.9 on `k` and > 0.7 on positions*. Both met; magnitudes essentially unbiased.

## The finding that matters more than the median — the tail

The cosine collapses with `max|W|`, monotonically, on **both** channels (binned medians):

| max\|W\| | `∂L/∂k` | n | `∂L/∂pts` | n |
|---|---|---|---|---|
| < 1 | **0.9986** | 30 | **0.9868** | 5 |
| 1–3 | 0.9769 | 109 | 0.9472 | 46 |
| 3–10 | 0.9256 | 112 | 0.8985 | 37 |
| 10–100 | 0.7395 | 42 | **0.1782** | 11 |
| > 100 | **0.0424** | 7 | −0.1617 | 1 |

The gradient is near-perfect in the bulk and **fails precisely in the near-mechanism tail** — the same
place the VALUE error concentrates (error rises 7× with `max|W|`, §7 of
`M2_RESIDUAL_AND_CONSTRAINTS.md`). Coherent rather than surprising: where the response is near-singular,
so is its derivative.

**Actionable consequence:** a trust-region designer should **gate on `max|W|`**, which the surrogate
computes cheaply, and hand those networks to the exact solver. That is a design rule derived from
measurement, not a tuning knob.

## Per family

| family | n (k) | med cos | | family | n (pts) | med cos |
|---|---|---|---|---|---|---|
| `disordered` | 125 | 0.9620 | | `disordered` | 45 | 0.9273 |
| `cells` | 71 | 0.9362 | | `cells` | 31 | 0.8985 |
| `random` | 42 | 0.9201 | | `bravais` | 10 | 0.8033 |
| `longrange` | 36 | 0.9410 | | `longrange` | 9 | 0.9556 |
| `bravais` | 13 | 0.9777 | | `basis` | 2 | 0.9184 |

⚠ **Do not read the thin rows.** `auxetic` (n=5), `basis` (n=1), `tiling` (n=1) on `k`, and `random`
(n=2), `tiling` (n=1) on positions carry no weight. Only the four-figure rows above are usable.

## What this run also had to fix first — and it was not the model

The FIRST pilot gave median cos **0.12** (`k`) and **−0.008** (`pts`), which reads as "the surrogate's
gradient is useless". It was the **reference** that was broken, not the model. `evaluate_v2.geo_of`
rebuilt `edge_vecs` as `bond_R[tri_bond]`, but `bond_R` is oriented by the GLOBAL bond list while the
builder orients by each TRIANGLE's corner order — **48 of 96 rows differ in sign** on a 32-triangle
mesh. `q_e = Δx Δxᵀ` is sign-EVEN, so `A(s)`, the GNN's inputs and the sim never noticed; the CURVATURE
constraint is not, so a rebuilt solver enforced angles belonging to no real mesh.

Measured, fresh solve vs the stored label: **median 1.5e-1, max 7.1e+02, wrong on 93 %** — against the
GNN's own 4.5e-2. The reference was three times worse than the model it was meant to judge.
Fixed (`5625254`); label reproduction → **1.3e-15**, `bravais` exactly 0. **Guarded** since, by
`mesh_build.check_edge_vecs` called from `DesignProblem.from_geo`.

**The methodological point:** the first gate here was "does `geo_at(pts0)` equal `geo_of`" — it passed
at 4e-16 while the whole reference was wrong, because it compared a reconstruction against a
reconstruction sharing the same bug. It was replaced by **label reproduction against a number produced
by the code that built the data**, which is the only check that could have caught it.

## Limitations

- **Self-loops excluded: 772 of 23 296** (3.3 %). A bond from a node to its own periodic image has
  `bond_u == bond_v`, so no orientation is recoverable from a schema that stores no `edge_vecs`. They
  are confined to 2–5 node cells (`cells`, `anchor`). Decided 2026-09-15: they are to be removed
  entirely by **supercell re-representation** (A1) — `C_eff` is invariant across supercells to 5e-16,
  so nothing is lost. The `M2_V2_PLAN §3.1c` claim that they vanish at N≥3 is **FALSE**: measured
  67 % at N=3, 33 % at N=4, 25 % at N=5, 8 % at N=6, 0 % only at N≥7.
- **Position channel is small-mesh only** (`n_node ≤ 64`), because the exact reference costs 2N solver
  rebuilds. Large-mesh position fidelity is untested.
- **One target form** (`ν−0.2`, `0.8E`). A different displacement could weight the two channels
  differently; not swept.
- **A frozen single checkpoint.** Nothing here says the tail failure is intrinsic rather than specific
  to this model.
- **Cosine, not step quality.** A good direction with a bad Hessian can still converge slowly. A0.3
  (descent through the frozen GNN vs M1 at matched budget) is the test that closes this, and it probes
  exactly the tail the table above says is weak.
