# M2 — the residual head, and putting the CONSTRAINTS into the model

**What.** Two architectural changes, in order: (1) predict the *deviation* from the analytic
per-triangle tensor rather than the tensor itself; (2) give the model the constraint operators the
solver actually imposes, instead of making it infer their effect from examples.

**Producer.** `Phase 5/m2/model_v3.py`, `Phase 5/m2/train_v3.py`.
Gates: `Phase 5/verifications/test_m2_head_v3.py`, `Phase 5/verifications/test_m2_constraints.py`.
float64, `OMP_NUM_THREADS=4`, seed 0, `data/dataset_v2_s0.npz` (41 431 samples).

---

## 1. The diagnosis these changes answer

Measured on the previous checkpoint (1300-network `bravais` holdout):

**Error is controlled by k-CONTRAST — 17×, monotonic:**

| contrast | n | MAE/σ | median ‖W‖ |
|---|---|---|---|
| 0 – 3 | 73 | **0.0240** | 0.00 |
| 3 – 10 | 275 | 0.0716 | 0.00 |
| 10 – 10² | 736 | 0.2106 | 2.23 |
| 10² – 10⁴ | 16 | 0.2250 | 2.28 |
| 10⁴ – 10⁹ | 200 | **0.3994** | 10.43 |

`THEORY_NOTES.md` independently measured the required cluster radius as ~1 for geometric disorder but
**4–6 at 100× contrast** ("soft channels / stiff backbones have a longer correlation length"). Our
data reaches contrast 10⁶ and the model has a fixed reach for every sample: **the physics has a
contrast-dependent screening length, a fixed-depth GNN has one fixed reach.**

**The k-field's own correlation length works the OPPOSITE way** — `xi_short` (near-white) is worst
at 0.2816, `xi_long` best at 0.1582, `xi_box` 0.1779, `iid` 0.1752. A long-wavelength stiffness
field is locally smooth and therefore *easy*; sharp local heterogeneity is what hurts. This confirms
`M2_LOCALITY.md`'s note that the long tail under correlated `k` is **inherited from the input**, not
a causal screening length.

**And it did not get the free part free**: where `W = 0` the answer *is* `A(s)`, a closed form in the
triangle's own three edges, and the model scored **0.0538** instead of 0.

## 2. Change 1 — the residual head

```
G = (I + X) · G_an · (I + X)ᵀ ,      G_an = diag(k_e / 16 ℓ_e²)
```

`G_an` is computed inside `forward()` from inputs already present (ℓ² is the trace of each carrier,
`Q[:,0,:] + Q[:,2,:]`; `k` from `tri_scalars[:, :3]`). The network predicts only the dimensionless
invariant 3×3 `X`, and **the readout's last layer is zero-initialised**, so training *starts* at
`X = 0`, i.e. exactly at `A(s)`.

Why it matters beyond the ~5% it fixes directly: `C` is dominated by `A` — predicting `A` alone
scores **1.1747** against the label σ (worse than the global mean, 0.9866), while the trained model
scores 0.2000 — so the loss was dominated by the easy term while 91% of the error sat in the
correction. The head makes the easy term exact and leaves the network only the hard one.

### Result

Leave-one-family-out on `bravais`, everything else identical to the run that produced 0.2040:

| | free head (reference) | **residual head** |
|---|---|---|
| best per-triangle MAE/σ | 0.2040 @ ep 130 | **0.2023 @ ep 104** |
| LR cuts used | 5 | **4** |
| epochs to reach 0.204 | 130 | **88** |

**Modestly better, and ~30% faster to converge.** It was ahead at most checkpoints early (roughly 2×
the pace to any given level) and the advantage narrowed in the low-LR grind.

**⚠ The weights are LOST.** The machine restarted at ~ep 106. The run predated the resume snapshots,
so none were written — and it would have failed to save anyway: the run tag encoded configuration
(holdout, filter, layers, width, epochs) but **not architecture**, so it was byte-identical to the
free-head run's tag and would have hit the no-overwrite guard after ~24 h. Both holes are now closed
(`HEAD_VERSION` in the tag; `--resume` snapshots at every evaluation, atomic via `.tmp` + `os.replace`).

**⚠ Caveat on the head's guarantee.** `X = 0` reproduces `A(s)` exactly and the model *starts* there,
but nothing forces `X → 0` on `W = 0` samples during training. "Exact where `W = 0`" is an
initialisation property, not a trained one, and should be measured on the next checkpoint.

## 3. Change 2 — the constraints

`C(s) = (1+W)ᵀA₃(1+W)`, and `W` solves a KKT system whose constraint operators are **geometry-only**
(`forward_solver_torch.py:613`). Their status in the model:

| operator | rows | nnz/row | structure | was it in the model? |
|---|---|---|---|---|
| `J_edge` | 1.5·N | 6 | per shared bond: `q_e·δg(s1) − q_e·δg(s2) = 0` | **yes, by accident** — `triangle_adjacency` *is* this pairing |
| `C_curv` | 0.5·N | 18 | per interior vertex: `Σ_{s∋v}(∂θ_v^s/∂g)·δg(s) = 0` | **NO — entirely absent** |
| `M_S` | 3 | N | area-weighted global mean | no |

### `J_edge` — verified, not assumed

The claim "adjacency already is the edge-compatibility coupling" had been asserted for a long time
and never checked. Measured:

- **pairing identical** — 177 solver edges → 354 directed, model 354, exact set match;
- **weights identical**, once the contraction is accounted for. The model stores `q_e` in the
  **state** convention (no factor 2 on shear) and contracts with `VEC3_METRIC = diag(1,2,1)`, which
  reproduces the **constraint** convention exactly: `max rel |m ⊙ bond_q − J_edge q| = 0.000e+00`.
  A raw vector-to-vector comparison shows rel 0.5 purely because the factor 2 lives in the metric
  rather than in the vector — `mesh_build.kkt_from_tri_bond`'s docstring warns about precisely this.

### `C_curv` — new `StarMP` channel

Triangle → vertex → triangle passing over the vertex stars. The weight `∂θ/∂g` is a **vec3**, the
same representation as the edge carriers, so it is split into a unit tensor `w/|w|` (equivariant,
direction) and its norm `|w|` (invariant, magnitude) — passing `w` raw would let its scale, which
spans orders of magnitude on sliver corners, swamp the tensor channels.

`model_v3.angle_gradient_vec` is a **literal port** of the solver's `_angle_gradient_vec`, and the
assembled operator is compared against the solver's own matrix:

| mesh | kernel vs solver | operator vs solver | rows | vec3 transform |
|---|---|---|---|---|
| random_patch(60) | **0.000e+00** | 4.441e-16 | 59 | 2.813e-14 |
| random_patch(120) | **0.000e+00** | 8.257e-16 | 116 | 2.813e-14 |
| bravais 1,1 | **0.000e+00** | **0.000e+00** | 16 | 2.813e-14 |

### `M_S` — new `GlobalMS` channel

Area-weighted mean pooled **per graph** and broadcast back. The quantity the constraint zeroes *is*
that mean, so the channel represents the constraint exactly rather than by proxy. Rank 3, and its
per-triangle influence falls as 1/N.

## 4. Why the constrained space has the dimension it does

A check that the constraint set is exactly the physically realisable subspace — the equivalence
`THEORY_NOTES.md` §5 asserts, measured here:

| mesh | 3N | J_edge | C_curv | M_S | rank | **3N − rank** | **2·nodes − 2** |
|---|---|---|---|---|---|---|---|
| random_patch(60) | 354 | 177 | 59 | 3 | 238 | **116** | **116** |
| random_patch(120) | 696 | 348 | 116 | 3 | 466 | **230** | **230** |
| bravais r4 | 96 | 48 | 16 | 3 | 66 | **30** | **30** |

**Exact on all three.** The per-triangle metric field is discontinuous — `δg(s1) ≠ δg(s2)` — but the
two triangles sharing an edge must agree on that edge's *length*, and imposing every such agreement
(plus zero discrete curvature, plus the mean) leaves precisely the fields some node configuration
could produce. The stacked operator is **rank-deficient by 1** (239 rows, rank 238), which is the
redundancy `CLAUDE.md` records as the root of the B-1 mis-solve, showing up independently here.

## 5. Gates

| gate | result |
|---|---|
| `G` invariant under rotation | **5.55e-17 / 1.11e-16** |
| `C → 𝒮C𝒮ᵀ` equivariant | **3.84e-16 / 1.92e-16** |
| SPD by construction | min eig **1.84e-03** |
| **untrained head == analytic `A(s)`** | **rel 0.000e+00** (exact, by zero-init) |
| nonzero `X` moves it | rel ~1e-01 (the correction channel is live) |
| reach, bond+star, no `M_S` | L=1: 0 beyond 2 union hops; L=2: 0 beyond 4 |
| `M_S` on | L=1 reaches **112/112** — global by construction |
| batched vs one-at-a-time | **0.000e+00** |

## 6. Three bugs the gates caught

**The global channel was silently OFF during evaluation.** `prepare()` set `area_w` but not
`tri_batch`, and `forward` enables `M_S` on `'tri_batch' in g` — so single-sample evaluation ran a
*different architecture* than batched training. Caught by comparing batched against one-at-a-time
output: **7.8e-02 apart**, now 0.000e+00.

**The zero-init made the receptive-field gate vacuous.** With the readout zeroed, `X ≡ 0` and
`G = G_an` depends only on each triangle's own edges — the counts fell to 2/6 and 2/13, i.e. only
the triangles owning the perturbed bond. The gate now randomises the readout first.

**The reach bound was wrong, twice.** Each layer is `TensorMP` (a bond hop) *then* `StarMP` (a star
hop), so a layer advances **2** union hops, not 1. The first gate versions asserted the wrong bound
and failed the model for the test's error.

## 7. Limitations

- **`M_S` destroys the finite receptive field** — one perturbation reaches every triangle in one
  layer. Faithful (the constraint *is* global), but it compounds the global normalisers already in
  `prepare()` (mean bond length, mean `k`), which are the prime suspect for the measured
  size-transfer degradation. `--no_global` exists so it can be ablated as a single variable.
- **The constrained model is untrained.** Everything here is architecture and gates; no training run
  has been done with the star or `M_S` channels.
- **Two variables at once** if star and `M_S` are enabled together — recommend `--no_star` /
  `--no_global` to keep runs single-variable.
- **`ℓ₀ = ℓ` throughout**, so that input channel still carries no information.
