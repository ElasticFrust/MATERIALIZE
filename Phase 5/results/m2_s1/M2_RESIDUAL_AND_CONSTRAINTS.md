# M2 — the residual head, and putting the CONSTRAINTS into the model

**What.** Two architectural changes, in order: (1) predict the *deviation* from the analytic
per-triangle tensor rather than the tensor itself; (2) give the model the constraint operators the
solver actually imposes, instead of making it infer their effect from examples.

**Producer.** `Phase 5/m2/model_v3.py`, `Phase 5/m2/train_v3.py`.
Gates: `Phase 5/verifications/test_m2_head_v3.py`, `Phase 5/verifications/test_m2_constraints.py`.
Scoring: `Phase 5/m2/evaluate_v2.py` (vs the independent sim),
`Phase 5/verifications/m2_error_strata.py` (the stratifications of §7).
float64, `OMP_NUM_THREADS=4`, seed 0, `data/dataset_v2_s0.npz` (41 431 samples).

> ## Headline (2026-09-10) — **the KILL CRITERION IS CLEARED**
>
> Both constraint channels pay, and the second one carried S1 over the line:
>
> | architecture | per-triangle MAE/σ | MAE(ν) vs the sim |
> |---|---|---|
> | v2 scalar messages | 0.5111 | — |
> | v3 tensor messages, free head | 0.2040 | 0.0550 |
> | + residual head | 0.2023 | — |
> | + `C_curv` (star) | 0.1209 | 0.0572 |
> | **+ `M_S` (area)** | **0.0925** | **0.0329** |
>
> **MAE(ν) 0.0329 against a kill line of 0.05 — cleared for the first time in S1.**
> **MAE(E)/E 3.85 % against a must-tier of 5 % — met.** The must-tier on ν (≤ 0.02) is **not** met;
> in-domain it is **0.0231**, i.e. ~15 % away. Median MAE(ν) is **0.0109**, so over half the holdout
> is already inside the must-tier and the mean is tail-driven. Full reading in §7.

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

## 7. The trained result — both constraint channels pay

Run `res_bravais_w10_h1_L5_ns48_h64_e400_star`: `StarMP` **on**, `M_S` **off** (single variable
against the residual head), everything else identical to the run that produced 0.2023. 315 758
parameters, 101 epochs, stopped on the **LR floor** (2.19e-06 < 1e-3 of initial) at 50.2 h.
`--w_max_cut 10` kept 25 812 / 39 475 training samples; the 1 300-network holdout is unfiltered.

| head | per-triangle MAE/σ | best epoch |
|---|---|---|
| free SPD | 0.2040 | 130 |
| residual | 0.2023 | 104 |
| **residual + `C_curv`** | **0.1209** | 100 |

**−40 %, and it is not an average moving on one bin.** Scored over the whole 1 300-network holdout
(`m2_error_strata.py`), it improves everywhere:

| k-contrast | n | free head | **star** | change | MAE(ν) | median ν |
|---|---|---|---|---|---|---|
| 0 – 3 | 73 | 0.0240 | **0.0193** | −20 % | 0.0368 | 0.0108 |
| 3 – 10 | 275 | 0.0716 | **0.0414** | −42 % | 0.0368 | 0.0142 |
| 10 – 10² | 736 | 0.2106 | **0.1111** | −47 % | 0.0493 | 0.0208 |
| 10² – 10⁴ | 16 | 0.2250 | **0.1190** | −47 % | 0.0559 | 0.0316 |
| 10⁴ – 10⁹ | 200 | 0.3994 | **0.2880** | −28 % | 0.1391 | 0.0890 |
| **ALL** | 1300 | 0.2000 | **0.1185** | **−41 %** | 0.0599 | — |

The contrast trend is **not** flattened: still ~15× end to end. Coupling structure and reach are
different deficits, and the constraint channel did not remove the reach one.

### The verdict, and where it actually comes from

400 networks scored against the **independent sim** (`evaluate_v2.py`):

| | mean | median |
|---|---|---|
| MAE(ν) | **0.0572** | 0.0204 |
| MAE(E)/E | 6.57 % | 2.28 % |
| **SOLVER vs SIM (the floor)** | **3.6e-08** | — |

**Must tier NOT met; kill criterion (MAE(ν) > 0.05) STILL TRIGGERED** — and the tensor error fell
41 % while MAE(ν) moved 0.0550 → 0.0572, i.e. *slightly worse*. Those two facts are not in conflict:
ν is a ratio of contractions of `C_eff`, so it is dominated by a handful of networks where the
denominator is small, while the per-triangle MAE is an average over 88 560 triangles.

Splitting on the **training-domain boundary** says where the mean lives:

| domain | n | MAE/σ | MAE(ν) | median ν |
|---|---|---|---|---|
| `max‖W‖ ≤ 10` — **trained on** | 1190 | 0.1007 | **0.0452** | 0.0195 |
| `max‖W‖ > 10` — **filtered out of training, kept in the holdout** | 110 | 0.3111 | **0.2190** | 0.1742 |

Those 110 networks (8.5 %) carry **22 % of the total per-triangle error and 31 % of MAE(ν)**.
*(An earlier reading of this split put it at "roughly two thirds"; measured, it is 31 %. The
in-domain figure was right, the attribution was not.)* So the headline is a mixture of a model
inside its domain at 0.0452 and pure extrapolation at 0.2190 — the domain restriction is a
deliberate, documented choice, and it has to be read with the number, not around it.

### A finding this measurement produced: the head is NOT exact where `W = 0`

| max‖W‖ | n | MAE/σ | MAE(ν) |
|---|---|---|---|
| **exactly 0** | **254** | **0.0405** | **0.0528** |
| 0 – 1 | 86 | 0.0546 | 0.0100 |
| 1 – 3 | 585 | 0.0909 | 0.0269 |
| 3 – 10 | 265 | 0.1953 | 0.0894 |
| > 10 | 110 | 0.3111 | 0.2190 |

The `W = 0` bin has the **lowest per-triangle tensor error of any bin and the second-highest
MAE(ν)** — 0.0528, itself above the kill line. This is §2's caveat, measured: `X = 0` reproduces
`A(s)` exactly and the model *starts* there, but nothing during training forces `X → 0` on `W = 0`
samples, and it drifts. These are the samples whose answer is a closed form the architecture already
contains.

*(Follow-up, §7b: `M_S` repaired most of this on its own — that bin fell to MAE(ν) **0.0205** — so
the `X → 0` penalty dropped from "the obvious next change" to a second-order lever worth ~11 % of
the headline. Recorded because the re-ranking only happened when the stratification was re-run; the
conclusion above was carried forward one round too long.)*

## 7b. Adding `M_S` — the AREA constraint, and the kill criterion clears

Run `res_bravais_w10_h1_L5_ns48_h64_e400_star_ms`: `StarMP` **and** `GlobalMS` on, everything else
identical to the star run. 397 328 parameters (+26 %), stopped on the **LR floor** at epoch 104,
best **0.0925 @ ep 94**; the last five evaluations sit at 0.0925/0.0927/0.0926/0.0928/0.0926 with
train loss pinned at 0.0159, so it is converged rather than merely slowed.

**Per-triangle MAE/σ 0.1209 → 0.0925, −23 %**, and it converged at a *lower level at every schedule
position*, not merely lower at the end: its epoch-82 score is one the star run never reached at any
epoch.

| k-contrast | n | star | **star + `M_S`** | MAE(ν) star | **MAE(ν) + `M_S`** |
|---|---|---|---|---|---|
| 0 – 3 | 73 | 0.0193 | **0.0094** | 0.0368 | **0.0107** |
| 3 – 10 | 275 | 0.0414 | **0.0257** | 0.0368 | **0.0145** |
| 10 – 10² | 736 | 0.1111 | **0.0801** | 0.0493 | **0.0246** |
| 10² – 10⁴ | 16 | 0.1190 | **0.0899** | 0.0559 | **0.0163** |
| 10⁴ – 10⁹ | 200 | 0.2880 | **0.2493** | 0.1391 | **0.1260** |
| **ALL** | 1300 | 0.1185 | **0.0908** | 0.0599 | **0.0372** |

**Against the independent sim** (400 networks, `evaluate_v2.py`):

| | mean | median | tier |
|---|---|---|---|
| MAE(ν) | **0.0329** | **0.0109** | kill line 0.05 — **CLEARED** |
| MAE(E)/E | **3.85 %** | **1.01 %** | must-tier 5 % — **MET** |
| baseline (dataset-mean ν) | 0.2795 | 0.2316 | — |

The must-tier on ν (≤ 0.02) is **not** met. In-domain it is **0.0231** (was 0.0452), i.e. ~15 %
away; the median 0.0109 is well inside it, so the shortfall is entirely in the tail.

| domain | n | MAE/σ | MAE(ν) |
|---|---|---|---|
| `max‖W‖ ≤ 10` — trained on | 1190 | 0.0742 | **0.0231** |
| `max‖W‖ > 10` — never trained on | 110 | 0.2706 | **0.1899** |

Those 110 now carry **25 % of the tensor error and 43 % of MAE(ν)** — up from 31 %, not because
they got worse (0.2190 → 0.1899, they improved) but because everything else improved faster.

**Two things that did NOT improve, and they matter more than the headline:**

- **The contrast spread WIDENED, ~15× → 26×.** Every bin improved, but the easy end improved roughly
  twice as much as the hard end. `M_S` is a rank-3 global coupling; it does nothing for the
  contrast-dependent **screening length** of §1. Reach is still an open, untouched deficit.
- **The model is still UNDERFIT** — and measured properly this time. `m2_v3_report`'s train score is
  computed on the **unfiltered** train split, so it includes the 34.6 % of near-mechanism samples
  `--w_max_cut` removed and the model never saw; that number (0.1771) is contaminated and must not
  be quoted. Measured on what it actually fitted (same filter, same σ, 500 networks each):

  | | per-triangle MAE/σ |
  |---|---|
  | **TRAIN** (fitted, `‖W‖ ≤ 10`) | **0.1214** |
  | **VAL** (`bravais`, `‖W‖ ≤ 10`) | **0.0708** |

  Train is **71 % worse** than validation — the opposite of overfitting; there is no generalisation
  gap to close. Caveat that bounds the strength: leave-one-family-out makes train and val different
  *distributions* (`bravais` is the easiest family), so part of that 71 % is family difficulty. A
  within-family split would separate them. But the direction is unambiguous and consistent with
  every earlier measurement: **capacity, not data.**

### `--w_max_cut`'s stated premise is REFUTED

The flag's help text says near-mechanism networks are "where the solver is least trustworthy". That
was never measured; what *was* measured (7× worse at high ‖W‖) is the **model's** error, a different
claim that got conflated with label quality. Measured now, on the 6 020 samples where
`build_dataset` actually computes `sim_gap` (`structure == 'dilution'`, chosen *because* the solver
was expected to be fragile there — so a worst case):

| max‖W‖ | n | median gap | mean gap | max gap | frac > 0.05 |
|---|---|---|---|---|---|
| 1 – 3 | 300 | 8.1e-11 | 2.9e-03 | 4.97e-02 | **0.000** |
| 3 – 10 | 1498 | 1.3e-10 | 9.2e-05 | 4.56e-02 | **0.000** |
| 10 – 30 | 1418 | 2.8e-10 | 2.1e-05 | 2.57e-02 | **0.000** |
| 30 – 100 | 1332 | 6.7e-10 | 1.3e-05 | 1.77e-02 | **0.000** |
| 100+ | 1472 | 7.1e-09 | 8.6e-06 | **7.04e-03** | **0.000** |

**The solver–sim gap does not degrade with ‖W‖ — it improves.** Not one sample in any bin exceeds
the 0.05 flag. So the tail's labels are sound and the regime is learnable in principle; the filter
excludes 34.6 % of the training data on a premise that does not hold for this dataset.

**⚠ Do not act on that alone.** The model is capacity-limited (above), so adding a large, unseen,
harder regime without adding capacity risks spreading the same capacity thinner and degrading the
in-domain result. Coverage and capacity are separate problems and capacity is the measured one.

*(`sim_gap` is `0.0` with status `not_checked` for every non-dilution sample — `build_dataset.py:527`.
A first pass at this analysis read those placeholder zeros as measurements and concluded the solver
was perfect everywhere. Restrict to the checked subset before reading that column.)*

## 8. Limitations

- **`M_S` destroys the finite receptive field** — one perturbation reaches every triangle in one
  layer. Faithful (the constraint *is* global), but it compounds the global normalisers already in
  `prepare()` (mean bond length, mean `k`), which are the prime suspect for the measured
  size-transfer degradation. `--no_global` exists so it can be ablated as a single variable.
- **DEPTH IS STILL UNANSWERED.** The depth-8 arm (`..._L8_..._star`) **diverged** at epoch 20
  (train 0.0479 → 0.4560, val 0.1884 → 1.0081, worse than the global-mean baseline) and never
  recovered; it was killed at epoch 22. Its best weights (ep 12, 0.1879) are rescued in
  `checkpoint_..._L8_..._star_rescued.pt` with `diverged=True` in the metadata, and its history in
  `run_..._star_DIVERGED.json`. **That 0.1879 is a floor, not a measurement of depth** — the arm is
  confounded and needs a clean re-run at a lower LR (diverging at 3e-3 with 8 layers is itself
  evidence the rate is too hot for that depth). This matters more now that the model is measured
  capacity-limited and the contrast spread has widened.
- **The constraints are REPRESENTED, not IMPOSED.** The channels carry the operators' structure and
  their exact weights; nothing in the model solves the KKT system or projects onto its null space.
  Whether representation suffices is what these runs measure.
- **`ℓ₀ = ℓ` throughout**, so that input channel still carries no information.
- **Only one holdout family (`bravais`) has been scored** at this architecture; the leave-one-out
  result is not yet known to transfer to the other families.
