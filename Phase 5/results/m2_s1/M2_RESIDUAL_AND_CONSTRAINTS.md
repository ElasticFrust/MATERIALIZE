# M2 — the residual head, and putting the CONSTRAINTS into the model

**What.** Two architectural changes, in order: (1) predict the *deviation* from the analytic
per-triangle tensor rather than the tensor itself; (2) give the model the constraint operators the
solver actually imposes, instead of making it infer their effect from examples.

**Producer.** `Phase 5/m2/model_v3.py`, `Phase 5/m2/train_v3.py`.
Gates: `Phase 5/verifications/test_m2_head_v3.py`, `Phase 5/verifications/test_m2_constraints.py`.
Scoring: `Phase 5/m2/evaluate_v2.py` (vs the independent sim),
`Phase 5/verifications/m2_error_strata.py` (the stratifications of §7).
float64, `OMP_NUM_THREADS=4`, seed 0, `data/dataset_v2_s0.npz` (41 431 samples).

> ## Headline (2026-09-11) -- READ THE MEDIAN, NOT THE MEAN
>
> The architecture work is real and replicated. **The tier verdict is not decidable from the mean**,
> because the mean is not reproducible at the sample size the criterion has always been evaluated at.
>
> | architecture | per-triangle MAE/sigma (`bravais` holdout) |
> |---|---|
> | v2 scalar messages | 0.5111 *(own-bulk oracle 0.4740 -- v2 was beaten BY it)* |
> | v3 tensor messages, free head | 0.2040 |
> | + residual head | 0.2023 |
> | + `C_curv` (star) | 0.1209 |
> | **+ `M_S` (area)** | **0.0925** |
>
> **Against the INDEPENDENT SIM on FRESHLY GENERATED, UNSEEN meshes** -- two independent builds
> (seeds 4321 and 777), 400 networks per family each:
>
> | family | MAE(nu) s4321 | MAE(nu) s777 | **median s4321** | **median s777** | verdict |
> |---|---|---|---|---|---|
> | `random` | 0.0256 | 0.0260 | 0.0099 | 0.0099 | passes both |
> | `bravais` | 0.0329 | 0.0396 | 0.0109 | 0.0105 | passes both |
> | `disordered` | 0.0583 | 0.0442 | 0.0142 | 0.0137 | **STRADDLES the line** |
> | `longrange` | 0.0750 | 0.1040 | 0.0121 | 0.0151 | fails both |
> | `cells` | 0.1613 | **0.2548** | 0.0466 | 0.0458 | fails both, worst |
>
> **The medians replicate to within 0.0004-0.003. The means swing by up to 58 %.** Same model, same
> protocol, two independent draws. That is the signature of a heavy tail: the mean is set by a
> handful of extreme networks and carries enormous sampling variance.
>
> **Consequences, and the first one is a spec problem, not a model problem:**
> - **the kill criterion is defined on the MEAN, and the mean is not reproducible at n = 400** -- so
>   a mean-based pass/fail at that sample size mostly measures how many outliers landed in the draw;
> - **every family's median is inside the kill line** (0.0099-0.0466), and `random`'s median 0.0099
>   is inside the MUST tier (0.02). More than half of every family is accurate; the failure is
>   entirely in the tail;
> - `cells` is the problem family -- worst mean, worst E (16.7 %/23.3 %), worst tail -- and its
>   cause is now measured (SS9).
>
> Full reading in SS7/7b; the generalisation and error-structure measurements in SS9.

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

## 7b. Adding `M_S` — the AREA constraint

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
| MAE(ν) | **0.0329** | **0.0109** | kill line 0.05 — cleared **on this family** (see §9) |
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
- **OVERFITTING IS RULED OUT. UNDERFITTING IS NOT ESTABLISHED** — and the difference is load-bearing,
  because it decides whether the next run should buy capacity. `m2_v3_report`'s train score is
  computed on the **unfiltered** train split, so it includes the 34.6 % of near-mechanism samples
  `--w_max_cut` removed and the model never saw; that number (0.1771) is contaminated and must not
  be quoted. Measured on what it actually fitted (same filter, same σ, 500 networks each):

  | | per-triangle MAE/σ |
  |---|---|
  | **TRAIN** (fitted, `‖W‖ ≤ 10`) | **0.1214** |
  | **VAL** (`bravais`, `‖W‖ ≤ 10`) | **0.0708** |

  **Read this carefully — train > val is NOT the signature of underfitting.** The textbook signatures
  are *underfit* = train ≈ val, both high; *overfit* = train ≪ val. Train **worse** than val is
  neither: it says the two sets are **different distributions**, which under leave-one-family-out
  they are by construction — `bravais` (regular lattices) is the easiest family, the training set is
  everything else.

  So what it establishes is exactly one thing: **the model is not memorising** — you cannot overfit a
  set you do worse on than on held-out data. There is no generalisation gap to close, so *more data
  is not indicated*.

  **It does NOT establish that capacity is the binding constraint.** That rests on weaker,
  circumstantial evidence: train error sits at 0.1214 rather than being driven toward zero; every
  earlier checkpoint showed train ≈ val; the run stopped on the LR floor with train loss pinned at
  0.0159. Suggestive, not decisive. *(An earlier version of this section asserted "capacity, not
  data" as settled, and a 5-day depth run was recommended on it. The user caught the inference.)*

  **THE DECISIVE TEST IS CHEAP — run it before buying capacity.** An **overfit probe**: train the
  same architecture to convergence on ~200 training networks only. If it can drive their error near
  zero, capacity is fine and the limit is optimisation or genuine ambiguity in the target; if it
  cannot, capacity is genuinely binding and depth/width is the right lever. Hours, not days — and it
  answers directly what a depth run would answer expensively and ambiguously.

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


## 9. Generalisation, and the STRUCTURE of the error  *(2026-09-11)*

Producers: `Phase 5/verifications/m2_fresh_holdout.py`, `m2_error_cancellation.py`,
`m2_validation_plots.py`. Figures: `m2_learning_curves.png`, `m2_parity_{nu,E}.png`,
`m2_by_family.png`, `m2_by_wmax.png`, `m2_error_cancellation.png`.

### 9a. A frozen model plus a seeded generator = honest test data on demand

Every number before this section came from ONE holdout family, `bravais`, because the hard families
were all trained on. The way out is not a re-split and a retrain (~2.4 days) but **generating fresh
data and scoring the frozen model on it (~25 min)** — the split has to be fixed before training, the
*measurement* does not. That capability is permanent: any future checkpoint can be scored on new
data without retraining anything.

### 9b. Generalisation to unseen meshes is GOOD

Matched comparison, both sides filtered at `‖W‖ ≤ 10` exactly as training was, same σ:

| | per-triangle MAE/σ |
|---|---|
| TRAIN (what it actually fitted) | 0.1197 |
| FRESH, unseen meshes | **0.1313** |

**+9.7 %.** The model transfers to meshes it has never seen at a ~10 % cost.

### 9c. The error CANCELS as √N — there is no systematic bias

`R = mean_s|err_s| / |mean_s err_s|` is the cancellation actually achieved; `R ~ √N` means
independent errors, `R ~ 1` means a systematic per-network bias.

| family | median n_tri | R | √N | **R/√N** |
|---|---|---|---|---|
| `cells` | 16 | 1.93 | 4.00 | 0.484 |
| `bravais` | 84 | 3.98 | 9.14 | 0.435 |
| `disordered` | 96 | 4.31 | 9.80 | 0.440 |
| `random` | 240 | 6.97 | 15.49 | 0.450 |
| **ALL** | 72 | 3.69 | 8.49 | **0.435** |

**`R` tracks √N with a CONSTANT prefactor across every family**, so there is **no large systematic
bias** — the errors behave as independent over blocks of `n_corr = 1/0.435² ≈ 5.3` triangles.

Two consequences:
- **the `cells` failure is just weak √N averaging on a small mesh**, exactly as the user argued it
  should be. `ν` is a contraction of the MEAN `C(s)`; with 16 triangles there is almost nothing to
  average. The measured MAE(ν) ratio `cells`/`random` = 6.3 against √(240/16) = 3.9, same order.
- **a bulk loss term is therefore NOT suppressing a bias** (there is none to suppress). It helps
  only through its *other* effect — per-graph weighting — which `--graph_balance` achieves directly
  and without the per-triangle/bulk trade (measured 0.6841 → 0.7326 at `--bulk_weight 1.0`).
  *(An earlier draft of this section explained the bulk term by the "null direction" argument —
  that a per-triangle loss cannot see correlated errors. True in principle, refuted here in fact.)*

**`n_corr ≈ 5.3 triangles` is close to `THEORY_NOTES`' independently measured cluster radius of 4–6
at high contrast** — the physics' own screening length. The model's errors appear correlated over
exactly the scale on which the physics couples. First quantitative handle on the reach deficit.

### 9d. The tier test is NOT REPRODUCIBLE at n = 400 — read the median

Two independent fresh builds (seeds 4321, 777), same frozen model, MAE(ν) vs the independent sim,
at **two sample sizes** — because the first question is whether the number is reproducible at all:

| family | s4321 n=400 | s777 n=400 | **s4321 n=2000** | **s777 n=2000** | median (all four) |
|---|---|---|---|---|---|
| `random` | 0.0256 | 0.0260 | 0.0280 | 0.0317 | 0.0099–0.0107 |
| `bravais` | 0.0329 | 0.0396 | 0.0331 | 0.0383 | 0.0101–0.0109 |
| `disordered` | 0.0583 | 0.0442 | 0.0568 | *(killed mid-run)* | 0.0133–0.0142 |
| `longrange` | 0.0750 | 0.1040 | 0.0820 | 0.1014 | 0.0121–0.0151 |
| `cells` | 0.1613 | **0.2548** | 0.2042 | 0.1578 | 0.0451–0.0466 |

**Five times the sample only halved the swing** — 58 % at n=400 → ~24 % at n=2000, about the √5 ≈ 2.2×
a finite-variance estimator would give, but nowhere near tight enough to decide a pass/fail at 0.05.
**The medians replicate to 0.0001–0.0014.**

The kill criterion is defined on the MEAN, so a pass/fail largely reports how many tail networks
landed in the draw — `disordered` straddles the line between two draws of the same model.

**Every family's median is inside the kill line**, and `random`'s (0.0099) is inside the MUST tier.
More than half of every family is accurate; the failure is entirely in the tail. **`cells` and
`longrange` fail on both draws; `random` and `bravais` pass on both.**

⚠ `cells` is also the only family with a NON-ZERO solver-vs-sim floor (MAE(ν) ≈ 0.014,
MAE(E)/E ≈ 1.3 %) — on small cells the labels themselves disagree with the sim, so ~9 % of the
model's error there is chasing a target the oracle disputes.


### 9e. `--bulk_weight` is REFUTED (preliminary); `--graph_balance` is untested

Identical config, seed and architecture; `--limit 5000`, L5, 15 epochs (NOT converged):

| variant | per-triangle val | bulk val |
|---|---|---|
| baseline | **0.3772** | **0.1016** |
| `--bulk_weight 1.0` | 0.4305 (+14 %) | 0.1018 (identical) |
| `--graph_balance` | *(one epoch only -- run killed)* | -- |

**The bulk term costs 14 % on the per-triangle metric and buys NOTHING on the bulk metric it exists
to improve.** Which is what §9c predicts: with no systematic bias to suppress, explicitly optimising
the mean cannot beat what the per-triangle field already delivers. Treat `--bulk_weight` as refuted
unless a converged run says otherwise; the flag stays (default 0) so the negative result is
reproducible.

`--graph_balance` remains the live hypothesis and is UNTESTED: it addresses the other effect, the
per-graph re-weighting (a 700-triangle mesh currently outweighs a 16-triangle one ~44:1), which is
the mechanism §9c actually identifies for the `cells` failure. Its single epoch-0 point was 0.6522
against baseline's 0.7423.


### 9f. Overfit probe — FINISHED 2026-09-12: the high-contrast error is a FITTING failure, not a generalisation gap

`--overfit_probe 200 --contrast_min 10000`, train == val, otherwise identical to the S1 run. The L5
arm was **resumed from its epoch-399 snapshot** (same model/optimiser/scheduler/history, cap raised
400 → 1600) and **stopped on its own measured rule at epoch 442**: 12 evaluations without a > 2 %
improvement, best **0.2213 @ ep 442**. L10 had already stopped itself at 200 epochs / 0.5258.

| | epochs | MAE/σ (on data seen every epoch) | train loss | lr at stop | ended by |
|---|---|---|---|---|---|
| **L5** | **442** | **0.2213** | 0.0552 | 7.29e-06 | **stopping rule** (was the 400 cap before) |
| **L10** | 200 | 0.5258 | 0.2322 | 2.43e-05 | stopping rule |

**The decisive comparison** (`Phase 5/verifications/m2_probe_vs_trained.py` →
`probe_vs_trained.json`): both checkpoints scored on the **same 200 samples**, with **one `sd`**
computed on that population, because the two runs each normalised by their own train-set `sd` and
their published numbers are therefore not on a common scale — the lesson of the "0.5144 was the train
split" error (§ 2026-08-27).

| | per-triangle MAE/σ | bulk MAE/σ |
|---|---|---|
| trained on all 40 775 (the 0.0925 model) | 0.3455 | 0.1220 |
| **overfit probe, these 200 only** | **0.2213** | **0.0615** |

**Reading.** Pointing all 397 328 parameters at 200 high-contrast samples, and training to
convergence on data it sees every epoch, improves per-triangle MAE/σ only **0.3455 → 0.2213 (−36 %)**
and gets nowhere near zero. So the high-contrast error is **not** a generalisation gap and **not** a
data shortage — the architecture **cannot fit that population even when free to memorise it**. This
is consistent with §9c (no systematic bias) and with the +9.7 % train→fresh-mesh gap: more data buys
nothing here.

**The bulk number localises it:** 0.0615 bulk against 0.2213 per-triangle. The probe fits each
graph's MEAN `C` well and fails on the per-triangle SPATIAL structure — the same split §9c found.

**Depth 10 stalls** (0.5258, lr collapsed to 2.43e-05), and the depth-8 arm diverged outright at
lr 3e-3: two independent signs that **depth past 5 is an OPTIMISATION problem in this architecture**,
not a free lever.

**Three limits on this verdict, all load-bearing:**
1. **"Free to memorise" is not established by counting.** The 200 samples carry 48 642 triangles =
   **291 852 target scalars against 397 328 parameters — only 1.4:1**, on a structured target. A
   clean capacity probe needs parameters ≫ targets (≈16 samples, ~30:1), so the honest claim is
   "cannot fit at this capacity ratio", not "cannot represent".
2. **Convergence rests on the plateau rule, not on an LR restart.** The lr fell 37× (2.70e-04 →
   7.29e-06) in the 36 epochs before stopping, and `CLAUDE.md` §3 is explicit that the decisive test
   is an **LR-restart probe** — reload the best checkpoint, restore the initial lr, train on. Not yet
   run; until it is, "0.2213 is the floor" is the weaker instrument's answer.
3. **Reach vs expressivity is still NOT separated.** This says the deficit is in fitting, not in
   data; it does not say which structural property is missing. (The earlier "reach is the deficit"
   claim stays withdrawn — see §9c.)

**The sizing error not to repeat:** 200 samples at batch 32 is **7 steps/epoch**, so the original
400-epoch cap was ~2 800 gradient steps against the real run's ~42 000. Size a capacity probe by
GRADIENT STEPS, not sample count.

**Evidence, and where it lives.** Cite the committed `run_v3_*.json`, **not** the `probe_*.log` files
— `*.log` is gitignored, so the logs are local scratch while the JSONs are in-repo and carry the
FULL per-evaluation `history` (epoch, train, val, bulk, lr) from epoch 0, a restart inheriting and
extending its parent's. L5 = `run_v3_res_bravais_w0_h1_L5_ns48_h64_e1600_star_ms_probe200_c10000.json`
(103 points, ep 0–442, best 0.2213).

**A bug this run exposed** (fixed, `train_v3.py:553`): the post-loop `per, bad = evaluate(...)`
unpacked 2 values from a 3-tuple, so **any run that reached its stopping rule crashed before writing
its checkpoint** — `blk` is consumed at the JSON dump. It survived because earlier runs were killed
or hit the epoch cap path. The `.resume` snapshot meant nothing was lost.

### 9g. TIGHT capacity probe (8 samples, 39.7:1) — CAPACITY IS REFUTED

> **ANSWERED BY §9h:** the expressivity-vs-optimisation question this section leaves open is settled
> PARTLY in the OPTIMISATION direction — LR restarts at a sensible rate took this section's
> "floor" of 0.1936 down to **0.1485**. But they do NOT reach zero, so a REPRESENTATIONAL deficit
> survives. Read §9g for the capacity refutation, which stands, then §9h for both halves.

§9f's verdict was bounded by a weak capacity ratio (1.4:1). This re-runs it at **39.7:1** — 8 samples
from the same contrast ≥ 10⁴ pool, same rng and seed, 1 670 triangles = 10 020 target scalars against
397 328 parameters. Sized by **gradient steps** this time: batch 2 → 4 steps/epoch, and it ran
**2 370 epochs ≈ 9 480 steps** against the 200-sample probe's ~3 100.
Command: `train_v3.py --overfit_probe 8 --contrast_min 10000 --batch 2 --epochs 6000 --eval_every 10
--patience 8 --stop_patience 20` (patience widened to keep the stop window ~800 steps, so a cheap
epoch cannot buy a premature "cannot fit").

Both probes scored against the real S1 model on their **own** sample set under **one `sd`**
(`m2_probe_vs_trained.py --n {8,200}`):

| probe | params : target scalars | grad steps | trained-on-40 775 | **probe** | probe bulk |
|---|---|---|---|---|---|
| 200 samples | 1.4 : 1 | ~3 100 | 0.3455 | **0.2213** | 0.0615 |
| **8 samples** | **39.7 : 1** | **~9 480** | 0.3248 | **0.1936** | 0.0571 |

**CAPACITY IS NOT THE BINDING CONSTRAINT.** A **28× better** parameter-to-target ratio and **3× the
gradient steps** moved the floor only **0.2213 → 0.1936 (−12.5 %)**. Two probes differing that much in
capacity land within 0.03 of each other, at ~0.2, on data they see every epoch. So §9f's bounded
claim is now unbounded in the direction that matters: **adding parameters will not fix the
high-contrast error, and neither will adding data (§9f).**

**Bulk 0.0571 vs per-triangle 0.1936** reproduces §9f exactly: the model fits each graph's MEAN `C`
and fails on the per-triangle SPATIAL structure. That is the deficit, stated in one number.

**WHAT IS STILL OPEN, and it is NOT a formality.** The run ended on the **LR FLOOR** —
`lr 2.19e-06 < 1e-3 of initial`, the scheduler having cut it repeatedly — **not** on the
no-improvement rule, and val was still creeping down (0.1945 → 0.1936 over the last 100 epochs,
~0.5 %). `CLAUDE.md` §3 is explicit that a clock- or decay-driven ending cannot distinguish *"found
the bottom"* from *"no longer allowed to walk"*, and that **the decisive test is an LR-RESTART
PROBE**. So the choice between
- **expressivity** — the architecture cannot represent the per-triangle high-contrast target, ⇒ the
  next move is STRUCTURAL (a new channel / a different message space), not a training knob; and
- **optimisation** — it can represent it but cannot be trained there from this initialisation at this
  schedule, ⇒ the next move is the schedule, the initialisation, or the loss,

**is not yet made** *(made in §9h: **optimisation**)*. Both probes ended with their lr exhausted, so
the agreement between them is *also* consistent with both hitting the same optimisation wall — which
is what §9h then measured. `--restart_lr` was added for it.

**What it does settle regardless:** neither more parameters nor more data is the lever, and
`--graph_balance` (a per-graph re-weighting) does not address a per-triangle spatial deficit either.

### 9h. LR-RESTART probes — the "floor" WAS PARTLY THE SCHEDULE (0.1936 → 0.1485 over two restarts), but restarts do NOT reach zero

`--restart_lr <rate>` added for this (`train_v3.py`), because `--resume` alone cannot do it:
`opt.load_state_dict` restores the **decayed** lr (measured 2.19e-06), so even a fresh scheduler keeps
it. Two restarts from §9g's epoch-2370 snapshot, both judged against its inherited best **0.1936**:

| restart rate | what happened | best reached | verdict |
|---|---|---|---|
| **3e-3** (the INITIAL lr) | threw 0.1936 up to **0.73**, stalled at 0.705, never re-entered the basin | 0.7012 | **instrument failure — measures nothing** |
| **2.7e-4** (a rate the run was still progressing at) | perturbed to 0.41, recovered past the old best within ~150 epochs | **0.1636 (−15.5 %)** | **the flat tail was the SCHEDULE** |

Runs, all committed with their full `history` (the `probe_*.log` files are gitignored scratch —
cite these): parent `..._probe8_c10000.json` (238 pts, ep 0–2370, 0.1936) · 3e-3 restart
`..._probe8_c10000_rlr.json` (259 pts, ep 0–2580, never beat 0.1936) · 2.7e-4 restart
`..._rlr0.00027.json` (313 pts, ep 0–3120, **0.1636**) · round 2 `..._rlr0.00027_rs2.json`
(375 pts, ep 0–3740, **0.1485**).

**So 0.1936 was not a floor.** §9g's stated caveat — that it ended on the lr floor with val still
creeping — was the right caveat, and cashing it in moved the number 15.5 %.

**"Restore the INITIAL lr" is the wrong probe here, and that is the reusable lesson.** `CLAUDE.md` §3
words the test that way, but 3e-3 is hotter than any rate at which this run ever made progress — the
parent's history shows 0.49 → 0.31 over a thousand epochs **all at 2.7e-4**, with 3e-3 only ever the
early phase. A restart hot enough to leave the basin cannot interrogate the basin's floor. Hence the
flag is a **RATE, not a switch**: pick a rate the run was observably still progressing at.

**A flaw in the first version of the flag, recorded so it is not repeated.** The stop clock was seeded
with the inherited `best`, which silently makes the criterion *"beat 0.1936 within `stop_patience`
evaluations"* — something a restart that first LOSES ground can essentially never satisfy. The 3e-3
run was killed 209 epochs in while still descending from 0.73. The clock now resets to **infinity**, so
a restart is judged on its OWN trajectory; whether it beat the inherited best is read off `best`.

**Round 2 (`--run_suffix rs2`, same 2.7e-4): 0.1636 → 0.1485, −9.2 %.** Real, and smaller than
round 1's −15.5 %. Restarts are **diminishing, not inexhaustible**:

| stage | MAE/σ | gain | ended by |
|---|---|---|---|
| 200-sample probe (1.4:1) | 0.2213 | — | lr decay |
| 8-sample probe (39.7:1) | 0.1936 | −12.5 % | lr floor |
| warm restart 1 @ 2.7e-4 | 0.1636 | **−15.5 %** | lr floor |
| warm restart 2 @ 2.7e-4 | **0.1485** | **−9.2 %** | lr floor |

**BOTH conclusions hold, and neither alone is the story.**
1. **The schedule was costing a real, bounded amount.** Two restarts took 0.1936 → 0.1485, **−23 %**
   cumulative. *Extrapolating* the 15.5 % → 9.2 % decay (ratio 0.59) puts the asymptote near **0.13**,
   ~30 % below the original "converged" value. ⚠ **That is a two-point extrapolation — a guess at the
   functional form, not a measurement.** A third round would test it; nothing here depends on it.
2. **Restarts do NOT reach zero, so a genuine FITTING deficit survives the schedule fix.** ~0.13–0.15
   on **8 samples the model sees every epoch, at 39.7:1 over-parametrisation**, is still a large
   error. With capacity refuted (§9g) and the schedule now accounted for, **the residual is
   representational** — so structural work (a new channel / a different message space) is back on the
   table, but measured against a *correct* baseline rather than a schedule-limited one.

**⚠ THE CONSEQUENCE FOR EVERY TRAINED NUMBER IN THIS DOCUMENT.** Every one came off the same
plateau-and-stop schedule — the ladder v2 0.5111 → 0.2040 → 0.2023 → 0.1209 → **0.0925**, and the tier
scores against the sim. **The RANKING is probably safe** (all arms shared the schedule), **but the
LEVELS are pessimistic, plausibly by ~20–30 %**, which is large next to the tier margins. **Do not
quote any of them as converged without re-checking under warm restarts.**

**PROTOCOL CHANGE THIS EARNS:** make **repeated warm restarts** (restart at a rate the run was still
progressing at, until a round buys < ~2 %) the standard, not a single decay to exhaustion — and make
it the **baseline any architectural change is measured against**, or a structural gain will be
confounded with schedule headroom the baseline never collected.

## 10. Limitations

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
