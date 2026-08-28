# v3 — tensor messages on triangle adjacency

**What.** Whether passing the network the **geometry itself** — the edge vector as a tensor `q_e`,
plus `k` and `ℓ₀`, and nothing derived — with **tensor-valued messages on triangle adjacency**, does
what v2's scalar messages on the node graph could not: predict the **per-triangle** `C(s)` on a
held-out topology family.

**Producer.** `Phase 5/m2/train_v3.py` (model `Phase 5/m2/model_v3.py`), scored and plotted by
`Phase 5/verifications/m2_v3_report.py`, ν/E vs sim by `Phase 5/m2/evaluate_v2.py`. Gates:
`Phase 5/verifications/test_m2_head_v3.py`. `OMP_NUM_THREADS=1`, float64, seed 0, `data/dataset.npz`.

---

## 1. Why v2 could not work — measured, not argued

v2's network **never saw the geometry**. `Q` appears nowhere in its `forward()`; it is used only
afterwards in `assemble(Q, G)`. What reached the network was, per bond, a single **scalar** length,
plus node-level **summary statistics** (min/max/std angular gap) and three triangle angles.

| where | what happened | why |
|---|---|---|
| `W = 0` | learned to **0.001 with ZERO message-passing layers** | `C(s) = A(s)` needs only each edge's own `(k, ℓ)` — both present in the scalars |
| `W ≠ 0` | nearest neighbours in feature space (distance 1.74) differ by **0.716** MAE/σ, against **0.697** for predicting the global mean | the scalars do not determine the target at all |

Depth did not rescue it: 0, 1 and 2 layers all landed at ~0.51. Three full runs — 0.5343 (with the
two scale bugs), 0.5111 (angles on), 0.5093 (angles off) — sit on that line.

The information that decides `W` is the **relative arrangement** of adjacent triangles, and v2
destroyed it twice: once by reducing each bond to a length, once by mean-aggregating **scalar**
messages between **nodes**, when the physics couples **triangles** through shared edges (edge
compatibility `J`, and the vertex angle sum in the curvature operator `𝒞`).

*The diagnosis is the user's; my own hypothesis — that the angle features were the missing
ingredient — was refuted by the ablation (0.5111 with, 0.5093 without).*

## 2. What v3 changes

**Inputs** — `q_e = vec3(Δx Δxᵀ)`, `k_e`, `ℓ₀,e`. Nothing derived: lengths, angles, degrees and gap
statistics are gone; the network forms them if it wants them. Absolute position is not passed (`C`
is translation-invariant to 5.9e-16, so it carries no information, and `bond_R` additionally carries
the periodic wrap that `pts[v] − pts[u]` does not). `ℓ₀` is passed although `ℓ₀ = ℓ` in all current
data, because it is a real input of `A(s)` and the residual-stress programme makes it independent.

**Messages** — on **triangle adjacency** (triangles sharing a bond), carrying both invariants
(`⟨T_s,T_t⟩`, `⟨T_s,q⟩`, `⟨T_t,q⟩`, `k`, `ℓ₀`) **and** tensors (a scalar-gated combination of `T_t`
and `q_b`), so orientation propagates instead of collapsing to a scalar at every hop.

**The algebra, measured before building on it** — the invariant inner product on vec3 `[xx, xy, yy]`
is `diag(1, 2, 1)`, i.e. `tr(AB)`, **not** the plain dot product, which changes by up to **11×**
under rotation. `G` must be **invariant** (`Q → 𝒮Q` already carries the rotation), so the readout
builds it from tensor inner products — which is where relative orientation finally reaches the
scalars.

## 3. Gates, all passed before training

| gate | result |
|---|---|
| `G` invariant under rotation | **2.2e-16 / 3.3e-16** |
| `C → 𝒮 C 𝒮ᵀ` equivariant | **4.9e-16** at two angles |
| SPD by construction | min eig **2.2e-02** |
| triangle adjacency | 336 directed edges over 112 triangles (~3 each) |
| **receptive field = exactly L hops** | L=1: 6/6 within, **0 beyond**; L=2: 13/13 within, **0 beyond** |
| `oracle_check` on the training data | **267 `W=0` samples exact** to machine precision |

The receptive-field gate had to perturb the **prepared tensors**, not the geometry: `lbar` and `k̄`
are global means, so perturbing an input moves every triangle through the normaliser. An earlier
version of that test did exactly that and "showed" influence on 112/112 triangles at L=2 — which is
impossible, and was the giveaway that the test, not the model, was wrong.

**Stability, a property of the design rather than a tuning detail:** the readout consumes `⟨T, T⟩`,
**quadratic** in the tensor state, so unnormalised growth squares into it — the first training step
saw a loss of **2e33**. Fixed by rescaling tensor channels by a scalar built from their own
invariants, equivariant **by construction** (a scalar commutes with `T → 𝒮T`); equivariance
re-verified at 2.2e-16 afterwards.

## 4. Result

**Training metric — v3 beats the own-bulk oracle; v2 never did.** Leave-one-family-out on
`bravais`, 8815 train / 900 val networks (88 560 val triangles), 150 epochs, seed 0. All rows
recomputed in ONE pass on the SAME split by `m2_v3_report.py` — which matters, see the correction
below.

| model | per-triangle MAE/σ | bulk MAE/σ | params |
|---|---|---|---|
| global mean | 1.0707 | 0.9505 | — |
| **own-bulk oracle (val)** | **0.4114** | 0 | — |
| own-bulk oracle (train) | 0.5414 | 0 | — |
| v2 scalar-msg | 0.5111 | 0.2903 | 11 270 |
| v2 scalar-msg (train) | 0.5009 | 0.2408 | — |
| **v3 tensor-msg** | **0.3906** | **0.2030** | 82 545 |
| v3 tensor-msg (train) | 0.3921 | 0.1631 | — |

SPD violations: **0** at every epoch, by construction of the head. Per-triangle correlation on the
holdout: `C_xxxx` **r = 0.912**, `C_xyxy` **r = 0.842** (`v3_scatter.png`).

**⚠ CORRECTION — the "0.5144 baseline" quoted throughout the earlier work is the TRAIN-split
value.** On the held-out `bravais` split the own-bulk oracle is **0.4114**. Two consequences: v3's
margin over it is **5%**, not the ~24% that was being announced against the mismatched reference;
and **v2 at 0.5111 is WORSE than predicting each network's own average** — v2 did not merely fail to
learn local structure, it was beaten by the trivial per-network predictor. The report script exists
precisely to compute model and baselines together on one split; the error was made verbally while
waiting for it.

**Headline metric (§1 must-tier) — FAILED, and the KILL CRITERION IS TRIGGERED.** 250 networks of
the held-out family, `MODEL → C_eff → ν, E` against the INDEPENDENT sim (`physical_homog.virial_C`,
not `sim_region_C6`):

| comparison | MAE(ν) | median | MAE(E)/E | median |
|---|---|---|---|---|
| **MODEL vs SIM (headline)** | **0.1619** | 0.0913 | **17.43 %** | 8.35 % |
| model vs solver labels | 0.1619 | 0.0913 | 17.43 % | 8.35 % |
| **SOLVER vs SIM (the floor)** | **0.0000** | 0.0000 | 0.00 % | 0.00 % |
| baseline: dataset mean ν | 0.3077 | 0.2500 | 130.37 % | 28.80 % |

- **must tier** (MAE(ν) ≤ 0.02, MAE(E)/E ≤ 5%): **NOT met** — 3× and 3.5× over.
- **kill criterion** (MAE(ν) > 0.05): **TRIGGERED** — 0.1619, and 0.0913 even on the median.

**The labels are clean, so all of this is the model's error.** `SOLVER vs SIM = 0.0000` on all 250
networks. There is no corrupted-label component and no floor effect to hide behind — a genuinely
useful negative result, since near-mechanism label corruption was the leading alternative
explanation.

**The relative-E denominator is floored** at `eps_E = 0.05 · median(E_sim) = 0.0341`, i.e.
`|E_m − E_p| / (|E_p| + eps_E)` — the convention `CLAUDE.md` §3 already uses for ν
(`/(|ν_sim| + ε_ν)`), approved 2026-08-27. Without it the metric is unusable: `E_sim` is
**bimodal**, ~4% of networks sitting at ~1e-5 against a median of 0.68, and those alone drove the
mean to **7648%** against a median of 8.86%. The unfloored value is still printed on every run so
the floor cannot quietly manufacture a pass.

### 4.1 The model is UNDERFIT — not overfit, not data-limited

**Train 0.3921 vs val 0.3906**: the held-out score is marginally *better* than the training score.
There is no generalisation gap at all. Adding data would not help — that is the lever when val is
much worse than train. Depth, width and epochs are the live levers, and `M2_LOCALITY.md`'s measured
recommendation is `n_layers` **4–5** against the **3** used here.

*(During the run I read the falling train loss against the flat val curve as overfitting. That was
wrong: train is a normalised MSE and val a normalised MAE, so their levels are not comparable. The
train-split score in `m2_v3_report.py` is what settles it, and it was added for exactly this reason.)*

### 4.2 Near-mechanism networks are much worse — but carry only ~29% of the error

Val error stratified by `max|W|` (the stored strain-concentration magnitude), 900 networks:

| max‖W‖ bin | n | MAE(ν) | median | share of total error |
|---|---|---|---|---|
| 0 – 3 | 615 | 0.1200 | 0.0820 | **53.5 %** |
| 3 – 10 | 178 | 0.1358 | 0.0882 | 17.5 % |
| 10 – 50 | 95 | 0.3366 | 0.2097 | 23.2 % |
| 50 – 200 | 11 | 0.6388 | 0.6131 | 5.1 % |
| > 200 | 1 | 0.8645 | 0.8645 | 0.6 % |

Error climbs monotonically 0.12 → 0.86, a **7× spread** — so near-mechanism networks *are* much
harder, and they are also where the solver itself is least trustworthy. **But they are rare:**
dropping everything above `max|W| = 10` removes 12% of networks and moves MAE(ν) only
**0.1531 → 0.1235**, still 2.5× over the kill threshold. **Over half the total error comes from
ordinary, well-conditioned networks with `max|W| < 3`.** Excluding near-mechanisms is worth doing —
it is a ~20% effect and removes the least trustworthy labels — but it is not the fix.

> **METHOD NOTE, worth keeping.** `corr(log₁₀ max|W|, |Δν|) = −0.045`, essentially **zero**, because
> the effect lives entirely in a sparse tail while the bulk is flat. Reporting only the correlation
> would have given "‖W‖ doesn't matter" — the exact opposite of the binned table. Same trap as the
> hexagon `rcond` correlation recorded in `CLAUDE.md` §3: **bin before you correlate.**

## 5. Limitations

- **The kill criterion is triggered, and that verdict stands as measured.** What it does *not* yet
  establish is the conclusion it exists to license — "the representation is inadequate" — because
  the model tested was underfit (§4.1) and below the recommended depth. The follow-up run (depth 5,
  `max|W| ≤ 10` training filter, Huber, 220 epochs) is what tests the representation. **No extra
  data is involved, so "do not scale data further" is respected.**
- **One family, one seed.** Only `bravais` was held out, and only seed 0 was run. The spread across
  families is unmeasured, and `bravais` may be unrepresentative — it is the most regular family.
- **The large-size holdout is unscored.** The ~1000-triangle bin the model never trained on is the
  real test of the locality argument, and it has not been run for v3.
- **Speed is unmeasured**, so the §1 stretch tier is untested. v3 costs 18× more per *training*
  epoch than v2, which says nothing about inference cost but is worth knowing before wiring it into
  a search loop.
- **The Huber loss and the `max|W|` filter are untested changes** at the time of writing — they
  enter with the follow-up run, and their effect is not separated from the depth change in it.
- **`ℓ₀ = ℓ` throughout**, so the `ℓ₀` input channel carries no information in this dataset. It is
  passed for the residual-stress programme, where it becomes independent — untested until then.

## 6. The DEEP run (depth 5, ‖W‖≤10 filter, Huber) — 2026-08-28

Depth 5, `ns=48 nt=10 hidden=64`, 178 173 params, batch 64, 220 epochs, `--w_max_cut 10`
(6599/8815 training samples kept, **holdout left UNFILTERED**), `--huber 1.0`. 11.5 h, 174 s/epoch.
Final per-triangle MAE/σ = **0.3561** (shallow: 0.3906).

**Three changes went in together — depth, filter and Huber — so their effects are NOT separated.**
That is a deliberate cost (one overnight run instead of three) and a real weakness of this result.

### 6.1 vs the independent sim, stratified by TRAINING DOMAIN

The filter creates a train/test domain mismatch by design: the holdout still contains 107 networks
with `max|W| > 10` that the model never trains on. Scoring must therefore be stratified, or the
filter looks like a regression when it is a domain restriction.

| domain | n | shallow d3 | **deep d5** |
|---|---|---|---|
| **trained (‖W‖≤10)** | 793 | 0.1235 / med 0.0833 | **0.1012 / med 0.0517** |
| excluded (‖W‖>10) | 107 | 0.3726 / med 0.2239 | 0.3930 / med 0.2504 |
| all | 900 | 0.1531 / med 0.0916 | 0.1359 / med 0.0599 |

**In-domain the deep model is 18 % better on the mean and 38 % better on the median**, and its
median (0.0517) sits essentially ON the 0.05 kill line. On the excluded regime it is only 5 % worse
than the model that *did* train there — so those networks are **intrinsically hard, not merely
unfamiliar**. They are 12 % of the holdout and carry **34 % of the total error**.

`MUST` tier still NOT met (0.02); `KILL` still TRIGGERED (0.1359 overall, 0.1012 in-domain).
Relative-E got worse on the mean (17.43 % → 26.96 %) while its median held (8.35 % → 8.95 %) — the
same fat-tail signature.

### 6.2 The regime FLIPPED: now a generalisation gap, so data IS the lever

| split | per-triangle MAE/σ |
|---|---|
| train (‖W‖≤10, equal-size subsample) | **0.2647** |
| val, in-domain (‖W‖≤10) | **0.3213** |
| val, unfiltered | 0.3496 |

A **21 % gap**, where the shallow model had none (0.3921 vs 0.3906). Capacity was the binding
constraint; it no longer is. **This is the condition under which more data becomes the right lever**,
and it is why the data scale-up was refused a day earlier and taken now — the diagnostic flipped,
not the argument.

### 6.3 ⚠ The large-size holdout looks like a triumph and is NOT one

The never-trained ~1000-triangle bin scores **MAE(ν) = 0.0360, kill criterion NOT triggered**. That
is an artefact:

| | per-triangle MAE/σ | bulk MAE(ν) | std(ν_true) | **ratio** |
|---|---|---|---|---|
| bravais family holdout (median 96 tri) | 0.3496 | 0.1359 | 0.4330 | **0.314** |
| large size holdout (median 1000 tri) | **0.4668** | 0.0360 | 0.1098 | **0.328** |

- **Per-triangle the model is WORSE on the large bin** (0.4668 vs 0.3496) — local accuracy *degrades*
  with size.
- The ν target there has **4× less spread**, and the bulk averages ~1000 triangles instead of ~96.
- **Normalised by target spread the two are identical** (0.328 vs 0.314).

So the good absolute score buys nothing: it is a narrower target plus more averaging. **The genuine
size-generalisation finding is the per-triangle number, and it is negative** — a real caveat on the
locality argument. One untested suspect: `prepare()` normalises by the GLOBAL mean bond length and
mean `k`, which are size-dependent couplings inside an otherwise local model.

> **This also exposes a flaw in the §1 criterion itself.** `MAE(ν) ≤ 0.02` is an ABSOLUTE threshold
> applied to sets whose ν spread differs 4×: it demands error ≤ 0.18·σ on the large bin but
> ≤ 0.046·σ on `bravais` — **four times harder for the same model**. The tier is not measuring one
> thing across families. Flagged, not changed — changing a success criterion mid-investigation is
> how results get manufactured.

### 6.4 Reproducibility note

Training is **bit-identical across 1, 4 and 8 BLAS threads** (train 2.0961/0.4630, val
0.9757/0.9267 on the probe). This is NOT in tension with the standing warning that BLAS thread count
changes *design* outcomes — that is the designer's L-BFGS path, a different code path. 4 threads is
the throughput optimum (88 → 49 s/epoch); 8 is worse (65 s) through oversubscription.

## 7. Figures

| file | what |
|---|---|
| `v3_learning_curves.png` | v2 vs v3 validation curves, log-y, with both baselines drawn as lines |
| `v3_components.png` | per-component MAE/σ, models side by side |
| `v3_scatter.png` | predicted vs true `C_xxxx` and `C_xyxy` on the held-out family |
