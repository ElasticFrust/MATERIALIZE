# M2 v2 — plan (forward surrogate first, edit-policy later)

**Status: PROPOSED, awaiting approval. No code written against this yet.**
Supersedes the "Next steps" of `M2.md` v1. Written 2026-08-24.

---

## 0. Decisions taken (were open; recorded here so they stop being re-litigated)

**D1 — What M2 IS.** `CLAUDE.md` §1/§2 called M2 a *GNN edit-policy* in four places while `M2.md`
and `model.py` (`ForwardGNN`) say *forward surrogate*. **Resolved: both, in order.** v2 is the
**forward surrogate**; the **edit-policy** is the endpoint and comes after. They are different
objects, and the surrogate is the natural **critic** inside a later policy, so nothing is wasted.
*(`CLAUDE.md` should be corrected to say "GNN surrogate → edit-policy" rather than "edit-policy".)*

**Why surrogate first, and the argument against it.** The surrogate is a **feasibility probe with a
clean pass/fail**: if a GNN cannot predict `C6` from a graph, it certainly cannot invert the map,
and the labels are free because the solver already produces them. The edit-policy's hard part is
*label design*, which should not be attacked before the representation question is settled.
**Against:** a surrogate approximates something we already have exactly *and* differentiably; its
only win is throughput and its error becomes a floor for anything built on it. That cost is accepted
here only because it doubles as the feasibility test.

**D2 — What it trains against.** Target the **tensor**, not the derived curves: `ν(θ), E(θ)` are
ratios/reciprocals of quartics in `C`, so predicting 74 numbers directly permits profiles **no
positive-definite `C` can produce**. Predict the tensor; derive `ν,E` with the solver's own
`c6_to_nuE` / `c6_to_nuE_theta`.

**D3 — Representation of the tilings.** Must be fixed BEFORE building (it changes node counts by
~50 %: honeycomb_r3 is 54 nodes as a fan, 36 as chords). **Default: `method='fan'`** — the shipped
default and the representation `test_hex_closed_form` validates to 4.4e-06. Record the choice in the
dataset provenance. *(Open: a chord-tiling arm is now available, `seeds.seed_tiling(..., 'chord')`.)*

**D4 — Threads pinned.** Measured 2026-08-23: BLAS thread count changes a design outcome
(ν −0.150 → −0.128, objective error 450×). The generator MUST pin `OMP/MKL_NUM_THREADS` and record
it per sample, or the labels carry unlabelled noise. See `b1_thread_local.py`.

---

## 1. Goal and success criteria

**Goal.** A graph → elastic-tensor surrogate accurate enough to replace solver forward passes inside
the M1 search loop, and trustworthy enough to serve later as a policy critic.

**Primary success metric — and it is deliberately harsh:**

> **MAE of `ν` and `E` against the INDEPENDENT SIM, on HELD-OUT TOPOLOGY FAMILIES.**

Not val loss; not agreement with the solver labels it trained on. `M2.md` records that the July
validation used a random split scored against its own training labels — which measures memorisation.
A design tool must work on a topology class it has never seen.

| tier | criterion | meaning |
|---|---|---|
| **must** | family-held-out `MAE(ν) ≤ 0.02`, `MAE(E)/E ≤ 5 %` | comparable to `gap_tol = 0.05`; usable as a pre-filter |
| **want** | `MAE(ν) ≤ 0.01` and SPD in 100 % of predictions | usable as a gradient source |
| **stretch** | ≥10× faster than the solver at equal accuracy on ≥200-node graphs | actually worth wiring into the loop |

**Kill criterion, stated up front.** If family-held-out `MAE(ν) > 0.05` after §4's protocol, the
representation is inadequate — **stop and report**, do not scale data further. That is the
feasibility answer, and it is a legitimate result.

---

## 2. Architecture

### 2.1 The head — physics-shaped, equivariant, and SPD by construction

The solver's own structure is `C(s) = (1+W)ᵀ A(s) (1+W)`, `A(s) = Σ_e c_e q_e q_eᵀ`, `q_e = Δx_e Δx_eᵀ`
(vec3). Pushing `(1+W)` through gives `C(s) = Σ_e c_e q̃_e q̃_eᵀ` with **`q̃_e = (1+W)ᵀ q_e`** — a sum
of outer products of *transformed* vectors. **A scalar-weighted sum of BARE `q_e q_eᵀ` is therefore
an AFFINE-ONLY ansatz and cannot represent the non-affine response.** *(This was caught by the user;
an earlier draft of this plan had exactly that error.)*

The exact, equivariant form: the three `q_e` of a triangle, when linearly independent, are a basis
of vec3. Stack them as columns `Q(s) = [q₁ q₂ q₃]` and predict, per triangle,

```
    passive :  C(s) = Q(s) · M Mᵀ · Q(s)ᵀ = (QM)(QM)ᵀ     M lower-triangular, softplus diagonal  (6 scalars)
    general :  C(s) = Q(s) ·   Γ   · Q(s)ᵀ                 Γ a free 3×3                          (9 scalars)
```

Properties:

- **Exactly expressive** — `Sym(3×3)` is 6-dimensional, so the passive form reaches *any* symmetric
  `C(s)`; no affine restriction remains.
- **Equivariant** — under a rotation `Q → 𝓡Q`, so `C → 𝓡C𝓡ᵀ`: the correct tensor law, while the
  network emits only rotation-**invariant** scalars. No capacity is spent re-learning a symmetry we
  know exactly.
- **SPD by construction** in the passive form (`XXᵀ`), so physicality is structural, not hoped for.
- **Interpretable** — at `W = 0`, `MMᵀ` is diagonal with entries `k_e/4ℓ_e²` (the bare tensor), so
  **the off-diagonal weights are exactly the non-affine content.** The model learns the correction.
- **`passive=False` admits ODD / ACTIVE elasticity** — a free `Γ` spans all of `3×3` including the
  antisymmetric part. SPD is a *passive-only* assumption and is a flag, not a hard-wired constraint.

**Limitation, recorded:** the parametrisation degenerates exactly where `A(s)` does — slivers or dead
`k` make the `q_e` dependent and `Q` singular. The surrogate inherits the solver's validity domain
(`CLAUDE.md` §3) rather than repairing it. Monitor `cond(Q)` per triangle and report it.

### 2.2 Assembly and the exact scaling symmetry

`C_eff = (1/N) Σ_s C(s)` — the **UNWEIGHTED** mean, matching the settled convention (`CLAUDE.md` §3;
area weighting biases ν and is tombstoned). Physical units via the same `8N/ΣS_s` factor.

**Exact symmetry to build in, free:** scale all `k` by λ ⇒ `A → λA`, `A⁻¹ → A⁻¹/λ`, so **`W` is
unchanged** and `C → λC`. Therefore: feed `k / mean(k)` to the network and multiply the predicted
`C` by `mean(k)`. `ν` is then scale-invariant by construction, as the physics requires.

### 2.3 Body

Keep v1's plain-torch message passing (no `torch_geometric`): `n_layers ≈ 4–5`, `hidden ≈ 128`.
Changes from v1:

- **Edge features → rotation-invariant only**: `[k/k̄, log(k/k̄), ℓ_e]` plus, per node, the sorted
  **angles between incident bonds**. Drop raw `dir_x, dir_y` from the *scalar* path — direction now
  enters only through `Q(s)` in the head, where it belongs.
- **Triangle-level readout**: message passing on nodes as now, then gather each triangle's three
  nodes + three edges to emit its 6 (or 9) weights. This is the structural change — v1 pooled
  globally and emitted `C6` directly, which is both non-equivariant and unable to express per-triangle
  structure.
- Optional later: simplicial/triangle-native message passing (`PLAN §9`).

---

## 3. Dataset

### 3.1 Families (the holdout axis)

From `seeds.seed_pool(include=('bravais','random','tiling','basis','auxetic'))`:

| family | source | what it contributes | current n |
|---|---|---|---|
| `bravais` | `seed_bravais` φ∈{0.8,1,1.2} × ψ∈{0.8,1} × η∈{0,0.15} | regular crystals, small `W`; the easy end | 12 |
| `random` | `random_patch` × 4 point processes (uniform / poisson_disk / blue_noise / graded) | disordered, large `W`; the bulk | `n_random` |
| `tiling` | square r4, honeycomb r3, kagome r3, square_octagon r3 | soft-`k` fictional edges, coordination ≠ 6 | 4 |
| `basis` | `honeycomb(3)`, `kagome(3)` | complex-basis crystals | 2 |
| `auxetic` | `auxetic_motifs(3)` (rotating squares, reentrant honeycomb) | **ν < 0**, the rare region | few |

**Balance is the problem, not size.** `random` can be generated without limit while `tiling`,
`basis` and `auxetic` are a handful of *topologies* each. So expand the scarce families along their
own parameters (reps, θ of rotating squares, v of reentrant honeycomb, η) rather than letting
`random` dominate — otherwise every held-out-family score is really "trained on random".

### 3.1b GEOMETRY (node positions) is a primary axis — and it fixes the balance problem

**Moving vertices on a FIXED topology creates essentially a new network**, and this project already
has the verified generator: **frozen-connectivity magnitude-η disorder** (`build_periodic_tf_mesh`,
`CLAUDE.md` §3 — perturb positions, **never re-triangulate**, no `uniform(−η,η)`). It is not mere
augmentation: it drives **ν from +1/3 down to ≈ −0.11**, so it traverses real physics.

This is the answer to §3.1's balance problem. `tiling`, `basis` and `auxetic` are a handful of
*topologies* each, but each expands into hundreds of distinct networks:

```
per topology:  η ∈ {0, 0.1, 0.2, 0.3, 0.4} × seeds (5, for η>0; η=0 is deterministic)
               × k-patterns (6) × sizes (3)      ≈ 450 samples per base topology
                                                 → ~10k across the zoo
```

Constraints, all from the project's own measurements:
- **η ≤ 0.42.** η=0.5 is singular; the sim health gate fires at **η ≥ 0.44** (only 3/10 seeds survive
  η=0.5). Try/except `UnhealthyGeometryError` and **record the surviving-seed count**, or the
  family average is a silently biased subset.
- **Never re-triangulate** — connectivity must stay frozen or the topology label is a lie.
- η=0 needs only one seed (deterministic); seeds matter only for η>0.

**⚠ This blurs the holdout.** At large η a honeycomb-topology network approaches a generic
disordered one, so families become similar and leave-one-family-out gets EASIER than it should be —
inflating the headline score. **Stratify:** hold families out at **low η**, where they are genuinely
distinct, and treat high-η as its own regime rather than letting it bridge the split. Report the
score as a function of η.

### 3.1c SIZE — vary it, and use it as an architecture test

Current build: **9 / 90 / 288** nodes (min/median/max).

`C_eff` is **INTENSIVE** — tile a crystal twice and `C` is unchanged. The §2.1 head gives
`C_eff = (1/N)Σ_s C(s)`, a **mean**, so this holds by construction. **v1 pooled `mean + sum`
globally, and the `sum` branch is EXTENSIVE** — its output grows with node count, which is simply
wrong for `C`. Size variation would have exposed it as a systematic bias; it is another reason the
head is replaced rather than tuned.

Plan: **train on 60–250 nodes** (cheap to label), **hold out a LARGE bin (500–1000) as a separate
generalisation test**. Large labels are expensive, so spending them on validation rather than
training is a real saving.

**Physical caveat — the receptive field.** `W` comes from a *global* constrained solve (compatibility
+ curvature couple the whole cell, like a Poisson problem), while a message-passing GNN sees only
`n_layers` hops. The surrogate can therefore only capture `W` insofar as the elastic response is
**screened** over a finite correlation length. **Prediction: accuracy degrades with cell size, and
fastest near a mechanism**, where the correlation length diverges — precisely where the solver is
fragile too. If observed, that is the architecture meeting a real limit, not a training failure;
mitigations are more layers, a global-context vector, or hierarchical message passing.

### 3.2 Sampling axes crossed with each topology

- **k-pattern**: `k0` (native/soft-fictional), `uniform`, `lognormal` (σ ∈ {0.3, 0.8}),
  `graded` (two amplitudes). ~6 per topology.
- **size**: `n_nodes ∈ {60, 120, 240}` for training; a separate 500–1000 bin held out (§3.1c).
- **seeds**: ≥5 per (topology, pattern, η>0); η=0 is deterministic (§3.1b).
- **η (geometry)**: {0, 0.1, 0.2, 0.3, 0.4}, capped at 0.42 — see §3.1b.

**Target ≈ 10 000 samples** (η makes this cheap), with **no family below ~10 %** of the total.

### 3.3 Labels

- Forward scan → **solver** `C6` (cheap, the bulk).
- Ingested designed networks (`Phase 5/networks/**/*.npz`) → **sim** `C6` where `C6_per` exists.
  These populate the auxetic/anisotropic corners the forward scan misses.
- **Record which label source each sample used.** Mixing solver and sim labels without a flag would
  make the validation uninterpretable.

### 3.4 Schema — store TRAJECTORIES, not just endpoints

Where a sample comes from an optimisation run, keep **every step** `(graph, k_i, C6_i)`, not only the
final network. It costs almost nothing (already computed) and it is exactly the demonstration data
the later edit-policy needs: `(G, C_i, C_target) → k_{i+1} − k_i`. One build, two consumers.

Per sample: `pts, bond_u, bond_v, bond_R, k, tri_bond, tri_verts, areas, BL1, BL2`, label `C6`,
derived `ν(θ), E(θ)`, descriptors (mean ν, mean E, anisotropy), and **provenance**: family, topology
id, k-pattern, seed, label source, trajectory id + step, tiling method, thread count, solver commit.

### 3.5 Two leakage traps — both would silently inflate the score

1. **Trajectory leakage.** Steps within one optimisation are highly correlated. Splitting by
   *sample* puts near-duplicates on both sides. **Split by trajectory id.**
2. **Ingest leakage.** A designed network inherits the family of the seed it came from. If it is not
   attributed, a held-out family reappears in training through the ingest. **Attribute every ingested
   network to its source family and hold it out with that family.**

---

## 4. Validation protocol

1. **Leave-one-family-out**, 5 folds. Train on 4 families, test on the 5th. This is the headline.
2. Score `ν, E` from the predicted `C` against the **independent sim** on the held-out family
   (`physical_homog`, not `_common.sim_region_C6` — that routes through the solver's own contraction
   and would be self-verification, `CLAUDE.md` §3).
3. **Also report the within-family random split** — not as a result, but to expose the gap between
   memorisation and generalisation. The difference is the number July did not have.
4. **Report SPD violation rate** (must be 0 by construction in passive mode — a nonzero rate means a
   bug, so it is a live assertion, not a metric).
5. **Report `cond(Q)` on failures** — tests whether errors concentrate where the parametrisation
   degenerates, i.e. whether the surrogate fails where the solver is also weak.

---

## 5. Milestones and gates

| # | deliverable | gate before proceeding |
|---|---|---|
| **M0** | fix `CLAUDE.md`'s "edit-policy" wording; pin threads + tiling method in the builder | — |
| **M1** | new head (§2.1) + invariant features, trained on the EXISTING 317-sample build | pipeline runs; SPD rate 100 %; sanity: predicts the regular lattice ν=1/3, E=2/√3 |
| **M2** | scaled, balanced dataset (§3), ~5 000 samples, trajectories + provenance | coverage cells ≫ 19; no family < 10 %; leakage checks pass |
| **M3** | full 5-fold leave-one-family-out training | the §1 **must** tier, or the kill criterion |
| **M4** | wire into M1's search as a pre-filter | solver calls per design reduced at unchanged design quality |
| **M5** | edit-policy on the stored trajectories | separate plan |

**M1 is deliberately on the small stale-free build**: it is an architecture check, not a science
result, and it costs minutes. Do not scale data before M1 passes.

---

## 6. Risks and open questions

- **The surrogate may simply not be needed.** If M4 shows the solver is not the bottleneck in the
  search, the honest outcome is to keep M2 as the policy critic only. Decide at M4, on measurement.
- **`ν` is a ratio and is ill-conditioned near `E → 0`.** MAE(ν) over a set containing near-mechanism
  networks may be dominated by a few samples. Report the distribution, not just the mean.
- **Size generalisation is untested by the family split** — a model trained on 60–240-node graphs may
  fail at 1000. Explicitly hold out the largest size bin as a secondary check.
- **Fan vs chord** (D3) is fixed to `fan` here, but the choice changes the graphs the GNN sees.
  If the pool ever switches, the dataset must be rebuilt, not fine-tuned.
- **Open:** whether to include `phase4` seeds (currently excluded from the default pool).
- **Open:** whether the edit-policy should act on `k` only, or on `k` + node positions. Positions
  have no cheap gradient (`CLAUDE.md` §3, SPSA), which is an argument for the policy to own them.
