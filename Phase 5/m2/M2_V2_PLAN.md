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

### 3.1b GEOMETRY — vary it per FAMILY, not with a blanket η

Moving vertices on a FIXED topology makes essentially a new network, and it is a real physical axis:
frozen-connectivity magnitude-η (`build_periodic_tf_mesh`, never re-triangulate) drives **ν from
+1/3 to ≈ −0.11** on the regular lattice.

**But η is NOT a universal axis, and an earlier draft of this plan wrongly used it as one.** Applied
blindly it is useless or actively harmful:

- `random` — already disordered; η adds nothing.
- **`auxetic` — η DESTROYS THE MOTIF.** Perturbing a re-entrant honeycomb makes it generic, so it
  *depopulates* the rare auxetic region it was supposed to fill. This is the worst case.
- `tiling` — degrades the mesh without clearly adding physics; the geometry *is* the tiling.

Use each family's **own natural parameter** instead:

| family | axis | why |
|---|---|---|
| `bravais` | **η** (≤0.42), φ, ψ | the family η was validated on |
| `auxetic` | **motif parameters** — rotating-square θ, re-entrant `v`, hexagon diameter `d` | traverses the family **while staying auxetic** |
| `tiling` | reps, k-pattern (+ optional chord/fan arm) | geometry is the tiling's definition |
| `basis` | basis parameters | same |
| `random` | point process, `n_nodes`, seed | disorder is already its axis |

**The hexagon diameter family is special: it has a CLOSED FORM**, `ν(d) = (4r²−1)/(3+4r−4r²)`,
matched to 4.4e-06 by `test_hex_closed_form`. It is simultaneously a continuous auxetic sampling
axis **and** a third-path ground truth — labels checkable without solver or sim. Use it as both.

η constraints where it IS used: **η ≤ 0.42** (0.5 singular; health gate fires at η≥0.44, only 3/10
seeds survive 0.5) — try/except `UnhealthyGeometryError` and **record the surviving-seed count**, or
the average is a silently biased subset. η=0 needs one seed.

**⚠ Any geometry axis blurs the holdout.** A heavily perturbed honeycomb approaches a generic
disordered network, making leave-one-family-out EASIER than it should be and inflating the headline.
**Stratify:** hold families out at LOW disorder, treat high disorder as its own regime, and report
the score as a function of the disorder parameter.

### 3.1c SIZE — from MINIMAL CELLS (analytic anchor) to large (generalisation test)

`C_eff` is **INTENSIVE** — tile a crystal twice and `C` is unchanged. The §2.1 head gives
`C_eff = (1/N)Σ_s C(s)`, a **mean**, so this holds by construction. **v1 pooled `mean + sum`, and the
`sum` branch is EXTENSIVE** — its output grows with node count, which is simply wrong for `C`. Size
variation would have exposed it as systematic bias; another reason the head is replaced, not tuned.

**MINIMAL CELLS ARE AN ANALYTIC ANCHOR, not merely cheap data.** The smallest periodic mesh is
`V=1, E=3, F=2` (Euler on the torus). In the small-cell limit the non-affine correction vanishes and

```
    C(s) -> A(s) = Σ_e (k_e / 4ℓ_e²) q_e q_eᵀ        (closed form)
```

which in the §2.1 parametrisation is the sharp prediction that **`MMᵀ` is DIAGONAL with entries
`k_e/4ℓ_e²`**. So tiny cells carry GROUND TRUTH, and they isolate the learning problem: the diagonal
is pinned analytically and everything to be learned sits in the **off-diagonals — exactly the
non-affine content**. This is an **M1 gate**, not just a sample.

It also inverts the receptive-field concern: a small cell fits entirely inside the GNN's receptive
field, so the model *can* be exact there, and the degradation with size becomes a measured curve.

*Caveat:* in a minimal cell every bond is a **self-loop** in the quotient graph (node to its own
periodic image, distinguished only by `bond_R`). Message passing must handle `u == v` sensibly —
test at M1.

Size plan: **minimal (2–20 tri) -> training bulk (60–250 nodes) -> held-out LARGE bin (500–1000)**.
Large labels are expensive, so they are spent on validation, not training.

**Physical caveat — the receptive field.** `W` comes from a *global* constrained solve (compatibility
+ curvature couple the whole cell, like a Poisson problem), while message passing sees only
`n_layers` hops. The surrogate captures `W` only insofar as the response is **screened** over a
finite correlation length. **Prediction: accuracy degrades with cell size, fastest near a
mechanism**, where that length diverges — exactly where the solver is fragile too. If observed, that
is a real architectural limit, not a training failure.

### 3.1d Is topology "immaterial — only its statistics"?

**Generically yes, and there is a hard version:** Maxwell counting. In 2D central-force networks the
isostatic point is `z_c = 4`; triangular is z=6 (rigid), kagome z=4 (marginal), honeycomb z=3
(floppy — **which is exactly why the tilings need soft fictional edges at all**). Statistics do real
predictive work.

**And it matches the model class:** a GNN with mean pooling *is* a learned statistic over local
environments — permutation-invariant, so it cannot memorise a specific wiring; it learns a
distribution over local motifs.

**But it FAILS precisely where the interest is.** Rotating-squares and re-entrant honeycomb are
auxetic because of a specific **collective geometric motif**, not because of their statistics: a
random network with the same `z` and length distribution is not auxetic. That is a collective mode,
not a local average — the same long-range issue as the receptive field.

**Split to carry forward:** statistics govern the generic/disordered bulk; specific structure governs
the rare, mechanism-driven corners a design tool exists to reach. **Consequence for the eventual
generative model:** it should emit **target statistics** (coordination, length/angle distributions,
motif frequencies) and then *realise* a network with them — far better posed than generating a graph
directly — with mechanisms handled as an explicit **motif vocabulary** rather than hoped for from
sampling.

*(Note: trajectory positions and broad geometry sampling are COMPLEMENTARY, not redundant. A
trajectory is a target-biased walk — demonstrations, for the policy. Broad sampling is unbiased
coverage — what the surrogate needs. A surrogate trained only on trajectory geometries would see
only the slice an optimiser visits en route to targets we happened to ask for.)*

### 3.1e THE GENERATOR SHOULD BE UNIT CELLS WITH BASES — not a fixed zoo

**Reframing (user, 2026-08-24): the UNIT CELL is the fundamental object; everything else is
repetition.** For a crystal, `C_eff` from the minimal cell **equals** `C_eff` from any supercell of
it — the periodic correction has the lattice's own periodicity. So a systematic sweep over **small
cells with N-node bases** is not a cheap corner of the space, it is a near-complete covering of the
**crystalline** part of it:

```
    N = 1  Bravais          N = 2  honeycomb-like        N = 3  kagome-like        N ≥ 4  the rest
```

The current zoo samples this with `bravais` (N=1) plus **two hand-picked** basis crystals. Replacing
that with an enumeration over (lattice vectors, basis size N, basis positions, connectivity) is
systematic, cheap to label, and far wider.

**Free gate that comes with it — SUPERCELL INVARIANCE.** Predict on the 1×1 cell and on its 3×3
supercell: the answers must be **identical**, with no labels required. This directly tests the
intensivity that v1's `sum` pooling would have violated, and it is a stronger statement than any
size-generalisation score.

### 3.1f DISORDER HAS A CORRELATION LENGTH — white-noise η is one corner

**The `η ≤ 0.42` bound is a REGULAR-LATTICE measurement and does not transfer** (user, 2026-08-24).
η is an *absolute* displacement, so on a mesh that already has short edges or thin triangles a
*small* η can collapse a triangle into a sliver. The safe range is mesh-dependent.

**Fix: stop gating on a scalar η; gate on SHAPE QUALITY.** `positions.tri_shape_quality` already
exists (its docstring records the sliver-vs-gap correlations). Perturb, then accept/reject on quality
and on the health gate, and **report the acceptance rate** — otherwise the surviving set is a biased
subset, the same trap as the η-sweep seed count.

**And white noise is only the zero-correlation-length corner.** The richer axis is a **correlated
displacement field**:

```
    u(x) = Σ_q A_q sin(q·x + φ_q)          sweep the spectrum ⇒ sweep the correlation length
```

`q` large recovers white-noise η; `q` small gives smooth long-wavelength deformation — **locally a
slightly-strained crystal, globally structured**. That regime is exactly what probes the
receptive-field question of §3.1c: a finite-hop GNN sees a locally-crystalline environment while the
global response differs. Sampling only white noise leaves that untested.

### 3.1g WIDENING THE SPACE — by physics, not by more randomness

The zoo is narrow, but "more random samples" is not the remedy. Widen along axes that move the
quantities physics says control `C`:

| axis | knob | why it matters | status |
|---|---|---|---|
| **coordination `z`** | **bond dilution** (soft-`k` subset) | Maxwell `z_c = 4` in 2D; sweeping z from 6 through isostatic is *the* rigidity axis | **ABSENT — biggest gap** |
| symmetry / anisotropy | lattice vectors, **basis size N** (§3.1e) | sets the tensor's symmetry class | 2 hand-picked bases |
| disorder correlation length | white noise → long-wavelength field (§3.1f) | separates local from collective response | white noise only |
| stiffness distribution | k-patterns; correlated k fields | | partly |
| mechanism presence | motif vocabulary (rotating squares, re-entrant, hexagon `d`) | the auxetic corners statistics cannot reach (§3.1d) | partly |

**Bond dilution is the biggest gap and it is available**: the tilings already use soft `k = eps` to
represent absent bonds, so dilution is a `k`-pattern, not a re-triangulation. It traverses **rigidity
percolation**, where `E → 0` and `ν` swings hard — a physically central region the zoo never visits.

**⚠ But it collides with a documented failure mode.** Dead / near-dead `k` is one of the two routes
to `A(s)` rank loss (`CLAUDE.md` §3): measured on a regular lattice driven to ν=−0.2, 11 % of bonds
dead gave **solver ν = −0.110 vs sim ν = +0.136 — opposite signs**. Note the contrast with the
hexagon gate, where `k_spoke = 1e-8` is *benign* because it is a SPOKE (the face can hinge) — so
softness itself is not the problem, its structural role is. Therefore dilution must be:
- **bounded** — keep `k_soft` in the benign soft-spring regime rather than driving it to zero;
- **health-gated**, and
- **cross-checked against the independent sim far more densely than elsewhere.**

It is simultaneously the richest axis and the one most likely to produce labels **the solver itself
gets wrong** — which makes it a place to validate before training, not after.

### 3.1h THE k FIELD — where the NONLINEARITY lives, so vary it as richly as the geometry

`A(s)` is **linear** in `k`, but `W` depends on `k` through `A⁻¹` — so **the k-distribution is
precisely what exercises the part of the map the GNN must actually learn.** At uniform `k`, `W` is a
pure function of geometry; contrast in `k` is what makes it non-trivial. The current zoo varies
roughly one of the four knobs below.

**Free simplification from §2.2's scaling symmetry.** `C` is homogeneous of degree 1 in a global `k`
scale (`W` invariant), so only the **SHAPE** `k/k̄` matters for `ν`, and the overall scale merely
sets `E`. The sampler therefore explores shape only — **one fewer dimension, for free.**

| knob | range | current |
|---|---|---|
| **marginal shape** | uniform · lognormal(σ) · **bimodal(f, ratio)** · **heavy-tailed (designed-like)** | lognormal only |
| **correlation length ξ** | iid (ξ=0) · **correlated field** · gradient (ξ≈L) · uniform (ξ=∞) | **only the two extremes** |
| **structure** (keyed to TOPOLOGY, not space) | **orientation** · **length** · **sublattice** · motif/region | absent |
| **contrast** `k_max/k_min` | decades, bounded (see below) | implicit in σ |

**Correlation length**, exactly as for geometry (§3.1f): `graded` is the single-mode limit and iid
`lognormal` is the zero-length limit — the plan has both endpoints and **nothing in between**. A
correlated log-normal field

```
    k(x) = exp( Σ_q A_q sin(q·x + φ_q) )
```

sweeps ξ, and it is the axis that tests whether `W` responds to **local** vs **long-range** stiffness
contrast — the screening question of §3.1c, now in the `k` channel rather than the geometry channel.

**Structure keyed to the topology is the biggest omission and the cheapest win:**
- **orientation-keyed** — `k` as a function of bond direction ⇒ produces anisotropic `C` directly and
  controllably (far more efficient than waiting for anisotropy to appear by chance);
- **sublattice-keyed** — e.g. the three bond orientations of a honeycomb get different `k`; breaks
  symmetry in a *designed* way, and pairs naturally with the basis enumeration of §3.1e;
- **length-keyed** — `k ∝ ℓ^α`, a one-parameter family, mimics fibre networks;
- **motif/region-keyed** — stiff inclusion in a soft matrix (the `two_region` / `glue_square_hole`
  idiom), i.e. a characteristic length between iid and global gradient.

*(Note the tilings' `k0` — natives at 1, fictional edges at `eps` — is ALREADY an instance of the
structured-bimodal class. It is the only one currently sampled.)*

**⚠ DEPLOYMENT SHIFT — the reason this is not optional.** The surrogate will run **inside the design
loop**, so it must be accurate on the `k` distributions **the optimiser actually produces**, and
those are measured **HEAVY-TAILED**: max/median **17** on a healthy design and **147** on a
degenerate one (`CLAUDE.md` §3, the B-4 colour-scale finding). Training on lognormal and deploying on
that is textbook distribution shift. The designed-network ingest (§3.3) covers it partly; heavy-tailed
`k` should be sampled **deliberately**, not hoped for.

**Unification: DILUTION IS NOT A SEPARATE AXIS.** §3.1g's bond dilution is the **large-contrast
corner of the bimodal family** (`k_lo → eps`). So the contrast sweep approaches rigidity percolation
*continuously*, and §3.1g's bounding + health-gating + denser sim cross-checks apply progressively
along the sweep rather than as a special case. Contrast is therefore the axis to ramp **last and most
carefully**, since it is the one that walks into the DEAD-k regime where the solver itself has been
measured to return the wrong sign.

### 3.2 Sampling axes crossed with each topology

- **k-pattern**: the four knobs of §3.1h — marginal shape × correlation length × topology-keyed
  structure × contrast. The v1 set (`k0`, `uniform`, `lognormal`, `graded`) is the ξ-extremes
  subset of this and is retained as the baseline arm.
- **size**: minimal cells + `n_nodes ∈ {60, 120, 240}` for training; 500–1000 held out (§3.1c).
- **seeds**: ≥5 per (topology, pattern, η>0); η=0 is deterministic (§3.1b).
- **geometry**: the family's OWN parameter (η only for `bravais`) — see §3.1b.

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
