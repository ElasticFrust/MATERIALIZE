# B-1 — investigation and fix, 2026-08-22 → 2026-08-24

> ## ✅ B-1 IS ROOT-CAUSED AND FIXED. **Read §4d (cause) and §4e (fix) first.**
> **§§1–4c are the INVESTIGATION TRAIL, kept in chronological order because the refutations are
> the useful part — but their headers and conclusions were written while the cause was still
> unknown and are superseded by §4d/§4e.** Do not quote §§1–4c as current status.

**What:** two full-suite repetition campaigns against audit **B-1** (solver nondeterminism), run as
the blocker ahead of M2, then the probe that actually found it. Campaign #1: 21 full
`test_inverse_design` runs, 10.32 h, arms alternating 1-thread / default-threads.

**Provenance:** commit `0b378d7`; working tree carried one uncommitted change to
`b1_overnight.py` (stderr capture, added before launch). Seeds are fixed by the suite.
Producer: `b1_overnight.py 10 --arms 1,0`. Analysis: `b1_excursion_analysis.py`,
`b1_thread_local.py`. Raw record: `overnight_20260822T201443Z.{csv,log}`.

---

## 1. Headline

| | |
|---|---|
| runs | 21 (10.32 h; 1458–2330 s each) |
| `[15]` B-1 excursions | **2** — run 14 `PASS_DRIFT` 8.0e-03, run 18 `FAIL` 1.18e-02 |
| observed `[15]` rate | 2/21 ≈ **9.5 %** (documented: 1 in 21 ≈ 4.8 %) — same order |
| excursion arm | **both on default threads; 0 in 11 on 1-thread** |
| `[4]` `test_local_region` | **11/11 FAIL on 1-thread, 10/10 PASS on default** |

Two independent findings came out of the night. **(A)** is settled and actionable. **(B)** was
still unexplained *at the time this section was written* — it is **resolved in §4d/§4e**.

---

## 2. (A) SETTLED — BLAS thread count changes a design outcome

`test_local_region` failed on every 1-thread run at exactly ν = −0.128 and passed on every
default-thread run at exactly −0.150. The overnight harness alternates arms strictly, so thread
count is **confounded with run parity** there. `b1_thread_local.py` breaks the confound by running
the problem alone — no suite context, no alternation:

```
threads=1  achieved=-0.127929  err=0.096566  FAIL
threads=4  achieved=-0.149881  err=0.000215  PASS
```

Same seed, same commit, same code. A threading difference already on record as **~1e-33** in the
forward path is amplified by 120 optimiser iterations into a **different basin**: ν moves by 0.022
and the objective error by **450×**.

Note `err` is `max_θ|ν(θ) − target|` while `achieved` is the θ-mean (`inverse_design.validate`), so
the failing design is angularly spread, not merely offset.

**Consequence for M2:** a training set labelled by this design path is **thread-count dependent**.
The dataset generator must pin the thread count and record it as provenance. The July set's thread
count was never recorded.

**Secondary consequence:** `test_local_region`'s docstring calls `reg=0.02` "correct and stable"
after the `reg=0.0` bistability. It is stable *at default threads only*.

---

## 3. (B) B-1 itself — better characterised, still unexplained *(AS OF 2026-08-23 — see §4d)*

Three excursions now exist on record (one from 2026-08-22, two from this run). All three are on
**case 1 (φ=ψ=1, η=0, seed 0 — the regular lattice)**, all on **4 threads**, all with the solver
**softer** than the oracle. Cases 2 (η=0.35) and 3 (ψ=0.6) have **0 hits in 21 draws each**.

### What is now established

1. **The deviation is DIFFUSE, not one component** — answering the question posed in
   `VERIFICATION_CAMPAIGN.md` §5.4 and unanswered since. Component deficits 0.42–1.20 %, with
   `xyxy` among the *smallest*. **The shear channel is not singled out, so the A-0 contraction
   class is excluded**; §5.4's own reading sends this to the solve.
2. **The oracle is bit-exact.** `c_phys` is identical (`max|ΔP| = 0`) across all three dumps and
   across two commits. B-1 lives in the solver path, full stop — previously an assumption.
3. **The wrong answers are DISCRETE and exactly reproducible.** Dump 0 (commit `502cb92`) and
   dump 2 (commit `0b378d7`) are **bit-identical**: `max|ΔC_solver| = 0.000e+00`, cos = 1.000000.
   Two distinct states so far, `vs` = 1.175383e-02 and 8.013331e-03, nearly collinear
   (cos = 0.9857) but not the same vector.

   **This is the strongest structural constraint available.** Bit-exact reproduction across
   different commits cannot come from accumulated rounding noise — the solver took a *different
   branch*, deterministically.
4. **Always over-compliant.** Every component of every excursion is softer than truth, never stiffer.

### Hypotheses tested and REFUTED

- **Dropped C2 (curvature) constraint.** Direction, diffuseness and the documented over-compliance
  all fit. Measured: dropping C2 gives 3.47e-02 (3× too large) and dropping C1 gives 1.47e-02;
  best fit along the C1 direction is λ = 0.69 with a 1.6e-03 residual on a 1.18e-02 excursion.
  **Neither toggle reproduces an excursion.**
- **Rank truncation at the `lstsq` cutoff.** `_intrinsic_dense_W` ends in `torch.linalg.lstsq(G, r)`
  on a `G` documented as having redundant rows; on CPU that is LAPACK `gelsy`, which truncates by
  effective rank against `rcond` — a genuine discrete branch, and ulp jitter flipping it would
  explain every observed asymmetry. **Refuted:** there is no knife edge. `σ_min = 6.21e-16` sits at
  **1.5e-04 × cutoff**, the tail gap is `2.43e-02 → 6.21e-16` (13 orders), **zero** singular values
  lie within 1e3× of the cutoff, and sweeping `rcond` over 1e-16 … 1e-5 leaves the answer unchanged
  at 2.2047e-13.
- **Ill-conditioning amplifying a ulp-level BLAS-kernel difference.** Refuted — the failing case is
  the *best*-conditioned of the three on every metric:

  | case | n_tri | cond(I3−Sw) | cond(A3)max | cond(G) |
  |---|---|---|---|---|
  | **1 REGULAR (fails)** | 336 | **1.015** | **4.57** | **4.5e16** |
  | 2 η=0.35 (clean) | 336 | 1.445 | 314 | 5.5e17 |
  | 3 ψ=0.6 (clean) | 576 | 1.017 | 11.1 | 1.5e18 |

  Consistent with the hexagon result already on record (`corr(log rcond, log|Δν|) = +0.73`, best
  accuracy at worst conditioning): **conditioning does not predict solver error in this project.**

  Cases 1 and 2 are also the *same size* and take the *same* dense path (threshold 600), so the
  dense/sparse selection is not the discriminator either. Only geometric regularity is.

---

## 4. Limitations — read before building on this

- **Every probe in §3's refutations ran in a clean isolated process, where B-1 never occurs**
  (documented: 0 in 30 isolated processes). They characterise the *healthy* state and infer
  backwards. They can refute a mechanism that requires a standing knife edge; they cannot identify
  what actually moved during an excursion.
- **The dumps record the wrong output and nothing upstream** — no `G`, no singular values, no
  effective rank, no intermediates. This is why three hypotheses could only be refuted. *(DONE
  2026-08-23: `_b1_dump_if_anomalous` now captures upstream state incl. `W`/`A3`, which is what
  localised the cause in §4d.)*
- **2 events is a small sample.** The arm asymmetry (2/10 default vs 0/11 1-thread) is suggestive,
  **not significant — Fisher exact two-sided p = 0.214**. The case asymmetry is the stronger of the
  two: all 3 excursions ever recorded are on case 1, against a uniform-over-cases null of
  p = (1/3)³ = **0.037** (this run alone, 2/2, gives 0.111). Neither reaches a standard threshold;
  both need more nights.
- **`[4]`'s thread result (§2) is settled independently** and does not depend on any of the above.
- One night at ~4.8–9.5 % per run cannot settle B-1's rate. The CSV is the cumulative record;
  append more nights before quoting a rate.

---

## 4b. Campaign #2 — 2026-08-23 (11 runs, 1 excursion)

Launched instrumented (upstream capture added after campaign #1). **11 runs, 5 default PASS,
6 one-thread FAIL — all six from `[4]`, not `[15]`.** One `[15]` excursion, on **run 3**.

**A machine suspend cost half the campaign:** run 4 recorded **19407.7 s (5.4 h)** against a 29 min
mean. The deadline is wall-clock, so the budget was consumed without buying runs — ~11 runs instead
of the projected ~20. *(The incremental write design held: the suspend cost time, not data.
**Fix for the next launch: budget by COMPLETED RUNS, not wall clock.**)*

### What run 3's excursion settled (`b1_anomaly_20260823T093703Z`, vs = 1.172970e-02)

1. **THREADING IS NOT NECESSARY.** It fired at **`threads=1, omp=1`**. Campaign #1's "0 in 11 on
   1-thread, both hits on default" was a small-sample artefact — its own p = 0.214 said so. Any
   mechanism requiring multi-threaded nondeterminism is dead.
2. **A 1-ULP INPUT DIFFERENCE BECOMES A 1.2 % OUTPUT ERROR.** `G` and `r` differ from a clean
   recompute by **4.441e-16** and 2.220e-16 — and 4.441e-16 is *exactly* `np.spacing(2.0)`, i.e.
   literally one representable step — while `C` moves by 1.17e-02. **~14 orders of amplification.**
3. **`G` is singular to working precision.** `cond ≈ 3e16`; `σ_min` is pure noise, measuring
   6.2e-16 / 9.5e-16 / 5.5e-15 on three runs of the same problem. Rank is stably 671/672.
4. **THE `lstsq` IS EXONERATED.** Feeding the anomalous `G, r` through `gelsy`, `gelsd`, `gelss`
   and `pinv` **all** reproduce the healthy `Λ` to ~1e-14. The solve is stable; the non-uniqueness
   of a rank-deficient solution is *not* the mechanism either.
5. Every other captured quantity is identical to full precision: `cond(I3−Sw)`
   (1.0147246528719565), `A3_cond_max` (4.5688), `r_norm` (1.1851), `G` symmetry (2.21e-15).

**By elimination the fault is in `W0` or `PinvJt`, in directions `J3` cannot see.** `J3` is
672×1008, so it has ≥336 null dimensions: a change of `W0` inside `ker(J3)` leaves `r = J3·W0`
untouched while still moving `W`. That is precisely the blind spot the first capture had.

### Status of the bisection

`_B1Capture` now also records the metric solve's input `A3` and output `W`, which splits the
remaining path in two — `W` matching the healthy run puts the fault **downstream in the contraction
to C**; `W` differing puts it **inside the metric solve**. It was added after run 3, so runs 4–11
carried it and **none of them hit — the bisection had no subject yet.** *(It got one on
2026-08-24 from `b1_persistence.py`, and the answer was `W`: see §4d.)*

### Combined rate

**3 `[15]` excursions in 32 campaign runs ≈ 9.4 %** (documented prior: 1 in 21 ≈ 4.8 %).
`[4]`'s thread dependence is now **17/17 fail on 1-thread, 15/15 pass on default** across both
campaigns, plus the isolated confirmation in §2.

---

## 4c. Bounding the excursion WITHOUT reproducing it (2026-08-23)

Waiting for a ~1-in-11 event that costs 30 min a draw is a bad instrument. These probes instead
**inject** candidate deviations into the healthy computation and ask whether any of them reproduces
a captured state. All are deterministic, take seconds, and need no excursion.
Script: `b1_excursion_analysis.py` (sections) + the injection probes recorded here.

**The pipeline is WELL-CONDITIONED.** A 1-ulp relative perturbation of `k` gives a ~1-ulp relative
change in `C`: amplification **0.60** (case 1), 2.64 (case 2), 0.82 (case 3). So B-1 is **not**
numerical instability — a perturbed input does not produce a wrong answer. Combined with the
bit-identical repeats across commits, this forces the conclusion that an anomalous run **executes
a structurally different computation**, rather than amplifying noise.

> **This retires the "1 ulp → 1.2 %, ~14 orders of amplification" phrasing used earlier in this
> doc's history and in commit `f5f61d0`.** The 1-ulp difference between the anomalous `G`/`r` and a
> clean recompute is a *symptom* of a slightly different upstream state, **not the cause** — feeding
> those exact `G, r` through any driver returns the HEALTHY `Λ`. The causal difference is elsewhere
> and its magnitude remains unmeasured.

**Injecting along `ker(J3)` — the inference it was built to test is REFUTED.** The final `W`
satisfies `J3·W = 0` by construction (measured 6.078e-15), so "the anomalous run landed on a
different point of `ker(J3)`" was the natural reading. `J3` is 672×1008 with rank 671, so
dim ker(J3) = **337**. Injecting `δ ∈ ker(J3)` (verified: `max|J3·δ| ~ 1e-17`):

- **`C` is remarkably insensitive to `W` inside the constraint manifold** — a **1 %** perturbation
  of `W` moves `C` by only **2e-04 … 5e-04**. Reaching 1.2e-02 would need `W` off by **30–40 %**.
- random `ker(J3)` directions give `cos` = −0.26, +0.08, −0.46, −0.16 against the observed
  deviation, while the **two real excursions agree with each other at cos = 0.9857**.

Both facts make the `ker(J3)` story implausible: it needs an enormous `W` error *and* one very
specific direction out of 337, hit twice independently. **Inference retracted.**

**Families that do NOT reproduce a captured state** (each measured):

| candidate | result |
|---|---|
| uniform rescale of `C` | best `a` = 0.990554 / 0.993842, cos ≈ 0.97 — but **21–24 % residual** |
| partial C1 drop | best λ = 0.69, **14 % residual** |
| under-applied `W` (`W → βW`) | deviation goes as (1−β)²; needs β ≈ 0.56, and still misses by 2.3e-02 |
| displacement in `ker(J3)` | wrong direction (above) |
| `lstsq` driver (`gelsy/gelsd/gelss/pinv`) | all reproduce the healthy `Λ` to ~1e-14 |
| dense vs sparse solve path | agree to **1.1e-08** — far from 1e-02 |

The excursion is ~80 % "everything softer by ~1 %" with an **irreducible ~20 % of structure that
nothing constructible reproduces.**

*(Incidental finding worth its own line: the dense and sparse implementations of the intrinsic solve
agree only to **1.1e-08**, not the 1.6e-12 the solver reaches against the oracle. Not the cause of
B-1, but it bounds how much the two paths may be treated as interchangeable.)*

### The next excursion is now a SHARP BINARY TEST *(it fired — the answer was `W`, see §4d)*

Because `C` is **linear in `A(s)`** but nearly flat in `W` within the constraint manifold:

- fault in **`W`** ⇒ `W` must differ by **~30–40 %** — unmissable
- fault in **`A(s)`** ⇒ `A3` need differ by only **~1 %**

The dump records both `W` and `A3` (+ checksums), so one more excursion discriminates on sight.
**Prediction, on economy of hypotheses: `A(s)`** — it needs a 1 % error where `W` needs 40 %, and a
uniformly smaller `A` gives exactly the diffuse, always-softer, near-uniform signature. `A3` is
built from `k` and geometry, both of which *should* be deterministic — so if this is right,
something **upstream of the solve** is moving. Flagged as a prediction, not a finding.

---

## 4d. ROOT CAUSE LOCALISED — the KKT constraint correction is applied WRONG (2026-08-23)

`b1_persistence.py`, first run, `reps=50`: **700 probes, 699 at bit-identical 2.20e-13, one
excursion at 1.172970e-02.** The probe made B-1 affordable — 700 draws in ~45 min against ~1 draw
per 30 min from a campaign, a ~500× improvement in draws per unit compute.

### Two facts from the sampling itself

- **The bad state is ONE-SHOT, not persistent.** The excursion lasted exactly one solve; all 49
  following probes returned to 2.20e-13. **This kills the "the process went bad and stayed bad"
  model that framed B-1 from the start.**
- It hit at **rep = 0** — the first solve after a context test. Given an excursion occurred,
  landing on rep 0 of a 50-probe burst is p = 0.02, so *"a preceding test leaves transient state
  that perturbs exactly the next solve"* is real signal. **Which test is NOT established**: it
  followed `test_homogeneity_regularizer`, but with a single event any given burst has p = 1/14.

### The cause

The dump carried the `W`/`A3` bisection payload, and it is unambiguous:

| quantity | healthy | anomalous |
|---|---|---|
| `A3` (bare tensor) | — | **bit-identical, `max|ΔA3| = 0.000e+00`** |
| `G` rank / cond | 671 / 4.523e16 | **identical** |
| `cond(I3−Sw)` | 1.0147246528719565 | **identical** |
| `W` Frobenius | 8.305366453251494 | 9.241718301662717 (**+11.3 %**) |
| **`max|J3·W|`** | **6.078e-15** | **5.427e-01** |

**Every input to the metric solve is bit-perfect; the output violates the constraints by fourteen
orders of magnitude.** `W` differs by 45.93 % (Frobenius) and **90.0 % of that difference lies in
the ROW SPACE of `J3`** — precisely the directions the constraints control.

Quantitatively, the correction is **~54 % applied**: `max|J3·W|` is 1.1851 with no correction
(`W0`), 6.1e-15 when correct, and 5.427e-01 in the excursion — i.e. 45.8 % of the uncorrected
residual survives. It is **not** any clean variant: it matches neither `W0`, nor edge-only (C1),
nor curvature-only (C2), nor the healthy `W`.

**This retro-explains every recorded property of B-1:** always **over-compliant** (an unenforced
constraint can only soften, hence the one-signed deviation); **diffuse** (the correction is global,
so no single component is singled out — consistent with excluding the A-0 shear channel);
**discrete and bit-identical across commits** (a specific wrong correction, not a continuum); and
**invisible upstream** (nothing feeding the solve is wrong, which is why every clean-process probe
in §4c found healthy inputs).

### What is still open

`W = W0 − PinvJt·Λ`, so the fault is in **`Λ` or `PinvJt`** — the dump does not yet capture either.
Note `G = J3·PinvJt` being bit-identical does **not** pin `PinvJt`, since `J3` has a 337-dimensional
null space. **Next instrumentation step: capture `Λ` and `PinvJt`**; that closes it completely.

The leading reading is that `torch.linalg.lstsq(G, r)` intermittently returns a wrong `Λ` for this
**singular** system (`G` rank 671/672, cond 3e16) — which would partially rehabilitate the pivoting
idea refuted in §3: the *rank decision* is stable (13-order gap), but `gelsy`'s *solution* for a
rank-deficient system need not be. §4c showed all four drivers return the healthy `Λ` from the
captured `G, r` **in a clean process**, which does not test immunity under the triggering
conditions. **If confirmed, the fix is to stop solving a knowingly singular system with a
pivoting-based driver.**

---

## 4e. THE FIX — a two-stage KKT guard in the core (2026-08-23/24)

`Phase 2/forward_solver_torch.py::_woodbury_solve_aw`. **Protected core** — gated by all five suites.

```
stage 1 (every solve, two mat-vecs):  orth = |Gᵀ(G Λ − r)| / (|G| max(|G Λ − r|, |r|))  >  1e-3 ?
stage 2 (only if triggered):          re-solve with the SVD driver; repair only if the
                                      CORRECTION TERM moves:  |PinvJt·ΔΛ| / max(|W0|, 1)  >  1e-6
```

**Why orthogonality and not the residual.** `Λ` is a valid least-squares solution iff its residual
is orthogonal to `range(G)`. When the constraint set is **inconsistent**, `|GΛ − r|` is irreducibly
nonzero and `J3·W ≠ 0` legitimately — so a raw-residual test fires on healthy solves. It did:
`sanity.py` at 1.9e-01 and the hexagon gate at 4e-08, both correct.

**Why the impact is normalised by `max(|W0|, 1)`.** `W` is a correction **to the identity** (`C` is
built from `1+W`), so its natural scale is 1, not its own magnitude. Dividing by `|W0|` alone
inflates without bound wherever `W ≈ 0` — exactly the uniform-k regular lattice, where `W ≡ 0`
identically — and produced spurious "repairs" of 1.9e+06, 3.9e+00 and 1.2e+01.

**Tolerances set from measurement, not reasoning** (three earlier ones were wrong because they
generalised a scale from too narrow a sample):

| | healthy | at failure |
|---|---|---|
| `orth`, clean meshes (η 0→0.45, anisotropic, uniform-k) | 3.1e-15 … 4.2e-14 | — |
| `orth`, designed mesh (39 solves in `optimize()`) | max **8.58e-08**, median 1.6e-12 | ~0.54 |
| trigger rate at 1e-3 | **0.00 %** | fires |
| `impact` on the `W ≡ 0` lattice | **2.10e-29** (old norm: 8.36e-15) | ~0.33–0.46 |

**No exception path.** A false trigger costs one extra solve and nothing else — earlier versions
could abort a design run on a merely ill-conditioned mesh, a worse failure than the bug. It emits a
`RuntimeWarning` on repair: **treat that warning as data — it is how the true rate gets measured.**

### Verification

- **All five gates pass**: `test_forward_solver` 8/8 (incl. gradients + large-N adjoint),
  `sanity.py`, `test_hex_closed_form` 3/3 (max |Δν| = 4.34e-06, the independent analytic oracle),
  `test_designer_surface` 5/5, `test_inverse_design` 16/16.
- **Injected the observed failure** (a Λ scaled by 0.542): the guard reports `orth` 5.372e-01 and
  repairs to 3.994e-15, giving a `C` within **2.08e-17** of the healthy answer. Unguarded, that same
  injection *is* the 1.2e-02 B-1 signature.
- **End-to-end**: `b1_persistence.py` 50, **0 excursions in 700 probes**, against **1 in 700**
  pre-fix — including the identical burst that produced the original.
- **Caught two genuine events in the wild** while verifying (`W` off by 3.788e-01 and 3.446e+00),
  each repaired, with every gate still passing.

### The rate, measured (2026-08-24, `b1_rate.py`)

The guard warns on every repair, so the rate is countable rather than inferred. Over the full suite
(whose `optimize()` loops issue thousands of intrinsic solves) plus 300 bare probes:

```
guarded intrinsic solves : 2488
repairs                  :    4
rate                     : 1.608e-03   (1 in 622)
95 % Wilson interval     : [6.25e-04, 4.13e-03]  =  1 in [242, 1599]
```

**Cross-checks against the independent estimate:** `b1_persistence.py` saw 1 excursion in 700
probes; 700 lies well inside the interval above. Two different instruments, same rate.
*(Wilson, not the normal approximation — at these rates `p ± 1.96√(p(1−p)/n)` is wrong and
collapses to zero width on a null result.)*

**RESOLVED — the large magnitudes are GENUINE.** The guard now refuses to repair unless the
orthogonality **provably improves** (a repair that does not improve it is a false positive by
definition), and the magnitudes survived that filter unchanged: min 1.765, median 2.534, max 48.07,
over 5 repairs in 2303 solves. So mis-solves are frequently far more catastrophic than the single
event dissected in §4d (~0.46) — established, not assumed.

```
run A (before the orth assertion)   4 / 2488  = 1 in 622   95 % CI  1 in [242, 1599]
run B (after  the orth assertion)   5 / 2303  = 1 in 461   95 % CI  1 in [197, 1078]
POOLED                              9 / 4791  = 1 in 532   95 % CI  1 in [280, 1012]
```

**THE LOOP CLOSES.** The pre-fix probe measured C-level excursions at **1 in 700 solves**; the guard
measures constraint mis-solves at **1 in 532 [280, 1012]**. Two independent instruments measuring
*different* quantities — observable tensor error versus constraint violation — agree. That means
essentially **every mis-solve produces a detectable error in `C`**, and it is the strongest
confirmation that the mechanism is correctly identified.

*(Earlier warning counts in this session are NOT usable for a rate — they were inflated by two buggy
stage-2 normalisations, dividing by `|Λ|` and then by `|W0|`. This is the first clean measurement.)*

### Honest scope

This is a **guard, not a cure**: whatever makes `lstsq` occasionally mis-solve this singular system
may remain, and the underlying `Λ`-vs-`PinvJt` question (§4d) is closed only in the sense that the
wrong result can no longer propagate. Given `G` is singular **by construction**, guarding is the
right call regardless — and the warning now measures the rate instead of inferring it.

*(Incidental, and NOT B-1: on ~1 in 39 designed-mesh solves `gelsy` and `gelsd` disagree by ~1.1e-02
in `W` while **both are valid least-squares solutions** (`orth` ≤ 8.6e-08) — `ker(G)`
non-uniqueness. Via the `ker(J3)` insensitivity in §4c that is only ~3e-04 in `C`, two orders below
B-1. Correctly not flagged, but it is a real residual source of solver variability.)*

---

## 5. Files

| file | role |
|---|---|
| `overnight_20260822T201443Z.csv` | one row per run: arm, verdict, seconds, detail |
| `overnight_20260822T201443Z.log` | full stdout (+stderr) of all 21 runs |
| `b1_anomaly_20260822T142234Z_eta0.0_s0.json` | excursion 1 (pre-existing, 1.175383e-02) |
| `b1_anomaly_20260823T030730Z_eta0.0_s0.json` | excursion 2 — run 14, 8.013331e-03 |
| `b1_anomaly_20260823T050059Z_eta0.0_s0.json` | excursion 3 — run 18, 1.175383e-02 (≡ excursion 1) |
| `../b1_excursion_analysis.py` | §3 — structure, rank test, conditioning |
| `../b1_thread_local.py` | §2 — the thread-count experiment |
