# B-1 overnight run — 2026-08-22/23

**What:** the first full-suite repetition campaign against audit **B-1** (solver nondeterminism),
run as the blocker ahead of M2. 21 full `test_inverse_design` runs, 10.32 h, arms alternating
1-thread / default-threads.

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

Two independent findings came out of the night. **(A)** is settled and actionable; **(B)** is
better characterised than before but still unexplained.

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

## 3. (B) B-1 itself — better characterised, still unexplained

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
  effective rank, no intermediates. This is why three hypotheses could only be refuted. **The single
  highest-value next step is to extend `_b1_dump_if_anomalous` to capture the upstream state**, so
  the next excursion is diagnostic rather than merely another data point.
- **2 events is a small sample.** The arm asymmetry (2/10 default vs 0/11 1-thread) is suggestive,
  **not significant — Fisher exact two-sided p = 0.214**. The case asymmetry is the stronger of the
  two: all 3 excursions ever recorded are on case 1, against a uniform-over-cases null of
  p = (1/3)³ = **0.037** (this run alone, 2/2, gives 0.111). Neither reaches a standard threshold;
  both need more nights.
- **`[4]`'s thread result (§2) is settled independently** and does not depend on any of the above.
- One night at ~4.8–9.5 % per run cannot settle B-1's rate. The CSV is the cumulative record;
  append more nights before quoting a rate.

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
