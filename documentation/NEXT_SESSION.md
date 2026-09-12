# NEXT SESSION — start here

**Rewritten 2026-08-22; §1 rewritten 2026-08-24 (B-1 FIXED); banner + §2–§3 rewritten 2026-08-25
(M2's decisions taken, work staged S0…S5).** Read `CLAUDE.md` §1–3, this file, and
**`documentation/VERIFICATION_CAMPAIGN.md` — Part I is the INDEX of what has already been measured,
one line per analysis script. Read it before proposing any new measurement, and say what you found,
including "nothing".** `AUDIT_2026-08.md` §6 STATUS holds the older backlog.

> **The goal is a LEARNED MODEL OF THE SOLUTION SPACE — not the GNN surrogate, and not any single
> optimised network** (`CLAUDE.md` §1). The forward map is many-to-one, so the endpoint is a
> generative / edit **neural network** (**GNN surrogate → edit-policy**, VAE, with an interpreter
> front-end mapping user intent → targets). Within that arc the **edit-policy is the endpoint (S5)**;
> the **forward surrogate comes first** because it is the cheap feasibility probe — if a GNN cannot
> predict `C6` from a graph it certainly cannot invert the map — and because it is the policy's
> natural **critic**. The differentiable designer (M1) is the *engine* that produces the labels,
> explicitly **not** the goal.
>
> *(An earlier version of this banner read "the goal is the GNN". That was drift; caught by the user
> 2026-08-25 and corrected here.)*
>
> **Nothing is blocking.** B-1 was root-caused and fixed 2026-08-24 (§1); M2's decisions **D1–D9 are
> taken** (2026-08-25); the live plan is **`Phase 5/m2/M2_V2_PLAN.md`**, staged **S0…S5**.
> **The next action is S1** (§2).

---

## THE TODO, in order

### 1. B-1 — ✅ **ROOT-CAUSED AND FIXED (2026-08-24).** No longer a blocker.
Full write-up: **`Phase 3/verifications/b1_dumps/B1_OVERNIGHT.md`** (§4d cause, §4e the fix and its
verification). Settled statement of record: `CLAUDE.md` §3.

**The cause.** The intrinsic solve ends in `torch.linalg.lstsq(G, r)` on a `G = J3·PinvJt` that is
**singular by construction** (redundant constraint rows; rank 671/672, cond ~3e16). Roughly **1 in
532 solves** its pivoting CPU driver returns a `Λ` that only *partially* applies the KKT correction:
every input stays **bit-perfect** while the returned `W` violates `J3·W = 0` by fourteen orders, and
`C_eff` comes out **~1 % over-compliant** — above design tolerances, indistinguishable from a real
result. That one mechanism explains B-1's whole signature (always over-compliant, diffuse,
discrete/bit-identical across commits, invisible upstream).

**The fix** (protected core, `_woodbury_solve_aw`): every solve tests the residual's orthogonality
`Gᵀ(GΛ−r) ≈ 0`; only on a trigger does it re-solve with the SVD driver, and it repairs only if the
correction term actually moves **and** the orthogonality provably improves. No exception path; warns
on repair.

**Verified:** all five gates green · injecting the observed failure repairs to within **2.08e-17** of
the healthy `C` · `b1_persistence.py` **0 excursions in 700** vs **1 in 700** pre-fix, same burst ·
rate **1 in 532, 95 % CI [280, 1012]**, cross-checking the probe's independent 1-in-700.

**What is still open about B-1** (none of it blocking): it is a **guard, not a cure** — the
library-level intermittency may remain, but a wrong result can no longer propagate silently; the
`Λ`-vs-`PinvJt` question of §4d is moot rather than answered; and the *trigger* (an excursion hits
the first solve after a context test, p = 0.02, but **which** test is unestablished, p = 1/14).

**New instruments, all in `Phase 3/verifications/`:** `b1_persistence.py` (the one that found it —
probes `[15]`'s case 1 repeatedly inside one real suite process, ~500× more draws per unit compute
than a suite-repetition campaign) · `b1_rate.py` (repair rate, Wilson interval) ·
`b1_excursion_analysis.py` (dump structure + the refutation trail) · `b1_thread_local.py` (§2 below).

> **The lesson that cost the most:** the overnight campaigns were the wrong instrument — 32 suite
> runs, ~20 h, 3 excursions, no cause. The probe found it in 45 min by making a draw cost 1.5 s
> instead of 30 min. **Prefer making the event cheap over buying more of it.**

> **RESULTS OF THE FIRST RUN: `Phase 3/verifications/b1_dumps/B1_OVERNIGHT.md`** — 21 full-suite
> runs, 10.32 h, **2 excursions caught** (8.0e-03 and 1.18e-02, both on default threads, 0 in 11 on
> 1-thread). B-1 is **much better characterised but NOT explained**; three mechanisms were refuted.
> A **separate and SETTLED** finding came out of it: **BLAS thread count changes design outcomes**
> (ν −0.150 → −0.128, objective error 450×, same seed/commit) — so **M2's dataset generator must
> pin the thread count and record it as provenance.** See §1a below for what is now known.

**Caught again 2026-08-22** during a routine gate run: `test_homogenization`, regular lattice
(φ=ψ=1, η=0, seed 0), **1.18e-02 against a 1.6e-12 baseline**, then 8/8 PASS on repeat in isolation.
Evidence in `Phase 3/verifications/b1_dumps/` (harness auto-dump
`b1_anomaly_20260822T142234Z_eta0.0_s0.json` + the suite log).

**Ruled out today:**
- **The cheap harness does not reproduce it.** New `b1_reproduce.py threads N` mode: suite context
  built once, then 360 probes across 1- and 4-thread arms → **0 hits in both**, every probe at
  exactly 1.00× baseline. ~17 hits were expected at the nominal rate. So replaying the 13 preceding
  tests is **not sufficient context** — consistent with "clusters in time with a persistent state
  transition", and it means the "~1 in 21" refers to something other than these probes.
- **Thread count is not the direct cause.** Threading *does* make the forward path nondeterministic
  (40 calls on the regular lattice: 2 distinct bit patterns at 4 threads, 1 bit-exact at 1 thread —
  isolated for the first time), but at **1.5e-33**, thirty orders below the failure, consistent with
  the 1–2 ulp envelope already on record.

**THE RUNNER — `Phase 3/verifications/b1_overnight.py`** (first run 2026-08-22/23; re-running
APPENDS a new timestamped CSV, so keep accumulating nights):

```
python "Phase 3/verifications/b1_overnight.py" 10          # 10-hour budget, arms 1-thread / default
python "Phase 3/verifications/b1_overnight.py" 10 --arms 1,0
```

Each iteration is a fresh subprocess running the full suite (~12 min), alternating a **1-thread** arm
and a **default-threads** arm, so the thread question is answered alongside. It appends one CSV row
and **flushes + fsyncs after every run** — a suspend at hour six must not cost the night (one g1_2
design already recorded 18266 s of wall clock from exactly that). Re-running APPENDS, so the CSV is
cumulative across nights. Output: `b1_dumps/overnight_<UTC>.{csv,log}`.

It flags two things: an outright `FAIL`, and `PASS_DRIFT` — a pass whose [15] line no longer shows
the usual `solver-vs-physical(tensor)=1.6e-12`, i.e. a quiet change in the number that is not yet
large enough to trip the assert. ~40 runs/night, ~2 expected events at the observed rate.

Useful control that fell out: `torch.set_num_threads(1)` makes the forward path bit-reproducible, so
"noise or real state change?" becomes decidable.

### 1a. What the 2026-08-22/23 run ESTABLISHED (full doc: `b1_dumps/B1_OVERNIGHT.md`)

**Facts, not inferences:**
- **The wrong answers are DISCRETE and bit-identical across different commits**
  (`max|ΔC_solver| = 0.000e+00`, cos = 1.000000, `502cb92` vs `0b378d7`). Rounding noise cannot do
  that — **the solver takes a different BRANCH, deterministically.** Strongest constraint available.
- **DIFFUSE, not one component** — answers `VERIFICATION_CAMPAIGN.md` §5.4's discriminating
  question, open since it was posed. `xyxy` is among the *smallest* deficits, so **the A-0
  contraction class is excluded** and §5.4's own reading sends this to the solve.
- **The oracle is bit-exact** across all dumps and both commits. B-1 is in the **solver** path —
  previously an assumption.
- **Always over-compliant**, and **only on the regular lattice** (3/3 of all excursions; 0/21 each
  on η=0.35 and ψ=0.6; uniform null p = 0.037). Thread-arm asymmetry is **not** significant (p = 0.214).

**Refuted — do not re-propose without new evidence:** dropped C2 (overshoots 3×); **rank truncation
at the `lstsq` cutoff** (no knife edge — `σ_min` is 1.5e-04 × cutoff, 13-order tail gap, `rcond`
swept 1e-16…1e-5 changes nothing); **ill-conditioning amplification** (the failing case is the
*best*-conditioned of the three).

**THE TRAP, and why those could only refute:** every one of those probes ran in a **clean isolated
process, where B-1 never occurs** (0 in 30 on record). They characterise the healthy state and infer
backwards. **The dumps record the wrong OUTPUT and nothing upstream** — no `G`, no singular values,
no effective rank. That is now fixed (`_b1_dump_if_anomalous` captures upstream state as of
2026-08-23), so **the next excursion should be diagnostic rather than another data point.**

### 1b. A PROPER CHORD TILING (small, before M2)
The tilings now use a **phantom-centre fan** (`seeds._fan_and_tag`, 2026-08-22) because the Delaunay
chord split produced **crossing edges** — 4 on `tiling_honeycomb_r3`, 5 on `_r4`, 2 on
`tiling_kagome_r2`, i.e. overlapping triangles rather than a mesh. The fan is correct by
construction and is the representation `test_hex_closed_form` already validates (4.4e-06).

**Still wanted:** a proper CHORD-based tiling — choose a non-crossing diagonal set per face instead
of adding a vertex. A chord tiling keeps the vertex set of the actual tiling (no phantom nodes,
no extra DOF), which matters for anything that reasons about the tiling's own coordination or feeds
node counts to M2. Small, self-contained, and the fan is a correct fallback until it exists.

*(`_reentrant_honeycomb` still uses the Delaunay chord split and is still tagged `mesh_ok=False` —
A-17's tail. Giving it the fan is the obvious follow-up.)*

### 2. M2 — the decisions are TAKEN; the work is staged S0…S5

**Live plan: `Phase 5/m2/M2_V2_PLAN.md`** (decisions **D1–D9 approved 2026-08-25**; §0 records each
one *together with the argument against it*, so they stop being re-litigated). The two questions this
section used to pose are answered:

- **What M2 IS (D1)** — *both, in order*: v2 is the **forward surrogate**; the **edit-policy is the
  endpoint** and comes after it, with the surrogate as its natural critic. They are different
  objects, so nothing is wasted.
- **What it trains against (D2)** — the **tensor**, not the derived curves. ν(θ),E(θ) are
  ratios/reciprocals of quartics in `C`, so predicting 74 numbers directly would permit profiles
  **no positive-definite `C` can produce**. Predict `C`; derive ν,E with the solver's own
  `c6_to_nuE` / `c6_to_nuE_theta`.

| stage | what | status |
|---|---|---|
| **S0** | `CLAUDE.md` wording; **pin `OMP/MKL_NUM_THREADS` + the tiling method in the builder** | wording ✅ (swept across *all* docs 2026-08-25); **thread + method pinning NOT DONE — the one S0 item still open** |
| **S0b** | dilution validity sweep, 266 cases | ✅ **`k_soft ≥ 1e-8` is safe at every `f ≤ 0.40`**, including sub-isostatic `z = 3.6`; below 1e-12 unusable. `results/dilution_validity/DILUTION_VALIDITY.md` |
| **S1 — MEASURED; VERDICT STILL PENDING (two runs in flight, 2026-09-06)** | head `C(s) = Q·MMᵀ·Qᵀ`; body is **v3 = TENSOR messages on TRIANGLE adjacency**, now with the **residual head** and the solver's **CONSTRAINT channels** (plan §2.4–2.5; the how-and-why is `documentation/GNN_GUIDE.md`) | see the block below |
| S2 | scaled, balanced dataset (~10 k), trajectories + provenance | after S1 — see §3 |
| S3 | 5-fold leave-one-family-out training | **must**: MAE(ν) ≤ 0.02 and MAE(E)/E ≤ 5 % **against the independent sim**; **kill criterion** MAE(ν) > 0.05 ⇒ stop and report, which is a legitimate result |
| S4 | wire in as an M1 pre-filter | solver calls per design reduced at unchanged design quality |
| S5 | **the edit-policy** — the actual endpoint of this arc | separate plan; trains on the trajectories S2 stores |

> ### S1 as of 2026-09-11 -- READ THE MEDIAN, NOT THE MEAN
>
> Full record: **`Phase 5/results/m2_s1/M2_RESIDUAL_AND_CONSTRAINTS.md`** (SS7/7b the trained
> results, **SS9 generalisation + error structure**). How the network works:
> **`documentation/GNN_GUIDE.md`**. Figures: `m2_{learning_curves,parity_nu,parity_E,by_family,
> by_wmax,error_cancellation}.png` in `Phase 5/results/m2_s1/`.
>
> **The architecture ladder is real and replicated** (`bravais` holdout, per-triangle MAE/sigma):
> v2 scalar 0.5111 (own-bulk oracle 0.4740 beat it) -> v3 tensor msgs 0.2040 -> residual head 0.2023
> -> **+`C_curv` 0.1209** -> **+`M_S` 0.0925**. Both constraint channels paid.
>
> **The TIER VERDICT is not decidable from the mean.** Two independent FRESH builds (seeds 4321,
> 777), frozen model, MAE(nu) vs the INDEPENDENT SIM, 400/family:
>
> | family | s4321 n=400 | s777 n=400 | **s4321 n=2000** | **s777 n=2000** | median (all 4) | verdict |
> |---|---|---|---|---|---|---|
> | `random` | 0.0256 | 0.0260 | 0.0280 | 0.0317 | 0.0099-0.0107 | passes all |
> | `bravais` | 0.0329 | 0.0396 | 0.0331 | 0.0383 | 0.0101-0.0109 | passes all |
> | `disordered` | 0.0583 | 0.0442 | 0.0568 | -- | 0.0133-0.0142 | **STRADDLES** |
> | `longrange` | 0.0750 | 0.1040 | 0.0820 | 0.1014 | 0.0121-0.0151 | fails all |
> | `cells` | 0.1613 | 0.2548 | 0.2042 | 0.1578 | 0.0451-0.0466 | fails all, worst |
> 
> **5x the sample only HALVED the swing** (58 % at n=400 -> ~24 % at n=2000), while the MEDIANS replicate to 0.0001-0.0014. The ranking is robust across two sample sizes and two seeds; the VALUES are not. A mean-based pass/fail at 0.05 is not a usable instrument here.
> 
> Heavy tail. **Every family's median
> is inside the kill line**; random's is inside the MUST tier. More than half of every family is
> accurate and the failure is ALL tail. **SPEC QUESTION FOR THE USER: should a mean-based kill
> criterion at n=400 remain the instrument?** Tuning architecture against it is tuning against noise.
>
> **Generalisation is FINE:** train 0.1197 vs fresh unseen meshes 0.1313 = **+9.7 %**.
>
> **Error structure:** `R = mean|err|/|mean err|` tracks **sqrt(N) at a constant 0.435** in every
> family => independent over blocks of ~5.3 TRIANGLES, **no systematic bias**. So `cells` fails
> simply because a 16-triangle mesh has almost no sqrt(N) averaging. A bulk loss term therefore
> suppresses no bias; it acts only by per-graph re-weighting, which `--graph_balance` does directly.
>
> **TWO CLAIMS WITHDRAWN -- do not re-derive:** (1) "n_corr ~ 5.3 matches the cluster radius 4-6" is
> a UNITS ERROR (count vs radius; 5.3 triangles ~ radius 1.3) and THEORY_NOTES may be stale;
> (2) "reach is the deficit, build multi-scale" is WITHDRAWN -- depth 5 gives 10 hops on
> 72-240-triangle meshes and the errors are correlated over ~1 hop, so the contrast trend does NOT
> distinguish reach from expressivity. Also `W=0` needs ORDERED geometry AND uniform k.
>
> ### NEXT STEPS, in order
>
> 1. **FINISH THE OVERFIT PROBE -- it ran but was STOPPED, and was MIS-SIZED.**
>    `train_v3.py --overfit_probe 200 --contrast_min 10000`, train == val, L5 vs L10.
>    Reached: **L5 ep 399 (the CAP, not the stopping rule) at MAE/sigma 0.2272 and STILL FALLING**
>    (0.4353 @ ep 125); **L10 ep 200 at 0.5258, STALLED** (lr collapsed to 2.43e-05).
>    - **Depth 10 stalls.** With the depth-8 arm having DIVERGED at lr 3e-3, that is two independent
>      signs that **depth past 5 is an OPTIMISATION problem here**, not a free lever.
>    - **Nothing can yet be concluded about reach vs expressivity** -- L5 had not converged.
>    - **THE SIZING ERROR, do not repeat it:** 200 samples at batch 32 is **7 steps/epoch**, so 400
>      epochs ~ 2 800 gradient steps against the real run's ~42 000. Size a capacity probe by
>      GRADIENT STEPS, not sample count. At ~0.1-0.3 s/sample this costs hours whatever the sample
>      size (fewer samples = fewer steps/epoch; more samples = longer epochs).
>    - **To finish:** raise `--epochs` far past 400 and run ONE arm at a time on all four threads;
>      L5 alone is cheaper and better-conditioned. Logs `Phase 5/results/m2_s1/probe_L{5,10}.log`.
> 2. **`--graph_balance` vs baseline -- UNTESTED and now the live hypothesis.** `--bulk_weight`
>    is **REFUTED** (preliminary, 15 ep, not converged): it cost 14 % on the per-triangle
>    metric and buys NOTHING on bulk (0.1018 vs baseline 0.1016) -- exactly as the
>    no-systematic-bias result predicts. `--graph_balance` attacks the OTHER effect, the
>    per-graph re-weighting (700-triangle meshes currently outweigh 16-triangle ones ~44:1),
>    which is the mechanism actually identified for the `cells` failure. Judge on per-family
>    MAE(nu) on FRESH meshes, `cells` in particular -- NOT on aggregate val, which the
>    variants deliberately re-weight.
> 3. **The `--w_max_cut` question.** All the failure is in the tail, and those networks are excluded
>    from TRAINING on a premise now measured FALSE (solver labels are sound at every |W|: max gap
>    4.97e-02 at |W| 1-3 falling to 7.04e-03 above 100, zero samples over 0.05). Training on them may
>    be the largest available gain and costs nothing architectural.
> 4. **Then the differentiable designer** -- gradient descent on (positions, k) through the frozen
>    GNN inside a solver-verified trust region. The GNN has no position-gradient asymmetry (positions
>    enter via `bond_R` -> `q_e`, all in torch), so it unlocks joint k+position design the exact
>    solver structurally cannot do (CLAUDE.md SS3 records positions need SPSA; FD #13). Topology stays
>    discrete and outside. **The user's framing: it is also another way to COMPARE solver vs GNN** --
>    descent actively hunts the surrogate's blind spots, a harder test than random-sample scoring.
>
> **NOT next:** more data (+9.7 % gap, nothing to buy); component-weighted loss (dropped -- fixed
> per-component weights cannot track per-network anisotropy; use an auxiliary loss on E itself if the
> bulk/balance work leaves E failing).

> ### ⚠ 2026-08-26 — TWO BUGS OF ONE CLASS, and the guard that now catches them
>
> Twice in one day the model "failed to learn" because it was asked for an output determined by a
> **per-sample factor its inputs cannot see**:
> 1. **unnormalised `Q`** — `C_pred` scaled as `lbar²` while the target and every input are
>    scale-invariant (`lbar` spans 0.752–1.600 across the dataset, ~4× in `C`);
> 2. **internal-vs-physical units** on a per-triangle target (per-crystal factor **18–220**).
>
> **Both passed every structural gate** — SPD, equivariance, intensivity, expressiveness — because
> those check the FORM of the head, not whether the target is REACHABLE. Both presented identically,
> as the model plateauing.
>
> **`train_v2.oracle_check` now runs before the first gradient step and RAISES**: on any `W = 0`
> sample it pushes the analytic `C(s) = A(s)` through the exact pipeline and demands machine
> precision. Negative-tested — reintroduce either bug and it fires.
>
> **Consequence: `Phase 5/results/m2_s1/M2_S1.md`'s training numbers are INVALID** (within-family
> 0.4613, leave-one-family-out 0.5343, and the "does not beat the bulk baseline" conclusion). They
> were measured with both bugs present. Its §1 parametrisation gates survive.
>
> **Rule that follows:** before training on any new target, push the ANALYTIC answer through the
> exact pipeline and demand machine precision. No structural gate substitutes for it.
>
> Two other things measured the same day, both correcting earlier claims of mine:
> * **a crystal is ONE sample, not one per triangle** — every triangle is equivalent by translation
>   and the primitive cell's two are inversion-related while `C` is even under inversion (4920
>   triangles over 50 crystals → 50 distinct values). Resolution must come from the (φ, ψ) grid.
> * **φ is NOT periodic** at fixed diagonal (ν runs 0 → +1/3 → 0 → −0.579 → −1.381 over [0,4]); the
>   earlier period-2 claim was an artefact of the Delaunay generator always taking the shorter
>   diagonal. What IS redundant is the diagonal flag: `(φ, a₁+a₂)` = `(φ+2, a₂−a₁)` to 0.00e+00.
> * **depth HURTS on a local target** — 2 layers 13× worse than 0 on the crystals. `n_layers` 4–5 is
>   now something to test, not assume.

> ### ⚠ 2026-08-27 — THREE THINGS FROM S1 NOT TO RE-LEARN
>
> 1. **The "0.5144 own-bulk baseline" is the TRAIN-split value; on the held-out split it is 0.4114.**
>    Consequences: v3's margin over it is 5 %, not 24 %, and **v2 was beaten by the trivial
>    per-network-mean predictor**. Always recompute model and baselines in ONE pass on ONE split —
>    `Phase 5/verifications/m2_v3_report.py` does exactly this and exists because of this error.
> 2. **MORE DATA IS NOT THE LEVER when train ≈ val.** The S1 model is underfit, so `dataset_large.npz`
>    stays on the shelf until depth opens a generalisation gap. This is *also* why the kill criterion,
>    though triggered, has not been invoked: it presupposes a trained representation.
> 3. **BIN BEFORE YOU CORRELATE.** Error rises 7× with `max|W|` (0.12 below 3, 0.86 above 200), yet
>    `corr(log₁₀ max|W|, |Δν|) = −0.045` — essentially zero, because the effect is a sparse tail on a
>    flat bulk. The correlation alone says the opposite of the truth. Same trap as the hexagon `rcond`
>    correlation in `CLAUDE.md` §3. *(Near-mechanisms carry only ~29 % of total error — filtering them
>    is a ~20 % effect, NOT the fix; over half the error is in ordinary networks with `max|W| < 3`.)*
>
> Also settled: **`SOLVER vs SIM = 0.0000` on all 250 scored networks** — the labels are clean, so all
> S1 error is the model's, and near-mechanism label corruption is REFUTED as the explanation. And the
> relative-E metric is now **floored** (`|E_p| + 0.05·median E_sim`, mirroring `ε_ν`) because `E_sim` is
> bimodal and drove the unfloored mean to 7648 % against a median of 8.86 %.

**The order is load-bearing: do NOT scale data before S1 passes.** S1 is deliberately tiny — an
architecture check with **analytic ground truth** (in the small-cell limit the non-affine correction
vanishes and `C(s) → A(s)`, so `MMᵀ` must come out diagonal with entries `k_e/4ℓ_e²`). It catches in
minutes what a 10 k build would otherwise expose only afterwards: extensive-vs-intensive pooling,
self-loop handling in minimal cells, and whether the head is parametrised correctly at all.

### 3. M2 — the dataset rebuild is **S2**, and comes only AFTER S1

The old labels are **verified stale** (37/41 drift, worst |Δν| = 1.73), so `m2/checkpoint.pt` is
unverified. `Phase 5/dataset/` and `m2/data/` are deliberately **ABSENT** so nothing can retrain on
them; the July set is archived at `validation_2026-08/attic/m2_dataset_2026-07_pre-A0/`. Record the
smoke-train metrics this time — `M2.md` still has `<FILL>` where the v1 numbers should be.

**But the rebuild is not the next action.** Building 10 k samples against an architecture that has
not passed its analytic gate is exactly what the staging exists to prevent.

What the rebuild must respect, all of it measured *since* this section was first written:

- **threads pinned and recorded per sample** (D4) — BLAS thread count changes a design outcome
  (ν −0.150 → −0.128, objective error 450×, same seed and commit);
- **dilution bounded at `k_soft ≥ 1e-8`** (S0b) — below 1e-12 the solver returns ν of the wrong
  *sign*, and no geometric health gate can detect it;
- **tiling representation recorded** (D3, default `fan`) — it changes node counts by ~50 %
  (honeycomb_r3: 54 nodes as a fan, 36 as chords), so switching it means rebuilding, not fine-tuning;
- **`phase4` seeds excluded** (D5) — they are non-periodic, wrapped heuristically, and would **leak
  across the leave-one-family-out split**;
- **the two leakage traps** — split by **trajectory id** (steps within one optimisation are
  near-duplicates), and **attribute every ingested design to its source family** so it is held out
  with that family;
- **balance, not size** — no family below ~10 %; `random` can be generated without limit while
  `tiling`, `basis` and `auxetic` are a handful of topologies each.

`Phase 5/results/reach_summary/REACH_SUMMARY.md` records the reachable envelope on one axis — what a
dataset must cover to be honest about the auxetic corners.

### 4. Deferred, deliberately
- **The tilings themselves are FIXED** (crossing chords → phantom-centre fan, see below). What is
  left as-is, by the user's call: **no run-time geometry guard**, and no re-measurement of whether
  the residual `tiling_honeycomb_r3` disagreement survives the fix.
- **`build_topologies` accepts meshes that `check_mesh_preconditions` rejects** — it only checks
  `areas > 0`. That is how an invalid mesh reached production. Deliberately not guarded.
- **LOW PRIORITY — the regulariser causal test, and the per-triangle regulariser it points at.**
  S0b showed the dilution validity boundary IS the solver's `eps = 1e-12 * A3.abs().max()`
  (`Phase 5/results/dilution_validity/DILUTION_VALIDITY.md` §5b): predicted `k_soft_crit = 2e-12`,
  measured jump between 1e-10 and 1e-12, on TWO bases whose different geometric prefactors (0.5 and
  0.093) each match their own prediction. The definitive test — change the constant and check the
  boundary moves proportionally — is **deliberately deferred** (user's call, 2026-08-25): the
  evidence is already triple-confirmed, the practical bound `k_soft ≥ 1e-8` has ~4 orders of margin
  and does not depend on it, and it edits the PROTECTED CORE, so it costs a full gate re-run to
  learn something we would act on identically.
  **The reason to keep it alive:** `eps` uses a **GLOBAL** max, so a soft triangle is regularised
  against the stiffest triangle in the whole mesh. A **per-triangle** relative regulariser
  (`eps_s = 1e-12 · |A(s)|max`) would scale with each triangle's own magnitude, not swamp the soft
  ones, and could push the validity boundary down by MANY ORDERS — a genuine widening of the
  solver's usable domain, not just a confirmation. That is the version worth doing, if any.
- goal2 over its full target range — its target set has never been scoped (directional/full-tensor).
- `A-8` open-boundary suite · `A-17` tail (centre-vertex re-representation) · `A-18` analytic-oracle
  generalisation · TODO 2.6 `open_stretch` guard.
- **FD #2 needs redesign, not re-aiming** — both proposed targets measured insufficient (below).

---

## What changed on 2026-08-21/22

**1. "Untrustworthy" is NOT "unreal".** `trustworthy` is
`gap = max_θ|Δν(θ)|/(|ν_sim(θ)|+0.05) + max_θ|ΔE(θ)|/|E_sim(θ)| < 0.05` — a **relative, PER-ANGLE
agreement test between two code paths**, not a physicality verdict. The +0.05 floor makes it a ~5%
relative test where |ν| ≫ 0.05 but an absolute |Δν| ≤ 0.0025 where |ν| ≪ 0.05, ~20× stricter near
zero. Tabulating only gap-passing rows **censors the extremes**.

> **Not a new observation.** `AUDIT_2026-08.md` already recorded for `g1_2_freed` that the optimiser
> "claims ν down to −0.761 over all 110 runs" while the trustworthy subset gives [+0.100, +0.340].
> Today added the *interpretation*: those designs are **49/49 sign-agreeing** between the two codes
> (median bulk |Δν| = 0.073), so they are not "a near-mechanism regime the sim rejects" as the audit
> concluded — they are designs both codes broadly agree about that fail a strict per-angle
> tolerance. And the fix below.

**2. Selection is SCORED, not vetoed** — in **both** `run_g1_2.design_one` and `designer.design()`:
`err + 0.5·gap`, with PHYSICALITY still a veto (non-SPD / non-finite / |ν|≥ν_max is a statement about
the network, audit A-5). `gap_tol` now only sets the reported `trustworthy` flag.
Measured cause of the old rule's damage: the **SPSA step size**. At `a = 0.15` the search barely
moves and returns a *target-independent* endpoint; at `a = 0.25` it reaches ν = −0.134 with a larger
gap, and the veto discarded it.

**3. g1_2 re-run: ALL TEN topologies reach auxetic ν.** `triangular` +0.038 → **−0.217**,
`flipped_tri_f8` −0.057 → −0.272, `tetrakis` −0.067 → −0.226, `rotating_squares` −0.259 → −0.398.
Deepest −0.436 (honeycomb, solver −0.488) at k ≡ 1. Trustworthy unchanged at 45/110 — the fix
recovered *reach*, not agreement.

**4. goal1: symmetric grid + repaired position budget → a conclusion overturned.** `NU_GRID` is now
13 points over [−0.95, +0.95]; `design_iso` now passes `spsa_a = 0.25` instead of falling through to
the library default 0.02 (travel per coordinate **0.202 → 2.53** lattice spacings, matching g1_2, at
the same step count and runtime). Result: **every contrast band reaches ν < 0**, including f = 0.99
at −0.336. So "forbid contrast and the design space collapses onto +1/3" was a **search artefact**.

> **Corrected physics:** the lever is **asymmetric**. **Contrast raises the CEILING** (+0.34 at
> f = 0.99 vs +0.90 at f = 0 — geometry cannot push ν above the uniform-lattice value). **Positions
> supply the FLOOR** (−0.29 … −0.34 at every band; −0.436 in g1_2 at k ≡ 1). At f = 0.99 the window
> is [−0.336, +0.338] — near-symmetric, ≈ ±1/3, *not* a point near +1/3.

Median |err| **0.198 → 0.0063**, success 33 % → 55 %. Cost: trustworthy 114 → 67 (bigger moves, more
disagreement). `quality_floor` also enabled at **1e-3** from its A/B — a degeneracy guard only;
**never raise it to buy agreement** (0.03+ multiplies median error ×17 and would destroy the −0.436
design).

**5. A(s) conditioning is not the whole story.** `rcond(A(s))` alone does **not** predict solver
error: on the hexagon closed form, driving `A(s)` to numerical rank-1 with soft spokes *improves*
accuracy by six orders (corr **+0.73**, best accuracy at worst conditioning). SOFT k is benign;
**DEAD** k and **slivers** are not — `CLAUDE.md` §3 now separates them. In `g1_2` (k ≡ 1, purely
geometric) the sliver route is confirmed hard (`quality_p05` corr −0.86, and the **tail** predicts
better than the worst triangle).

**6. A THIRD verification path already exists** — `test_hex_closed_form.py` is analytic and
independent of both solver and sim (4.4e-06). **A-18 generalises it; it is not the first.**

**7. The tilings had CROSSING CHORDS — fixed.** See the section below; every result involving a
`tiling` topology predates the fix.

**8. New instruments:** `plotting.plot_ranges` (reach intervals; pale = full reach, solid = agreeing
sub-range, filled/hollow markers), `Phase 5/results/reach_summary/` (every ν achieved, on one axis),
`conditioning_probe.py`, `hex_conditioning_check.py`, `g1_2_solver_recheck.py`,
`g1_2_triangular_start_probe.py`, `b1_reproduce.py threads`.

---

## RESOLVED 2026-08-22 (end of session): the tilings had CROSSING CHORDS

The user found it: the chord triangulation produced **edges that cross other edges**. Measured —
`tiling_honeycomb_r3` **4 crossings**, `_r4` **5**, `tiling_kagome_r2` **2**; all three already
tagged `mesh_ok=False` by A-17, and all three **still entered `goal1`'s design pool**, because
`build_topologies` only checks `areas > 0` and never calls `check_mesh_preconditions`. So the
worst-disagreement network in every goal1 run was an **invalid mesh**, not a subtle solver problem.

**Fixed** by `seeds._fan_and_tag`: a phantom centre vertex inside every non-triangular face, fanned
to its corners. A fan cannot cross — every added edge joins a face's own centre to its own corner.
All seven tilings now have **0 crossings and `mesh_ok=True`** (three were False). Euler V−E+F = 0
confirms the face traversal on the torus. `test_designer_surface` [5] used to *assert* the honeycomb
tiling must fail as not-closed; it now asserts it passes.

**Consequence:** every result involving a `tiling` topology predates the fix — `goal1` (all three
runs) and `g1_2` both draw from this pool. Node and bond counts changed (honeycomb_r3: 36→54 nodes,
110→162 bonds), so those runs are not comparable to future ones and should be re-run before any
tiling-based conclusion is trusted.

**Still open, and deliberately unguarded:** `build_topologies` accepts meshes that
`check_mesh_preconditions` rejects. The gate exists and is correct; nothing calls it at pool-build
time. No run-time guard was added (user's call) — but this is why an invalid mesh reached production.

## The residual discrepancy (probably explained by the above)

`tiling_honeycomb_r3` — a 36-node chord-triangulated honeycomb — is the worst solver-vs-sim
disagreement in every goal1 run: **0.311 → 0.323 → 0.708** as the position budget grew. It is **one
deterministic network recurring**, not three independent samples (the tiling pool is fixed and each
topology appears once per sweep) — an earlier version of this file overstated it.

The 08-18 instance was **healthy on every axis** (`rcond_min` 5e-03, shape quality 0.475 ≈ 20°,
`k_min/mean` 0.65), so neither §3 route explains it. The current 0.708 instance has **not** been
re-measured. The live hypothesis (the user's) is that **position optimisation degrades the
geometry**, hardest where k is constrained but the target still looks chaseable — which fits the
disagreement more than doubling when travel went 13×, and fits the worst case sitting in `medium`
rather than at extreme ν.

**Decisions taken:** the tilings are **legitimate networks** — chord triangulation with all edges at
designed k is a real spring network, just not the tiling it is named after. **Honouring `k0` is NOT
the fix** (soft chords carry the face's shear, so it would only make everything near-mechanism). A
centre-vertex fan would be a better *representation of a honeycomb* (what A-17's tail and the hexagon
gate use) but is not automatically a better network. **No run-time geometry guard is being added.**
Note `run_goal1`'s CSV records `kmin_avg` and `solver_sim_gap` but **no geometry health**, so this is
not answerable from the results without recomputing.

---

## Things not to re-learn the hard way

- **The project keeps re-deriving what it already knows**, because findings live in prose spread over
  a 1300-line audit and a dozen results docs rather than in the data. Three times in one session I
  proposed analyses that already existed (per-triangle localisation *with a ‖W‖ field*; the
  sliver-vs-gap correlations in `positions.tri_shape_quality`'s docstring; the `ab_quality_floor`
  trade-off). **Check `VERIFICATION_CAMPAIGN.md` and `verification_tools/README.md` first, and say
  what you found — including "nothing".** That rule is in `CLAUDE.md` §3.
  **FIXED 2026-08-25:** the rule used to point at a document that could not serve it —
  `VERIFICATION_CAMPAIGN.md` was a *campaign plan*, not an index, and 7 of 9 sampled analyses were
  absent from it. It now opens with **Part I, a real index**: one row per analysis script giving the
  question it answers, where the answer lives, and the headline number where one is on record.
- **Read reach over ALL runs, never the trustworthy subset.** This concealed a real result four
  separate times, including in `run_goal1`'s own printed summary, which would have reported
  "medium: [+0.15, +0.38]" and hidden the −0.377.
- **`ab_quality_floor` is a SHAPE floor, not a stiffness floor.** The mislabel was in three places
  (its results doc, FD #2, and the audit) — all now corrected.
- **A re-run must be closed under producer→consumer.** The 08-18 re-run ran the producers, never the
  plotters, so figures and docs quoted July numbers over August data.
- **Look at the rendered figure.** Six plotter defects this session were found by eye, none by an
  exit code — a dropped topology, missing legend entries, a legend covering data, a `$\nu$` title
  rendering as a newline, filled/hollow markers indistinguishable against their own bar, and bars
  1.7 slots wide after a scale went stale.
- **An edit whose assertion fails can still get committed** (`eb0b8b8` → corrected by `612e8cf`).
- **Wall-clock timings in logs may be fiction** — one design recorded 18266 s from a machine suspend.
- **ν < 0 is reached and sim-confirmed** in `auxetic_sweep` (to −0.6009), `g1_2` (49 designs, 49/49
  sign-agreeing, to −0.436) and goal1 (to −0.82 at f = 0, and ν < 0 at *every* contrast band).
