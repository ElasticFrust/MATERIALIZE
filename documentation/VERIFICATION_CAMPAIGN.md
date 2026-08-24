# The post-audit verification campaign — plan

> **Status:** draft for approval, 2026-08-16. Nothing here has been run.
> Predecessor: `documentation/AUDIT_2026-08.md` (register), fix order item **#8**.

---

## 1. Why this campaign exists

Every saved design in `Phase 5/networks/` and every number in `Phase 5/results/` was produced
**and verified** with instruments now known to have been wrong. Since 2026-08 the following were
found and fixed:

| defect | effect on the old campaigns |
|---|---|
| **A-0** shear-channel contraction | `C_xyxy` over-stiff by 29–90% wherever `W ≠ 0`; reported ν(θ) dipped to −0.30 where the truth was +0.13 |
| **A-10** ν,E convention | design path reported **ν_yx, E_y** (y only) while the oracle reported the direction-average — every anisotropic solver-vs-sim ν/E comparison mixed two different quantities (Δν up to 0.093, all of it artefact) |
| **A-9 / A-1** oracle independence | per-triangle and regional "independent" checks routed through the solver's own contraction — self-verification |
| **A-2** ranking metric | `target_err_sim` was an ABSOLUTE `max(|Δν|,|ΔE|)`; for E ≫ 1 it ignored ν, **and it ranked the designs** |
| **A-12** data loss | `run_g1_2.py` discarded 58 of 110 runs, filtered by the broken instrument |
| **A-11** survivorship | failed runs dropped, so success rates had a `N_ok/N_ok` denominator |

`Phase 5/PLAN.md` already states the conclusion: **re-run, don't re-analyse.** `goal2` and
`goal2_attempts` are explicitly INVALIDATED (39/39 and 98/102 designs worsened; mean target error
0.05 → 0.48 and 0.20 → 0.71); `g1_2` has 21 of 53 over tolerance; `goal1` is the most robust at
6/110.

**This campaign is the first measurement of the project's results with instruments that work.**

## 2. What must be true before starting

- [x] independent oracle at bulk, per-triangle and regional level (A-9), gated by `[7]`/`[8]`
- [x] ν,E convention consistent between design path and oracle (A-10) — agreement 1.4e-12
- [x] designer surface honest: relative channel-resolved error, both sides reported, physicality gate,
      survives unhealthy candidates (A-1…A-6)
- [x] drivers record failures and keep every run (A-11, A-12); health checks unified (A-13)
- [x] artifacts stamped with commit / dirty / seed (B-3)
- [x] `reg` explicit at every `optimize()` site (B-2)
- [x] **B-1 understood AND FIXED (2026-08-24)** — see §5. *The constraint it imposed on how results
      may be quoted is LIFTED for runs made after the fix.*

## 3. Scope — what gets re-run

### Stage 0 — re-verify the layers BELOW the designer first

Dependencies point inward (`Phase 5 → Phase 3 → Phase 2`), so the core and the design layer are
re-verified **before** any Phase 5 campaign starts. Lower priority than the campaigns themselves —
these layers have been under continuous gating all through the audit and are expected to pass — but
running them first means a Phase 5 result can never be blamed on an unverified layer underneath, and
it costs little.

| what | covers |
|---|---|
| `Phase 2/test_forward_solver.py` | 8 gates: crystal ν=1/3; foam; intrinsic gradient; autograd-vs-FD; large-N adjoint; legacy Woodbury; **[7]** bulk `C_eff` vs energy Hessian component-wise; **[8]** per-triangle + regional `C(s)` vs the independent oracle |
| ~~`Phase 2/verification_open_domain`~~ | **not runnable** — it is a directory of PNG outputs, not a check (corrected 2026-08-16 when Stage 0 was first run). Open-boundary verification remains incomplete: audit **A-8** |
| `Phase 3/test_inverse_design.py` | 16 tests incl. **[15]** solver-vs-oracle tensor + virial-vs-energy, and the strain/stress + homogeneity family |
| `Phase 3/verifications/verify_lattice.py` | the Phase 3 lattice check |
| `verification_tools/accuracy_vs_disorder.py` | 9 network families × η ≤ 0.42 — end-to-end solver-vs-oracle, the broadest single statement of solver accuracy we have |
| `Phase 5/verifications/sanity.py` | ν=1/3, E=2/√3 on **both** solver and sim — the gate that must be run before trusting anything |
| `Phase 5/verifications/test_designer_surface.py` | the A-2…A-6 designer surface (4/4) |

Record every number. These are also the **B-1 sample**: each `test_inverse_design` run is one draw
at the ~1-in-21 anomaly, and `[15]` now dumps automatically (§5). *Do not skip Stage 0 on the grounds
that "it passed yesterday" — that is precisely the reasoning B-1 punishes.*

Deliberately NOT in Stage 0: the superseded legacy island in `verification_tools/` (it does not run —
`verification_tools/README.md` §3), and **A-14**/contraction-isolation (parked).

### Stage 1+ — the Phase 5 campaigns

| campaign | what it is | old status | priority |
|---|---|---|---|
| `goal1` | ν-target sweep across topology classes and disorder bands, 110 runs | most robust (6/110 over tol) — still measured with A-0 + A-10 | **1** |
| `g1_2` | positions-only design at k=1, 10 topologies × ν grid, 110 runs | 21/53 over tol; **58 runs unrecoverable** (A-12) | **2** |
| `goal2` | full anisotropic tensor targets | **INVALIDATED** — leaned hardest on the broken component | **3** |
| `goal2_attempts` | as above, wider attempt set | **INVALIDATED** (98/102 worsened) | **4** |

Order is deliberate: `goal1` is the cheapest and the best understood, so it is the shakedown run —
if the pipeline is wrong, it shows up there before the expensive anisotropic work.

## 4. Method

**Per campaign:**
1. Run the driver unchanged except for the audit fixes already in place. **Record the commit and
   confirm the tree is clean before starting** — `save_network` stamps `dirty`, and a dirty stamp
   means the artifact is not reproducible from its commit alone.
2. Every run is saved, trustworthy or not (A-12). Untrusted designs land as `UNTRUSTED_*`.
3. Every failure is recorded with `status=FAILED:<Type>` (A-11), so the denominator is honest.
4. End each campaign with its results doc (`Phase 5/results/<campaign>/<CAMPAIGN>.md`) — *what,
   method, key numbers, limitations, figure links*, plus commit/seed. **This closes C-1**, which was
   deliberately deferred out of the audit rather than fabricated.

**Comparison against the old results** is by *re-derivation, not by reading old numbers*: for each
campaign report the new distribution, and separately the old-vs-new shift where a matched comparison
is possible. Old `target_err_sim` values are **not comparable** — A-2 redefined the metric.

**Figures** come from `plotting.py` primitives only, loading saved networks — never re-optimising.

## 5. B-1 — ✅ ROOT-CAUSED AND FIXED (2026-08-24)

> **READ THIS FIRST — the rest of this section is the historical record of the investigation and its
> constraints, kept because the reasoning is instructive. It is SUPERSEDED as a live constraint.**
>
> **Cause:** the intrinsic solve ends in `torch.linalg.lstsq(G, r)` on a `G = J3·PinvJt` that is
> **singular by construction** (redundant constraint rows; rank 671/672, cond ~3e16). About **1 in
> 532 solves**, its pivoting CPU driver returns a `Λ` that only partially applies the KKT correction:
> every input stays **bit-perfect** while the returned `W` violates `J3·W = 0` by fourteen orders
> (90 % of the error in `J3`'s row space) and `C_eff` comes out **~1 % over-compliant**. That single
> mechanism explains the entire signature recorded below — always over-compliant, diffuse,
> discrete/bit-identical across commits, invisible upstream.
>
> **Fixed** in the protected core (`_woodbury_solve_aw`) by a two-stage guard that tests residual
> orthogonality every solve, and on a trigger re-solves with the SVD driver — repairing only if the
> correction term moves *and* the orthogonality provably improves. Warns on repair. All five gates
> green; `b1_persistence.py` **0 excursions in 700** vs **1 in 700** pre-fix.
>
> **Rate:** 1 in 532, 95 % CI 1 in [280, 1012] (`b1_rate.py`), cross-checking the probe's independent
> 1-in-700 — two instruments on *different* observables, so essentially every mis-solve produced a
> detectable `C` error. Severity: repaired `|ΔW|` median 2.5, max 48.
>
> **Full account:** `Phase 3/verifications/b1_dumps/B1_OVERNIGHT.md` §4d (cause) and §4e (fix).
> Settled statement: `CLAUDE.md` §3.
>
> **Instruments (all `Phase 3/verifications/`) — which question each answers:**
>
> | script | question |
> |---|---|
> | `b1_persistence.py` | Is the bad state PERSISTENT or one-shot, and after which test? **Found the cause.** Probes `[15]`'s case 1 repeatedly inside one real suite process — ~1.5 s a draw vs ~30 min, ≈500× more draws per unit compute |
> | `b1_rate.py` | How often does the guard repair? Counts repairs over guarded solves, Wilson interval |
> | `b1_excursion_analysis.py` | What STRUCTURE do the captured excursions have? Diffuse-vs-one-component, rank test, conditioning — the refutation trail |
> | `b1_thread_local.py` | Does BLAS thread count change a design outcome? **Yes** — breaks the arm/parity confound |
> | `b1_overnight.py` | Full-suite repetition campaign (the EXPENSIVE route; superseded by `b1_persistence.py` for cause-hunting, still valid for rate-in-the-wild) |
> | `b1_reproduce.py` | Earlier cheap-harness attempts (`suitectx`, `threads`) — recorded as INSUFFICIENT context: 0 hits in 360 probes |
>
> **Consequence for §5.1–5.4 below:** point 1 ("no headline number may rest on a single run") and
> point 2's stronger form are **lifted for post-fix runs**; the guard now catches the failure they
> were guarding against. Point 3 (the campaign as diagnostic) is superseded by `b1_persistence.py`,
> which buys ~500× more draws per unit compute. Point 4's discriminating question **was answered**:
> the deviation is DIFFUSE, which excluded the A-0 contraction class and correctly sent the hunt to
> the solve. **Numbers from BEFORE 2026-08-24 still carry the old caveat.**

**Measured 2026-08-16:** `test_inverse_design` [15]'s solver-vs-oracle tensor comparison intermittently
reads **~1.4e-04** instead of its usual **1.6e-12** — rate **1 in 21 suite runs**. It requires suite
context (0 in 30 isolated processes) and is stochastic given that context. Cause **unknown**; ruled
out by measurement: the A-10 change, ill-conditioning (relative condition number 0.05–0.08 in k),
perturbation of `G` through the singular `lstsq` (1–3×, no amplification even at σ_min = 4.7e-17),
Delaunay flips, float32, caching, method switching, try/except fallbacks, the 500/600 threshold
window, and state from any of the 13 preceding tests.

**Consequences, which are binding on how this campaign reports:**
1. **No headline number may rest on a single run.** A solver quantity can be wrong by ~1e-4 roughly
   5% of the time. Quote distributions over the campaign's many runs, not one design's number.
2. ~~~1e-4 is far below every design tolerance in use (`gap_tol` 0.05), so **it does not threaten the
   campaign's conclusions** — it threatens single-measurement claims about solver *accuracy*.~~
   **WRONG — corrected 2026-08-17.** B-1 was caught outside the test suite returning **ν = +2.76886
   where the truth is +2.950455: a 6 % error**, well ABOVE `gap_tol` and indistinguishable from a real
   result. It does hit ordinary solver calls, and it can threaten conclusions. The binding rule is
   therefore stronger: **no solver number is trustworthy without repetition OR an independent-sim
   cross-check.** The sim is what caught this instance.
3. **The campaign is the diagnostic.** `[15]` now dumps the full per-case tensors to
   `Phase 3/verifications/b1_dumps/` whenever the comparison exceeds 1e-9 — free when healthy. Running
   the suite before/after each campaign yields the sample that dedicated runs would otherwise have to
   buy (~7 h for one expected catch).
4. **The discriminating question is already framed:** is the deviation ONE tensor component or
   diffuse? A single component — especially shear-shear — points at the contraction (the A-0 channel).
   Diffuse points at the solve. The dump records exactly this; `[15]` previously reported only
   max-over-cases, which is why it was never known.

## 6. Cost and staging

Each campaign is hours of compute. **Staged, with a checkpoint between each** — nothing runs
unattended for a whole campaign without a look at the previous one:

0. **Stage 0 — re-verify Phase 2 and Phase 3** (§3). Fast relative to the campaigns; establishes that
   the layers underneath are sound before any Phase 5 number is produced.
1. **Shakedown:** `goal1`, small `n_runs`, verify the pipeline end to end, inspect the results doc.
2. Full `goal1` → checkpoint.
3. `g1_2` → checkpoint.
4. `goal2`, `goal2_attempts` → checkpoint.
5. Cross-campaign synthesis + the re-derivation of the mean-field ≈1.4× figure under the unweighted
   homogenisation (**A-7** — the old number came through the tombstoned `Ceff_nuE` and must not be
   re-cited until this is done).

**Gates before each stage** (the fast subset of Stage 0, re-run every time — they are also fresh B-1
draws): `Phase 2/test_forward_solver.py` 8/8, `Phase 3/test_inverse_design.py` 16/16,
`Phase 5/verifications/sanity.py` exact (ν=1/3, E=2/√3 on both paths),
`test_designer_surface.py` 4/4.

**Stop conditions:** sanity.py moving at all; a gate failing; a `[15]` dump showing a deviation that
is NOT the known ~1e-4 signature; or the re-run reproducing the old (pre-fix) numbers, which would
mean the fixes are not actually in the executed path.

## 7. What this campaign does NOT do

- It does not retire the oracle. That happens only once the solver is trusted (`CLAUDE.md` §1).
- It does not touch residual stress / incompatible ḡ / curved geometry — the sim assembles from a
  stress-free ḡ = I and is **not** a ground truth there (FUTURE_DIRECTIONS #1).
- It does not resolve **A-8** (open-boundary suite incomplete) or **A-14**/metric-A (parked).
- It is not a substitute for understanding B-1; it is the cheapest way to gather evidence on it.
