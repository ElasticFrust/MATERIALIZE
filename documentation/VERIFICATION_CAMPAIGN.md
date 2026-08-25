# Verification — the INDEX of what has been measured, and the campaign plan

> **Part I is the INDEX** — one line per analysis script saying **which question it answers** and
> **where the answer lives**. `CLAUDE.md` §3 makes reading it mandatory *before proposing a new
> measurement*, and requires you to **say what you found, including "nothing"**. Companion indexes:
> `verification_tools/README.md` (what in that directory is live, legacy or dead) and the results
> docs themselves (`Phase 5/results/*/`, `verification_tools/plots/*/`).
>
> **Part II is the post-audit campaign plan** (drafted 2026-08-16), kept below unchanged.
>
> *Index added 2026-08-25. Until then this file was the campaign plan only, while `CLAUDE.md` §0/§3
> and `NEXT_SESSION.md` already described it as "the index of what has already been measured" — so
> the anti-re-derivation rule pointed at a document that could not serve it. That is the drift this
> section closes; three analyses had already been re-proposed after having been done.*

---

# PART I — THE INDEX

**How to use it.** Find the row whose *question* is closest to yours. If one exists, read its answer
before proposing work. The **question** column is what the script's own module docstring says it
asks; the **answer** column is where its output and results doc live. Headline numbers appear only
where they are quoted from a results doc or from `CLAUDE.md` §3 — an empty headline means *the
answer exists, go read it*, not *nothing was found*.

**Not in this index:** the gates (`Phase 2/test_forward_solver.py`, `Phase 3/test_inverse_design.py`,
`Phase 5/verifications/{test_designer_surface,test_hex_closed_form,sanity}.py`) — those are pass/fail
regressions, listed in `CLAUDE.md` §3 "Regressions (the full gate set)"; and the **dead legacy
island**, listed in §I.8 below precisely so it is not mistaken for evidence.

## I.1 — Is the solver right? (solver vs the INDEPENDENT oracle)

| script | the question it answers | answer lives in |
|---|---|---|
| `verification_tools/accuracy_vs_disorder.py` | How well does the solver do on DISORDERED networks? — tensor-level accuracy vs η | `plots/accuracy_vs_disorder/ACCURACY_VS_DISORDER.md` — **the broadest single statement of solver accuracy we have** (9 families × η ≤ 0.42) |
| `verification_tools/per_triangle_C_comparison.py` | Per-triangle `C(s)`: the solver's contraction vs the independent energy-Hessian oracle — visualised | `plots/per_triangle_C/PER_TRIANGLE_C.md` — **records corr(&#124;ΔC&#124;, ‖W‖) = +0.39**, and that an earlier draft overstating it as "tracks ‖W‖" was corrected |
| `verification_tools/recheck_sweep_nu_E_eta.py` | Re-check ν(η) and E(η) — disordered networks and VD rigidity contrasts — against stored results | `plots/recheck_nu_E_eta/RECHECK_NU_E_ETA.md` |
| `verification_tools/verify_solver_sweep.py` | Complete η sweep of the production forward solver vs simulation | `plots/dg_solver_sweep_*.npz` — ⚠ **stale artifact**: internal-unit E (low by 18.4752) and area-weighted ν (`ARCHITECTURE.md` §5) |
| `verification_tools/verify_solver_final.py` | 10 (network structure × rigidity-distribution) combinations — solver vs sim | `plots/dg_solver_final_10networks.npz` (the `.png` is **not in the tree** — re-run to regenerate) |
| `verification_tools/verify_irregular_VD_eta.py` | Sim-vs-solver on IRREGULAR (disordered) networks, in PHYSICAL units | `plots/dg_irregular_VD_eta_*.png` |
| `verification_tools/verify_solver_open.py` | OPEN-DOMAIN (non-periodic) `method='intrinsic'` vs simulation | script output — open-boundary coverage is **incomplete, audit A-8** |
| `verification_tools/verify_soft_region.py` | What happens when the six bonds around one vertex are made soft? | declared output `plots/dg_soft_region_per_triangle.png` — **artifact not in the tree; re-run to regenerate** |
| `verification_tools/verify_soft_circles_50x50{,_directions}.py` | Two circular soft regions, three network structures — per-triangle response, and in the OTHER directions | declared output `plots/dg_soft_circles_50x50*.png` — **artifacts not in the tree; re-run to regenerate** |
| `Phase 5/verifications/hex_solver_validation.py`, `hex_response_plot.py`, `hex_nu_linear.py` | Where does the solver work? — the hexagon diameter family against the **CLOSED FORM**, both triangulations, both directional responses | `Phase 5/results/hex_validation/HEX_VALIDATION.md` — **the third path**, independent of solver *and* sim, matched to 4.4e-06 |
| `Phase 5/verifications/single_hexagon.py` | The solver on the smallest meaningful case (one hexagon, non-periodic, six triangles) | script output |

## I.2 — Where does the solver STOP being right? (validity boundaries)

| script | the question it answers | answer lives in |
|---|---|---|
| `Phase 5/verifications/dilution_validity.py` | Where does BOND DILUTION break the solver? (M2 decision D8) | `results/dilution_validity/DILUTION_VALIDITY.md` — **the boundary is in SOFTNESS, not dilution fraction**: `k_soft ≥ 1e-8` safe at every `f ≤ 0.40` incl. sub-isostatic `z = 3.6`; `k_soft ≤ 1e-12` breaks at the first nonzero `f`; **6 of 266 cases returned ν of the opposite sign** |
| `Phase 5/verifications/dilution_regulariser_check.py` | Is that boundary just the solver's REGULARISER? | same doc §5b — **yes, quantitatively**: `eps = 1e-12·max&#124;A3&#124;` predicts `k_soft_crit = 2e-12`; `λ_min ≈ 0.5·k_soft·max&#124;A&#124;` over 8 decades; two bases with *different* geometric prefactors each match their own prediction |
| `Phase 5/verifications/dilution_networks_figure.py` | What do the broken networks LOOK like? | same doc §3 — **structurally indistinguishable from the safe ones; no geometric health gate can catch it** |
| `Phase 5/verifications/hex_conditioning_check.py` | Does a rank-deficient `A(s)` actually mean a wrong answer? | `results/hex_validation/` + `CLAUDE.md` §3 — **no**: driving `A(s)` to numerical rank-1 with soft spokes *improves* accuracy by six orders; corr(log₁₀ rcond, log₁₀&#124;Δν&#124;) = **+0.73** |
| `Phase 5/verifications/conditioning_probe.py` | Does `A(s)` conditioning explain where solver and sim disagree? | `results/conditioning_probe/CONDITIONING_PROBE.md` — SOFT k is benign; **DEAD k and SLIVERS are not** (`quality_p05` corr −0.86 in g1_2, and the *tail* predicts better than the worst triangle) |
| `verification_tools/exact_kinematics_check.py` | Does linearising the kinematics cost anything? | script output + `CLAUDE.md` §3 — **nothing for `C_eff` at an unstressed reference** (models 1 vs 3 differ by O(h)); it bites only under **prestress** (audit A-15) |
| `verification_tools/finite_amplitude_check.py` | What does the solver's constitutive approximation cost at finite amplitude? | `plots/finite_amplitude/FINITE_AMPLITUDE.md` |

## I.3 — Is the verification APPARATUS itself sound?

| script | the question it answers | answer lives in |
|---|---|---|
| `Phase 5/verifications/blast_radius_shear_fix.py` | Blast radius of the 2026-08 shear-channel core fix over every SAVED design | `results/shear_fix/SHEAR_FIX.md` — the per-campaign invalidation table |
| `Phase 3/verifications/relayer_a7b.py` | Did the A-7b re-layering change any number? (before/after equivalence) | `relayer_a7b_out/RELAYER_A7B.md` |
| `Phase 5/verifications/ab_quality_floor.py` | Does a SHAPE-QUALITY floor on the position search buy trustworthiness? | `results/ab_quality_floor/AB_QUALITY_FLOOR.md` — it is a **shape** floor, not a stiffness floor; `quality_floor = 1e-3`, and **raising it to buy agreement multiplies median error ×17** |
| `Phase 5/verifications/campaign_shakedown.py` | Does a campaign driver run end-to-end on a tiny subset, without modifying it? | script output (pipeline check, not a physics result) |
| `verification_tools/compat_projection.py` | Is the mean-field's residual error just INCOMPATIBILITY? | `plots/` — needs `dg_analysis_data/` **regenerated first** (`pbc_dg_analysis.py`) |
| `verification_tools/test_mean_isolation.py` | Is the mean/normalisation constraint THE failure of the metric-space solve? | script output; the settled answer is that C3 must be **area-weighted** (`CLAUDE.md` §3) |
| `verification_tools/test_curvature_operator.py`, `test_angle_response.py` | Does the discrete curvature (C2 / St-Venant) operator do what the derivation says? | declared output `plots/dg_curvature_operator.png` — **not in the tree**; and both scripts need `dg_analysis_data/` regenerated first (`pbc_dg_analysis.py`) |
| `verification_tools/pbc_dg_analysis.py` | Per-triangle non-affine Δg under a single macroscopic strain | **the producer of `dg_analysis_data/`, RETIRED 2026-08-18** — regenerable; inventory in `validation_2026-08/retired_dg_analysis_data.txt` |

## I.4 — B-1: the intermittent KKT mis-solve (ROOT-CAUSED AND FIXED 2026-08-24)

Full account `Phase 3/verifications/b1_dumps/B1_OVERNIGHT.md` §4d (cause) / §4e (fix); settled
statement `CLAUDE.md` §3. **Do not re-propose the refuted mechanisms** — dropped C2, `lstsq` rank
truncation, ill-conditioning amplification — without new evidence.

| script | the question it answers | answer lives in |
|---|---|---|
| `b1_persistence.py` | Is the bad state PERSISTENT or one-shot, and after which test? | **found the cause** — ~1.5 s a draw vs ~30 min, ≈500× more draws per unit compute; **0 excursions in 700 post-fix vs 1 in 700 pre-fix** |
| `b1_rate.py` | How often does the guard repair? | **1 in 532 solves, 95 % CI 1 in [280, 1012]**; repaired &#124;ΔW&#124; median 2.5, max 48 |
| `b1_excursion_analysis.py` | What STRUCTURE do the captured excursions have? | **DIFFUSE, not one component** — which excluded the A-0 contraction class and correctly sent the hunt to the solve |
| `b1_thread_local.py` | Does BLAS THREAD COUNT change a design outcome? | **YES** — ν −0.150 → −0.128, objective error 450×, same seed and commit ⇒ **M2's generator must pin threads** (decision D4) |
| `b1_overnight.py` | Full-suite repetition campaign (rate in the wild) | `b1_dumps/overnight_<UTC>.{csv,log}` — the EXPENSIVE route; superseded for cause-hunting by `b1_persistence.py` |
| `b1_reproduce.py` | Can a cheap harness reproduce it? | **no** — 0 hits in 360 probes; recorded as INSUFFICIENT context |

## I.5 — What can the DESIGNER actually reach?

> **Read reach over ALL runs, never the trustworthy subset** — that filter concealed a real result
> four separate times. `trustworthy` is a per-angle two-code *agreement* test, not a physicality
> verdict.

| script | the question it answers | answer lives in |
|---|---|---|
| `Phase 5/verifications/run_goal1.py` (+ `plot_goal1.py`) | ν-target sweep across topology classes and disorder bands (110 runs) | `results/goal1/GOAL1.md` — **every contrast band reaches ν < 0**; contrast raises the CEILING, positions supply the FLOOR |
| `Phase 5/verifications/run_goal1_frontier.py` | The POSITIVE-ν frontier probe | `results/goal1_frontier/GOAL1_FRONTIER.md` |
| `Phase 5/verifications/run_g1_2.py` (+ `plot_g1_2.py`) | Isotropic ν by GEOMETRIC DISTORTION ALONE (k ≡ 1) | `results/g1_2/G1_2.md` — **all ten topologies reach auxetic ν**; deepest −0.436 sim / −0.488 solver |
| `Phase 5/verifications/run_g1_2_freed.py` | The same with the fictional bracing edges FREED | `results/g1_2_freed/G1_2_FREED.md` |
| `Phase 5/verifications/g1_2_solver_recheck.py` | Can the SOLVER side of every design be recovered, so untrustworthy runs can be plotted? | `results/g1_2/` — **49/49 sign-agreeing** between the two codes |
| `Phase 5/verifications/g1_2_triangular_start_probe.py` | Can the position optimiser reach what random η-disorder reaches? | `results/g1_2/` |
| `Phase 5/verifications/run_goal2.py`, `run_goal2_attempts.py` | DIRECTIONAL / full-tensor targets, and the failed attempts | `results/goal2/GOAL2.md`, `results/goal2_attempts_rerun/GOAL2.md` — its **target set has never been scoped**; deliberately deferred |
| `Phase 5/verifications/run_anisotropic{,4,4_pos}.py` | A highly anisotropic ν(θ); a REALIZABLE cos4θ version; does moving nodes close the gap? | `Phase 5/aniso*_response.png`, `*_run.log` — positions help the loss but **not the anisotropy-amplitude ceiling** (an empirical topology/size bound, not a harmonic limit) |
| `Phase 5/verifications/verify_positions.py` | Does vertex-position optimisation actually help? | script output + `Phase 5/networks/` |
| `Phase 5/verifications/plot_reach_summary.py` | **What Poisson ratio has this project ACTUALLY achieved?** — one figure over every experiment | `results/reach_summary/REACH_SUMMARY.md` |
| `Phase 3/verifications/` cases — `auxetic_sweep`, `auxetic_patch`, `graded_nu`, `anisotropy`, `two_region`, `strain_stress`, `regimes`, `large16k`, `dir_aux_ribbon`, `reference_metric` | Does design → realise → simulate reproduce each class of target? | `Phase 3/verifications/README.md` §Cases, and each case directory |

## I.6 — Representation & mesh questions

| script | the question it answers | answer lives in |
|---|---|---|
| `Phase 5/verifications/chord_vs_fan_figure.py` | Phantom-centre FAN vs non-crossing CHORDS on the same tiling | `results/chord_vs_fan/CHORD_VS_FAN.md` — the Delaunay chord split produced **crossing edges**; a fan cannot cross, by construction |
| *(measurement, no script)* | How many crossings did each tiling have? | `results/tiling_inspect/TILING_INSPECT.md` — honeycomb_r3 **4**, _r4 **5**, kagome_r2 **2**. **Every `tiling` result predates the fix** and must be re-run before it is trusted |
| `Phase 5/verifications/dhex_family.py`, `reentrant_family.py`, `reentrant_move.py` | The re-entrant honeycomb by the DIAMETER algorithm, as a rib skeleton, and built the CORRECT way (fixed topology, move vertices) | `results/reentrant/REENTRANT.md` |
| `Phase 3/verifications/topologies_overview.py` | What do all the topologies used across the cases look like? | `Phase 3/verifications/topologies_overview.png` |
| `Phase 3/verifications/verify_lattice.py` | Does `make_lattice` geometry work end-to-end, before any plotting? | gate; script output |

## I.7 — M2 premises (the GNN)

| script | the question it answers | answer lives in |
|---|---|---|
| `Phase 5/verifications/supercell_invariance.py` | Is `C_eff` really INTENSIVE across supercells? — the premise under "train small, deploy large" and under the label-free supercell gate | `results/supercell_invariance/SUPERCELL_INVARIANCE.md` — **yes, to 5e-16 … 7e-13** on four chains including the sharp anisotropic one; independently confirms that v1's `sum`-pooling branch is unphysical |
| *(see I.2 — dilution)* | What may M2 safely sample along the coordination axis? | `k_soft ≥ 1e-8`, full `f ∈ [0, 0.40]` usable |

## I.8 — DEAD: the legacy island (do NOT cite as evidence)

`verification_tools/test_cluster_{Ceff,VD,rigidity,Ceff_rigidity}.py` and
`test_intrinsic_{metric,VD}.py` — **these do not execute.** They call `Ceff_nuE`, the legacy
**area-weighted** metric average, tombstoned 2026-08-10 as physically wrong; it raises
`NotImplementedError`, so any path reaching it dies on first call. Full status:
`verification_tools/README.md` §3. Their *conclusions* have since been re-established far more
strongly and independently, but **numbers computed through `Ceff_nuE` are stale** — in particular the
"single-site mean-field ≈1.4× over-compliance" figure, which **must not be re-cited until it is
re-derived** under the unweighted homogenisation (audit A-7; queued as Part II §6 stage 5).

---

# PART II — THE POST-AUDIT CAMPAIGN PLAN

## The post-audit verification campaign — plan (drafted 2026-08-16)

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
