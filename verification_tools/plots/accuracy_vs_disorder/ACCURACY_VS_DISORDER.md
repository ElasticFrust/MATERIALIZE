# How the calculations perform on DISORDERED networks

**What.** Tensor-level accuracy of the homogenisation across the full disorder range, on nine
network families — because the crystal (η=0) is the degenerate case (W ≡ 0, everything agrees to
round-off) and says nothing about the machinery. `test_forward_solver` [7]/[8] check only 2–3 η
values; this resolves the whole curve.

**Method.** `verification_tools/accuracy_vs_disorder.py`, N=8 (128 tri), η ∈ [0, 0.5] step 0.02,
3 seeds, 9 families. Everything is compared **component-wise at the tensor level**, never through
reduced ν,E — so none of it is affected by the open A-10 convention question. Commit after `e092d72`.

Families: frozen-connectivity magnitude-η with uniform k (the canonical disorder); the same under
five VD contrasts `k = 1 + tanh(a(|R|−1))`, a ∈ {−10, −2, +5, +10, +100}; random binary k (×10)
— rigidity disorder *uncorrelated* with geometry, which VD is not; re-triangulated η (the other
disorder intent — a topology scan); and an anisotropic base lattice (ψ=0.6).

---

> **What this residual is NOT.** It is *not* solver-vs-sim disagreement.
>
> All comparisons here compute the SAME object two ways: C(s) = (1+W)ᵀA(s)(1+W), via the SAME
> `_compute_actual_elastic_tensor`, with the SAME A(s). They differ only in **where W comes from**,
> and are scored against the same reference — `energy_C_per_triangle`, the Hessian ∂²U_s/∂g² of
> triangle s's own relaxed energy. Measured N=10, η=0.30, VD a=+5:
>
> | source of W | per-triangle | bulk |
> |---|---|---|
> | sim's relaxed **u** → per-triangle F_s → Δg = FᵀF − I (**exact** strain) — *this is what [7]/[8] use* | 1.05e-02 | 5.37e-03 |
> | the same **u**, same F_s, but Δg = (F−I)+(F−I)ᵀ (**linearised** strain) | **5.47e-13** | **2.33e-13** |
> | the **solver's own** W — the intrinsic constrained solve over δg. **No displacement field is involved at all** | **7.54e-10** | 5.45e-12 |
>
> Rows 1–2 share a displacement field; **row 3 does not have one** — the solver works in metric
> space (W is (N,9), per-triangle, no nodal DOF). The two strain measures in rows 1–2 differ by
> 5.84e-03 at DELTA=1e-3, which accounts for row 1 exactly and scales linearly with DELTA. So the
> ~1e-2 figures below measure a strain-measure mismatch at finite probe amplitude — a property of
> the isolation construction, chosen to make the check sensitive to the CONTRACTION. Row 3 is the
> genuine cross-formulation check (metric-space constrained solve vs nodal relaxation) and is ~8
> orders tighter. See audit A-15.


## 1. Headline

| quantity (mean over seeds), frozen η family | η=0 | η=0.2 | η=0.4 | η=0.5 |
|---|---|---|---|---|
| **oracle control** — virial vs energy, *no solver* | 1.1e-13 | 1.1e-13 | 1.1e-13 | 1.1e-13 |
| **END-TO-END bulk** — solver's own `forward()` vs oracle | 2.2e-13 | 5.8e-14 | 4.6e-12 | 3.8e-02 |
| **END-TO-END per-triangle** — solver's own `forward()` vs oracle | 2.2e-13 | 8.4e-12 | 7.5e-10 | 4.1e-01 |
| contraction-isolation, bulk (gated by [7]) | 2.2e-13 | ~9e-4 | ~1e-3 | ~1e-3 |
| contraction-isolation, per-triangle (gated by [8]) | 2.2e-13 | ~3e-3 | ~7e-3 | ~1.2e-2 |
| isolation, per-triangle **with the A-0 defect** (reference) | 4.4e-13 | ~1e0 | ~1e0 | ~2e0 |

**Read the first three rows for "is the solver right"; the last three only for "is the contraction
right".** They are different questions and they differ by up to 8 orders (§2).

- **The oracle does not drift.** The control is flat at 1.1e-13 across every family and every η, so
  growth elsewhere is never the yardstick.
- **End-to-end, the solver is essentially exact out to η=0.4** — 1e-13 … 1e-9 in *all nine
  families*, per-triangle as well as bulk. Far tighter than the isolation rows, which carry an
  O(DELTA) strain-measure artefact (see the box above).
- **The one genuine end-to-end failure is η=0.5**, and it splits by *constructor*, not by physics —
  §3.
- **The A-0 defect sits 2–3 orders above the correct isolation curve at every η > 0** and is
  *indistinguishable* at η = 0 — the blindness `CLAUDE.md` §3 warns about, shown directly.

## 2. The stiffness trend belongs to the ISOLATION metric — **not** to the solver

> **Correction (2026-08-15).** An earlier revision of this section concluded that accuracy is
> governed by proximity to a mechanism, and set a practical criterion "trust the per-triangle
> read-back while E ≳ 1e-2". **That was measured on the isolation metric only, and it is wrong as a
> statement about the solver.** The end-to-end rows, added afterwards, refute it. Superseded text
> kept out; the corrected result follows.

Plotting the **isolation** error against bulk E, all nine families do collapse onto one trend
spanning 8 decades — that part reproduces. But adding the solver's **own** `forward()` shows the
trend is a property of *how the isolation metric is built*, not of the solver. VD a=+100, the most
extreme family:

| η | 0.0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 |
|---|---|---|---|---|---|---|
| bulk E | 1.15e+00 | 1.11e-02 | 1.36e-03 | 2.57e-04 | 4.35e-04 | 4.99e-03 |
| **isolation** per-triangle | 3.2e-13 | 4.3e-02 | 2.8e+00 | 4.6e+02 | 1.8e+04 | 4.8e+05 |
| **END-TO-END** per-triangle | 2.2e-13 | 3.2e-05 | 3.2e-05 | 5.0e-05 | 1.2e-04 | 8.8e-01 |

E falls four orders between η=0 and η=0.3; the isolation metric climbs 15 orders; **the end-to-end
error stays flat at ~5e-05.** At E = 2.6e-04 — two decades below the old "E ≳ 1e-2" line — the
solver's own prediction still matches the independent oracle to five digits.

**Root cause of the isolation trend.** `W = D·D⁻¹` is built by pushing the **sim's** displacement
`u` (from a *geometrically linear* solve) through `MO.tri_metric_change`, which is the
*geometrically exact* Δg = FᵀF − ḡ. The mismatch is O(|u|²). Softening the network grows |u|, so the
artefact grows with it — it measures a **kinematic mismatch between two strain measures**, not solver
error. This is the same artefact the box at the top of this document quantifies (row 1 vs row 2:
1.05e-02 vs 5.47e-13), now shown to be the whole content of the "stiffness trend".

**Consequence for the gates.** [7] and [8] are built on the isolation construction, so they inherit
this artefact: part of what they measure is probe amplitude, not correctness. That is the root cause
under audit **A-14**, and the fix is to extract W through ε rather than Δg (noise floor ~1e-2 →
~5e-13). **The near-mechanism caution in `CLAUDE.md` §3 is not thereby retracted** — it is a
statement about the *true nonlinear* response diverging from the linear one, which this sweep does
not measure (both sides here are linear). What is retracted is the claim that *this measurement*
demonstrated it.

## 3. The one real end-to-end failure: η=0.5, and it splits by constructor

End-to-end agreement is 1e-13 … 1e-9 for **η ≤ 0.4 in every family**. At **η = 0.5** it splits
perfectly along which builder made the mesh:

| family | end-to-end per-triangle at η=0.5 |
|---|---|
| frozen η uniform k; VD a ∈ {−10,−2,+5,+10,+100}; random binary k — all `build_geometry` | **1.6e-01 … 8.8e-01** |
| re-triangulated η; anisotropic ψ=0.6 — both `make_lattice` | **2.3e-09 / 4.0e-09** |

η<0.5 is the documented domain for frozen-connectivity disorder, and `make_lattice` re-triangulates,
so η=0.5 is not a singular point for it — consistent with the split. **The root cause is OPEN.** Two
candidates were tested and **refuted**:

- **slivers / mesh shape** — on frozen η=0.5, corr(log err, log shape-quality) = **−0.15**; the
  worst-10 triangles have median error 1.27e-01 against 8.74e-02 for the best-10 (same order), and
  the single worst triangle has quality 0.80. The error is spread **globally**, not on bad triangles.
- **stiffness / near-mechanism** — frozen η=0.5 has bulk **E = 0.546**, a perfectly stiff material,
  and still fails at 4.1e-01.

No mechanism should be asserted for this until one is measured.

## 4. What this says about the gate's tolerance

[8]'s 2e-2 is calibrated on **its own case list** — frozen-connectivity meshes up to η=0.35, where
the worst is 7.7e-3. It is **not** a universal validity claim. This sweep finds families that exceed
it at large η while being perfectly healthy numerically:

- VD a=−10 crosses ~2e-2 around η ≈ 0.42;
- anisotropic base ψ=0.6 reaches ~2.5e-2 near η ≈ 0.36–0.42;
- re-triangulated η reaches ~1.5e-2 by η=0.5.

None of these are in [8]'s case list, so the gate does not currently see them. That is a **coverage
observation, not a failure** — but it means "the solver agrees to within 2e-2" should not be stated
as a general property. Either the gate's cases should be widened or the claim should be qualified;
logged for the audit register rather than silently fixed here.

## 5. Limitations

- One size (N=8, 128 triangles) and 3 seeds — no size-convergence study; the min–max bands are from
  3 realisations only.
- Periodic cells only (open-boundary per-triangle remains audit **A-8**).
- Both metrics compare **linear** predictions to a **linear** oracle, so nothing here bounds the
  *finite-amplitude* error. That is measured separately by
  `verification_tools/finite_amplitude_check.py`: at 10% strain the solver's energy differs from real
  Hookean springs by ~5% (uniaxial) / ~0.7% (shear), **first order in strain**, with the constitutive
  and response-linearisation errors partially cancelling.
- At η ≥ ~0.48 some realisations are rejected by the sim's health gate (sliver triangles), so the
  far-right points average over fewer seeds — the shaded band in the per-family figure marks where.
- The residual's O(DELTA) origin was established separately (`PER_TRIANGLE_C.md` §3); this sweep
  does not re-derive it.
