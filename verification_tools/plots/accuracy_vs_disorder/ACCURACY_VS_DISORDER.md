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

## 1. Headline

| quantity (mean over seeds) | η=0 | η=0.2 | η=0.4 | η=0.5 |
|---|---|---|---|---|
| **oracle control** — virial vs energy, *no solver* | 1.1e-13 | 1.1e-13 | 1.1e-13 | 1.1e-13 |
| solver **bulk** vs oracle (gated by [7]) | 2.2e-13 | ~9e-4 | ~1e-3 | ~1e-3 |
| solver **per-triangle** vs oracle (gated by [8]) | 2.2e-13 | ~3e-3 | ~7e-3 | ~1.2e-2 |
| per-triangle **with the A-0 defect** (reference) | 4.4e-13 | ~1e0 | ~1e0 | ~2e0 |

(frozen η / typical family; VD a=+100 is the exception — §3.)

Three things follow:

- **The oracle does not drift.** The control is flat at 1.1e-13 across every family and every η.
  So the growth in the other rows is the solver (and the strain measure), not the yardstick.
- **Per-triangle costs a factor 3–5 over bulk**, consistently — fluctuations that cancel in the
  average do not cancel locally. Both grow slowly with η.
- **The A-0 defect sits 2–3 orders above the correct curve for every family at every η > 0** — and
  is *indistinguishable* at η = 0. The gate discriminates everywhere except the crystal, which is
  exactly the blindness `CLAUDE.md` §3 warns about, shown directly.

## 2. The controlling variable is stiffness, not η

`accuracy_vs_stiffness.png` is the important figure. Plotting per-triangle error against the bulk
modulus E of each realisation, **all nine families collapse onto a single trend**, spanning 8
decades in E and 8 in error. η is only a proxy; what actually governs accuracy is **how close the
realisation is to a mechanism**:

| bulk E of the realisation | per-triangle agreement |
|---|---|
| E ≳ 0.1 (a stiff material) | 1e-3 … 1e-2 — comfortably inside the gate |
| E ≈ 1e-2 | ~2e-2 — crosses the gate tolerance |
| E ≈ 1e-3 | ~1e0 |
| E ≲ 1e-4 (floppy) | 1e1 … 1e5 — the linear read-back is meaningless |

**Practical criterion:** trust the per-triangle read-back while **E ≳ 1e-2**; below that, treat it
as unvalidated regardless of how the design was produced. This is the documented near-mechanism
failure mode (`CLAUDE.md` §3: "near one, W is ill-conditioned and the linear read-back diverges from
the true nonlinear relaxation — a soft-eigenvalue issue, **not** a linearisation artefact"), now
with a number attached.

## 3. VD a=+100 is pathological, and it is the geometry's fault, not the solver's

At a=+100 the stiffness is effectively binary (k ∈ {0,2}) and disorder drives the network floppy
fast — measured on the frozen-η family:

| η | 0.0 | 0.1 | 0.2 | 0.3 | 0.5 |
|---|---|---|---|---|---|
| bulk E | 1.155 | 1.1e-2 | 1.5e-3 | 1.5e-4 | 1.8e-4 |
| per-triangle error | 3.2e-13 | 3.2e-2 | 4.0e-1 | 4.5e0 | 2.1e2 |

E falls four orders by η=0.3. The absolute disagreement |ΔC| stays modest, but |C| itself collapses,
so the *relative* error explodes — and by η=0.5 the disagreement is 200× the tensor magnitude, i.e.
the solver's C(s) genuinely bears no relation to the physical one. This is not a normalisation
artefact: it is the mechanism regime, and it is detected clearly.

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
- The solver side is fed the **sim's** measured W, isolating the contraction. The *solver's own* W
  vs the sim's is a separate question not measured here.
- At η ≥ ~0.48 some realisations are rejected by the sim's health gate (sliver triangles), so the
  far-right points average over fewer seeds — the shaded band in the per-family figure marks where.
- The residual's O(DELTA) origin was established separately (`PER_TRIANGLE_C.md` §3); this sweep
  does not re-derive it.
