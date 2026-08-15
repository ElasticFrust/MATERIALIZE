# Re-check: ν(η) and E(η) — disordered networks and VD contrasts

**What.** Recovered the ν(η), E(η) response of disordered periodic networks and of five
virtual-distortion rigidity contrasts, and checked it three ways after the A-7b re-layering.

**Method.** `verification_tools/recheck_sweep_nu_E_eta.py`, reusing
`verify_solver_sweep.one_realisation` (the recorded code path — no reimplementation).
N=20 (800 triangles), **η ∈ [0, 0.5] step 0.02 (26 points)**, 10 seeds per η, 6 cases:
disordered (uniform k) plus VD `k = 1 + tanh(a·(|R|−1))` for a ∈ {−10, −2, +5, +10, +100}.
Commit `55850e9`; reference commit `0ad2e49` (immediately pre-A-7b).

Figures: `recheck_nu_eta.png`, `recheck_E_eta.png` — solver and independent sim **overlaid in the
same panel** per case (project convention, CLAUDE.md §3), mean±σ over seeds.
Raw: `recheck_raw.npz`, `recheck.json`.

---

## 1. Did the re-layering move the physics? No — exactly zero.

Matched elementwise against the same sweep run at `0ad2e49` (same N, same η grid, same seeds):

| | |
|---|---|
| values compared | 26 η × 10 seeds × 6 cases × 4 quantities = **6240** |
| worst relative deviation | **0.000e+00** — bit-identical |
| health-gate skip pattern (which seeds the sim refused) | **identical** |

A pure move should move nothing, and nothing moved.

## 2. Known value at η = 0

The regular lattice, on both paths independently:

| path | ν | E |
|---|---|---|
| independent sim | 0.333333 (Δ = 2.8e-16) | 1.154701 (Δ = 1.3e-13) |
| solver | 0.333333 (Δ = 5.0e-09) | 1.154701 (Δ = 1.7e-08) |

against the analytic ν = 1/3, E = 2/√3 = 1.154701.

## 3. Solver vs independent sim — the physics check

Same realisation, worst case over all η and seeds. Split by whether the sample is still a *material*
or has collapsed toward a mechanism:

| case | max\|Δν\| | max ΔE/E | max\|Δν\| (E>0.05) | max ΔE/E (E>0.05) |
|---|---|---|---|---|
| disordered | 0.0147 | 0.032 | 0.0147 | 0.032 |
| VD a=−10 | 0.0882 | 0.165 | 0.0489 | 0.115 |
| VD a=−2 | 0.0165 | 0.047 | 0.0165 | 0.047 |
| VD a=+5 | 0.1181 | 0.079 | 0.1181 | 0.079 |
| VD a=+10 | 0.1005 | 0.137 | 0.1005 | 0.137 |
| **VD a=+100** | **0.5913** | **0.849** | **0.0453** | **0.211** |

> **CORRECTION (2026-08-15, later the same day).** The numbers in this table are **not** a measure of
> solver-vs-sim physics. They compare the solver's ν,E — which are **ν_yx and E_y, the y-direction
> only** — against the sim's, which are **½(ν_xy+ν_yx) and ½(Ex+Ey)**. That is audit **A-10**, and on
> anisotropic/VD networks it dominates everything here. Applying the SAME reduction to BOTH tensors:
>
> | case (N=14, η=0.30) | Δν as tabulated | Δν, same reduction |
> |---|---|---|
> | disordered | 0.0003 | **8.6e-14** |
> | VD a=−10 | 0.0791 | **1.4e-12** |
> | VD a=+10 | 0.0929 | **2.4e-12** |
> | VD a=+5 (η=0.40) | 0.0756 | **1.4e-12** |
>
> So solver and sim agree to ~1e-12 on these; the 0.08–0.09 was the convention difference. The
> "disordered" row looks small only because that mesh is nearly isotropic, where the two reductions
> coincide. **Read the table below as a comparison of two ν,E CONVENTIONS, not as accuracy.**

**The one large disagreement is understood and expected.** At a = +100 the stiffness is effectively
binary (k ∈ {0, 2}) and the network is at or past a floppy mode: E collapses from 1.1547 at η=0 to
0.056 by η=0.04 and ~0 thereafter, and **81% of its samples have E_sim < 0.05**. `CLAUDE.md` §3
records exactly this regime — near a mechanism W is ill-conditioned and the linear read-back
diverges from the true nonlinear relaxation, a soft-eigenvalue issue, *not* a linearisation
artefact. Restricting to samples that are still a material collapses the discrepancy from 0.59 to
0.045 in ν, i.e. into line with every other case. It is a statement about the sample, not the solver.

## 4. The physics recovered

- **Auxetic band (disordered):** ν goes +0.3333 at η=0 → **−0.145** at η=0.5, smoothly and
  monotonically, crossing zero near η ≈ 0.42. Consistent with the recorded band (ν → ≈ −0.1).
- **VD a<0** (stiffer where compressed) *raises* ν first — to ≈0.55 (a=−10) and ≈0.41 (a=−2) around
  η ≈ 0.35–0.4 — then drops it sharply at the largest η.
- **VD a>0** (stiffer where stretched) drives strong auxeticity: ν → ≈ −0.5 by η = 0.5 for both
  a=+5 and a=+10, crossing zero near η ≈ 0.19 (a=+5) and η ≈ 0.13 (a=+10).
- **E falls monotonically with η in every case**, fastest for large |a|.

The 0.02 η step matters: at the legacy 0.05 spacing the a=−10 and a=−2 maxima and the zero
crossings are only 1–2 points wide and read as sampling artefacts.

## 5. Two things found while doing this

**(a) The old stored sweep is superseded and must not be used as a reference.**
`verification_tools/plots/dg_solver_sweep_20x20.npz` *looks* like the reference for exactly this
quantity. It is not:
- its E is in the solver's **internal units** — E(η=0) = 0.0625 against the physical 1.154701, a
  factor 18.4752086 = (2/√3)/0.0625 to 12 digits — it predates `physical_units`;
- its ν at η>0 came from the legacy **area-weighted** metric average (`Ceff_nuE`), tombstoned
  2026-08-10 as physically wrong *because it biases ν on disordered meshes* — precisely this regime.
  ν(η=0) still agrees exactly, since at η=0 all triangles have equal area and the weighting is a
  no-op — which is why the staleness is easy to miss;
- it also predates the shear-channel fix.

Reproducing its numbers today would mean the corrections had been undone. Logged in
`documentation/ARCHITECTURE.md` §5 as a standing warning about stored artifacts.

**(b) The sim's health gate fires in this sweep, and callers must handle it.** At η ≥ 0.44 the
perturbed lattice produces sliver triangles; `physical_homog` screens them and raises
`UnhealthyGeometryError` rather than letting scipy/LAPACK hard-crash. The first version of this
script did not catch it and died mid-sweep at η=0.45. It now records those realisations as NaN and
reports them (4/10 seeds at η=0.44, 5/10 at η=0.46), so the high-η averages are over the seeds that
are physically meaningful. This is the documented contract (CLAUDE.md §3) and is worth remembering
for any new sweep that reaches large disorder.

## 6. Limitations

- One system size (N=20, 800 triangles); no finite-size study here.
- 10 seeds per point — the ±σ bands at large η in the VD a=±10/+100 panels are wide, and the
  a=+100 case is dominated by mechanism-proximity rather than by disorder.
- The high-η end of every curve is averaged over fewer than 10 seeds where the health gate fired.
- Bulk ν, E only — this says nothing about per-triangle fields, which remain covered only by the
  shared-contraction route (audit **A-9**, still open).
