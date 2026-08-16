# `verification_tools/` — what is live, what is legacy

This directory is **not a library**. `CLAUDE.md` §2: it is the independent oracle plus a pile of
experiment scripts, and it is *temporary and retireable* — production code in Phases 2/3 must not
depend on it. Read this before running or citing anything in here.

---

## 1. The oracle — LIVE, load-bearing, do not touch casually

| file | role |
|---|---|
| `physical_homog.py` | the independent oracle: `relax`, `virial_C`/`virial_nuE`, `energy_C`/`energy_nuE`, `energy_C_per_triangle`, `energy_C_region`, `require_healthy_mesh` / `UnhealthyGeometryError` |
| `sim_assembly.py` | `assemble_K_faff` — the oracle's own stiffness assembly |

**INVARIANT (`CLAUDE.md` §3, audit A-9): these two share NO code with the design path.**
`physical_homog` keeps its **own** `DELTA`/`MODES` rather than importing them — that duplication is
**deliberate; do not "fix" it**. Otherwise "checked against the sim" degrades into the code checking
itself, which is exactly how the 2026-08 shear-channel defect (A-0) survived for months.

Gated by `Phase 2/test_forward_solver.py` [7]/[8] and `Phase 3/test_inverse_design.py` [15].

## 2. Current experiment scripts — LIVE, runnable

`accuracy_vs_disorder.py` (+ `plots/accuracy_vs_disorder/`), `per_triangle_C_comparison.py`,
`recheck_sweep_nu_E_eta.py`, `exact_kinematics_check.py`, `finite_amplitude_check.py`,
`compat_projection.py`, `pbc_dg_analysis.py`, and the `test_*`/`verify_*` files not listed in §3.

Each writes into `plots/<name>/` with a results `.md` beside its outputs.

## 3. The legacy `test_cluster_*` / early `verify_*` island — SUPERSEDED, DOES NOT RUN

`test_cluster_Ceff.py`, `test_cluster_VD.py`, `test_cluster_rigidity.py`,
`test_cluster_Ceff_rigidity.py`, `test_intrinsic_metric.py`, `test_intrinsic_VD.py`.

**Status: these do not execute.** Their comparison routines call `Ceff_nuE`, the legacy
**area-weighted** metric average, which was **tombstoned 2026-08-10 as physically wrong** (it biases
ν on unequal-area meshes and overstates auxeticity; `A(s)` carries no area prefactor, so the correct
homogenisation is the UNWEIGHTED mean of `C(s)`). `Ceff_nuE` now raises `NotImplementedError`, so any
path reaching it dies on first call. Audit **A-7**.

**Why they are still here.** These modules also hold pieces nothing else has yet re-homed —
`mf_W3` (mean-field W), and `edge_op` / `curv_op` / `mean_op` / `intrinsic_solve` — used by
`test_intrinsic_VD.py`. Deleting the modules would take those with them.

**What they are NOT any more.** Until 2026-08-15 this cluster was genuinely load-bearing: it supplied
`vec3`, `tri_metric_change`, `bare_tensor`, `build_geometry`, `set_VD`, `assemble_K_faff` to Phases
2/3, and `Phase 3/inverse_design.py` imported `test_cluster_VD` directly. **A-7b moved all of that**
into `Phase 2/{metric_ops,mesh_build,solver_build}.py` + `verification_tools/sim_assembly.py`.
Verified 2026-08-16: `_common.py`, `Phase 2/test_forward_solver.py` and `Phase 3/inverse_design.py`
import **none** of this cluster; its only remaining importers are its own members and
`Phase 3/verifications/relayer_a7b.py` (the A-7b migration instrument).

**Citing them.** `documentation/MATERIALIZE.md`, `Phase 2/SOLVER_GUIDE.md` and
`INTRINSIC_METRIC_SOLVE.md` cite these as the provenance of older results. Those *conclusions* are
sound and have since been re-established far more strongly and independently (`test_inverse_design`
[15] vs the energy-Hessian tensor at 1.6e-12). But **specific numbers computed through `Ceff_nuE`
are stale** — in particular the "single-site mean-field ≈1.4× over-compliance" figure.
**Do not re-cite that number without re-deriving it** with the unweighted homogenisation (queued for
the post-audit verification campaign).

**Deleted 2026-08-16** (A-7, orphaned — imported by nothing, dead on first call):
`verify_gb_formulation.py`, `verify_intrinsic_solver.py`. Git-restorable.

## 4. House rules

- **Tests and verification scripts SHOULD import the oracle** — that is their job. The
  no-oracle-imports rule governs which way *production* code depends, not who may run a check.
- Run everything with `C:\Users\doron\anaconda3\python.exe` (`python`/`python3` on PATH are broken
  Windows Store stubs). float64 throughout.
- Every experiment ends with a results `.md` carrying its commit/config/seed.
