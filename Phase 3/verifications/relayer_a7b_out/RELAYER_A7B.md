# A-7b — re-layering the shared library out of the oracle layer

**What.** Lifted the shared geometry / metric-tensor library out of `verification_tools/` — the
temporary, retireable validation oracle — into the core layer, so the design layer stops depending
on code that is scheduled for deletion.

**Why.** `CLAUDE.md` §2 states the architecture as *dependencies point inward toward the core; the
core never references the layers above it*, and marks `verification_tools/` temporary. That was
inverted: `Phase 3/inverse_design.py` built its meshes, its periodic constraint topology and its
`ElasticSolver` itself out of four scripts in there. Audit finding **A-7b**
(`documentation/AUDIT_2026-08.md`), blocker, queued first after the audit by the user's decision of
2026-08-14: *"it can't be that what was essentially verifications code for experiments will be
central. That breaks hierarchy."*

Commits: `d2d0ba9` (instrument + baseline) · `d553f75` (new modules) · `fa03257` (re-point all
consumers) · docs/results commit.

---

## 1. The violation was wider than the finding recorded

A-7b named `test_cluster_VD` only. `inverse_design.py:33-36` actually imported from **four**
`verification_tools` scripts:

```python
from verify_solver_open import clean_tri, build_open_mesh
import test_cluster_VD as VD                      # build_geometry, set_VD
from test_intrinsic_VD import kkt_from_tri_bond
from verify_solver_sweep import make_solver       # + _mount
```

`make_solver`/`_mount` *construct and mount the `ElasticSolver`* — solver construction living in a
sweep script. Fixing only the listed symbols would have left the design layer rooted in the
retireable layer. **Lesson for reading this audit: where a finding names specific symbols, re-derive
the full set before scoping the fix.**

## 2. What moved

| new module | contents | from |
|---|---|---|
| `Phase 2/metric_ops.py` | `vec3`, `tri_metric_change`, `bare_tensor` | `test_cluster_Ceff.py`, `test_cluster_rigidity.py` |
| `Phase 2/mesh_build.py` | `build_geometry`, `set_VD`, `kkt_from_tri_bond`, `clean_tri`, `build_open_mesh` | `test_cluster_VD.py`, `test_intrinsic_VD.py`, `verify_solver_open.py` |
| `Phase 2/solver_build.py` | `make_solver`, `_mount` | `verify_solver_sweep.py` |
| `verification_tools/sim_assembly.py` | `assemble_K_faff` | `test_cluster_rigidity.py` |

`assemble_K_faff` did **not** change layer — it feeds `physical_homog`, so it is genuinely
oracle-side. It only left an experiment script whose `main()` runs a 30×3-seed sweep, so
`verification_tools/` now reads as `{physical_homog, sim_assembly}` = the oracle, everything else =
experiments.

Bodies are verbatim. 43 consumer files were re-pointed (plus the 4 new modules and the instrument);
`verification_tools/` is no longer on `inverse_design.py`'s `sys.path` at all, so the inversion
cannot return silently. `verification_tools/` is net **−67 lines**: code left the layer.

**Deliberately not moved** — reasons in `AUDIT_2026-08.md` A-7b: `c6_to_nuE` and `W3_to_W9` (would
prejudge the open finding **A-10**, three disagreeing ν/E reductions); `DELTA`/`MODES`
(already oracle-side on both copies, so not part of the inversion — logged **A-7d**);
`_common.py`'s mesh constructors (a separate 798-line untangling — logged **A-7c**).

## 3. What static checking missed — and what caught it

Grepping for `CE.`, `TR.`, `VD.` per file was **not sufficient**, and believing it would have shipped
a break. `_common.py` was acting as an implicit **re-export hub**: 39 sites across 13 files reach
*through* it — `C.CE.vec3`, `C.CE.tri_metric_change`, `C.TR.assemble_K_faff` — a pattern the
symbol-level grep cannot see, because the text is `C.CE.` and not `CE.`. Dropping the now-unused
`import test_cluster_Ceff as CE` from `_common` therefore broke the whole `strain_stress/` family.

It was caught by **running `Phase 3/test_inverse_design.py`**, which failed
`test_strain_stress_equivalence` with `module '_common' has no attribute 'CE'`, and then by the
import inventory, which found five more (two of them only *transitively*, via
`design_and_verify` and `mode_selective_colocated`).

Fixed by re-pointing the reach-throughs to `C.MO.` / `C.SA.` rather than restoring `CE`/`TR` as
aliases in `_common` — keeping the name `CE` for what is now `metric_ops` would have preserved
exactly the misleading label this task exists to remove.

**Generalisable:** an import-level refactor is not verifiable by grep alone. Attribute access
through an intermediate module is invisible to it, and only execution finds it.

## 4. Two things that fell out of the move

- **A circular import dissolved.** `test_intrinsic_VD.nuE` imported `make_solver` *lazily*, with a
  docstring explaining it was to break a cycle with `verify_solver_sweep`, which imported
  `kkt_from_tri_bond` back from it. Both now come from the core, which depends on neither, so it is
  a plain top-level import.
- **The two `bare_tensor` variants became one.** k-aware (`test_cluster_rigidity`, 11 callers) and
  k-less (`test_cluster_Ceff`, 4 callers) are now a single function reading `tri_k` with default 1.
  Exactly equivalent to both: the k-aware callers always set `tri_k`, and the k-less callers pass
  `pbc_dg_analysis.build_periodic_tf_mesh` meshes, whose return dict contains no `tri_k`/`bond_k`.
  Checked numerically on a mesh of each kind, not just argued — and the instrument asserts the
  no-`tri_k` premise rather than trusting it.

## 5. Verification

Instrument: `Phase 3/verifications/relayer_a7b.py` (`--out`, `--compare`). It resolves the moved
symbols from the **new** homes, falling back to the **old**, so one script runs on both sides.

**The plan's original criterion had to change, and the measurement is the reason.** It called for
bit-identity to 1e-15 everywhere. `forward(method='intrinsic')` is **not bit-reproducible**: calling
the *same pre-move* `make_solver` twice in one process gives ν differing by 1.7e-16 (two distinct
doubles), C_eff by 2.2e-16, per-triangle C6 by 5.0e-16, run-order dependent. That is audit **B-1**
(solver nondeterminism, source unidentified), entirely independent of this work. So:

- everything deterministic — including the **fully constructed solver**: geometry buffers plus the
  C1 edge-compatibility, C2 curvature and C3 normalisation operators — is required **bit-identical**.
  This is *stronger* than the plan had, and it pins `make_solver` without a forward solve at all.
- values downstream of a forward solve are compared at relative **1e-13**, ~100× the measured
  envelope.

| check | result |
|---|---|
| 9 moved pure-NumPy symbols, old vs new, both present | **bit-identical** |
| constructed solver state (buffers + C1/C2/C3 operators), old vs new | **bit-identical** |
| `Phase 2/test_forward_solver.py` | **7/7**; test [7] component-wise C_eff vs `physical_homog.energy_C` on W≠0 meshes, worst **1.67e-03** — identical to baseline |
| `Phase 5/verifications/sanity.py` | ν = 0.333333, E = 1.154701 on **both** solver and sim |
| `Phase 3/test_inverse_design.py` | **16/16 ALL PASSED**, matching baseline (`p3_after.log`) |
| import inventory, 145 → 149 files | **no regressions**; the 4 new modules import ok, the 2 pre-existing failures are unchanged, and one flaky `TIMEOUT` resolved to ok |
| `inverse_design` transitive imports | **zero** oracle modules |

Test [7] is the load-bearing one: `CLAUDE.md` §3 records that the crystal gate is *structurally
blind* to the homogenisation (W ≡ 0 on the regular lattice), so ν=1/3 could not see a contraction
error. [7] compares the full tensor component-wise against an independent energy Hessian on meshes
where W ≠ 0.

**Baseline honesty.** The first `before.json` was captured after the new modules already existed, so
`_resolve()` picked the new layout and it recorded the new code — useless as an "old" reference. The
genuine baseline was re-taken from commit `0ad2e49` in an isolated `git worktree`
(`true_before.json`).

**The inventory's verdict needed correcting too.** As first written it flagged *any* ok↔not-ok
flip as a failure, and duly reported FAIL for `verify_positions.py: TIMEOUT → ok` — an
*improvement*, caused by the per-file cap being wall-clock while the baseline ran under CPU
contention. It now separates a regression (`ok → not-ok`, fatal) from an improvement, otherwise it
would cry wolf on every future run. The two surviving failures are pre-existing and unrelated:
`anisotropy/rot_target_seed_probe.py` (its own `_common` sys.path is broken) and `make_pdf.py`
(`markdown2` not installed).

**A side effect worth recording.** The import inventory executes each file up to its last
module-level import; in files with code *between* imports that still runs real work, and it
regenerated five tracked artifacts (`sphere_response_compare.png`, three `Phase 5/networks/*.npz`,
`reentrant_by_move.png`). They were restored from git, and the "after" inventory is run in a
throwaway worktree so the live tree cannot be touched.

## 6. Files

- `relayer_a7b.py` — the instrument (`--out`, `--compare`, `--imports-from`)
- `true_before.json` / `.log` — genuine pre-change baseline (worktree at `0ad2e49`, old layout)
- `final.json` / `.log` — post-change, taken in a throwaway worktree of the finished commit
- `p3_after.log` — the Phase 3 suite, 16/16
- `before.json`, `step1.json` — the first, superseded baseline and its step-1 re-run; kept only
  because §5 "Baseline honesty" refers to them
