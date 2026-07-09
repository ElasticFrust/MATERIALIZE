# Code review — Phase 2 (flagged) / Phase 3 (refactored) / verification_tools (targeted)

Date: 2026-07-08/09. Scope: `Phase 2/forward_solver_torch.py` (read-only, protected core),
`Phase 3/verifications/` (full refactor), `verification_tools/` (only files Phase 2/3 actually
import). Phase 4 was explicitly excluded from this pass.

## What changed

All consolidation landed in `Phase 3/verifications/_common.py` (the existing shared-helper
module) and a new `verification_tools/_bootstrap.py`. No script's plotted output or achieved
ν/E changed — every consolidation was verified against the original inline code before being
applied (see **Verification** below).

### `_common.py` — new shared functions

| Function | Replaces | Previously duplicated in |
|---|---|---|
| `open_stretch` / `open_stretch_nu` | the open-boundary cut-and-stretch spring solver (`nwb_of`/`Kmat`/clamp-and-solve) | `two_region/demo.py`, `response_fields.py`, `large16k/make_verify_auxetic.py`, `make_cut_stretch.py`, `make_patch_isolate.py` (5x) |
| `load_network` (fixed) | now restores `edge_vecs`/`actual_len2` directly, eliminating the `augment()`/`open_truss()`/`edges_meta()` shim | `large16k/make_patch_response.py`, `make_patch_response_coarse.py`, `make_local_response.py`, `make_cut_stretch.py`, `make_patch_isolate.py` |
| `unit_mode_response`, `bare_stress`, `tensor_mag` | `mode_fields`/`unit_mode_strains` (relax under 3 unit macro-strain modes, compute stress) | `large16k/make_patch_response.py`, `make_patch_response_coarse.py`, `make_local_response.py` (3x) |
| `nuE_row_grid` | the "2-row (local ν / local E), one column per topology, region marked" plot layout | `auxetic_patch/make_maps.py:per_case()`, `large16k/large_designs.py:maps_figure()` |
| `box`, `triangle_verts` | trivial exact-duplicate helpers | `auxetic_patch/design_and_verify.py`, `design_all.py`, `large16k/large_designs.py` |
| `decoupled_ENu_design` | the "E differs only in R_E, ν differs only in a different R_ν" design recipe (fixed targets: R_E→1.8 vs bg 1.0, R_ν→-0.30 vs bg 0.20) | `auxetic_patch/design_and_verify.py:group3()`, `design_all.py:run_decoupled()` |
| `glue_square_hole` | punch-a-hole + `glue()` + measure + open-stretch pipeline | `two_region/inclusion_square.py`, `inclusion_rigidity_only.py` |

Once `open_stretch` was centralized, the fragile `import demo as D` / `import ribbon as RB`
cross-imports (scripts reusing each other's `cut_stretch` as ad hoc libraries) mostly evaporated —
`inclusion_square.py`, `inclusion_square_fields.py`, `ribbon.py`, `ribbon_fields.py` now call
`C.open_stretch` directly.

### `verification_tools/_bootstrap.py` (new)

Mounts `sys.path` (this dir + `../Phase 2`) and sets `matplotlib.use('Agg')` — the boilerplate
that was copy-pasted at the top of every script here. Only applied to the 3 files that had the
*full* boilerplate **and** are actually imported by Phase 2/3 (traced via the real import graph,
not guessed): `test_cluster_VD.py`, `test_cluster_Ceff_rigidity.py`, `pbc_dg_analysis.py`.

### Resolution policy used for every merge

Before merging any "duplicate," the actual behavior was diffed (not just matching names), and
ties were broken in this order: (1) existing shared module wins over a per-script copy, (2)
most-used copy wins, (3) most-recently-touched wins, (4) most-detailed/complete plotting wins,
(5) plain version wins over a specialized fork. Never write a fresh implementation from scratch —
always adapt one of the existing, already-validated copies.

## Two "duplicates" that were investigated and deliberately NOT merged

- **`vd_demo/vd_decompose.py` / `vd_sweep_sim.py`** still use `test_cluster_VD.build_geometry`
  instead of `_common.make_lattice`. Not a style inconsistency: `build_geometry` deliberately keeps
  **identical bond connectivity** across different `eta` values (both scripts' docstrings state
  this explicitly — a VD k-field computed on the disordered geometry is transferred bond-for-bond
  onto the regular geometry). `make_lattice` re-triangulates from scratch on every call, including
  after perturbation, which would silently break that connectivity guarantee. Left untouched.
- **ν/E-from-Voigt-tensor conversion in `test_cluster_Ceff.py` vs `physical_homog.py`**: not a
  duplicate. `physical_homog.py`'s own docstring states its formula is the corrected *physical*
  (virial) homogenization, and explicitly warns that `test_cluster_Ceff`'s is "the legacy metric
  average ... which biases ν on disordered/anisotropic meshes." Merging would either break
  `physical_homog`'s purpose or silently reintroduce the bias into `test_cluster_Ceff`. Left
  untouched — matches the existing `[[homogenization-physical-not-metric]]` convention.

## Flagged only — no code changed (per explicit scope decisions)

### `Phase 2/forward_solver_torch.py` (protected core, no edits without separate approval)

- `_woodbury_kkt_sparse` (lines ~699-823, ~125 lines) is **dead code** — never called anywhere;
  `_woodbury_kkt_sparse_combined` (~826-1020) supersedes it and is the only one invoked (from
  line ~424).
- `_assemble_dense_J` (~291-302) is rebuilt from scratch on **every** `forward()` call even though
  the constraint Jacobian is geometry-only and doesn't change across an `optimize()` run's
  iterations (up to 80 × restarts) — real loop-invariant recomputation.
- The ν/E formula is duplicated: computed inline in `forward()` (~454-460) and separately in
  `Phase 3/inverse_design.py:c6_to_nuE` (~52-58). A true fix means one importing from the other,
  which touches `forward_solver_torch.py`.

### Possibly-stale / misleadingly-named files (left as-is, not renamed or deleted)

- `make_pdf.py` (repo root) — hardcoded paths from a different machine/environment, not
  parameterized, not imported anywhere.
- `Disc_2_Cont_optimized.py` (repo root) — name references a solver (D2C) that its own docstring
  says was deleted; current contents are just mesh/topology generators, still imported by
  `Phase 2/test_forward_solver.py`, `Phase 3/test_inverse_design.py`, `Phase 4/data/topology_generators.py`,
  `verification_tools/verify_solver_open.py`.
- `verification_tools/verify_solver_open.py`, `verify_irregular_VD_eta.py` — possibly superseded by
  the newer PBC-based verification scripts; not confirmed either way.
- `verification_tools/test_*.py` naming — none are pytest-discoverable tests (no `test_...()`
  functions/asserts); ambiguous alongside the `verify_*.py` files doing the same kind of thing.

### `verification_tools/` files not imported by Phase 2 or Phase 3 (out of scope, untouched)

Traced import graph: Phase 2 imports nothing from `verification_tools/`; Phase 3 pulls in only
`test_cluster_VD`, `test_cluster_rigidity`, `test_cluster_Ceff`, `test_cluster_Ceff_rigidity`,
`physical_homog`, `pbc_dg_analysis` (directly or transitively). Everything else here still carries
the copy-pasted bootstrap boilerplate but was left alone:
`compat_projection.py`, `verify_gb_formulation.py`, `verify_intrinsic_solver.py`,
`verify_soft_circles_50x50.py`, `verify_soft_circles_50x50_directions.py`, `verify_soft_region.py`,
`verify_solver_final.py`, `verify_solver_open.py`, `verify_solver_sweep.py`,
`verify_irregular_VD_eta.py`, `test_angle_response.py`, `test_curvature_operator.py`,
`test_intrinsic_metric.py`, `test_intrinsic_VD.py`, `test_mean_isolation.py`.

### Also noted, not fixed (out of scope, discovered incidentally)

`verification_tools/pbc_dg_analysis.py`'s `write_readme()` crashes on Windows with
`UnicodeEncodeError` (writes a `Δ` character without `encoding='utf-8'`) — pre-existing bug,
unrelated to anything touched in this review.

## Verification performed

- Direct numeric-equivalence tests: every consolidated function compared against the original
  inline formula on the same lattice — bit-for-bit identical, except one deliberate case
  (`make_patch_response.py`'s per-forcing stress now precomputes per-mode stress and linearly
  combines instead of combining strain first; differs at ~1e-16 relative — floating-point
  reordering only).
- Ran every load-only script (loads a saved `.npz`, no re-optimization) end-to-end against real
  saved networks — all produced physics matching their documented expectations.
- Ran every script that calls `optimize()` end-to-end too (`design_and_verify.py`, `design_all.py`'s
  full 40-case sweep, `large_designs.py`'s full 16k-triangle sweep, `inclusion_square.py`,
  `inclusion_rigidity_only.py`, `ribbon.py`): **all 56 regenerated network files from
  `design_all.py` + `large_designs.py` are byte-for-byte identical to the pre-existing committed
  versions** — the strongest available confirmation that the refactor reproduces the exact
  original optimizer trajectory, not just similar-looking results. The 3 files that did differ
  (`inclusion_square.py`'s outputs) are expected: that script uses unseeded randomized
  multi-restart optimization, so it's non-reproducible run-to-run by construction, before and
  after this refactor alike.
- Ran both regression suites (`Phase 2/test_forward_solver.py`, `Phase 3/test_inverse_design.py`) —
  both pass, unaffected as expected since neither file was touched.
- Grepped for stale references to every removed local helper (`augment()`, per-file `cut_stretch`
  variants, `import demo as D`, etc.) — none found.
