# Campaign Stage 4b — `goal2_attempts` re-run

**Provenance.** Commit in `COMMIT.txt`, clean tree. Fresh output dir
`Phase 5/{networks,results}/goal2_attempts_rerun/` — the 115 pre-fix artifacts are PRESERVED, not
overwritten, so the old-vs-new comparison stays possible.

## 1. Result

13/13 cases. **kept 42 · dropped (untrustworthy) 52 · out-ranked 65** — 159 attempts total.
**Attempts cluster away from target in 13/13 cases**, i.e. the selection story holds: the designs the
pipeline keeps really are the good ones, not an arbitrary slice.

| provenance | kept err | attempt err |
|---|---|---|
| crystal | 0.015 – 0.025 | 0.055 – 0.208 |
| hand orthotropic | 0.012 – 0.119 | 0.077 – 0.837 |
| random SPD | 0.056 – 0.449 | 0.709 – 1.202 |

Same ordering as Stage 4: crystal targets easiest, random SPD hardest — consistent with random SPD
tensors not necessarily being realizable by these networks.

`PLAN.md` recorded the pre-fix set as **INVALIDATED** (98/102 designs worsened, mean error 0.20 →
0.71, worst 4.14). The re-run's kept designs sit at 0.012–0.449, so — as in Stage 4 — **the
capability survives; it was the artifacts that were broken.**

## 2. Two defects found, and a NEAR-MISS worth recording

The first attempt **crashed** (exit 1) on an uncaught `UnhealthyGeometryError`, same degenerate
signature as Stage 4's pool bug (`min area 1.110e-16, mean 4.330e-01`).

**(a) Unguarded sim call.** The sim raises on near-singular geometry *by design* (`CLAUDE.md` §3:
"callers just try/except it"). Unguarded, ONE bad saved design killed the entire run. This is exactly
the **A-4** coverage gap flagged at the end of Stage 4 — the A-4 fix wrapped `verify()` inside
`designer.design()` but not this path. Now records and skips.

**(b) THE NEAR-MISS — it was re-analysing, not re-running.** The crash occurred inside the ADD-ONLY
*reload* branch: it had found 115 previously saved `.npz` files and was re-reading them instead of
re-designing. Those were produced under the **pre-fix solver** and with the **unscreened pool** Stage 4
had just fixed.

**Had those files not happened to contain the degenerate geometry, Stage 4b would have completed
cleanly and reported warmed-over pre-fix numbers as a re-run.** The campaign's premise — "re-run,
don't re-analyse" — would have been silently violated, with healthy-looking output. The Stage 4 pool
bug crashed the one path that would have produced a fake result.

Fixed by `FORCE_REDESIGN` (env `G2ATT_FORCE_REDESIGN`), with the reasoning recorded at the flag:
add-only reload is right for incremental work and wrong for a re-run campaign.

**Generalisable lesson:** an "add-only / resume if outputs exist" convenience is a correctness hazard
in a re-run campaign. Any driver with that behaviour must be checked before a stage is trusted.

## 3. Limitations

- One run per case; the B-1 constraint applies (every number here is sim-cross-checked, which is what
  the trustworthy/dropped split is).
- `random SPD` errors may reflect unrealizable targets rather than designer limits.
- The old-vs-new comparison is by re-derivation, not by reading the stale set: old `target_err_sim`
  used a different metric definition (audit A-2) and is not directly comparable.
