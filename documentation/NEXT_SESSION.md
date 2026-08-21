# NEXT SESSION — start here

**Written 2026-08-18, end of the audit programme.** Repo state: working tree clean, everything
pushed, all five gates green. Read `CLAUDE.md` §1–3, this file, then `documentation/AUDIT_2026-08.md`
**§6 STATUS** (which lists what is left and why).

---

## The three things to do first

### 1. Rebuild + retrain M2 — mechanical, unblocked, long compute

The training labels are **verified stale**: 37 of 41 sampled labels drift, worst |Δν| = **1.73**
(`Phase 5/m2/M2.md` has the per-family table). `checkpoint.pt` learned a map the current solver
disagrees with, so **every M2 number is unverified** until this is redone.

```
python "Phase 5/dataset.py"            # regenerate Phase 5/dataset/  (nets + dataset.npz)
python "Phase 5/m2/build_dataset.py"   # regenerate Phase 5/m2/data/dataset.npz
python "Phase 5/m2/train.py"           # retrain
```

`Phase 5/dataset/` and `Phase 5/m2/data/` are **absent on purpose** so nothing can retrain on the old
labels by accident. The July data is archived at
`validation_2026-08/attic/m2_dataset_2026-07_pre-A0/` — **keep it**, it is the only record of what the
current checkpoint learned and the baseline for a before/after comparison.

Two things to fix while retraining, both already flagged in `M2.md`:
- the smoke-train metrics were **never recorded** (`<FILL>` placeholders) — record them this time;
- validate on **held-out topology FAMILIES** against the **independent sim**, not a random split
  against the solver labels the model trained on.

### 2. B-1 — the one real blocker

Solver nondeterminism, **localised to `W`** (the constrained solve): the first divergence is always
there, never in geo/k/bare, and not in the contraction. ~1 run in 21, error up to **6 %**. Clusters in
time with a persistent state transition; import order refuted. Harness: `Phase 3/verifications/b1_reproduce.py`
(modes `fingerprint`, `repeat`, `bisect`, `commits`, `suitectx`, `stages`, `imports`).

It cannot be scheduled — it has to be caught. It did not fire in any gate run on 2026-08-18, which
proves nothing. **Everything else is clean now, so this is the top correctness item.**

### 3. `FUTURE_DIRECTIONS` #2 — stability constraint, re-aimed

Worth doing early because it is the *cause* of failures this audit kept hitting (g1_2's auxetic
targets, `verify_lattice`, what `ab_quality_floor` could only mitigate). **But #2 as written targets
the wrong quantity** — see the correction block now at the top of that entry. Penalise the
**per-triangle conditioning of `A(s)`**, not the smallest global stiffness eigenvalue — **but read
FD #2's 2026-08-21 block first: `rcond(A(s))` alone is measurably NOT a predictor of solver error
(hexagon: corr +0.73, best accuracy at worst conditioning), and goal1's worst case is healthy on
every conditioning axis**; `CLAUDE.md` §3
shows the latter gives a *false all-clear* for the dominant failure. Ready-made test:
`verify_lattice`'s regular-lattice ν=−0.2 case (solver −0.112 vs sim +0.137) — the constraint works
iff that stops flipping sign.

---

## Also ready, no blockers

| | |
|---|---|
| **A-8** open-boundary verification suite | was parked "until everything is clean" — **that condition is now met** |
| **A-17 tail** | `honeycomb tiling`, `reentrant_honeycomb` are not Delaunay-representable; centre-vertex re-representation (the hexagon gate proves that representation is sound). Currently tagged `mesh_ok=False`, so nothing is silently wrong |
| **A-18** | analytic basis-cell oracle — a genuine THIRD path (everything today compares two codes) and the basis for the interactive applet |

## Science, longer range

**#1 residual stress / incompatible ḡ (★★★★★)** is the project's highest-value extension, but it is
**blocked on an open research question** — the correct finite-size split ḡ = ḡ_bg + δḡ. Implementing
before that is settled would bake in a choice that must later be undone. Then #3 differentiable
open-boundary, #4 topology/connectivity design, #13 differentiable positions. Endgame: M2 scale-up →
**VAE + interpreter** (no code exists for either yet), then 3D (#11).

---

## Things not to re-learn the hard way

- **An artifact without a B-3 provenance stamp predates the August fixes and is suspect.**
  `validation_2026-08/scan_provenance.py` is the detector. 520 → 77 unstamped; the 77 are deliberate
  keeps with no regenerated counterpart.
- **A re-run set must be closed under producer→consumer.** A script that only *loads and plots* sails
  through any cache-stowing: `fig5` rendered a 9-July network in 7.5 s and exited 0, and its field was
  materially wrong ([−7.7, +19.2] vs [−3.4, +15.8] once fixed).
- **Exit codes lie.** Four defects this campaign were found by *inspecting outputs*; none by an exit
  code. `verify_positions` printed FAILED and returned 0. The runner now has `verdict()` and
  `stale_inputs()` for this.
- **Suspiciously fast = suspicious.** fig1 (41 s), fig5 (7.5 s), goal2_attempts (42 s) were each a
  stale cache, each caught only by eye.
- **On this box, piping stdout gives cp1252** — anything printing `ν` dies with `UnicodeEncodeError`
  *after* the compute and *before* the save. Always `PYTHONIOENCODING=utf-8`. (Hit four times in one
  session.)
- **ν < 0 IS reached and sim-confirmed** — `auxetic_sweep`: 36/70 rows below −0.05, all stable, down
  to **−0.6009** across regular / aniso_shr / aniso_str / disorder_hi / disorder_lo. g1_2's null result
  is about *its* setup (k-only on fixed braced tilings), **not** a general limit. Do not let the g1_2
  headline overwrite this.
