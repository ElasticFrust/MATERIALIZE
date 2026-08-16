# Verification campaign — Stage 0: re-verify Phase 2 and Phase 3

**What.** Re-verification of the layers beneath the designer, before any Phase 5 campaign runs.
Dependencies point inward (`Phase 5 → Phase 3 → Phase 2`), so a surprising Phase 5 result must never
be attributable to an unverified layer underneath. Plan: `documentation/VERIFICATION_CAMPAIGN.md` §3.

**Provenance.** Commit **`5234913`**, working tree **clean** (`COMMIT.txt`, `DIRTY.txt` beside this
file). Anaconda python, float64. Raw logs: `*.log` in this directory.

---

## 1. Results

| check | result | verdict |
|---|---|---|
| `Phase 2/test_forward_solver.py` | 8/8 — crystal ν=0.333333; **[7]** bulk `C_eff` vs energy Hessian worst 1.65e-02; **[8]** per-triangle worst 9.74e-03, regional worst 1.67e-02 | **PASS** |
| `Phase 3/test_inverse_design.py` | 16/16 — **[15]** solver-vs-oracle tensor **1.6e-12**, virial-vs-energy 1.1e-13 | **PASS** |
| `Phase 3/verifications/verify_lattice.py` | geometry PASS; **design section has a real problem** → §3 | **PASS (but see §3)** |
| `Phase 5/verifications/sanity.py` | ν=0.333333, E=1.154701 on **both** solver and sim — exact | **PASS** |
| `Phase 5/verifications/test_designer_surface.py` | 4/4 (A-2…A-6) | **PASS** |
| `verification_tools/accuracy_vs_disorder.py` | 9 families × η ≤ 0.42, 3 seeds | **PASS** → §2 |

**No B-1 anomaly fired.** `[15]` read 1.6e-12 and the always-on dump
(`Phase 3/verifications/b1_dumps/`, threshold 1e-9) stayed empty. At the measured ~1-in-21 rate this
is the expected outcome of a single run and is **one clean draw, not evidence of absence** — the
mistake made and retracted on 2026-08-16.

## 2. The disorder sweep reproduced EXACTLY

Every family matched the values recorded on 2026-08-15, to the digit:

| family | max end-to-end (η ≤ 0.42) |
|---|---|
| re-triangulated η | 1.9e-10 |
| VD a=−2 | 3.0e-09 |
| frozen η, uniform k | 4.0e-09 |
| anisotropic ψ=0.6 | 4.3e-09 |
| VD a=+5 | 5.0e-09 |
| VD a=+10 | 7.7e-09 |
| random binary k | 1.3e-08 |
| VD a=−10 | 6.0e-08 |
| **VD a=+100** | **1.3e-04** |

Worst over the whole domain **1.31e-04**, identical to the previous run. Bit-level reproducibility of
a multi-hour sweep across a day and several intervening commits is itself a useful result.

### Correction found by this stage

The claim **"the oracle control is flat at 1.1e-13 across every family and every η"** — repeated in
`ACCURACY_VS_DISORDER.md`, the register and memory — is **wrong for VD a=+100**:

| η | 0.00 | 0.18 | 0.30 | 0.42 |
|---|---|---|---|---|
| oracle control (virial vs energy, no solver) | 1.1e-13 | 8.8e-13 | 3.5e-11 | **1.3e-10** |

Three orders of drift; median 1.7e-12. True for the other eight families. **Conclusion unaffected** —
the yardstick is still ~6 orders tighter than the 1.3e-04 it is used to measure — but the oracle is
not perfectly stiffness-independent, and any claim made near a mechanism should quote the control
beside it. Corrected in all three places.

## 3. `verify_lattice.py` — a real defect, NOT fixed here

Its design section reported:

```
regular      target=-0.20  solver=-0.190  sim=-0.445  CHECK
aniso_shr    target=-0.20  solver=-0.201  sim=-0.201  PASS
disorder_lo  target=-0.20  solver=-0.190  sim=-0.195  PASS
ALL GEOMETRY PASS
```

A **0.255** solver-vs-sim gap on the simplest geometry, while the other two agree to ~0.005.

**Three distinct problems, established by reproducing it faithfully** (`make_topology('regular', 8)`,
which is what it uses — not `make_lattice`):

1. **Wildly nondeterministic.** Four independent runs at reg ∈ {0.003, 0.01, 0.02, 0.05} landed at
   solver ν = **+0.304, +0.101, +0.163, +0.138** — *none* near −0.2, while the script's own run
   reached −0.190. Identical code, outcomes spanning 0.5 in ν. This is B-1 **stage 1** (the real,
   understood `optimize()` bistability) appearing OUTSIDE the test suite that B-2 fixed.
2. **`reg=0.003`**, below `CLAUDE.md` §3's stated 0.01–0.05, with no restarts.
3. **It gates nothing.** `ok` is computed from the geometry section only; the design section's
   `CHECK` never reaches the verdict, so `ALL GEOMETRY PASS` prints regardless of how far off the
   design is. The name is technically honest; next to those lines it reads as a global pass.

**Open question, not answered here:** whether ν = −0.2 is reachable at all on a *regular* lattice by
k-design alone. The η-disorder family only reaches ν ≈ −0.11, so the target may be unrealizable for
that topology — in which case the optimiser is chasing an impossible goal and settling on whatever
soft configuration it finds, which would explain why the sim disagrees precisely when the solver
"succeeds".

**Deliberately not fixed in this stage.** Whether the target is realizable, and whether the design
section should gate, are design decisions for the project owner, not a cleanup. Recommendation:
assert on `solver_sim_gap`, raise `reg` to 0.02 with restarts, and either pick a reachable target or
document it as a known-unreachable probe.

*Method note:* the first diagnostic tried here (`k_min/mean < 1e-4` ⇒ "near-mechanism") was
**discarded as unreliable** — the register records that softplus underflows to exactly 0 in
successful designs too, so it does not discriminate.

## 4. Verdict

**Stage 0 PASSES.** Phase 2 and Phase 3 are sound; the designer surface and the sanity gate are
exact; the broadest solver-accuracy statement reproduces bit-for-bit. Stage 1 (`goal1` shakedown) may
proceed.

Two items leave this stage open, neither blocking: the `verify_lattice` design section (§3) and the
standing B-1 constraint — **no headline number may rest on a single run**.

## 5. Limitations

- One run of each check. For everything except `accuracy_vs_disorder` (which reproduced a prior run)
  that is a single draw — see the B-1 note in §1.
- `Phase 2/verification_open_domain` is a directory of PNG outputs, **not** a runnable check; the plan
  listed it as one. Open-boundary verification remains incomplete — audit **A-8**.
- Contraction-isolation / **A-14** are parked at the user's direction and were not examined.
- The legacy `verification_tools` island does not run at all (`verification_tools/README.md` §3) and
  was deliberately excluded.
