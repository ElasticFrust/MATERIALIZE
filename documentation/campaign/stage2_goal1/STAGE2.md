# Campaign Stage 2 + 2b — `goal1` re-run, and the positive-ν frontier

**Provenance.** Stage 2 at commit `5987e70`, **clean tree**; Stage 2b at `9b7c7eb`, clean.
Seeds `topo=0`, `shuffle=12345` (now named constants, stamped into every artifact — audit B-3).
Logs: `run_goal1.log` here, `../stage2b_frontier/run_frontier.log`.
Per-experiment docs: `Phase 5/results/goal1/GOAL1.md` (corrected),
`Phase 5/results/goal1_frontier/GOAL1_FRONTIER.md` (new).

---

## 1. Stage 2 — `goal1` re-run under working instruments

110 runs, 76.6 min. This is the first measurement of `goal1` with the A-0 shear fix, the A-10 ν,E
convention, the independent oracle (A-9/A-1), and honest drivers (A-11/A-12).

| | old (pre-fix) | **new** |
|---|---|---|
| trustworthy | 100/110 | **89/110** |
| median ν-error (k+pos) | 0.082 | **0.098** |
| success \|err\| < 0.05 | 42 % | **44.9 %** |

**Reachable ν window by contrast band** (matched comparison — same topologies, same seeds):

| band | f | old | **new** | negative reach |
|---|---|---|---|---|
| soft | 0.00 | [−0.82, +0.45] | **[−0.78, +0.45†]** | retreated 0.04 |
| large | 0.10 | [−0.31, +0.44] | **[−0.18, +0.45†]** | retreated 0.13 |
| medium | 0.50 | [−0.00, +0.33] | **[+0.07, +0.39]** | retreated 0.07 |
| small | 0.90 | [+0.05, +0.34] | **[+0.16, +0.34]** | retreated 0.11 |
| none | 0.99 | [+0.16, +0.33] | **[+0.20, +0.33]** | retreated 0.04 |

† still censored by this grid — resolved in §2.

### Two findings

**(a) The auxetic reach retreated in ALL FIVE bands.** That is the **A-0 signature**: the shear defect
*overstated auxeticity*, exactly as the register predicted. Five out of five moving the same direction
is very unlikely to be optimiser noise — though the individual magnitudes do carry that uncertainty,
since the optimiser is nondeterministic (B-1 stage 1). **The networks were never as auxetic as the old
campaign claimed.**

**(b) Untrustworthy runs more than doubled, 10 → 21 — and that is the fixes WORKING.** The old gap was
computed with the wrong ν convention (A-10) and against a partly self-referential oracle (A-1), so it
systematically *understated* solver-sim disagreement. 19 % of designs are now correctly flagged where
the old instrument saw 9 %. The old headline "100/110 trustworthy" was an artifact of a blunt
instrument, not a better result.

**Unaffected:** the central claim — *auxeticity requires stiffness contrast; clamp `f` toward uniform
and the negative-ν region disappears* — holds, and the monotone collapse from [−0.78, +0.45] at f=0 to
[+0.20, +0.33] at f=0.99 is if anything sharper than before.

## 2. Stage 2b — the "+0.45 ceiling" was the grid edge

`goal1`'s ν grid stops at +0.45, so its reported *maximum* of +0.45 measured **where the search
stopped**, not what the networks can do. (The negative end is genuine: the grid runs to −0.9 and the
designer falls short at −0.78.) Probed with ν up to 0.95, 24 runs, 15.1 min:

- **`f=0` reaches ν = +0.906**, with solver-sim gap **0.0000** at several points (ν*=0.80 → +0.8007,
  err 0.0007; ν*=0.90 → +0.8915, err 0.0085). The ceiling was **twice** what was reported.
- **`f=0.1` has a REAL frontier at ≈ +0.63** — targets 0.80/0.90/0.95 all saturate at +0.59…+0.63 with
  gap 0.0000. Measured, not censored.

**Corrected `f=0` window: [−0.78, +0.91]** — near-symmetric. Roughly **half the positive half of the
design space was invisible** to every previous `goal1` run.

The positive frontier depends on stiffness contrast exactly as the negative one does, which
*strengthens* `goal1`'s claim: ν → +1 requires the shear modulus G → 0 and ν → −1 requires the area
modulus K → 0, both achievable only through bond-level contrast, so a floor `f` truncates both ends.

High ν is also the **shear-soft** channel — the one A-0 corrupted — so a clean result there is a
strong end-to-end check on that repair.

## 3. Verdict

**Stage 2 and 2b both complete.** `goal1` is re-measured, its censored frontier is resolved, and both
per-experiment docs are written (closing part of **C-1** for this experiment, as planned).

Carried forward: the B-1 constraint — no headline number here rests on a single run; every frontier
value above is supported by several agreeing runs, and only trustworthy runs are used.

## 4. Limitations

- The optimiser is nondeterministic, so a *single* run's achieved ν carries spread; all claims here
  rest on multiple runs.
- 7 of 24 frontier runs were untrustworthy, concentrated at ν ≥ 0.70 in `soft` (gaps up to 0.93) —
  consistent with the `A(s)` validity limit (`CLAUDE.md` §3): extreme ν at f=0 drives bonds to zero,
  which breaks per-triangle invertibility. Excluded from the frontier numbers.
- Only `f ∈ {0.0, 0.1}` probed above +0.45; higher-`f` frontiers come from `goal1`, where they are
  genuine.
- Isotropic scalar ν only. Directional targets are `goal2`'s job (Stage 4).
