# Campaign Stage 3 — `g1_2` re-run, and a defect found in the experiment itself

**Provenance.** Commit `a98a249`, clean tree. Seeds as recorded by the driver. 110 runs, logs here.

> ## ⚠ STATUS: the physics conclusions of this stage are WITHHELD
> A defect was found in `g1_2`'s topology construction (§3) that invalidates its result for 4 of its
> 10 topologies. The instrument findings (§1, §2) stand; the "what can positions achieve" conclusion
> does not, and is not stated here. **Re-run required** — see §5.

---

## 1. What the re-run measured

110 runs, 0 failures. Against the correct instruments: **~24 % trustworthy** (solver-vs-sim gap
< 0.05), median gap 0.394, max 7.34.

The old campaign saved 53 of 110 as "trustworthy" using an instrument later found ~200× too lenient,
discarding 58 unrecoverably. The old figure was a blunter measurement, not a better result. **Had
A-12 not been fixed, this run would have saved ~21 designs and discarded the 89 that carry the
finding.**

## 2. The extreme gaps are REAL — verified independently

The worst stored gap (7.34) was re-computed from the saved network, outside the driver:

```
UNTRUSTED_g12_rotating_squares_m095.npz
   solver nu = -0.6764        sim nu = +0.1887        |dnu| = 0.8651
   min shape quality 0.046    (median 0.823 over 66 triangles)
   min/mean area     0.061    k uniform: True
```

One triangle at quality 0.046 among 65 near 0.82. **The solver reported a strongly auxetic success
where the truth is positive — opposite sign.** The existing area guard sat **61× away from firing**
(floor 1e-3, actual 0.061), which is why it never triggered.

This is the *geometric* route to `A(s)` rank loss (`CLAUDE.md` §3) — k is uniform here, so the
stiffness route is impossible by construction. It is an independent confirmation of that validity
condition through the other channel.

## 3. THE DEFECT — the auxetic motifs are braced, so they are not auxetic

`build_topologies` states it plainly: *"each a FULLY triangulated periodic network with **ALL edges
real** (k=1)"*. Four of the ten topologies (`honeycomb`, `square_octagon`, `rotating_squares`,
`reentrant_honeycomb`) are **not triangulations**; they require *fictional* edges purely to
triangulate them. Setting k=1 on those welds the hinges shut.

Measured directly:

| motif | fictional edges | all real, k=1 | **bracing freed (k=1e-6)** |
|---|---|---|---|
| rotating squares | 16 / 101 | solver +0.2660, sim +0.2890 | **solver −0.9805, sim −1.0000** |
| reentrant honeycomb | **98 / 194** | solver +0.2629, sim +0.3032 | **solver −1.0443, sim −1.0833** |

**The motifs are strongly auxetic (ν ≈ −1) and the solver evaluates them accurately** (−0.98 vs
−1.00; −1.04 vs −1.08). Nothing is wrong with a re-entrant structure or with the solver's handling of
it. For `reentrant_honeycomb`, **over half the network (98/194 bonds) is scaffolding**, and bracing it
converts a ν ≈ −1.08 auxetic into a ν ≈ +0.30 truss.

So `g1_2` engineers the defining property out of four topologies *before the experiment begins*, then
reports what positions-only design can achieve on them. **That result is an artifact of the bracing.**

Additional finding from the same check — the **undistorted** structures, before any position move:

| topology | solver ν | sim ν | gap | min quality |
|---|---|---|---|---|
| `square_octagon` | +0.0966 | +0.3177 | **0.2211** | 0.4524 |
| `reentrant_honeycomb` | +0.2629 | +0.3032 | 0.0403 | 0.5000 |
| `rotating_squares` | +0.2660 | +0.2890 | 0.0231 | 0.8660 |
| (triangular, kagome, tetrakis, foam, flipped ×2) | — | — | **0.0000** | — |

`square_octagon` is **already broken with no position optimisation at all**. For that topology the
problem is structural, not the search.

## 4. Stage 3b — the shape-quality floor (PROVISIONAL)

`spsa_positions` gained a `quality_floor` parameter (default **0.0 = off**), rejecting steps that
drive any triangle below a shape-quality bound. Motivation: the existing area guard never fires
(untrusted designs sit at median min/mean area 0.047, **47× above** its 1e-3 floor), and the measured
predictors of log10(gap) over 124 saved designs are

    min shape quality -0.523    min ANGLE -0.505    min/mean AREA -0.418    min EDGE LENGTH -0.107

so constraining the smallest **distance** would do almost nothing — a sliver has long edges and a tiny
angle. A/B, 6 topologies × 2 targets × 4 floors:

| floor | trustworthy | median gap | median err |
|---|---|---|---|
| 0.000 | 17 % | 0.660 | 0.189 |
| 0.020 | 42 % | 0.419 | 0.254 |
| 0.050 | **50 %** | 0.367 | 0.274 |
| 0.100 | 50 % | 0.381 | 0.288 |

Trustworthiness triples; achieved error rises 52 %. **No default is being set from this**: three of the
six topologies are the braced motifs of §3, so the improvement partly reflects designs that were
already compromised. `quality_floor` ships **off**.

*Method note:* the first version of this A/B returned byte-identical results across all four floors,
which reads as "the floor does nothing". It was invalid — it ran a reduced search
(`n_outer=2, spsa_steps=20`) that never distorted geometry (final min quality ~0.72), so the floor had
nothing to reject. **A guard can only be tested under conditions that trigger the failure.** The
positive control (`minq`) was the tell. Fixed to drive the real `design_one` search.

## 5. What must happen before Stage 3's physics can be stated

1. **Re-run `g1_2` with fictional edges at near-zero k**, not 1 — the configuration the
   `is_fictional` mask exists for, and the one where the motifs are actually auxetic.
2. **Re-run the 3b A/B** on the corrected topology set, then set `quality_floor`'s default from it.
3. Investigate `square_octagon`'s undistorted gap of 0.221 separately — it is not a search problem.

## 6. Corrections made to my own earlier claims in this stage

- **"Positions-only at uniform k cannot reach ν < 0" — WRONG, twice.** (i) It read a
  *trustworthy-filtered* table as physics: 28 of 110 runs reached ν < 0, down to **−0.2945**, all
  flagged untrustworthy. (ii) The run's own η-disorder reference band reaches **ν = −0.115** at uniform
  k, printed in the same log — the established result (`CLAUDE.md`: η-disorder drives ν +1/3 → ~−0.11).
  Position change at uniform k demonstrably produces auxeticity.
- The remaining question — whether those 28 sim-auxetic designs are genuine or artifacts — is open,
  and cannot be settled on a topology set with the §3 defect.
