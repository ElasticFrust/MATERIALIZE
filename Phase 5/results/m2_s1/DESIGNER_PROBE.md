# A0.3 — designing THROUGH the frozen surrogate, against M1

**Producer:** `Phase 5/verifications/m2_designer_probe.py`
**Data:** `designer_probe_full.json` (this directory)
**Model:** `checkpoint_v3_res_bravais_w10_h1_L5_ns48_h64_e400_star_ms.pt`, frozen
**Population:** `dataset_fresh_s4321.npz` — fresh UNSEEN meshes, ≤120 nodes, self-loops excluded
**Run:** 2026-09-15, 20 targets, float64, `OMP_NUM_THREADS=1`, seed 0

---

## Why this test exists

A0.2 measured `cos(∇L_solver, ∇L_gnn)` pointwise. But a cosine is a statement about one point, and
**descent is a statement about a trajectory** — one that deliberately walks toward wherever the
surrogate claims the loss is low, which is exactly where a surrogate is most likely to be wrong. So
this is adversarial in a way random-sample scoring cannot be.

## Method

Both arms run the **identical** optimiser: L-BFGS + strong Wolfe on `raw`, squashed through softplus
(`inverse_design._softplus`), from the **same** initialisation with the **same** `reg = 0.02`.
**Only the forward model differs.** Targets are isotropic ν∈[−0.45, +0.60] (inside the measured reach
envelope) with `E = fE·E_base`, `fE ∈ [0.7, 1.3]`, so every request is reachable.

The question avoids budget-matching entirely:

> **How many exact-solver calls does M1 need to reach the design the surrogate produces for free?**

M1's best-so-far trace *is* a quality-vs-solver-calls curve, because every closure call is one solver
call. The surrogate's optimisation costs **zero**; one call then scores it.

## Result

| | median loss vs target (solver) | vs INDEPENDENT SIM | solver calls |
|---|---|---|---|
| start, no design | 4.526e-01 | 4.526e-01 | 0 |
| **GNN, free** | **6.412e-02** | **6.047e-02** | **0** |
| GNN + 4 refine rounds | 6.371e-02 | 6.027e-02 | 4 |
| M1, full run | 2.779e-02 | 2.779e-02 | **86** (range 34–101) |

> ### The S4 number
> **M1 needs a median of 9 exact-solver calls to match the design the surrogate produces for free** —
> matched on **20 of 20** targets (min 1, max 32), against the **86** M1 spends in a full run.

- **Within ~2× of a full M1 run, for nothing:** floored ratio **1.82**. Unfloored median 1.82, but
  **p90 = 20.2** — a heavy tail, so the median is the honest summary and the mean would not be.
- Improves on the starting network on **90 %** (18/20); **beats M1 on 0 %**.
- The independent sim tracks the solver throughout (6.05e-02 vs 6.41e-02), as expected from the
  measured near-zero solver-vs-sim floor on these families. It is a real second path, not a copy —
  the per-row values differ beyond 1e-12.

### Stratified by `max|W|`, where A0.2 predicted trouble

| max\|W\| | n | GNN free | M1 | ratio (floored) |
|---|---|---|---|---|
| < 3 | 12 | 5.188e-02 | 2.779e-02 | **0.82** |
| 3–10 | 6 | 9.952e-02 | 3.558e-02 | 0.96 |
| 10–100 | 2 | 1.946e-01 | 2.896e-02 | **6.10** |

Consistent with A0.2's gradient result — the surrogate degrades with `max|W|` — though n=2 in the top
bin carries no weight on its own.

### The two failures, and one of them is NOT the tail

On **2 of 20 targets the surrogate's design was WORSE than doing nothing**:

| family | ν* | start → GNN free | max\|W\| | n_tri |
|---|---|---|---|---|
| `tiling` | −0.01 | 8.895e-01 → **4.004e+00** | **0.8** | 64 |
| `disordered` | +0.21 | 1.886e-01 → 3.148e-01 | 11.8 | 96 |

The `disordered` case fits the `max|W|` story. **The `tiling` one does not** — `max|W| = 0.8` puts it
in the *easiest* bin, where the surrogate is otherwise better than M1. What `tiling` does have is
almost no training data: **0.2 % of `dataset_v2_s0`** (92 of 41 431 samples). So the worst failure
here looks like a **family-coverage** failure rather than a near-mechanism one — which is a direct
argument for A1's rebalancing, and a reminder that `max|W|` is not the only axis of risk.

**Consequence for a designer:** one solver call is enough to catch this (keep the better of start and
proposal), and that is the cheapest possible insurance. It also means the surrogate must not be used
open-loop.

## Limitations

- **`k` channel only.** M1's position channel is SPSA — a different and far more expensive baseline —
  so that comparison is its own experiment. A0.2 measured the position *gradient* at median cos 0.915.
- **The refinement scheme failed, and that is MY scheme, not a property of the surrogate.**
  6.412e-02 → 6.371e-02 with a **75 %** reject rate. It re-optimises from a *randomly perturbed
  restart*, which is a weak way to spend a solver call; a real trust region would line-search along
  the surrogate's proposed step. Read this row as "this refinement fails", not "refinement cannot
  help".
- **n = 20**, one target family (isotropic ν with a scaled E), one frozen checkpoint, ≤120 nodes.
- **Both arms use `n_restarts = 1`.** The production designer uses 3, so M1 here is weaker than
  `design_on_topology` in full configuration.

## Reading, against A0's pre-registered rule

The rule's green branch required median cos > 0.9 on `k` (**met**, 0.950), > 0.7 on positions
(**met**, 0.915), **and** "A0.3 matches M1 within its own scatter". **That last clause is NOT met** —
the surrogate lands ~1.8× above a full M1 run and never beats it.

What *is* established: the surrogate is a **strong initialiser and pre-filter — worth ~9 exact solves,
free, on every target tried — but not a replacement for the solver.** That is precisely the role S4
specifies for it, and it is enough for the edit-policy's critic, which needs ranking and gradients
rather than final answers. It is not enough to design open-loop.
