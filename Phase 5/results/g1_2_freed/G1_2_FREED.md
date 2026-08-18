# g1_2_freed — the "freed" variant: can ν go negative if the bracing is released?

**Script** `Phase 5/verifications/run_g1_2_freed.py` · companion to
[`../g1_2/`](../g1_2/) (`run_g1_2.py`)

## What

`g1_2` braces its fictional (triangulating) edges at k=1. This variant **frees** them, asking
whether the auxetic targets `g1_2` could not reach become reachable once the bracing is released.
110 runs, 10 topologies, ν targets from −0.95 to +0.90.

## Key numbers

```
runs 110      trustworthy 33      topologies 10

nu_achieved_sim over ALL runs        : [-0.761, +0.950]
nu_achieved_sim over TRUSTWORTHY runs: [+0.100, +0.340]
```

**That gap is the result.** Freeing the bracing *does* let the optimiser produce strongly negative ν
— down to −0.761 — but **every such design fails the independent-sim honesty check**. Restricted to
designs the sim confirms, the reachable set is [+0.100, +0.340]: the same positive-only window
`run_g1_2` reports with the bracing in place.

So the two experiments agree from opposite directions: **within this family, ν < 0 is reachable only
by entering the near-mechanism regime where the linear read-back cannot be trusted.** Freeing the
bracing changes what the optimiser will *claim*, not what the network will *do*.

## Limitations

- Only 33/110 runs are trustworthy, so the trustworthy window rests on a third of the data.
- The lower edge (+0.100) equals the smallest positive target on the grid, so it is set by **what
  was sampled**, not by what is achievable — the true edge is unknown from this run.
- Untrustworthy designs are saved with an `UNTRUSTED_` filename prefix. They are kept as evidence of
  the failure mode and **must not** be read as designs.
- Does not contradict `run_goal2`, which *does* land a negative-ν **full-tensor** target
  trustworthily. The difference is the target class and the topology pool, not the sign of ν.

## Outputs

`results.csv` (110 rows, per-run), `results.npz`, `topologies.csv` (the 10 topologies with
coordination signatures), `eta_reference.npz`.
