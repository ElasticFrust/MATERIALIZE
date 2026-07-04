# Forward-solver verification — summary

Verification of `Phase 2/forward_solver_torch.py` (`method='intrinsic'`) against the
PBC spring-network simulation. Run on CPU torch 2.12.1.

## What was checked

**1. Unit regression** — `Phase 2/test_forward_solver.py` (5/5 pass):
- Regular triangular lattice, uniform k → **ν = 0.333333 exactly** (1/3), E > 0.
- Foam disorder → finite, no NaN.
- Intrinsic **gradients flow**; **autograd matches finite-difference to 6 digits**.
- Legacy `method='woodbury'` still runs.

**2. Sim-vs-solver sweep** — `verify_solver_sweep.py`, 20×20, 3 realisations,
η = 0 → 0.5 (step 0.1), for geometric disorder and virtual-distortion (VD) rigidity
contrasts `k = 1 + tanh(a(|R|−1))`, a ∈ {−10, −2, +5, +10, +100}.
Plot: `plots/sim_vs_solver_sweep.png`.

## Result

| Case | Solver vs simulation |
|---|---|
| Disordered (geometric, k=1) | ν, E overlap through η=0.5; hairline ν gap only at η=0.5 |
| VD a = −10, −2, +5, +10 | ν, E track the simulation within ±std across all η; tiny divergence only at η=0.5 |
| VD a = +100 (near-binary) | breaks down at η ≳ 0.3 — simulation ν/E diverge with large variance, solver stays bounded |

**Conclusion: the solver is not broken.** It reproduces the simulation for geometric
disorder and moderate rigidity contrast (|a| ≤ 10) through η = 0.5. The a = +100
near-binary case is the **known** limitation (correlation length exceeds the solver's
per-triangle/edge/vertex locality — see `ANALYTICAL_MODEL_STATUS.md` §7.2); the large
simulation variance there also reflects near-percolation singularity in the ground truth.

## Reproduce

```
# env: CPU torch; set KMP_DUPLICATE_LIB_OK=TRUE on conda+MKL
python "Phase 2/test_forward_solver.py"
python verification_tools/verify_solver_sweep.py     # full: 20×20 ×10 + 50×50; heavier
```

The reduced first-pass plot here used 3 realisations at η step 0.1. `verify_solver_sweep.py`
as-is runs the full 10-realisation 20×20 ensemble plus a single 50×50.
