# Benchmarking Session Summary — June 2026

## What we did

### 1. Found and fixed three bugs in `_woodbury_kkt_sparse_combined`

The refactored unified KKT function (`Phase 2/forward_solver_torch.py`) had three bugs
relative to the original `_woodbury_kkt_sparse`:

| Bug | Location | Effect |
|-----|----------|--------|
| Spurious `/N` on `Vy3`, `gs3` | non-AW Woodbury branch | Woodbury correction ~800× too small (effectively disabled) |
| K0 einsum transposed (`'el,elm'` → `'el,eml'`) | edge-KKT K0 matrix | ~7% error in W for edge methods |
| K0 einsum transposed (`'pl,plm'` → `'pl,pml'`) | angle-KKT K0 matrix | ~25% error in W for angle methods |

Also added explicit `float64` casting at entry to `forward()` to prevent silent
float32 degradation from PyTorch training callers.

The original `_woodbury_kkt_sparse` (pre-refactor) was always correct — bugs were
introduced during the refactor to the combined function.

### 2. Verified fixes

- Before/after comparison plot confirms new `_woodbury_kkt_sparse_combined` matches
  the original `_woodbury_kkt_sparse` to numerical precision after all three fixes.
- Plot: `benchmarking/new/before_after_fix.png`

### 3. Ran 20×20 all-methods benchmark

Six MF variants vs. spring-network simulation, 20×20 foam, 10 trials, η = 0–0.5:

**Methods:** Std, AW, Std+edge, AW+edge, Std+full (edge+angle KKT), AW+full  
**Mesh types:** DF (distort-first Delaunay) and TF (triangulate-first)  
**Plot:** `benchmarking/new/all6_comparison.png`  
**Data:** `mf20_{df,tf}_data.npz`, `mf20_full_{df,tf}_data.npz`, `sim20_{df,tf}_data.npz`

### 4. Key finding: MF fails for E at large geometric disorder

Even with uniform k=1 springs (geometric disorder only, no rigidity heterogeneity),
all MF methods dramatically underpredict E at high η:

- Simulation: E/E₀ drops ~13% from η=0 to η=0.5 (DF mesh)
- Best MF method (AW): drops ~73%
- All other methods: 97–99% drop or divergence

This updates a previous assumption that "geometric disorder alone is well-captured."
That was only true for η ≤ 0.20.

For ν: MF qualitatively tracks the trend at low-to-moderate η but overshoots
negative values (or diverges) at high η.

### 5. Why the MF keeps missing

- **E**: The non-affine W correction grows with disorder and can reduce C_eff → 0.
  The real network has a stiff backbone that is invisible to per-triangle averaging.
- **KKT edge constraints**: Amplify the problem by coupling frustrated triangles;
  large constraint multipliers further reduce C_eff.
- **Angle constraints**: Rest metrics don't satisfy Σθ=2π in disordered foam
  (intrinsic frustration), making the angle-KKT Gram matrix near-singular at η ≥ 0.30.

### 6. Current state of the solver

| Feature | Status |
|---------|--------|
| Non-AW Woodbury (`Std`, `AW`) | Correct, matches original |
| Edge KKT (`Std+edge`, `AW+edge`) | Correct after /N and einsum fixes |
| Angle KKT (`Std+full`, `AW+full`) | Correct numerically; diverges at η ≥ 0.30 due to physics (frustrated reference) |
| float64 precision | Enforced at `forward()` entry |

## Files changed

| File | Change |
|------|--------|
| `Phase 2/forward_solver_torch.py` | /N fix, float64, two K0 einsum fixes |
| `benchmarking/run_20x20_comparison.py` | New: 4-method MF vs sim |
| `benchmarking/run_before_after_fix.py` | New: before/after KKT fix |
| `benchmarking/run_all6_comparison.py` | New: all 6 methods |
| `benchmarking/new/comparison_20x20.png` | 4-method result plot |
| `benchmarking/new/before_after_fix.png` | Before/after verification plot |
| `benchmarking/new/all6_comparison.png` | All-6-methods result plot |
| `ANALYTICAL_MODEL_STATUS.md` | Updated with bug fixes and benchmark results |

## What's next

The MF + KKT framework is essentially exhausted for E accuracy at large η.
Options going forward:

1. **GNN / Phase 4**: Learn the microstructure→property map directly from simulation.
   This sidesteps the MF breakdown entirely.
2. **Cluster CPA**: Embed 2–3 triangle clusters instead of single triangles;
   captures short-range correlations without full simulation.
3. **Restrict to η ≤ 0.20**: In this regime MF is accurate (E error < 10%, ν error < 5%)
   and the pipeline is reliable for inverse design.
