# MATERIALIZE

Forward homogenisation and inverse design of 2D elastic spring networks
(mechanical metamaterials). Given a triangulated network of springs, compute its
macroscopic elastic response (effective tensor, Poisson's ratio, Young's modulus);
the goal is the inverse — design microstructures that realise a target response,
including auxetic (negative-Poisson) behaviour.

## Repository layout (after the 2026 cleanup)

Core:
- **`Phase 2/forward_solver_torch.py`** — the canonical, differentiable forward
  solver. Default `method='intrinsic'`: a metric-space solve with edge (length),
  curvature (angle), and area-weighted-mean constraints that reproduces the PBC
  simulation. Legacy `method='woodbury'` is the old single-site mean field.
- **`Disc_2_Cont_optimized.py`** — mesh/topology generators only (crystal, foam).
  The old NumPy mean-field solver that used to live here was removed (git history).

Docs / theory:
- **`ANALYTICAL_MODEL_STATUS.md`** — what the solver does and how it works.
- **`INTRINSIC_METRIC_SOLVE.md`** — full derivation of the intrinsic solve
  (energy, the three constraints, KKT system, correctness proof).
- **`THEORY_NOTES.md`** — why the single-site mean field fails.
- **`derivation_edge_compatibility.pdf`** — edge-compatibility derivation (G&B notation).
- **`Tutorial.md`** — GNN + VAE background (for the planned ML phase).

Planned / in flux:
- **`Phase 4/PLAN.md`** — the GNN-surrogate + CVAE inverse-design plan.
  **Out of date — update against the current solver before use.**
- **`Phase 4/data/`** — topology generators + rigidity patterns (kept; scope TBD).

## Quick start

```python
import sys; sys.path.insert(0, 'Phase 2')
import Disc_2_Cont_optimized as D2C
from forward_solver_torch import from_triangulation

tri = D2C.generate_foam_points(size=(4, 4), eta=0.2)
solver, k, l0 = from_triangulation(tri)
res = solver(k, l0)                      # method='intrinsic' by default
print(res['poisson'].item(), res['young'].item())
```
