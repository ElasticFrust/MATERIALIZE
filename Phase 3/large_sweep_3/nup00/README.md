# Target Poisson Ratio: nu = +0.0

## Run Configuration

- **Target nu**: +0.0
- **Mesh size**: (10, 10)
- **Design variables**: rigidities, rest_lengths, both
- **Optimizer**: L-BFGS, lr=0.05, tol=1e-14, max_iter=500
- **Random restarts per case**: 3 (seeds: 7, 144, 281)

## Summary

- **Total cases**: 24 (8 topologies x 3 design variables)
- **Converged (loss < 0.01)**: 22/24
- **Best result**: aniso_crystal / rigidities (loss = 2.81e-20)

## Results by Topology

### iso_crystal
_Isotropic crystal — regular triangular lattice, shape=(1,1), orientation=0_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | +0.000000 | 9.91e-14 | Yes | 520 | 0.4 |
| rest_lengths | +0.000001 | 3.59e-13 | Yes | 520 | 3.9 |
| both | +0.000001 | 1.14e-12 | Yes | 520 | 5.1 |

### aniso_crystal
_Anisotropic crystal — stretched lattice, shape=(1.5, 0.8), orientation=pi/6_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | -0.000000 | 2.81e-20 | Yes | 40 | 0.3 |
| rest_lengths | -0.000000 | 5.89e-17 | Yes | 60 | 0.7 |
| both | -0.000000 | 4.09e-17 | Yes | 40 | 0.4 |

### foam_eta02_42
_Foam eta=0.2 (seed 42) — Delaunay BEFORE perturbation_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | -0.000000 | 5.14e-17 | Yes | 60 | 0.7 |
| rest_lengths | +0.000000 | 3.97e-15 | Yes | 60 | 0.7 |
| both | -0.000000 | 1.34e-17 | Yes | 60 | 0.9 |

### foam_eta02_137
_Foam eta=0.2 (seed 137) — Delaunay BEFORE perturbation_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | -0.000000 | 1.32e-17 | Yes | 60 | 0.6 |
| rest_lengths | +0.000000 | 4.21e-15 | Yes | 60 | 0.7 |
| both | +0.000000 | 1.19e-17 | Yes | 40 | 0.3 |

### foam_eta045_256
_Foam eta=0.45 (seed 256) — Delaunay BEFORE perturbation_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | -0.000758 | 5.74e-07 | No | 520 | 3.2 |
| rest_lengths | -0.000741 | 5.49e-07 | No | 520 | 3.2 |
| both | -0.000788 | 6.21e-07 | No | 520 | 4.5 |

### foam_eta045_314
_Foam eta=0.45 (seed 314) — Delaunay BEFORE perturbation_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | -0.001659 | 2.75e-06 | No | 520 | 1.0 |
| rest_lengths | -2.017671 | 4.07e+00 | No | 520 | 3.6 |
| both | +0.101879 | 1.04e-02 | No | 520 | 3.4 |

### poisson_999
_Poisson random network (seed 999) — Delaunay AFTER perturbation, eta=0.3_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | +0.000000 | 1.47e-17 | Yes | 40 | 0.2 |
| rest_lengths | +0.000000 | 3.96e-17 | Yes | 60 | 0.6 |
| both | -0.000000 | 1.94e-17 | Yes | 60 | 0.9 |

### poisson_1337
_Poisson random network (seed 1337) — Delaunay AFTER perturbation, eta=0.3_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | +0.000000 | 1.78e-17 | Yes | 60 | 0.6 |
| rest_lengths | +0.000000 | 1.54e-17 | Yes | 60 | 0.6 |
| both | -0.000000 | 2.87e-17 | Yes | 60 | 0.9 |
