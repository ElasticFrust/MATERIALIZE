# Target Poisson Ratio: nu = +0.9

## Run Configuration

- **Target nu**: +0.9
- **Mesh size**: (10, 10)
- **Design variables**: rigidities, rest_lengths, both
- **Optimizer**: L-BFGS, lr=0.05, tol=1e-14, max_iter=500
- **Random restarts per case**: 3 (seeds: 7, 144, 281)

## Summary

- **Total cases**: 24 (8 topologies x 3 design variables)
- **Converged (loss < 0.01)**: 19/24
- **Best result**: poisson_1337 / rigidities (loss = 1.79e-22)

## Results by Topology

### iso_crystal
_Isotropic crystal — regular triangular lattice, shape=(1,1), orientation=0_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | +0.900000 | 1.80e-17 | Yes | 60 | 0.6 |
| rest_lengths | +0.900000 | 3.87e-17 | Yes | 60 | 0.6 |
| both | +0.900000 | 1.96e-17 | Yes | 60 | 0.8 |

### aniso_crystal
_Anisotropic crystal — stretched lattice, shape=(1.5, 0.8), orientation=pi/6_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | +0.900000 | 1.50e-17 | Yes | 40 | 0.3 |
| rest_lengths | +0.900000 | 1.83e-18 | Yes | 40 | 0.2 |
| both | +0.900000 | 2.20e-17 | Yes | 40 | 0.4 |

### foam_eta02_42
_Foam eta=0.2 (seed 42) — Delaunay BEFORE perturbation_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | +0.900000 | 3.97e-17 | Yes | 60 | 0.6 |
| rest_lengths | +0.900000 | 6.83e-16 | Yes | 60 | 0.7 |
| both | +0.900000 | 1.48e-17 | Yes | 60 | 0.8 |

### foam_eta02_137
_Foam eta=0.2 (seed 137) — Delaunay BEFORE perturbation_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | +0.900000 | 2.05e-17 | Yes | 60 | 0.6 |
| rest_lengths | +0.900000 | 9.86e-16 | Yes | 60 | 0.7 |
| both | +0.900000 | 5.25e-16 | Yes | 60 | 0.9 |

### foam_eta045_256
_Foam eta=0.45 (seed 256) — Delaunay BEFORE perturbation_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | -0.000707 | 8.11e-01 | No | 520 | 3.2 |
| rest_lengths | -0.000722 | 8.11e-01 | No | 520 | 3.3 |
| both | -0.000762 | 8.11e-01 | No | 520 | 3.8 |

### foam_eta045_314
_Foam eta=0.45 (seed 314) — Delaunay BEFORE perturbation_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | -0.001701 | 8.13e-01 | No | 520 | 2.9 |
| rest_lengths | +0.867071 | 1.08e-03 | No | 520 | 2.0 |
| both | NaN | 1.18e-02 | No | 520 | 11.8 |

### poisson_999
_Poisson random network (seed 999) — Delaunay AFTER perturbation, eta=0.3_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | +0.900000 | 3.22e-17 | Yes | 60 | 0.5 |
| rest_lengths | +0.900000 | 6.74e-17 | Yes | 60 | 0.6 |
| both | +0.900000 | 2.25e-14 | Yes | 520 | 1.8 |

### poisson_1337
_Poisson random network (seed 1337) — Delaunay AFTER perturbation, eta=0.3_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | +0.900000 | 1.79e-22 | Yes | 40 | 0.2 |
| rest_lengths | +0.900000 | 1.38e-16 | Yes | 60 | 0.6 |
| both | +0.900000 | 1.38e-17 | Yes | 60 | 0.7 |
