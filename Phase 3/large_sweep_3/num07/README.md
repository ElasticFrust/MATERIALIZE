# Target Poisson Ratio: nu = -0.7

## Run Configuration

- **Target nu**: -0.7
- **Mesh size**: (10, 10)
- **Design variables**: rigidities, rest_lengths, both
- **Optimizer**: L-BFGS, lr=0.05, tol=1e-14, max_iter=500
- **Random restarts per case**: 3 (seeds: 7, 144, 281)

## Summary

- **Total cases**: 24 (8 topologies x 3 design variables)
- **Converged (loss < 0.01)**: 21/24
- **Best result**: aniso_crystal / rigidities (loss = 1.60e-23)

## Results by Topology

### iso_crystal
_Isotropic crystal — regular triangular lattice, shape=(1,1), orientation=0_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | +0.000000 | 4.90e-01 | No | 520 | 1.1 |
| rest_lengths | +0.000000 | 4.90e-01 | No | 520 | 3.5 |
| both | +0.000000 | 4.90e-01 | No | 520 | 3.9 |

### aniso_crystal
_Anisotropic crystal — stretched lattice, shape=(1.5, 0.8), orientation=pi/6_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | -0.700000 | 1.60e-23 | Yes | 40 | 0.2 |
| rest_lengths | -0.702166 | 4.69e-06 | No | 520 | 2.3 |
| both | -0.700000 | 4.48e-17 | Yes | 60 | 0.8 |

### foam_eta02_42
_Foam eta=0.2 (seed 42) — Delaunay BEFORE perturbation_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | -0.705278 | 2.79e-05 | No | 520 | 4.0 |
| rest_lengths | -0.699997 | 7.44e-12 | Yes | 520 | 3.0 |
| both | -0.700000 | 1.78e-16 | Yes | 60 | 1.1 |

### foam_eta02_137
_Foam eta=0.2 (seed 137) — Delaunay BEFORE perturbation_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | -0.700000 | 2.17e-15 | Yes | 80 | 1.4 |
| rest_lengths | -0.700000 | 5.73e-14 | Yes | 520 | 3.0 |
| both | -0.700000 | 3.31e-15 | Yes | 80 | 1.7 |

### foam_eta045_256
_Foam eta=0.45 (seed 256) — Delaunay BEFORE perturbation_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | -0.699933 | 4.53e-09 | No | 520 | 1.9 |
| rest_lengths | -0.700034 | 1.17e-09 | No | 520 | 1.2 |
| both | -0.700087 | 7.64e-09 | No | 520 | 1.7 |

### foam_eta045_314
_Foam eta=0.45 (seed 314) — Delaunay BEFORE perturbation_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | -0.699999 | 9.18e-13 | Yes | 520 | 1.2 |
| rest_lengths | -0.706542 | 4.28e-05 | No | 520 | 1.6 |
| both | -0.685955 | 1.97e-04 | No | 520 | 4.4 |

### poisson_999
_Poisson random network (seed 999) — Delaunay AFTER perturbation, eta=0.3_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | -0.700000 | 5.31e-15 | Yes | 40 | 0.3 |
| rest_lengths | -0.700000 | 2.39e-16 | Yes | 80 | 1.0 |
| both | -0.700000 | 3.37e-16 | Yes | 60 | 1.0 |

### poisson_1337
_Poisson random network (seed 1337) — Delaunay AFTER perturbation, eta=0.3_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | -0.700000 | 2.00e-16 | Yes | 40 | 0.3 |
| rest_lengths | -0.700000 | 1.45e-17 | Yes | 80 | 0.8 |
| both | -0.700000 | 2.31e-16 | Yes | 60 | 0.9 |
