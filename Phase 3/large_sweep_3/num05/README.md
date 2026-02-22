# Target Poisson Ratio: nu = -0.5

## Run Configuration

- **Target nu**: -0.5
- **Mesh size**: (10, 10)
- **Design variables**: rigidities, rest_lengths, both
- **Optimizer**: L-BFGS, lr=0.05, tol=1e-14, max_iter=2000
- **Random restarts per case**: 3 (seeds: 7, 144, 281)

## Summary

- **Total cases**: 24 (8 topologies x 3 design variables)
- **Converged (loss < 0.01)**: 20/24
- **Best result**: foam_eta045_314 / rigidities (loss = 1.58e-18)

## Results by Topology

### iso_crystal
_Isotropic crystal — regular triangular lattice, shape=(1,1), orientation=0_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | -0.000001 | 2.50e-01 | No | 2020 | 4.7 |
| rest_lengths | +0.000000 | 2.50e-01 | No | 2020 | 5.5 |
| both | +0.000000 | 2.50e-01 | No | 2020 | 15.5 |

### aniso_crystal
_Anisotropic crystal — stretched lattice, shape=(1.5, 0.8), orientation=pi/6_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | -0.500000 | 7.79e-15 | Yes | 40 | 0.5 |
| rest_lengths | -0.499328 | 4.51e-07 | No | 2020 | 9.3 |
| both | -0.500000 | 2.10e-17 | Yes | 60 | 0.9 |

### foam_eta02_42
_Foam eta=0.2 (seed 42) — Delaunay BEFORE perturbation_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | -0.500000 | 7.93e-17 | Yes | 80 | 1.0 |
| rest_lengths | -0.500000 | 1.42e-17 | Yes | 80 | 0.8 |
| both | -0.500000 | 4.85e-17 | Yes | 60 | 0.8 |

### foam_eta02_137
_Foam eta=0.2 (seed 137) — Delaunay BEFORE perturbation_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | -0.500000 | 8.37e-16 | Yes | 60 | 0.7 |
| rest_lengths | -0.500000 | 2.66e-17 | Yes | 80 | 0.9 |
| both | -0.500000 | 2.00e-16 | Yes | 80 | 1.3 |

### foam_eta045_256
_Foam eta=0.45 (seed 256) — Delaunay BEFORE perturbation_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | -0.499971 | 8.50e-10 | No | 2020 | 6.1 |
| rest_lengths | -0.500000 | 1.30e-16 | Yes | 40 | 0.3 |
| both | -0.500002 | 5.55e-12 | Yes | 2020 | 8.5 |

### foam_eta045_314
_Foam eta=0.45 (seed 314) — Delaunay BEFORE perturbation_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | -0.500000 | 1.58e-18 | Yes | 40 | 0.2 |
| rest_lengths | -0.500015 | 2.10e-10 | No | 2020 | 7.6 |
| both | -1.110817 | 3.73e-01 | No | 2020 | 22.1 |

### poisson_999
_Poisson random network (seed 999) — Delaunay AFTER perturbation, eta=0.3_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | -0.500000 | 3.61e-17 | Yes | 60 | 0.5 |
| rest_lengths | -0.500000 | 1.28e-17 | Yes | 80 | 0.8 |
| both | -0.500000 | 2.65e-17 | Yes | 60 | 0.8 |

### poisson_1337
_Poisson random network (seed 1337) — Delaunay AFTER perturbation, eta=0.3_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | -0.500000 | 4.63e-17 | Yes | 60 | 0.4 |
| rest_lengths | -0.500000 | 2.96e-17 | Yes | 80 | 0.8 |
| both | -0.500000 | 3.70e-17 | Yes | 60 | 0.7 |
