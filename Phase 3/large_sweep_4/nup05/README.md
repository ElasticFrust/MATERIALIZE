# Target Poisson Ratio: nu = 0.5 (ISOTROPIC)

## Run Configuration

- **Target nu**: 0.5
- **Constraint**: ISOTROPIC (weight_isotropy=10.0)
- **Mesh size**: (10, 10)
- **Design variables**: rigidities, rest_lengths, both
- **Optimizer**: L-BFGS, lr=0.05, tol=1e-14, max_iter=2000
- **Random restarts per case**: 3 (seeds: 7, 144, 281)

## Summary

- **Total cases**: 24 (8 topologies x 3 design variables)
- **Converged (loss < 0.01)**: 12/24
- **Best result**: foam_eta02_42 / rigidities (loss = 1.59e-17)

## Results by Topology

### iso_crystal
_Isotropic crystal -- regular triangular lattice, shape=(1,1), orientation=0_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | +0.336282 | 2.73e-02 | No | 2020 | 14.6 |
| rest_lengths | +0.336290 | 2.73e-02 | No | 2020 | 13.6 |
| both | +0.336285 | 2.73e-02 | No | 2020 | 18.4 |

### aniso_crystal
_Anisotropic crystal -- stretched lattice, shape=(1.5, 0.8), orientation=pi/6_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | +0.290577 | 3.83e-01 | No | 2020 | 23.3 |
| rest_lengths | +0.291583 | 3.86e-01 | No | 2020 | 20.3 |
| both | +0.295177 | 3.82e-01 | No | 2020 | 26.8 |

### foam_eta02_42
_Foam eta=0.2 (seed 42) -- Delaunay BEFORE perturbation_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | +0.500000 | 1.59e-17 | Yes | 120 | 1.7 |
| rest_lengths | +0.500000 | 3.58e-17 | Yes | 120 | 1.5 |
| both | +0.500000 | 1.01e-16 | Yes | 120 | 2.3 |

### foam_eta02_137
_Foam eta=0.2 (seed 137) -- Delaunay BEFORE perturbation_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | +0.500000 | 2.54e-17 | Yes | 180 | 2.6 |
| rest_lengths | +0.500000 | 4.55e-17 | Yes | 140 | 1.7 |
| both | +0.500000 | 2.37e-15 | Yes | 120 | 2.5 |

### foam_eta045_256
_Foam eta=0.45 (seed 256) -- Delaunay BEFORE perturbation_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | -0.301727 | 7.12e-01 | No | 2020 | 8.8 |
| rest_lengths | -0.295092 | 7.19e-01 | No | 2020 | 11.6 |
| both | -0.299904 | 7.12e-01 | No | 2020 | 10.3 |

### foam_eta045_314
_Foam eta=0.45 (seed 314) -- Delaunay BEFORE perturbation_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | -2.861033 | 3.35e+00 | No | 2020 | 17.3 |
| rest_lengths | -2.843734 | 3.36e+00 | No | 2020 | 17.3 |
| both | -2.879886 | 3.35e+00 | No | 2020 | 23.4 |

### poisson_999
_Poisson random network (seed 999) -- Delaunay AFTER perturbation, eta=0.3_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | +0.500000 | 1.72e-17 | Yes | 120 | 1.5 |
| rest_lengths | +0.500000 | 1.72e-16 | Yes | 120 | 1.5 |
| both | +0.500000 | 1.88e-17 | Yes | 100 | 1.8 |

### poisson_1337
_Poisson random network (seed 1337) -- Delaunay AFTER perturbation, eta=0.3_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | +0.500000 | 2.28e-17 | Yes | 200 | 2.9 |
| rest_lengths | +0.500000 | 2.78e-15 | Yes | 120 | 1.7 |
| both | +0.500000 | 1.70e-16 | Yes | 120 | 2.3 |
