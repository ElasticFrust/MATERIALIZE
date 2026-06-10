# Target Poisson Ratio: nu = -0.7 (ISOTROPIC)

## Run Configuration

- **Target nu**: -0.7
- **Constraint**: ISOTROPIC (weight_isotropy=10.0)
- **Mesh size**: (10, 10)
- **Design variables**: rigidities, rest_lengths, both
- **Optimizer**: L-BFGS, lr=0.05, tol=1e-14, max_iter=2000
- **Random restarts per case**: 3 (seeds: 7, 144, 281)

## Summary

- **Total cases**: 24 (8 topologies x 3 design variables)
- **Converged (loss < 0.01)**: 0/24
- **Best result**: foam_eta045_256 / rigidities (loss = 2.10e-01)

## Results by Topology

### iso_crystal
_Isotropic crystal -- regular triangular lattice, shape=(1,1), orientation=0_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | +0.316469 | 1.05e+00 | No | 2020 | 18.4 |
| rest_lengths | +0.316561 | 1.05e+00 | No | 2020 | 13.8 |
| both | +0.316356 | 1.05e+00 | No | 2020 | 20.4 |

### aniso_crystal
_Anisotropic crystal -- stretched lattice, shape=(1.5, 0.8), orientation=pi/6_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | +0.226583 | 1.11e+00 | No | 2020 | 21.6 |
| rest_lengths | +0.252112 | 1.11e+00 | No | 2020 | 18.8 |
| both | +0.221842 | 1.11e+00 | No | 2020 | 23.4 |

### foam_eta02_42
_Foam eta=0.2 (seed 42) -- Delaunay BEFORE perturbation_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | +0.110550 | 8.41e-01 | No | 2020 | 17.3 |
| rest_lengths | -0.195347 | 5.11e-01 | No | 2020 | 8.0 |
| both | -0.051502 | 6.98e-01 | No | 2020 | 17.4 |

### foam_eta02_137
_Foam eta=0.2 (seed 137) -- Delaunay BEFORE perturbation_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | -0.091505 | 6.33e-01 | No | 2020 | 17.7 |
| rest_lengths | -0.204142 | 4.92e-01 | No | 2020 | 9.6 |
| both | -0.153547 | 5.87e-01 | No | 2020 | 24.0 |

### foam_eta045_256
_Foam eta=0.45 (seed 256) -- Delaunay BEFORE perturbation_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | -0.318067 | 2.10e-01 | No | 2020 | 7.1 |
| rest_lengths | -0.313408 | 2.26e-01 | No | 2020 | 14.6 |
| both | -0.319356 | 2.12e-01 | No | 2020 | 11.1 |

### foam_eta045_314
_Foam eta=0.45 (seed 314) -- Delaunay BEFORE perturbation_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | -2.870014 | 2.34e+00 | No | 2020 | 17.5 |
| rest_lengths | -2.879812 | 2.34e+00 | No | 2020 | 20.7 |
| both | -2.766036 | 2.33e+00 | No | 2020 | 26.4 |

### poisson_999
_Poisson random network (seed 999) -- Delaunay AFTER perturbation, eta=0.3_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | -0.257434 | 2.13e-01 | No | 2020 | 12.5 |
| rest_lengths | -0.208298 | 4.66e-01 | No | 2020 | 17.3 |
| both | -0.182860 | 5.50e-01 | No | 2020 | 18.6 |

### poisson_1337
_Poisson random network (seed 1337) -- Delaunay AFTER perturbation, eta=0.3_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | -0.088439 | 5.47e-01 | No | 2020 | 16.0 |
| rest_lengths | -0.236491 | 4.62e-01 | No | 2020 | 20.5 |
| both | -0.137636 | 5.23e-01 | No | 2020 | 20.7 |
