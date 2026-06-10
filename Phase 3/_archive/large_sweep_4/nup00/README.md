# Target Poisson Ratio: nu = 0.0 (ISOTROPIC)

## Run Configuration

- **Target nu**: 0.0
- **Constraint**: ISOTROPIC (weight_isotropy=10.0)
- **Mesh size**: (10, 10)
- **Design variables**: rigidities, rest_lengths, both
- **Optimizer**: L-BFGS, lr=0.05, tol=1e-14, max_iter=2000
- **Random restarts per case**: 3 (seeds: 7, 144, 281)

## Summary

- **Total cases**: 24 (8 topologies x 3 design variables)
- **Converged (loss < 0.01)**: 10/24
- **Best result**: foam_eta02_42 / rest_lengths (loss = 4.48e-18)

## Results by Topology

### iso_crystal
_Isotropic crystal -- regular triangular lattice, shape=(1,1), orientation=0_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | +0.327626 | 1.09e-01 | No | 2020 | 15.4 |
| rest_lengths | +0.327639 | 1.09e-01 | No | 2020 | 13.7 |
| both | +0.327649 | 1.09e-01 | No | 2020 | 20.6 |

### aniso_crystal
_Anisotropic crystal -- stretched lattice, shape=(1.5, 0.8), orientation=pi/6_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | +0.261726 | 3.44e-01 | No | 2020 | 21.0 |
| rest_lengths | +0.267627 | 3.41e-01 | No | 2020 | 20.6 |
| both | +0.279377 | 3.40e-01 | No | 2020 | 28.5 |

### foam_eta02_42
_Foam eta=0.2 (seed 42) -- Delaunay BEFORE perturbation_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | +0.012193 | 1.25e-02 | No | 2020 | 17.1 |
| rest_lengths | -0.000000 | 4.48e-18 | Yes | 120 | 1.6 |
| both | -0.013683 | 1.78e-02 | No | 2020 | 22.0 |

### foam_eta02_137
_Foam eta=0.2 (seed 137) -- Delaunay BEFORE perturbation_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | -0.000000 | 4.14e-15 | Yes | 340 | 5.4 |
| rest_lengths | -0.000000 | 2.33e-16 | Yes | 140 | 1.9 |
| both | +0.000000 | 1.51e-16 | Yes | 140 | 2.6 |

### foam_eta045_256
_Foam eta=0.45 (seed 256) -- Delaunay BEFORE perturbation_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | -0.308875 | 1.59e-01 | No | 2020 | 8.0 |
| rest_lengths | -0.301827 | 1.70e-01 | No | 2020 | 11.9 |
| both | -0.306670 | 1.60e-01 | No | 2020 | 12.8 |

### foam_eta045_314
_Foam eta=0.45 (seed 314) -- Delaunay BEFORE perturbation_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | -2.883122 | 2.61e+00 | No | 2020 | 16.0 |
| rest_lengths | -2.851109 | 2.60e+00 | No | 2020 | 16.3 |
| both | -2.892770 | 2.59e+00 | No | 2020 | 21.6 |

### poisson_999
_Poisson random network (seed 999) -- Delaunay AFTER perturbation, eta=0.3_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | +0.000000 | 1.95e-16 | Yes | 140 | 1.9 |
| rest_lengths | +0.000000 | 3.79e-16 | Yes | 120 | 1.6 |
| both | +0.000000 | 1.71e-17 | Yes | 120 | 2.0 |

### poisson_1337
_Poisson random network (seed 1337) -- Delaunay AFTER perturbation, eta=0.3_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | +0.000000 | 7.43e-17 | Yes | 180 | 2.4 |
| rest_lengths | -0.000000 | 1.68e-16 | Yes | 120 | 1.6 |
| both | -0.000000 | 1.13e-16 | Yes | 140 | 2.6 |
