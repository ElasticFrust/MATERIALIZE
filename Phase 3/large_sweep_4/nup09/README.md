# Target Poisson Ratio: nu = 0.9 (ISOTROPIC)

## Run Configuration

- **Target nu**: 0.9
- **Constraint**: ISOTROPIC (weight_isotropy=10.0)
- **Mesh size**: (10, 10)
- **Design variables**: rigidities, rest_lengths, both
- **Optimizer**: L-BFGS, lr=0.05, tol=1e-14, max_iter=2000
- **Random restarts per case**: 3 (seeds: 7, 144, 281)

## Summary

- **Total cases**: 24 (8 topologies x 3 design variables)
- **Converged (loss < 0.01)**: 4/24
- **Best result**: poisson_999 / rest_lengths (loss = 3.90e-13)

## Results by Topology

### iso_crystal
_Isotropic crystal -- regular triangular lattice, shape=(1,1), orientation=0_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | +0.343865 | 3.15e-01 | No | 2020 | 15.5 |
| rest_lengths | +0.343723 | 3.15e-01 | No | 2020 | 13.2 |
| both | +0.343849 | 3.15e-01 | No | 2020 | 21.0 |

### aniso_crystal
_Anisotropic crystal -- stretched lattice, shape=(1.5, 0.8), orientation=pi/6_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | +0.308063 | 7.73e-01 | No | 2020 | 22.5 |
| rest_lengths | +0.317836 | 7.73e-01 | No | 2020 | 22.0 |
| both | +0.322572 | 7.73e-01 | No | 2020 | 22.4 |

### foam_eta02_42
_Foam eta=0.2 (seed 42) -- Delaunay BEFORE perturbation_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | +0.594004 | 1.65e-01 | No | 2020 | 15.1 |
| rest_lengths | +0.766348 | 2.08e-02 | No | 2020 | 8.6 |
| both | +0.559703 | 2.22e-01 | No | 2020 | 24.8 |

### foam_eta02_137
_Foam eta=0.2 (seed 137) -- Delaunay BEFORE perturbation_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | +0.863551 | 4.61e-02 | No | 2020 | 19.2 |
| rest_lengths | +0.895919 | 3.01e-05 | No | 2020 | 10.4 |
| both | +0.965195 | 3.76e-02 | No | 2020 | 18.2 |

### foam_eta045_256
_Foam eta=0.45 (seed 256) -- Delaunay BEFORE perturbation_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | -0.292668 | 1.51e+00 | No | 2020 | 7.2 |
| rest_lengths | -0.287341 | 1.51e+00 | No | 2020 | 13.4 |
| both | -0.293178 | 1.51e+00 | No | 2020 | 13.8 |

### foam_eta045_314
_Foam eta=0.45 (seed 314) -- Delaunay BEFORE perturbation_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | -2.856973 | 4.31e+00 | No | 2020 | 17.0 |
| rest_lengths | -2.849101 | 4.32e+00 | No | 2020 | 18.5 |
| both | -2.872456 | 4.31e+00 | No | 2020 | 21.1 |

### poisson_999
_Poisson random network (seed 999) -- Delaunay AFTER perturbation, eta=0.3_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | +0.741034 | 1.06e-01 | No | 2020 | 19.0 |
| rest_lengths | +0.900000 | 3.90e-13 | Yes | 2020 | 8.8 |
| both | +0.900017 | 3.93e-09 | No | 2020 | 12.5 |

### poisson_1337
_Poisson random network (seed 1337) -- Delaunay AFTER perturbation, eta=0.3_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | +0.889280 | 2.89e-02 | No | 2020 | 20.8 |
| rest_lengths | +0.900001 | 2.70e-12 | Yes | 2020 | 8.7 |
| both | +1.297610 | 2.08e-01 | No | 2020 | 25.4 |
