# Target Poisson Ratio: nu = -0.5 (ISOTROPIC)

## Run Configuration

- **Target nu**: -0.5
- **Constraint**: ISOTROPIC (weight_isotropy=10.0)
- **Mesh size**: (10, 10)
- **Design variables**: rigidities, rest_lengths, both
- **Optimizer**: L-BFGS, lr=0.05, tol=1e-14, max_iter=2000
- **Random restarts per case**: 3 (seeds: 7, 144, 281)

## Summary

- **Total cases**: 24 (8 topologies x 3 design variables)
- **Converged (loss < 0.01)**: 0/24
- **Best result**: poisson_999 / both (loss = 6.57e-02)

## Results by Topology

### iso_crystal
_Isotropic crystal -- regular triangular lattice, shape=(1,1), orientation=0_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | +0.319510 | 6.83e-01 | No | 2020 | 15.2 |
| rest_lengths | +0.319618 | 6.83e-01 | No | 2020 | 13.7 |
| both | +0.319365 | 6.83e-01 | No | 2020 | 23.7 |

### aniso_crystal
_Anisotropic crystal -- stretched lattice, shape=(1.5, 0.8), orientation=pi/6_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | +0.237308 | 7.93e-01 | No | 2020 | 19.3 |
| rest_lengths | +0.255306 | 7.89e-01 | No | 2020 | 19.0 |
| both | +0.239778 | 7.95e-01 | No | 2020 | 25.1 |

### foam_eta02_42
_Foam eta=0.2 (seed 42) -- Delaunay BEFORE perturbation_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | -0.067791 | 4.30e-01 | No | 2020 | 13.9 |
| rest_lengths | -0.106537 | 3.25e-01 | No | 2020 | 21.6 |
| both | +0.003334 | 3.77e-01 | No | 2020 | 27.3 |

### foam_eta02_137
_Foam eta=0.2 (seed 137) -- Delaunay BEFORE perturbation_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | +0.018429 | 4.23e-01 | No | 2020 | 16.4 |
| rest_lengths | -0.129309 | 3.48e-01 | No | 2020 | 20.4 |
| both | -0.001418 | 3.89e-01 | No | 2020 | 24.2 |

### foam_eta045_256
_Foam eta=0.45 (seed 256) -- Delaunay BEFORE perturbation_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | -0.317523 | 9.82e-02 | No | 2020 | 6.7 |
| rest_lengths | -0.309971 | 1.12e-01 | No | 2020 | 13.6 |
| both | -0.316216 | 9.87e-02 | No | 2020 | 11.1 |

### foam_eta045_314
_Foam eta=0.45 (seed 314) -- Delaunay BEFORE perturbation_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | -2.868375 | 2.32e+00 | No | 2020 | 17.0 |
| rest_lengths | -2.858109 | 2.32e+00 | No | 2020 | 15.7 |
| both | -2.801743 | 2.31e+00 | No | 2020 | 25.3 |

### poisson_999
_Poisson random network (seed 999) -- Delaunay AFTER perturbation, eta=0.3_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | -0.106384 | 2.58e-01 | No | 2020 | 15.8 |
| rest_lengths | -0.193926 | 2.55e-01 | No | 2020 | 19.3 |
| both | -0.248105 | 6.57e-02 | No | 2020 | 14.6 |

### poisson_1337
_Poisson random network (seed 1337) -- Delaunay AFTER perturbation, eta=0.3_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | -0.041992 | 3.12e-01 | No | 2020 | 14.3 |
| rest_lengths | -0.282863 | 1.48e-01 | No | 2020 | 11.6 |
| both | -0.238438 | 7.99e-02 | No | 2020 | 22.5 |
