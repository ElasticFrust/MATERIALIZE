# Target Poisson Ratio: nu = 0.3 (ISOTROPIC)

## Run Configuration

- **Target nu**: 0.3
- **Constraint**: ISOTROPIC (weight_isotropy=10.0)
- **Mesh size**: (10, 10)
- **Design variables**: rigidities, rest_lengths, both
- **Optimizer**: L-BFGS, lr=0.05, tol=1e-14, max_iter=2000
- **Random restarts per case**: 3 (seeds: 7, 144, 281)

## Summary

- **Total cases**: 24 (8 topologies x 3 design variables)
- **Converged (loss < 0.01)**: 15/24
- **Best result**: foam_eta02_42 / rigidities (loss = 1.29e-17)

## Results by Topology

### iso_crystal
_Isotropic crystal -- regular triangular lattice, shape=(1,1), orientation=0_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | +0.332753 | 1.09e-03 | No | 2020 | 11.0 |
| rest_lengths | +0.332753 | 1.09e-03 | No | 2020 | 8.4 |
| both | +0.332753 | 1.09e-03 | No | 2020 | 13.3 |

### aniso_crystal
_Anisotropic crystal -- stretched lattice, shape=(1.5, 0.8), orientation=pi/6_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | +0.290038 | 3.06e-01 | No | 2020 | 21.4 |
| rest_lengths | +0.280330 | 3.10e-01 | No | 2020 | 21.2 |
| both | +0.289301 | 3.06e-01 | No | 2020 | 26.1 |

### foam_eta02_42
_Foam eta=0.2 (seed 42) -- Delaunay BEFORE perturbation_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | +0.300000 | 1.29e-17 | Yes | 80 | 1.0 |
| rest_lengths | +0.300000 | 1.23e-16 | Yes | 100 | 1.4 |
| both | +0.300000 | 1.46e-16 | Yes | 100 | 2.0 |

### foam_eta02_137
_Foam eta=0.2 (seed 137) -- Delaunay BEFORE perturbation_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | +0.300000 | 9.43e-17 | Yes | 100 | 1.4 |
| rest_lengths | +0.300000 | 1.31e-16 | Yes | 140 | 2.0 |
| both | +0.300000 | 4.87e-16 | Yes | 100 | 2.2 |

### foam_eta045_256
_Foam eta=0.45 (seed 256) -- Delaunay BEFORE perturbation_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | -0.302898 | 4.32e-01 | No | 2020 | 8.1 |
| rest_lengths | -0.297658 | 4.40e-01 | No | 2020 | 14.5 |
| both | -0.302214 | 4.32e-01 | No | 2020 | 17.5 |

### foam_eta045_314
_Foam eta=0.45 (seed 314) -- Delaunay BEFORE perturbation_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | -2.860439 | 2.99e+00 | No | 2020 | 16.9 |
| rest_lengths | -2.855604 | 3.00e+00 | No | 2020 | 16.8 |
| both | -2.727984 | 2.98e+00 | No | 2020 | 28.5 |

### poisson_999
_Poisson random network (seed 999) -- Delaunay AFTER perturbation, eta=0.3_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | +0.300000 | 4.67e-17 | Yes | 80 | 1.0 |
| rest_lengths | +0.300000 | 2.31e-17 | Yes | 100 | 1.3 |
| both | +0.300000 | 4.66e-17 | Yes | 100 | 1.7 |

### poisson_1337
_Poisson random network (seed 1337) -- Delaunay AFTER perturbation, eta=0.3_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | +0.300000 | 1.96e-17 | Yes | 120 | 1.6 |
| rest_lengths | +0.300000 | 2.31e-17 | Yes | 100 | 1.2 |
| both | +0.300000 | 2.27e-17 | Yes | 100 | 1.7 |
