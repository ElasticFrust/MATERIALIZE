# Target Poisson Ratio: nu = -0.9 (ISOTROPIC)

## Run Configuration

- **Target nu**: -0.9
- **Constraint**: ISOTROPIC (weight_isotropy=10.0)
- **Mesh size**: (10, 10)
- **Design variables**: rigidities, rest_lengths, both
- **Optimizer**: L-BFGS, lr=0.05, tol=1e-14, max_iter=2000
- **Random restarts per case**: 3 (seeds: 7, 144, 281)

## Summary

- **Total cases**: 24 (8 topologies x 3 design variables)
- **Converged (loss < 0.01)**: 0/24
- **Best result**: foam_eta045_256 / both (loss = 4.03e-01)

## Results by Topology

### iso_crystal
_Isotropic crystal -- regular triangular lattice, shape=(1,1), orientation=0_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | +0.313530 | 1.50e+00 | No | 2020 | 42.9 |
| rest_lengths | +0.313580 | 1.50e+00 | No | 2020 | 18.0 |
| both | +0.313410 | 1.50e+00 | No | 2020 | 21.8 |

### aniso_crystal
_Anisotropic crystal -- stretched lattice, shape=(1.5, 0.8), orientation=pi/6_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | +0.214738 | 1.51e+00 | No | 2020 | 21.4 |
| rest_lengths | +0.247976 | 1.50e+00 | No | 2020 | 20.5 |
| both | +0.210128 | 1.51e+00 | No | 2020 | 23.9 |

### foam_eta02_42
_Foam eta=0.2 (seed 42) -- Delaunay BEFORE perturbation_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | +0.000126 | 1.07e+00 | No | 2020 | 17.3 |
| rest_lengths | -0.212835 | 7.76e-01 | No | 2020 | 6.4 |
| both | +0.060937 | 1.17e+00 | No | 2020 | 22.0 |

### foam_eta02_137
_Foam eta=0.2 (seed 137) -- Delaunay BEFORE perturbation_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | -0.068921 | 9.57e-01 | No | 2020 | 17.5 |
| rest_lengths | -0.269247 | 8.61e-01 | No | 2020 | 10.1 |
| both | -0.029557 | 9.04e-01 | No | 2020 | 20.2 |

### foam_eta045_256
_Foam eta=0.45 (seed 256) -- Delaunay BEFORE perturbation_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | -0.323700 | 4.04e-01 | No | 2020 | 10.7 |
| rest_lengths | -0.318243 | 4.19e-01 | No | 2020 | 11.1 |
| both | -0.323871 | 4.03e-01 | No | 2020 | 11.2 |

### foam_eta045_314
_Foam eta=0.45 (seed 314) -- Delaunay BEFORE perturbation_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | -2.869908 | 2.45e+00 | No | 2020 | 16.0 |
| rest_lengths | -2.881842 | 2.44e+00 | No | 2020 | 21.2 |
| both | -2.068633 | 2.27e+00 | No | 2020 | 15.5 |

### poisson_999
_Poisson random network (seed 999) -- Delaunay AFTER perturbation, eta=0.3_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | -0.305522 | 9.46e-01 | No | 2020 | 17.8 |
| rest_lengths | -0.270442 | 7.33e-01 | No | 2020 | 19.7 |
| both | -0.236822 | 8.87e-01 | No | 2020 | 30.8 |

### poisson_1337
_Poisson random network (seed 1337) -- Delaunay AFTER perturbation, eta=0.3_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | -0.174698 | 8.65e-01 | No | 2020 | 16.4 |
| rest_lengths | -0.346878 | 6.52e-01 | No | 2020 | 10.1 |
| both | -0.138048 | 8.93e-01 | No | 2020 | 20.0 |
