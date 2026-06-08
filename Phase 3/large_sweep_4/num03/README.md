# Target Poisson Ratio: nu = -0.3 (ISOTROPIC)

## Run Configuration

- **Target nu**: -0.3
- **Constraint**: ISOTROPIC (weight_isotropy=10.0)
- **Mesh size**: (10, 10)
- **Design variables**: rigidities, rest_lengths, both
- **Optimizer**: L-BFGS, lr=0.05, tol=1e-14, max_iter=2000
- **Random restarts per case**: 3 (seeds: 7, 144, 281)

## Summary

- **Total cases**: 24 (8 topologies x 3 design variables)
- **Converged (loss < 0.01)**: 3/24
- **Best result**: poisson_999 / both (loss = 3.64e-03)

## Results by Topology

### iso_crystal
_Isotropic crystal -- regular triangular lattice, shape=(1,1), orientation=0_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | +0.322589 | 3.94e-01 | No | 2020 | 13.5 |
| rest_lengths | +0.322769 | 3.94e-01 | No | 2020 | 12.6 |
| both | +0.322633 | 3.94e-01 | No | 2020 | 20.6 |

### aniso_crystal
_Anisotropic crystal -- stretched lattice, shape=(1.5, 0.8), orientation=pi/6_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | +0.247906 | 5.55e-01 | No | 2020 | 21.4 |
| rest_lengths | +0.258884 | 5.50e-01 | No | 2020 | 17.7 |
| both | +0.249116 | 5.55e-01 | No | 2020 | 24.0 |

### foam_eta02_42
_Foam eta=0.2 (seed 42) -- Delaunay BEFORE perturbation_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | +0.052872 | 2.11e-01 | No | 2020 | 14.7 |
| rest_lengths | -0.130473 | 1.16e-01 | No | 2020 | 7.2 |
| both | +0.038861 | 1.87e-01 | No | 2020 | 23.7 |

### foam_eta02_137
_Foam eta=0.2 (seed 137) -- Delaunay BEFORE perturbation_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | -0.038939 | 1.77e-01 | No | 2020 | 17.1 |
| rest_lengths | -0.066798 | 1.79e-01 | No | 2020 | 18.3 |
| both | +0.053783 | 1.94e-01 | No | 2020 | 22.1 |

### foam_eta045_256
_Foam eta=0.45 (seed 256) -- Delaunay BEFORE perturbation_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | -0.311931 | 6.34e-02 | No | 2020 | 7.9 |
| rest_lengths | -0.306714 | 7.55e-02 | No | 2020 | 9.6 |
| both | -0.312270 | 6.41e-02 | No | 2020 | 12.4 |

### foam_eta045_314
_Foam eta=0.45 (seed 314) -- Delaunay BEFORE perturbation_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | -2.867262 | 2.37e+00 | No | 2020 | 17.6 |
| rest_lengths | -2.858086 | 2.37e+00 | No | 2020 | 21.8 |
| both | -2.895109 | 2.37e+00 | No | 2020 | 22.2 |

### poisson_999
_Poisson random network (seed 999) -- Delaunay AFTER perturbation, eta=0.3_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | -0.078501 | 9.93e-02 | No | 2020 | 17.5 |
| rest_lengths | -0.241256 | 3.75e-03 | No | 2020 | 7.2 |
| both | -0.242799 | 3.64e-03 | No | 2020 | 33.5 |

### poisson_1337
_Poisson random network (seed 1337) -- Delaunay AFTER perturbation, eta=0.3_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | -0.070142 | 9.01e-02 | No | 2020 | 16.9 |
| rest_lengths | -0.233650 | 4.79e-03 | No | 2020 | 7.3 |
| both | -0.214388 | 1.40e-02 | No | 2020 | 21.8 |
