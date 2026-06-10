# Target Poisson Ratio: nu = 0.7 (ISOTROPIC)

## Run Configuration

- **Target nu**: 0.7
- **Constraint**: ISOTROPIC (weight_isotropy=10.0)
- **Mesh size**: (10, 10)
- **Design variables**: rigidities, rest_lengths, both
- **Optimizer**: L-BFGS, lr=0.05, tol=1e-14, max_iter=2000
- **Random restarts per case**: 3 (seeds: 7, 144, 281)

## Summary

- **Total cases**: 24 (8 topologies x 3 design variables)
- **Converged (loss < 0.01)**: 10/24
- **Best result**: poisson_999 / rigidities (loss = 1.22e-16)

## Results by Topology

### iso_crystal
_Isotropic crystal -- regular triangular lattice, shape=(1,1), orientation=0_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | +0.339950 | 1.32e-01 | No | 2020 | 15.6 |
| rest_lengths | +0.339980 | 1.32e-01 | No | 2020 | 15.0 |
| both | +0.339956 | 1.32e-01 | No | 2020 | 19.7 |

### aniso_crystal
_Anisotropic crystal -- stretched lattice, shape=(1.5, 0.8), orientation=pi/6_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | +0.299110 | 5.40e-01 | No | 2020 | 21.5 |
| rest_lengths | +0.304843 | 5.41e-01 | No | 2020 | 20.6 |
| both | +0.305327 | 5.38e-01 | No | 2020 | 26.5 |

### foam_eta02_42
_Foam eta=0.2 (seed 42) -- Delaunay BEFORE perturbation_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | +0.580231 | 1.02e-01 | No | 2020 | 18.6 |
| rest_lengths | +0.699999 | 2.89e-12 | Yes | 2020 | 8.9 |
| both | +0.699634 | 7.34e-03 | No | 2020 | 23.1 |

### foam_eta02_137
_Foam eta=0.2 (seed 137) -- Delaunay BEFORE perturbation_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | +0.700000 | 1.95e-16 | Yes | 240 | 3.8 |
| rest_lengths | +0.700000 | 8.18e-15 | Yes | 160 | 2.2 |
| both | +0.700000 | 4.65e-15 | Yes | 160 | 3.5 |

### foam_eta045_256
_Foam eta=0.45 (seed 256) -- Delaunay BEFORE perturbation_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | -0.300521 | 1.07e+00 | No | 2020 | 7.1 |
| rest_lengths | -0.289829 | 1.08e+00 | No | 2020 | 10.5 |
| both | -0.295967 | 1.07e+00 | No | 2020 | 15.2 |

### foam_eta045_314
_Foam eta=0.45 (seed 314) -- Delaunay BEFORE perturbation_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | -2.861769 | 3.79e+00 | No | 2020 | 17.2 |
| rest_lengths | -2.849559 | 3.80e+00 | No | 2020 | 16.9 |
| both | -2.872064 | 3.79e+00 | No | 2020 | 21.4 |

### poisson_999
_Poisson random network (seed 999) -- Delaunay AFTER perturbation, eta=0.3_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | +0.700000 | 1.22e-16 | Yes | 200 | 2.8 |
| rest_lengths | +0.700000 | 9.69e-15 | Yes | 120 | 1.6 |
| both | +0.700000 | 2.50e-15 | Yes | 160 | 3.0 |

### poisson_1337
_Poisson random network (seed 1337) -- Delaunay AFTER perturbation, eta=0.3_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | +0.580259 | 9.11e-02 | No | 2020 | 16.8 |
| rest_lengths | +0.700000 | 5.82e-14 | Yes | 2020 | 7.8 |
| both | +0.700000 | 4.51e-15 | Yes | 360 | 8.0 |
