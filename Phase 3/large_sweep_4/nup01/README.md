# Target Poisson Ratio: nu = 0.1 (ISOTROPIC)

## Run Configuration

- **Target nu**: 0.1
- **Constraint**: ISOTROPIC (weight_isotropy=10.0)
- **Mesh size**: (10, 10)
- **Design variables**: rigidities, rest_lengths, both
- **Optimizer**: L-BFGS, lr=0.05, tol=1e-14, max_iter=2000
- **Random restarts per case**: 3 (seeds: 7, 144, 281)

## Summary

- **Total cases**: 24 (8 topologies x 3 design variables)
- **Converged (loss < 0.01)**: 12/24
- **Best result**: poisson_1337 / rigidities (loss = 1.33e-17)

## Results by Topology

### iso_crystal
_Isotropic crystal -- regular triangular lattice, shape=(1,1), orientation=0_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | +0.329322 | 5.35e-02 | No | 2020 | 16.6 |
| rest_lengths | +0.329337 | 5.35e-02 | No | 2020 | 12.6 |
| both | +0.329329 | 5.35e-02 | No | 2020 | 19.3 |

### aniso_crystal
_Anisotropic crystal -- stretched lattice, shape=(1.5, 0.8), orientation=pi/6_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | +0.268299 | 3.14e-01 | No | 2020 | 19.8 |
| rest_lengths | +0.270953 | 3.11e-01 | No | 2020 | 20.0 |
| both | +0.282794 | 3.09e-01 | No | 2020 | 23.8 |

### foam_eta02_42
_Foam eta=0.2 (seed 42) -- Delaunay BEFORE perturbation_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | +0.100000 | 1.45e-17 | Yes | 160 | 2.1 |
| rest_lengths | +0.100000 | 4.30e-16 | Yes | 100 | 1.3 |
| both | +0.100000 | 4.35e-16 | Yes | 120 | 2.3 |

### foam_eta02_137
_Foam eta=0.2 (seed 137) -- Delaunay BEFORE perturbation_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | +0.100000 | 1.37e-16 | Yes | 140 | 1.9 |
| rest_lengths | +0.100000 | 1.98e-16 | Yes | 140 | 1.9 |
| both | +0.100000 | 1.40e-17 | Yes | 120 | 2.1 |

### foam_eta045_256
_Foam eta=0.45 (seed 256) -- Delaunay BEFORE perturbation_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | -1.272699 | 2.03e+00 | No | 2020 | 4.0 |
| rest_lengths | -0.300340 | 2.40e-01 | No | 2020 | 10.9 |
| both | -0.305020 | 2.31e-01 | No | 2020 | 11.3 |

### foam_eta045_314
_Foam eta=0.45 (seed 314) -- Delaunay BEFORE perturbation_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | -2.859677 | 2.72e+00 | No | 2020 | 16.9 |
| rest_lengths | -2.858214 | 2.71e+00 | No | 2020 | 16.5 |
| both | -2.865715 | 2.71e+00 | No | 2020 | 23.5 |

### poisson_999
_Poisson random network (seed 999) -- Delaunay AFTER perturbation, eta=0.3_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | +0.100000 | 7.08e-17 | Yes | 100 | 1.3 |
| rest_lengths | +0.100000 | 2.22e-17 | Yes | 100 | 1.2 |
| both | +0.100000 | 1.58e-17 | Yes | 100 | 1.6 |

### poisson_1337
_Poisson random network (seed 1337) -- Delaunay AFTER perturbation, eta=0.3_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | +0.100000 | 1.33e-17 | Yes | 120 | 1.5 |
| rest_lengths | +0.100000 | 5.89e-17 | Yes | 100 | 1.1 |
| both | +0.100000 | 3.82e-17 | Yes | 100 | 1.8 |
