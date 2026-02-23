# Target Poisson Ratio: nu = -0.1 (ISOTROPIC)

## Run Configuration

- **Target nu**: -0.1
- **Constraint**: ISOTROPIC (weight_isotropy=10.0)
- **Mesh size**: (10, 10)
- **Design variables**: rigidities, rest_lengths, both
- **Optimizer**: L-BFGS, lr=0.05, tol=1e-14, max_iter=2000
- **Random restarts per case**: 3 (seeds: 7, 144, 281)

## Summary

- **Total cases**: 24 (8 topologies x 3 design variables)
- **Converged (loss < 0.01)**: 8/24
- **Best result**: poisson_999 / both (loss = 1.66e-17)

## Results by Topology

### iso_crystal
_Isotropic crystal -- regular triangular lattice, shape=(1,1), orientation=0_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | +0.325932 | 1.85e-01 | No | 2020 | 15.1 |
| rest_lengths | +0.325987 | 1.85e-01 | No | 2020 | 12.8 |
| both | +0.325986 | 1.85e-01 | No | 2020 | 18.8 |

### aniso_crystal
_Anisotropic crystal -- stretched lattice, shape=(1.5, 0.8), orientation=pi/6_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | +0.258468 | 3.94e-01 | No | 2020 | 19.5 |
| rest_lengths | +0.264206 | 3.91e-01 | No | 2020 | 20.3 |
| both | +0.259446 | 3.94e-01 | No | 2020 | 25.4 |

### foam_eta02_42
_Foam eta=0.2 (seed 42) -- Delaunay BEFORE perturbation_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | +0.102402 | 8.56e-02 | No | 2020 | 18.3 |
| rest_lengths | -0.080528 | 4.19e-04 | No | 2020 | 6.8 |
| both | +0.053381 | 6.90e-02 | No | 2020 | 22.8 |

### foam_eta02_137
_Foam eta=0.2 (seed 137) -- Delaunay BEFORE perturbation_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | +0.052235 | 6.46e-02 | No | 2020 | 17.7 |
| rest_lengths | -0.085653 | 2.34e-04 | No | 2020 | 9.2 |
| both | +0.000144 | 6.93e-02 | No | 2020 | 23.2 |

### foam_eta045_256
_Foam eta=0.45 (seed 256) -- Delaunay BEFORE perturbation_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | -0.311051 | 1.09e-01 | No | 2020 | 8.3 |
| rest_lengths | -0.303458 | 1.19e-01 | No | 2020 | 9.2 |
| both | -0.308614 | 1.08e-01 | No | 2020 | 11.0 |

### foam_eta045_314
_Foam eta=0.45 (seed 314) -- Delaunay BEFORE perturbation_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | -2.866014 | 2.50e+00 | No | 2020 | 17.1 |
| rest_lengths | -2.859996 | 2.50e+00 | No | 2020 | 16.1 |
| both | -2.893942 | 2.50e+00 | No | 2020 | 22.3 |

### poisson_999
_Poisson random network (seed 999) -- Delaunay AFTER perturbation, eta=0.3_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | -0.100000 | 9.55e-15 | Yes | 140 | 2.1 |
| rest_lengths | -0.100000 | 3.00e-17 | Yes | 120 | 1.4 |
| both | -0.100000 | 1.66e-17 | Yes | 120 | 2.1 |

### poisson_1337
_Poisson random network (seed 1337) -- Delaunay AFTER perturbation, eta=0.3_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | -0.100000 | 2.15e-17 | Yes | 400 | 5.9 |
| rest_lengths | -0.100000 | 1.88e-15 | Yes | 120 | 1.7 |
| both | -0.100000 | 8.90e-16 | Yes | 180 | 3.6 |
