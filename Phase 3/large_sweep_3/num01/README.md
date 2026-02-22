# Target Poisson Ratio: nu = -0.1

## Run Configuration

- **Target nu**: -0.1
- **Mesh size**: (10, 10)
- **Design variables**: rigidities, rest_lengths, both
- **Optimizer**: L-BFGS, lr=0.05, tol=1e-14, max_iter=2000
- **Random restarts per case**: 3 (seeds: 7, 144, 281)

## Summary

- **Total cases**: 24 (8 topologies x 3 design variables)
- **Converged (loss < 0.01)**: 19/24
- **Best result**: aniso_crystal / rigidities (loss = 6.80e-19)

## Results by Topology

### iso_crystal
_Isotropic crystal — regular triangular lattice, shape=(1,1), orientation=0_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | +0.000000 | 1.00e-02 | No | 2020 | 4.9 |
| rest_lengths | +0.000000 | 1.00e-02 | No | 2020 | 5.1 |
| both | +0.000000 | 1.00e-02 | No | 2020 | 14.2 |

### aniso_crystal
_Anisotropic crystal — stretched lattice, shape=(1.5, 0.8), orientation=pi/6_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | -0.100000 | 6.80e-19 | Yes | 40 | 0.3 |
| rest_lengths | -0.100000 | 6.00e-17 | Yes | 60 | 0.7 |
| both | -0.100000 | 1.15e-17 | Yes | 80 | 1.4 |

### foam_eta02_42
_Foam eta=0.2 (seed 42) — Delaunay BEFORE perturbation_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | -0.100000 | 2.59e-17 | Yes | 60 | 0.6 |
| rest_lengths | -0.100000 | 4.07e-17 | Yes | 80 | 0.8 |
| both | -0.100000 | 1.27e-17 | Yes | 60 | 0.7 |

### foam_eta02_137
_Foam eta=0.2 (seed 137) — Delaunay BEFORE perturbation_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | -0.100000 | 2.77e-17 | Yes | 60 | 0.5 |
| rest_lengths | -0.100000 | 4.47e-17 | Yes | 80 | 0.8 |
| both | -0.100000 | 1.73e-17 | Yes | 60 | 0.8 |

### foam_eta045_256
_Foam eta=0.45 (seed 256) — Delaunay BEFORE perturbation_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | -0.099992 | 5.76e-11 | No | 2020 | 8.3 |
| rest_lengths | -0.099982 | 3.09e-10 | No | 2020 | 7.6 |
| both | -0.100011 | 1.17e-10 | No | 2020 | 12.8 |

### foam_eta045_314
_Foam eta=0.45 (seed 314) — Delaunay BEFORE perturbation_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | -0.100000 | 1.00e-16 | Yes | 40 | 0.2 |
| rest_lengths | -1.827798 | 2.99e+00 | No | 2020 | 18.3 |
| both | NaN | 7.82e+00 | No | 2020 | 54.5 |

### poisson_999
_Poisson random network (seed 999) — Delaunay AFTER perturbation, eta=0.3_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | -0.100000 | 3.47e-17 | Yes | 60 | 0.5 |
| rest_lengths | -0.100000 | 1.12e-16 | Yes | 60 | 0.7 |
| both | -0.100000 | 2.15e-17 | Yes | 60 | 0.7 |

### poisson_1337
_Poisson random network (seed 1337) — Delaunay AFTER perturbation, eta=0.3_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | -0.100000 | 1.34e-17 | Yes | 40 | 0.3 |
| rest_lengths | -0.100000 | 4.14e-17 | Yes | 60 | 0.7 |
| both | -0.100000 | 4.19e-17 | Yes | 60 | 0.8 |
