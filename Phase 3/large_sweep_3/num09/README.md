# Target Poisson Ratio: nu = -0.9

## Run Configuration

- **Target nu**: -0.9
- **Mesh size**: (10, 10)
- **Design variables**: rigidities, rest_lengths, both
- **Optimizer**: L-BFGS, lr=0.05, tol=1e-14, max_iter=2000
- **Random restarts per case**: 3 (seeds: 7, 144, 281)

## Summary

- **Total cases**: 24 (8 topologies x 3 design variables)
- **Converged (loss < 0.01)**: 17/24
- **Best result**: aniso_crystal / rigidities (loss = 1.43e-18)

## Results by Topology

### iso_crystal
_Isotropic crystal — regular triangular lattice, shape=(1,1), orientation=0_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | -0.000000 | 8.10e-01 | No | 2020 | 13.7 |
| rest_lengths | +0.000000 | 8.10e-01 | No | 2020 | 13.7 |
| both | +0.000000 | 8.10e-01 | No | 2020 | 26.0 |

### aniso_crystal
_Anisotropic crystal — stretched lattice, shape=(1.5, 0.8), orientation=pi/6_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | -0.900000 | 1.43e-18 | Yes | 40 | 0.2 |
| rest_lengths | -0.876970 | 5.30e-04 | No | 2020 | 20.0 |
| both | -0.900000 | 2.62e-16 | Yes | 40 | 0.6 |

### foam_eta02_42
_Foam eta=0.2 (seed 42) — Delaunay BEFORE perturbation_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | -0.766577 | 1.78e-02 | No | 2020 | 22.3 |
| rest_lengths | -0.826225 | 5.44e-03 | No | 2020 | 19.4 |
| both | -0.707322 | 3.71e-02 | No | 2020 | 35.0 |

### foam_eta02_137
_Foam eta=0.2 (seed 137) — Delaunay BEFORE perturbation_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | -0.900000 | 3.74e-14 | Yes | 2020 | 17.1 |
| rest_lengths | -0.890762 | 8.53e-05 | No | 2020 | 23.4 |
| both | -0.900000 | 1.68e-15 | Yes | 100 | 1.8 |

### foam_eta045_256
_Foam eta=0.45 (seed 256) — Delaunay BEFORE perturbation_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | -0.900025 | 6.49e-10 | No | 2020 | 12.9 |
| rest_lengths | -0.900033 | 1.08e-09 | No | 2020 | 12.9 |
| both | -0.900001 | 4.07e-13 | Yes | 2020 | 13.2 |

### foam_eta045_314
_Foam eta=0.45 (seed 314) — Delaunay BEFORE perturbation_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | -0.900003 | 9.12e-12 | Yes | 2020 | 16.4 |
| rest_lengths | NaN | 3.83e+00 | No | 2020 | 82.3 |
| both | -1.112709 | 4.52e-02 | No | 2020 | 45.6 |

### poisson_999
_Poisson random network (seed 999) — Delaunay AFTER perturbation, eta=0.3_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | -0.900000 | 1.46e-16 | Yes | 60 | 0.6 |
| rest_lengths | -0.900000 | 9.25e-16 | Yes | 100 | 1.4 |
| both | -0.900000 | 1.06e-16 | Yes | 60 | 0.8 |

### poisson_1337
_Poisson random network (seed 1337) — Delaunay AFTER perturbation, eta=0.3_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | -0.900000 | 1.06e-16 | Yes | 40 | 0.4 |
| rest_lengths | -0.900000 | 9.35e-15 | Yes | 100 | 1.3 |
| both | -0.900000 | 1.67e-17 | Yes | 60 | 0.9 |
