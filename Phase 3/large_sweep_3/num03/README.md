# Target Poisson Ratio: nu = -0.3

## Run Configuration

- **Target nu**: -0.3
- **Mesh size**: (10, 10)
- **Design variables**: rigidities, rest_lengths, both
- **Optimizer**: L-BFGS, lr=0.05, tol=1e-14, max_iter=500
- **Random restarts per case**: 3 (seeds: 7, 144, 281)

## Summary

- **Total cases**: 24 (8 topologies x 3 design variables)
- **Converged (loss < 0.01)**: 20/24
- **Best result**: aniso_crystal / both (loss = 9.74e-20)

## Results by Topology

### iso_crystal
_Isotropic crystal — regular triangular lattice, shape=(1,1), orientation=0_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | +0.000000 | 9.00e-02 | No | 520 | 2.8 |
| rest_lengths | +0.000000 | 9.00e-02 | No | 520 | 3.5 |
| both | +0.000000 | 9.00e-02 | No | 520 | 3.9 |

### aniso_crystal
_Anisotropic crystal — stretched lattice, shape=(1.5, 0.8), orientation=pi/6_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | -0.300000 | 1.89e-17 | Yes | 40 | 0.3 |
| rest_lengths | -0.300000 | 1.79e-16 | Yes | 80 | 1.1 |
| both | -0.300000 | 9.74e-20 | Yes | 40 | 0.3 |

### foam_eta02_42
_Foam eta=0.2 (seed 42) — Delaunay BEFORE perturbation_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | -0.300000 | 4.89e-17 | Yes | 60 | 0.6 |
| rest_lengths | -0.300000 | 2.46e-17 | Yes | 80 | 0.8 |
| both | -0.300000 | 2.67e-17 | Yes | 40 | 0.4 |

### foam_eta02_137
_Foam eta=0.2 (seed 137) — Delaunay BEFORE perturbation_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | -0.300000 | 3.87e-17 | Yes | 60 | 0.6 |
| rest_lengths | -0.300000 | 3.03e-17 | Yes | 80 | 0.8 |
| both | -0.300000 | 2.73e-16 | Yes | 60 | 0.9 |

### foam_eta045_256
_Foam eta=0.45 (seed 256) — Delaunay BEFORE perturbation_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | -0.299889 | 1.23e-08 | No | 520 | 2.2 |
| rest_lengths | -0.300023 | 5.25e-10 | No | 520 | 1.9 |
| both | -0.299881 | 1.42e-08 | No | 520 | 3.1 |

### foam_eta045_314
_Foam eta=0.45 (seed 314) — Delaunay BEFORE perturbation_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | -0.300002 | 3.31e-12 | Yes | 520 | 1.8 |
| rest_lengths | -0.299909 | 8.32e-09 | No | 520 | 2.4 |
| both | -0.493973 | 3.76e-02 | No | 520 | 2.4 |

### poisson_999
_Poisson random network (seed 999) — Delaunay AFTER perturbation, eta=0.3_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | -0.300000 | 3.22e-17 | Yes | 40 | 0.2 |
| rest_lengths | -0.300000 | 2.53e-15 | Yes | 60 | 0.7 |
| both | -0.300000 | 1.06e-17 | Yes | 40 | 0.4 |

### poisson_1337
_Poisson random network (seed 1337) — Delaunay AFTER perturbation, eta=0.3_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | -0.300000 | 2.43e-17 | Yes | 40 | 0.2 |
| rest_lengths | -0.300000 | 1.44e-15 | Yes | 60 | 0.7 |
| both | -0.300000 | 2.96e-17 | Yes | 60 | 0.8 |
