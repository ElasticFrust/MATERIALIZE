# Target Poisson Ratio: nu = +0.5

## Run Configuration

- **Target nu**: +0.5
- **Mesh size**: (10, 10)
- **Design variables**: rigidities, rest_lengths, both
- **Optimizer**: L-BFGS, lr=0.05, tol=1e-14, max_iter=500
- **Random restarts per case**: 3 (seeds: 7, 144, 281)

## Summary

- **Total cases**: 24 (8 topologies x 3 design variables)
- **Converged (loss < 0.01)**: 20/24
- **Best result**: foam_eta02_137 / rigidities (loss = 1.48e-17)

## Results by Topology

### iso_crystal
_Isotropic crystal — regular triangular lattice, shape=(1,1), orientation=0_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | +0.500000 | 1.69e-17 | Yes | 60 | 0.6 |
| rest_lengths | +0.500000 | 2.29e-17 | Yes | 60 | 0.6 |
| both | +0.500000 | 1.67e-17 | Yes | 60 | 0.8 |

### aniso_crystal
_Anisotropic crystal — stretched lattice, shape=(1.5, 0.8), orientation=pi/6_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | +0.500000 | 4.90e-16 | Yes | 40 | 0.4 |
| rest_lengths | +0.500000 | 2.53e-17 | Yes | 60 | 0.7 |
| both | +0.500000 | 1.48e-17 | Yes | 60 | 0.8 |

### foam_eta02_42
_Foam eta=0.2 (seed 42) — Delaunay BEFORE perturbation_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | +0.500000 | 2.27e-16 | Yes | 60 | 0.7 |
| rest_lengths | +0.500000 | 4.63e-17 | Yes | 60 | 0.6 |
| both | +0.500000 | 1.88e-17 | Yes | 60 | 0.7 |

### foam_eta02_137
_Foam eta=0.2 (seed 137) — Delaunay BEFORE perturbation_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | +0.500000 | 1.48e-17 | Yes | 60 | 0.6 |
| rest_lengths | +0.500000 | 2.26e-17 | Yes | 60 | 0.6 |
| both | +0.500000 | 2.45e-17 | Yes | 60 | 0.7 |

### foam_eta045_256
_Foam eta=0.45 (seed 256) — Delaunay BEFORE perturbation_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | +0.335262 | 2.71e-02 | No | 520 | 2.6 |
| rest_lengths | -0.000726 | 2.51e-01 | No | 520 | 2.7 |
| both | -0.000780 | 2.51e-01 | No | 520 | 3.4 |

### foam_eta045_314
_Foam eta=0.45 (seed 314) — Delaunay BEFORE perturbation_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | -0.001728 | 2.52e-01 | No | 520 | 2.8 |
| rest_lengths | +0.571554 | 5.12e-03 | No | 520 | 3.5 |
| both | +0.477222 | 5.19e-04 | No | 520 | 2.9 |

### poisson_999
_Poisson random network (seed 999) — Delaunay AFTER perturbation, eta=0.3_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | +0.500000 | 2.29e-17 | Yes | 60 | 0.6 |
| rest_lengths | +0.500000 | 1.17e-15 | Yes | 60 | 0.7 |
| both | +0.500000 | 2.25e-17 | Yes | 60 | 0.8 |

### poisson_1337
_Poisson random network (seed 1337) — Delaunay AFTER perturbation, eta=0.3_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | +0.500000 | 2.06e-17 | Yes | 60 | 0.5 |
| rest_lengths | +0.500000 | 8.81e-16 | Yes | 60 | 0.7 |
| both | +0.500000 | 4.36e-17 | Yes | 60 | 0.9 |
