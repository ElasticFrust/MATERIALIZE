# Target Poisson Ratio: nu = +0.1

## Run Configuration

- **Target nu**: +0.1
- **Mesh size**: (10, 10)
- **Design variables**: rigidities, rest_lengths, both
- **Optimizer**: L-BFGS, lr=0.05, tol=1e-14, max_iter=500
- **Random restarts per case**: 3 (seeds: 7, 144, 281)

## Summary

- **Total cases**: 24 (8 topologies x 3 design variables)
- **Converged (loss < 0.01)**: 19/24
- **Best result**: foam_eta02_42 / rigidities (loss = 1.46e-17)

## Results by Topology

### iso_crystal
_Isotropic crystal — regular triangular lattice, shape=(1,1), orientation=0_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | +0.100000 | 1.96e-17 | Yes | 80 | 0.8 |
| rest_lengths | +0.100000 | 2.37e-17 | Yes | 80 | 0.9 |
| both | +0.100000 | 2.45e-17 | Yes | 60 | 0.9 |

### aniso_crystal
_Anisotropic crystal — stretched lattice, shape=(1.5, 0.8), orientation=pi/6_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | +0.100000 | 3.84e-17 | Yes | 60 | 0.6 |
| rest_lengths | +0.100000 | 2.19e-16 | Yes | 60 | 0.9 |
| both | +0.100000 | 1.87e-17 | Yes | 40 | 0.4 |

### foam_eta02_42
_Foam eta=0.2 (seed 42) — Delaunay BEFORE perturbation_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | +0.100000 | 1.46e-17 | Yes | 60 | 0.6 |
| rest_lengths | +0.100000 | 1.98e-16 | Yes | 60 | 0.6 |
| both | +0.100000 | 1.67e-17 | Yes | 60 | 0.9 |

### foam_eta02_137
_Foam eta=0.2 (seed 137) — Delaunay BEFORE perturbation_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | +0.100000 | 1.48e-17 | Yes | 80 | 0.8 |
| rest_lengths | +0.100000 | 1.61e-15 | Yes | 60 | 0.7 |
| both | +0.100000 | 2.16e-17 | Yes | 60 | 0.8 |

### foam_eta045_256
_Foam eta=0.45 (seed 256) — Delaunay BEFORE perturbation_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | -0.000692 | 1.01e-02 | No | 520 | 2.3 |
| rest_lengths | -0.000752 | 1.02e-02 | No | 520 | 3.1 |
| both | -0.000801 | 1.02e-02 | No | 520 | 3.4 |

### foam_eta045_314
_Foam eta=0.45 (seed 314) — Delaunay BEFORE perturbation_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | -0.001692 | 1.03e-02 | No | 520 | 3.0 |
| rest_lengths | +0.133017 | 1.09e-03 | No | 520 | 2.4 |
| both | -0.001691 | 1.03e-02 | No | 520 | 4.1 |

### poisson_999
_Poisson random network (seed 999) — Delaunay AFTER perturbation, eta=0.3_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | +0.100000 | 2.33e-17 | Yes | 60 | 0.5 |
| rest_lengths | +0.100000 | 6.08e-17 | Yes | 60 | 0.7 |
| both | +0.100000 | 2.04e-17 | Yes | 60 | 0.8 |

### poisson_1337
_Poisson random network (seed 1337) — Delaunay AFTER perturbation, eta=0.3_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | +0.100000 | 1.59e-17 | Yes | 60 | 0.6 |
| rest_lengths | +0.100000 | 4.45e-17 | Yes | 60 | 0.6 |
| both | +0.100000 | 1.83e-17 | Yes | 60 | 0.8 |
