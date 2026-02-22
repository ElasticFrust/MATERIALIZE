# Target Poisson Ratio: nu = 0.7

## Run Configuration

- **Target nu**: 0.7
- **Mesh size**: (10, 10)
- **Design variables**: rigidities, rest_lengths, both
- **Optimizer**: L-BFGS, lr=0.05, tol=1e-14, max_iter=2000
- **Random restarts per case**: 3 (seeds: 7, 144, 281)

## Summary

- **Total cases**: 24 (8 topologies x 3 design variables)
- **Converged (loss < 0.01)**: 18/24
- **Best result**: poisson_1337 / rigidities (loss = 3.20e-21)

## Results by Topology

### iso_crystal
_Isotropic crystal — regular triangular lattice, shape=(1,1), orientation=0_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | +0.700000 | 2.41e-17 | Yes | 60 | 0.5 |
| rest_lengths | +0.700000 | 5.97e-16 | Yes | 60 | 0.7 |
| both | +0.700000 | 1.49e-17 | Yes | 60 | 0.7 |

### aniso_crystal
_Anisotropic crystal — stretched lattice, shape=(1.5, 0.8), orientation=pi/6_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | +0.700000 | 1.68e-17 | Yes | 40 | 0.3 |
| rest_lengths | +0.700000 | 1.25e-17 | Yes | 60 | 0.8 |
| both | +0.700000 | 2.78e-17 | Yes | 60 | 0.7 |

### foam_eta02_42
_Foam eta=0.2 (seed 42) — Delaunay BEFORE perturbation_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | +0.700000 | 2.09e-17 | Yes | 40 | 0.3 |
| rest_lengths | +0.700000 | 1.57e-17 | Yes | 60 | 0.5 |
| both | +0.700000 | 3.29e-17 | Yes | 60 | 0.8 |

### foam_eta02_137
_Foam eta=0.2 (seed 137) — Delaunay BEFORE perturbation_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | +0.700000 | 2.44e-17 | Yes | 60 | 0.6 |
| rest_lengths | +0.700000 | 1.99e-17 | Yes | 60 | 0.5 |
| both | +0.700000 | 2.73e-17 | Yes | 60 | 0.8 |

### foam_eta045_256
_Foam eta=0.45 (seed 256) — Delaunay BEFORE perturbation_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | -0.000731 | 4.91e-01 | No | 2020 | 9.9 |
| rest_lengths | -0.000750 | 4.91e-01 | No | 2020 | 21.8 |
| both | -0.000776 | 4.91e-01 | No | 2020 | 11.1 |

### foam_eta045_314
_Foam eta=0.45 (seed 314) — Delaunay BEFORE perturbation_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | -0.001696 | 4.92e-01 | No | 2020 | 10.0 |
| rest_lengths | +0.590090 | 1.21e-02 | No | 2020 | 8.5 |
| both | +0.505542 | 3.78e-02 | No | 2020 | 20.2 |

### poisson_999
_Poisson random network (seed 999) — Delaunay AFTER perturbation, eta=0.3_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | +0.700000 | 7.93e-15 | Yes | 40 | 0.4 |
| rest_lengths | +0.700000 | 5.40e-16 | Yes | 60 | 0.7 |
| both | +0.700000 | 2.86e-17 | Yes | 60 | 0.9 |

### poisson_1337
_Poisson random network (seed 1337) — Delaunay AFTER perturbation, eta=0.3_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | +0.700000 | 3.20e-21 | Yes | 40 | 0.2 |
| rest_lengths | +0.700000 | 9.61e-16 | Yes | 60 | 0.7 |
| both | +0.700000 | 1.05e-20 | Yes | 60 | 0.6 |
