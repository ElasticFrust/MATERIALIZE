# Target Poisson Ratio: nu = +0.3

## Run Configuration

- **Target nu**: +0.3
- **Mesh size**: (10, 10)
- **Design variables**: rigidities, rest_lengths, both
- **Optimizer**: L-BFGS, lr=0.05, tol=1e-14, max_iter=500
- **Random restarts per case**: 3 (seeds: 7, 144, 281)

## Summary

- **Total cases**: 24 (8 topologies x 3 design variables)
- **Converged (loss < 0.01)**: 20/24
- **Best result**: poisson_999 / both (loss = 1.25e-17)

## Results by Topology

### iso_crystal
_Isotropic crystal — regular triangular lattice, shape=(1,1), orientation=0_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | +0.300000 | 2.50e-17 | Yes | 60 | 0.6 |
| rest_lengths | +0.300000 | 1.50e-17 | Yes | 60 | 0.6 |
| both | +0.300000 | 7.42e-17 | Yes | 60 | 0.9 |

### aniso_crystal
_Anisotropic crystal — stretched lattice, shape=(1.5, 0.8), orientation=pi/6_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | +0.300000 | 2.29e-17 | Yes | 60 | 0.7 |
| rest_lengths | +0.300000 | 1.45e-17 | Yes | 60 | 0.7 |
| both | +0.300000 | 1.75e-17 | Yes | 60 | 0.7 |

### foam_eta02_42
_Foam eta=0.2 (seed 42) — Delaunay BEFORE perturbation_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | +0.300000 | 1.33e-17 | Yes | 60 | 0.5 |
| rest_lengths | +0.300000 | 2.55e-17 | Yes | 60 | 0.6 |
| both | +0.300000 | 1.48e-17 | Yes | 60 | 1.1 |

### foam_eta02_137
_Foam eta=0.2 (seed 137) — Delaunay BEFORE perturbation_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | +0.300000 | 3.04e-17 | Yes | 60 | 0.7 |
| rest_lengths | +0.300000 | 1.45e-17 | Yes | 60 | 0.6 |
| both | +0.300000 | 2.16e-17 | Yes | 60 | 0.9 |

### foam_eta045_256
_Foam eta=0.45 (seed 256) — Delaunay BEFORE perturbation_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | -0.000714 | 9.04e-02 | No | 520 | 2.9 |
| rest_lengths | -0.000819 | 9.05e-02 | No | 520 | 3.0 |
| both | -0.000737 | 9.04e-02 | No | 520 | 2.9 |

### foam_eta045_314
_Foam eta=0.45 (seed 314) — Delaunay BEFORE perturbation_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | -0.001698 | 9.10e-02 | No | 520 | 3.1 |
| rest_lengths | +0.241386 | 3.44e-03 | No | 520 | 2.5 |
| both | +0.303926 | 1.54e-05 | No | 520 | 4.5 |

### poisson_999
_Poisson random network (seed 999) — Delaunay AFTER perturbation, eta=0.3_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | +0.300000 | 2.47e-17 | Yes | 60 | 0.5 |
| rest_lengths | +0.300000 | 3.44e-16 | Yes | 60 | 0.7 |
| both | +0.300000 | 1.25e-17 | Yes | 60 | 0.9 |

### poisson_1337
_Poisson random network (seed 1337) — Delaunay AFTER perturbation, eta=0.3_

| Design Variable | Achieved nu | Loss | Converged | Iterations | Time (s) |
|---|---|---|---|---|---|
| rigidities | +0.300000 | 1.85e-17 | Yes | 60 | 0.7 |
| rest_lengths | +0.300000 | 3.45e-16 | Yes | 60 | 0.7 |
| both | +0.300000 | 1.50e-17 | Yes | 60 | 0.8 |
