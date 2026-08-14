# Phase 5 / M2 — GNN forward surrogate

M2 v1 is a **learned forward surrogate** that amortises the differentiable solver:
given a periodic network graph (node positions + per-bond stiffness `k`), it predicts
the homogenised elastic tensor **C6 (6 numbers)**; `nu(theta)`, `E(theta)` derive from C6
exactly as the solver does. See [`M2.md`](M2.md) for the full write-up.

All scripts use `C:\Users\doron\anaconda3\python.exe` (float64). No `torch_geometric`
dependency — the GNN is plain torch (`index_add_` scatter).

## Files
- `build_dataset.py` — scaled, coverage-driven dataset builder (forward scan + ingest).
- `model.py` — `ForwardGNN` (plain-torch message-passing) + C6->nu,E helpers.
- `train.py` — training scaffold (batch-of-graphs, MSE on normalised C6) + smoke run.
- `data/dataset.npz` — consolidated dataset (all graphs + labels). `data/coverage.png`.
- `results/` — `loss_curve.png`, `pred_vs_true.png` (smoke-train diagnostics).
- `checkpoint.pt` — trained weights + C6 normalisation stats + config.

## 1. (Re)build the dataset
```bash
# modest default (~few min): 24 random topologies x 4 k-patterns + ingest all networks/
"C:\Users\doron\anaconda3\python.exe" "Phase 5\m2\build_dataset.py"

# overnight scale-up: many more topologies (parameterized)
"C:\Users\doron\anaconda3\python.exe" "Phase 5\m2\build_dataset.py" --scale
"C:\Users\doron\anaconda3\python.exe" "Phase 5\m2\build_dataset.py" --n_random 500 --n_nodes 160
```
Sources combined: (A) **forward scan** of the seed zoo (`seeds.seed_pool`: Bravais +
random point-processes + tilings + complex-basis + auxetic) x k-patterns
(`k0` / `uniform` / `lognormal` / `graded`), solver-labelled; (B) **ingest** of ALL
`Phase 5/networks/**/*.npz` (incl. any `goal1/`, `goal2/` from sibling runs) using their
independent-sim `C6_per -> region C6` — these populate the rare, interesting regions.

## 2. Train
```bash
# smoke / scaffold check (proves the pipeline learns end-to-end)
"C:\Users\doron\anaconda3\python.exe" "Phase 5\m2\train.py"

# longer / larger
"C:\Users\doron\anaconda3\python.exe" "Phase 5\m2\train.py" --epochs 800 --hidden 128 --n_layers 5
```
Loss = MSE on C6 components normalised by their **train** std; 80/20 train/val split.
Reports val MAE on the **derived** `nu` and `E`; saves loss curve + predicted-vs-true scatter.

## 3. Predict with a trained model
```python
import sys; sys.path.insert(0, r"Phase 5\m2")
from train import load_model, predict_c6
from model import c6_pred_to_nuE, c6_pred_to_nuE_theta
model, ckpt = load_model()                 # loads checkpoint.pt
C6 = predict_c6(model, geo, k)             # geo from _common; k = per-bond stiffness
nu, E = c6_pred_to_nuE(C6)                 # scalar isotropic-mean nu,E
nu_th, E_th = c6_pred_to_nuE_theta(C6)     # directional (37,) each
```

## Status
This is a **SCAFFOLD**: the data pipeline, model, and training loop run end-to-end and
the surrogate demonstrably learns, but it is **not a converged model**. To reach a usable
surrogate, scale the dataset (`--scale`) and train longer (see `M2.md` "Next steps").
