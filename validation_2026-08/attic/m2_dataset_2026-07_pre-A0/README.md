# M2 training set — 2026-07, PRE-A-0 baseline (archived, do not train on this)

Archived 2026-08-18 from `Phase 5/dataset/` and `Phase 5/m2/data/`. Kept, not deleted, because it is
the **only record of what the existing M2 checkpoint actually learned** — without it the old model's
behaviour cannot be explained, and a before/after comparison after retraining is impossible.

## Why it is retired

Built 23 July 2026, i.e. **before** the August forward-map fixes (**A-0** shear contraction,
**A-10** ν convention). Its labels were therefore computed with the old map. Measured 2026-08-18 by
recomputing ν(θ) from each sample's stored geometry + k with the current solver — **37 of 41 sampled
labels drift by more than 1e-3**:

| family | n | median \|Δν\| | max \|Δν\| |
|---|---|---|---|
| tiling | 5 | 4.79e-01 | **1.73e+00** |
| auxetic | 3 | 5.23e-01 | 1.55e+00 |
| honeycomb | 2 | 5.32e-01 | 6.08e-01 |
| random | 12 | 2.22e-01 | 5.62e-01 |
| kagome | 1 | 2.45e-01 | 2.45e-01 |
| flipped | 1 | 2.27e-01 | 2.27e-01 |
| bravais | 16 | 6.55e-02 | 1.74e-01 |
| jitter | 1 | 2.02e-02 | 2.02e-02 |

So **the GNN was trained on labels wrong by ν ≈ 0.2–1.7 across most of its training set.** The
ordering is consistent with A-0 biting hardest where the strain-concentration `W` is largest
(`bravais`, the most regular family, drifts least) — but **no subset survived**: even bravais is
stale at a median of 0.066. There is no salvageable slice here.

## Rebuilding

```
python "Phase 5/dataset.py"            # regenerates Phase 5/dataset/ (nets + dataset.npz)
python "Phase 5/m2/build_dataset.py"   # regenerates Phase 5/m2/data/dataset.npz (graphs + labels)
python "Phase 5/m2/train.py"           # retrain
```

Until those run, `Phase 5/m2/train.py` has no input — that is intended, since training on the old
labels is exactly what must not happen by accident.

`Phase 5/m2/checkpoint.pt` (tracked) is the model trained on THIS data. Treat its outputs as
unverified until retrained.

Contents: `Phase5_dataset/` (121 sample `.npz` + `dataset.npz` + `coverage.png`), `Phase5_m2_data/`.
