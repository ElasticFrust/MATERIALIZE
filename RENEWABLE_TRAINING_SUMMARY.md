# Renewable Per-Chunk Training — Rationale & Monitoring Guide

## What Changed

### Before
Training loaded a fixed pre-generated dataset saved as `.pt` chunk files.
On every epoch the training loop re-read the same `chunk_0000.pt …
chunk_0039.pt` files, shuffling within each chunk but always using the same
~80 000 samples.  The model was therefore exposed to the same topology
instances, the same rigidity realisations, and the same Poisson-ratio labels
on every pass.

### After
A new **renewable mode** replaces disk reads with live data generation.  Before
each chunk is trained, `generate_chunk_live()` is called: it draws a fresh
random batch of samples (new topologies, new rigidity patterns, new σ levels,
new rest lengths) via `multiprocessing.Pool` and passes them directly to the
`DataLoader`.  No `.pt` file is written or read; the data exists only in RAM
for the duration of that chunk's training step, then is discarded.

**Concretely, with the defaults (40 chunks × ~1 860 samples/chunk):**

| Mode | Unique samples seen after N epochs |
|------|------------------------------------|
| Static dataset | ~80 000 (always the same) |
| Renewable (this PR) | N × ~74 400 (all distinct) |


## Why Renewable Per-Chunk Matters

### 1. Prevents Memorisation
With a static dataset the model can — and eventually will — fit idiosyncrasies
of specific parameter realisations rather than the underlying physics.
Symptoms: val/train loss gap widens, or val loss plateaus even though the model
clearly has capacity left.  Renewable training makes this impossible: no sample
is ever seen twice.

### 2. Effective Data Augmentation Without Storage Cost
The variation axes (18 topologies × 3 mesh sizes × 3 design-variable types ×
8 rigidity patterns × 4 σ values = 648 base configs, each with independent
random draws) define an astronomically large distribution.  Storing an
exhaustive pre-generated dataset would require hundreds of GB.  Live generation
costs ~0.5 s/sample on CPU; at 4 workers this is typically fast enough to keep
the GPU saturated between chunks.

### 3. Better Generalisation Across All Config Combinations
Static chunk files were generated with a fixed random seed at generation time,
so some config combinations may be over- or under-represented by chance.
Renewable sampling re-draws the config list each chunk, ensuring every epoch
covers the full design space more uniformly.

### 4. Curriculum Emerges Naturally
Hard samples (extreme σ, unusual topology × pattern combinations that produce
edge-case Poisson ratios) appear at the same rate throughout training rather
than being clustered in specific chunk files.  This tends to produce smoother
loss curves and more robust convergence.

### 5. Val / Test Sets Remain Static
Validation and test splits are still loaded from pre-saved `.pt` files.  This
is intentional: stable evaluation requires that the same samples are scored
every epoch so improvements in val_mse are meaningful.  Only the *training*
stream is made renewable.


## Why Continuous Monitoring is Essential

### Environment Instability
This compute environment kills long-running processes (OOM, preemption, node
timeouts).  A training run of 300 epochs × 40 chunks × ~30 s/chunk ≈ 100 hours
total cannot complete in a single unattended run without a safety net.

### Auto-restart Preserves Progress
The `run_renewable_training.py` monitor:
1. Launches GNN training as a subprocess with a logfile.
2. Polls process health every **20 minutes** (configurable via `--check_interval`).
3. On process death (any non-zero exit), reads the latest checkpoint and
   restarts immediately with `--resume`.
4. Because `save_checkpoint()` runs every 5 chunks inside the epoch, at most
   ~5 × 30 s = 2.5 minutes of work is lost on any restart.
5. After GNN converges (`best_model.pt` saved), automatically starts CVAE
   training under the same monitor loop.

### Log Files
| File | Contents |
|------|----------|
| `gnn_renewable.log` | Full stdout/stderr of GNN training |
| `cvae_renewable.log` | Full stdout/stderr of CVAE training |
| `monitor_renewable.log` | Timestamped health-check events, restarts, checkpoint status |

### How to Check Progress Manually
```bash
# See last training output
tail -20 gnn_renewable.log

# See monitor events
tail -30 monitor_renewable.log

# Check checkpoint epoch
python3 -c "
import torch
c = torch.load('Phase 4/checkpoints/latest_checkpoint.pt', weights_only=False)
print(f'epoch={c[\"epoch\"]}  chunk={c.get(\"chunk_idx\")}  best_val_mse={c[\"best_val_mse\"]:.6f}')
"
```

### How to Re-run if Monitor Itself Dies
```bash
cd /home/user/MATERIALIZE
nohup python run_renewable_training.py \
    --renewable_n_chunks 40 \
    --renewable_chunk_size 2000 \
    --renewable_n_workers 4 \
    --check_interval 1200 \
    > monitor_nohup.log 2>&1 &
echo "Monitor PID: $!"
```
The monitor injects `--resume` on every restart automatically, so training
continues from where it left off regardless of how many times the environment
kills the process.


## Key Files Modified

| File | Change |
|------|--------|
| `Phase 4/data/generate_dataset.py` | Added `generate_chunk_live(n_samples, n_workers)` |
| `Phase 4/gnn/train.py` | Added `train_epoch_renewable()`, `--renewable*` CLI flags |
| `Phase 4/cvae/train.py` | Added `train_epoch_renewable_cvae()`, `--renewable*` CLI flags |
| `run_renewable_training.py` | New orchestration + monitor script |


## Quick Reference — CLI Flags

```
# GNN with renewable training (no pre-generated dataset needed):
python -m gnn.train \
    --data_dir ./Phase\ 4/processed \
    --renewable \
    --renewable_n_chunks 40 \
    --renewable_chunk_size 2000 \
    --renewable_n_workers 4 \
    --epochs 300 --batch_size 32 --patience 30

# CVAE with renewable training:
python -m cvae.train \
    --data_dir ./Phase\ 4/processed \
    --gnn_checkpoint ./Phase\ 4/checkpoints/best_model.pt \
    --renewable \
    --renewable_n_chunks 40 \
    --renewable_chunk_size 2000 \
    --renewable_n_workers 4 \
    --epochs 300 --batch_size 32 --patience 40

# Full pipeline with monitoring (recommended):
python run_renewable_training.py \
    --check_interval 1200 \
    --renewable_n_workers 4
```
