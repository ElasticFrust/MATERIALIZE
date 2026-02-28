#!/usr/bin/env python
"""Production dataset generation script.

Generates train/val/test splits with progress logging.
Run from Phase 4 directory: python run_generate_dataset.py
"""
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "Phase 2"))
sys.path.insert(0, str(PROJECT_ROOT / "Phase 3"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import numpy as np
from data.generate_dataset import generate_dataset

OUTPUT_DIR = Path(__file__).resolve().parent / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Production sizes
SPLITS = [
    ('train', 80000),
    ('val',   10000),
    ('test',  10000),
]
N_WORKERS = 8

total_start = time.time()

for split, n in SPLITS:
    print(f"\n{'='*60}")
    print(f"Generating {split} split ({n} samples, {N_WORKERS} workers)")
    print(f"{'='*60}")
    t0 = time.time()
    data_list = generate_dataset(n, n_workers=N_WORKERS)
    elapsed = time.time() - t0

    save_path = OUTPUT_DIR / f'{split}.pt'
    torch.save(data_list, save_path)

    rate = len(data_list) / elapsed if elapsed > 0 else 0
    print(f"\nSaved {len(data_list)} samples to {save_path}")
    print(f"Time: {elapsed:.0f}s ({rate:.1f} samples/sec)")

    # Summary stats
    nus = [d.y_nu.item() for d in data_list]
    from collections import Counter
    patterns = Counter(d.rig_pattern for d in data_list)
    topos = Counter(d.topo_name for d in data_list)
    print(f"Poisson range: [{min(nus):.3f}, {max(nus):.3f}], mean={np.mean(nus):.3f}")
    print(f"Patterns: {dict(patterns)}")
    print(f"Topos ({len(topos)} unique): {dict(topos)}")
    sys.stdout.flush()

total_elapsed = time.time() - total_start
print(f"\n{'='*60}")
print(f"DONE. Total time: {total_elapsed/60:.1f} minutes")
print(f"{'='*60}")
