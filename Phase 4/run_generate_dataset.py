#!/usr/bin/env python
"""Production dataset generation script.

Generates train/val/test splits using torch.multiprocessing with
file_system sharing strategy to avoid fd exhaustion.
"""
import sys
import time
import gc
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "Phase 2"))
sys.path.insert(0, str(PROJECT_ROOT / "Phase 3"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.multiprocessing as mp

import numpy as np
from data.generate_dataset import (
    _generate_sample_config_list,
    _worker,
)

OUTPUT_DIR = Path(__file__).resolve().parent / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Production sizes
SPLITS = [
    ('train', 80000),
    ('val',   10000),
    ('test',  10000),
]
CHUNK_SIZE = 5000   # generate in chunks for progress + memory safety
N_WORKERS = 12

total_start = time.time()

for split, n in SPLITS:
    print(f"\n{'='*60}")
    print(f"Generating {split} split ({n} samples, {N_WORKERS} workers)")
    print(f"{'='*60}")
    sys.stdout.flush()

    t0 = time.time()
    all_data = []
    remaining = n

    while remaining > 0:
        batch = min(CHUNK_SIZE, remaining)
        configs = _generate_sample_config_list(batch)

        with mp.Pool(N_WORKERS) as pool:
            results = pool.map(_worker, configs)

        chunk = [d for d in results if d is not None]
        all_data.extend(chunk)
        remaining -= batch
        elapsed = time.time() - t0
        rate = len(all_data) / elapsed if elapsed > 0 else 0
        eta = remaining / rate / 60 if rate > 0 else 0
        print(f"  [{split}] {len(all_data)}/{n} valid samples "
              f"({elapsed/60:.1f}min elapsed, ~{eta:.0f}min remaining, "
              f"{rate:.1f} samples/sec)")
        sys.stdout.flush()
        del chunk, results, configs
        gc.collect()

    elapsed = time.time() - t0
    save_path = OUTPUT_DIR / f'{split}.pt'
    torch.save(all_data, save_path)

    rate = len(all_data) / elapsed if elapsed > 0 else 0
    size_mb = save_path.stat().st_size / 1024 / 1024
    print(f"\nSaved {len(all_data)} samples to {save_path} ({size_mb:.1f} MB)")
    print(f"Time: {elapsed/60:.1f} min ({rate:.1f} samples/sec)")

    # Summary stats
    nus = [d.y_nu.item() for d in all_data]
    patterns = Counter(d.rig_pattern for d in all_data)
    topos = Counter(d.topo_name for d in all_data)
    print(f"Poisson range: [{min(nus):.3f}, {max(nus):.3f}], mean={np.mean(nus):.3f}")
    print(f"Patterns: {dict(patterns)}")
    print(f"Topos ({len(topos)} unique): {dict(topos)}")
    sys.stdout.flush()

    del all_data
    gc.collect()

total_elapsed = time.time() - total_start
print(f"\n{'='*60}")
print(f"DONE. Total time: {total_elapsed/60:.1f} minutes")
print(f"{'='*60}")
