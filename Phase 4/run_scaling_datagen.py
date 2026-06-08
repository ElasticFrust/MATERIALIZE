#!/usr/bin/env python
"""Generate train-only datasets at scaled sizes for the scaling test.

Usage:
    python run_scaling_datagen.py --output_dir ./scaling_5k --n_train 5000
    python run_scaling_datagen.py --output_dir ./scaling_10k --n_train 10000
"""
import sys
import time
import gc
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "Phase 2"))
sys.path.insert(0, str(PROJECT_ROOT / "Phase 3"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.multiprocessing as mp
import numpy as np
from collections import Counter

from data.generate_dataset import _generate_sample_config_list, _worker

parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', required=True)
parser.add_argument('--n_train', type=int, required=True)
parser.add_argument('--n_workers', type=int, default=4)
parser.add_argument('--chunk_size', type=int, default=1000)
args = parser.parse_args()

output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

print(f"Generating train split: {args.n_train} samples -> {output_dir}")
t0 = time.time()
all_data = []
remaining = args.n_train

while remaining > 0:
    batch = min(args.chunk_size, remaining)
    configs = _generate_sample_config_list(batch)
    with mp.Pool(args.n_workers) as pool:
        results = pool.map(_worker, configs)
    chunk = [d for d in results if d is not None]
    all_data.extend(chunk)
    remaining -= batch
    elapsed = time.time() - t0
    rate = len(all_data) / elapsed if elapsed > 0 else 0
    eta = remaining / rate / 60 if rate > 0 else 0
    print(f"  {len(all_data)}/{args.n_train} samples "
          f"({elapsed/60:.1f}min, ~{eta:.0f}min remaining, {rate:.1f} s/s)")
    sys.stdout.flush()
    del chunk, results, configs
    gc.collect()

save_path = output_dir / 'train.pt'
torch.save(all_data, save_path)
elapsed = time.time() - t0
nus = [d.y_nu.item() for d in all_data]
topos = Counter(d.topo_name for d in all_data)
print(f"\nSaved {len(all_data)} samples to {save_path} ({save_path.stat().st_size/1024/1024:.1f} MB)")
print(f"Time: {elapsed/60:.1f} min")
print(f"Poisson range: [{min(nus):.3f}, {max(nus):.3f}], mean={np.mean(nus):.3f}")
print(f"Topologies ({len(topos)}): {dict(topos)}")
