#!/usr/bin/env python
"""Resumable production dataset generation script.

Generates train/val/test splits using torch.multiprocessing with
file_system sharing strategy. Saves each chunk to disk immediately
and does NOT accumulate data in memory, avoiding OOM.

Final output is saved as chunked directories (processed/train/, etc.)
with a manifest file. Training code can use load_split() to load.

Usage:
    python run_generate_dataset_resumable.py          # fresh start or resume
    python run_generate_dataset_resumable.py --fresh   # force fresh start
"""
import sys
import time
import gc
import json
import argparse
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
CHUNK_SIZE = 2000
N_WORKERS = 4


def get_split_dir(split):
    d = OUTPUT_DIR / f'{split}_chunks'
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_existing_chunks(split):
    """Find already-generated chunk files for a split."""
    return sorted(get_split_dir(split).glob('chunk_*.pt'))


def count_existing_samples(split):
    """Count samples in saved chunks using manifest if available."""
    manifest_path = get_split_dir(split) / 'manifest.json'
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        return manifest['total_samples']
    # Fallback: count by loading
    total = 0
    for chunk_path in get_existing_chunks(split):
        data = torch.load(chunk_path, weights_only=False)
        total += len(data)
        del data
    gc.collect()
    return total


def split_is_complete(split):
    """Check if generation is done for this split."""
    manifest_path = get_split_dir(split) / 'manifest.json'
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        return manifest.get('complete', False)
    return False


def save_manifest(split, chunk_info, total_samples, complete=False, stats=None):
    """Save a manifest file with chunk metadata."""
    manifest = {
        'split': split,
        'total_samples': total_samples,
        'n_chunks': len(chunk_info),
        'chunks': chunk_info,
        'complete': complete,
    }
    if stats:
        manifest['stats'] = stats
    manifest_path = get_split_dir(split) / 'manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)


def compute_stats_streaming(split):
    """Compute summary statistics without loading all data at once."""
    all_nus = []
    pattern_counts = Counter()
    topo_counts = Counter()
    for chunk_path in get_existing_chunks(split):
        data = torch.load(chunk_path, weights_only=False)
        for d in data:
            all_nus.append(d.y_nu.item())
            pattern_counts[str(d.rig_pattern)] += 1
            topo_counts[d.topo_name] += 1
        del data
        gc.collect()
    return {
        'nu_min': float(min(all_nus)),
        'nu_max': float(max(all_nus)),
        'nu_mean': float(np.mean(all_nus)),
        'patterns': dict(pattern_counts),
        'n_unique_topos': len(topo_counts),
        'topos': dict(topo_counts),
    }


def generate_split(split, n, fresh=False):
    """Generate one split with resumability. Does NOT accumulate in memory."""
    if not fresh and split_is_complete(split):
        manifest_path = get_split_dir(split) / 'manifest.json'
        with open(manifest_path) as f:
            manifest = json.load(f)
        print(f"\n[{split}] Already complete "
              f"({manifest['total_samples']} samples in "
              f"{manifest['n_chunks']} chunks)")
        return

    # Check existing partial progress
    if fresh:
        for f in get_existing_chunks(split):
            f.unlink()
        manifest_path = get_split_dir(split) / 'manifest.json'
        if manifest_path.exists():
            manifest_path.unlink()
        existing_count = 0
        chunk_idx = 0
        chunk_info = []
    else:
        chunks = get_existing_chunks(split)
        chunk_idx = len(chunks)
        if chunk_idx > 0:
            existing_count = count_existing_samples(split)
            # Rebuild chunk_info from existing files
            chunk_info = []
            for cp in chunks:
                d = torch.load(cp, weights_only=False)
                chunk_info.append({
                    'file': cp.name,
                    'n_samples': len(d),
                })
                del d
            gc.collect()
            print(f"  Resuming: {existing_count} samples in {chunk_idx} chunks")
        else:
            existing_count = 0
            chunk_info = []

    print(f"\n{'='*60}")
    print(f"Generating {split} split ({n} total, {N_WORKERS} workers)")
    if existing_count > 0:
        print(f"  Resuming from {existing_count} existing samples")
    print(f"{'='*60}")
    sys.stdout.flush()

    t0 = time.time()
    generated_this_run = 0
    total_generated = existing_count
    remaining = n - total_generated

    while remaining > 0:
        batch = min(CHUNK_SIZE, remaining)
        configs = _generate_sample_config_list(batch)

        with mp.Pool(N_WORKERS) as pool:
            results = pool.map(_worker, configs)

        chunk = [d for d in results if d is not None]
        del results

        # Save chunk to disk immediately
        chunk_name = f'chunk_{chunk_idx:04d}.pt'
        chunk_path = get_split_dir(split) / chunk_name
        torch.save(chunk, chunk_path)

        chunk_info.append({
            'file': chunk_name,
            'n_samples': len(chunk),
        })

        generated_this_run += len(chunk)
        total_generated += len(chunk)
        chunk_idx += 1
        remaining -= batch

        # Update manifest after each chunk (for resumability)
        save_manifest(split, chunk_info, total_generated, complete=False)

        elapsed = time.time() - t0
        rate = generated_this_run / elapsed if elapsed > 0 else 0
        eta = remaining / rate / 60 if rate > 0 else 0
        print(f"  [{split}] {total_generated}/{n} samples "
              f"({elapsed/60:.1f}min, ~{eta:.0f}min left, "
              f"{rate:.1f}/sec) [chunk {chunk_idx} saved]")
        sys.stdout.flush()

        # Free chunk from memory immediately
        del chunk, configs
        gc.collect()

    # Compute stats by streaming through chunks
    elapsed = time.time() - t0
    print(f"\n  Computing statistics...")
    sys.stdout.flush()
    stats = compute_stats_streaming(split)

    # Save final manifest
    save_manifest(split, chunk_info, total_generated, complete=True, stats=stats)

    rate = generated_this_run / elapsed if elapsed > 0 else 0
    total_size = sum(f.stat().st_size for f in get_existing_chunks(split))
    size_mb = total_size / 1024 / 1024
    print(f"\nDone: {total_generated} samples in {chunk_idx} chunks ({size_mb:.1f} MB)")
    print(f"Time: {elapsed/60:.1f} min ({rate:.1f} samples/sec)")
    print(f"Poisson: [{stats['nu_min']:.3f}, {stats['nu_max']:.3f}], "
          f"mean={stats['nu_mean']:.3f}")
    print(f"Patterns: {stats['patterns']}")
    print(f"Topos ({stats['n_unique_topos']} unique): {stats['topos']}")
    sys.stdout.flush()

    gc.collect()


def load_split(split, data_dir=None):
    """Load a split from chunk files. Use this in training code.

    Returns list of PyG Data objects.
    """
    if data_dir is None:
        data_dir = OUTPUT_DIR
    split_dir = data_dir / f'{split}_chunks'

    # Try chunked format first
    if split_dir.exists():
        chunks = sorted(split_dir.glob('chunk_*.pt'))
        all_data = []
        for chunk_path in chunks:
            all_data.extend(torch.load(chunk_path, weights_only=False))
        return all_data

    # Fallback to single-file format
    single = data_dir / f'{split}.pt'
    if single.exists():
        return torch.load(single, weights_only=False)

    raise FileNotFoundError(f"No data found for split '{split}' in {data_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fresh', action='store_true',
                        help='Force fresh start, discard partial data')
    args = parser.parse_args()

    total_start = time.time()

    for split, n in SPLITS:
        generate_split(split, n, fresh=args.fresh)

    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"DONE. Total time: {total_elapsed/60:.1f} minutes")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
