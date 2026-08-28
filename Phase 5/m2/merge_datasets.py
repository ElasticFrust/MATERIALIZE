r"""Merge M2 dataset shards built at different seeds into one npz.

Why this exists: `build_dataset.py` is single-threaded and takes ~1.6 h per shard, so scaling the
training set means building several shards (different `--seed`) and concatenating them, rather than
one enormous serial build.

THE FORMAT. `build_dataset.save` uses a POINTER SCHEME: variable-size per-sample arrays (`pts`,
`bond_R`, `C6_per`, ...) are concatenated flat, with a companion `<key>_ptr` of cumulative offsets,
`len == n_samples + 1`. Everything else is one row per sample. So the merge is uniform:

    <key>_ptr   ->  concat(A_ptr, B_ptr[1:] + A_ptr[-1])      # re-base B's offsets onto A's tail
    everything  ->  concat along axis 0

THE LEAKAGE TRAP, and the reason this is not a three-line `np.concatenate`. `M2_V2_PLAN.md` §3.5
records two ways a split can silently inflate its score, one of them being a TRAJECTORY split that
lets frames of one trajectory land on both sides. Shards built at different seeds can reuse the same
`traj_id` / `topology_id` strings, so a naive merge would fuse two unrelated trajectories under one
id -- and the splitter would then happily put "the same" trajectory in train and val. Every id is
therefore NAMESPACED by its shard before concatenation.

Run:
    python "Phase 5/m2/merge_datasets.py" --out data/dataset_merged.npz \
           data/dataset_xl.npz data/dataset_xl_s1.npz
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

#: ids namespaced per shard so trajectories from different builds can never be fused (see §3.5)
ID_KEYS = ('traj_id', 'topology_id')


def merge(paths, out, verbose=True):
    """Concatenate `paths` into `out`. Returns the merged sample count."""
    shards = []
    for p in paths:
        with np.load(p, allow_pickle=False) as d:
            shards.append({k: d[k] for k in d.files})
        if verbose:
            n = len(shards[-1]['C6'])
            print('  %-34s %6d samples, %3d arrays' % (os.path.basename(p), n,
                                                       len(shards[-1])))

    keys = set(shards[0])
    for i, s in enumerate(shards[1:], 1):
        if set(s) != keys:
            raise SystemExit('shard %d has different keys than shard 0 (%s); built by different '
                             'builder versions?' % (i, sorted(set(s) ^ keys)))

    # namespace the ids BEFORE concatenating
    for tag, s in zip((os.path.splitext(os.path.basename(p))[0] for p in paths), shards):
        for k in ID_KEYS:
            if k in s:
                s[k] = np.array(['%s/%s' % (tag, v) for v in s[k]])

    merged = {}
    for k in sorted(keys):
        if k.endswith('_ptr'):
            parts, base = [shards[0][k]], shards[0][k][-1]
            for s in shards[1:]:
                parts.append(s[k][1:] + base)
                base = base + s[k][-1]
            merged[k] = np.concatenate(parts)
        else:
            merged[k] = np.concatenate([s[k] for s in shards], axis=0)

    n = len(merged['C6'])
    # the pointer invariant is the one thing that silently corrupts every downstream sample if
    # wrong, so assert it rather than trust the arithmetic
    for k in [k for k in merged if k.endswith('_ptr')]:
        base = k[:-4]
        if len(merged[k]) != n + 1:
            raise SystemExit('%s has length %d, expected n+1 = %d' % (k, len(merged[k]), n + 1))
        if merged[k][-1] != len(merged[base]):
            raise SystemExit('%s ends at %d but %s has %d rows'
                             % (k, merged[k][-1], base, len(merged[base])))
    if len(set(merged['traj_id'])) != sum(len(set(s['traj_id'])) for s in shards):
        raise SystemExit('traj_id namespacing failed -- ids collide across shards, which would let '
                         'one trajectory land on both sides of the split (plan section 3.5)')

    np.savez_compressed(out, **merged)
    if verbose:
        print('  -> %s   %d samples, %d distinct traj_id' % (out, n, len(set(merged['traj_id']))))
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('shards', nargs='+', help='npz shards to merge (as built by build_dataset.py)')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    paths = [p if os.path.isabs(p) else os.path.join(HERE, p) for p in a.shards]
    out = a.out if os.path.isabs(a.out) else os.path.join(HERE, a.out)
    if os.path.exists(out):
        raise SystemExit('refusing to overwrite %s -- merge writes a NEW file' % out)
    merge(paths, out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
