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


def _dupe_mask(merged):
    """True where a sample exactly repeats an earlier one.

    Shards built at different seeds share their DETERMINISTIC families: the bravais (phi, psi,
    diagonal) grid and the disordered (eta, seed) grid are not seed-dependent, so ~37 % of `bravais`
    and ~36 % of `disordered` repeat exactly, while `cells`/`random` (seeded k-fields) do not repeat
    at all. Those repeats are the SAME NETWORK, not new evidence: keeping them would double-weight
    part of the holdout and part of the training distribution.

    The key is the full per-sample label `C6` (6 float64, bit-exact) plus the family and triangle
    count -- distinct networks agreeing to the last bit in all six components is not a realistic
    collision."""
    ptr = merged['C6_per_ptr']
    ntri = np.diff(ptr)
    seen, dup = set(), np.zeros(len(merged['C6']), bool)
    for i, (row, fam, nt) in enumerate(zip(merged['C6'], merged['family'], ntri)):
        key = (bytes(np.ascontiguousarray(row)), str(fam), int(nt))
        if key in seen:
            dup[i] = True
        else:
            seen.add(key)
    return dup


def merge(paths, out, dedupe=True, verbose=True):
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

    if dedupe:
        dup = _dupe_mask(merged)
        if dup.any():
            keep = ~dup
            ptr_keys = [k for k in merged if k.endswith('_ptr')]
            for k in ptr_keys:
                base = k[:-4]
                lens = np.diff(merged[k])[keep]
                parts = [merged[base][merged[k][i]:merged[k][i + 1]]
                         for i in np.where(keep)[0]]
                merged[base] = (np.concatenate(parts, axis=0) if parts
                                else merged[base][:0])
                merged[k] = np.concatenate([[0], np.cumsum(lens)])
            for k in [k for k in merged if not k.endswith('_ptr')
                      and k[:-4] not in [p[:-4] for p in ptr_keys]
                      and k not in [p[:-4] for p in ptr_keys]]:
                merged[k] = merged[k][keep]
            n = int(keep.sum())
            if verbose:
                print('  deduped: dropped %d exact repeats (%.1f %%), %d remain'
                      % (int(dup.sum()), 100 * dup.mean(), n))

    np.savez_compressed(out, **merged)
    if verbose:
        print('  -> %s   %d samples, %d distinct traj_id' % (out, n, len(set(merged['traj_id']))))
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('shards', nargs='+', help='npz shards to merge (as built by build_dataset.py)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--no_dedupe', action='store_true',
                    help='keep exact repeats (deterministic families recur across seeds)')
    a = ap.parse_args()
    paths = [p if os.path.isabs(p) else os.path.join(HERE, p) for p in a.shards]
    out = a.out if os.path.isabs(a.out) else os.path.join(HERE, a.out)
    if os.path.exists(out):
        raise SystemExit('refusing to overwrite %s -- merge writes a NEW file' % out)
    merge(paths, out, dedupe=not a.no_dedupe)
    return 0


if __name__ == '__main__':
    sys.exit(main())
