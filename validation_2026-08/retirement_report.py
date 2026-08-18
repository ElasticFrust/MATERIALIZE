r"""What is safe to retire — the A-19 cleanup, as evidence rather than a guess.

For every directory holding UNSTAMPED `.npz` (pre-B-3, hence pre-August forward-map fixes; see
audit A-19), report how many files, how much disk, and **which scripts still reference that
directory**. Reference is checked at DIRECTORY granularity on purpose: most consumers glob
(`design_g2att_{case}_*.npz`), so matching exact basenames would badly undercount and make orphaned
data look live.

  referenced  -> retiring needs the consumer re-run or removed FIRST (this is the fig5 trap:
                 its producer was out of scope, so it re-rendered a 9-July network and exited 0)
  orphaned    -> no script mentions it; safe to retire on its own

Run:  python validation_2026-08/retirement_report.py
"""
import collections
import glob
import json
import os

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
SKIP = ('validation_2026-08', '.git', '__pycache__')


def unstamped():
    import numpy as np
    out = []
    for f in glob.glob(os.path.join(REPO, '**', '*.npz'), recursive=True):
        if any(s in f for s in SKIP):
            continue
        try:
            z = np.load(f, allow_pickle=True)
            m = z['meta'].item() if 'meta' in z.files else {}
            m = json.loads(m) if isinstance(m, str) else (m or {})
        except Exception:
            m = {}
        if not m.get('commit'):
            out.append(f)
    return out


def sources():
    for f in glob.glob(os.path.join(REPO, '**', '*.py'), recursive=True):
        if any(s in f for s in SKIP):
            continue
        try:
            yield os.path.relpath(f, REPO), open(f, encoding='utf-8', errors='replace').read()
        except OSError:
            continue


def main():
    files = unstamped()
    src = list(sources())
    by_dir = collections.defaultdict(list)
    for f in files:
        by_dir[os.path.relpath(os.path.dirname(f), REPO)].append(f)

    print(f'{len(files)} unstamped .npz in {len(by_dir)} directories\n')
    print(f'{"files":>6} {"MB":>7}  directory / referencing scripts')
    total_orphan_mb = 0.0
    for d, fs in sorted(by_dir.items(), key=lambda kv: -len(kv[1])):
        mb = sum(os.path.getsize(f) for f in fs) / 1e6
        # Match the LAST TWO path components, both separators. Matching only the leaf makes a
        # directory called `networks` match the WORD "networks" in ~60 unrelated files, which
        # inflates every count and makes orphaned data look live.
        parts = d.replace(chr(92), '/').split('/')
        frag = '/'.join(parts[-2:]) if len(parts) > 1 else parts[-1]
        cands = {frag, frag.replace('/', chr(92)), os.path.basename(d) + '.npz'}
        stems = {os.path.splitext(os.path.basename(f))[0] for f in fs}
        refs = sorted({s for s, txt in src
                       if 'retirement_report' not in s and 'scan_provenance' not in s
                       and (any(c in txt for c in cands) or any(st in txt for st in stems))})
        print(f'{len(fs):6d} {mb:7.1f}  {d}')
        if refs:
            for r in refs[:4]:
                print(f'{"":16}referenced by {r}')
            if len(refs) > 4:
                print(f'{"":16}... and {len(refs)-4} more')
        else:
            total_orphan_mb += mb
            print(f'{"":16}ORPHANED — no script references it')
    print(f'\northaned total: {total_orphan_mb:.1f} MB')


if __name__ == '__main__':
    main()
