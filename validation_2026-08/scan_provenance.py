r"""Repo-wide provenance scan of saved networks — the A-19 detector.

`save_network` stamps `commit`/`dirty`/`saved_utc`/`seed` (audit **B-3**). Anything WITHOUT that
stamp was written before B-3 existed, hence before the August forward-map fixes (**A-0** shear
contraction, **A-10** ν convention) — so its *recorded response* (`C6`, ν, E) is stale even though
its geometry and k are still perfectly good numbers.

Unstamped is therefore a **sufficient filter for "suspect"**, and it is the durable answer to A-19:
tag the class, rather than redesigning files one at a time.

Measured 2026-08-18: 1171 npz, 651 stamped, **520 unstamped — 492 of them dated July 2026**.

Run:  python validation_2026-08/scan_provenance.py [--csv out.csv]
"""
import argparse
import collections
import csv
import datetime as dt
import glob
import json
import os

SKIP = ('validation_2026-08', '.git')


def meta_of(path):
    import numpy as np
    try:
        z = np.load(path, allow_pickle=True)
        m = z['meta'].item() if 'meta' in z.files else {}
        return json.loads(m) if isinstance(m, str) else (m or {})
    except Exception:
        return {}


def scan(root='.'):
    for f in glob.glob(os.path.join(root, '**', '*.npz'), recursive=True):
        if any(s in f for s in SKIP):
            continue
        m = meta_of(f)
        yield (f, m.get('commit') or '', m.get('saved_utc') or '',
               dt.date.fromtimestamp(os.path.getmtime(f)).isoformat())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default=None)
    a = ap.parse_args()
    rows = list(scan())
    unst = [r for r in rows if not r[1]]
    print(f'npz {len(rows)}   stamped {len(rows)-len(unst)}   UNSTAMPED (suspect) {len(unst)}')
    by_month = collections.Counter(r[3][:7] for r in unst)
    for k in sorted(by_month):
        print(f'   unstamped, mtime {k}: {by_month[k]}')
    print('\nunstamped by directory:')
    for d, n in collections.Counter(os.path.dirname(r[0]) for r in unst).most_common():
        print(f'  {n:5d}  {d}')
    if a.csv:
        with open(a.csv, 'w', newline='', encoding='utf-8') as fh:
            w = csv.writer(fh)
            w.writerow(['path', 'commit', 'saved_utc', 'mtime_date', 'suspect'])
            for r in rows:
                w.writerow([*r, 'YES' if not r[1] else ''])
        print('wrote', a.csv)


if __name__ == '__main__':
    main()
