r"""Audit M2 label quality: does the solver agree with the INDEPENDENT sim on the data we TRAIN on?

Provenance: `M2_V2_PLAN.md` section 3.1g requires a dense sim cross-check for DILUTION specifically.
`build_dataset.sim_check` implements exactly that -- and only that. Every other sample is stored with
`sim_status='not_checked'`, `sim_ok=True` and `sim_gap=0.0`.

THE PROBLEM THIS AUDITS: `sim_ok=True` on an unchecked sample is a DEFAULT, not a measurement, and a
stored `sim_gap` of 0.0 is indistinguishable from a genuine zero. Measured on `dataset_v2_s0.npz`:
84.9 % of samples are unchecked, and `auxetic`, `tiling`, `basis`, `anchor` and `random_large` have
ZERO checked samples. Since the model is trained on solver labels, any disagreement there is a
CEILING on what any surrogate can achieve -- `evaluate_v2` prints SOLVER-vs-SIM as "the floor" for
exactly this reason, but only over the family it scores.

So this samples the UNCHECKED population, stratified by family, and compares the STORED label
against the independent sim (`_common.sim_bulk_C6` -> `physical_homog.virial_C`; NOT
`sim_region_C6`, which routes the sim back through the solver's own contraction and would be
self-verification).

DO NOT RE-RUN THE SOLVER HERE. The first version of this script called `build_dataset.sim_check`,
which rebuilds a `DesignProblem` from the geo and recomputes the label -- and `evaluate_v2.geo_of`
reconstructs a geo with PLACEHOLDER `tri_verts` and `centroids` (zeros). That is harmless for the
sim path, which does not read them, and fatal for the solver path, which does. The result was a
measurement of the RECONSTRUCTION, reported as 68.5 % of labels failing: a false alarm contradicted
by `evaluate_v2`'s own SOLVER-vs-SIM = 0.0000 on 250 bravais networks. Verified directly --
12 of 12 recomputations failed to reproduce their own stored label (rel 1e-1 to 5e-1).
The stored label IS the solver's output; comparing it to the sim is the whole measurement.

Run:
    python "Phase 5/verifications/m2_label_audit.py" --per_family 40
"""
import argparse
import json
import os
import sys
import time
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
M2DIR = os.path.join(REPO, 'Phase 5', 'm2')
sys.path.insert(0, os.path.join(REPO, 'Phase 3', 'verifications'))
import _common as C                                                       # noqa: E402
sys.path.insert(0, os.path.join(REPO, 'Phase 5'))
sys.path.insert(0, M2DIR)
import build_dataset as B                                                 # noqa: E402
import evaluate_v2 as EV                                                  # noqa: E402
import train_v2 as T                                                      # noqa: E402

warnings.simplefilter('ignore')
RESULTS = os.path.join(REPO, 'Phase 5', 'results', 'm2_s1')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', default=os.path.join(M2DIR, 'data', 'dataset_v2_s0.npz'))
    ap.add_argument('--per_family', type=int, default=40, help='unchecked samples drawn per family')
    ap.add_argument('--all', action='store_true',
                    help='audit EVERY unchecked sample and write a per-sample verdict mask, turning '
                         "the stored sim_ok=True default into a measurement")
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    raw = T.load(a.data)
    with np.load(a.data, allow_pickle=False) as d:
        status = np.array([str(x) for x in d['sim_status']])
    unchecked = [i for i, s in enumerate(status) if s == 'not_checked']
    byfam = {}
    for i in unchecked:
        byfam.setdefault(raw[i]['family'], []).append(i)

    rng = np.random.default_rng(a.seed)
    print('auditing %s: %d unchecked of %d\n' % (os.path.basename(a.data), len(unchecked), len(raw)))
    print('  %-14s %5s %8s %9s %9s %9s %7s' %
          ('family', 'n', 'median', 'p90', 'max', 'FAILED', 'err'))

    rows, t0 = {}, time.time()
    verdict = np.full(len(raw), np.nan)          # per-sample gap; NaN = not audited / sim errored
    for fam in sorted(byfam):
        idx = byfam[fam]
        pick = (np.arange(len(idx)) if a.all
                else rng.choice(len(idx), min(a.per_family, len(idx)), replace=False))
        gaps, nerr = [], 0
        for j in sorted(pick):
            g = raw[idx[j]]
            try:
                c6_sim = np.asarray(C.sim_bulk_C6(EV.geo_of(g)), float)
            except Exception:                                             # noqa: BLE001
                nerr += 1
                continue
            c6_stored = np.asarray(g['C6'], float)          # the solver's own output, as trained on
            denom = max(float(np.abs(c6_sim).max()), 1e-30)
            gap = float(np.abs(c6_stored - c6_sim).max() / denom)
            if not np.isfinite(gap):
                nerr += 1
                continue
            gaps.append(gap)
            verdict[idx[j]] = gap
        if not gaps:
            print('  %-14s %5d  (all %d attempts errored)' % (fam, 0, nerr))
            continue
        gaps = np.array(gaps)
        bad = int((gaps > B.SIM_GAP_TOL).sum())
        rows[fam] = dict(n=len(gaps), median=float(np.median(gaps)), p90=float(np.percentile(gaps, 90)),
                         max=float(gaps.max()), failed=bad, errors=nerr)
        print('  %-14s %5d %8.2e %9.2e %9.2e %6d (%3.0f%%) %7d'
              % (fam, len(gaps), np.median(gaps), np.percentile(gaps, 90), gaps.max(),
                 bad, 100 * bad / len(gaps), nerr))

    allg = np.concatenate([np.full(r['n'], r['median']) for r in rows.values()]) if rows else []
    tot_n = sum(r['n'] for r in rows.values())
    tot_bad = sum(r['failed'] for r in rows.values())
    print('\n  TOTAL %d audited, %d exceed the builder tolerance %.2f  (%.1f %%)   [%.0fs]'
          % (tot_n, tot_bad, B.SIM_GAP_TOL, 100 * tot_bad / max(tot_n, 1), time.time() - t0))
    print('  For reference the DILUTION samples, which ARE checked, fail at 2.53 %.')

    os.makedirs(RESULTS, exist_ok=True)
    if a.all:
        # The MASK is the deliverable: a per-sample, measured verdict replacing the sim_ok=True
        # default. Saved beside the dataset so a trainer can exclude bad labels without a rebuild.
        mp = a.data.replace('.npz', '_labelgap.npz')
        audited = np.isfinite(verdict)
        np.savez_compressed(mp, gap=verdict, audited=audited,
                            sim_gap_tol=np.array(B.SIM_GAP_TOL))
        print('  per-sample gaps -> %s   (%d audited, %d exceed tol)'
              % (mp, int(audited.sum()), int((verdict[audited] > B.SIM_GAP_TOL).sum())))
    out = os.path.join(RESULTS, 'label_audit_%s.json' % os.path.basename(a.data).replace('.npz', ''))
    with open(out, 'w', encoding='utf-8') as fh:
        json.dump(dict(data=os.path.basename(a.data), per_family=a.per_family, seed=a.seed,
                       sim_gap_tol=B.SIM_GAP_TOL, n_unchecked=len(unchecked), n_total=len(raw),
                       by_family=rows), fh, indent=2)
    print('  ->', out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
