r"""M2 -- score a TRAINED surrogate on FRESHLY GENERATED data it has never seen.

Provenance: MATERIALIZE `Phase 5/m2/M2_V2_PLAN.md` S1; model `Phase 5/m2/model_v3.py`;
generator `Phase 5/m2/build_dataset.py`.

WHY THIS EXISTS (user, 2026-09-11).  A held-out split has to be chosen BEFORE training, so every
question we failed to ask up front seemed to cost a fresh multi-day run.  It does not: the model is
already trained and frozen, so **any data it has never seen is a valid holdout**, and generating
that data costs a dataset build (0.155 s/sample) rather than a training run (~2.4 days).

What that buys over re-splitting and retraining:

  * **the hard families, at last.**  Every number so far came from the `bravais` family holdout --
    ordered geometry, median |W| 2.00 -- while `random` (median |W| 14.03) and `disordered` were
    ENTIRELY trained on.  A fresh build with a new seed produces new point clouds, hence new MESHES,
    in exactly those families.
  * **matched distributions**, which no previous split had.  The fresh samples come from the SAME
    generators with a different seed, so train-vs-fresh is finally a like-for-like comparison and
    the over/underfit question (train ~ fresh => underfit; train << fresh => overfit) becomes
    answerable.  The `bravais` holdout could never answer it: it is measurably EASIER than the
    training set, which alone explained the train > val reading.

NOTE ON NAMING (the dataset is misleading here).  `topology_id` does NOT mean topology in this
project's sense (connectivity).  It is the seed record's name -- a specific (positions,
connectivity) pair, i.e. a **MESH** -- and `geom_variant` re-Delaunays (`build_dataset.py:488`), so
a variant is a different connectivity too.  Nothing in this dataset identifies connectivity alone.
`train_v2.load` surfaces it as `mesh_id`.

The normaliser is the ORIGINAL training split's label sigma, so every number here is directly
comparable with the trainer's and with `m2_error_strata.py` -- a fresh sigma would silently rescale
the metric.

Run:
  python "Phase 5/verifications/m2_fresh_holdout.py" \
      --ckpt checkpoint_v3_res_bravais_w10_h1_L5_ns48_h64_e400_star_ms.pt \
      --fresh "Phase 5/m2/data/dataset_fresh_s1234.npz"
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
M2DIR = os.path.join(REPO, 'Phase 5', 'm2')
sys.path.insert(0, M2DIR)
sys.path.insert(0, REPO)
# Phase 5 import preamble (CLAUDE.md section 3): `_common` wires sys.path + the solver stack first.
sys.path.insert(0, os.path.join(REPO, 'Phase 3', 'verifications'))
import _common as _C                                                        # noqa: E402,F401
import model_v3 as M3                                                       # noqa: E402
import train_v2 as T2                                                       # noqa: E402
import train_v3 as T3                                                       # noqa: E402
from inverse_design import c6_to_nuE                                        # noqa: E402

torch.set_default_dtype(torch.float64)
RESULTS = os.path.join(REPO, 'Phase 5', 'results', 'm2_s1')
W_EDGES = [0.0, 1e-9, 1.0, 3.0, 10.0, np.inf]


def _usable(gs):
    return [g for g in gs if g['sim_ok'] and g['size_bin'] != 'large_holdout']


def _score(net, gs, sd, n, seed=0):
    """Per-sample per-triangle MAE/sigma and |dnu| against the stored solver label."""
    sel = np.random.default_rng(seed).choice(len(gs), min(n, len(gs)), replace=False)
    rows = []
    with torch.no_grad():
        for i in sel:
            g = gs[i]
            p = T3.predict(net, T3.prepare(g)).numpy()
            t = np.asarray(g['C6_per'], float)
            nu_m, E_m = c6_to_nuE(torch.as_tensor(p.mean(0)))
            nu_s, E_s = c6_to_nuE(torch.as_tensor(np.asarray(g['C6'], float)))
            rows.append(dict(mae=float((np.abs(p - t) / sd).mean()),
                             d_nu=float(abs(float(nu_m) - float(nu_s))),
                             rel_E=float(abs(float(E_m) - float(E_s)) / max(abs(float(E_s)), 1e-30)),
                             w_max=float(g['w_max']), family=g['family'], mesh=g['mesh_id']))
    return rows


def _agg(rows):
    return dict(n=len(rows),
                mae=float(np.mean([r['mae'] for r in rows])),
                mae_nu=float(np.mean([r['d_nu'] for r in rows])),
                median_nu=float(np.median([r['d_nu'] for r in rows])),
                rel_E=float(np.mean([r['rel_E'] for r in rows])))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', default=os.path.join(M2DIR, 'data', 'dataset_v2_s0.npz'),
                    help='the ORIGINAL dataset -- supplies the training samples and the sigma')
    ap.add_argument('--fresh', default=os.path.join(M2DIR, 'data', 'dataset_fresh_s1234.npz'))
    ap.add_argument('--ckpt', default='checkpoint_v3_res_bravais_w10_h1_L5_ns48_h64_e400_star_ms.pt')
    ap.add_argument('--holdout', default='bravais', help='the family the checkpoint was trained AGAINST')
    ap.add_argument('--w_max_cut', type=float, default=10.0,
                    help='the training filter of the run being scored; the matched-distribution '
                         'comparison is taken on this subset, since the filter applied to TRAINING '
                         'ONLY and an unfiltered holdout is therefore a different distribution')
    ap.add_argument('--n', type=int, default=600, help='samples scored per group')
    a = ap.parse_args()

    orig = _usable(T2.load(a.data))
    fresh = _usable(T2.load(a.fresh))
    tr = [g for g in orig if g['family'] != a.holdout]          # what the checkpoint trained on
    sd = np.concatenate([np.asarray(g['C6_per'], float) for g in tr]).std(0).clip(1e-12)

    # --- GATE: the fresh meshes must actually be new, or this measures nothing ------------------
    tr_mesh = {g['mesh_id'] for g in tr}
    fr_mesh = {g['mesh_id'] for g in fresh}
    shared = tr_mesh & fr_mesh
    print('original %d samples (%d trained on, %d meshes) | fresh %d samples (%d meshes)'
          % (len(orig), len(tr), len(tr_mesh), len(fresh), len(fr_mesh)))
    print('  shared meshes: %d of %d fresh (%.1f %%)'
          % (len(shared), len(fr_mesh), 100 * len(shared) / max(len(fr_mesh), 1)))
    fresh_new = [g for g in fresh if g['mesh_id'] not in tr_mesh]
    print('  -> scoring %d fresh samples on UNSEEN meshes' % len(fresh_new))
    if not fresh_new:
        raise SystemExit('every fresh mesh was already in training -- rebuild with a different seed. '
                         '(Parametric families such as `bravais` regenerate identically; the seeded '
                         'point-cloud families are the ones that produce new meshes.)')

    net = M3.from_checkpoint(torch.load(os.path.join(M2DIR, a.ckpt), weights_only=False))
    cut = a.w_max_cut

    groups = {
        # matched pair: both sides filtered exactly as TRAINING was, so the comparison is like-for-like
        'TRAIN (fitted, |W|<=%g)' % cut: [g for g in tr if float(g['w_max']) <= cut],
        'FRESH, unseen meshes (|W|<=%g)' % cut: [g for g in fresh_new if float(g['w_max']) <= cut],
        'FRESH, unseen meshes (all |W|)': fresh_new,
    }
    out = {'ckpt': a.ckpt, 'fresh': os.path.basename(a.fresh), 'n_fresh_new': len(fresh_new),
           'shared_meshes': len(shared), 'groups': {}, 'by_family': {}, 'by_w': {}}
    print('\n  %-34s %6s %10s %9s %9s' % ('group', 'n', 'MAE/sigma', 'MAE(nu)', 'med(nu)'))
    scored = {}
    for nm, gs in groups.items():
        if not gs:
            continue
        rows = _score(net, gs, sd, a.n)
        scored[nm] = rows
        m = _agg(rows)
        out['groups'][nm] = m
        print('  %-34s %6d %10.4f %9.4f %9.4f' % (nm, m['n'], m['mae'], m['mae_nu'], m['median_nu']))

    key = 'FRESH, unseen meshes (all |W|)'
    rows = scored[key]
    print('\n  fresh, BY FAMILY:')
    print('  %-14s %6s %10s %9s' % ('family', 'n', 'MAE/sigma', 'MAE(nu)'))
    for f in sorted({r['family'] for r in rows}):
        sel = [r for r in rows if r['family'] == f]
        m = _agg(sel)
        out['by_family'][f] = m
        print('  %-14s %6d %10.4f %9.4f' % (f, m['n'], m['mae'], m['mae_nu']))

    print('\n  fresh, BY max|W| (the training filter sat at %g):' % cut)
    print('  %-14s %6s %10s %9s' % ('max|W|', 'n', 'MAE/sigma', 'MAE(nu)'))
    for i in range(len(W_EDGES) - 1):
        sel = [r for r in rows if W_EDGES[i] <= r['w_max'] < W_EDGES[i + 1]]
        if not sel:
            continue
        m = _agg(sel)
        lab = '%g-%g' % (W_EDGES[i], W_EDGES[i + 1])
        out['by_w'][lab] = m
        print('  %-14s %6d %10.4f %9.4f' % (lab, m['n'], m['mae'], m['mae_nu']))

    tr_key = 'TRAIN (fitted, |W|<=%g)' % cut
    fr_key = 'FRESH, unseen meshes (|W|<=%g)' % cut
    if tr_key in out['groups'] and fr_key in out['groups']:
        t_, f_ = out['groups'][tr_key]['mae'], out['groups'][fr_key]['mae']
        out['gap_pct'] = float(100 * (f_ - t_) / max(t_, 1e-30))
        print('\n  MATCHED COMPARISON (both sides |W| <= %g, same sigma, same generators):' % cut)
        print('    train %.4f  vs  fresh-unseen-mesh %.4f   -> fresh is %+.1f %% ' % (t_, f_, out['gap_pct']))
        print('    train ~ fresh => UNDERFIT (capacity/optimisation);  '
              'train << fresh => OVERFIT (data/regularisation)')

    os.makedirs(RESULTS, exist_ok=True)
    stem = os.path.splitext(a.ckpt)[0].replace('checkpoint_', '')
    p = os.path.join(RESULTS, 'fresh_%s.json' % stem)
    with open(p, 'w', encoding='utf-8') as fh:
        json.dump(out, fh, indent=2)
    print('  ->', p)
    return 0


if __name__ == '__main__':
    sys.exit(main())
