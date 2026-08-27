r"""M2 S1 -- score v2 (scalar messages) against v3 (TENSOR messages) on the SAME holdout, and plot.

Provenance: MATERIALIZE `Phase 5/m2/M2_V2_PLAN.md` S1; head `C(s) = Q (M M^T) Q^T` of
`model_v2.py`; the v3 architecture change (tensor messages on triangle adjacency) is documented in
`model_v3.py`.  Figures go through the root `plotting.py`, per CLAUDE.md section 3.

TWO BASELINES, and the distinction is the whole point of the exercise:

  * `global`     -- predict the TRAIN-split mean C6 for every triangle.  Trivially available.
  * `own bulk`   -- predict each network's OWN bulk mean `C_eff` for every one of its triangles.
                    This is an ORACLE: it is handed the answer's average and asked only for the
                    fluctuation.  A model that merely matches it has learned the bulk response and
                    NOTHING about the per-triangle field -- which is exactly what v2 did (0.509 vs
                    0.5144).  Beating `own bulk` is therefore the bar for "the network learned the
                    mechanics", not beating `global`.

Compute and render are separated: scoring writes `v3_report.npz`, plotting reads it (`--plots_only`).

Run:
  python "Phase 5/verifications/m2_v3_report.py"
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
import model_v2 as M2                                                       # noqa: E402
import model_v3 as M3                                                       # noqa: E402
import train_v2 as T2                                                       # noqa: E402
import train_v3 as T3                                                       # noqa: E402
import plotting as P                                                        # noqa: E402

torch.set_default_dtype(torch.float64)
RESULTS = os.path.join(REPO, 'Phase 5', 'results', 'm2_s1')
COMP = ['C_xxxx', 'C_xxxy', 'C_xxyy', 'C_xyxy', 'C_xyyy', 'C_yyyy']


def _split(data, holdout):
    raw = [g for g in T2.load(data) if g['sim_ok'] and g['size_bin'] != 'large_holdout']
    return ([g for g in raw if g['family'] != holdout],
            [g for g in raw if g['family'] == holdout])


def _score_v2(ck_path, va):
    ck = torch.load(ck_path, weights_only=False)
    angles = ck['n_node_feat'] == M2.N_NODE_FEAT
    net = M2.ForwardGNNv2(hidden=ck['hidden'], n_layers=ck['layers'],
                          n_node_feat=ck['n_node_feat'], n_tri_feat=ck['n_tri_feat'])
    net.load_state_dict(ck['state'])
    net.eval()
    pred = []
    with torch.no_grad():
        for g in va:
            pred.append(T2.predict(net, T2.prepare(g, angles=angles))[0].numpy())
    return pred, sum(p.numel() for p in net.parameters())


def _score_v3(ck_path, va):
    ck = torch.load(ck_path, weights_only=False)
    net = M3.ForwardGNNv3(ns=ck['ns'], nt=ck['nt'], hidden=ck['hidden'], n_layers=ck['layers'])
    net.load_state_dict(ck['state'])
    net.eval()
    pred = []
    with torch.no_grad():
        for g in va:
            pred.append(T3.predict(net, T3.prepare(g)).numpy())
    return pred, sum(p.numel() for p in net.parameters())


def _metrics(pred, targ, sd):
    """Per-component MAE (per-triangle and bulk) plus the SPD violation rate."""
    per = np.stack([np.abs(p - t).mean(0) for p, t in zip(pred, targ)]).mean(0)
    bulk = np.stack([np.abs(p.mean(0) - t.mean(0)) for p, t in zip(pred, targ)]).mean(0)
    eig = np.linalg.eigvalsh(M2.c6_to_sym3(torch.as_tensor(np.concatenate(pred))).numpy())
    return dict(per=per, bulk=bulk, per_n=float((per / sd).mean()),
                bulk_n=float((bulk / sd).mean()),
                spd=float((eig.min(-1) < -1e-10).mean()))


def compute(a):
    tr, va = _split(a.data, a.holdout)
    targ = [np.asarray(g['C6_per'], float) for g in va]
    tr_all = np.concatenate([np.asarray(g['C6_per'], float) for g in tr])
    sd = tr_all.std(0).clip(1e-12)
    gmean = tr_all.mean(0)
    print('holdout %r: train %d / val %d networks, %d val triangles'
          % (a.holdout, len(tr), len(va), sum(len(t) for t in targ)))

    out = {}
    out['global'] = _metrics([np.tile(gmean, (len(t), 1)) for t in targ], targ, sd)
    out['own bulk'] = _metrics([np.tile(t.mean(0), (len(t), 1)) for t in targ], targ, sd)
    # A TRAIN-split score on an equal-sized random subsample. This is the diagnostic that
    # separates the two ways a model can land on the own-bulk oracle: if it does not beat the
    # oracle even on data it was FITTED to, the shortfall is capacity/optimisation; if it beats it
    # on train and not on val, the shortfall is generalisation. Without this the plateau is
    # ambiguous, and the ambiguity is what cost the previous round of v2 runs.
    sub = np.random.default_rng(0).choice(len(tr), min(len(va), len(tr)), replace=False)
    trs = [tr[i] for i in sorted(sub)]
    tr_targ = [np.asarray(g['C6_per'], float) for g in trs]
    out['own bulk (train)'] = _metrics([np.tile(t.mean(0), (len(t), 1)) for t in tr_targ],
                                       tr_targ, sd)

    for nm, ck, fn in [('v2 scalar-msg', a.ck_v2, _score_v2), ('v3 tensor-msg', a.ck_v3, _score_v3)]:
        if not os.path.exists(ck):
            print('  SKIP %s: no checkpoint at %s' % (nm, ck))
            continue
        pred, npar = fn(ck, va)
        out[nm] = _metrics(pred, targ, sd)
        out[nm]['params'] = npar
        out['%s (train)' % nm] = _metrics(fn(ck, trs)[0], tr_targ, sd)
        if nm.startswith('v3'):
            np.savez_compressed(os.path.join(RESULTS, 'v3_scatter.npz'),
                                pred=np.concatenate(pred), targ=np.concatenate(targ))

    print('\n  %-14s %10s %10s %8s %9s' % ('model', 'per-tri', 'bulk', 'SPD', 'params'))
    for nm, m in out.items():
        print('  %-14s %10.4f %10.4f %8.5f %9s'
              % (nm, m['per_n'], m['bulk_n'], m['spd'], m.get('params', '-')))

    np.savez_compressed(os.path.join(RESULTS, 'v3_report.npz'), label_std=sd,
                        data=np.array(os.path.basename(a.data)),
                        **{'%s|%s' % (nm, k): np.asarray(v)
                           for nm, m in out.items() for k, v in m.items()})
    with open(os.path.join(RESULTS, 'v3_report.json'), 'w', encoding='utf-8') as fh:
        json.dump(dict(holdout=a.holdout, n_train=len(tr), n_val=len(va),
                       n_val_tri=int(sum(len(t) for t in targ)),
                       label_std=[float(x) for x in sd],
                       models={nm: {k: (float(v) if np.ndim(v) == 0 else [float(x) for x in v])
                                    for k, v in m.items()} for nm, m in out.items()}), fh, indent=2)
    return out


def plots(a):
    import matplotlib.pyplot as plt
    d = np.load(os.path.join(RESULTS, 'v3_report.npz'))
    sd = d['label_std']
    models = sorted({k.split('|')[0] for k in d.files if '|' in k
                     and '(train)' not in k})
    base = float(d['own bulk|per_n'])

    # (1) learning curves, v2 vs v3, with BOTH baselines drawn as lines
    runs = {}
    for nm, f in [('v2 scalar-msg', 'run_bravais.json'), ('v3 tensor-msg', 'run_v3_bravais.json')]:
        p = os.path.join(RESULTS, f)
        if not os.path.exists(p):
            continue
        h = json.load(open(p, encoding='utf-8'))['history']
        # the two trainers name the same quantity differently ('val_per_tri' vs 'val'); both are the
        # per-triangle MAE / label sigma on the holdout, so plot whichever key is present
        key = 'val_per_tri' if 'val_per_tri' in h[0] else 'val'
        runs[nm] = ([r['epoch'] for r in h], [r[key] for r in h])
    if runs:
        P.save_fig(P.plot_learning_curves(
            runs, baselines={'own-bulk oracle': base, 'global mean': float(d['global|per_n'])},
            title='M2 S1 -- leave-one-family-out (%s), %s'
                  % (a.holdout, str(d['data']) if 'data' in d.files else '?')),
            os.path.join(RESULTS, 'v3_learning_curves.png'))

    # (2) per-component MAE/std, models side by side
    def bars(ax):
        w, x = 0.8 / len(models), np.arange(6)
        for i, nm in enumerate(models):
            ax.bar(x + i * w - 0.4 + w / 2, d['%s|per' % nm] / sd, w, label=nm)
        ax.set_xticks(x)
        ax.set_xticklabels(COMP, rotation=30, fontsize=8)
        ax.set_ylabel('per-triangle MAE / label sigma')
        ax.grid(axis='y', alpha=0.25)
        ax.set_axisbelow(True)
        ax.legend(fontsize=8)
    P.save_element(bars, os.path.join(RESULTS, 'v3_components.png'), figsize=(7.0, 4.2))

    # (3) predicted vs true, the two components that matter: the axial and the SHEAR channel
    # (C_xyxy is where the A-0 contraction defect lived, and where W shows up most strongly)
    sp = os.path.join(RESULTS, 'v3_scatter.npz')
    if os.path.exists(sp):
        s = np.load(sp)
        fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.4))
        for ax, j in zip(axes, [0, 3]):
            t, p = s['targ'][:, j], s['pred'][:, j]
            lo, hi = np.percentile(np.concatenate([t, p]), [0.5, 99.5])
            ax.hexbin(t, p, gridsize=60, bins='log', extent=(lo, hi, lo, hi), cmap='viridis')
            ax.plot([lo, hi], [lo, hi], 'k--', lw=1.4)
            ax.set_xlabel('true %s' % COMP[j])
            ax.set_ylabel('predicted')
            ax.set_title('%s   (r = %.3f)' % (COMP[j], np.corrcoef(t, p)[0, 1]))
            ax.set_box_aspect(1)
        fig.suptitle('v3 per-triangle prediction, held-out family %r' % a.holdout, fontsize=12)
        fig.tight_layout()
        P.save_fig(fig, os.path.join(RESULTS, 'v3_scatter.png'))
    print('figures written to', RESULTS)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', default=os.path.join(M2DIR, 'data', 'dataset.npz'))
    ap.add_argument('--holdout', default='bravais')
    ap.add_argument('--ck_v2', default=os.path.join(M2DIR, 'checkpoint_v2_bravais.pt'))
    ap.add_argument('--ck_v3', default=os.path.join(M2DIR, 'checkpoint_v3_bravais.pt'))
    ap.add_argument('--plots_only', action='store_true')
    a = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)
    if not a.plots_only:
        compute(a)
    plots(a)
    return 0


if __name__ == '__main__':
    sys.exit(main())
