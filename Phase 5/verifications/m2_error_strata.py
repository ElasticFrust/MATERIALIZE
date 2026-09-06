r"""M2 S1 -- WHERE the surrogate's error lives: stratify per-triangle MAE by k-CONTRAST and by |W|.

Provenance: MATERIALIZE `Phase 5/m2/M2_V2_PLAN.md` S1; model `Phase 5/m2/model_v3.py`.

WHY THIS EXISTS AS A SCRIPT.  The headline per-triangle MAE/sigma is an average over a holdout whose
difficulty varies by more than an order of magnitude, so it can move for reasons that have nothing to
do with the change being tested.  Two stratifications carry the diagnosis:

  * **k-CONTRAST** (`max k / min k` per network) -- measured monotone and 17x from end to end on the
    free head.  `THEORY_NOTES.md` independently measured the required cluster radius as ~1 for
    geometric disorder but 4-6 at 100x contrast, i.e. the PHYSICS has a contrast-dependent screening
    length while a fixed-depth GNN has one fixed reach.  This is the curve any depth or coupling
    change has to move.
  * **max|W|** -- the training-domain split.  `--w_max_cut 10` DROPS near-mechanism networks from
    training and leaves the holdout unfiltered, so part of the headline is pure extrapolation onto
    networks the model was never shown.  Reporting the two sides separately is the difference
    between "the model is bad" and "the model is being scored outside its domain".

These tables were computed ad hoc for the free-head and residual-head runs and existed only in a
terminal; every number cited in `Phase 5/results/m2_s1/M2_RESIDUAL_AND_CONSTRAINTS.md` now comes from
here instead.  Compute and render are separated: scoring writes the JSON, `--plots_only` reads it.

Run:
  python "Phase 5/verifications/m2_error_strata.py" \
      --ckpt checkpoint_v3_res_bravais_w10_h1_L5_ns48_h64_e400_star.pt
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
# Phase 5 import preamble (CLAUDE.md section 3): `_common` wires the rest of sys.path and the
# solver stack, and must be imported BEFORE anything from `inverse_design`.
sys.path.insert(0, os.path.join(REPO, 'Phase 3', 'verifications'))
import _common as _C                                                         # noqa: E402,F401
import model_v3 as M3                                                        # noqa: E402
import train_v2 as T2                                                        # noqa: E402
import train_v3 as T3                                                        # noqa: E402
import plotting as P                                                         # noqa: E402
from inverse_design import c6_to_nuE                                         # noqa: E402

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)
RESULTS = os.path.join(REPO, 'Phase 5', 'results', 'm2_s1')

#: contrast bin edges. Decades above 10, because that is how the quantity is distributed: uniform-k
#: and mildly disordered networks sit at 1-3 while the designed/dilute ones reach 1e6.
CONTRAST_EDGES = [0.0, 3.0, 10.0, 1e2, 1e4, 1e9]
#: |W| bin edges. 10 is not arbitrary -- it is `--w_max_cut`, the TRAINING DOMAIN boundary.
W_EDGES = [0.0, 1e-9, 1.0, 3.0, 10.0, np.inf]


def _bin(x, edges):
    """Index of the bin containing `x` (right-open), clipped into range."""
    return int(np.clip(np.searchsorted(edges, x, side='right') - 1, 0, len(edges) - 2))


def _label(edges, i):
    lo, hi = edges[i], edges[i + 1]
    fmt = (lambda v: '%g' % v if v < 1e3 else '%.0e' % v)
    return '%s-%s' % (fmt(lo), fmt(hi))


def compute(a):
    raw = [g for g in T2.load(a.data)
           if g['sim_ok'] and g['size_bin'] != 'large_holdout' and g['family'] == a.holdout]
    if not raw:
        raise SystemExit('no holdout networks for family %r in %s' % (a.holdout, a.data))
    # The normaliser is the TRAIN-split label sigma, exactly as the trainer and `m2_v3_report` use,
    # so the numbers here are comparable with the headline rather than a second, private scale.
    tr = [g for g in T2.load(a.data)
          if g['sim_ok'] and g['size_bin'] != 'large_holdout' and g['family'] != a.holdout]
    sd = np.concatenate([np.asarray(g['C6_per'], float) for g in tr]).std(0).clip(1e-12)

    net = M3.from_checkpoint(torch.load(os.path.join(M2DIR, a.ckpt), weights_only=False))
    print('checkpoint %s  ->  %d holdout networks (family %r)' % (a.ckpt, len(raw), a.holdout))

    rows = []
    with torch.no_grad():
        for i, g in enumerate(raw):
            t = T3.prepare(g)
            pred = T3.predict(net, t).numpy()
            targ = np.asarray(g['C6_per'], float)
            k = np.asarray(g['k'], float)
            # nu AND the per-triangle tensor error, because they can move in opposite directions:
            # the star channel cut the tensor error 41 % while MAE(nu) vs the sim went from 0.0550
            # to 0.0572. nu here is model-vs-SOLVER-LABEL, not vs the sim -- justified ONLY because
            # `evaluate_v2` measured SOLVER vs SIM at 3.6e-08 on this same family, i.e. the label
            # and the sim agree to eight decimals here. On any family where that floor is not
            # negligible this substitution is invalid and the sim must be run.
            nu_m, E_m = c6_to_nuE(torch.as_tensor(pred.mean(0)))
            nu_s, E_s = c6_to_nuE(torch.as_tensor(np.asarray(g['C6'], float)))
            rows.append(dict(mae=float((np.abs(pred - targ) / sd).mean()),
                             d_nu=float(abs(float(nu_m) - float(nu_s))),
                             rel_E=float(abs(float(E_m) - float(E_s))
                                         / max(abs(float(E_s)), 1e-30)),
                             contrast=float(k.max() / max(k.min(), 1e-300)),
                             w_max=float(g['w_max']), n_tri=len(targ),
                             k_pattern=g.get('k_pattern', '?')))
            if (i + 1) % 200 == 0:
                print('   %d/%d' % (i + 1, len(raw)))

    def agg(sel):
        return dict(n=len(sel),
                    mae=float(np.mean([r['mae'] for r in sel])),
                    mae_nu=float(np.mean([r['d_nu'] for r in sel])),
                    median_nu=float(np.median([r['d_nu'] for r in sel])),
                    rel_E=float(np.mean([r['rel_E'] for r in sel])),
                    median_w=float(np.median([r['w_max'] for r in sel])),
                    median_contrast=float(np.median([r['contrast'] for r in sel])))

    out = dict(ckpt=a.ckpt, data=os.path.basename(a.data), holdout=a.holdout, n=len(rows),
               overall=float(np.mean([r['mae'] for r in rows])),
               overall_nu=float(np.mean([r['d_nu'] for r in rows])), strata={})
    for name, key, edges in [('contrast', 'contrast', CONTRAST_EDGES), ('w_max', 'w_max', W_EDGES)]:
        st = []
        for b in range(len(edges) - 1):
            sel = [r for r in rows if _bin(r[key], edges) == b]
            if sel:
                st.append(dict(bin=_label(edges, b), **agg(sel)))
        out['strata'][name] = st

    print('\n  overall per-triangle MAE/sigma = %.4f   MAE(nu) = %.4f   (n = %d)'
          % (out['overall'], out['overall_nu'], out['n']))
    for name, st in out['strata'].items():
        print('\n  %-14s %6s %10s %9s %9s %11s'
              % (name, 'n', 'MAE/sigma', 'MAE(nu)', 'med(nu)', 'median |W|'))
        for r in st:
            print('  %-14s %6d %10.4f %9.4f %9.4f %11.2f'
                  % (r['bin'], r['n'], r['mae'], r['mae_nu'], r['median_nu'], r['median_w']))

    # The training-domain split, stated on its own because it is the whole reading of the headline:
    # `--w_max_cut` removes these networks from TRAINING and the holdout deliberately keeps them,
    # so part of every reported average is pure extrapolation.
    ins = [r for r in rows if r['w_max'] <= a.w_max_cut]
    outs = [r for r in rows if r['w_max'] > a.w_max_cut]
    out['in_domain'] = agg(ins) if ins else None
    out['out_domain'] = agg(outs) if outs else None
    if ins and outs:
        for key, tot in (('mae', out['overall']), ('mae_nu', out['overall_nu'])):
            out['out_domain']['share_of_' + key] = float(
                len(outs) * out['out_domain'][key] / (len(rows) * tot))
        print('\n  TRAINING DOMAIN (max|W| <= %g): %d networks   MAE/sigma %.4f   MAE(nu) %.4f'
              % (a.w_max_cut, len(ins), out['in_domain']['mae'], out['in_domain']['mae_nu']))
        print('  OUTSIDE it (never trained on): %d networks   MAE/sigma %.4f   MAE(nu) %.4f'
              % (len(outs), out['out_domain']['mae'], out['out_domain']['mae_nu']))
        print('  those %d carry %.0f %% of the total per-triangle error and %.0f %% of MAE(nu)'
              % (len(outs), 100 * out['out_domain']['share_of_mae'],
                 100 * out['out_domain']['share_of_mae_nu']))

    os.makedirs(RESULTS, exist_ok=True)
    stem = os.path.splitext(a.ckpt)[0].replace('checkpoint_', '')
    path = os.path.join(RESULTS, 'strata_%s.json' % stem)
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(out, fh, indent=2)
    print('  ->', path)
    return path


def plots(paths):
    """One grouped-bar panel per stratification, all checkpoints side by side."""
    loaded = [json.load(open(p, encoding='utf-8')) for p in paths]
    names = [os.path.basename(p)[len('strata_'):-len('.json')] for p in paths]
    for strat, xlabel in [('contrast', 'k-contrast  (max k / min k)'),
                          ('w_max', 'max |W|  (training domain: <= 10)')]:
        bins = list(dict.fromkeys(r['bin'] for d in loaded for r in d['strata'].get(strat, [])))

        def draw(ax, strat=strat, bins=bins, xlabel=xlabel):
            w, x = 0.8 / max(len(loaded), 1), np.arange(len(bins))
            for i, (d, nm) in enumerate(zip(loaded, names)):
                by = {r['bin']: r['mae'] for r in d['strata'].get(strat, [])}
                ax.bar(x + i * w - 0.4 + w / 2, [by.get(b, np.nan) for b in bins], w, label=nm)
            ax.set_xticks(x)
            ax.set_xticklabels(bins, rotation=20, fontsize=8)
            ax.set_xlabel(xlabel)
            ax.set_ylabel('per-triangle MAE / label sigma')
            ax.grid(axis='y', alpha=0.25)
            ax.set_axisbelow(True)
            ax.legend(fontsize=7)

        P.save_element(draw, os.path.join(RESULTS, 'strata_%s.png' % strat), figsize=(7.0, 4.2))
    print('figures written to', RESULTS)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', default=os.path.join(M2DIR, 'data', 'dataset_v2_s0.npz'))
    ap.add_argument('--ckpt', default='checkpoint_v3_res_bravais_w10_h1_L5_ns48_h64_e400_star.pt')
    ap.add_argument('--holdout', default='bravais')
    ap.add_argument('--w_max_cut', type=float, default=10.0,
                    help='the TRAINING filter used by the run being scored; the in/out-of-domain '
                         'split is reported against it, so it must match the run')
    ap.add_argument('--plots_only', nargs='*', default=None,
                    help='skip scoring and plot these strata_*.json files instead')
    a = ap.parse_args()
    if a.plots_only is not None:
        plots(a.plots_only or [os.path.join(RESULTS, f) for f in sorted(os.listdir(RESULTS))
                               if f.startswith('strata_') and f.endswith('.json')])
        return 0
    plots([compute(a)])
    return 0


if __name__ == '__main__':
    sys.exit(main())
