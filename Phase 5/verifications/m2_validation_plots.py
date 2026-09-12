r"""M2 S1 -- the figure set: the architecture ladder, and validation against SOLVER and SIM.

Provenance: MATERIALIZE `Phase 5/m2/M2_V2_PLAN.md` S1; model `Phase 5/m2/model_v3.py`.
All rendering goes through the root `plotting.py` (CLAUDE.md section 3); this file is pure
load-and-arrange, it computes nothing and re-runs no model.

WHAT IT DRAWS, and why each exists

  1. `m2_learning_curves.png` -- the four-step ARCHITECTURE LADDER on one axis, with the own-bulk
     oracle and the global mean drawn as LINES. The oracle is the bar that matters: it is handed
     each network's own bulk `C_eff` and asked only for the fluctuation, so a model that merely
     matches it has learned the bulk response and nothing about the per-triangle field -- which is
     exactly what v2 did.

  2/3. `m2_parity_nu.png`, `m2_parity_E.png` -- predicted vs the INDEPENDENT SIM, two rows:
     the `bravais` holdout (the family the run was trained against) and FRESH UNSEEN MESHES.
     Each row carries the same panel twice: once with the SOLVER's own scatter against the sim
     overlaid (the floor -- no model trained on solver labels can beat it), once coloured by
     `max|W|`. Without the floor in the same panel there is no way to tell the model's residual
     from the two code paths' own disagreement.

  4. `m2_by_family.png`, 5. `m2_by_wmax.png` -- the same numbers STRATIFIED, with the kill line and
     the must-tier drawn in. The headline average hides more than an order of magnitude: `cells`
     and `random` have near-identical per-triangle error and a 5.7x difference in MAE(nu), because
     nu is a contraction of the MEAN of C(s) and averages over triangle count.

Run:
  python "Phase 5/verifications/m2_validation_plots.py"
"""
import argparse
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
RESULTS = os.path.join(REPO, 'Phase 5', 'results', 'm2_s1')
sys.path.insert(0, REPO)
import plotting as P                                                        # noqa: E402
import matplotlib.pyplot as plt                                             # noqa: E402

#: the four-step ladder. `(label, source)`; a `.json` carries `history`, a `.log` is parsed --
#: the RESIDUAL-head run has no json because the machine restarted before it could write one.
LADDER = [
    ('v3 tensor msgs, free head', 'run_v3_bravais_w10_h1_L5_ns48_h64_e400.json'),
    ('+ residual head', 'v3_residual_bravais.log'),
    ('+ C_curv (star)', 'run_v3_res_bravais_w10_h1_L5_ns48_h64_e400_star.json'),
    ('+ M_S (area)', 'run_v3_res_bravais_w10_h1_L5_ns48_h64_e400_star_ms.json'),
]
#: baselines on the CURRENT dataset's bravais split (`v3_report.json`, recomputed 2026-09-11)
OWN_BULK, GLOBAL_MEAN = 0.4740, 0.9866
KILL_NU, MUST_NU, MUST_E = 0.05, 0.02, 5.0          # section 1 tiers (MAE(E)/E in per cent)
_EP = re.compile(r'^\s*ep\s+(\d+)\s+train\s+\S+\s+val MAE/std\s+(\S+)')


def _curve(src):
    """(epochs, val MAE/sigma) from either a run json's history or a training log."""
    p = os.path.join(RESULTS, src)
    if not os.path.exists(p):
        return None
    if src.endswith('.json'):
        h = json.load(open(p, encoding='utf-8'))['history']
        return [r['epoch'] for r in h], [r['val'] for r in h]
    xs, ys = [], []
    for line in open(p, encoding='utf-8', errors='replace'):
        m = _EP.match(line)
        if m:
            xs.append(int(m.group(1))); ys.append(float(m.group(2)))
    return (xs, ys) if xs else None


def _ev(name):
    p = os.path.join(RESULTS, name)
    return json.load(open(p, encoding='utf-8')) if os.path.exists(p) else None


def _stats(d):
    """MAE(nu) and MAE(E)/E for model-vs-sim and solver-vs-sim, from a saved eval json."""
    nm, ns, np_ = (np.asarray(d[k], float) for k in ('nu_model', 'nu_solver', 'nu_sim'))
    em, es, ep = (np.asarray(d[k], float) for k in ('E_model', 'E_solver', 'E_sim'))
    eps = float(d['eps_E'])
    return dict(n=len(nm),
                mae_nu=float(np.mean(np.abs(nm - np_))),
                mae_nu_solver=float(np.mean(np.abs(ns - np_))),
                relE=float(100 * np.mean(np.abs(em - ep) / (np.abs(ep) + eps))),
                relE_solver=float(100 * np.mean(np.abs(es - ep) / (np.abs(ep) + eps))))


def fig_learning(out):
    runs = {}
    for lab, src in LADDER:
        c = _curve(src)
        if c:
            runs[lab] = c
        else:
            print('  SKIP curve %r (%s not found)' % (lab, src))
    if not runs:
        return
    fig = P.plot_learning_curves(
        runs, baselines={'own-bulk oracle (the real bar)': OWN_BULK,
                         'global mean': GLOBAL_MEAN},
        title='M2 S1 -- architecture ladder, leave-one-family-out (bravais)',
        figsize=(7.4, 5.0))
    P.save_fig(fig, out)
    print('  ->', os.path.basename(out))


def fig_parity(kind, rows, out):
    """kind = 'nu' or 'E'; rows = [(row label, eval json dict), ...]."""
    mk, sk, pk = ('%s_model' % kind, '%s_solver' % kind, '%s_sim' % kind)
    sym = r'$\nu$' if kind == 'nu' else '$E$'
    fig, axes = plt.subplots(len(rows), 2, figsize=(9.6, 4.6 * len(rows)), squeeze=False)
    sm = None
    for r, (lab, d) in enumerate(rows):
        truth = np.asarray(d[pk], float)
        # LEFT: model and solver both against the sim -- the floor shares the panel
        P.parity_panel(axes[r][0], truth,
                       {'model': np.asarray(d[mk], float),
                        'solver (the floor)': np.asarray(d[sk], float)},
                       xlabel='sim %s' % sym, ylabel='predicted %s' % sym,
                       title='%s -- model & solver vs sim' % lab)
        # RIGHT: the model alone, coloured by strain concentration
        w = np.asarray(d.get('w_max', []), float)
        if w.size == truth.size:
            sm = P.parity_panel(axes[r][1], truth, {'model': np.asarray(d[mk], float)},
                                xlabel='sim %s' % sym, ylabel='model %s' % sym,
                                colour_by=w, title=r'%s -- coloured by max$\|W\|$' % lab)
        else:
            axes[r][1].set_visible(False)
    if sm is not None:
        fig.colorbar(sm, ax=axes[:, 1].tolist(), label=r'max$\|W\|$', fraction=0.045)
    fig.suptitle('M2 -- %s against the INDEPENDENT SIM' % sym, fontsize=12)
    P.save_fig(fig, out)
    print('  ->', os.path.basename(out))


def fig_bars(cats, series_nu, series_E, out, xlabel):
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    P.grouped_bars(axes[0], cats, series_nu, ylabel=r'MAE($\nu$) vs sim', xlabel=xlabel,
                   hline=KILL_NU, hline_label='kill line 0.05')
    axes[0].axhline(MUST_NU, color='green', ls=':', lw=1.3, label='must-tier 0.02')
    axes[0].legend(fontsize=7)
    P.grouped_bars(axes[1], cats, series_E, ylabel='MAE(E)/E  [%]', xlabel=xlabel,
                   hline=MUST_E, hline_label='must-tier 5 %')
    fig.tight_layout()
    P.save_fig(fig, out)
    print('  ->', os.path.basename(out))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--stem', default='v3_res_bravais_w10_h1_L5_ns48_h64_e400_star_ms')
    a = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)
    print('figures ->', RESULTS)

    fig_learning(os.path.join(RESULTS, 'm2_learning_curves.png'))

    # aggregate rows: the family trained AGAINST, and fresh unseen meshes (hardest family)
    rows = []
    for lab, name in [('bravais holdout (trained against)', 'eval_%s_bravais.json' % a.stem),
                      ('FRESH unseen meshes (random)', 'eval_%s_random.json' % a.stem)]:
        d = _ev(name)
        if d:
            rows.append((lab, d))
        else:
            print('  SKIP parity row %r (%s not found)' % (lab, name))
    if rows:
        fig_parity('nu', rows, os.path.join(RESULTS, 'm2_parity_nu.png'))
        fig_parity('E', rows, os.path.join(RESULTS, 'm2_parity_E.png'))

    # per-FAMILY, fresh meshes -- one eval json per family
    fam_files = sorted(f for f in os.listdir(RESULTS)
                       if f.startswith('eval_%s_' % a.stem) and f.endswith('.json'))
    fams, s_nu, s_E = [], {'model vs sim': [], 'solver vs sim (floor)': []}, \
                      {'model vs sim': [], 'solver vs sim (floor)': []}
    for f in fam_files:
        d = _ev(f)
        if not d:
            continue
        st = _stats(d)
        fams.append('%s\n(n=%d)' % (f[len('eval_%s_' % a.stem):-len('.json')], st['n']))
        s_nu['model vs sim'].append(st['mae_nu'])
        s_nu['solver vs sim (floor)'].append(st['mae_nu_solver'])
        s_E['model vs sim'].append(st['relE'])
        s_E['solver vs sim (floor)'].append(st['relE_solver'])
    if fams:
        fig_bars(fams, s_nu, s_E, os.path.join(RESULTS, 'm2_by_family.png'), 'family / dataset')

    # per-max|W|, pooled over every eval json that carries w_max
    edges = [0.0, 1e-9, 1.0, 3.0, 10.0, np.inf]
    labels = ['0', '0-1', '1-3', '3-10', '>10']
    nu_m, nu_s, e_m, e_s, ns = [], [], [], [], []
    pool = [(_ev(f)) for f in fam_files]
    pool = [d for d in pool if d and len(d.get('w_max', [])) == len(d['nu_model'])]
    if pool:
        W = np.concatenate([np.asarray(d['w_max'], float) for d in pool])
        NM, NS, NP = (np.concatenate([np.asarray(d[k], float) for d in pool])
                      for k in ('nu_model', 'nu_solver', 'nu_sim'))
        EM, ES, EP = (np.concatenate([np.asarray(d[k], float) for d in pool])
                      for k in ('E_model', 'E_solver', 'E_sim'))
        eps = float(np.mean([d['eps_E'] for d in pool]))
        for i in range(len(edges) - 1):
            m = (W >= edges[i]) & (W < edges[i + 1])
            ns.append(int(m.sum()))
            nu_m.append(float(np.mean(np.abs(NM[m] - NP[m]))) if m.any() else None)
            nu_s.append(float(np.mean(np.abs(NS[m] - NP[m]))) if m.any() else None)
            e_m.append(float(100 * np.mean(np.abs(EM[m] - EP[m]) / (np.abs(EP[m]) + eps)))
                       if m.any() else None)
            e_s.append(float(100 * np.mean(np.abs(ES[m] - EP[m]) / (np.abs(EP[m]) + eps)))
                       if m.any() else None)
        cats = ['%s\n(n=%d)' % (l, n) for l, n in zip(labels, ns)]
        fig_bars(cats, {'model vs sim': nu_m, 'solver vs sim (floor)': nu_s},
                 {'model vs sim': e_m, 'solver vs sim (floor)': e_s},
                 os.path.join(RESULTS, 'm2_by_wmax.png'), r'max$\|W\|$  (training filter at 10)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
