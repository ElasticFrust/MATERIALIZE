r"""M2 -- do the per-triangle errors CANCEL in the bulk mean, or are they systematically correlated?

Provenance: MATERIALIZE `Phase 5/m2/M2_V2_PLAN.md` S1; model `Phase 5/m2/model_v3.py`.

THE QUESTION (user, 2026-09-11).  "In principle the model should reproduce the exact tensor for
every triangle, which must give the same bulk -- so why should a bulk loss term matter at all?"

It matters because the per-triangle objective has a NULL DIRECTION.  Write `err_s` for the error on
triangle `s`.  The loss minimises

    per-triangle  =  mean_s |err_s|

while `nu` and `E` are read off

    bulk          =  |mean_s err_s|

and those rank models differently.  Two models with IDENTICAL per-triangle loss can have wildly
different bulk error:

  * errors random and zero-mean across triangles  ->  they cancel, |mean err| ~ sigma/sqrt(N);
  * errors systematically biased one way          ->  nothing cancels, |mean err| ~ sigma.

So the per-triangle loss never sees the CORRELATION STRUCTURE of its own errors, which is the whole
difference between a usable and a useless bulk prediction.

WHAT THIS SCRIPT MEASURES.  The cancellation ratio per network,

    R  =  mean_s |err_s|  /  |mean_s err_s|

against `sqrt(n_tri)`.  `R ~ sqrt(N)` means independent errors (the bulk is nearly free, and a bulk
loss term is treating a symptom); `R ~ 1` means fully correlated errors (a systematic per-network
bias that the per-triangle loss cannot see, and which a bulk term directly attacks).

WHY IT DECIDES SOMETHING.  It says whether `--bulk_weight` addresses the real mechanism before a
multi-day run is spent on it, and it explains -- or refutes -- the measured 6.3x spread in MAE(nu)
between `cells` (~6-24 triangles) and `random` (116-720) at near-identical per-triangle error.

Run:
  python "Phase 5/verifications/m2_error_cancellation.py"
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
RESULTS = os.path.join(REPO, 'Phase 5', 'results', 'm2_s1')
sys.path.insert(0, M2DIR)
sys.path.insert(0, REPO)
import model_v3 as M3                                                       # noqa: E402
import train_v2 as T2                                                       # noqa: E402
import train_v3 as T3                                                       # noqa: E402
import plotting as P                                                        # noqa: E402

torch.set_default_dtype(torch.float64)
torch.set_num_threads(2)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', default=os.path.join(M2DIR, 'data', 'dataset_fresh_s4321.npz'))
    ap.add_argument('--ckpt', default='checkpoint_v3_res_bravais_w10_h1_L5_ns48_h64_e400_star_ms.pt')
    ap.add_argument('--ref', default=os.path.join(M2DIR, 'data', 'dataset_v2_s0.npz'),
                    help='dataset supplying the TRAINING sigma, so numbers stay comparable')
    ap.add_argument('--holdout', default='bravais')
    ap.add_argument('--n', type=int, default=1200)
    a = ap.parse_args()

    tr = [g for g in T2.load(a.ref)
          if g['sim_ok'] and g['size_bin'] != 'train_unused' and g['family'] != a.holdout]
    sd = np.concatenate([np.asarray(g['C6_per'], float) for g in tr]).std(0).clip(1e-12)
    gs = [g for g in T2.load(a.data) if g['sim_ok'] and g['size_bin'] != 'large_holdout']
    sel = np.random.default_rng(0).choice(len(gs), min(a.n, len(gs)), replace=False)
    net = M3.from_checkpoint(torch.load(os.path.join(M2DIR, a.ckpt), weights_only=False))
    print('checkpoint %s  ->  %d networks from %s' % (a.ckpt, len(sel), os.path.basename(a.data)))

    rows = []
    with torch.no_grad():
        for i in sel:
            g = gs[i]
            err = (T3.predict(net, T3.prepare(g)).numpy()
                   - np.asarray(g['C6_per'], float)) / sd          # (n_tri, 6), sigma-normalised
            per = float(np.abs(err).mean())                        # mean_s |err_s|
            blk = float(np.abs(err.mean(0)).mean())                # |mean_s err_s|
            rows.append(dict(per=per, blk=blk, R=per / max(blk, 1e-30), n_tri=err.shape[0],
                             family=g['family'], w_max=float(g['w_max'])))

    print('\n  %-12s %6s %8s %10s %10s %8s %9s %8s'
          % ('family', 'n', 'med n_tri', 'per-tri', 'bulk', 'R', 'sqrt(N)', 'R/sqrtN'))
    out = {'ckpt': a.ckpt, 'data': os.path.basename(a.data), 'by_family': {}}
    for f in sorted({r['family'] for r in rows}) + ['ALL']:
        sel_r = rows if f == 'ALL' else [r for r in rows if r['family'] == f]
        if not sel_r:
            continue
        nt = np.array([r['n_tri'] for r in sel_r], float)
        R = np.array([r['R'] for r in sel_r])
        rec = dict(n=len(sel_r), med_ntri=float(np.median(nt)),
                   per=float(np.mean([r['per'] for r in sel_r])),
                   blk=float(np.mean([r['blk'] for r in sel_r])),
                   R=float(np.median(R)), sqrtN=float(np.median(np.sqrt(nt))))
        rec['R_over_sqrtN'] = rec['R'] / max(rec['sqrtN'], 1e-30)
        out['by_family'][f] = rec
        print('  %-12s %6d %8.0f %10.4f %10.4f %8.2f %9.2f %8.3f'
              % (f, rec['n'], rec['med_ntri'], rec['per'], rec['blk'], rec['R'], rec['sqrtN'],
                 rec['R_over_sqrtN']))

    allr = out['by_family']['ALL']
    print('\n  R = mean_s|err_s| / |mean_s err_s|   (cancellation actually achieved)')
    print('  R ~ sqrt(N)  => errors INDEPENDENT, the bulk is nearly free')
    print('  R ~ 1        => errors CORRELATED, a systematic per-network bias the per-triangle')
    print('                  loss cannot see -- which is what a bulk term attacks')
    print('  measured overall: R = %.2f against sqrt(N) = %.2f  ->  R/sqrt(N) = %.3f'
          % (allr['R'], allr['sqrtN'], allr['R_over_sqrtN']))

    def draw(ax):
        nt = np.array([r['n_tri'] for r in rows], float)
        ax.scatter(np.sqrt(nt), [r['R'] for r in rows], s=10, alpha=0.5, linewidths=0,
                   label='per network')
        hi = np.sqrt(nt).max()
        ax.plot([1, hi], [1, hi], 'k--', lw=1.3, label=r'independent errors: $R=\sqrt{N}$')
        ax.axhline(1.0, color='crimson', ls=':', lw=1.3, label='fully correlated: $R=1$')
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel(r'$\sqrt{n_{\rm tri}}$'); ax.set_ylabel(r'cancellation ratio $R$')
        ax.grid(alpha=0.25); ax.set_axisbelow(True); ax.legend(fontsize=7); ax.set_box_aspect(1)
    P.save_element(draw, os.path.join(RESULTS, 'm2_error_cancellation.png'), figsize=(5.4, 5.0))

    with open(os.path.join(RESULTS, 'error_cancellation.json'), 'w', encoding='utf-8') as fh:
        json.dump(out, fh, indent=2)
    print('  -> error_cancellation.json + m2_error_cancellation.png')
    return 0


if __name__ == '__main__':
    sys.exit(main())
