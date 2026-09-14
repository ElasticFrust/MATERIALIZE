"""Is the high-contrast error a CAPACITY floor or a generalisation gap?

Scores the REAL S1 checkpoint (`..._L5_..._star_ms`, per-triangle MAE/sigma 0.0925 on the `bravais`
holdout) on the EXACT 200 samples the overfit probe memorised, so the two numbers are on one
population and one metric. The probe is free to memorise those 200; the real model saw ~198 of them
inside its 40k train set. If the probe's score is close to the real model's, focusing all capacity on
the hard population buys nothing -- i.e. the architecture is at its representational FLOOR there, and
neither more data nor better generalisation can help.

Why not read it off the contrast strata table (M2_RESIDUAL_AND_CONSTRAINTS.md SS7b, 0.2493 at
contrast 1e4-1e9): that row is the `bravais` VAL split, all one family, while the probe's 200 are
114 random / 45 disordered / 29 cells / 10 longrange / 2 bravais. Different populations.

Read-only: loads two checkpoints, writes one JSON. Implements nothing new -- reuses train_v3's own
`evaluate`, so the metric is identical by construction rather than by reimplementation.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
M2DIR = os.path.join(HERE, '..', 'm2')
sys.path.insert(0, M2DIR)

torch.set_default_dtype(torch.float64)

import model_v3 as M3                                                     # noqa: E402
import train_v2 as T2                                                     # noqa: E402
import train_v3 as T3                                                     # noqa: E402

TRAINED = os.path.join(M2DIR, 'checkpoint_v3_res_bravais_w10_h1_L5_ns48_h64_e400_star_ms.pt')
PROBES = {200: os.path.join(M2DIR, 'checkpoint_v3_res_bravais_w0_h1_L5_ns48_h64_e1600_star_ms'
                                  '_probe200_c10000.pt'),
          8: os.path.join(M2DIR, 'checkpoint_v3_res_bravais_w0_h1_L5_ns48_h64_e6000_star_ms'
                                '_probe8_c10000.pt')}
RESULTS = os.path.join(HERE, '..', 'results', 'm2_s1')


#: everything that determines which samples `--overfit_probe` drew. `seed` alone is NOT enough:
#: the draw is `rng(seed).choice(len(raw), n)`, and `raw` is built from `data` then SHRUNK by
#: `contrast_min`, so the same seed over a different pool selects entirely different networks.
DRAW_KEYS = ('seed', 'data', 'overfit_probe', 'contrast_min')


def draw_from_checkpoint(ck, name, expect_n=None):
    """-> the four parameters that reproduce a probe's sample draw; FAIL FAST if any is missing.

    Extracted from `main` so `test_m2_train_surface.py` exercises the shipped refusal rather than a
    copy of it.
    """
    draw = {}
    for key in DRAW_KEYS:
        if key not in ck:
            raise SystemExit(
                'probe checkpoint %s does not record %r, so its sample draw cannot be reproduced. '
                'Checkpoints written before 2026-09-14 predate this; re-save it with the value from '
                'its run tag.' % (name, key))
        draw[key] = ck[key]
    if expect_n is not None and int(draw['overfit_probe']) != int(expect_n):
        raise SystemExit('--n %d but %s was trained on %d samples'
                         % (expect_n, name, int(draw['overfit_probe'])))
    return draw


def build(ck):
    """Rebuild the net from a checkpoint's own recorded architecture (never assumed)."""
    net = M3.ForwardGNNv3(ns=ck['ns'], nt=ck['nt'], hidden=ck['hidden'],
                          n_layers=ck['layers'],
                          use_star=ck.get('use_star', True),
                          use_global=ck.get('use_global', True))
    net.load_state_dict(ck['state'])
    net.eval()
    return net


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--n', type=int, default=200, choices=sorted(PROBES),
                    help='which overfit probe to compare: 200 samples (1.4:1 params-to-targets) '
                         'or 8 (39.7:1). Both drawn from the same pool with the same rng and seed, '
                         'so the 8 are a SUBSET-like draw of the same population, not a new one.')
    a = ap.parse_args()
    out_path = os.path.join(RESULTS, 'probe_vs_trained%s.json'
                            % ('' if a.n == 200 else '_n%d' % a.n))

    # THE SAMPLE DRAW IS READ FROM THE PROBE CHECKPOINT, never assumed. The probe's 200 (or 8)
    # samples were never saved as a list -- they were DRAWN, by `rng(seed).choice(len(raw), n)` -- so
    # this script has to reproduce the draw exactly. Four things determine it: the seed, the dataset
    # file, the contrast filter (which SHRINKS `raw` before the draw, changing what every index
    # means), and the count. Hardcoding any of them encodes an assumption about how a DIFFERENT
    # program was invoked, and a mismatch produces a normal-looking table for the wrong population --
    # which is exactly how a resume against the wrong `--data` went unnoticed until a log line gave
    # it away (NEXT_SESSION, 2026-09-13). Taking all four from the checkpoint makes a mismatch
    # impossible rather than merely documented.
    probe_ck = torch.load(PROBES[a.n], weights_only=False)
    draw = draw_from_checkpoint(probe_ck, os.path.basename(PROBES[a.n]), expect_n=a.n)
    trained_ck = torch.load(TRAINED, weights_only=False)
    if trained_ck.get('data') != draw['data']:
        raise SystemExit('the two checkpoints were trained on different datasets (%s vs %s), so '
                         'there is no common population to score them on'
                         % (trained_ck.get('data'), draw['data']))
    data_path = os.path.join(M2DIR, 'data', str(draw['data']))
    print('draw reproduced from the probe checkpoint: data=%s seed=%d contrast_min=%g n=%d'
          % (draw['data'], int(draw['seed']), float(draw['contrast_min']), a.n))
    raw = [g for g in T2.load(data_path) if g['sim_ok'] and g['size_bin'] != 'large_holdout']
    raw = [g for g in raw if float(g.get('contrast', 0.0)) >= float(draw['contrast_min'])]
    # The probe's selection, reproduced exactly: same rng, same seed (--seed, default 0), same pool.
    sel = np.random.default_rng(int(draw['seed'])).choice(len(raw), a.n, replace=False)
    sub = [raw[i] for i in sorted(sel)]
    samples = [T3.prepare(g) for g in sub]
    # ONE sd for BOTH models, computed on the population being scored. The two runs normalised by
    # their own train-set sd, so their published numbers are not on a common scale -- recomputing
    # model and reference in ONE pass on ONE split is the lesson of the "0.5144 was the train
    # split" error (M2_RESIDUAL_AND_CONSTRAINTS.md, 2026-08-27).
    sd = torch.cat([t['target'] for t in samples]).std(0).clamp_min(1e-12)

    out = {'n': len(samples), 'draw': {k: (float(v) if k != 'data' else v)
                                       for k, v in draw.items()}, 'n_triangles': int(sum(len(s['target']) for s in samples)),
           'label_std': [float(x) for x in sd]}
    for name, path in (('trained_full_40k', TRAINED),
                       ('overfit_probe_%d' % a.n, PROBES[a.n])):
        ck = torch.load(path, weights_only=False)
        per, bad, blk = T3.evaluate(build(ck), samples, sd)
        out[name] = dict(mae_over_std=float((per / sd).mean()),
                         bulk_mae_over_std=float((blk / sd).mean()),
                         spd_violation=float(bad), checkpoint=os.path.basename(path))
        print('%-18s per-triangle MAE/sigma %.4f   bulk %.4f' % (name, out[name]['mae_over_std'],
                                                                 out[name]['bulk_mae_over_std']))
    with open(out_path, 'w', encoding='utf-8') as fh:
        json.dump(out, fh, indent=2)
    print('wrote %s' % out_path)
    return 0


if __name__ == '__main__':
    sys.exit(main())
