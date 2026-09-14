"""Does the CONDITIONING of the solver's constraint system predict the surrogate's error?

THE QUESTION. Every lever that could be measured has been closed: capacity (section 9g, 39.7:1),
data volume (+9.7 %), the schedule on the full arm (9i), `--bulk_weight`, `--graph_balance`, and the
near-mechanism tail (9j, which REDISTRIBUTED error rather than reducing it). All three independent
routes say the model is REPRESENTATION-limited. This asks WHICH representational property is missing,
and it discriminates between two candidates no previous experiment separated:

  (a) REACH      -- information cannot travel far enough. Already doubted: depth 5 gives ~10 hops on
                    72-240-triangle meshes and the errors are correlated over ~1 hop.
  (b) ITERATIONS -- `W(s)` is the solution of a GLOBAL constrained system, i.e. an operator INVERSE.
                    A fixed-depth message-passing net is a fixed number of relaxation sweeps, and the
                    sweeps a linear solve needs grow with the system's CONDITION NUMBER. Under (b)
                    the error should track cond(G), with k-contrast only a PROXY for it.

WHY cond(G) IS THE RIGHT MATRIX. The intrinsic solve ends in `lstsq(G, r)` with `G = J3 @ PinvJt`
(`forward_solver_torch.py`), the constraint Gram matrix -- so G is precisely the operator whose
inverse sets `W`, and its conditioning is what a relaxation scheme would have to iterate against.

EFFECTIVE conditioning, not `np.linalg.cond`. G is SINGULAR BY CONSTRUCTION -- the constraint rows
are redundant (audit B-1: rank 671/672, cond ~3e16 on the regular lattice), so a raw condition number
is ~1e16 for every sample and discriminates nothing. We take sigma_max / sigma_min over the spectrum
ABOVE the same rcond cutoff `lstsq` itself uses (eps * max(shape) * sigma_max).

The core is PROTECTED and is not touched: G is captured by monkey-patching `torch.linalg.lstsq` for
the duration of one forward solve -- the technique `Phase 3/verifications/b1_excursion_analysis.py`
already uses to instrument this same call.

BIN, DO NOT CORRELATE. Error rises 7x with max|W| while corr(log10 max|W|, |dnu|) = -0.045, because
the effect is a sparse tail on a flat bulk (CLAUDE.md section 3). Everything here is reported as
binned medians, and the decisive test is STRATIFIED: within a contrast bin, does cond(G) still
separate the error? If it does, conditioning is the variable and contrast was the proxy.

Run:  C:\\Users\\doron\\anaconda3\\python.exe "Phase 5/verifications/m2_conditioning_vs_error.py"
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
M2DIR = os.path.join(REPO, 'Phase 5', 'm2')
sys.path.insert(0, os.path.join(REPO, 'Phase 3', 'verifications'))
import _common as C                                                       # noqa: E402,F401
sys.path.insert(0, os.path.join(REPO, 'Phase 5'))
sys.path.insert(0, M2DIR)

import torch                                                              # noqa: E402
torch.set_default_dtype(torch.float64)

import model_v3 as M3                                                     # noqa: E402
import train_v3 as T3                                                     # noqa: E402
import train_v2 as T2                                                     # noqa: E402
from evaluate_v2 import geo_of                                            # noqa: E402
from inverse_design import DesignProblem, c6_to_nuE                       # noqa: E402
import solve_probe as SP                                                  # noqa: E402

RESULTS = os.path.join(REPO, 'Phase 5', 'results', 'm2_s1')


def capture_G(prob, k):
    """Run one forward solve, capturing G from the `lstsq` the intrinsic path ends in.

    Both the capture and the effective-conditioning arithmetic live in
    `Phase 3/verifications/solve_probe.py`, shared with `b1_excursion_analysis.py` -- the cutoff
    convention is easy to get subtly wrong and a wrong one silently redefines "rank".
    """
    def _run():
        with torch.no_grad():
            return prob.forward(torch.as_tensor(np.asarray(k, float)), physical_units=True)

    return SP.capture_lstsq_lhs(_run)                       # the FIRST lstsq is the KKT solve


def bin_report(name, x, err, nbins=5):
    """Median error per quantile bin of `x`, plus the spread that variable explains."""
    x, err = np.asarray(x, float), np.asarray(err, float)
    ok = np.isfinite(x) & np.isfinite(err)
    x, err = x[ok], err[ok]
    qs = np.quantile(x, np.linspace(0, 1, nbins + 1))
    qs[-1] *= 1 + 1e-9
    rows, meds = [], []
    for i in range(nbins):
        m = (x >= qs[i]) & (x < qs[i + 1])
        if m.sum() == 0:
            continue
        meds.append(float(np.median(err[m])))
        rows.append((float(qs[i]), float(qs[i + 1]), int(m.sum()), meds[-1]))
    spread = max(meds) / max(min(meds), 1e-30) if meds else float('nan')
    print('\n  by %s  (median |dnu| per quintile)' % name)
    print('    %-26s %6s %10s' % ('bin', 'n', 'median'))
    for lo, hi, n, md in rows:
        print('    %-26s %6d %10.4f' % ('%.3g .. %.3g' % (lo, hi), n, md))
    print('    SPREAD explained (max/min) = %.2fx' % spread)
    return {'bins': rows, 'spread': spread}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', default=os.path.join(M2DIR, 'data', 'dataset_v2_s0.npz'))
    ap.add_argument('--ckpt', default='checkpoint_v3_res_bravais_w10_h1_L5_ns48_h64_e400_star_ms.pt')
    ap.add_argument('--holdout', default='bravais')
    ap.add_argument('--n', type=int, default=400, help='networks to solve (each needs an SVD of G)')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    raw = [g for g in T2.load(a.data) if g['sim_ok'] and g['size_bin'] != 'large_holdout']
    val = [g for g in raw if g['family'] == a.holdout]
    rng = np.random.default_rng(a.seed)
    if len(val) > a.n:
        val = [val[i] for i in sorted(rng.choice(len(val), a.n, replace=False))]
    print('scoring %d %s networks (seed %d)' % (len(val), a.holdout, a.seed))

    net = M3.from_checkpoint(torch.load(os.path.join(M2DIR, a.ckpt), weights_only=False))
    # NO exception handling in the loop, deliberately (CLAUDE.md: fail-fast by default). This path
    # touches only the SOLVER, which "degrades gracefully and needs no guard" -- the catchable
    # `UnhealthyGeometryError` belongs to the SIM entry points, which are never called here. So there
    # is no expected benign failure, and anything that does fail is a bug worth stopping for. An
    # earlier broad `except Exception` swallowed a TypeError on all 12 smoke samples, reported them as
    # "skipped", and cost a debugging cycle -- and had it hit only SOME samples it would have biased
    # the population silently.
    rec = []
    for i, g in enumerate(val):
        prob = DesignProblem.from_geo(geo_of(g))
        _, G = capture_G(prob, g['k'])
        if G is None:
            continue
        ce, ndrop = SP.effective_cond(G)
        t = T3.prepare(g)
        with torch.no_grad():
            pred = T3.predict(net, t).numpy()
        tgt = np.asarray(g['C6_per'], float)
        # `c6_to_nuE` is torch-native, and the TARGET is the stored bulk `C6` -- both exactly as
        # `m2_error_strata.py` does it, so the two scripts define nu identically rather than
        # nearly so.
        nu_p = float(c6_to_nuE(torch.as_tensor(pred.mean(0)))[0])
        nu_t = float(c6_to_nuE(torch.as_tensor(np.asarray(g['C6'], float)))[0])
        rec.append(dict(cond=ce, rank_drop=ndrop, contrast=float(g['contrast']),
                        n_tri=int(len(tgt)), w_max=float(g['w_max']),
                        dnu=abs(nu_p - nu_t),
                        mae=float(np.abs(pred - tgt).mean())))
        if (i + 1) % 50 == 0:
            print('  ... %d/%d' % (i + 1, len(val)))

    if not rec:
        raise SystemExit('nothing scored')
    cond = np.array([r['cond'] for r in rec])
    dnu = np.array([r['dnu'] for r in rec])
    print('\n%d networks solved.  cond(G) spans %.2e .. %.2e (median %.2e); rank drop median %d'
          % (len(rec), cond.min(), cond.max(), np.median(cond),
             int(np.median([r['rank_drop'] for r in rec]))))

    out = {'ckpt': a.ckpt, 'n': len(rec), 'by': {}}
    for nm, key in (('cond(G)', 'cond'), ('k-contrast', 'contrast'),
                    ('n_tri', 'n_tri'), ('max|W|', 'w_max')):
        out['by'][nm] = bin_report(nm, [r[key] for r in rec], dnu)

    # THE DECISIVE TEST: within a contrast bin, does cond(G) still separate the error?
    print('\n  STRATIFIED -- within each k-contrast tercile, split by median cond(G):')
    ct = np.array([r['contrast'] for r in rec], float)
    edges = np.quantile(ct, [0, 1 / 3, 2 / 3, 1.0])
    edges[-1] *= 1 + 1e-9
    strat = []
    for i in range(3):
        m = (ct >= edges[i]) & (ct < edges[i + 1])
        if m.sum() < 8:
            continue
        cm = np.median(cond[m])
        lo, hi = dnu[m & (cond <= cm)], dnu[m & (cond > cm)]
        if not len(lo) or not len(hi):
            continue
        r = float(np.median(hi) / max(np.median(lo), 1e-30))
        strat.append({'contrast_bin': [float(edges[i]), float(edges[i + 1])], 'n': int(m.sum()),
                      'median_lo_cond': float(np.median(lo)),
                      'median_hi_cond': float(np.median(hi)), 'ratio': r})
        print('    contrast %-18s n=%3d   low-cond %.4f   high-cond %.4f   ratio %.2fx'
              % ('%.3g..%.3g' % (edges[i], edges[i + 1]), m.sum(),
                 np.median(lo), np.median(hi), r))
    out['stratified'] = strat
    out['records'] = rec
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, 'conditioning_vs_error_%s.json'
                        % os.path.splitext(a.ckpt)[0].replace('checkpoint_v3_', ''))
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(out, fh, indent=2)
    print('\n  ->', path)
    return 0


if __name__ == '__main__':
    sys.exit(main())
