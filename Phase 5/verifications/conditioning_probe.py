r"""Phase 5 — does A(s) CONDITIONING explain where the solver and the sim disagree?

THE QUESTION
------------
`CLAUDE.md` §3 names two independent routes to per-triangle `A(s)` rank loss, where the solver's
inverse is set by its regulariser rather than by physics and `C(s)` can be arbitrarily wrong:

  (1) LOW k      -- soft/dead edges. Guarded only INDIRECTLY by `reg` (a k-variance penalty).
  (2) DEGENERATE GEOMETRY -- slivers make the three q_e outer products near-coplanar. Guarded only
      by `require_healthy_mesh`, via a triangle-AREA proxy.

Neither guard inspects `A(s)`. Nothing has been fixed: there is no conditioning check on any live
path, and `positions.quality_floor` -- the right-shaped (angle/shape) guard -- is implemented but
DEFAULTS TO 0.0 (off) pending a calibration that was never done.

What HAS been measured (2026-08-16, recorded in `positions.tri_shape_quality`): geometric sliver
proxies correlate with log10(gap) at -0.52 (shape quality), -0.51 (min angle), -0.42 (area),
-0.11 (edge length, useless). But shape quality is a PROXY for A(s) rank loss, not A(s) itself.

WHY BOTH EXPERIMENTS
--------------------
In `g1_2`, k = 1 EXACTLY, so A(s) = sum_e (1/4l^2) q_e q_e^T is a purely GEOMETRIC quantity --
A(s) conditioning and shape quality carry the same information there, and route (1) cannot arise.
Only `goal1` designs k across five contrast bands, so only goal1 can separate the two routes. It
also holds the largest single disagreement in the project (|dnu| = 0.311, in the `medium` band at
an unremarkable nu ~ +0.25).

WHAT IT MEASURES (per design)
-----------------------------
A(s) is assembled EXACTLY as the solver does it (`forward_solver_torch.forward`):
    q_geom_e = [vx^2, 2 vx vy, vy^2];  A3 = sum_e (k_e / 4 l_e^2) q_e q_e^T
then per triangle rcond = lambda_min / lambda_max, and per design the tail statistics (worst
triangle, 5th percentile, fraction below thresholds) alongside shape quality, area and k-contrast.
These are correlated against that design's solver-vs-sim |dnu|.

A correlation here is EVIDENCE FOR the mechanism, not proof: both codes are linear formulations and
no third path exists to say which one is right where they differ (audit A-18).

Determinism: a pure recompute of saved designs -- no RNG, so no seed to take or emit.

Run:  C:\\Users\\doron\\anaconda3\\python.exe "Phase 5/verifications/conditioning_probe.py"
Writes: Phase 5/results/conditioning_probe/{g1_2,goal1}_conditioning.csv + PROBE.md inputs
"""
# ---- §0 preamble (verbatim; this file lives in Phase 5/verifications/) ------------------------
import os, sys, csv
import numpy as np
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'Phase 3', 'verifications'))
import _common as C                 # wires the rest of sys.path + the solver stack
import torch                        # noqa: E402
torch.set_default_dtype(torch.float64)   # REQUIRED — the whole stack is float64

sys.path.insert(0, os.path.join(REPO, 'Phase 5'))
import positions                                           # noqa: E402  — tri_shape_quality
sys.path.insert(0, os.path.join(REPO, 'Phase 2'))
from persistence import _provenance                        # noqa: E402

OUTDIR = os.path.join(REPO, 'Phase 5', 'results', 'conditioning_probe')


def a3_rcond(geo, bond_k):
    """Per-triangle lambda_min/lambda_max of A(s), assembled EXACTLY as the solver assembles it.

    Mirrors `forward_solver_torch.forward`: q_geom = [vx^2, 2 vx vy, vy^2] per edge and
    A3 = sum_e (k_e / 4 l_e^2) q_e q_e^T. Reproducing the solver's own packing is the point -- a
    re-derived A(s) would be measuring a different matrix from the one that gets inverted."""
    ev = np.asarray(geo['edge_vecs'], float)                   # (N,3,2) per-triangle edge vectors
    vx, vy = ev[:, :, 0], ev[:, :, 1]
    q = np.stack([vx ** 2, 2.0 * vx * vy, vy ** 2], axis=2)    # (N,3,3) [tri, edge, comp]
    l2 = (ev ** 2).sum(2)                                      # (N,3) edge length^2
    ke = np.asarray(geo['tri_k'], float) if 'tri_k' in geo else None
    if ke is None or ke.shape != l2.shape:                     # fall back to per-bond -> per-tri
        ke = np.asarray(bond_k, float)[np.asarray(geo['tri_bond'])]
    stiff = ke / (4.0 * np.maximum(l2, 1e-300))                # k_e / 4 l_e^2
    A3 = np.einsum('ne,nei,nej->nij', stiff, q, q)             # (N,3,3) symmetric
    w = np.linalg.eigvalsh(A3)                                 # ascending
    return np.abs(w[:, 0]) / np.maximum(np.abs(w[:, -1]), 1e-300)


def design_row(path, bond_k=None):
    """Per-design tail statistics of A(s) conditioning, shape quality, area and k-contrast."""
    geo, k, _C6, _meta = C.load_network(path)
    rc = a3_rcond(geo, k)
    q = positions.tri_shape_quality(geo)
    a = np.asarray(geo['areas'], float)
    kk = np.asarray(k, float)
    return dict(n_tri=len(rc),
                rcond_min=float(rc.min()), rcond_p05=float(np.percentile(rc, 5)),
                frac_rcond_lt_1e3=float((rc < 1e-3).mean()),
                frac_rcond_lt_1e4=float((rc < 1e-4).mean()),
                quality_min=float(q.min()), quality_p05=float(np.percentile(q, 5)),
                area_min_over_mean=float(a.min() / a.mean()),
                k_min_over_mean=float(kk.min() / max(kk.mean(), 1e-300)),
                frac_k_lt_1e3=float((kk < 1e-3 * max(kk.mean(), 1e-300)).mean()))


def corr(x, y):
    """Pearson r on log10 of the response, and Spearman (rank) r -- reported together because the
    relation need not be linear and a rank correlation cannot be flattered by a few extreme points."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return float('nan'), float('nan')
    rx = np.argsort(np.argsort(x[m])).astype(float)
    ry = np.argsort(np.argsort(y[m])).astype(float)
    return (float(np.corrcoef(x[m], y[m])[0, 1]), float(np.corrcoef(rx, ry)[0, 1]))


def collect_g1_2():
    """g1_2: k = 1 EXACTLY, so A(s) is purely geometric -- route (1) cannot arise here."""
    res = os.path.join(REPO, 'Phase 5', 'results', 'g1_2')
    rows = list(csv.DictReader(open(os.path.join(res, 'results.csv'))))
    d = np.load(os.path.join(res, 'solver_recheck.npz'), allow_pickle=True)
    dnu = {int(r): (float(abs(np.mean(ns) - np.mean(nv))), float(np.abs(ns - nv).max()))
           for r, ns, nv, ok in zip(d['run_id'], d['nu_sim'], d['nu_solver'], d['ok']) if ok}
    out = []
    for r in rows:
        rid = int(float(r['run_id']))
        if rid not in dnu or not r['design_path'] or not os.path.exists(r['design_path']):
            continue
        row = design_row(r['design_path'])
        row.update(run_id=rid, topo=r['topo'], nu_target=float(r['nu_target']),
                   nu_achieved=float(r['nu_achieved_sim']), gap=float(r['solver_sim_gap']),
                   dnu_bulk=dnu[rid][0], dnu_worst=dnu[rid][1])
        out.append(row)
    return out


def collect_goal1():
    """goal1: k is DESIGNED across five contrast bands -- the only place route (1) is testable."""
    res = os.path.join(REPO, 'Phase 5', 'results', 'goal1')
    out = []
    for r in csv.DictReader(open(os.path.join(res, 'results.csv'))):
        p = r['design_path']
        if not p or not os.path.exists(p):
            continue
        row = design_row(p)
        row.update(run_id=int(float(r['run_id'])), topo=r['topo_class'], band=r['band'],
                   band_f=float(r['band_f']), nu_target=float(r['nu_target']),
                   nu_achieved=float(r['nu_sim_full']), gap=float(r['solver_sim_gap']),
                   dnu_bulk=abs(float(r['nu_sim_full']) - float(r['nu_solver_full'])),
                   dnu_worst=float('nan'))
        out.append(row)
    return out


def report(name, rows, predictors):
    print(f'\n=== {name}: {len(rows)} designs ===')
    y = np.log10(np.maximum([r['dnu_bulk'] for r in rows], 1e-6))
    print(f"  response = log10|dnu_bulk|   ({sum(1 for r in rows if r['dnu_bulk'] < 1e-6)} designs "
          f"agree to <1e-6 and sit at the floor)")
    print(f"  {'predictor':22} {'pearson':>9} {'spearman':>9}")
    for p in predictors:
        pr, sp = corr([r[p] for r in rows], y)
        print(f'  {p:22} {pr:9.3f} {sp:9.3f}')
    big = sorted(rows, key=lambda r: -r['dnu_bulk'])[:5]
    print('  worst-disagreement designs:')
    for r in big:
        print(f"    |dnu|={r['dnu_bulk']:.4f} nu={r['nu_achieved']:+.3f} {str(r['topo'])[:18]:18} "
              f"rcond_min={r['rcond_min']:.2e} q_min={r['quality_min']:.4f} "
              f"k_min/mean={r['k_min_over_mean']:.2e}")


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    commit, dirty, saved = _provenance()
    preds = ['rcond_min', 'rcond_p05', 'frac_rcond_lt_1e3', 'frac_rcond_lt_1e4',
             'quality_min', 'quality_p05', 'area_min_over_mean',
             'k_min_over_mean', 'frac_k_lt_1e3']
    for name, rows in (('g1_2', collect_g1_2()), ('goal1', collect_goal1())):
        if not rows:
            print(f'{name}: no designs found'); continue
        report(name, rows, preds)
        dest = os.path.join(OUTDIR, f'{name}_conditioning.csv')
        with open(dest, 'w', newline='') as fh:
            fh.write(f'# commit={commit} dirty={dirty} saved_utc={saved}\n')
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
        print(f'  -> {dest}')
    print(f'\n[commit {commit[:7]} dirty={dirty}]')


if __name__ == '__main__':
    main()
