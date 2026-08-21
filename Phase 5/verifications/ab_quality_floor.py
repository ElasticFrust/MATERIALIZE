"""Stage 3b — does a SHAPE-QUALITY floor on the position search buy trustworthiness?

**The problem.** `g1_2` (positions-only design at uniform k=1) re-ran on 2026-08-16 with working
instruments: only **~24 % of runs are trustworthy** (solver-vs-sim gap < 0.05), median gap 0.394,
max 7.34. SPSA moves vertices with nothing stopping it walking out of the SOLVER's validity domain:
`A(s) = sum_e (k_e/4l^2) q_e q_e^T` loses rank when two of a triangle's edges go near-PARALLEL, and
there the solver's inverse is set by its regulariser, not by physics (`CLAUDE.md` §3). k is uniform
here, so this is purely the GEOMETRIC route to that failure.

**Why the existing guard misses it.** `spsa_positions` already rejects steps that drop a triangle
below `1e-3 x mean area`. But on the 124 saved designs the UNTRUSTED ones have median min/mean area
**0.047 — 47x above that floor**. It essentially never fires. Measured predictors of log10(gap):

    min shape quality  corr -0.523     min ANGLE      corr -0.505
    min/mean AREA      corr -0.418     min EDGE LEN   corr -0.107   <-- nearly useless

A sliver has LONG edges and a tiny angle, so a minimum-DISTANCE constraint would pass it. Quality
(equivalently min angle) is the thing to constrain.

**What this script does NOT do:** pick a threshold off that correlation. corr ~-0.5 says quality
predicts the gap, not that any particular floor is right; calibrating a gate from correlational data
at n=124 is the same "tolerance set from noise" error made and retracted over B-1 earlier that day.
So `quality_floor` shipped defaulting to 0.0 (OFF) until this A/B was run. **RESULT (2026-08-21,
40 runs under the scored selection): the default is now 1e-3** — free (trustworthy 38% either way,
median err 0.0061 -> 0.0043, within noise at n=8) and it blocks only the numerically broken, 2 of
110 designs. Floors >= 0.03 double trustworthiness but raise median error 17x, i.e. they forbid the
designs that were the point — see the table in `positions.spsa_positions`.

**Design.** Same topologies, same targets, same seeds; the ONLY difference is `quality_floor`. Each
arm reports the trustworthy fraction, the gap distribution, and the achieved design error -- because
a floor RESTRICTS THE DESIGN SPACE, so the honest question is not "does trustworthiness go up" but
**"what does the trustworthiness cost in reach?"**

Read it as: floor helps iff trustworthy fraction rises MATERIALLY while median achieved error does
not degrade much. If both move together, the floor is just forbidding the designs that were the
point.

Run:  python ab_quality_floor.py [n_topo] [floors...]
      python ab_quality_floor.py 6 0.0 0.02 0.05 0.10
Out:  Phase 5/results/ab_quality_floor/
"""
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'Phase 3', 'verifications'))
import _common as C                                                    # noqa: E402
sys.path.insert(0, os.path.join(REPO, 'Phase 5'))
sys.path.insert(0, HERE)
import physical_homog as PH                                            # noqa: E402
import positions                                                       # noqa: E402
import run_g1_2 as G                                                   # noqa: E402

torch.set_default_dtype(torch.float64)
OUT = os.path.join(REPO, 'Phase 5', 'results', 'ab_quality_floor')
NU_TARGETS = (-0.30, 0.10)          # one auxetic (hard, drives slivers), one mild
GAP_TOL = 0.05


def one(topo, nu_target, floor, seed=None):
    """One positions-only design through the REAL driver, with `floor` injected.

    Deliberately calls `run_g1_2.design_one` rather than rolling a reduced search here. A first
    version of this A/B used its own `design_with_positions(n_outer=2, spsa_steps=20)` call, and
    every floor returned BYTE-IDENTICAL results — because that gentler search never distorted the
    geometry (final min quality ~0.72) so the floor had nothing to reject. The real driver uses
    several jittered restarts across a range of SPSA step sizes and reaches min quality ~0.05.
    **A guard can only be tested under the conditions that trigger the failure**, so the floor is
    injected by patching `positions.spsa_positions` and the genuine search is run."""
    t0 = time.time()
    orig = positions.spsa_positions

    def patched(*a, **kw):
        kw['quality_floor'] = floor
        return orig(*a, **kw)

    positions.spsa_positions = patched
    try:
        geoB, kB, rep, loss = G.design_one(topo, nu_target)
    except PH.UnhealthyGeometryError as e:
        return dict(floor=floor, nu_target=nu_target, status=f'UNHEALTHY:{str(e)[:40]}',
                    gap=float('nan'), nu=float('nan'), err=float('nan'),
                    minq=float('nan'), secs=time.time() - t0)
    finally:
        positions.spsa_positions = orig
    nu = float(rep['nu_sim'].mean())
    gap = float(rep['solver_sim_gap'])
    return dict(floor=floor, nu_target=nu_target, status='ok', gap=gap, nu=nu,
                err=abs(nu - nu_target), minq=float(positions.tri_shape_quality(geoB).min()),
                secs=time.time() - t0)


def main():
    n_topo = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    floors = [float(x) for x in sys.argv[2:]] or [0.0, 0.02, 0.05, 0.10]
    os.makedirs(OUT, exist_ok=True)

    topos, _sigs = G.build_topologies()          # returns (list_of_records, signature_audit)
    topos = topos[:n_topo]
    print(f'A/B quality floor — {n_topo} topologies x {len(NU_TARGETS)} targets x '
          f'{len(floors)} floors = {n_topo*len(NU_TARGETS)*len(floors)} runs', flush=True)

    rows = []
    for ti, t in enumerate(topos):
        for nu_t in NU_TARGETS:
            for fl in floors:
                r = one(t, nu_t, fl)          # design_one seeds itself identically per topo/target,
                r['topo'] = t['name']         # so the arms are matched pairs differing only in floor
                rows.append(r)
                print(f"  {r['topo'][:22]:22s} nu*={nu_t:+.2f} floor={fl:.3f} -> "
                       f"nu={r['nu']:+.4f} err={r['err']:.4f} gap={r['gap']:.4f} "
                       f"minq={r['minq']:.4f} {r['status']}", flush=True)

    with open(os.path.join(OUT, 'ab_results.json'), 'w') as fh:
        json.dump(rows, fh, indent=2)

    print('\n' + '=' * 72)
    print(f"{'floor':>7} {'n':>4} {'trustworthy':>12} {'median gap':>11} "
          f"{'median err':>11} {'median minq':>12}")
    for fl in floors:
        s = [r for r in rows if r['floor'] == fl and r['status'] == 'ok']
        if not s:
            print(f'{fl:>7.3f} {0:>4}   (all unhealthy)'); continue
        g = np.array([r['gap'] for r in s]); e = np.array([r['err'] for r in s])
        q = np.array([r['minq'] for r in s])
        print(f'{fl:>7.3f} {len(s):>4} {100*np.mean(g < GAP_TOL):>11.0f}% '
              f'{np.median(g):>11.4f} {np.median(e):>11.4f} {np.median(q):>12.4f}')
    print('=' * 72)
    print('floor HELPS iff trustworthy% rises materially WITHOUT median err degrading much;')
    print('if both move together it is forbidding the designs that were the point.')
    print(f'\nwrote {OUT}')


if __name__ == '__main__':
    main()
