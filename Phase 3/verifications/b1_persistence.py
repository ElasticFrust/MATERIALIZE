r"""B-1: is the bad state PERSISTENT or ONE-SHOT — and which test triggers it?

THE PROBLEM THIS SOLVES
-----------------------
Every campaign so far buys ~1 excursion per ~20 runs at ~30 min a run: a terrible sampling rate for
something documented as needing suite context (0 in 30 ISOLATED processes). `b1_reproduce.py`'s
`suitectx`/`threads` modes tried to cheat by REPLAYING the preceding tests and saw 0 hits in 360
probes -- replay is not sufficient context.

So this runs the REAL suite, in one process, and simply **probes [15]'s case 1 repeatedly along the
way** -- after every test, and many times over. A probe is a solver-vs-oracle comparison costing
seconds, against ~30 min for a whole suite run, so this raises the number of B-1 draws per unit
compute by orders of magnitude WITHOUT faking the context.

WHAT IT DISCRIMINATES (the open question nothing has answered)
--------------------------------------------------------------
- **PERSISTENT** state ("the process went bad and stayed bad"): once an excursion appears, the
  repeats that follow keep showing it. `n_bad` climbs and stays.
- **ONE-SHOT** transient: a single probe deviates and its immediate neighbours do not.
Either answer is progress, and the `after` column says WHICH test's execution preceded the change --
the first direct evidence about the trigger.

Every probe over `B1_DUMP_THRESHOLD` writes a full dump through the normal
`test_inverse_design._b1_dump_if_anomalous` path, so any hit arrives with the upstream `W`/`A3`
capture already attached. Per the analysis in `b1_dumps/B1_OVERNIGHT.md` §4c the next excursion is a
binary test: `W` off by ~30-40% => the metric solve; `A3` off by ~1% => the bare tensor.

Run:  C:\Users\doron\anaconda3\python.exe "Phase 3/verifications/b1_persistence.py" [reps] [--full]
        reps    probes after each test (default 8)
        --full  run the whole 16-test suite as context (default: the 13 tests preceding [15],
                which is what the documented "suite context" refers to)
Out:  b1_dumps/persistence_<UTC>.csv  -- one row per probe, flushed as it goes
      plus a normal b1_anomaly_*.json (+ _Gr.npz) for every probe that trips the threshold.
"""
import csv
import datetime as dt
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'Phase 3'))
sys.path.insert(0, HERE)

import torch                                                     # noqa: E402
import test_inverse_design as T                                  # noqa: E402
import _common as C                                              # noqa: E402
import physical_homog as PH                                      # noqa: E402
import sim_assembly as SA                                        # noqa: E402
from inverse_design import DesignProblem                         # noqa: E402

# [15]'s case 1 -- the ONLY case that has ever produced an excursion (3/3; 0/21 each on the other
# two). Rebuilt exactly as the test does.
PHI, PSI, ETA, SEED = 1.0, 1.0, 0.0, 0
BASELINE = 1.6e-12                          # the healthy solver-vs-oracle tensor agreement

# The suite's own order, copied from `test_inverse_design.__main__` (which builds it as a local, so
# it cannot be imported). The 13 entries before `test_homogenization` are exactly the "13 preceding
# tests" the B-1 notes refer to. Keep in sync with the runner.
SUITE_ORDER = ['test_round_trip', 'test_property_auxetic', 'test_property_E', 'test_local_region',
               'test_mixed_global_local', 'test_large_N_adjoint', 'test_open_domain',
               'test_directional', 'test_constrain_isotropic', 'test_strain_stress_equivalence',
               'test_strain_stress_autograd', 'test_strain_stress_design',
               'test_homogeneity_regularizer', 'test_homogenization', 'test_isotropization']


def free_of(geo):
    """Free DOFs, matching `test_homogenization`'s nested helper (pin the first node)."""
    return np.arange(2, 2 * len(geo['pts']))


def _case():
    geo = C.make_lattice(PHI, PSI, half=6.0, eta=ETA, seed=SEED)
    rng = np.random.default_rng(SEED)
    k = 0.5 + rng.random(len(geo['bond_u']))
    geo['bond_k'] = k
    geo['tri_k'] = k[geo['tri_bond']]
    return geo, k


def probe(geo, k, c_phys, free):
    """One solver-vs-oracle comparison. Returns (vs, cap) with the upstream capture attached.

    `c_phys` is passed in because the ORACLE IS BIT-EXACT across runs (measured: max|dP| = 0 over
    four dumps and two commits), so recomputing it every probe would only buy cost. The solver side
    is rebuilt from scratch each time -- that is the side under test."""
    prob = DesignProblem.from_geo(geo)
    with T._B1Capture() as cap:
        cs = C.solver_region_C6(prob, torch.as_tensor(k))
    c_solver = np.array([[cs[0], cs[2], cs[1]], [cs[2], cs[5], cs[4]], [cs[1], cs[4], cs[3]]])
    vs = float(np.abs(c_solver - c_phys).max() / np.abs(c_phys).max())
    return vs, c_solver, cap


def main():
    reps = int(sys.argv[1]) if len(sys.argv) > 1 and not sys.argv[1].startswith('-') else 8
    full = '--full' in sys.argv

    torch.manual_seed(0)                     # the runner does this before the suite -- same context

    geo, k = _case()
    free = free_of(geo)
    c_phys = PH.energy_C(geo, free, SA.assemble_K_faff)          # independent oracle, computed ONCE

    tests = [getattr(T, n) for n in SUITE_ORDER]
    if not full:                             # the 13 tests PRECEDING [15] -- the documented context
        tests = tests[:SUITE_ORDER.index('test_homogenization')]

    os.makedirs(T.B1_DUMP_DIR, exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    path = os.path.join(T.B1_DUMP_DIR, f'persistence_{stamp}.csv')
    print(f'B-1 persistence probe: {len(tests)} context tests x {reps} probes each '
          f'(+{reps} before any test)', flush=True)
    print(f'  csv: {path}', flush=True)

    n_bad = 0
    with open(path, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['i', 'after', 'rep', 'vs', 'ratio_to_baseline', 'secs'])
        fh.flush()
        i = 0

        def burst(after):
            nonlocal i, n_bad
            for r in range(reps):
                t0 = time.time()
                vs, c_solver, cap = probe(geo, k, c_phys, free)
                i += 1
                bad = vs > T.B1_DUMP_THRESHOLD
                n_bad += bad
                if bad:                       # normal dump path => W/A3 capture comes along
                    T._b1_dump_if_anomalous(PHI, PSI, ETA, SEED, vs, c_solver, c_phys, cap)
                w.writerow([i, after, r, f'{vs:.6e}', f'{vs / BASELINE:.3f}',
                            f'{time.time() - t0:.2f}'])
                fh.flush()
                os.fsync(fh.fileno())
                if bad:
                    print(f'  [{i:4d}] after={after:28} rep={r} vs={vs:.3e}  '
                          f'{vs / BASELINE:.1f}x baseline   <<< EXCURSION', flush=True)

        burst('<none: fresh process>')
        for t in tests:
            try:
                t()
            except Exception as e:                               # noqa: BLE001 — context, not a gate
                print(f'  (context test {t.__name__} raised {type(e).__name__}: {e})', flush=True)
            burst(t.__name__)
            print(f'  ...after {t.__name__:32} probes={i} excursions={n_bad}', flush=True)

    print(f'\n{n_bad} excursion(s) in {i} probes.', flush=True)
    if n_bad:
        print('  -> read the `after` column: the burst where vs first departs from ~1.0x baseline '
              'names the test whose execution preceded the change, and whether it PERSISTS.',
              flush=True)
    else:
        print('  -> no excursion. Probing is ~100x cheaper per draw than a suite run, so keep '
              'raising `reps` before concluding the context is insufficient.', flush=True)


if __name__ == '__main__':
    main()
