r"""B-1: measure the GUARD'S REPAIR RATE over a large, honest denominator.

WHY
---
"~1 in 700 solves" came from a single observation (`b1_persistence.py`, 700 probes, 1 excursion).
Now that `_woodbury_solve_aw` carries the two-stage guard (audit B-1) and emits a `RuntimeWarning`
whenever it repairs, the rate is directly countable: run the real suite -- whose `optimize()` loops
issue thousands of intrinsic solves -- and divide repairs by solves.

Counting only REPAIRS, not stage-1 triggers: a trigger that stage 2 clears is harmless by design,
and recomputing stage 1 here would double the cost of every solve.

Earlier warning counts in this session are NOT usable for a rate -- they were inflated by two
buggy stage-2 normalisations (dividing by |Lam| and by |W0|, both of which blow up where W ~ 0).
This is the first clean measurement.

Reports a Wilson 95 % interval, because at these rates the naive p +/- 1.96*sqrt(p(1-p)/n) is wrong
(and gives a zero-width interval if no repair is seen).

Run:  C:\Users\doron\anaconda3\python.exe "Phase 3/verifications/b1_rate.py" [--probes N]
        --probes N   extra bare probes after the suite (default 0; the suite alone is thousands)
Out:  stdout only -- append the numbers to b1_dumps/B1_OVERNIGHT.md.
"""
import math
import os
import sys
import time
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'Phase 3'))
sys.path.insert(0, HERE)

import numpy as np                                               # noqa: E402
import _common as C                          # MUST precede the solver imports: it wires sys.path
import torch                                                     # noqa: E402
import forward_solver_torch as FST                               # noqa: E402
import test_inverse_design as T                                  # noqa: E402

SUITE_ORDER = ['test_round_trip', 'test_property_auxetic', 'test_property_E', 'test_local_region',
               'test_mixed_global_local', 'test_large_N_adjoint', 'test_open_domain',
               'test_directional', 'test_constrain_isotropic', 'test_strain_stress_equivalence',
               'test_strain_stress_autograd', 'test_strain_stress_design',
               'test_homogeneity_regularizer', 'test_homogenization', 'test_isotropization']

count = {'solves': 0, 'repairs': 0, 'magnitudes': []}


def wilson(k, n, z=1.96):
    """95 % Wilson interval -- correct at k = 0 and at tiny p, unlike the normal approximation."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def main():
    probes = 0
    if '--probes' in sys.argv:
        probes = int(sys.argv[sys.argv.index('--probes') + 1])

    real = FST._woodbury_solve_aw

    def counted(A3, w, J3=None):
        if J3 is not None and J3.shape[0] > 0:                   # only guarded solves count
            count['solves'] += 1
        return real(A3, w, J3)

    FST._woodbury_solve_aw = counted

    def hook(message, category, filename, lineno, file=None, line=None):
        text = str(message)
        if 'audit B-1' in text and 'repaired' in text:
            count['repairs'] += 1
            try:                                                 # pull the reported magnitude
                count['magnitudes'].append(float(text.split('by ')[1].split()[0]))
            except (IndexError, ValueError):
                pass

    warnings.simplefilter('always')
    warnings.showwarning = hook

    torch.manual_seed(0)                                         # the suite's own context
    t0 = time.time()
    for name in SUITE_ORDER:
        try:
            getattr(T, name)()
        except Exception as e:                                   # noqa: BLE001 — context, not a gate
            print(f'  ({name} raised {type(e).__name__}: {e})', flush=True)
        print(f'  after {name:34} solves={count["solves"]:7d} repairs={count["repairs"]}',
              flush=True)

    if probes:
        geo = C.make_lattice(1.0, 1.0, half=6.0, eta=0.0, seed=0)
        rng = np.random.default_rng(0)
        k = 0.5 + rng.random(len(geo['bond_u']))
        geo['bond_k'] = k; geo['tri_k'] = k[geo['tri_bond']]
        from inverse_design import DesignProblem
        for _ in range(probes):
            p = DesignProblem.from_geo(geo)
            with torch.no_grad():
                p.solver.forward(torch.as_tensor(k)[p.tri_bond], rest_lengths=p.rl_ref,
                                 method='intrinsic', physical_units=True)
        print(f'  after {probes} bare probes{"":16} solves={count["solves"]:7d} '
              f'repairs={count["repairs"]}', flush=True)

    n, r = count['solves'], count['repairs']
    lo, hi = wilson(r, n)
    print(f'\nB-1 guard repair rate over {time.time()-t0:.0f} s')
    print(f'  guarded intrinsic solves : {n}')
    print(f'  repairs                  : {r}')
    print(f'  rate                     : {r/n if n else 0:.3e}   '
          f'(1 in {n/r:.0f})' if r else f'  rate                     : 0 observed')
    print(f'  95 % Wilson interval     : [{lo:.3e}, {hi:.3e}]  '
          f'= 1 in [{1/hi if hi else float("inf"):.0f}, '
          f'{1/lo if lo else float("inf"):.0f}]')
    if count['magnitudes']:
        m = np.array(count['magnitudes'])
        print(f'  repaired |dW| relative   : min {m.min():.3e}  median {np.median(m):.3e}  '
              f'max {m.max():.3e}')


if __name__ == '__main__':
    main()
