r"""Gate the A0.4 TRACE HOOK in `optimize` — the edit-policy's demonstration recorder.

`optimize` is verified-truth code, so the hook is additive and default-off. What is checked:

  [1] `optimize` is REPEATABLE within a process AFTER the first call. Measured while building this
      gate: run0 differs from run1 by ~1.3e-03 in `k`, while run1 == run2 EXACTLY — a first-call
      effect (warm-up / lazy init), not drift and not noise.
  [2] tracing performs ZERO solver calls while the optimiser runs, so it cannot perturb it. This
      targets the MECHANISM because the OUTCOME cannot be used: two identical untraced calls diverge
      by 3.5e-02 in `k` at n_iter=40 (bistable; any last-bit difference flips the basin).
  [3] a row is one CLOSURE EVALUATION, so `len(trace) == len(history)`. Consecutive rows may be
      line-search siblings at one iterate; the guard against near-duplicate leakage is the split by
      `traj_id` (every row of a run on one side), not the row granularity.
  [4] the row carries what the policy needs, and `grad_raw` is the SOLVER'S gradient at that
      iterate: `dL/dk = grad_raw / sigmoid(BETA*raw)`, checked against finite differences.
  [5] the bulk label is the UNWEIGHTED mean of the per-triangle field (`CLAUDE.md` §3), so the two
      stored quantities cannot drift apart.

Run:
    python "Phase 3/verifications/test_inverse_design_trace.py"
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, HERE)
import _common as C                                                       # noqa: E402,F401
from inverse_design import DesignProblem, Objective, optimize, _loss, BETA  # noqa: E402

torch.set_default_dtype(torch.float64)


def _problem(N=8, eta=0.2, seed=3):
    prob = DesignProblem.periodic(N=N, eta=eta, seed=seed)
    objs = [Objective('nu', target=-0.10, weight=1.0)]
    return prob, objs


def test_repeatable_after_first_call():
    """[1] Repeatable within a process AFTER the first call.

    Asserts the shape the code actually has rather than a determinism it does not provide."""
    prob, objs = _problem()
    out = [optimize(prob, objs, mode='k', n_iter=25, n_restarts=1, seed=0, reg=0.02, verbose=False)
           for _ in range(3)]
    d10 = float(np.abs(out[1]['k'].numpy() - out[0]['k'].numpy()).max())
    d21 = float(np.abs(out[2]['k'].numpy() - out[1]['k'].numpy()).max())
    ok = d21 == 0.0
    print('[1] repeatable after the first call: run2 vs run1 max|dk| = %.3e (must be 0); '
          'run1 vs run0 = %.3e (first-call effect, informational)  %s'
          % (d21, d10, 'OK' if ok else 'FAIL'))
    return 0 if ok else 1


def test_tracing_adds_no_solver_call_during_the_optimisation():
    """[2] Tracing performs ZERO solver calls while the optimiser is running.

    This targets the MECHANISM rather than the outcome, and it does so because the outcome cannot be
    used. `optimize` is not bit-reproducible: two IDENTICAL UNTRACED calls on this problem at
    n_iter=40 diverge by **3.5e-02** in `k` — even in a fresh subprocess — because the optimisation
    is bistable there and any last-bit difference (a B-1 guard repair, thread scheduling) flips the
    basin. Comparing final designs would therefore measure basin selection, and no tolerance can fix
    that: the effect is 3.5e-02, seven orders above anything tracing could plausibly do.

    So instead: count `prob.forward` calls. If tracing inserts none into the optimiser's own call
    sequence, it cannot perturb the solver's state, and the recorded responses are computed strictly
    afterwards by `_trace_fill_response` — exactly one per row. That is exact, deterministic, and it
    is the property the design actually claims."""
    prob, objs = _problem()
    real = prob.forward
    n = {'calls': 0}

    def counted(*a, **kw):
        n['calls'] += 1
        return real(*a, **kw)

    prob.forward = counted
    try:
        n['calls'] = 0
        base = optimize(prob, objs, mode='k', n_iter=20, n_restarts=1, seed=0, reg=0.02,
                        verbose=False)
        untraced_calls, untraced_evals = n['calls'], len(base['history'])

        n['calls'] = 0
        tr = []
        got = optimize(prob, objs, mode='k', n_iter=20, n_restarts=1, seed=0, reg=0.02,
                       verbose=False, trace=tr)
        traced_calls, traced_evals = n['calls'], len(got['history'])
    finally:
        prob.forward = real

    # untraced: one forward per closure call, plus the single final re-evaluation in `optimize`
    per_eval_untraced = untraced_calls - untraced_evals
    # traced: the same, plus EXACTLY one extra per recorded row, all in the post-pass
    overhead = traced_calls - traced_evals - per_eval_untraced
    ok = (traced_evals == len(tr) and overhead == len(tr)
          and per_eval_untraced == traced_calls - traced_evals - len(tr))
    print('[2] solver calls: untraced %d for %d evals; traced %d for %d evals + %d rows. '
          'Tracing overhead = %d == rows, all AFTER the optimisation  %s'
          % (untraced_calls, untraced_evals, traced_calls, traced_evals, len(tr), overhead,
             'OK' if ok else 'FAIL'))
    return 0 if ok else 1


def test_one_row_per_evaluation():
    """[3] One row per CLOSURE EVALUATION, steps 0..n-1."""
    prob, objs = _problem()
    tr = []
    res = optimize(prob, objs, mode='k', n_iter=30, n_restarts=1, seed=0, reg=0.02, verbose=False,
                   trace=tr)
    n_eval, n_row = len(res['history']), len(tr)
    steps = [r['step'] for r in tr]
    ok = n_row == n_eval and steps == list(range(n_row)) and n_row > 1
    print('[3] %d rows == %d closure evaluations, steps 0..n-1  %s'
          % (n_row, n_eval, 'OK' if ok else 'FAIL'))
    return 0 if ok else 1


def test_row_content_and_gradient():
    """[4] The row holds the policy's (state, signal), and `grad_raw` is the SOLVER's gradient.

    Checked at the FIRST row, where the parametrisation is healthy. Measured across one trajectory,
    rel |FD − analytic| is 2.3e-09 at row 0, 1.2e-05 at row 3 and 5.6e-04 at row 6 — while `k_min`
    falls from 1.7e-01 to ~1e-07. It is the FINITE-DIFFERENCE REFERENCE that degrades as the
    optimiser drives bonds toward DEAD k (`CLAUDE.md` §3: `A(s)` loses rank there and the inverse is
    set by the regulariser), not the recorded gradient. `h = 1e-5` for the same reason — smaller
    steps are worse here, the signature of an ill-conditioned `f` rather than of a bug.

    That `k_min` reaches ~1e-07 within six iterations at `reg = 0.02` is itself worth knowing: the
    demonstration trajectories WILL contain near-dead-k states."""
    prob, objs = _problem()
    tr = []
    optimize(prob, objs, mode='k', n_iter=6, n_restarts=1, seed=0, reg=0.02, verbose=False, trace=tr)
    need = ('k', 'C6', 'C6_per', 'loss', 'raw_k', 'grad_k', 'w_max', 'k_min', 'seed', 'threads')
    missing = [x for x in need if x not in tr[-1]]

    r = tr[0]
    raw = torch.as_tensor(r['raw_k'])
    dk = r['grad_k'] / torch.sigmoid(BETA * raw).numpy()
    rng = np.random.default_rng(0)
    k0 = torch.as_tensor(r['k'])
    h = 1e-5 * float(k0.mean())
    worst = 0.0
    for _ in range(4):
        d = rng.normal(size=k0.shape); d /= np.linalg.norm(d)
        with torch.no_grad():
            fp = float(_loss(prob, objs, k0 + h * torch.as_tensor(d), None, 0.02))
            fm = float(_loss(prob, objs, k0 - h * torch.as_tensor(d), None, 0.02))
        fd = (fp - fm) / (2 * h)
        an = float((dk * d).sum())
        worst = max(worst, abs(fd - an) / max(abs(fd), abs(an), 1e-30))
    ok = not missing and worst < 1e-7
    print('[4] row keys complete (%s); dL/dk from grad_raw vs FD at row 0: worst rel %.3e; '
          'k_min over the run %.2e -> %.2e  %s'
          % ('all present' if not missing else 'MISSING ' + ','.join(missing), worst,
             tr[0]['k_min'], tr[-1]['k_min'], 'OK' if ok else 'FAIL'))
    return 0 if ok else 1


def test_bulk_is_mean_of_per_triangle():
    """[5] `C6` must be the UNWEIGHTED mean of `C6_per` — the settled homogenisation, so the stored
    bulk label can never drift from the field it is derived from (an area weight here would bias ν
    on unequal-area meshes)."""
    prob, objs = _problem()
    tr = []
    optimize(prob, objs, mode='k', n_iter=4, n_restarts=1, seed=0, reg=0.02, verbose=False, trace=tr)
    worst = max(float(np.abs(r['C6'] - r['C6_per'].mean(0)).max()
                      / max(np.abs(r['C6']).max(), 1e-300)) for r in tr)
    ok = worst < 1e-10
    print('[5] C6 == unweighted mean of C6_per over %d rows: worst rel %.3e  %s'
          % (len(tr), worst, 'OK' if ok else 'FAIL'))
    return 0 if ok else 1


def main():
    bad = 0
    print('inverse_design TRACE-HOOK tests')
    for fn in (test_repeatable_after_first_call,
               test_tracing_adds_no_solver_call_during_the_optimisation,
               test_one_row_per_evaluation, test_row_content_and_gradient,
               test_bulk_is_mean_of_per_triangle):
        bad += fn()
    print('\n%s' % ('ALL PASSED' if bad == 0 else '%d TEST(S) FAILED' % bad))
    return 1 if bad else 0


if __name__ == '__main__':
    raise SystemExit(main())
