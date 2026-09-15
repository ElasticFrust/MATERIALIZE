r"""A0.3 -- design THROUGH the frozen surrogate, and score it against M1 at a matched budget.

WHY THIS AND NOT ANOTHER SCORE.  A0.2 measured `cos(grad L_solver, grad L_gnn)` and found 0.950 (k) /
0.915 (positions) -- but also that the cosine COLLAPSES in the near-mechanism tail (0.04 at
`max|W| > 100`).  A cosine is a statement about one point; descent is a statement about a trajectory,
and descent does not sample the space uniformly -- it actively walks toward wherever the surrogate
says the loss is low, which is exactly where a surrogate is most likely to be wrong.  So this is an
ADVERSARIAL test in a way random-sample scoring cannot be, and it is the last of A0's three gates.

THE CURRENCY IS EXACT-SOLVER CALLS.  The surrogate's claim is not "better designs", it is "the same
design for fewer exact solves" -- which is also what S4 is defined on.  So the headline is:

    **how many exact-solver calls does M1 need to reach the design the surrogate produces for FREE?**

    M1 arm  : L-BFGS through the EXACT solver. Every closure call is one solver call, so its
              best-so-far trace IS a quality-vs-solver-calls curve.
    GNN arm : the SAME L-BFGS through the FROZEN surrogate, costing ZERO solver calls; one call then
              scores the result. `--refine` further rounds each spend ONE call to accept or reject a
              perturbed re-optimisation, tracing the rest of the curve.

THE ONLY DIFFERENCE IS THE FORWARD MODEL -- same parametrisation (`raw` through softplus), same
optimiser (L-BFGS + strong Wolfe), same init, same `reg`.  The FIRST version of this probe got that
wrong: M1 ran L-BFGS on `raw` while the GNN arm ran Adam on `k` with a hard clamp, so it compared two
OPTIMISERS and would have charged the difference to the surrogate.  It also could not match budgets,
because strong Wolfe makes several closure calls per iteration (M1 drew 29-32 calls against the GNN
arm's 15-25).  Asking "how many calls to match" removes the need to match budgets at all.

SCOPE: the `k` channel.  M1's position channel is SPSA, a different and far more expensive baseline,
so that comparison is its own experiment; A0.2 already measured the position GRADIENT (median
cos 0.915).

SCORED ON BOTH the exact solver and the INDEPENDENT SIM -- scoring only on the solver would ask
whether descent minimised the thing it was pointed at (`CLAUDE.md` §3 tier (B)).

Run:
    python "Phase 5/verifications/m2_designer_probe.py" --n_targets 20 --n_iter 80 --refine 4
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'Phase 3', 'verifications'))
import _common as C                                                       # noqa: E402
from inverse_design import (DesignProblem, ANG, c6_to_nu_theta,           # noqa: E402
                            c6_to_E_theta, _softplus, _inv_softplus)
sys.path.insert(0, os.path.join(REPO, 'Phase 5'))
sys.path.insert(0, os.path.join(REPO, 'Phase 5', 'm2'))
import model_v3 as M3                                                     # noqa: E402
import train_v2 as T2                                                     # noqa: E402
import train_v3 as T3                                                     # noqa: E402
import evaluate_v2 as EV                                                  # noqa: E402
sys.path.insert(0, HERE)
from m2_gradient_fidelity import loss_from_c6, solver_c6, gnn_c6, geo_at  # noqa: E402

torch.set_default_dtype(torch.float64)
RESULTS = os.path.join(REPO, 'Phase 5', 'results', 'm2_s1')


# ------------------------------------------------------------------ scoring
def score(geo, k, nu_t, E_t):
    """Achieved-vs-target on the EXACT SOLVER and on the INDEPENDENT SIM.

    Two different code paths, per `CLAUDE.md` §3: the solver is what the design was pointed at, the
    sim is whether the network physically does it.  `sim_bulk_C6` routes to the independent virial,
    never `sim_region_C6` (which pushes the sim's relaxation back through the solver's own
    contraction and would be self-verification)."""
    out = {}
    prob = DesignProblem.from_geo(geo)
    with torch.no_grad():
        c6s = solver_c6(prob, torch.as_tensor(np.asarray(k, float)))
        out['err_solver'] = float(loss_from_c6(c6s, nu_t, E_t))
        out['nu_solver'] = float(c6_to_nu_theta(c6s, ANG).mean())
    try:
        g2 = dict(geo); C.apply_k_to_geo(g2, np.asarray(k, float))
        c6p = torch.as_tensor(np.asarray(C.sim_bulk_C6(g2), float))
        out['err_sim'] = float(loss_from_c6(c6p, nu_t, E_t))
        out['nu_sim'] = float(c6_to_nu_theta(c6p, ANG).mean())
    except Exception as e:                                                # noqa: BLE001
        out['err_sim'] = float('nan'); out['nu_sim'] = float('nan')
        out['sim_error'] = type(e).__name__
    return out


# ------------------------------------------------------------------ the two arms
#
# THE ONLY DIFFERENCE BETWEEN THEM IS THE FORWARD MODEL.  Same parametrisation (`raw`, squashed
# through softplus so k > 0 without a hard constraint -- `inverse_design._softplus`), same optimiser
# (L-BFGS + strong Wolfe, the settings `optimize` uses), same initialisation, same `reg`.
#
# The first version of this probe got that wrong: M1 ran L-BFGS on `raw` while the GNN arm ran Adam
# on `k` with a hard clamp, so it compared two OPTIMISERS and attributed the difference to the
# surrogate.  It also could not match budgets, because strong Wolfe makes several closure calls per
# L-BFGS iteration (M1 drew 29-32 solver calls where the GNN arm drew 15-25).
#
# Both defects disappear with the right question, which is also the one S4 is defined on:
# **how many EXACT-SOLVER CALLS does M1 need to reach the design the surrogate produces for free?**
# No budget matching is required -- the curve answers it directly.

def _lbfgs_design(forward_c6, raw0, nu_t, E_t, reg, n_iter, trace=None):
    """L-BFGS + strong Wolfe on `raw`, against whatever forward model is passed.

    `trace`, if given, receives the PURE objective (no reg) at every closure call, so the caller can
    build a best-so-far curve in units of forward-model evaluations."""
    raw = raw0.clone().requires_grad_(True)
    opt = torch.optim.LBFGS([raw], lr=1.0, max_iter=n_iter,
                            line_search_fn='strong_wolfe', tolerance_grad=1e-12)

    def closure():
        opt.zero_grad()
        k = _softplus(raw)
        pure = loss_from_c6(forward_c6(k), nu_t, E_t)
        if trace is not None:
            trace.append(float(pure))
        loss = pure + (reg * ((k - k.mean()) ** 2).mean() if reg > 0 else 0.0)
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        return _softplus(raw).detach()


def arm_m1(geo, raw0, nu_t, E_t, n_iter, reg):
    """M1: L-BFGS through the EXACT solver. EVERY closure call is one solver call."""
    prob = DesignProblem.from_geo(geo)
    trace = []
    k = _lbfgs_design(lambda kk: solver_c6(prob, kk), raw0, nu_t, E_t, reg, n_iter, trace)
    return k.numpy(), np.minimum.accumulate(np.array(trace, float)), len(trace)


def arm_gnn(net, g, geo, raw0, nu_t, E_t, n_iter, reg, refine):
    """The surrogate arm: L-BFGS through the FROZEN GNN, costing ZERO solver calls.

    `k_free` is the design the surrogate produces with no solver contact at all; one solver call
    scores it. Each of `refine` further rounds re-optimises from a perturbed restart and spends ONE
    solver call to accept or reject it, so the arm traces out quality against solver calls.

    SCOPE: `k` only. M1's position channel is SPSA (`positions.design_with_positions`), a different
    and far more expensive baseline, so comparing the surrogate's position descent against it is its
    own experiment -- see the limitations in GRADIENT_FIDELITY.md. A0.2 already measured the position
    GRADIENT at median cos 0.915."""
    prob = DesignProblem.from_geo(geo)
    fwd = lambda kk: gnn_c6(net, g, k=kk)                                 # noqa: E731

    k_free = _lbfgs_design(fwd, raw0, nu_t, E_t, reg, n_iter)
    with torch.no_grad():
        best_true = float(loss_from_c6(solver_c6(prob, k_free), nu_t, E_t))
    best_k, calls, rejects = k_free, 1, 0
    curve = [best_true]

    raw = _inv_softplus(best_k)
    for j in range(refine):
        gen = torch.Generator().manual_seed(1000 + j)
        cand = _lbfgs_design(fwd, raw + 0.05 * torch.randn(raw.shape, generator=gen),
                             nu_t, E_t, reg, max(n_iter // 4, 5))
        with torch.no_grad():
            true = float(loss_from_c6(solver_c6(prob, cand), nu_t, E_t))
        calls += 1
        if true < best_true:
            best_true, best_k, raw = true, cand, _inv_softplus(cand)
        else:
            rejects += 1
        curve.append(best_true)
    return best_k.numpy(), k_free.numpy(), calls, rejects / max(refine, 1), np.array(curve)


# ------------------------------------------------------------------ targets
def make_targets(n, rng):
    """Isotropic (nu, E-scale) targets spanning the MEASURED reach envelope.

    `results/reach_summary/`: k-design reaches [-0.823, +0.899] at zero contrast and the auxetic
    sweep [-0.601, +0.301]. Sampling [-0.45, +0.60] stays inside verified reach on both, so a miss
    is the optimiser's and not an unreachable request. E is requested as a FACTOR of the network's
    own base E, which keeps the target reachable on every topology."""
    nus = np.linspace(-0.45, 0.60, n)
    fE = rng.uniform(0.7, 1.3, n)
    return nus, fE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=os.path.join(
        REPO, 'Phase 5', 'm2', 'checkpoint_v3_res_bravais_w10_h1_L5_ns48_h64_e400_star_ms.pt'))
    ap.add_argument('--data', default=os.path.join(
        REPO, 'Phase 5', 'm2', 'data', 'dataset_fresh_s4321.npz'))
    ap.add_argument('--n_targets', type=int, default=20)
    ap.add_argument('--n_iter', type=int, default=80, help='L-BFGS iterations, BOTH arms')
    ap.add_argument('--refine', type=int, default=4,
                    help='surrogate trust-region rounds (ONE solver call each)')
    ap.add_argument('--reg', type=float, default=0.02, help='k-variance penalty (CLAUDE.md S3)')
    ap.add_argument('--max_nodes', type=int, default=120)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--tag', default='')
    a = ap.parse_args()

    print('checkpoint %s' % os.path.basename(a.ckpt))
    print('k channel. Both arms: L-BFGS + strong Wolfe on softplus(raw), same init, same reg=%.3g.'
          % a.reg)
    print('ONLY the forward model differs. M1 pays one solver call per closure; the surrogate pays '
          'ZERO until it is scored.')
    net = M3.from_checkpoint(torch.load(a.ckpt, map_location='cpu', weights_only=False))
    pool = [g for g in T2.load(a.data)
            if g['sim_ok'] and not EV.has_self_loops(g) and len(g['pts']) <= a.max_nodes]
    print('%d usable networks (no self-loops, <= %d nodes)' % (len(pool), a.max_nodes))

    rng = np.random.default_rng(a.seed)
    nus, fEs = make_targets(a.n_targets, rng)
    picks = rng.choice(len(pool), a.n_targets, replace=False)

    rows, t0 = [], time.time()
    for i, (gi, nu0, fE) in enumerate(zip(picks, nus, fEs)):
        g = pool[int(gi)]
        geo = EV.geo_of(g)
        prob = DesignProblem.from_geo(geo)
        with torch.no_grad():
            base = solver_c6(prob, torch.as_tensor(np.asarray(g['k'], float)))
            E_base = c6_to_E_theta(base, ANG)
        nu_t = torch.full((len(ANG),), float(nu0), dtype=torch.float64)
        E_t = (fE * E_base).detach()

        # ONE init shared by both arms -- the only difference must be the forward model
        gen = torch.Generator().manual_seed(a.seed + i)
        nb = len(g['bond_u'])
        raw0 = _inv_softplus(torch.ones(nb)) + 0.3 * torch.randn(nb, generator=gen)

        k_m1, curve_m1, calls_m1 = arm_m1(geo, raw0, nu_t, E_t, a.n_iter, a.reg)
        k_gn, k_free, calls_gn, rej, _ = arm_gnn(net, g, geo, raw0, nu_t, E_t,
                                                 a.n_iter, a.reg, a.refine)

        s_m1, s_gn, s_fr = (score(geo, k, nu_t, E_t) for k in (k_m1, k_gn, k_free))
        s_00 = score(geo, np.asarray(g['k'], float), nu_t, E_t)
        hit = curve_m1 <= s_fr['err_solver']
        reach = int(np.argmax(hit) + 1) if hit.any() else -1
        rows.append(dict(family=str(g.get('family', '?')), n_tri=len(g['tri_bond']),
                         w_max=float(g.get('w_max', np.nan)), nu_target=float(nu0), fE=float(fE),
                         start=s_00, m1=s_m1, gnn=s_gn, gnn_free=s_fr,
                         calls_m1=calls_m1, calls_gnn=calls_gn, reject_rate=rej,
                         m1_calls_to_match_free=reach))
        print('  [%2d/%2d] %-11s nu*=%+.2f | start %.2e | M1 %.2e (%d calls) | GNNfree %.2e '
              '(0 calls) | GNN+%d %.2e | M1 needs %s to match GNNfree  (%.0fs)'
              % (i + 1, a.n_targets, rows[-1]['family'], nu0, s_00['err_solver'],
                 s_m1['err_solver'], calls_m1, s_fr['err_solver'], a.refine,
                 s_gn['err_solver'], (str(reach) if reach > 0 else '>%d' % calls_m1),
                 time.time() - t0))

    def col(key, sub):
        return np.array([r[key][sub] for r in rows], float)

    print('')
    print('=== A0.3: design through the frozen surrogate vs M1 (k channel) ===')
    for sub, lab in (('err_solver', 'SOLVER'), ('err_sim', 'INDEPENDENT SIM')):
        s0, m1, fr, gn = (col(k, sub) for k in ('start', 'm1', 'gnn_free', 'gnn'))
        ok = np.isfinite(m1) & np.isfinite(gn) & np.isfinite(fr) & np.isfinite(s0)
        print('')
        print('  achieved-vs-target loss on the %s  (n=%d)' % (lab, ok.sum()))
        print('    start, no design            median %.3e' % np.median(s0[ok]))
        print('    GNN free (0 solver calls)   median %.3e   ratio to M1 %.2f'
              % (np.median(fr[ok]), np.median(fr[ok] / np.maximum(m1[ok], 1e-300))))
        print('    GNN + refine                median %.3e   ratio to M1 %.2f'
              % (np.median(gn[ok]), np.median(gn[ok] / np.maximum(m1[ok], 1e-300))))
        print('    M1 (exact solver, full run) median %.3e' % np.median(m1[ok]))
        print('    GNN-free improves on the start on %.0f%% of targets; beats M1 on %.0f%%'
              % (100 * (fr[ok] < s0[ok]).mean(), 100 * (fr[ok] < m1[ok]).mean()))

    reach = np.array([r['m1_calls_to_match_free'] for r in rows], float)
    got = reach > 0
    cm1 = np.array([r['calls_m1'] for r in rows], float)
    print('')
    print('  THE S4 NUMBER -- exact-solver calls M1 needs to match the surrogate FREE design:')
    print('    matched on %d of %d targets; median %s calls (M1 spends %d in a full run)'
          % (int(got.sum()), len(rows),
             ('%.0f' % np.median(reach[got])) if got.any() else 'n/a', int(np.median(cm1))))
    if got.sum() < len(rows):
        print('    on %d target(s) M1 never matched it within its own run'
              % (len(rows) - int(got.sum())))
    rej = np.array([r['reject_rate'] for r in rows], float)
    print('  trust-region reject rate over the refine rounds: median %.0f%%' % (100 * np.median(rej)))

    # A RATIO IS MEANINGLESS WHEN THE REFERENCE APPROACHES ZERO. M1 can reach ~1e-06 on an easy
    # target, which sent a pilot bin to a "ratio" of 48509 -- an artefact of the denominator, not a
    # surrogate failure. Floor it by the median M1 loss (the same idea as the eps_E floor in
    # evaluate_v2, and as CLAUDE.md's eps_nu), and print the ABSOLUTE medians alongside so the
    # ratio can never be read on its own.
    wm = np.array([r['w_max'] for r in rows], float)
    fr_a, m1_a = col('gnn_free', 'err_solver'), col('m1', 'err_solver')
    floor = float(np.median(m1_a[np.isfinite(m1_a)]))
    gr = fr_a / (m1_a + floor)
    print('')
    print('  by max|W| (binned medians; A0.2 found the GRADIENT degrades here). Ratio floored by the'
          ' median M1 loss %.2e:' % floor)
    print('    %-8s %3s %12s %12s %8s' % ('max|W|', 'n', 'GNNfree', 'M1', 'ratio'))
    for lo, hi, nm in ((0, 3, '<3'), (3, 10, '3-10'), (10, 100, '10-100'), (100, 1e9, '>100')):
        m = (wm >= lo) & (wm < hi) & np.isfinite(gr)
        if m.sum():
            print('    %-8s %3d %12.3e %12.3e %8.2f'
                  % (nm, int(m.sum()), np.median(fr_a[m]), np.median(m1_a[m]), np.median(gr[m])))

    os.makedirs(RESULTS, exist_ok=True)
    dst = os.path.join(RESULTS, 'designer_probe%s.json' % (('_' + a.tag) if a.tag else ''))
    with open(dst, 'w') as f:
        json.dump(dict(config=vars(a), rows=rows), f, indent=1)
    print('')
    print('-> %s' % dst)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
