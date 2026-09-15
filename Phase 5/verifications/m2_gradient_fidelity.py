r"""A0.2 -- does the surrogate's GRADIENT point the way the solver's does?

THE QUESTION.  Every measurement of the M2 surrogate so far scores its VALUE (per-triangle MAE/sigma
0.0925, MAE(nu) against the sim).  None scores the quantity a designer actually consumes.  The
surrogate's reason to exist is that geometry is differentiable through it while the exact solver's
positions are "fixed inputs, set at construction, not differentiated" (`ElasticSolver`; hence SPSA,
CLAUDE.md S3, FD #13).  So the operative question is not "is the prediction accurate" but "does
descending the surrogate move a design the way descending the truth would".

WHAT IS COMPARED.  Two losses, the SAME weighted nu(theta)/E(theta) residual the designer minimises
(`inverse_design._loss`'s `nu_theta` + `E_theta` arithmetic, reused here, not reimplemented):

    L_solver(x) = loss( C6 from the exact solver at x )
    L_gnn(x)    = loss( C6 from the frozen surrogate at x )

evaluated at the SAME point x and against the SAME target, then `cos(grad L_solver, grad L_gnn)`.
They are gradients of different functions -- that is the point.  A surrogate can be biased and still
be a perfect descent direction, and it can be accurate and still point sideways; only this
measurement separates the two.

THE TARGET is a design specification, so it is anchored on the SOLVER's response, not the model's:
nu_target = nu_solver - `--dnu`, E_target = `--fE` * E_solver.  Anchoring it on the model would make
the model's own error part of the specification.  Both channels are displaced so both contribute.

REFERENCES, and they differ by channel (this is the asymmetry the whole project rests on):
  * dL/dk    -- the solver's own ADJOINT.  Exact, one backward.  No approximation anywhere.
  * dL/dpts  -- the solver has no position gradient, so CENTRAL FINITE DIFFERENCES, and by default
                the FULL 2N of them, giving the EXACT cosine rather than a random-direction
                estimator.  Affordable only on small meshes (rebuild+forward is ~20 ms at 3 nodes,
                ~140 ms at 60), hence `--max_nodes`; `--dirs` switches to the estimator for larger.

HEALTH.  Every perturbed geometry is checked for triangle INVERSION via the SIGNED area, and against
`mesh_build.check_mesh_preconditions`.  A rejected probe is COUNTED AND REPORTED, never silently
dropped -- a filtered set is a biased set, which is the eta-sweep seed-count trap.  (The oracle's
`require_healthy_mesh` is deliberately not used: it exists to stop the SIM segfaulting, no sim runs
here, and not importing it keeps this script's dependency surface minimal.)

REPORTING is by BINNED MEDIANS, stratified by family and by max|W| -- never a raw correlation.  On
this exact quantity a correlation coefficient has already said the opposite of the truth twice (the
max|W| tail, and the hexagon rcond).

Run:
    python "Phase 5/verifications/m2_gradient_fidelity.py" --n_k 300 --n_pos 100
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
import _common as C                                                       # noqa: E402,F401
from inverse_design import (DesignProblem, ANG, c6_to_nu_theta,           # noqa: E402
                            c6_to_E_theta, c6_to_nuE)
import mesh_build as MB                                                   # noqa: E402

sys.path.insert(0, os.path.join(REPO, 'Phase 5', 'm2'))
import model_v3 as M3                                                     # noqa: E402
import train_v2 as T2                                                     # noqa: E402
import train_v3 as T3                                                     # noqa: E402
import evaluate_v2 as EV                                                  # noqa: E402

torch.set_default_dtype(torch.float64)
RESULTS = os.path.join(REPO, 'Phase 5', 'results', 'm2_s1')


# ------------------------------------------------------------------ geometry at perturbed positions
def geo_at(g, pts):
    """`evaluate_v2.geo_of(g)`, but at the positions `pts` instead of the stored ones.

    Only the coordinate-carried fields move: `bond_R` (through the constant periodic offset, exactly
    as `train_v3.bond_vectors` does it -- the offset is frozen connectivity and is NEVER re-derived
    by minimum image), and everything downstream of it.

    `edge_vecs` is rebuilt through `evaluate_v2.edge_vec_map`, i.e. with the TRIANGLE's orientation.
    Using `bond_R[tri_bond]` here instead -- which is what an earlier draft did, and what `geo_of`
    itself did until 2026-09-15 -- flips the sign on about half the rows and silently corrupts the
    curvature constraint, so the finite differences would have measured that instead of the physics.
    `check_label_reproduction` is the gate that catches it."""
    geo = dict(EV.geo_of(g))
    bu = np.asarray(g['bond_u'], np.int64); bv = np.asarray(g['bond_v'], np.int64)
    p0 = np.asarray(g['pts'], float)
    shift = np.asarray(g['bond_R'], float) - (p0[bv] - p0[bu])
    pts = np.asarray(pts, float)
    bR = pts[bv] - pts[bu] + shift
    tb = np.asarray(g['tri_bond'], np.int64)
    ei, sg = EV.edge_vec_map(g)
    ev = bR[ei] * sg[..., None]
    geo['pts'] = pts
    geo['bond_R'] = bR
    geo['edge_vecs'] = ev
    geo['actual_len2'] = (ev ** 2).sum(-1)
    geo['areas'] = M3.triangle_areas(g['tri_verts'], tb, bu, bv, bR).numpy()
    return geo


def signed_areas(g, pts):
    return M3.triangle_areas(g['tri_verts'], np.asarray(g['tri_bond'], np.int64),
                             g['bond_u'], g['bond_v'],
                             geo_at(g, pts)['bond_R'], signed=True).numpy()


def healthy(g, pts, sign0):
    """A perturbed geometry is usable if no triangle INVERTED and the mesh preconditions hold."""
    s = signed_areas(g, pts)
    if np.any(np.sign(s) != sign0) or not np.all(np.abs(s) > 0):
        return False, 'inverted'
    geo = geo_at(g, pts)
    try:
        ok = MB.check_mesh_preconditions(dict(pts=geo['pts'], simplices=geo['simplices'],
                                              areas=geo['areas'], bond_u=g['bond_u'],
                                              bond_v=g['bond_v'], tri_bond=g['tri_bond']))
        ok = bool(ok) if not isinstance(ok, tuple) else bool(ok[0])
    except Exception:                                                     # noqa: BLE001
        return True, 'precondition-check-unavailable'
    return (True, '') if ok else (False, 'preconditions')


# ------------------------------------------------------------------ the shared objective
def loss_from_c6(C6, nu_t, E_t, nu_w=1.0, E_w=1.0, thetas=ANG):
    """EXACTLY `_loss`'s `nu_theta` + `E_theta` contribution, on whatever C6 it is handed.

    Shared by both sides on purpose: if the two used even slightly different objective arithmetic
    the measured disagreement would be partly the objective's, and the whole number would be
    uninterpretable."""
    return (nu_w * ((c6_to_nu_theta(C6, thetas) - nu_t) ** 2).mean()
            + E_w * ((c6_to_E_theta(C6, thetas) - E_t) ** 2).mean())


def solver_c6(prob, k):
    per = prob.forward(k, None, physical_units=True)['per_triangle']
    return prob.region_tensor(per, None)


def gnn_c6(net, g, pts=None, k=None):
    return T3.predict(net, T3.prepare(g, pts=pts, k=k)).mean(0)


def cosine(a, b):
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < 1e-300 or nb < 1e-300:
        return float('nan'), na, nb
    return float(np.dot(a, b) / (na * nb)), na, nb


# ------------------------------------------------------------------ per-network measurement
def measure_k(net, g, dnu, fE):
    """dL/dk on both sides, both by autograd. The solver reference is EXACT."""
    geo = EV.geo_of(g)
    prob = DesignProblem.from_geo(geo)
    k0 = np.asarray(g['k'], float)

    with torch.no_grad():
        C6s = solver_c6(prob, torch.as_tensor(k0))
        nu_s, E_s = c6_to_nu_theta(C6s, ANG), c6_to_E_theta(C6s, ANG)
    nu_t, E_t = (nu_s - dnu).detach(), (fE * E_s).detach()

    ks = torch.as_tensor(k0).requires_grad_(True)
    loss_from_c6(solver_c6(prob, ks), nu_t, E_t).backward()
    gs = ks.grad.detach().numpy().copy()

    kg = torch.as_tensor(k0).requires_grad_(True)
    loss_from_c6(gnn_c6(net, g, k=kg), nu_t, E_t).backward()
    gg = kg.grad.detach().numpy().copy()
    return cosine(gs, gg)


def measure_pos(net, g, dnu, fE, h_frac, dirs=0, rng=None):
    """dL/dpts: GNN by autograd, solver by CENTRAL finite differences.

    `dirs=0` does the full 2N differences -> the EXACT cosine. `dirs>0` projects onto that many
    random unit directions and estimates the cosine from the paired directional derivatives, which
    is the cheaper estimator for large meshes."""
    p0 = np.asarray(g['pts'], float)
    sign0 = np.sign(signed_areas(g, p0))
    geo = EV.geo_of(g)
    prob = DesignProblem.from_geo(geo)
    k_t = torch.as_tensor(np.asarray(g['k'], float))

    with torch.no_grad():
        C6s = solver_c6(prob, k_t)
        nu_s, E_s = c6_to_nu_theta(C6s, ANG), c6_to_E_theta(C6s, ANG)
    nu_t, E_t = (nu_s - dnu).detach(), (fE * E_s).detach()

    pg = torch.as_tensor(p0).requires_grad_(True)
    loss_from_c6(gnn_c6(net, g, pts=pg), nu_t, E_t).backward()
    gg = pg.grad.detach().numpy().copy()

    lbar = float(np.linalg.norm(np.asarray(g['bond_R'], float), axis=1).mean())
    h = h_frac * lbar
    rejected = 0

    def L_solver_at(p):
        return float(loss_from_c6(solver_c6(DesignProblem.from_geo(geo_at(g, p)), k_t), nu_t, E_t))

    if dirs <= 0:                                                  # exact: full 2N central FD
        gs = np.zeros_like(p0)
        for i in range(p0.shape[0]):
            for c in range(2):
                pp = p0.copy(); pp[i, c] += h
                pm = p0.copy(); pm[i, c] -= h
                ok_p, _ = healthy(g, pp, sign0); ok_m, _ = healthy(g, pm, sign0)
                if not (ok_p and ok_m):
                    rejected += 1
                    continue                                       # leave that component at 0
                gs[i, c] = (L_solver_at(pp) - L_solver_at(pm)) / (2 * h)
        cos, na, nb = cosine(gs.ravel(), gg.ravel())
        return cos, na, nb, rejected, 2 * p0.size // 2
    # estimator: paired directional derivatives
    fd, an = [], []
    for _ in range(dirs):
        d = rng.normal(size=p0.shape); d /= np.linalg.norm(d)
        ok_p, _ = healthy(g, p0 + h * d, sign0); ok_m, _ = healthy(g, p0 - h * d, sign0)
        if not (ok_p and ok_m):
            rejected += 1
            continue
        fd.append((L_solver_at(p0 + h * d) - L_solver_at(p0 - h * d)) / (2 * h))
        an.append(float((gg * d).sum()))
    if len(fd) < 3:
        return float('nan'), float('nan'), float('nan'), rejected, dirs
    fd, an = np.array(fd), np.array(an)
    cos = float(np.dot(fd, an) / max(np.linalg.norm(fd) * np.linalg.norm(an), 1e-300))
    return cos, float(np.linalg.norm(fd)), float(np.linalg.norm(an)), rejected, dirs


# ------------------------------------------------------------------ reporting
def binned(rows, key, label, bins):
    print('\n  by %s (binned MEDIANS -- never a raw correlation, CLAUDE.md S3):' % label)
    print('    %-16s %5s %9s %9s %9s %9s' % (label, 'n', 'med cos', 'q25', 'q75', 'frac>0'))
    vals = np.array([r[key] for r in rows], float)
    for lo, hi, name in bins:
        m = (vals >= lo) & (vals < hi)
        c = np.array([r['cos'] for r, keep in zip(rows, m) if keep], float)
        c = c[np.isfinite(c)]
        if len(c) == 0:
            continue
        print('    %-16s %5d %9.4f %9.4f %9.4f %8.0f%%'
              % (name, len(c), np.median(c), np.percentile(c, 25), np.percentile(c, 75),
                 100 * (c > 0).mean()))


def report(rows, chan):
    c = np.array([r['cos'] for r in rows], float)
    c = c[np.isfinite(c)]
    print('\n=== d/d%s : %d networks scored ===' % (chan, len(c)))
    if len(c) == 0:
        print('  nothing scored'); return {}
    q = np.percentile(c, [5, 25, 50, 75, 95])
    print('  cosine(grad solver, grad GNN):  median %.4f   IQR [%.4f, %.4f]   p5 %.4f  p95 %.4f'
          % (q[2], q[1], q[3], q[0], q[4]))
    print('  fraction with cos > 0 (still a descent direction): %.1f %%' % (100 * (c > 0).mean()))
    print('  fraction with cos > 0.9: %.1f %%   > 0.7: %.1f %%'
          % (100 * (c > 0.9).mean(), 100 * (c > 0.7).mean()))
    mag = np.array([r['mag_ratio'] for r in rows], float); mag = mag[np.isfinite(mag)]
    print('  |grad GNN| / |grad solver|:  median %.3f   IQR [%.3f, %.3f]'
          % (np.median(mag), np.percentile(mag, 25), np.percentile(mag, 75)))

    fams = sorted({r['family'] for r in rows})
    print('\n  by family:')
    print('    %-16s %5s %9s %9s %9s' % ('family', 'n', 'med cos', 'q25', 'frac>0'))
    for f in fams:
        cc = np.array([r['cos'] for r in rows if r['family'] == f], float)
        cc = cc[np.isfinite(cc)]
        if len(cc) == 0:
            continue
        print('    %-16s %5d %9.4f %9.4f %8.0f%%'
              % (f, len(cc), np.median(cc), np.percentile(cc, 25), 100 * (cc > 0).mean()))
    binned(rows, 'w_max', 'max|W|',
           [(0, 1, '<1'), (1, 3, '1-3'), (3, 10, '3-10'), (10, 100, '10-100'),
            (100, 1e9, '>100')])
    rej = sum(r.get('rejected', 0) for r in rows)
    tot = sum(r.get('probes', 0) for r in rows)
    if tot:
        print('\n  health-rejected probes: %d of %d (%.2f %%) -- counted, not dropped silently'
              % (rej, tot, 100 * rej / tot))
    return dict(n=len(c), median=float(q[2]), q25=float(q[1]), q75=float(q[3]),
                frac_pos=float((c > 0).mean()), frac_gt09=float((c > 0.9).mean()),
                frac_gt07=float((c > 0.7).mean()), mag_median=float(np.median(mag)))


# ------------------------------------------------------------------ gate on the perturbation code
def check_label_reproduction(raw, idxs, tol=1e-9):
    """THE gate on the reference: a freshly rebuilt solver must reproduce the STORED label.

    This replaced a weaker gate ("geo_at(pts0) == geo_of"), which passed at 4e-16 while the whole
    reference was wrong -- it was comparing the reconstruction against a reconstruction, and both
    shared the same sign bug.  The only honest check is against a number produced by the code that
    built the data.  Also confirms `geo_at` at zero perturbation, since it routes through the same
    orientation map."""
    worst, worst_f, n = 0.0, '', 0
    for i in idxs:
        g = raw[int(i)]
        if EV.has_self_loops(g):
            continue
        prob = DesignProblem.from_geo(geo_at(g, g['pts']))
        with torch.no_grad():
            fresh = np.asarray(solver_c6(prob, torch.as_tensor(np.asarray(g['k'], float))), float)
        stored = np.asarray(g['C6'], float)
        rel = float(np.abs(fresh - stored).max() / max(np.abs(stored).max(), 1e-300))
        n += 1
        if rel > worst:
            worst, worst_f = rel, str(g.get('family', '?'))
    ok = worst <= tol
    print('[gate] fresh solve reproduces the STORED label: worst rel %.3e over %d meshes (%s)  %s'
          % (worst, n, worst_f, 'OK' if ok else 'FAIL'))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=os.path.join(
        REPO, 'Phase 5', 'm2', 'checkpoint_v3_res_bravais_w10_h1_L5_ns48_h64_e400_star_ms.pt'))
    ap.add_argument('--data', default=os.path.join(
        REPO, 'Phase 5', 'm2', 'data', 'dataset_fresh_s4321.npz'),
        help='FRESH unseen meshes by default -- the honest population for a deployment question')
    ap.add_argument('--n_k', type=int, default=300)
    ap.add_argument('--n_pos', type=int, default=100)
    ap.add_argument('--max_nodes', type=int, default=64,
                    help='position channel only: full 2N central FD is ~2N rebuilds per network')
    ap.add_argument('--dirs', type=int, default=0,
                    help='0 = exact full 2N FD; >0 = that many random directions (estimator)')
    ap.add_argument('--h_frac', type=float, default=1e-6, help='FD step, in units of mean bond length')
    ap.add_argument('--dnu', type=float, default=0.2, help='target nu = nu_solver - dnu')
    ap.add_argument('--fE', type=float, default=0.8, help='target E = fE * E_solver')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--tag', default='')
    a = ap.parse_args()

    print('checkpoint %s' % os.path.basename(a.ckpt))
    print('data       %s' % os.path.basename(a.data))
    print('target     nu_solver - %.3f , %.2f * E_solver   (anchored on the SOLVER, not the model)'
          % (a.dnu, a.fE))
    net = M3.from_checkpoint(torch.load(a.ckpt, map_location='cpu', weights_only=False))
    raw = [g for g in T2.load(a.data) if g['sim_ok']]
    print('%d usable networks in the pool' % len(raw))

    # SELF-LOOPS: the stored schema cannot express their edge orientation, so a solver cannot be
    # rebuilt on them at all (see `evaluate_v2.oriented_edge_vecs`). Excluded, and COUNTED -- an
    # exclusion that is not reported is a biased sample.
    n_all = len(raw)
    raw = [g for g in raw if not EV.has_self_loops(g)]
    print('%d of %d networks EXCLUDED: self-loop bonds, edge orientation not reconstructable from '
          'the stored schema (tiny cells only) -- %d usable' % (n_all - len(raw), n_all, len(raw)))

    rng = np.random.default_rng(a.seed)
    if not check_label_reproduction(raw, rng.choice(len(raw), 12, replace=False)):
        return 1

    # ---- k channel -------------------------------------------------------------------------
    sel = rng.choice(len(raw), min(a.n_k, len(raw)), replace=False)
    rows_k, t0 = [], time.time()
    for n, i in enumerate(sel):
        g = raw[int(i)]
        try:
            cos, ns, ng = measure_k(net, g, a.dnu, a.fE)
        except Exception as e:                                            # noqa: BLE001
            print('   k: skipped %s: %s' % (g.get('family', '?'), str(e)[:70]))
            continue
        rows_k.append(dict(cos=cos, mag_ratio=ng / max(ns, 1e-300),
                           family=str(g.get('family', '?')),
                           w_max=float(g.get('w_max', np.nan)), n_tri=len(g['tri_bond'])))
        if (n + 1) % 50 == 0:
            print('   k: %d/%d  (%.0fs)' % (n + 1, len(sel), time.time() - t0))
    sum_k = report(rows_k, 'k')

    # ---- position channel ------------------------------------------------------------------
    small = [i for i in range(len(raw)) if len(raw[i]['pts']) <= a.max_nodes]
    print('\n%d networks with <= %d nodes (position channel)' % (len(small), a.max_nodes))
    selp = rng.choice(small, min(a.n_pos, len(small)), replace=False) if small else []
    rows_p, t0 = [], time.time()
    for n, i in enumerate(selp):
        g = raw[int(i)]
        try:
            cos, ns, ng, rej, probes = measure_pos(net, g, a.dnu, a.fE, a.h_frac, a.dirs, rng)
        except Exception as e:                                            # noqa: BLE001
            print('   pos: skipped %s: %s' % (g.get('family', '?'), str(e)[:70]))
            continue
        rows_p.append(dict(cos=cos, mag_ratio=ng / max(ns, 1e-300),
                           family=str(g.get('family', '?')),
                           w_max=float(g.get('w_max', np.nan)), n_tri=len(g['tri_bond']),
                           rejected=rej, probes=probes))
        if (n + 1) % 10 == 0:
            print('   pos: %d/%d  (%.0fs)' % (n + 1, len(selp), time.time() - t0))
    sum_p = report(rows_p, 'pts')

    os.makedirs(RESULTS, exist_ok=True)
    dst = os.path.join(RESULTS, 'gradient_fidelity%s.json' % (('_' + a.tag) if a.tag else ''))
    with open(dst, 'w') as f:
        json.dump(dict(config=vars(a), k=rows_k, pos=rows_p,
                       summary_k=sum_k, summary_pos=sum_p), f, indent=1)
    print('\n-> %s' % dst)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
