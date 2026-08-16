"""Phase 5 — VERTEX-POSITION OPTIMIZATION (derivative-free, SPSA) for the inverse designer.

The differentiable solver designs per-bond k on a FIXED geometry (positions are baked into
`DesignProblem.from_geo` and are NOT differentiable).  This module adds the missing design
axis: it OPTIMIZES the node positions themselves with SPSA (simultaneous perturbation
stochastic approximation) — 2 loss evaluations per step estimate a descent direction over ALL
coordinates at once — alternating with the (expensive, differentiable) k-design:

    round = [k-design on current geo]  ->  [SPSA position polish at FIXED k]
            ->  [re-Delaunay: topology may adapt -> k re-designed next round]

Between re-Delaunays the SPSA probes keep the SAME simplices (via
`triangulation.geo_from_simplices`), so a perturbation measures a SMOOTH geometry change; the
periodic re-Delaunay then lets edge flips happen so topology and positions explore jointly.

SPSA variant used (and why): with Bernoulli +-1 perturbations the raw SPSA gradient estimate is
g = s*Delta with a SINGLE scalar s = (L+ - L-)/(2c), i.e. every coordinate has the same
magnitude |s|.  |s| scales with the (tiny) loss and made raw steps ~1e-4 lattice units — no
progress in a small budget.  We therefore take the standard descent DIRECTION but a controlled
magnitude: step = a_k * sign(L+ - L-) * Delta (per-coordinate displacement exactly a_k, the
classic sign-SPSA / normalized-SPSA — identical direction, robust scale), plus an
accept-if-better-or-slightly-worse rule and a degenerate-triangle guard.

PROJECT RULE: position optimization NEVER re-triangulates — connectivity is frozen (distortion only).
`redelaunay_every` therefore DEFAULTS TO 0 (disabled) everywhere; pass a positive value only if you
deliberately want joint position+topology exploration (not the default, and not for fixed-topology work).

Public API
----------
    loss_at(geo, k, nu_target, E_target, nu_weight=1.0, E_weight=1.0)
                                                               -> float  (objective residual)
    spsa_positions(geo, k, nu_target, E_target, n_steps=40, a=0.02, c=0.01, seed=0,
                   redelaunay_every=0, nu_weight=1.0, E_weight=1.0)
                                                               -> geo    (carries 'spsa_history',
                                                                          'topology_changed')
    design_with_positions(nu_target, E_target, geo0, n_outer=3, spsa_steps=40, n_iter=80,
                          n_restarts=1, reg=0.02, redelaunay_every=0,
                          nu_weight=1.0, E_weight=1.0)          -> (geo, k, history)

**`nu_weight`/`E_weight` MUST match the weights the k-design used** — `CLAUDE.md` §3 makes this the
critical rule of position optimisation: mismatched weights mean the polish silently optimises a
DIFFERENT loss and destroys anisotropy. They were previously omitted from this very cheat-sheet
(audit B-3b), so a reader following it was led into precisely the documented failure mode.
`designer.design()` passes them through for you.
"""
# ---- §0 preamble (verbatim; positions.py lives directly in Phase 5/) --------------------------
import os, sys
import numpy as np
import torch
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))   # repo root from Phase 5/
sys.path.insert(0, os.path.join(REPO, 'Phase 3', 'verifications'))
import _common as C                 # sets up all other sys.path and imports the solver stack
from inverse_design import (DesignProblem, Objective, optimize, validate, ANG,
                            c6_to_nuE, c6_to_nuE_theta)
from inverse_design import _loss as _design_loss     # the exact scalar loss optimize() minimises
torch.set_default_dtype(torch.float64)   # REQUIRED — the whole stack is float64

import triangulation
import designer


# ---- helpers ---------------------------------------------------------------------------------
def _profile(x):
    """Broadcast a scalar or length-37 target into a numpy profile over ANG."""
    return np.array(np.broadcast_to(np.asarray(x, float), ANG.shape), float)


def _as_k_tensor(k):
    return k.detach() if torch.is_tensor(k) else torch.as_tensor(np.asarray(k, float))


def _tris_from_geo(geo):
    """Explicit (nt,3,3) [base,sx,sy] triangle list recovered from ANY geo dict (works for
    non-Delaunay/flipped geos too): the integer image shift of each vertex is read off the
    stored image-correct tri_verts."""
    pts = np.asarray(geo['pts'], float)
    box = np.array([float(geo['BL1'][0]), float(geo['BL2'][1])])
    simp = np.asarray(geo['simplices'], np.int64)
    sft = np.rint((np.asarray(geo['tri_verts'], float) - pts[simp]) / box).astype(np.int64)
    return np.concatenate([simp[:, :, None], sft], axis=2)


def _wrap(pts, tris, Lx, Ly):
    """Wrap pts into [0,Lx)x[0,Ly) AND compensate the tris image shifts so the real coordinates
    (hence the geometry) are exactly unchanged."""
    box = np.array([Lx, Ly])
    w = np.floor(pts / box).astype(np.int64)         # integer wrap counts per point
    pts_w = pts - w * box
    tris_w = np.asarray(tris, np.int64).copy()
    tris_w[:, :, 1:3] += w[tris_w[:, :, 0]]
    return pts_w, tris_w


def _edge_keys(tris):
    """Set of canonical (translation-invariant) edge keys of a tris list — two tris lists over the
    SAME pts describe the same topology iff their key sets are equal."""
    keys = set()
    for t in range(len(tris)):
        for i, j in ((0, 1), (1, 2), (2, 0)):
            keys.add(triangulation._edge_key(tris[t, i], tris[t, j]))
    return keys


# ---- 1. the position-design loss --------------------------------------------------------------
def loss_at(geo, k, nu_target, E_target, nu_weight=1.0, E_weight=1.0):
    """Objective residual of one solver forward on `geo` with FIXED per-bond `k`: the SAME
    weighted MSE `designer.design_on_topology` optimises (nu_theta + E_theta), WITHOUT the
    k-regulariser (k is fixed here, so it would only add a constant).  Cheap — one forward per call
    — this is the SPSA evaluation function.  `nu_weight`/`E_weight` MUST match the weights used by
    the k-design, else the position polish optimises a different objective (e.g. flattening a
    down-weighted E at the expense of a prioritised nu anisotropy)."""
    prob = DesignProblem.from_geo(geo)
    objs = [Objective('nu_theta', _profile(nu_target), thetas=ANG, weight=nu_weight),
            Objective('E_theta',  _profile(E_target),  thetas=ANG, weight=E_weight)]
    with torch.no_grad():
        return float(_design_loss(prob, objs, _as_k_tensor(k), None, 0.0))


# ---- 2. SPSA over the node positions (k fixed) ------------------------------------------------
def spsa_positions(geo, k, nu_target, E_target, n_steps=40, a=0.02, c=0.01, seed=0,
                   redelaunay_every=0, verbose=False, nu_weight=1.0, E_weight=1.0):
    """Derivative-free position polish of `geo` at FIXED designed `k` (SPSA, sign-normalized —
    see module docstring).  Per step: perturb ALL positions by +-c_k*Delta (Delta random +-1 per
    coordinate) ON THE SAME SIMPLICES (smooth geometry probe, no accidental edge flips), evaluate
    `loss_at` twice, move every coordinate by a_k against the estimated descent direction, then
    accept-if-better-or-slightly-worse; reject any step that collapses a triangle below 1e-3 of
    the mean area.  Every `redelaunay_every` steps the points are re-Delaunayed: if the topology
    is unchanged the same bond ordering (hence `k`) is kept; if it CHANGED, the new-topology geo
    is returned immediately with `geo['topology_changed']=True` so the caller re-designs k.

    Returns a geo dict (best positions found, wrapped into the box) carrying extra keys
    'spsa_history' (accepted-loss trace), 'spsa_n_reject', 'topology_changed'."""
    Lx, Ly = float(geo['BL1'][0]), float(geo['BL2'][1])
    pts = np.asarray(geo['pts'], float).copy()
    tris = _tris_from_geo(geo)                        # preserves even non-Delaunay topologies
    rng = np.random.default_rng(seed)
    kt = _as_k_tensor(k)

    L_cur = loss_at(geo, kt, nu_target, E_target, nu_weight, E_weight)
    best_pts, best_tris, best_L = pts.copy(), tris.copy(), L_cur
    history = [L_cur]
    n_reject = 0
    rej_streak = 0
    a_eff = float(a)
    A_stab = max(1.0, 0.1 * n_steps)                  # standard SPSA stability offset
    area_floor = 1e-3

    for step in range(n_steps):
        ak = a_eff / (step + 1 + A_stab) ** 0.602
        ck = c / (step + 1) ** 0.101

        # two smooth-geometry probes on the SAME simplices (no wrap needed: geo_from_simplices
        # is happy with points slightly outside the box)
        Delta = rng.choice([-1.0, 1.0], size=pts.shape)
        gp = triangulation.geo_from_simplices(pts + ck * Delta, tris, Lx, Ly)
        gm = triangulation.geo_from_simplices(pts - ck * Delta, tris, Lx, Ly)
        mean_area = 0.5 * (gp['areas'].mean() + gm['areas'].mean())
        if min(gp['areas'].min(), gm['areas'].min()) < area_floor * mean_area:
            n_reject += 1                             # degenerate probe — skip this step
            history.append(L_cur)
            continue
        Lp = loss_at(gp, kt, nu_target, E_target, nu_weight, E_weight)
        Lm = loss_at(gm, kt, nu_target, E_target, nu_weight, E_weight)

        # sign-SPSA update: raw ghat = (Lp-Lm)/(2ck) * Delta has one shared magnitude, so the
        # normalized step is exactly a_k * sign(Lp-Lm) * Delta (same direction, controlled size)
        s = np.sign(Lp - Lm)
        if s == 0.0:
            history.append(L_cur)
            continue
        pts_new = pts - ak * s * Delta

        g_new = triangulation.geo_from_simplices(pts_new, tris, Lx, Ly)
        if g_new['areas'].min() < area_floor * g_new['areas'].mean():
            n_reject += 1; rej_streak += 1            # degenerate triangle — reject
        else:
            L_new = loss_at(g_new, kt, nu_target, E_target, nu_weight, E_weight)
            if L_new <= L_cur * 1.05:                 # accept if better or only slightly worse
                pts, L_cur = pts_new, L_new
                rej_streak = 0
                if L_new < best_L:
                    best_pts, best_tris, best_L = pts.copy(), tris.copy(), L_new
            else:
                n_reject += 1; rej_streak += 1
        if rej_streak >= 3:                           # persistent rejection -> shrink the step
            a_eff *= 0.5
            rej_streak = 0
        history.append(L_cur)
        if verbose:
            print(f"    spsa step {step:3d}: L={L_cur:.4e} (best {best_L:.4e}, a_k={ak:.4f})")

        # periodic re-Delaunay: let the topology adapt to the moved points
        if redelaunay_every and (step + 1) % redelaunay_every == 0 and step + 1 < n_steps:
            pts_w, tris_w = _wrap(pts, tris, Lx, Ly)
            tris_new = triangulation.delaunay_tris(pts_w, Lx, Ly)
            if _edge_keys(tris_w) == _edge_keys(tris_new):
                pts, tris = pts_w, tris_w             # same topology — keep bond order (k valid)
            else:                                     # topology CHANGED — k invalid: hand back
                g = triangulation.geo_from_simplices(pts_w, tris_new, Lx, Ly)
                g['spsa_history'] = history
                g['spsa_n_reject'] = n_reject
                g['topology_changed'] = True
                return g

    pts_w, tris_w = _wrap(best_pts, best_tris, Lx, Ly)
    g = triangulation.geo_from_simplices(pts_w, tris_w, Lx, Ly)
    g['spsa_history'] = history
    g['spsa_n_reject'] = n_reject
    g['topology_changed'] = False
    return g


# ---- 3. the alternating position+k design loop ------------------------------------------------
def design_with_positions(nu_target, E_target, geo0, n_outer=3, spsa_steps=40, n_iter=80,
                          n_restarts=1, reg=0.02, spsa_a=0.02, spsa_c=0.01,
                          redelaunay_every=0, seed=0, verbose=True,
                          nu_weight=1.0, E_weight=1.0):
    """Joint position+k design by alternation, starting from topology `geo0`:

        round = [k-design (designer.design_on_topology)] -> [SPSA position polish at fixed k]
                -> [re-Delaunay inside SPSA; if topology changed, k is re-designed next round]

    plus one final k-design on the last geometry.  Tracks the best (geo, k) by the design loss
    (`loss_at` — the same objective residual for every candidate, so k-only and k+positions are
    compared on an equal footing).  Returns (best_geo, best_k, history) where history is a list of
    (stage, outer_round, loss) tuples: stage 'k' = after a k-design, 'spsa' = after the position
    polish of that round."""
    geo = geo0
    history = []
    best = None                                       # (geo, k, loss)

    def _kdesign(g, outer, stage='k'):
        nonlocal best
        _, _, res = designer.design_on_topology(g, nu_target, E_target, n_iter=n_iter,
                                                n_restarts=n_restarts, reg=reg,
                                                nu_weight=nu_weight, E_weight=E_weight)
        L = loss_at(g, res['k'], nu_target, E_target, nu_weight, E_weight)
        history.append((stage, outer, L))
        if best is None or L < best[2]:
            best = (g, res['k'], L)
        return res['k'], L

    for outer in range(n_outer):
        k, Lk = _kdesign(geo, outer)
        geo2 = spsa_positions(geo, k, nu_target, E_target, n_steps=spsa_steps,
                              a=spsa_a, c=spsa_c, seed=seed + outer,
                              redelaunay_every=redelaunay_every,
                              nu_weight=nu_weight, E_weight=E_weight)
        if geo2['topology_changed']:
            # k no longer matches geo2's bonds; the polish gains up to the change are in geo2's
            # positions, and the NEXT round's k-design re-designs k on the new topology
            history.append(('spsa', outer, geo2['spsa_history'][-1]))
        else:
            Ls = loss_at(geo2, k, nu_target, E_target, nu_weight, E_weight)
            history.append(('spsa', outer, Ls))
            if Ls < best[2]:
                best = (geo2, k, Ls)
        if verbose:
            print(f"  outer {outer}: k-design loss {Lk:.4e} -> spsa {history[-1][2]:.4e}"
                  f"{'  [topology changed]' if geo2['topology_changed'] else ''}"
                  f"  (rejects {geo2['spsa_n_reject']})")
        geo = geo2

    _kdesign(geo, n_outer, stage='k-final')           # consistent (geo,k) on the final geometry
    if verbose:
        print(f"  final k-design loss {history[-1][2]:.4e}; best overall {best[2]:.4e}")
    return best[0], best[1], history
