"""Phase 5 Goal 1 — ISOTROPIC inverse designer with a per-bond stiffness CONTRAST FLOOR.

Goal 1 targets a *scalar* (isotropic) Poisson ratio nu with Young's modulus E=1, and asks how the
achievable nu depends on how much stiffness CONTRAST the network is allowed.  This module adds the
one NEW piece Goal 1 needs on top of the existing Phase 5 stack (designer.py / positions.py /
seeds.py / triangulation.py, all imported, none modified):

    a k-designer whose per-bond stiffnesses are BOUNDED to a contrast band [f, 1]*scale.

Parameterisation (the contrast floor f = min(k)/avg(k) lower bound):
    raw (n_bond,) is the free optimiser variable;
    kshape = f + (1-f)*sigmoid(raw)     -> kshape in [f, 1]     (min(kshape)/avg(kshape) >= f)
    k      = scale * kshape             -> a uniform rescale (E is linear in scale, nu invariant)
Bands: soft f=0.0, large f=0.1, medium f=0.5, small f=0.9, none f=0.99.
f=0 recovers the unconstrained [0,1] range (bonds may go fully soft -> mechanisms/auxetics);
f->1 forces a nearly-uniform lattice (contrast suppressed).

Design logic
------------
The elastic tensor is LINEAR in the per-bond k (harmonic solver, fixed geometry), so scaling ALL
bonds by a constant `scale` scales the physical tensor -> E by `scale` and leaves nu unchanged.  We
therefore optimise only the SHAPE (relative stiffness distribution kshape) toward a FLAT
nu(theta)=nu_target profile, then set `scale = 1/E_shape` once so E=1 exactly, for free.

Public API
----------
    KBANDS                                                     -> dict(name->f)
    kshape_from_raw(raw, f)                                    -> torch (n_bond,) in [f,1]
    nuE_theta_solver(geo, k)                                   -> (nu_theta, E_theta) numpy (37,)
    design_iso_k(geo, nu_target, f, n_iter=150, n_restarts=2, w_iso=0.5, seed=0)
                                                              -> (k numpy, nu_ach, E_ach)
    design_iso(geo, nu_target, f, n_outer=2, spsa_steps=20, n_iter=150, n_restarts=2,
               w_iso=0.5, seed=0, verbose=False, return_stages=False)
                                                              -> (geo, k, nu_ach, E_ach[, konly])
"""
# ---- §0 preamble (verbatim; design_iso.py lives directly in Phase 5/) -------------------------
import os, sys
import numpy as np
import torch
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))   # repo root from Phase 5/
sys.path.insert(0, os.path.join(REPO, 'Phase 3', 'verifications'))
import _common as C                 # sets up all other sys.path and imports the solver stack
from inverse_design import (DesignProblem, Objective, optimize, validate, ANG,
                            c6_to_nuE, c6_to_nuE_theta)
torch.set_default_dtype(torch.float64)   # REQUIRED — the whole stack is float64

import positions

# canonical contrast bands (f = min(k)/avg(k) lower bound)
KBANDS = {'soft': 0.0, 'large': 0.1, 'medium': 0.5, 'small': 0.9, 'none': 0.99}


# ---- helpers ---------------------------------------------------------------------------------
def kshape_from_raw(raw, f):
    """Contrast-floor map: kshape = f + (1-f)*sigmoid(raw) in [f,1], so min/avg >= f for any raw."""
    return f + (1.0 - f) * torch.sigmoid(raw)


def _as_np(k):
    return k.detach().numpy() if torch.is_tensor(k) else np.asarray(k, float)


def nuE_theta_solver(geo, k):
    """Solver's directional nu(theta), E(theta) for a designed per-bond `k` (numpy (37,), (37,))."""
    prob = DesignProblem.from_geo(geo)
    with torch.no_grad():
        out = prob.forward(torch.as_tensor(_as_np(k)))
        C6 = prob.region_tensor(out['per_triangle'], None)
        nu_th, E_th = c6_to_nuE_theta(C6, ANG)
    return nu_th.numpy(), E_th.numpy()


def _rescale_to_E1(geo, k):
    """Rescale k by 1/mean(E_theta) so the SOLVER's mean E = 1 (nu is scale-invariant, unchanged).
    Returns (k_scaled numpy, nu_mean, E_mean=~1)."""
    nu_th, E_th = nuE_theta_solver(geo, k)
    Em = float(E_th.mean())
    k2 = _as_np(k) / Em
    return k2, float(nu_th.mean()), 1.0


# ---- 1. the constrained-k isotropic designer (SHAPE optimisation, then rescale E=1) -----------
def design_iso_k(geo, nu_target, f, n_iter=150, n_restarts=2, w_iso=0.5, seed=0, lr=0.05):
    """Design per-bond k on ONE fixed topology toward a FLAT (isotropic) nu(theta)=nu_target with
    E=1, subject to the contrast floor f (min(k)/avg(k) >= f).

    Optimises the SHAPE only: loss = mean((nu_theta - nu_target)^2) + w_iso*(var(nu_theta)
    + var(E_theta)) forwarding the solver with k=kshape (scale=1).  E is NOT in the loss (it is
    scale-free at this stage); after the best shape is found, `scale = 1/E_achieved` sets E=1 for
    free.  Best over `n_restarts`.  Returns (k numpy, nu_achieved, E_achieved) — the last two the
    SOLVER's isotropic-mean values at the final scaled k."""
    prob = DesignProblem.from_geo(geo)
    nbond = prob.n_bond
    best = None
    for r in range(n_restarts):
        g = torch.Generator().manual_seed(seed + r)
        raw = (0.3 * torch.randn(nbond, generator=g)).requires_grad_(True)
        opt = torch.optim.Adam([raw], lr=lr)
        for _ in range(n_iter):
            opt.zero_grad()
            kshape = kshape_from_raw(raw, f)
            out = prob.forward(kshape)
            C6 = prob.region_tensor(out['per_triangle'], None)
            nu_th, E_th = c6_to_nuE_theta(C6, ANG)
            loss = ((nu_th - nu_target) ** 2).mean() + w_iso * (nu_th.var() + E_th.var())
            loss.backward()
            opt.step()
        with torch.no_grad():
            kshape = kshape_from_raw(raw, f)
            out = prob.forward(kshape)
            C6 = prob.region_tensor(out['per_triangle'], None)
            nu_th, E_th = c6_to_nuE_theta(C6, ANG)
            L = float(((nu_th - nu_target) ** 2).mean() + w_iso * (nu_th.var() + E_th.var()))
        if best is None or L < best[0]:
            best = (L, kshape.detach().clone(), float(E_th.mean()))
    _, kshape, E_shape = best
    scale = 1.0 / E_shape if E_shape > 0 else 1.0
    k = (scale * kshape).numpy()
    nu_ach, E_ach = nuE_theta_solver(geo, k)          # exact solver readout at the final scaled k
    return k, float(nu_ach.mean()), float(E_ach.mean())


# ---- 2. alternation with SPSA position polish (nu only; E fixed by rescale) -------------------
def design_iso(geo, nu_target, f, n_outer=2, spsa_steps=20, n_iter=150, n_restarts=2,
               w_iso=0.5, seed=0, verbose=False, return_stages=False,
               spsa_a=0.25, spsa_c=0.05):
    """Joint k + vertex-position isotropic design by alternation, starting from topology `geo`:

        round = [design_iso_k]  ->  [positions.spsa_positions at fixed k, nu-only]
                -> [re-Delaunay inside SPSA; if topology changed, k re-designed next round]

    **`spsa_a`/`spsa_c` are passed explicitly (2026-08-22) — they used to be omitted**, so positions
    ran at the library defaults a=0.02, c=0.01. Over `spsa_steps=20` x `n_outer=2` that is **0.202
    lattice spacings** of possible travel per coordinate, against **2.68** for `g1_2` at a=0.25:
    13x less, and an eighth of even the a=0.15 arm measured FAILING to reach auxetic nu
    (`g1_2_triangular_start_probe.py`). goal1's position optimisation was therefore effectively OFF,
    and its high-contrast-floor rows were search-limited, not physics-limited — they reported little
    more than the seed's own nu near 1/3 while `g1_2` reaches -0.436 at k=1 EXACTLY, which is a
    stricter constraint than f=0.99. Defaults now a=0.25, c=0.05: travel 2.53, matching g1_2, at the
    SAME step count and therefore the same runtime (travel scales linearly in `a`).

    SPSA polishes positions toward the FLAT nu target with E_weight=0 (E is set by the k-rescale,
    so the position polish must NOT chase it).  After any fixed-k position move, k is rescaled to
    restore E=1 (nu is scale-invariant, so this is free and keeps every candidate at E=1).  Keeps
    the best (geo, k) by |nu_solver - nu_target|.

    Returns (geo, k, nu_ach, E_ach); if `return_stages`, also returns the k-only stage tuple
    (geo0, k_konly, nu_konly, E_konly) so the caller records stage (ii) without recomputing it."""
    # round-0 k-design == the k-ONLY stage (ii)
    k0, nu0, E0 = design_iso_k(geo, nu_target, f, n_iter, n_restarts, w_iso, seed)
    konly = (geo, k0, nu0, E0)

    best = [geo, k0, nu0, E0, abs(nu0 - nu_target)]   # [geo, k, nu, E, nu_err]
    cur_geo, cur_k = geo, k0

    for outer in range(n_outer):
        if outer > 0:                                  # round-0 k already computed above
            cur_k, nu_k, E_k = design_iso_k(cur_geo, nu_target, f, n_iter, n_restarts, w_iso,
                                            seed + outer)
            err_k = abs(nu_k - nu_target)
            if err_k < best[4]:
                best = [cur_geo, cur_k, nu_k, E_k, err_k]

        geo2 = positions.spsa_positions(cur_geo, cur_k, nu_target=nu_target, E_target=1.0,
                                        n_steps=spsa_steps, a=spsa_a, c=spsa_c, seed=seed + outer,
                                        nu_weight=1.0, E_weight=0.0)
        topo_changed = bool(geo2.get('topology_changed'))
        if not topo_changed:
            k2, nu2, E2 = _rescale_to_E1(geo2, cur_k)  # positions moved; restore E=1 (nu invariant)
            err2 = abs(nu2 - nu_target)
            if err2 < best[4]:
                best = [geo2, k2, nu2, E2, err2]
        if verbose:
            print(f"  outer {outer}: nu_err {best[4]:.4f}"
                  f"{'  [topology changed]' if topo_changed else ''}")
        cur_geo = geo2                                  # k for the new geo is (re)designed next round

    # final k-design on the last geometry (consistent (geo,k))
    kf, nuf, Ef = design_iso_k(cur_geo, nu_target, f, n_iter, n_restarts, w_iso, seed + n_outer)
    errf = abs(nuf - nu_target)
    if errf < best[4]:
        best = [cur_geo, kf, nuf, Ef, errf]

    if return_stages:
        return best[0], best[1], best[2], best[3], konly
    return best[0], best[1], best[2], best[3]
