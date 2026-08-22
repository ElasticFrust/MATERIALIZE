"""Phase 5 M1 core — the search-based inverse designer.

Given a target directional response nu(theta), E(theta), the designer:
  1. builds a BOUNDED pool of candidate triangulated topologies (structured seeds + random
     patches + auxetic motifs + position-jitter variants + NON-Delaunay edge-flip variants),
  2. optimises the per-bond rigidities k on EACH topology toward the target (differentiable
     solver, several restarts for diversity),
  3. ranks the topologies by design loss and keeps the best few (several distinct designs for
     the same target),
  4. VERIFIES each kept design with an INDEPENDENT full-PBC simulation (a different code path
     from the solver it optimised against) and records the solver-vs-sim gap, and
  5. saves each design (geometry + k + per-triangle tensor + metadata) under Phase 5/networks/.

Public API
----------
    design_on_topology(geo, nu_target, E_target, n_iter=120, n_restarts=3, reg=0.02)
                                                 -> (prob, objs, res)
    verify(geo, k, nu_target=None, E_target=None)               -> report dict
    topology_pool(n_random=8, n_nodes=120, n_flip_variants=4, flips=6, seed=0) -> list[geo]
    search(nu_target, E_target, pool, keep=5, **design_kw)      -> list[(geo, k, loss)]
    design(nu_target, E_target, tag, pool=None, keep=5, **design_kw) -> DesignReports
    run_physicality_checks(C6, ...) -> (ok, failures);  PHYSICALITY_CHECKS is the extensible registry

`design()` returns `DesignReports` — a list of the KEPT reports (so every existing caller is
unchanged) that additionally carries the DROPPED candidates on `.rejected`, each with why it was
dropped and an `outscored_kept` flag when it formally beat everything kept.

Do NOT modify _common.py / seeds.py / triangulation.py / Phase 2/forward_solver_torch.py — import.
"""
# ---- §0 preamble (verbatim; designer.py lives directly in Phase 5/) ---------------------------
import os, sys
import numpy as np
import torch
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))   # repo root from Phase 5/
sys.path.insert(0, os.path.join(REPO, 'Phase 3', 'verifications'))
import _common as C                 # sets up all other sys.path and imports the solver stack
from inverse_design import (DesignProblem, Objective, optimize, validate, ANG,
                            c6_to_nuE, c6_to_nuE_theta)
torch.set_default_dtype(torch.float64)   # REQUIRED — the whole stack is float64

import seeds
import triangulation
import phys_targets
# The oracle's own exception type. Imported DIRECTLY, not reached through `C.PH.…` — that
# re-export reach-through is the pattern audit A-7b exists to remove. This is not a new cross-layer
# dependency: `verify()` already calls the oracle via `C.sim_relax`/`C.sim_bulk_C6`, and catching
# what it deliberately raises is what CLAUDE.md §3 asks of callers. (The no-oracle-imports rule
# governs the DESIGN path in Phase 2/3; this is the verification path.)
import physical_homog as PH
import mesh_build as MB              # check_mesh_preconditions — the solver's own mesh gate (A-17)

# ν is O(1) and can cross zero, so every ν comparison here is relative WITH A FLOOR; E is strictly
# positive so it is relative outright. ONE constant, shared by the solver-vs-sim gap (verify()) and
# the achieved-vs-target error, so the two headline numbers are on the SAME scale — audit A-2, where
# they were not. At ν→0 the effective absolute ν-floor is tol*EPS_NU, so EPS_NU and any tolerance
# compared against these numbers are coupled (CLAUDE.md §3, verification discipline).
EPS_NU = 0.05

# Default solver-vs-sim honesty tolerance. One constant so `design()`'s default and the self-test's
# assertion cannot drift apart (audit A-6: the assert hardcoded 0.05, shadowing the argument).
GAP_TOL = 0.05

# Selection score: target error and solver-vs-sim disagreement WEIGHED, not one vetoing the other
# (2026-08-22). A large gap means the two CODE PATHS disagree about a network, not that the network
# is unreal, so it belongs in the ranking as a cost and in the report as its own number. Vetoing on
# it discarded designs that did the job: measured in `run_g1_2`, the rule kept a barely-moved design
# at gap 0.031 over one that reached nu=-0.134 at gap 0.371, and a whole experiment then read as
# "distortion cannot reach auxetic nu". PHYSICALITY stays a veto — that IS a statement about the
# network (non-SPD, non-finite, |nu|>=nu_max), and it catches what the gap cannot (audit A-5).
# 0.5 is a judgement, not a fit: on 110 saved designs every lambda in [0,1] picked the same winners.
SCORE_LAMBDA = 0.5


def _score(rep):
    """Selection score, lower is better: achieved-vs-target plus SCORE_LAMBDA x solver-vs-sim."""
    return float(rep['target_err_sim'] + SCORE_LAMBDA * rep['solver_sim_gap'])


# ---- small helpers ---------------------------------------------------------------------------
def _profile(x):
    """Broadcast a scalar or length-37 target into a numpy profile over ANG."""
    return np.array(np.broadcast_to(np.asarray(x, float), ANG.shape), float)


def _rel_err(nu, E, nu_ref, E_ref):
    """Relative directional error of (nu, E) against a reference profile, split by channel.

    Returns (err_nu, err_E, err_nu + err_E). Same structure as verify()'s solver-vs-sim gap. Used
    for achieved-vs-target with the REFERENCE = the TARGET: the target is fixed and known, so the
    metric cannot be flattered by achieving a large E (audit A-2 — the old metric was an ABSOLUTE
    max(|dnu|,|dE|), on which a target with E >> 1 let the E term swamp nu entirely, and it was the
    RANKING key)."""
    err_nu = float((np.abs(nu - nu_ref) / (np.abs(nu_ref) + EPS_NU)).max())
    err_E = float((np.abs(E - E_ref) / np.maximum(np.abs(E_ref), 1e-12)).max())
    return err_nu, err_E, err_nu + err_E


# ---- physicality checks: an EXTENSIBLE registry -----------------------------------------------
# CLAUDE.md §3 assigns the caller/designer the "is the achieved RESPONSE physical" gate (it needs the
# solver, and the sim must not depend on the solver). Audit A-5: no designer path applied any such
# check to a DESIGNED network — `phys_targets.is_physical` was only ever applied to generated
# TARGETS. Kept as a registry rather than one hardcoded call so further checks are APPENDED here
# without touching verify()/design().
#
# Contract for a check:  fn(C6, geo=None, k=None, rep=None) -> (ok: bool, reason: str)
# `reason` is only read when ok is False, and should name the violated criterion.

def _chk_realizable(C6, geo=None, k=None, rep=None):
    """SPD + finite + E in range + |nu| bounded + reciprocity (phys_targets.is_physical)."""
    return phys_targets.is_physical(C6)


def _chk_mesh(C6, geo=None, k=None, rep=None):
    """The SOLVER's own mesh preconditions (audit A-17) — closed combinatorics, no inverted triangles.

    This does not ask whether the RESPONSE is physical; it asks whether the solver was entitled to
    compute one at all. Where either condition fails the solver is simply wrong (on `rotating_squares`,
    known answer nu = -1: sim -1.00000, solver -0.685), and nothing else in the pipeline notices —
    the design merely looks like a bad one. Phase 5 is PBC-only, hence `periodic=True`."""
    if geo is None:
        return True, ''                                  # nothing to check
    ok, failures = MB.check_mesh_preconditions(geo, periodic=True)
    return ok, '; '.join(failures)


PHYSICALITY_CHECKS = [('realizable', _chk_realizable), ('mesh', _chk_mesh)]


def run_physicality_checks(C6, geo=None, k=None, rep=None, checks=None):
    """Run every registered physicality check on an ACHIEVED response tensor.

    Returns (ok_all, failures) with failures = [(check_name, reason), ...] — empty iff ok_all.
    ALL checks run (no short-circuit) so a rejection log lists every violated criterion, not just
    the first. `checks` overrides the registry (used by the tests)."""
    failures = []
    for name, fn in (PHYSICALITY_CHECKS if checks is None else checks):
        ok, reason = fn(C6, geo=geo, k=k, rep=rep)
        if not ok:
            failures.append((name, str(reason)))
    return (not failures), failures


class DesignReports(list):
    """The list of KEPT design reports, with the rejected ones carried alongside on `.rejected`.

    A plain list to every existing caller (`for r in reports`, `reports[0]`, `len(reports)`) — the
    six call sites are untouched — while keeping the dropped candidates available instead of
    discarding them: a design can be rejected for being unphysical yet still be the one you want to
    look at, and one that formally OUTSCORED everything kept is flagged `outscored_kept`."""

    def __init__(self, kept=(), rejected=()):
        super().__init__(kept)
        self.rejected = list(rejected)


def _tag(geo, name, is_fic=None):
    """Carry seed identity + fictional-edge mask ON the geo dict (extra keys are harmless to the
    solver / save_network) so a plain list[geo] pool preserves the metadata design() saves."""
    geo['seed_name'] = name
    geo['is_fictional'] = (np.zeros(len(geo['bond_R']), bool)
                           if is_fic is None else np.asarray(is_fic, bool))
    return geo


def _is_nondelaunay(geo):
    return str(geo.get('seed_name', '')).startswith('flipped_')


# ---- 1. design k on ONE fixed topology -------------------------------------------------------
def design_on_topology(geo, nu_target, E_target, n_iter=120, n_restarts=3, reg=0.02,
                       nu_weight=1.0, E_weight=1.0):
    """Optimise k on ONE fixed topology toward the directional target (best over restarts).
    `nu_weight`/`E_weight` trade off the two objectives — e.g. raise nu_weight / lower E_weight to
    prioritise matching a directional nu(theta) profile over hitting an exact stiffness."""
    prob = DesignProblem.from_geo(geo)
    objs = [Objective('nu_theta', np.asarray(nu_target), thetas=ANG, weight=nu_weight),
            Objective('E_theta',  np.asarray(E_target),  thetas=ANG, weight=E_weight)]
    res = optimize(prob, objs, mode='k', n_iter=n_iter, n_restarts=n_restarts, reg=reg,
                   verbose=False)
    return prob, objs, res      # res['k'] is the designed per-bond stiffness


# ---- 2. independent-sim verification ---------------------------------------------------------
def verify(geo, k, nu_target=None, E_target=None):
    """INDEPENDENT-sim check + solver-vs-sim gap for a designed (geo, k).

    Runs a full-PBC relaxation (`sim_per_triangle_C6` — a DIFFERENT code path from the solver the
    design was optimised against) to get the sim's directional nu(theta),E(theta); then the
    solver's own readout for the same k; then their gap (the honesty check).  If targets are given
    also returns `target_err_sim` — how well the INDEPENDENT sim hits the requested target."""
    C.apply_k_to_geo(geo, k)
    u_modes = C.sim_relax(geo)                  # ONE relaxation, shared by both readouts
    C6_per  = C.sim_per_triangle_C6(geo, u_modes)          # spatial pattern (shared contraction)
    C6_bulk = C.sim_bulk_C6(geo, u_modes)                  # bulk: INDEPENDENT (virial), audit A-1/A-9
    nu_sim, E_sim = C.nu_E_theta(C6_bulk, ANG)                       # numpy (37,)

    prob = DesignProblem.from_geo(geo)
    out  = prob.forward(torch.as_tensor(np.asarray(k, float)) if not torch.is_tensor(k) else k)
    C6_solver = prob.region_tensor(out['per_triangle'], None)
    nu_slv, E_slv = (t.numpy() for t in c6_to_nuE_theta(C6_solver, ANG))

    # RELATIVE solver-vs-sim gap (honesty check), reference = the SIM (the truth being compared to).
    gap_nu, gap_E, gap = _rel_err(nu_slv, E_slv, nu_sim, E_sim)
    rep = dict(nu_sim=nu_sim, E_sim=E_sim, nu_solver=nu_slv, E_solver=E_slv,
               solver_sim_gap=gap, solver_sim_gap_nu=gap_nu, solver_sim_gap_E=gap_E,
               C6_per=C6_per, C6_bulk=C6_bulk)

    # PHYSICALITY of the ACHIEVED response, on the INDEPENDENT bulk tensor (audit A-5).
    rep['physical_ok'], rep['physical_failures'] = run_physicality_checks(
        C6_bulk, geo=geo, k=k, rep=rep)

    if nu_target is not None and E_target is not None:
        nu_t, E_t = _profile(nu_target), _profile(E_target)
        # achieved-vs-target on BOTH sides — CLAUDE.md §3(B) requires the solver residual as well as
        # the independent sim's, and only the sim's was ever recorded (audit A-3). Ranking still uses
        # the SIM (see design()'s docstring: a low solver loss can be a mirage on floppy topologies).
        e_nu, e_E, e = _rel_err(nu_sim, E_sim, nu_t, E_t)
        rep.update(target_err_sim=e, target_err_sim_nu=e_nu, target_err_sim_E=e_E)
        s_nu, s_E, s = _rel_err(nu_slv, E_slv, nu_t, E_t)
        rep.update(target_err_solver=s, target_err_solver_nu=s_nu, target_err_solver_E=s_E)
    return rep


# ---- 3. the candidate topology pool ----------------------------------------------------------
def topology_pool(n_random=8, n_nodes=120, n_flip_variants=4, flips=6, seed=0):
    """A BOUNDED (~15-25) pool of candidate topologies for the search.  Three kinds of moves:
      (a) a handful of STRUCTURED seeds from the zoo (Bravais + tilings + auxetic motifs + random
          patches across point processes);
      (b) POSITION-JITTER variants of a couple of promising seeds (perturb points, re-Delaunay) —
          reach new DELAUNAY topologies;
      (c) NON-DELAUNAY edge-flip variants (triangulation.random_flipped_geo) — reach topologies
          Delaunay can never produce (satisfies M1b / §10 criterion 5).
    Returns a plain list[geo]; each geo carries `seed_name` + `is_fictional` as extra dict keys."""
    pool = []

    # (a) structured seeds ---------------------------------------------------------------------
    for name, geo in seeds.seed_bravais(phi_vals=(1.0, 1.3), psi_vals=(1.0,),
                                        etas=(0.0, 0.2), seeds=(0,), half=6):
        pool.append(_tag(geo, name))
    for tname, reps in (('square', 4), ('honeycomb', 3), ('kagome', 3)):
        rec = seeds.seed_tiling(tname, reps)
        pool.append(_tag(rec['geo'], rec['name'], rec['is_fictional']))
    for rec in seeds.auxetic_motifs(reps=3):
        pool.append(_tag(rec['geo'], rec['name'], rec['is_fictional']))
    procs = ('poisson_disk', 'blue_noise', 'uniform', 'graded')
    randels = []
    for i in range(n_random):
        rec = seeds.random_patch(n_nodes, seed=seed + i, process=procs[i % len(procs)])
        randels.append(rec)
        pool.append(_tag(rec['geo'], rec['name']))

    # (b) position-jitter variants (reach NEW Delaunay topologies) -----------------------------
    jitter_bases = [('bravais_reg', C.make_lattice(1.0, 1.0, half=6))]
    if randels:
        jitter_bases.append((randels[0]['name'], randels[0]['geo']))
    rng = np.random.default_rng(seed + 999)
    for bname, bgeo in jitter_bases:
        Lx, Ly = float(bgeo['BL1'][0]), float(bgeo['BL2'][1])
        pts = np.asarray(bgeo['pts'], float) + rng.normal(0.0, 0.1, (len(bgeo['pts']), 2))
        pts[:, 0] %= Lx; pts[:, 1] %= Ly
        pool.append(_tag(C._periodic_delaunay(pts, Lx, Ly), f'jitter_{bname}'))

    # (c) NON-Delaunay flipped variants (reach topologies Delaunay can't) -----------------------
    flip_bases = [('reg', C.make_lattice(1.0, 1.0, half=6))]
    if randels:
        flip_bases.append(('rand', randels[0]['geo']))
    for i in range(n_flip_variants):
        bname, bgeo = flip_bases[i % len(flip_bases)]
        s = seed + 100 + i
        gf = triangulation.random_flipped_geo(bgeo, flips, seed=s)
        pool.append(_tag(gf, f'flipped_{bname}_f{flips}_s{s}'))

    return pool


# ---- 4. search over the pool -----------------------------------------------------------------
def _rank_pool(nu_target, E_target, pool, **design_kw):
    """Design k on every topology in the pool; return ALL (geo, k, loss) sorted by loss (best
    first).  Shared by search() (keep-best slice) and design() (diversity-aware selection)."""
    results = []
    for geo in pool:
        _, _, res = design_on_topology(geo, nu_target, E_target, **design_kw)
        results.append((geo, res['k'], float(res['loss'])))
    results.sort(key=lambda r: r[2])
    return results


def search(nu_target, E_target, pool, keep=5, **design_kw):
    """Scan a POOL of topologies; design k on each; return the `keep` best (lowest loss) as
    (geo, k, loss) — several distinct designs for the same target."""
    return _rank_pool(nu_target, E_target, pool, **design_kw)[:keep]


# ---- 5. main entry point ---------------------------------------------------------------------
def design(nu_target, E_target, tag, pool=None, keep=5, gap_tol=GAP_TOL,
           optimize_positions=True, pos_budget=None, **design_kw):
    """Design several distinct networks for a target nu(theta),E(theta).

    `optimize_positions` (default True): after selecting the kept designs, additionally run the
    derivative-free VERTEX-POSITION optimization (positions.design_with_positions) on the TOP
    design only (budget); if it improves the design loss AND is physical AND improves the SCORE it
    replaces the top design (seed_name gets '+pos').  `pos_budget` is an optional
    dict overriding the modest default polish budget
    dict(n_outer=2, spsa_steps=25, n_iter=60, n_restarts=1).

    Pipeline: build/accept a pool -> rank topologies by (cheap, solver) design loss -> VERIFY a
    generous candidate set with the INDEPENDENT sim -> DROP the UNPHYSICAL (non-SPD, non-finite,
    |nu|>=nu_max — a statement about the network) -> rank the survivors by `_score` =
    `target_err_sim + SCORE_LAMBDA*solver_sim_gap` -> keep the best `keep` -> SAVE each to
    Phase 5/networks/design_<tag>_<rank>.npz.  Returns a list of report dicts.

    **The gap is a COST, not a veto** (2026-08-22). It says the two CODE PATHS disagree about a
    network, not that the network is unreal; vetoing on it discards designs that did the job (see
    SCORE_LAMBDA). `gap_tol` therefore no longer filters — it only sets the `trustworthy` FLAG on
    each report and annotates rejected ones. Kept designs may include some the codes disagree about,
    which is why every report carries `solver_sim_gap` alongside `target_err_sim`: two numbers,
    never merged into one verdict.

    Why rank by the sim, not the solver loss: a low solver loss can be a mirage on exotic/floppy
    topologies; the honest ranking is the independent simulation's distance to the requested target.
    To satisfy §10 criterion 5 a NON-Delaunay design is swapped in if none is kept.
    If NO candidate is trustworthy, we fall back to all verified (so the caller still gets results,
    flagged `trustworthy=False`)."""
    if pool is None:
        pool = topology_pool()

    # (i) cheap solver-loss ranking → a generous candidate set to verify
    ranked = _rank_pool(nu_target, E_target, pool, **design_kw)
    candidates = ranked[:max(2 * keep, keep + 3)]

    # (ii) INDEPENDENT-sim verification of each candidate.
    #      The sim self-protects by RAISING on a near-singular geometry (a sliver would otherwise
    #      segfault LAPACK uncatchably), so a single bad candidate used to abort the whole search —
    #      CLAUDE.md §3: "callers just try/except it" (audit A-4). Such a candidate is recorded as
    #      rejected and the search continues.
    verified, unhealthy = [], []
    for (geo, k, loss) in candidates:
        try:
            verified.append((geo, k, loss, verify(geo, k, nu_target, E_target)))
        except PH.UnhealthyGeometryError as e:
            unhealthy.append((geo, k, loss, str(e)))

    # (iii) SELECT: PHYSICALITY is the veto — non-SPD / non-finite / |nu|>=nu_max is a statement
    #       about the network itself, and it catches what the gap cannot (they can agree and still
    #       be unphysical, audit A-5). The solver-vs-sim GAP is NOT a veto (2026-08-22): it says the
    #       two code paths disagree, which is a reason to weigh a design down, not to pretend it
    #       does not exist. It enters the ranking via `_score` and is reported as its own number.
    #       Fall back to all verified if nothing qualifies, so a caller always gets results.
    ok = [v for v in verified if v[3]['physical_ok']]
    usable = ok if ok else verified

    # (iv) rank survivors by the SCORE (sim's distance to target + weighted disagreement)
    usable.sort(key=lambda v: _score(v[3]))
    kept = usable[:keep]

    # (v) ensure a (trustworthy) NON-Delaunay design appears (§10 criterion 5)
    kept_ids = {id(v) for v in kept}
    if not any(_is_nondelaunay(v[0]) for v in kept):
        flip = next((v for v in usable if _is_nondelaunay(v[0]) and id(v) not in kept_ids), None)
        if flip is not None:
            kept = (kept[:-1] + [flip]) if kept else [flip]

    # (v.5) vertex-position polish of the TOP design (positions.py; opt-out via
    #       optimize_positions=False).  Replaces the top design only if the polished one BOTH
    #       improves the design loss and passes the independent-sim honesty check.
    if optimize_positions and kept:
        import positions                              # lazy import — avoids import cycles
        budget = dict(n_outer=2, spsa_steps=25, n_iter=60, n_restarts=1)
        budget.update(pos_budget or {})
        nw = design_kw.get('nu_weight', 1.0)          # match the k-design's objective weighting so
        ew = design_kw.get('E_weight', 1.0)           # the polish doesn't optimise a different loss
        geo_t, k_t, loss_t, rep_t = kept[0]
        L0 = positions.loss_at(geo_t, k_t, nu_target, E_target, nw, ew)
        geoP, kP, _hist = positions.design_with_positions(nu_target, E_target, geo_t,
                                                          verbose=False, nu_weight=nw, E_weight=ew,
                                                          **budget)
        LP = positions.loss_at(geoP, kP, nu_target, E_target, nw, ew)
        if LP < L0:
            try:                                      # a polished-into-slivers geometry just means
                repP = verify(geoP, kP, nu_target, E_target)   # "keep the unpolished top" (A-4)
            except PH.UnhealthyGeometryError:
                repP = None
            # polished design replaces the top only if it is PHYSICAL and its SCORE improves —
            # previously it also had to clear `gap_tol`, which rejected polishes that reached the
            # target while the two codes drifted apart (same veto mistake as site iii).
            if (repP is not None and repP['physical_ok']
                    and _score(repP) < _score(kept[0][3])):
                geoP = _tag(geoP, str(geo_t.get('seed_name', 'unknown')) + '+pos')
                kept[0] = (geoP, kP, LP, repP)

    # (vi) save + report
    ndir = os.path.join(os.path.dirname(__file__), 'networks')
    os.makedirs(ndir, exist_ok=True)
    nu_t, E_t = _profile(nu_target), _profile(E_target)

    def _save(geo, k, loss, rep, path):
        """Persist one design + the numbers it was judged on (save-then-load: figures never
        re-optimise). Shared by the kept and the rejected sets — a rejected design is saved too."""
        mask = np.asarray(geo.get('is_fictional', np.zeros(len(geo['bond_R']), bool)), bool)
        C.apply_k_to_geo(geo, k)
        C.save_network(path, geo, k, C6_per=rep['C6_per'],
                       target_nu=nu_t.tolist(), target_E=E_t.tolist(),
                       loss=loss, solver_sim_gap=rep['solver_sim_gap'],
                       target_err_sim=rep['target_err_sim'],
                       target_err_solver=rep['target_err_solver'],
                       physical_ok=bool(rep['physical_ok']),
                       physical_failures=[f'{n}: {r}' for n, r in rep['physical_failures']],
                       seed_name=str(geo.get('seed_name', 'unknown')),
                       is_fictional=mask.tolist())

    def _report(rank, geo, k, loss, rep, path):
        return dict(rank=rank, path=path, loss=loss,
                    seed_name=str(geo.get('seed_name', 'unknown')),
                    is_nonDelaunay=_is_nondelaunay(geo),
                    trustworthy=bool(rep['solver_sim_gap'] < gap_tol),
                    physical_ok=bool(rep['physical_ok']),
                    physical_failures=list(rep['physical_failures']),
                    target_err_sim=rep['target_err_sim'],
                    target_err_sim_nu=rep['target_err_sim_nu'],
                    target_err_sim_E=rep['target_err_sim_E'],
                    target_err_solver=rep['target_err_solver'],
                    solver_sim_gap=rep['solver_sim_gap'],
                    nu_sim=rep['nu_sim'], E_sim=rep['E_sim'])

    reports = []
    for rank, (geo, k, loss, rep) in enumerate(kept):
        path = os.path.join(ndir, f'design_{tag}_{rank}.npz')
        _save(geo, k, loss, rep, path)
        reports.append(_report(rank, geo, k, loss, rep, path))

    # (vii) the REJECTED designs — kept ON THE SIDE, not discarded. A design dropped for being
    #       unphysical (or for a large gap, or for an unhealthy geometry) can still be the one worth
    #       looking at, and one that formally BEAT everything kept is the most interesting of all:
    #       `outscored_kept` says so explicitly rather than letting it vanish silently.
    kept_ids = {id(v) for v in kept}
    best_kept = min((r['target_err_sim'] for r in reports), default=np.inf)
    rejected = []
    for i, (geo, k, loss, rep) in enumerate([v for v in verified if id(v) not in kept_ids]):
        path = os.path.join(ndir, f'design_{tag}_rejected_{i}.npz')
        _save(geo, k, loss, rep, path)
        r = _report(None, geo, k, loss, rep, path)
        why = [f'{n}: {rs}' for n, rs in rep['physical_failures']]
        # The gap is NOT a rejection reason any more (it is scored). Still recorded, because a large
        # disagreement is the main thing a reader of a rejected design wants to know.
        if rep['solver_sim_gap'] >= gap_tol:
            why.append(f"(not a rejection) solver-vs-sim gap {rep['solver_sim_gap']:.4f} "
                       f">= gap_tol {gap_tol}")
        r['rejected_because'] = why or ['not in the top `keep` (no defect)']
        r['outscored_kept'] = bool(rep['target_err_sim'] < best_kept)
        rejected.append(r)
    for i, (geo, k, loss, msg) in enumerate(unhealthy):
        rejected.append(dict(rank=None, path=None, loss=loss,
                             seed_name=str(geo.get('seed_name', 'unknown')),
                             is_nonDelaunay=_is_nondelaunay(geo),
                             trustworthy=False, physical_ok=False,
                             physical_failures=[('geometry', msg)],
                             rejected_because=[f'unhealthy geometry: {msg}'],
                             outscored_kept=False, target_err_sim=np.inf))

    return DesignReports(reports, rejected)


# ---- self-test / demo ------------------------------------------------------------------------
def _demo():
    print("=" * 78)
    print("DESIGNER DEMO — target: isotropic auxetic  nu(theta) = -0.2 (flat), E(theta) = 1.0 (flat)")
    print("=" * 78)
    nu_target, E_target = -0.2, 1.0

    # bounded pool + moderate optimiser budget so the end-to-end run finishes in minutes
    pool = topology_pool(n_random=4, n_nodes=70, n_flip_variants=3, flips=6, seed=0)
    print(f"pool: {len(pool)} candidate topologies "
          f"({sum(_is_nondelaunay(g) for g in pool)} non-Delaunay / flipped)")

    reports = design(nu_target, E_target, tag='auxetic', pool=pool, keep=5,
                     n_iter=80, n_restarts=2, reg=0.02)

    # results table
    print("\n" + "-" * 92)
    print(f"{'rank':>4} {'seed_name':38s} {'nonDel?':>8} {'loss':>11} "
          f"{'tgt_err_sim':>12} {'solver_gap':>11}")
    print("-" * 92)
    for r in reports:
        print(f"{r['rank']:>4} {r['seed_name'][:38]:38s} {str(r['is_nonDelaunay']):>8} "
              f"{r['loss']:>11.3e} {r['target_err_sim']:>12.4f} {r['solver_sim_gap']:>11.4f}")
    print("-" * 92)

    # honest acceptance summary (§10) --------------------------------------------------------
    # design() no longer DROPS on the gap (it scores it), so the kept set may include designs the
    # two code paths disagree about — `trustworthy` below reports that, it does not define the set.
    trustworthy = [r for r in reports if r['trustworthy']]
    best = min(trustworthy or reports, key=lambda r: r['target_err_sim'])
    max_gap_kept = max((r['solver_sim_gap'] for r in reports), default=0.0)
    has_nondel = any(r['is_nonDelaunay'] for r in reports)

    print(f"\nsaved {len(reports)} designs ({len(trustworthy)} trustworthy):")
    for r in reports:
        print(f"    [{r['rank']}] {r['path']}")

    print(f"\nbest independent-sim design: rank {best['rank']} '{best['seed_name']}' "
          f"(non-Delaunay={best['is_nonDelaunay']}, trustworthy={best['trustworthy']})")
    print(f"    target_err_sim = {best['target_err_sim']:.4f}  = nu "
          f"{best['target_err_sim_nu']:.4f} + E {best['target_err_sim_E']:.4f}   "
          f"(RELATIVE, audit A-2 — not an absolute nu error)")
    print(f"    target_err_solver = {best['target_err_solver']:.4f}   "
          f"(the optimiser's own residual, CLAUDE.md §3(B) wants both)")
    print(f"    mean sim nu = {best['nu_sim'].mean():+.4f}  vs target {nu_target:+.4f}")
    print(f"    mean sim E  = {best['E_sim'].mean():+.4f}  vs target {E_target:+.4f}")
    print(f"    nu(theta) sim range: [{best['nu_sim'].min():+.4f}, {best['nu_sim'].max():+.4f}]")
    print(f"    E(theta)  sim range: [{best['E_sim'].min():+.4f}, {best['E_sim'].max():+.4f}]")
    print(f"max solver-vs-sim gap over KEPT (trustworthy) designs = {max_gap_kept:.4f}")

    # asserts: pipeline works, keeps only INDEPENDENTLY-VERIFIED designs, diverse, reaches non-Delaunay
    assert len(reports) >= 3, "expected SEVERAL alternative designs"
    assert trustworthy, "no trustworthy (small solver-vs-sim gap) design among the kept set"
    assert best['solver_sim_gap'] < GAP_TOL, \
        "best design is not trustworthy (solver-vs-sim gap too big)"
    assert has_nondel, "no NON-Delaunay (flipped) design among the kept set"
    # NB `target_err_sim` is now the RELATIVE, channel-summed error (audit A-2) — err_nu + err_E with
    # err_nu = max|dnu|/(|nu_t|+EPS_NU) — NOT the old absolute max(|dnu|,|dE|). This 0.1 is a
    # readable-report threshold only (never an assert), but the two channels are printed beside it so
    # the number cannot be mistaken for an absolute nu error the way the old wording invited.
    if best['target_err_sim'] < 0.1:
        print(f"\nOK: best trustworthy design's INDEPENDENT sim is within 0.1 RELATIVE of the target "
              f"(err={best['target_err_sim']:.4f} = nu {best['target_err_sim_nu']:.4f} "
              f"+ E {best['target_err_sim_E']:.4f}).")
    else:
        print(f"\nNOTE: the bounded pool did not reach nu={nu_target} within 0.1 relative "
              f"(err={best['target_err_sim']:.4f} = nu {best['target_err_sim_nu']:.4f} "
              f"+ E {best['target_err_sim_E']:.4f}; best sim nu mean = {best['nu_sim'].mean():+.4f}); "
              f"reporting best achieved honestly. Closest topology: '{best['seed_name']}'.")
    if getattr(reports, 'rejected', None):
        out = [r for r in reports.rejected if r.get('outscored_kept')]
        print(f"\n{len(reports.rejected)} candidate(s) rejected and kept on the side"
              + (f"; {len(out)} FORMALLY OUTSCORED the kept set:" if out else "."))
        for r in out:
            print(f"    '{r['seed_name']}' err={r['target_err_sim']:.4f} "
                  f"< best kept {best['target_err_sim']:.4f} — dropped: {'; '.join(r['rejected_because'])}")

    print(f"\nDESIGNER DEMO DONE — best independent-sim (nu,E) = "
          f"({best['nu_sim'].mean():+.4f}, {best['E_sim'].mean():+.4f}) vs target "
          f"({nu_target:+.4f}, {E_target:+.4f})")


if __name__ == '__main__':
    _demo()
