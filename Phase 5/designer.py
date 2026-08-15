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
    design(nu_target, E_target, tag, pool=None, keep=5, **design_kw) -> list[report]

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


# ---- small helpers ---------------------------------------------------------------------------
def _profile(x):
    """Broadcast a scalar or length-37 target into a numpy profile over ANG."""
    return np.array(np.broadcast_to(np.asarray(x, float), ANG.shape), float)


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

    # RELATIVE solver-vs-sim gap (honesty check): E strictly relative (E>0); nu relative with a
    # floor eps_nu since nu is O(1) and can cross 0. At nu->0 the absolute nu-floor is gap_tol*eps_nu,
    # so eps_nu and the tolerance are coupled. See CLAUDE.md 3 (verification discipline).
    eps_nu = 0.05
    gap_nu = float((np.abs(nu_sim - nu_slv) / (np.abs(nu_sim) + eps_nu)).max())
    gap_E  = float((np.abs(E_sim - E_slv) / np.maximum(np.abs(E_sim), 1e-12)).max())
    gap = gap_nu + gap_E
    rep = dict(nu_sim=nu_sim, E_sim=E_sim, nu_solver=nu_slv, E_solver=E_slv,
               solver_sim_gap=gap, C6_per=C6_per)
    if nu_target is not None and E_target is not None:
        nu_t, E_t = _profile(nu_target), _profile(E_target)
        rep['target_err_sim'] = float(max(np.abs(nu_sim - nu_t).max(),
                                          np.abs(E_sim - E_t).max()))
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
def design(nu_target, E_target, tag, pool=None, keep=5, gap_tol=0.05,
           optimize_positions=True, pos_budget=None, **design_kw):
    """Design several distinct networks for a target nu(theta),E(theta).

    `optimize_positions` (default True): after selecting the kept designs, additionally run the
    derivative-free VERTEX-POSITION optimization (positions.design_with_positions) on the TOP
    design only (budget); if it improves the design loss AND still passes the independent-sim
    gap check it replaces the top design (seed_name gets '+pos').  `pos_budget` is an optional
    dict overriding the modest default polish budget
    dict(n_outer=2, spsa_steps=25, n_iter=60, n_restarts=1).

    Pipeline: build/accept a pool -> rank topologies by (cheap, solver) design loss -> VERIFY a
    generous candidate set with the INDEPENDENT sim -> DROP untrustworthy designs (solver-vs-sim
    `gap >= gap_tol`: e.g. soft-edged tilings that optimise to a low solver loss but are really
    near-mechanisms the solver mis-predicts) -> rank the trustworthy survivors by how well the
    INDEPENDENT SIM hits the target (`target_err_sim`) -> keep the best `keep` -> SAVE each to
    Phase 5/networks/design_<tag>_<rank>.npz.  Returns a list of report dicts.

    Why rank by the sim, not the solver loss: a low solver loss can be a mirage on exotic/floppy
    topologies; the honest ranking is the independent simulation's distance to the requested target.
    To satisfy §10 criterion 5 a (trustworthy) NON-Delaunay design is swapped in if none is kept.
    If NO candidate is trustworthy, we fall back to all verified (so the caller still gets results,
    flagged `trustworthy=False`)."""
    if pool is None:
        pool = topology_pool()

    # (i) cheap solver-loss ranking → a generous candidate set to verify
    ranked = _rank_pool(nu_target, E_target, pool, **design_kw)
    candidates = ranked[:max(2 * keep, keep + 3)]

    # (ii) INDEPENDENT-sim verification of each candidate
    verified = [(geo, k, loss, verify(geo, k, nu_target, E_target))
                for (geo, k, loss) in candidates]

    # (iii) drop untrustworthy (large solver-vs-sim gap); fall back to all if none survive
    trust = [v for v in verified if v[3]['solver_sim_gap'] < gap_tol]
    usable = trust if trust else verified

    # (iv) rank survivors by how well the INDEPENDENT sim hits the target
    usable.sort(key=lambda v: v[3]['target_err_sim'])
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
            repP = verify(geoP, kP, nu_target, E_target)
            if repP['solver_sim_gap'] < gap_tol:
                geoP = _tag(geoP, str(geo_t.get('seed_name', 'unknown')) + '+pos')
                kept[0] = (geoP, kP, LP, repP)

    # (vi) save + report
    ndir = os.path.join(os.path.dirname(__file__), 'networks')
    os.makedirs(ndir, exist_ok=True)
    nu_t, E_t = _profile(nu_target), _profile(E_target)

    reports = []
    for rank, (geo, k, loss, rep) in enumerate(kept):
        mask = np.asarray(geo.get('is_fictional', np.zeros(len(geo['bond_R']), bool)), bool)
        path = os.path.join(ndir, f'design_{tag}_{rank}.npz')
        C.apply_k_to_geo(geo, k)
        C.save_network(path, geo, k, C6_per=rep['C6_per'],
                       target_nu=nu_t.tolist(), target_E=E_t.tolist(),
                       loss=loss, solver_sim_gap=rep['solver_sim_gap'],
                       target_err_sim=rep['target_err_sim'],
                       seed_name=str(geo.get('seed_name', 'unknown')),
                       is_fictional=mask.tolist())
        reports.append(dict(rank=rank, path=path, loss=loss,
                            seed_name=str(geo.get('seed_name', 'unknown')),
                            is_nonDelaunay=_is_nondelaunay(geo),
                            trustworthy=bool(rep['solver_sim_gap'] < gap_tol),
                            target_err_sim=rep['target_err_sim'],
                            solver_sim_gap=rep['solver_sim_gap'],
                            nu_sim=rep['nu_sim'], E_sim=rep['E_sim']))
    return reports


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
    # design() already dropped untrustworthy designs (large solver-vs-sim gap), so the kept set is
    # trustworthy; rank the honest best by the INDEPENDENT sim's distance to the target.
    trustworthy = [r for r in reports if r['trustworthy']]
    best = min(trustworthy or reports, key=lambda r: r['target_err_sim'])
    max_gap_kept = max((r['solver_sim_gap'] for r in reports), default=0.0)
    has_nondel = any(r['is_nonDelaunay'] for r in reports)

    print(f"\nsaved {len(reports)} designs ({len(trustworthy)} trustworthy):")
    for r in reports:
        print(f"    [{r['rank']}] {r['path']}")

    print(f"\nbest independent-sim design: rank {best['rank']} '{best['seed_name']}' "
          f"(non-Delaunay={best['is_nonDelaunay']}, trustworthy={best['trustworthy']})")
    print(f"    target_err_sim = {best['target_err_sim']:.4f}  "
          f"(aim < ~0.1 on nu for this hard global auxetic target)")
    print(f"    mean sim nu = {best['nu_sim'].mean():+.4f}  vs target {nu_target:+.4f}")
    print(f"    mean sim E  = {best['E_sim'].mean():+.4f}  vs target {E_target:+.4f}")
    print(f"    nu(theta) sim range: [{best['nu_sim'].min():+.4f}, {best['nu_sim'].max():+.4f}]")
    print(f"    E(theta)  sim range: [{best['E_sim'].min():+.4f}, {best['E_sim'].max():+.4f}]")
    print(f"max solver-vs-sim gap over KEPT (trustworthy) designs = {max_gap_kept:.4f}")

    # asserts: pipeline works, keeps only INDEPENDENTLY-VERIFIED designs, diverse, reaches non-Delaunay
    assert len(reports) >= 3, "expected SEVERAL alternative designs"
    assert trustworthy, "no trustworthy (small solver-vs-sim gap) design among the kept set"
    assert best['solver_sim_gap'] < 0.05, "best design is not trustworthy (solver-vs-sim gap too big)"
    assert has_nondel, "no NON-Delaunay (flipped) design among the kept set"
    if best['target_err_sim'] < 0.1:
        print("\nOK: the best trustworthy design's INDEPENDENT sim hits the auxetic target within ~0.1.")
    else:
        print(f"\nNOTE: the bounded pool did not reach nu={nu_target} within 0.1 "
              f"(best sim nu mean = {best['nu_sim'].mean():+.4f}); reporting best achieved honestly. "
              f"Closest topology: '{best['seed_name']}'.")

    print(f"\nDESIGNER DEMO DONE — best independent-sim (nu,E) = "
          f"({best['nu_sim'].mean():+.4f}, {best['E_sim'].mean():+.4f}) vs target "
          f"({nu_target:+.4f}, {E_target:+.4f})")


if __name__ == '__main__':
    _demo()
