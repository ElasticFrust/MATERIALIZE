"""Gate for the designer's VERIFICATION SURFACE — audit A-2 … A-6.

These are the four behavioural changes made when closing A-2/A-3/A-4/A-5/A-6, each tested on
CONSTRUCTED inputs so the whole file runs in seconds — no topology search, no optimiser. The
expensive end-to-end path is covered separately by `designer.py`'s own `_demo` and by `sanity.py`.

  [1] metric      — the achieved-vs-target error is RELATIVE and channel-resolved, and shares
                    EPS_NU with the solver-vs-sim gap so the two headline numbers are on one scale.
                    Includes the A-2 REGRESSION: with the old absolute max(|dnu|,|dE|) a target with
                    E >> 1 let the E channel swamp nu entirely — and it was the RANKING key.
  [2] gate        — run_physicality_checks accepts a sound tensor and rejects non-SPD / |nu|>=nu_max
                    ones, naming the violated criterion.
  [3] registry    — PHYSICALITY_CHECKS is EXTENSIBLE: an appended check runs, and its failure
                    surfaces with its own name. ALL checks run (no short-circuit).
  [4] rejection   — design() PARTITIONS instead of filtering: an unphysical candidate is dropped
                    from `kept` but preserved on `.rejected` with the reason, flagged
                    `outscored_kept` when it formally beat everything kept; and a candidate whose
                    geometry the sim refuses (UnhealthyGeometryError) no longer aborts the search.

Run:  python test_designer_surface.py     (anaconda python — the stack is float64)
"""
import os
import sys

import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'Phase 3', 'verifications'))
import _common as C                                                    # noqa: E402
sys.path.insert(0, os.path.join(REPO, 'Phase 5'))
import designer                                                        # noqa: E402
import physical_homog as PH                                            # noqa: E402
from inverse_design import ANG                                         # noqa: E402

torch.set_default_dtype(torch.float64)
FLAT = np.ones_like(ANG)


# ---- [1] the metric ---------------------------------------------------------------------------
def test_metric():
    # hand-computed: nu off by 0.10 against a target of 0.20 -> 0.10/(0.20+0.05) = 0.4
    #                E  off by 0.20 against a target of 2.00 -> 0.20/2.00          = 0.1
    e_nu, e_E, e = designer._rel_err(0.30 * FLAT, 2.20 * FLAT, 0.20 * FLAT, 2.00 * FLAT)
    assert abs(e_nu - 0.4) < 1e-12 and abs(e_E - 0.1) < 1e-12 and abs(e - 0.5) < 1e-12, \
        f"hand-computed relative error mismatch: {e_nu}, {e_E}, {e}"

    # the metric takes the MAX over theta, not the mean: one bad angle must show up
    nu = 0.20 * FLAT.copy(); nu[7] = 0.45
    e_nu2, _, _ = designer._rel_err(nu, 2.0 * FLAT, 0.20 * FLAT, 2.0 * FLAT)
    assert abs(e_nu2 - 0.25 / 0.25) < 1e-12, f"max over theta not taken: {e_nu2}"

    # A-2 REGRESSION. Target nu=0.20, E=50 (a stiff target). A design that misses nu badly
    # (0.20 -> -0.30) but tracks E well must NOT look better than one that nails nu and misses E by
    # the same *relative* amount. Under the OLD absolute max(|dnu|,|dE|) the nu error (0.5) was
    # invisible next to any E error, so nu was effectively ignored while ranking.
    nu_t, E_t = 0.20 * FLAT, 50.0 * FLAT
    bad_nu = designer._rel_err(-0.30 * FLAT, 50.0 * FLAT, nu_t, E_t)[2]     # nu wrong, E perfect
    bad_E = designer._rel_err(0.20 * FLAT, 51.0 * FLAT, nu_t, E_t)[2]       # nu perfect, E 2% off
    assert bad_nu > bad_E, f"nu channel still swamped by E: {bad_nu:.4f} vs {bad_E:.4f}"
    old_bad_nu = max(np.abs(-0.30 - 0.20), np.abs(50.0 - 50.0))             # = 0.5
    old_bad_E = max(np.abs(0.20 - 0.20), np.abs(51.0 - 50.0))               # = 1.0  -> ranked WORSE
    assert old_bad_E > old_bad_nu, "the old absolute metric was supposed to mis-rank these"
    print(f"  [1] metric: relative + channel-resolved (hand-check exact); A-2 regression held "
          f"(new {bad_nu:.3f}>{bad_E:.3f}, old {old_bad_nu:.1f}<{old_bad_E:.1f})  OK")


# ---- [2] the physicality gate ----------------------------------------------------------------
def test_gate():
    from inverse_design import isotropic_c6
    ok, fails = designer.run_physicality_checks(np.asarray(isotropic_c6(0.3, 1.0), float))
    assert ok and not fails, f"a sound isotropic tensor was rejected: {fails}"

    bad = np.zeros(6)                       # all-zero tensor: not SPD
    ok_b, fails_b = designer.run_physicality_checks(bad)
    assert not ok_b and fails_b, "an all-zero (non-SPD) tensor passed the gate"
    assert any('SPD' in r or 'finite' in r for _, r in fails_b), \
        f"rejection reason should name SPD/finiteness, got {fails_b}"

    nan = np.full(6, np.nan)
    ok_n, fails_n = designer.run_physicality_checks(nan)
    assert not ok_n, "a non-finite tensor passed the gate"
    print(f"  [2] gate: sound tensor accepted; non-SPD rejected ({fails_b[0][1][:38]}...), "
          f"non-finite rejected  OK")


# ---- [3] the registry is extensible ----------------------------------------------------------
def test_registry():
    from inverse_design import isotropic_c6
    good = np.asarray(isotropic_c6(0.3, 1.0), float)
    seen = []

    def _always_fails(C6, geo=None, k=None, rep=None):
        seen.append('ran')
        return False, 'deliberate test failure'

    before = list(designer.PHYSICALITY_CHECKS)           # snapshot, NOT a hardcoded expectation:
    n_reg = len(before)                                  # the registry is meant to grow (A-17 added
                                                         # 'mesh'), so assert the INVARIANT instead
    checks = before + [('selftest', _always_fails)]
    ok, fails = designer.run_physicality_checks(good, checks=checks)
    assert seen == ['ran'], "an appended check did not run"
    assert not ok and ('selftest', 'deliberate test failure') in fails, \
        f"appended check's failure did not surface: {fails}"

    # ALL checks run (no short-circuit) so the log lists EVERY violated criterion. A zero tensor
    # fails 'realizable'; 'mesh' abstains without a geo; 'selftest' always fails.
    both = designer.run_physicality_checks(np.zeros(6), checks=checks)[1]
    assert len(both) >= 2, f"expected at least the realizable+selftest failures, got {both}"
    # the module registry itself is untouched by passing `checks`
    assert list(designer.PHYSICALITY_CHECKS) == before, "registry was mutated by passing checks="
    print(f"  [3] registry: appended check ran and surfaced; {len(both)} failures reported from "
          f"{n_reg}+1 checks; module registry unmutated  OK")


# ---- [4] design() partitions, and survives an unhealthy candidate -----------------------------
def _fake_rep(target_err, gap, physical_ok, failures=()):
    return dict(nu_sim=0.1 * FLAT, E_sim=1.0 * FLAT, nu_solver=0.1 * FLAT, E_solver=1.0 * FLAT,
                solver_sim_gap=gap, solver_sim_gap_nu=gap / 2, solver_sim_gap_E=gap / 2,
                C6_per=np.zeros((1, 6)), C6_bulk=np.zeros(6),
                physical_ok=physical_ok, physical_failures=list(failures),
                target_err_sim=target_err, target_err_sim_nu=target_err / 2,
                target_err_sim_E=target_err / 2, target_err_solver=target_err,
                target_err_solver_nu=target_err / 2, target_err_solver_E=target_err / 2)


def test_rejection_path():
    geos = [designer._tag(C.make_lattice(1.0, 1.0, half=3.0), f'selftest_{i}') for i in range(3)]
    ks = [np.ones(len(g['bond_u'])) for g in geos]

    # candidate 0: physical, decent target error   -> should be KEPT
    # candidate 1: UNPHYSICAL but a BETTER target error -> rejected, and outscored_kept
    # candidate 2: the sim refuses the geometry    -> rejected, must NOT abort the search
    reps = {id(geos[0]): _fake_rep(0.50, 0.01, True),
            id(geos[1]): _fake_rep(0.10, 0.01, False, [('realizable', 'not SPD (min eig -1e-03)')])}

    def fake_rank_pool(nu_t, E_t, pool, **kw):
        return [(g, k, 0.0) for g, k in zip(geos, ks)]

    def fake_verify(geo, k, nu_target=None, E_target=None):
        if id(geo) == id(geos[2]):
            raise PH.UnhealthyGeometryError('sliver triangle (selftest)')
        return reps[id(geo)]

    orig_rank, orig_verify = designer._rank_pool, designer.verify
    designer._rank_pool, designer.verify = fake_rank_pool, fake_verify
    try:
        reports = designer.design(0.1, 1.0, tag='_selftest', pool=list(geos), keep=1,
                                  optimize_positions=False)
    finally:
        designer._rank_pool, designer.verify = orig_rank, orig_verify

    assert len(reports) == 1, f"expected 1 kept design, got {len(reports)}"
    assert reports[0]['seed_name'] == 'selftest_0', \
        f"the UNPHYSICAL candidate was kept: {reports[0]['seed_name']}"
    assert reports[0]['physical_ok'], "kept design should be physical"

    rej = {r['seed_name']: r for r in reports.rejected}
    assert 'selftest_1' in rej, "the unphysical candidate was DISCARDED instead of kept on the side"
    assert rej['selftest_1']['outscored_kept'], \
        "the unphysical candidate formally beat the kept one and should say so"
    assert any('SPD' in w for w in rej['selftest_1']['rejected_because']), \
        f"reason should name the physicality failure: {rej['selftest_1']['rejected_because']}"
    assert 'selftest_2' in rej and any('unhealthy' in w.lower()
                                       for w in rej['selftest_2']['rejected_because']), \
        "the unhealthy-geometry candidate should be recorded as rejected"

    # the rejected-but-better design is SAVED, so it stays inspectable
    assert rej['selftest_1']['path'] and os.path.exists(rej['selftest_1']['path']), \
        "a rejected design should still be saved to disk"

    ndir = os.path.join(REPO, 'Phase 5', 'networks')
    for f in os.listdir(ndir):                          # clean up this test's own artifacts only
        if f.startswith('design__selftest_'):
            os.remove(os.path.join(ndir, f))
    print(f"  [4] rejection: unphysical dropped from kept but preserved (outscored_kept=True, "
          f"reason given, saved); unhealthy geometry did not abort the search  OK")


# ---- [5] the A-17 mesh preconditions are enforced --------------------------------------------
def test_mesh_preconditions():
    """The solver's OWN mesh gate (A-17): closed combinatorics + no inverted triangles.

    Validated against every mesh whose solver behaviour was MEASURED on 2026-08-17 — the gate must
    pass exactly those where the solver is exact and fail exactly those where it is wrong."""
    import mesh_build as MB
    import seeds
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import hex_solver_validation as HV
    import single_hexagon as SH

    # PASS: the solver agrees with the sim to 0.00000-1e-11 on all of these
    for lab, g in (('make_lattice', C.make_lattice(1.0, 1.0, half=3)),
                   ('build_geometry', MB.build_geometry(6, 0.2, 1)),
                   ('kagome', seeds.kagome(reps=3)['geo']),
                   ('centre hexagon d=0.6', HV.build_dhex(2, 2, 0.6)[0]),
                   ('chord hexagon d=1.5', HV.build_chords(2, 2, 1.5)[0])):
        ok, f = MB.check_mesh_preconditions(g, periodic=True)
        assert ok, f'{lab} should PASS the mesh gate, got {f}'

    # REPAIRED 2026-08-17: seeds now break the Delaunay tie (cocircular points) so these are valid
    # and EXACT (gap 0.0000000) where they used to read 0.22-0.38. They must PASS now.
    for lab, rec in (('square_octagon', seeds.seed_tiling('square_octagon', 3)),
                     ('rotating_squares', seeds._rotating_squares(reps=4, theta_deg=25.0)),
                     ('honeycomb', seeds.honeycomb(reps=4)),
                     # REPAIRED 2026-08-22 by the PHANTOM-CENTRE FAN (`seeds._fan_and_tag`). These
                     # used to be tagged mesh_ok=False and asserted to FAIL below: the Delaunay
                     # chord split produced edges that CROSSED each other (4 on honeycomb_r3, 5 on
                     # _r4, 2 on kagome_r2) — overlapping triangles, not a mesh. A fan cannot cross:
                     # every added edge joins a face's own centre to its own corner. The old comment
                     # here said "no tie-break gives both a manifold AND every native rib", which is
                     # true of DELAUNAY and is exactly why the representation was changed.
                     ('honeycomb tiling', seeds.seed_tiling('honeycomb', 3)),
                     ('kagome tiling', seeds.seed_tiling('kagome', 2))):
        ok, f = MB.check_mesh_preconditions(rec['geo'], periodic=True)
        assert ok, f'{lab} should be REPAIRED and PASS now, got {f}'
        assert rec.get('mesh_ok', True), f'{lab} should be tagged mesh_ok=True'

    # STILL not closed: `_reentrant_honeycomb` still uses the Delaunay chord split
    # (`_triangulate_and_tag`), so A-17's tail applies to it. TAGGED, not raised, so the gate
    # rejects it at point of use instead of crashing every driver that builds it. Giving it the fan
    # treatment is the obvious follow-up.
    for lab, rec in (('reentrant_honeycomb', seeds._reentrant_honeycomb(reps=4)),):
        ok, f = MB.check_mesh_preconditions(rec['geo'], periodic=True)
        assert not ok and any('not closed' in x or 'torus' in x for x in f), \
            f'{lab} should FAIL as not-closed, got {f}'
        assert rec.get('mesh_ok') is False, f'{lab} should be tagged mesh_ok=False, got {rec.get("mesh_ok")}'

    # FAIL (2) inverted: the chord triangulation folds once the hexagon is non-convex (gap 8.6)
    ok, f = MB.check_mesh_preconditions(HV.build_chords(2, 2, 0.6)[0], periodic=True)
    assert not ok and any('INVERTED' in x for x in f), f'folded chord mesh should FAIL, got {f}'

    # an OPEN mesh legitimately has boundary bonds in ONE triangle -- the single hexagon is EXACT
    tri, _ = SH.hexagon(2.0)
    om = MB.build_open_mesh(tri)
    assert MB.check_mesh_preconditions(om, periodic=False)[0], 'open hexagon should PASS when open'
    assert not MB.check_mesh_preconditions(om, periodic=True)[0], \
        'the periodic flag must matter: an open mesh is not closed'

    # and it is wired into the designer's registry
    names = [n for n, _ in designer.PHYSICALITY_CHECKS]
    assert 'mesh' in names, f'mesh check not registered: {names}'
    bad = HV.build_chords(2, 2, 0.6)[0]
    ok_all, fails = designer.run_physicality_checks(np.zeros(6), geo=bad)
    assert not ok_all and any(n == 'mesh' for n, _ in fails), \
        f'registry did not surface the mesh failure: {fails}'
    print('  [5] mesh preconditions: 7 good meshes pass (honeycomb+kagome tilings REPAIRED by '
          'the phantom-centre fan), 1 not-closed + 1 folded fail, open mesh '
          'passes only when judged open, registry wired  OK')


if __name__ == '__main__':
    print('designer verification-surface tests (audit A-2 ... A-6, A-17)')
    failed = 0
    for t in (test_metric, test_gate, test_registry, test_rejection_path, test_mesh_preconditions):
        try:
            t()
        except Exception as e:
            failed += 1
            print(f"  FAIL {t.__name__}: {type(e).__name__}: {e}")
    print('ALL PASSED' if not failed else f'{failed} TEST(S) FAILED')
    sys.exit(1 if failed else 0)
