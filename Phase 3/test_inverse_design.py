"""
Tests for Phase 3 inverse design (inverse_design.py). Run: python "Phase 3/test_inverse_design.py"

Covers: round-trip recovery, property (nu incl. auxetic) targeting, full-tensor matching, a LOCAL
region target, a MIXED global+local design, a large-N (>600 tri, adjoint) design, an open mesh, and
directional ν(θ)/E(θ) profile design (program / isotropise / flat-E).
"""
import os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(HERE, 'verifications'))
import Disc_2_Cont_optimized as D2C
from inverse_design import (DesignProblem, Objective, optimize, validate,
                            per_triangle_strain_stress, region_mean_vec3)
import _common as C
import physical_homog as PH
import sim_assembly as SA
torch.set_default_dtype(torch.float64)


def test_round_trip():
    prob = DesignProblem.periodic(N=14, eta=0.3, seed=0)
    rng = torch.Generator().manual_seed(1)
    k_true = 0.3 + 1.5 * torch.rand(prob.n_bond, generator=rng)
    tgt = prob.region_tensor(prob.forward(k_true)['per_triangle'], None).detach()
    # reg EXPLICITLY 0.0 (CLAUDE.md §3 "you MUST pass reg" = state it, not default into it): the
    # target is built FROM k_true, so this is a RECOVERY test. reg penalises k-variance, and k_true
    # is a random spread — any reg>0 biases the optimum away from the solution being recovered.
    res = optimize(prob, [Objective('tensor', tgt)], mode='k', n_iter=120, reg=0.0, verbose=False)
    err = validate(prob, res['k'], res['l0'], [Objective('tensor', tgt)])[0]['err']
    assert err < 1e-3, f"round-trip tensor err {err:.2e}"
    print(f"  [1] round-trip: full-tensor err={err:.2e}  OK")


def test_property_auxetic():
    # reg per CLAUDE.md 3 (never leave the reg=0.0 default: k drifts into soft channels -- here it
    # collapsed to k_min/mean ~1e-10). reg alone is NOT enough on this target: from a single init
    # LBFGS lands in a bad basin at every reg, so take the restarts the designer itself uses.
    prob = DesignProblem.periodic(N=14, eta=0.3, seed=2)
    objs = [Objective('nu', target=-0.20)]
    res = optimize(prob, objs, mode='k', n_iter=100, n_restarts=3, reg=0.02, verbose=False)
    rep = validate(prob, res['k'], res['l0'], objs)[0]
    assert rep['err'] < 0.02, f"nu achieved {rep['achieved']:.3f} vs target -0.20"
    print(f"  [2] auxetic target nu=-0.20: achieved {rep['achieved']:+.3f}  OK")


def test_property_E():
    prob = DesignProblem.periodic(N=14, eta=0.2, seed=3)
    base = validate(prob, torch.ones(prob.n_bond), None, [Objective('E', 0.0)])[0]['achieved']
    tgtE = 0.6 * base
    res = optimize(prob, [Objective('E', tgtE)], mode='k', n_iter=100, reg=0.02, verbose=False)
    rep = validate(prob, res['k'], res['l0'], [Objective('E', tgtE)])[0]
    assert rep['err'] / tgtE < 0.03, f"E achieved {rep['achieved']:.3f} vs {tgtE:.3f}"
    print(f"  [3] E target {tgtE:.3f} (0.6*base): achieved {rep['achieved']:.3f}  OK")


def test_local_region():
    # `region` (not `reg`) -- `reg` is optimize()'s regularisation weight, which this now passes.
    # At the reg=0.0 default the outcome is bistable: repeated identical runs give err 0.0000 or
    # 0.0664 (a near-mechanism, k_min/mean ~1e-8). reg=0.01-0.02 is correct and stable AT THE
    # DEFAULT THREAD COUNT ONLY -- measured 2026-08-23, this test is bistable in the BLAS thread
    # count: threads=1 lands at nu=-0.1279 (err 0.0966, FAIL) and threads=4 at -0.1499 (err
    # 0.000215) on the same seed and commit, 17/17 vs 15/15 across two overnight campaigns and
    # confirmed in isolation by `verifications/b1_thread_local.py`. A ~1e-33 threading difference
    # in the forward path is amplified by 120 L-BFGS iterations into a different basin -- which is
    # why M2's dataset generator must PIN the thread count. reg=0.05 over-constrains (err 0.11).
    prob = DesignProblem.periodic(N=16, eta=0.2, seed=4)
    c = prob.centroids.mean(0)
    region = prob.region_in_circle(c, radius=0.20 * (prob.centroids[:, 0].max() - prob.centroids[:, 0].min()))
    objs = [Objective('nu', target=-0.15, region=region, weight=1.0)]
    res = optimize(prob, objs, mode='k', n_iter=120, reg=0.02, verbose=False)
    rep = validate(prob, res['k'], res['l0'], objs)[0]
    assert rep['err'] < 0.03, f"local nu achieved {rep['achieved']:.3f} vs -0.15"
    print(f"  [4] local region ({len(region)} tri) nu=-0.15: achieved {rep['achieved']:+.3f}  OK")


def test_mixed_global_local():
    prob = DesignProblem.periodic(N=18, eta=0.2, seed=5)
    c = prob.centroids.mean(0)
    reg = prob.region_in_circle(c, radius=0.18 * (prob.centroids[:, 0].max() - prob.centroids[:, 0].min()))
    objs = [Objective('nu', target=+0.25, region=None, weight=1.0),          # global positive
            Objective('nu', target=-0.20, region=reg, weight=2.0)]           # local auxetic patch
    res = optimize(prob, objs, mode='k', n_iter=150, reg=0.02, verbose=False)
    rep = validate(prob, res['k'], res['l0'], objs)
    g, l = rep[0], rep[1]
    assert g['err'] < 0.05 and l['err'] < 0.05, f"global {g['achieved']:.3f}, local {l['achieved']:.3f}"
    print(f"  [5] mixed: global nu={g['achieved']:+.3f} (->+0.25), "
          f"local patch nu={l['achieved']:+.3f} (->-0.20)  OK")


def test_large_N_adjoint():
    prob = DesignProblem.periodic(N=20, eta=0.3, seed=6)          # 800 tri > 600 -> adjoint path
    assert prob.n_tri > 600
    objs = [Objective('nu', target=0.0)]
    res = optimize(prob, objs, mode='k', n_iter=60, reg=0.02, verbose=False)
    rep = validate(prob, res['k'], res['l0'], objs)[0]
    assert res['history'][-1] < res['history'][0] and rep['err'] < 0.03, \
        f"large-N nu achieved {rep['achieved']:.3f}"
    print(f"  [6] large-N adjoint ({prob.n_tri} tri) nu=0: achieved {rep['achieved']:+.3f}  OK")


def test_open_domain():
    np.random.seed(7)
    tri = D2C.generate_foam_points((5, 5), 0.2)
    prob = DesignProblem.open(tri)
    objs = [Objective('nu', target=0.10)]
    res = optimize(prob, objs, mode='k', n_iter=100, reg=0.02, verbose=False)
    rep = validate(prob, res['k'], res['l0'], objs)[0]
    assert rep['err'] < 0.03, f"open nu achieved {rep['achieved']:.3f} vs 0.10"
    print(f"  [7] open mesh ({prob.n_tri} tri) nu=0.10: achieved {rep['achieved']:+.3f}  OK")


def test_directional():
    """Directional ν(θ)/E(θ) objectives: program a profile, isotropise to a flat level, flat E."""
    from inverse_design import ANG
    prob = DesignProblem.periodic(N=16, eta=0.3, seed=8)
    prof = 0.25 + 0.25 * np.cos(4 * ANG)                          # (a) program a REALIZABLE 4-fold ν(θ)
    o = [Objective('nu_theta', prof)]                            # (pure cos2θ is OFF the ν(θ) manifold)
    res = optimize(prob, o, mode='k', n_iter=140, reg=0.02, verbose=False)
    ea = validate(prob, res['k'], res['l0'], o)[0]['err']
    assert ea < 0.09, f"nu(theta) profile maxerr {ea:.3f}"
    o2 = [Objective('nu_theta', -0.10)]                          # (b) isotropise to flat ν0=-0.1
    res2 = optimize(prob, o2, mode='k', n_iter=150, reg=0.02, verbose=False)
    got = validate(prob, res2['k'], res2['l0'], o2)[0]['achieved']
    spread, mean = float(np.ptp(got)), float(got.mean())
    assert spread < 0.06 and abs(mean + 0.10) < 0.03, f"isotropise spread {spread:.3f} mean {mean:.3f}"
    o3 = [Objective('E_theta', 0.9)]                             # (c) flat E(θ) target
    res3 = optimize(prob, o3, mode='k', n_iter=100, reg=0.02, verbose=False)
    gE = validate(prob, res3['k'], res3['l0'], o3)[0]['achieved']
    assert abs(gE.mean() - 0.9) < 0.12, f"E(theta) flat mean {gE.mean():.3f}"
    print(f"  [8] directional: nu-profile maxerr={ea:.3f}, isotropise spread={spread:.3f}@{mean:+.3f}, "
          f"E-flat mean={gE.mean():.3f}  OK")


def test_constrain_isotropic():
    """EXACT (direction-complete) local constraints give an ISOTROPIC response; the scalar 'ish' does not."""
    from inverse_design import constrain, isotropic_c6, c6_to_nu_theta, ANG
    prob = DesignProblem.periodic(N=16, eta=0.3, seed=9)
    c = prob.centroids.mean(0)
    patch = prob.region_in_circle(c, 0.22 * (prob.centroids[:, 0].max() - prob.centroids[:, 0].min()))

    def nu_spread(k):
        C6 = prob.region_tensor(prob.forward(k)['per_triangle'], patch)
        nu = c6_to_nu_theta(C6, ANG).detach().numpy()
        return float(np.ptp(nu)), float(nu.mean())

    # reg per CLAUDE.md 3 -- at the reg=0.0 default this lands on a near-mechanism (spread 0.14,
    # mean -0.47 instead of a flat -0.2); reg=0.02 alone restores it (spread 1e-4).
    # legacy single-direction 'nu_dir' leaves the patch anisotropic; scalar 'nu' (=isotropic) fixes it
    ish = optimize(prob, [Objective('nu_dir', -0.2, region=patch)], mode='k', n_iter=140,
                   reg=0.02, verbose=False)
    sp_ish, _ = nu_spread(ish['k'])
    ex = optimize(prob, [Objective('nu', -0.2, region=patch)], mode='k', n_iter=160,
                  reg=0.02, verbose=False)
    sp, mn = nu_spread(ex['k'])
    assert sp < 0.05 and abs(mn + 0.2) < 0.03 and sp < sp_ish, \
        f"exact flat ν spread {sp:.3f} mean {mn:.3f} (ish spread {sp_ish:.3f})"
    # exact isotropic tensor (ν AND E) -> tiny isotropy residual
    it = optimize(prob, constrain(region=patch, tensor=isotropic_c6(-0.2, 0.6)), mode='k',
                  n_iter=180, reg=0.02, verbose=False)
    iso = validate(prob, it['k'], None, [Objective('isotropy', region=patch)])[0]['err']
    assert iso < 0.06, f"isotropy residual {iso:.3f}"
    print(f"  [9] constrain: exact flat nu spread={sp:.3f}@{mn:+.3f} (ish={sp_ish:.2f}), "
          f"isotropic-tensor residual={iso:.3f}  OK")


def test_strain_stress_equivalence():
    """The differentiable per_triangle_strain_stress(out['bare'], out['W'], load) must reproduce
    _common.unit_mode_response's INDEPENDENT NumPy ground truth (physical_homog.relax), for each of
    the 3 unit macro modes, on a mesh with a random (non-uniform) k. Residual is expected at O(DELTA)
    -- unit_mode_response's tri_metric_change is the EXACT nonlinear metric change Fs.T@Fs-I
    (evaluated at the linearly-relaxed displacement), while the solver's W is a LINEAR-response
    strain-concentration operator; they agree only to leading (linear) order in the small macro
    perturbation PH.DELTA=1e-3. A near-zero residual here would actually indicate a convention
    MISMATCH masked by coincidence, not a better match -- see PH.DELTA-scaled tolerance below."""
    geo = C.make_lattice(1.0, 1.0, half=6.0, eta=0.15, seed=1)
    rng = np.random.default_rng(2)
    k = 0.5 + rng.random(len(geo['bond_u']))
    geo['bond_k'] = k; geo['tri_k'] = k[geo['tri_bond']]
    prob = DesignProblem.from_geo(geo)
    out = prob.forward(torch.as_tensor(k), physical_units=True)
    bare, W = out['bare'], out['W']

    eps_ref, sig_ref = C.unit_mode_response(geo)               # ground truth, list of 3 (nt,2,2)
    Dgt = [F.T @ F - np.eye(2) for F in PH.Fk]
    maxrel = 0.0
    for mode in range(3):
        load_dg = C.MO.vec3(Dgt[mode]) / PH.DELTA              # linear-normalised, matches eps_ref's scale
        strain_t, stress_t = per_triangle_strain_stress(bare, W, load_dg)
        eps_ref_vec3 = C.MO.vec3(eps_ref[mode])
        sig_ref_vec3 = np.stack([sig_ref[mode][:, 0, 0], sig_ref[mode][:, 0, 1], sig_ref[mode][:, 1, 1]], -1)
        de = np.abs(strain_t.detach().numpy() - eps_ref_vec3).max() / np.abs(eps_ref_vec3).max()
        ds = np.abs(stress_t.detach().numpy() - sig_ref_vec3).max() / np.abs(sig_ref_vec3).max()
        maxrel = max(maxrel, de, ds)
    assert maxrel < 5 * PH.DELTA, f"strain/stress relative residual {maxrel:.2e} >> O(DELTA)={PH.DELTA:.0e}"
    print(f"  [10] strain/stress vs unit_mode_response: max relative residual={maxrel:.2e} "
          f"(O(DELTA)={PH.DELTA:.0e} expected)  OK")


def test_strain_stress_autograd():
    """Autograd matches finite differences for d(loss)/d(raw_k) through a 'stress' (global) and a
    'strain' (sub-region) Objective -- mirrors Phase 2/test_forward_solver.py's autograd-vs-FD check,
    exercising the actual _loss branches (not just the per_triangle_strain_stress helper directly)."""
    from inverse_design import _loss, _params_to_kl, _init_raw
    prob = DesignProblem.periodic(N=10, eta=0.2, seed=7)
    reg = prob.region_in_circle(prob.centroids.mean(0),
                                0.25 * (prob.centroids[:, 0].max() - prob.centroids[:, 0].min()))
    load = np.array([1.0, 0.0, 0.0])
    cases = [('stress', None, torch.tensor([0.1, 0.0, 0.02])),
             ('strain', reg, torch.tensor([0.9, 0.0, -0.1]))]
    for kind, region, target in cases:
        objs = [Objective(kind, target=target, region=region, load=load)]
        raw_k = _init_raw(prob, 'k', seed=0)['k'].detach().clone().requires_grad_(True)

        def loss_of(raw_k_val):
            k_bond, l0_bond = _params_to_kl({'k': raw_k_val}, prob, 'k')
            return _loss(prob, objs, k_bond, l0_bond)

        loss_of(raw_k).backward()
        g_auto = raw_k.grad[0].item()
        eps = 1e-5
        with torch.no_grad():
            kp = raw_k.detach().clone(); kp[0] += eps
            km = raw_k.detach().clone(); km[0] -= eps
            g_fd = (loss_of(kp).item() - loss_of(km).item()) / (2 * eps)
        denom = max(1.0, abs(g_fd))
        assert abs(g_auto - g_fd) / denom < 1e-3, f"{kind}: autograd {g_auto} vs FD {g_fd}"
        print(f"  [11] {kind} autograd vs finite-diff: {g_auto:.6e} vs {g_fd:.6e}  OK")


def test_strain_stress_design():
    """Design k to a REACHABLE per-triangle stress target (global) and strain target (sub-region) --
    reachability established by computing the target FROM an actual k_true (mirrors test_round_trip).
    Whole-cell strain is degenerate (region-mean strain = load exactly), so the strain case uses a
    sub-region; stress has no such degeneracy and is targeted globally."""
    prob = DesignProblem.periodic(N=12, eta=0.2, seed=11)
    load = np.array([1.0, 0.0, 0.3])
    rng = torch.Generator().manual_seed(12)
    k_true = 0.4 + 1.2 * torch.rand(prob.n_bond, generator=rng)
    out_true = prob.forward(k_true, physical_units=True)
    strain_true, stress_true = per_triangle_strain_stress(out_true['bare'], out_true['W'], load)

    stress_target = region_mean_vec3(stress_true, None).detach()
    objs = [Objective('stress', target=stress_target, region=None, load=load)]
    # reg EXPLICITLY 0.0, same reason as test_round_trip: both targets here are computed FROM
    # k_true, so these are recovery problems and a k-variance penalty biases away from the answer.
    res = optimize(prob, objs, mode='k', n_iter=150, reg=0.0, verbose=False)
    rep = validate(prob, res['k'], res['l0'], objs)[0]
    # matching all 3 stress-vec3 components simultaneously converges to a few % relative (a coarse
    # 3-number summary of a high-dim k has many near-equally-good solutions; not full k_true recovery)
    assert rep['err'] < 0.06 * float(stress_target.abs().max()), f"stress err {rep['err']:.4f}"
    print(f"  [12] stress design (global): err={rep['err']:.4f}  OK")

    reg = prob.region_in_circle(prob.centroids.mean(0),
                                0.25 * (prob.centroids[:, 0].max() - prob.centroids[:, 0].min()))
    strain_target = region_mean_vec3(strain_true, reg).detach()
    objs2 = [Objective('strain', target=strain_target, region=reg, load=load)]
    res2 = optimize(prob, objs2, mode='k', n_iter=150, reg=0.0, verbose=False)
    rep2 = validate(prob, res2['k'], res2['l0'], objs2)[0]
    assert rep2['err'] < 0.05 * float(strain_target.abs().max()), f"strain err {rep2['err']:.4f}"
    print(f"  [13] strain design (sub-region, {len(reg)} tri): err={rep2['err']:.4f}  OK")


def test_homogeneity_regularizer():
    """A demanding regional 'nu' target (auxetic -0.5 on a sub-region) with homogeneity=0 lets the
    optimiser satisfy the region-MEAN via a few very-different (hinge-like) triangles -- large
    per-triangle nu variance within the region even though the mean is on target. homogeneity>0
    should shrink that variance sharply while barely moving the achieved mean off target. Calibrated
    empirically (see plan) -- min(k)==0 turned out NOT to be a reliable indicator (softplus
    underflows to exactly 0 for very negative raw-k regardless, and is common even in successful
    designs elsewhere in this repo, e.g. two_region/ribbon.py's own auxetic patch); the per-triangle
    nu VARIANCE within the region is the direct, honest metric the regularizer actually controls."""
    from inverse_design import c6_to_nuE, _region_rows
    prob = DesignProblem.periodic(N=14, eta=0.25, seed=20)
    reg = prob.region_in_circle(prob.centroids.mean(0),
                                0.20 * (prob.centroids[:, 0].max() - prob.centroids[:, 0].min()))

    def region_nu_var(k):
        """Per-triangle nu variance within `reg`, robust to the rare near-mechanism (near-singular
        C6, e.g. from a near-zero-k triangle) triangle producing a non-finite nu -- an occasional,
        expected edge case for a demanding regional target, not a sign homogeneity() is broken."""
        with torch.no_grad():
            nu_pertri, _ = c6_to_nuE(prob.forward(k, physical_units=True)['per_triangle'])
            vals = _region_rows(nu_pertri, reg)
            finite = vals[torch.isfinite(vals)]
            return float(finite.var())

    # reg EXPLICITLY 0.0 on BOTH arms, and deliberately so. `reg` penalises k-VARIANCE, which
    # suppresses per-triangle nu variance as a side effect — the very quantity this test measures.
    # Turning it on would confound the homogeneity effect with the reg effect and would also weaken
    # the res0 CONTROL, whose whole job is to be the hinge-like high-variance baseline. The two arms
    # must differ ONLY in `homogeneity`. (Near-mechanism risk is handled by n_restarts on res1.)
    res0 = optimize(prob, [Objective('nu', target=-0.5, region=reg, weight=1.0)],
                    mode='k', n_iter=150, reg=0.0, verbose=False)
    var0 = region_nu_var(res0['k'])

    objs1 = [Objective('nu', target=-0.5, region=reg, weight=1.0, homogeneity=1.0)]
    # n_restarts>1: this demanding regional target occasionally lands LBFGS in a visibly worse local
    # optimum on a single run (observed run-to-run, even at a fixed seed -- LBFGS/threading
    # nondeterminism); n_restarts is this codebase's own mechanism for exactly that, so use it rather
    # than just loosening the achieved-error tolerance to paper over an unlucky single run. Even with
    # n_restarts=3, repeated calibration runs showed achieved err mostly ~0.00-0.01 with an occasional
    # outlier up to ~0.09 -- the 0.12 bound below has margin over that observed tail, not a blind guess.
    res1 = optimize(prob, objs1, mode='k', n_iter=150, n_restarts=4, reg=0.0, verbose=False)
    rep1 = validate(prob, res1['k'], res1['l0'], objs1)[0]
    var1 = region_nu_var(res1['k'])

    assert var1 < 0.3 * var0, f"homogeneity didn't shrink region nu-variance: {var1:.4f} vs {var0:.4f}"
    assert rep1['err'] < 0.12, f"homogeneity=1.0 pulled achieved nu off target: err={rep1['err']:.3f}"
    print(f"  [14] homogeneity regularizer: region nu-variance {var0:.4f} -> {var1:.4f} "
          f"(achieved nu={rep1['achieved']:+.3f})  OK")


# ---- audit B-1: always-on anomaly capture for [15] -------------------------------------------
# B-1 stage 3: this comparison intermittently reads ~1.4e-04 (once ~1.17e-02) instead of its usual
# 1.6e-12 — measured rate ~1 in 21 suite runs, 2026-08-16. It needs SUITE CONTEXT (0 anomalies in 30
# isolated processes) and is stochastic GIVEN that context, so a bisect cannot corner it: two full
# bisects both drew clean. Cause unknown; ruled out by measurement: the A-10 change, ill-conditioning
# (relative condition number 0.05-0.08 w.r.t. k), perturbing G through the singular lstsq (1-3x, no
# amplification even at sigma_min=4.7e-17), a Delaunay flip, a float32 leak, make_lattice caching,
# method switching, a try/except fallback, the 500/600 threshold window, and state from any of the 13
# preceding tests.
#
# [15] only ever reported max-over-the-3-cases, so WHICH case deviates and whether the deviation is
# ONE tensor component or diffuse has never been recorded — the sharpest unused clue. This captures
# exactly that, costs nothing when nothing is wrong, and lets the verification campaign collect the
# diagnostic for free instead of buying dedicated runs. Threshold 1e-9 sits ~600x above the worst
# healthy case (1.6e-12) and ~1e5 below the anomaly.
B1_DUMP_THRESHOLD = 1e-9
B1_DUMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'verifications', 'b1_dumps')


class _B1Capture:
    """Record the INPUTS of the solver's internal linear algebra during one forward solve, so that
    an anomalous [15] can be diagnosed instead of merely counted.

    WHY THIS EXISTS (2026-08-23). The first overnight campaign caught two excursions and refuted
    three candidate mechanisms, but could not identify the cause, because the dump recorded the
    wrong OUTPUT and nothing upstream — and every follow-up probe had to run in a clean isolated
    process, where B-1 never occurs. Capturing `G`, the constraint solve and the batched inverse at
    the MOMENT of an excursion is the only way to see which upstream quantity actually moved.

    Cost when healthy is a few clones and nothing else: **all analysis is deferred to dump time**
    (an SVD per solve would be far too expensive to run unconditionally). The solver's internals are
    reached by wrapping `torch.linalg`, so the PROTECTED CORE is untouched.
    """

    def __enter__(self):
        self.rec = {'lstsq': [], 'solve_cond_in': [], 'inv_in': [], 'W': [], 'A3': []}
        self._real = (torch.linalg.lstsq, torch.linalg.solve, torch.linalg.inv)
        real_lstsq, real_solve, real_inv = self._real

        # Also wrap the intrinsic solve itself. BISECTION (added after the 2026-08-23 09:37
        # excursion): that dump proved G and r are identical to 1 ulp and that Lam is stable across
        # gelsy/gelsd/gelss/pinv -- so the lstsq is NOT the culprit. The divergence therefore lives
        # in W0/PinvJt, in directions J3 cannot see (J3 is 672x1008, >=336 null dimensions, and a
        # change of W0 in ker(J3) leaves r=J3*W0 untouched while still moving W). Capturing W's
        # INPUT (A3) and OUTPUT splits the pipeline in two: if W matches the healthy run the fault
        # is downstream in the contraction to C; if it differs, it is inside the metric solve.
        import forward_solver_torch as _FST
        self._fst = _FST
        self._real_w = _FST._woodbury_solve_aw

        def woodbury_aw(A3, w, J3=None):
            out = self._real_w(A3, w, J3)
            self.rec['A3'].append(A3.detach().clone())
            self.rec['W'].append(out.detach().clone())
            return out

        _FST._woodbury_solve_aw = woodbury_aw

        def lstsq(A, B, *a, **kw):
            self.rec['lstsq'].append((A.detach().clone(), B.detach().clone()))
            return real_lstsq(A, B, *a, **kw)

        def solve(A, B, *a, **kw):                      # (I3 - Sw), tiny: keep the matrix itself
            self.rec['solve_cond_in'].append(A.detach().clone())
            return real_solve(A, B, *a, **kw)

        def inv(A, *a, **kw):                           # batched A3 (N,3,3)
            self.rec['inv_in'].append(A.detach().clone())
            return real_inv(A, *a, **kw)

        torch.linalg.lstsq, torch.linalg.solve, torch.linalg.inv = lstsq, solve, inv
        return self

    def __exit__(self, *exc):
        torch.linalg.lstsq, torch.linalg.solve, torch.linalg.inv = self._real
        self._fst._woodbury_solve_aw = self._real_w
        return False                                    # never swallow an exception

    def summarise(self, path_npz):
        """Deferred analysis — called ONLY on an anomaly. Returns a JSON-able dict and writes the
        full `G`/`r` to `path_npz` so a post-mortem can redo any algebra it likes."""
        out = {}
        if self.rec['lstsq']:
            G, r = self.rec['lstsq'][0]
            Gn, rn = G.numpy(), r.numpy()
            sv = np.linalg.svd(Gn, compute_uv=False)
            cut = float(np.finfo(np.float64).eps * max(Gn.shape) * sv[0])
            out['G'] = dict(shape=list(Gn.shape), sigma=sv.tolist(), rcond_none_cutoff=cut,
                            rank_at_cutoff=int((sv > cut).sum()),
                            sigma_min_over_cutoff=float(sv[-1] / cut),
                            cond=float(sv[0] / sv[-1]) if sv[-1] > 0 else float('inf'),
                            symmetry=float(np.abs(Gn - Gn.T).max()), r_norm=float(np.abs(rn).max()))
            extra = {}
            if self.rec['W']:                     # the bisection payload, see __enter__
                extra['W'] = self.rec['W'][0].numpy()
                extra['A3'] = self.rec['A3'][0].numpy()
                out['W'] = dict(shape=list(extra['W'].shape),
                                absmax=float(np.abs(extra['W']).max()),
                                fro=float(np.linalg.norm(extra['W'])),
                                checksum=float(extra['W'].sum()))
                out['A3_checksum'] = float(extra['A3'].sum())
            np.savez_compressed(path_npz, G=Gn, r=rn, **extra)
        if self.rec['solve_cond_in']:
            out['I3_minus_Sw_cond'] = [float(np.linalg.cond(A.numpy()))
                                       for A in self.rec['solve_cond_in']]
        if self.rec['inv_in']:
            out['A3_cond_max'] = max(float(np.max(np.linalg.cond(A.numpy())))
                                     for A in self.rec['inv_in'])
        return out


def _b1_dump_if_anomalous(phi, psi, eta, seed, vs, c_solver, c_phys, cap=None):
    """Record a full picture of an anomalous [15] comparison. Never raises and never alters the
    test's verdict — the asserts are untouched; this only observes.

    `cap`: an exited `_B1Capture` holding the solve's upstream inputs, or None."""
    try:
        if vs <= B1_DUMP_THRESHOLD:
            return
        import datetime, json, platform
        os.makedirs(B1_DUMP_DIR, exist_ok=True)
        scale = float(np.abs(c_phys).max())
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        commit, dirty, _ = C._provenance()
        upstream = {}
        if cap is not None:
            try:                                  # a failed post-mortem must not lose the dump
                upstream = cap.summarise(os.path.join(
                    B1_DUMP_DIR, f'b1_anomaly_{stamp}_eta{eta}_s{seed}_Gr.npz'))
            except Exception as e:                                          # noqa: BLE001
                upstream = {'summarise_failed': f'{type(e).__name__}: {e}'}
        rec = dict(when=stamp, commit=commit, dirty=dirty, vs=vs,
                   case=dict(phi=phi, psi=psi, eta=eta, seed=seed),
                   # is the deviation ONE component or diffuse? -> the discriminating question
                   dC_over_scale=(np.abs(c_solver - c_phys) / scale).tolist(),
                   c_solver=c_solver.tolist(), c_phys=c_phys.tolist(),
                   # upstream state AT THE MOMENT OF THE EXCURSION (added 2026-08-23) — the whole
                   # point: a clean process cannot reproduce B-1, so this is the only look inside.
                   upstream=upstream,
                   torch_threads=torch.get_num_threads(),
                   omp=os.environ.get('OMP_NUM_THREADS', 'unset'),
                   platform=platform.platform(), cpu_count=os.cpu_count())
        path = os.path.join(B1_DUMP_DIR, f'b1_anomaly_{stamp}_eta{eta}_s{seed}.json')
        with open(path, 'w') as fh:
            json.dump(rec, fh, indent=2)
        print(f"  [15] *** B-1 ANOMALY CAPTURED *** vs={vs:.3e} "
              f"(case eta={eta} seed={seed}) -> {path}", flush=True)
    except Exception as e:                                   # noqa: BLE001 — a diagnostic must never
        print(f"  [15] (B-1 dump failed, ignored: {type(e).__name__}: {e})", flush=True)  # break a gate


def test_homogenization():
    """The homogenisation function -- turning a relaxed network into effective (ν, E) -- checked
    three ways that MUST all agree: (a) the forward solver's own homogenised forward(), (b)
    physical_homog's VIRIAL-stress route, (c) physical_homog's ENERGY-HESSIAN route. The virial IS
    dU/dε, so (b) and (c) are genuinely separate computations whose agreement is a real
    self-consistency check of the homogenisation (not a tautology); (a) is the differentiable solver
    the designer optimises through, which the physical ground truth must confirm -- COMPONENT BY
    COMPONENT, since ν,E are only two contractions of the tensor and are weakly sensitive to its
    shear-shear entry. Across regular + disordered + anisotropic topologies with a random per-bond k,
    plus the analytic regular-lattice value ν=1/3, E=2/√3 at uniform k."""

    def free_of(geo):
        return np.arange(2, 2 * len(geo['pts']))

    geo = C.make_lattice(1.0, 1.0, half=6.0)                      # regular, uniform k=1
    geo['bond_k'] = np.ones(len(geo['bond_u'])); geo['tri_k'] = geo['bond_k'][geo['tri_bond']]
    nu0, E0 = PH.virial_nuE(geo, PH.relax(geo, free_of(geo), SA.assemble_K_faff))
    assert abs(nu0 - 1 / 3) < 0.02 and abs(E0 - 2 / np.sqrt(3)) < 0.05, \
        f"regular lattice analytic: nu={nu0:.4f} (~0.3333), E={E0:.4f} (~{2/np.sqrt(3):.4f})"

    # solver vs physical is compared at the TENSOR level: scalar ν/E is convention-dependent for an
    # anisotropic tensor (physical_homog._voigt_nuE averages Ex,Ey; the solver's c6_to_nuE uses a
    # different reduction), so the two legitimately disagree on the scalar summary while the full
    # effective tensor -- the unambiguous object the homogenisation actually produces -- agrees. The
    # virial-vs-energy pair DO share _voigt_nuE's convention, so their scalar (ν,E) is a fair check.
    # The oracle here is PH.energy_C -- an INDEPENDENT code path. It must NOT be C.sim_region_C6:
    # that routes the sim's relaxation through the solver's own _compute_actual_elastic_tensor, so
    # comparing against it is self-verification and hides any defect in the contraction itself
    # (it hid the 2026-08 shear-channel one; cf. Phase 2/test_forward_solver.py [7]).
    worst_ve, worst_vs = 0.0, 0.0
    for phi, psi, eta, seed in [(1.0, 1.0, 0.0, 0), (1.0, 1.0, 0.35, 1), (1.0, 0.6, 0.0, 2)]:
        geo = C.make_lattice(phi, psi, half=6.0, eta=eta, seed=seed)
        rng = np.random.default_rng(seed)
        k = 0.5 + rng.random(len(geo['bond_u']))
        geo['bond_k'] = k; geo['tri_k'] = k[geo['tri_bond']]
        free = free_of(geo)
        nu_v, E_v = PH.virial_nuE(geo, PH.relax(geo, free, SA.assemble_K_faff))   # virial route
        nu_e, E_e = PH.energy_nuE(geo, free, SA.assemble_K_faff)                  # energy-Hessian route
        prob = DesignProblem.from_geo(geo)
        with _B1Capture() as cap:                       # observes only; see the class docstring
            cs = C.solver_region_C6(prob, torch.as_tensor(k))                     # differentiable solver
        c_solver = np.array([[cs[0], cs[2], cs[1]], [cs[2], cs[5], cs[4]],        # -> Voigt [xx,yy,xy]
                             [cs[1], cs[4], cs[3]]])
        c_phys = PH.energy_C(geo, free, SA.assemble_K_faff)          # INDEPENDENT energy-Hessian tensor
        ve = max(abs(nu_v - nu_e), abs(E_v - E_e) / abs(E_v))
        vs = float(np.abs(c_solver - c_phys).max() / np.abs(c_phys).max())
        _b1_dump_if_anomalous(phi, psi, eta, seed, vs, c_solver, c_phys, cap)  # audit B-1, see below
        assert ve < 3e-3, f"virial vs energy-Hessian disagree (phi={phi},psi={psi},eta={eta}): {ve:.2e}"
        assert vs < 0.01, f"solver vs physical homogenisation tensor disagree (phi={phi},psi={psi},eta={eta}): {vs:.2e}"
        worst_ve, worst_vs = max(worst_ve, ve), max(worst_vs, vs)
    print(f"  [15] homogenisation: regular nu={nu0:.4f}/E={E0:.4f}; worst virial-vs-energy(ν,E)={worst_ve:.1e}, "
          f"solver-vs-physical(tensor)={worst_vs:.1e}  OK")


def test_isotropization():
    """Isotropisation via the 'isotropy' objective as a DESIGN driver (not just the readback of test
    [9]) -- as a ROUND TRIP that is guaranteed reachable and honest: (1) take a disordered (nearly
    isotropic) base and DESIGN a strongly directional response into it (2-fold E(θ)); confirm the
    result is genuinely anisotropic; (2) RE-design the same lattice with the 'isotropy' objective and
    confirm the anisotropy collapses. Anisotropy is measured with inverse_design._anisotropy on the
    INDEPENDENT sim tensor (not the solver's own readback). Two subtleties this test pins down:
      - A geometric CRYSTAL's anisotropy (e.g. aniso_str's compressed rows) lives in the reference
        metric and CANNOT be removed by k-scaling -- isotropising it fails (anisotropy even grows). So
        the base must be one whose anisotropy k-design can actually reach: a disordered lattice with
        the anisotropy DESIGNED in, exactly as here. (This is also why disordered bases are the ones
        the repo isotropises in tests [8]/[9].)
      - The 'isotropy' objective alone leaves the LEVEL free, so on its own it degenerates (collapses
        the tensor toward zero -> anisotropy 0/0 = nan); anchoring the modulus with an 'E' objective
        (the `constrain(isotropic=True, E=...)` idiom) makes it well-posed."""
    from inverse_design import _anisotropy, ANG
    prob, geo = C.make_case('disorder_hi', 14, seed=3)
    res_a = optimize(prob, [Objective('E_theta', 1.0 * (1 + 0.5 * np.cos(2 * ANG)))],
                     mode='k', n_iter=140, reg=1e-4, verbose=False)          # (1) DESIGN anisotropy in
    C.apply_k_to_geo(geo, res_a['k'])
    C6_a = C.sim_region_C6(geo, None)
    a0 = float(_anisotropy(torch.as_tensor(C6_a)))
    _, E0 = C.c6_nuE(C6_a)                                                   # anchor level at the anisotropic E
    assert a0 > 0.1, f"induced base should be clearly anisotropic, got {a0:.4f}"
    res_i = optimize(prob, [Objective('isotropy', weight=1.0), Objective('E', target=E0, weight=0.3)],
                     mode='k', n_iter=160, reg=1e-4, verbose=False)          # (2) isotropise it back
    C.apply_k_to_geo(geo, res_i['k'])
    C6_i = C.sim_region_C6(geo, None)
    a1 = float(_anisotropy(torch.as_tensor(C6_i)))
    _, E1 = C.c6_nuE(C6_i)
    assert np.isfinite(a1) and a1 < 0.2 * a0, \
        f"isotropy objective didn't isotropise: anisotropy {a0:.4f} -> {a1:.4f}"
    assert E1 > 0.3 * E0, f"isotropy collapsed the modulus (E {E0:.3f} -> {E1:.3f}), not a real isotropisation"
    print(f"  [16] isotropization (disordered round-trip): anisotropy {a0:.4f} -> {a1:.4f}, "
          f"E {E0:.3f} -> {E1:.3f} (sim-confirmed)  OK")


if __name__ == '__main__':
    torch.manual_seed(0)
    tests = [test_round_trip, test_property_auxetic, test_property_E, test_local_region,
             test_mixed_global_local, test_large_N_adjoint, test_open_domain, test_directional,
             test_constrain_isotropic, test_strain_stress_equivalence, test_strain_stress_autograd,
             test_strain_stress_design, test_homogeneity_regularizer, test_homogenization,
             test_isotropization]
    print("inverse_design tests")
    failed = 0
    for t in tests:
        try:
            t()
        except Exception as e:
            failed += 1
            print(f"  FAIL {t.__name__}: {e}")
    print("ALL PASSED" if failed == 0 else f"{failed} TEST(S) FAILED")
    sys.exit(1 if failed else 0)
