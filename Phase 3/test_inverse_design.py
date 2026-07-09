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
torch.set_default_dtype(torch.float64)


def test_round_trip():
    prob = DesignProblem.periodic(N=14, eta=0.3, seed=0)
    rng = torch.Generator().manual_seed(1)
    k_true = 0.3 + 1.5 * torch.rand(prob.n_bond, generator=rng)
    tgt = prob.region_tensor(prob.forward(k_true)['per_triangle'], None).detach()
    res = optimize(prob, [Objective('tensor', tgt)], mode='k', n_iter=120, verbose=False)
    err = validate(prob, res['k'], res['l0'], [Objective('tensor', tgt)])[0]['err']
    assert err < 1e-3, f"round-trip tensor err {err:.2e}"
    print(f"  [1] round-trip: full-tensor err={err:.2e}  OK")


def test_property_auxetic():
    prob = DesignProblem.periodic(N=14, eta=0.3, seed=2)
    objs = [Objective('nu', target=-0.20)]
    res = optimize(prob, objs, mode='k', n_iter=100, verbose=False)
    rep = validate(prob, res['k'], res['l0'], objs)[0]
    assert rep['err'] < 0.02, f"nu achieved {rep['achieved']:.3f} vs target -0.20"
    print(f"  [2] auxetic target nu=-0.20: achieved {rep['achieved']:+.3f}  OK")


def test_property_E():
    prob = DesignProblem.periodic(N=14, eta=0.2, seed=3)
    base = validate(prob, torch.ones(prob.n_bond), None, [Objective('E', 0.0)])[0]['achieved']
    tgtE = 0.6 * base
    res = optimize(prob, [Objective('E', tgtE)], mode='k', n_iter=100, verbose=False)
    rep = validate(prob, res['k'], res['l0'], [Objective('E', tgtE)])[0]
    assert rep['err'] / tgtE < 0.03, f"E achieved {rep['achieved']:.3f} vs {tgtE:.3f}"
    print(f"  [3] E target {tgtE:.3f} (0.6*base): achieved {rep['achieved']:.3f}  OK")


def test_local_region():
    prob = DesignProblem.periodic(N=16, eta=0.2, seed=4)
    c = prob.centroids.mean(0)
    reg = prob.region_in_circle(c, radius=0.20 * (prob.centroids[:, 0].max() - prob.centroids[:, 0].min()))
    objs = [Objective('nu', target=-0.15, region=reg, weight=1.0)]
    res = optimize(prob, objs, mode='k', n_iter=120, verbose=False)
    rep = validate(prob, res['k'], res['l0'], objs)[0]
    assert rep['err'] < 0.03, f"local nu achieved {rep['achieved']:.3f} vs -0.15"
    print(f"  [4] local region ({len(reg)} tri) nu=-0.15: achieved {rep['achieved']:+.3f}  OK")


def test_mixed_global_local():
    prob = DesignProblem.periodic(N=18, eta=0.2, seed=5)
    c = prob.centroids.mean(0)
    reg = prob.region_in_circle(c, radius=0.18 * (prob.centroids[:, 0].max() - prob.centroids[:, 0].min()))
    objs = [Objective('nu', target=+0.25, region=None, weight=1.0),          # global positive
            Objective('nu', target=-0.20, region=reg, weight=2.0)]           # local auxetic patch
    res = optimize(prob, objs, mode='k', n_iter=150, verbose=False)
    rep = validate(prob, res['k'], res['l0'], objs)
    g, l = rep[0], rep[1]
    assert g['err'] < 0.05 and l['err'] < 0.05, f"global {g['achieved']:.3f}, local {l['achieved']:.3f}"
    print(f"  [5] mixed: global nu={g['achieved']:+.3f} (->+0.25), "
          f"local patch nu={l['achieved']:+.3f} (->-0.20)  OK")


def test_large_N_adjoint():
    prob = DesignProblem.periodic(N=20, eta=0.3, seed=6)          # 800 tri > 600 -> adjoint path
    assert prob.n_tri > 600
    objs = [Objective('nu', target=0.0)]
    res = optimize(prob, objs, mode='k', n_iter=60, verbose=False)
    rep = validate(prob, res['k'], res['l0'], objs)[0]
    assert res['history'][-1] < res['history'][0] and rep['err'] < 0.03, \
        f"large-N nu achieved {rep['achieved']:.3f}"
    print(f"  [6] large-N adjoint ({prob.n_tri} tri) nu=0: achieved {rep['achieved']:+.3f}  OK")


def test_open_domain():
    np.random.seed(7)
    tri = D2C.generate_foam_points((5, 5), 0.2)
    prob = DesignProblem.open(tri)
    objs = [Objective('nu', target=0.10)]
    res = optimize(prob, objs, mode='k', n_iter=100, verbose=False)
    rep = validate(prob, res['k'], res['l0'], objs)[0]
    assert rep['err'] < 0.03, f"open nu achieved {rep['achieved']:.3f} vs 0.10"
    print(f"  [7] open mesh ({prob.n_tri} tri) nu=0.10: achieved {rep['achieved']:+.3f}  OK")


def test_directional():
    """Directional ν(θ)/E(θ) objectives: program a profile, isotropise to a flat level, flat E."""
    from inverse_design import ANG
    prob = DesignProblem.periodic(N=16, eta=0.3, seed=8)
    prof = 0.25 + 0.25 * np.cos(4 * ANG)                          # (a) program a REALIZABLE 4-fold ν(θ)
    o = [Objective('nu_theta', prof)]                            # (pure cos2θ is OFF the ν(θ) manifold)
    res = optimize(prob, o, mode='k', n_iter=140, verbose=False)
    ea = validate(prob, res['k'], res['l0'], o)[0]['err']
    assert ea < 0.09, f"nu(theta) profile maxerr {ea:.3f}"
    o2 = [Objective('nu_theta', -0.10)]                          # (b) isotropise to flat ν0=-0.1
    res2 = optimize(prob, o2, mode='k', n_iter=150, verbose=False)
    got = validate(prob, res2['k'], res2['l0'], o2)[0]['achieved']
    spread, mean = float(np.ptp(got)), float(got.mean())
    assert spread < 0.06 and abs(mean + 0.10) < 0.03, f"isotropise spread {spread:.3f} mean {mean:.3f}"
    o3 = [Objective('E_theta', 0.9)]                             # (c) flat E(θ) target
    res3 = optimize(prob, o3, mode='k', n_iter=100, verbose=False)
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

    # legacy single-direction 'nu_dir' leaves the patch anisotropic; scalar 'nu' (=isotropic) fixes it
    ish = optimize(prob, [Objective('nu_dir', -0.2, region=patch)], mode='k', n_iter=140, verbose=False)
    sp_ish, _ = nu_spread(ish['k'])
    ex = optimize(prob, [Objective('nu', -0.2, region=patch)], mode='k', n_iter=160, verbose=False)
    sp, mn = nu_spread(ex['k'])
    assert sp < 0.05 and abs(mn + 0.2) < 0.03 and sp < sp_ish, \
        f"exact flat ν spread {sp:.3f} mean {mn:.3f} (ish spread {sp_ish:.3f})"
    # exact isotropic tensor (ν AND E) -> tiny isotropy residual
    it = optimize(prob, constrain(region=patch, tensor=isotropic_c6(-0.2, 0.6)), mode='k',
                  n_iter=180, verbose=False)
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
        load_dg = C.CE.vec3(Dgt[mode]) / PH.DELTA              # linear-normalised, matches eps_ref's scale
        strain_t, stress_t = per_triangle_strain_stress(bare, W, load_dg)
        eps_ref_vec3 = C.CE.vec3(eps_ref[mode])
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
    res = optimize(prob, objs, mode='k', n_iter=150, verbose=False)
    rep = validate(prob, res['k'], res['l0'], objs)[0]
    # matching all 3 stress-vec3 components simultaneously converges to a few % relative (a coarse
    # 3-number summary of a high-dim k has many near-equally-good solutions; not full k_true recovery)
    assert rep['err'] < 0.06 * float(stress_target.abs().max()), f"stress err {rep['err']:.4f}"
    print(f"  [12] stress design (global): err={rep['err']:.4f}  OK")

    reg = prob.region_in_circle(prob.centroids.mean(0),
                                0.25 * (prob.centroids[:, 0].max() - prob.centroids[:, 0].min()))
    strain_target = region_mean_vec3(strain_true, reg).detach()
    objs2 = [Objective('strain', target=strain_target, region=reg, load=load)]
    res2 = optimize(prob, objs2, mode='k', n_iter=150, verbose=False)
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

    res0 = optimize(prob, [Objective('nu', target=-0.5, region=reg, weight=1.0)],
                    mode='k', n_iter=150, verbose=False)
    var0 = region_nu_var(res0['k'])

    objs1 = [Objective('nu', target=-0.5, region=reg, weight=1.0, homogeneity=1.0)]
    # n_restarts>1: this demanding regional target occasionally lands LBFGS in a visibly worse local
    # optimum on a single run (observed run-to-run, even at a fixed seed -- LBFGS/threading
    # nondeterminism); n_restarts is this codebase's own mechanism for exactly that, so use it rather
    # than just loosening the achieved-error tolerance to paper over an unlucky single run. Even with
    # n_restarts=3, repeated calibration runs showed achieved err mostly ~0.00-0.01 with an occasional
    # outlier up to ~0.09 -- the 0.12 bound below has margin over that observed tail, not a blind guess.
    res1 = optimize(prob, objs1, mode='k', n_iter=150, n_restarts=4, verbose=False)
    rep1 = validate(prob, res1['k'], res1['l0'], objs1)[0]
    var1 = region_nu_var(res1['k'])

    assert var1 < 0.3 * var0, f"homogeneity didn't shrink region nu-variance: {var1:.4f} vs {var0:.4f}"
    assert rep1['err'] < 0.12, f"homogeneity=1.0 pulled achieved nu off target: err={rep1['err']:.3f}"
    print(f"  [14] homogeneity regularizer: region nu-variance {var0:.4f} -> {var1:.4f} "
          f"(achieved nu={rep1['achieved']:+.3f})  OK")


if __name__ == '__main__':
    torch.manual_seed(0)
    tests = [test_round_trip, test_property_auxetic, test_property_E, test_local_region,
             test_mixed_global_local, test_large_N_adjoint, test_open_domain, test_directional,
             test_constrain_isotropic, test_strain_stress_equivalence, test_strain_stress_autograd,
             test_strain_stress_design, test_homogeneity_regularizer]
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
