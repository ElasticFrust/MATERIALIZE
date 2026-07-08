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
import Disc_2_Cont_optimized as D2C
from inverse_design import DesignProblem, Objective, optimize, validate
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


if __name__ == '__main__':
    torch.manual_seed(0)
    tests = [test_round_trip, test_property_auxetic, test_property_E, test_local_region,
             test_mixed_global_local, test_large_N_adjoint, test_open_domain, test_directional,
             test_constrain_isotropic]
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
