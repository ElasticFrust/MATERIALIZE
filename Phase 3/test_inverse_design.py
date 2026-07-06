"""
Tests for Phase 3 inverse design (inverse_design.py). Run: python "Phase 3/test_inverse_design.py"

Covers: round-trip recovery, property (nu incl. auxetic) targeting, full-tensor matching, a LOCAL
region target, a MIXED global+local design, a large-N (>600 tri, adjoint) design, and an open mesh.
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


if __name__ == '__main__':
    torch.manual_seed(0)
    tests = [test_round_trip, test_property_auxetic, test_property_E, test_local_region,
             test_mixed_global_local, test_large_N_adjoint, test_open_domain]
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
