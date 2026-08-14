import os, sys
import numpy as np
import torch
# NOTE: PLAN §0 preamble uses '..' (repo root from a script in Phase 5/); this file lives in
# Phase 5/verifications/, so it needs '..','..' to reach the repo root. (See report to caller.)
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))   # repo root
sys.path.insert(0, os.path.join(REPO, 'Phase 3', 'verifications'))
import _common as C                 # sets up all other sys.path and imports the solver stack
from inverse_design import (DesignProblem, Objective, optimize, validate, ANG,
                            c6_to_nuE, c6_to_nuE_theta)
torch.set_default_dtype(torch.float64)   # REQUIRED — the whole stack is float64

NU_T = 1.0 / 3.0
E_T = 2.0 / np.sqrt(3.0)   # ≈ 1.1547


def main():
    # ---- regular triangular lattice: solver readout ----
    geo = C.make_lattice(1.0, 1.0, half=6)
    prob = DesignProblem.from_geo(geo)
    out = prob.forward(torch.ones(prob.n_bond))          # uniform k=1
    nu, E = (float(x) for x in c6_to_nuE(prob.region_tensor(out['per_triangle'], None)))
    print(f"[solver] nu={nu:.6f}  E={E:.6f}   (target nu={NU_T:.6f}, E={E_T:.6f})")
    assert abs(nu - NU_T) < 1e-2,  f"regular nu should be 1/3, got {nu}"
    assert abs(E - E_T) < 5e-2, f"regular E should be 2/sqrt(3)~1.1547, got {E}"

    # ---- independent simulation agrees ----
    C.apply_k_to_geo(geo, torch.ones(prob.n_bond))
    nu_s, E_s = C.sim_region_nuE(geo)
    print(f"[sim   ] nu={float(nu_s):.6f}  E={float(E_s):.6f}")
    assert abs(nu_s - NU_T) < 1e-2 and abs(E_s - E_T) < 5e-2, \
        f"sim mismatch: nu_s={nu_s}, E_s={E_s}"

    # ---- directional objective round-trip (designer API) ----
    prob2 = DesignProblem.from_geo(C.make_lattice(1.0, 1.0, half=5))
    objs = [Objective('nu_theta', -0.1, thetas=ANG),
            Objective('E_theta', 1.0, thetas=ANG)]
    res = optimize(prob2, objs, mode='k', n_iter=60, reg=0.02, verbose=False)
    rep = validate(prob2, res['k'], res['l0'], objs)
    print("[directional round-trip] optimize/validate/Objective(nu_theta,E_theta) OK")
    for r in rep:
        ach = r['achieved']
        ach_mean = float(np.mean(ach)) if hasattr(ach, '__len__') else float(ach)
        print(f"    kind={r['kind']:9s}  achieved(mean)={ach_mean:+.4f}  err={r['err']:.4f}")
    # baseline nu of the un-designed lattice is +1/3; confirm the design moved it toward -0.1
    nu_rec = next(r for r in rep if r['kind'] == 'nu_theta')
    nu_ach = float(np.mean(nu_rec['achieved']))
    print(f"    nu moved {NU_T:+.4f} (baseline) -> {nu_ach:+.4f} (target -0.1000): "
          f"{'toward target' if nu_ach < NU_T else 'WRONG DIRECTION'}")
    assert nu_ach < NU_T, f"designed nu did not move toward -0.1 (got {nu_ach})"

    print("SANITY PASSED")


if __name__ == '__main__':
    main()
