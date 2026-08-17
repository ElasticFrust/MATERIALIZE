r"""GATE: the solver against a CLOSED-FORM solution (hexagon diameter family).

The strongest known-answer check in the repo, and the cheapest: 7 nodes, 6 triangles, milliseconds
per point, no periodicity, and an ANALYTIC reference independent of both the solver and the sim.

    nu(r) = -((2 - 4r)(r + 1/2)) / (3 + 4r - 4r^2)  =  (4r^2 - 1)/(3 + 4r - 4r^2),   r = d/2

for a hexagon with a RIGID perimeter (six edges of length 1) and FREE hinges, where d is the
separation of two opposite vertices. Derivation: |V0V1| = 1 gives y^2 = 1 - (r-1/2)^2, the cell
dimensions along the single soft mode are L_x = 1 + 2r (the LATTICE CONSTANT, not the hexagon's own
width 2r — that is the step that is easy to get wrong, and using 2r predicts nu = 2/3 at the regular
hexagon instead of 1) and L_y = 2y; differentiating gives the expression above.

**Why this gate earns its place.** It pins three independent landmarks — nu = -0.2 at d = 1/2
(re-entrant/auxetic), nu = 0 exactly at d = 1, and nu = 1 at the regular hexagon d = 2 (the textbook
value, where the response is also isotropic) — and it is SENSITIVE TO THE SHEAR CHANNEL, so it would
have caught **A-0** (C_xyxy over-stiff by 29-90%) immediately. Measured 2026-08-17: max deviation
4.4e-06 over d in [0.05, 2] in the free-hinge limit.

The physical model here has FINITE hinge stiffness, so the comparison is made at k_spoke = 1e-8. The
residual at the designer's default k_spoke = 1e-3 is ~0.3-2.7% and is NOT solver error: it is first
order in k_spoke (each tenfold reduction divides it by ten), which this file also asserts.

Run:  python test_hex_closed_form.py     (anaconda python — the stack is float64)
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'Phase 3', 'verifications'))
import _common as C                                                    # noqa: E402  (wires sys.path)
sys.path.insert(0, os.path.join(REPO, 'Phase 5'))
sys.path.insert(0, os.path.join(REPO, 'Phase 2'))
sys.path.insert(0, HERE)
from hex_nu_linear import nu_closed_form, nu_solver_single             # noqa: E402

torch.set_default_dtype(torch.float64)
TOL_FREE = 1e-4                     # free-hinge limit vs the closed form (measured 4.4e-06)


def test_closed_form():
    """Sweep d and compare the free-hinge solver against the analytic nu(r)."""
    ds = np.linspace(0.05, 2.0, 40)
    got = np.array([nu_solver_single(d, 1e-8) for d in ds])
    want = nu_closed_form(ds)
    err = np.max(np.abs(got - want))
    assert err < TOL_FREE, f'solver vs closed form: max |dnu| = {err:.3e} (tol {TOL_FREE})'
    print(f'  [1] closed form over d in [0.05, 2], 40 pts: max |dnu| = {err:.2e}  OK')


def test_landmarks():
    """The three analytic landmarks, each pinned independently."""
    for d, want, what in ((0.5, -0.2, 're-entrant / auxetic'),
                          (1.0, 0.0, 'sign change'),
                          (2.0, 1.0, 'regular hexagon (textbook honeycomb)')):
        got = nu_solver_single(d, 1e-8)
        assert abs(got - want) < 1e-4, f'd={d} ({what}): got {got:+.6f}, want {want:+.4f}'
    print('  [2] landmarks: nu(0.5)=-0.2, nu(1)=0, nu(2)=+1 (regular hexagon, isotropic)  OK')


def test_hinge_stiffness_is_first_order():
    """The residual at finite k_spoke must be FIRST ORDER in k_spoke, not a solver defect.

    A tenfold softer hinge must divide the deviation by ~10. If this ever fails, the residual is
    NOT the model difference and the agreement above should not be trusted."""
    d = 2.0
    want = nu_closed_form(d)
    errs = [abs(nu_solver_single(d, k) - want) / abs(want) for k in (1e-2, 1e-3, 1e-4)]
    for a, b in zip(errs, errs[1:]):
        ratio = a / max(b, 1e-300)
        assert 6.0 < ratio < 16.0, f'residual not first order in k_spoke: ratio {ratio:.2f}'
    print(f'  [3] residual is O(k_spoke): {errs[0]:.2e} -> {errs[1]:.2e} -> {errs[2]:.2e} '
          f'(ratios {errs[0]/errs[1]:.1f}, {errs[1]/errs[2]:.1f})  OK')


if __name__ == '__main__':
    print('hexagon closed-form gate (analytic reference; would have caught A-0)')
    failed = 0
    for t in (test_closed_form, test_landmarks, test_hinge_stiffness_is_first_order):
        try:
            t()
        except Exception as e:
            failed += 1
            print(f'  FAIL {t.__name__}: {type(e).__name__}: {e}')
    print('ALL PASSED' if not failed else f'{failed} TEST(S) FAILED')
    sys.exit(1 if failed else 0)
