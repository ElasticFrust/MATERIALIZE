r"""Gate the CONSTRAINT geometry the model consumes against the SOLVER's own operators.

`model_v3.angle_gradient_vec` and `vertex_stars` re-derive the curvature constraint `C_curv`:
one row per interior vertex, `sum_{s in star(v)} (dtheta_v^s/dg) . dg(s) = 0`.  The solver builds
exactly that in `forward_solver_torch._build_intrinsic_constraints`.

These must AGREE, not merely resemble each other.  The model is being handed the constraint the
solver imposes; if the convention drifts (sign of the edge vector at a corner, the factor 2 on the
shear component, which vertices count as interior) the model learns a constraint the physics does
not have, and nothing downstream would reveal it -- the loss would simply be a little worse and we
would blame the architecture.

Run:
    python "Phase 5/verifications/test_m2_constraints.py"
"""
import os
import sys
import warnings

import numpy as np

R = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, os.path.join(R, 'Phase 3', 'verifications'))
import _common as C                                                       # noqa: E402
sys.path.insert(0, os.path.join(R, 'Phase 5'))
sys.path.insert(0, os.path.join(R, 'Phase 5', 'm2'))
sys.path.insert(0, os.path.join(R, 'Phase 2'))
import torch                                                              # noqa: E402
import seeds as S                                                         # noqa: E402
import model_v3 as M3                                                     # noqa: E402
import forward_solver_torch as FST                                        # noqa: E402
from inverse_design import DesignProblem                                  # noqa: E402

warnings.simplefilter('ignore')
torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)


def main():
    bad = 0
    for name, rec in (('random_patch(60)', S.random_patch(60, seed=1)),
                      ('random_patch(120)', S.random_patch(120, seed=2)),
                      ('bravais 1,1', S.bravais_lattice(1.0, 1.0, reps=4))):
        geo = rec['geo']
        n_tri = len(geo['tri_bond'])

        # ---- [1] the scalar kernel, elementwise against the solver's own function --------------
        rng = np.random.default_rng(0)
        A = rng.normal(size=(200, 2)); B = rng.normal(size=(200, 2))
        mine = M3.angle_gradient_vec(A, B)
        theirs = np.array([FST._angle_gradient_vec(a, b) for a, b in zip(A, B)])
        d1 = float(np.abs(mine - theirs).max())
        print('[1] %-18s angle_gradient_vec vs solver: max|d| = %.3e' % (name, d1))
        bad += d1 > 1e-12

        # ---- [2] the assembled operator: same rows, same weights -------------------------------
        prob = DesignProblem.from_geo(geo)
        Csp = prob.solver._C_curv_sp                       # (n_interior_vertex, 3*n_tri)
        st_t, st_v, st_w, n_v = M3.vertex_stars(
            geo['simplices'] if 'simplices' in geo else geo['tri_verts'],
            geo['tri_bond'], geo['bond_u'], geo['bond_v'], geo['bond_R'])

        # rebuild the solver's matrix from OUR incidence and compare the two sparse operators
        mineM = np.zeros((n_v, 3 * n_tri))
        for t_, v_, w_ in zip(st_t, st_v, st_w):
            mineM[v_, 3 * t_:3 * t_ + 3] += w_
        theirsM = np.asarray(Csp.todense())
        # rows are the same SET but not necessarily in the same ORDER (each side numbers interior
        # vertices by first appearance in its own loop), so compare as sorted row multisets
        def canon(M):
            M = M[np.abs(M).sum(1) > 1e-14]
            return M[np.lexsort(M.T[::-1])]
        a_, b_ = canon(mineM), canon(theirsM)
        if a_.shape != b_.shape:
            print('[2] %-18s ROW COUNT MISMATCH: mine %s vs solver %s' % (name, a_.shape, b_.shape))
            bad += 1
        else:
            d2 = float(np.abs(a_ - b_).max())
            print('[2] %-18s C_curv operator vs solver: rows %d, max|d| = %.3e'
                  % (name, a_.shape[0], d2))
            bad += d2 > 1e-12

        # ---- [3] the weights are vec3 -- they must ROTATE like the carriers --------------------
        th = 0.7
        Rm = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        w_rot = M3.angle_gradient_vec(A @ Rm.T, B @ Rm.T)
        # dtheta/dg is a form on the metric, so under x -> Rx it transforms by the vec3 rotation
        Vr = M3.vec3_rotation(torch.tensor(th)).numpy()
        pred = mine @ np.linalg.inv(Vr)
        d3 = float(np.abs(w_rot - pred).max() / max(np.abs(mine).max(), 1e-300))
        print('[3] %-18s weights transform as vec3: rel %.3e' % (name, d3))
        bad += d3 > 1e-10

    print('\n%s' % ('ALL CONSTRAINT GATES PASSED' if bad == 0 else 'FAILURES: %d' % bad))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
