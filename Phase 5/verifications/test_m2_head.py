r"""GATE: is the M2 v2 head EXPRESSIVE, EQUIVARIANT and SPD -- before any training.

`M2_V2_PLAN.md` S1 lists gates for the trained model.  This file checks the ones that are properties
of the PARAMETRISATION, not of the weights, so they can be settled in seconds and before a single
gradient step.  If the head cannot represent the answer even with oracle weights, no amount of
training will help, and that is worth knowing first.

    [1] CLOSED FORM      with M M^T = diag(k_e / 16 l_e^2) the head reproduces the solver's C(s)
                         exactly on the OPEN single triangle, where W == 0 and C(s) = A(s).
    [2] AFFINE CELLS     the same oracle weights reproduce C(s) on any cell with W == 0 (the
                         regular triangular lattice, where nu = 1/3 and E = 2/sqrt(3)).
    [3] EXPRESSIVE       for a GENERAL mesh with W != 0, solving Q G Q^T = C(s) per triangle
                         recovers the solver's C(s) to machine precision -- i.e. the ansatz is not
                         merely affine.  This is the check that the plan's earlier
                         `sum_e c_e q_e q_e^T` form would have FAILED.
    [4] EQUIVARIANCE     rotate the geometry by theta: C must map to R C R^T with R the vec3
                         rotation, while the head's scalar inputs are unchanged.
    [5] SPD              the passive form gives positive-semidefinite C(s) for ARBITRARY raw
                         weights, including adversarial ones -- physicality is structural.
    [6] INTENSIVITY      mean pooling reproduces C_eff across a supercell; a sum branch cannot.

Run:  C:\Users\doron\anaconda3\python.exe "Phase 5/verifications/test_m2_head.py"
"""
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'Phase 3', 'verifications'))
import _common as C                                                       # noqa: E402
sys.path.insert(0, os.path.join(REPO, 'Phase 5'))
sys.path.insert(0, os.path.join(REPO, 'Phase 5', 'm2'))
import torch                                                              # noqa: E402
import model_v2 as M                                                      # noqa: E402
import seeds as S                                                         # noqa: E402
from inverse_design import DesignProblem                                  # noqa: E402

warnings.simplefilter('ignore')
torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

PASS, FAIL = [], []


def check(name, ok, detail):
    (PASS if ok else FAIL).append(name)
    print('  [%s] %-34s %s' % ('OK  ' if ok else 'FAIL', name, detail))


def solver_C(prob, k):
    """Per-triangle C(s) as the solver returns it, in INTERNAL units (see model_v2's UNITS note)."""
    with torch.no_grad():
        out = prob.forward(torch.as_tensor(k))
        return M.c6_to_sym3(out['per_triangle']), out


def tri_k_and_len2(geo, k):
    """Per-triangle-edge stiffness and squared length, in the solver's edge order."""
    tb = np.asarray(geo['tri_bond'])
    return np.asarray(k)[tb], np.asarray(geo['actual_len2'])


def main():
    print(__doc__.split('Run:')[0].strip().splitlines()[0])
    print()

    # ---- [1] the closed form on the open single triangle ---------------------------------------
    worst = 0.0
    for shape in ('equilateral', 'right', 'scalene'):
        for kv in (np.ones(3), np.array([3.0, 0.5, 1.7]), np.array([0.2, 4.0, 1.0])):
            rec = S.anchor_single_triangle(shape, kv)
            prob = DesignProblem.open(rec['tri'])
            C_solver, out = solver_C(prob, kv)
            ev = np.asarray(prob.edge_vecs if hasattr(prob, 'edge_vecs') else
                            prob.solver.edge_vecs, float).reshape(1, 3, 2)
            l2 = (ev ** 2).sum(-1)
            Q = M.edge_carriers(ev)
            G = M.closed_form_weights(kv.reshape(1, 3), l2)
            C_head, _ = M.assemble(Q, G)
            worst = max(worst, float((C_head - C_solver).abs().max()))
    check('[1] closed form, 1 triangle', worst < 1e-12,
          'max|C_head - C_solver| = %.2e over 3 shapes x 3 k' % worst)

    # ---- [2] affine cells: the same oracle weights, W == 0 -------------------------------------
    rec = [r for r in S.seed_cells_anchors() if r.get('geo') is not None][0]
    geo, k = rec['geo'], rec['k0']
    geo['bond_k'] = k
    geo['tri_k'] = k[geo['tri_bond']]
    prob = DesignProblem.from_geo(geo)
    C_solver, out = solver_C(prob, k)
    wmax = float(np.abs(np.asarray(out['W'], float)).max())
    ktri, l2 = tri_k_and_len2(geo, k)
    Q = M.edge_carriers(geo['edge_vecs'])
    C_head, Ceff = M.assemble(Q, M.closed_form_weights(ktri, l2))
    err = float((C_head - C_solver).abs().max())
    check('[2] closed form, affine cell', err < 1e-12,
          'max|dC| = %.2e on %s (max|W| = %.1e)' % (err, rec['name'], wmax))

    # ---- [3] expressiveness on a mesh with W != 0 ----------------------------------------------
    geo = S.random_patch(60, seed=0)['geo']
    rng = np.random.default_rng(0)
    k = np.exp(rng.normal(0, 0.8, len(geo['bond_u'])))
    geo['bond_k'] = k
    geo['tri_k'] = k[geo['tri_bond']]
    prob = DesignProblem.from_geo(geo)
    C_solver, out = solver_C(prob, k)
    wmax = float(np.abs(np.asarray(out['W'], float)).max())
    Q = M.edge_carriers(geo['edge_vecs'])
    # solve Q G Q^T = C  for G, per triangle -- the ansatz is exact iff this reproduces C
    Qi = torch.linalg.inv(Q)
    G_fit = Qi @ C_solver @ Qi.transpose(-1, -2)
    C_fit, _ = M.assemble(Q, G_fit)
    err = float((C_fit - C_solver).abs().max() / C_solver.abs().max())
    check('[3] expressive, W != 0', err < 1e-9,
          'rel max|dC| = %.2e with max|W| = %.2f (affine-only ansatz would fail)' % (err, wmax))

    off = float((G_fit - torch.diag_embed(torch.diagonal(G_fit, dim1=-2, dim2=-1))).abs().max())
    dia = float(torch.diagonal(G_fit, dim1=-2, dim2=-1).abs().max())
    check('[3b] off-diagonals carry W', off / dia > 1e-3,
          'max|offdiag|/max|diag| = %.3f  (0 would mean the response is affine)' % (off / dia))

    # ---- [4] equivariance ----------------------------------------------------------------------
    theta = 0.7
    R2 = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    Q_rot = M.edge_carriers(np.asarray(geo['edge_vecs']) @ R2.T)
    C_rot, _ = M.assemble(Q_rot, G_fit)                       # SAME scalar weights, rotated basis
    Rv = M.vec3_rotation(theta)
    C_expect = Rv @ C_solver @ Rv.T
    err = float((C_rot - C_expect).abs().max() / C_expect.abs().max())
    check('[4] equivariance', err < 1e-10,
          'rel max|C(rot) - R C R^T| = %.2e at theta = %.2f' % (err, theta))

    # ---- [5] SPD by construction ---------------------------------------------------------------
    net = M.ForwardGNNv2(hidden=8, n_layers=1, passive=True)
    raw = torch.randn(500, 6) * 50.0                          # adversarially large
    G = net._to_G(raw)
    eig_G = torch.linalg.eigvalsh(0.5 * (G + G.transpose(-1, -2)))
    C_any, _ = M.assemble(M.edge_carriers(geo['edge_vecs'])[:1].expand(500, 3, 3), G)
    eig_C = torch.linalg.eigvalsh(0.5 * (C_any + C_any.transpose(-1, -2)))
    check('[5] SPD by construction', float(eig_G.min()) >= -1e-10 and float(eig_C.min()) >= -1e-10,
          'min eig(G) = %.2e, min eig(C) = %.2e over 500 random raw weights'
          % (float(eig_G.min()), float(eig_C.min())))

    # ---- [6] intensivity of mean pooling -------------------------------------------------------
    outs = []
    for reps in (4, 6, 8):
        g = S.bravais_lattice(1.0, 1.0, reps=reps)['geo']
        kk = np.ones(len(g['bond_u']))
        ktri, l2 = tri_k_and_len2(g, kk)
        Qc = M.edge_carriers(g['edge_vecs'])
        _, Ceff = M.assemble(Qc, M.closed_form_weights(ktri, l2))
        outs.append(Ceff)
    d_mean = max(float((outs[i] - outs[0]).abs().max() / outs[0].abs().max()) for i in (1, 2))
    n_tri = [len(S.bravais_lattice(1.0, 1.0, reps=r)['geo']['simplices']) for r in (4, 6, 8)]
    check('[6] intensivity (mean pool)', d_mean < 1e-12,
          'rel max|dC_eff| = %.2e across n_tri = %s' % (d_mean, n_tri))

    print('\n%d passed, %d failed' % (len(PASS), len(FAIL)))
    if FAIL:
        print('FAILED:', ', '.join(FAIL))
    return 1 if FAIL else 0


if __name__ == '__main__':
    sys.exit(main())
