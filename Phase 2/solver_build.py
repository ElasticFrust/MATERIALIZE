"""
Construct an `ElasticSolver` from a PERIODIC geometry dict.

Purpose: `forward_solver_torch.ElasticSolver` is built from (points, simplices, edges), which is
enough for an OPEN mesh but not for a torus: on a periodic cell the edge vectors must be
image-corrected (unwrapped) rather than recomputed from wrapped node positions, and the intrinsic
solve additionally needs the interior-edge constraint topology. This module supplies exactly that
mounting step, so `Phase 3/inverse_design.py` and the verification scripts share one construction
path instead of each re-deriving it.

Implements the periodic case of Grossman & Boudaoud, PRR 2026 (arXiv:2309.07844) §II; see
`Phase 2/SOLVER_GUIDE.md` §2 and `INTRINSIC_METRIC_SOLVE.md` for the constraint set (C1) edge
compatibility, (C2) zero discrete Gaussian curvature, (C3) area-weighted normalisation.

Layering: **core-adjacent** — the only module here that imports the core. Pairs with
`mesh_build.py` (which produces the geo dict and the `kkt` arrays) and `metric_ops.py`.

History: lifted verbatim by the A-7b re-layering (`documentation/AUDIT_2026-08.md`) from
`verification_tools/verify_solver_sweep.py` — the design layer was constructing its solver from a
sweep script inside the temporary, retireable oracle layer.

KNOWN WART, moved as-is and deliberately NOT fixed here: `_mount` reaches into the solver's
internals (assigning `edge_vecs`, `actual_length2`, `area_weights`, `kkt_arrays` and calling the
private `_build_intrinsic_constraints`). The clean fix is a real periodic constructor on
`ElasticSolver`, which is a change to the PROTECTED core and needs explicit approval plus a
re-gate; it is logged rather than done. Until then this module is the single place that wart
lives, instead of being spread across the callers.
"""
import numpy as np
import torch

import forward_solver_torch as fst


def _mount(solver, ev, l2, areas, kkt, sx):
    """Override an ElasticSolver's geometry buffers with periodic-correct arrays."""
    solver.edge_vecs      = torch.as_tensor(ev, dtype=torch.float64)
    solver.actual_length2 = torch.as_tensor(l2, dtype=torch.float64)
    solver.area_weights   = torch.as_tensor(areas / areas.sum(), dtype=torch.float64)
    solver.kkt_arrays     = kkt
    solver._build_intrinsic_constraints(np.asarray(sx))
    return solver


def make_solver(geo, kkt):
    """`ElasticSolver` for a periodic geometry dict, with unwrapped edges and the PBC constraints.

    Args:
        geo: geometry dict from `mesh_build.build_geometry` (or any dict carrying `pts`,
             `simplices`, `edge_vecs`, `areas`) — `edge_vecs` MUST be the unwrapped, image-correct
             vectors, since the solver cannot recover them from the wrapped `pts`.
        kkt: interior-edge constraint arrays `(s1, s2, q)` from `mesh_build.kkt_from_tri_bond`.
    Returns:
        the mounted `ElasticSolver`, ready for `forward(tri_k, rest_lengths=..., method='intrinsic')`.
    """
    pts, sx, ev = geo['pts'], geo['simplices'], geo['edge_vecs']
    edges = np.stack([sx[:, [1, 0]], sx[:, [2, 0]], sx[:, [2, 1]]], axis=1)   # (e01,e02,e12)
    s = fst.ElasticSolver(pts, sx, edges)
    return _mount(s, ev, (ev ** 2).sum(2), geo['areas'], kkt, sx)
