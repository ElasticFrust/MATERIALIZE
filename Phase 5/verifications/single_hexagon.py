r"""ONE hexagon, NON-PERIODIC, six triangles — the solver on the smallest meaningful case.

User's spec (2026-08-17): a single hexagon, six perimeter edges HARD (k=1) and length 1; a
fictitious CENTRE vertex with six SOFT radial spokes gives exactly 6 triangles. Control the
separation `d` of two opposite rim vertices while every perimeter edge stays length 1:

    y = sqrt(1 - ((1-d)/2)^2)
    rim = [(+d/2,0), (+1/2,+y), (-1/2,+y), (-d/2,0), (-1/2,-y), (+1/2,-y)]

    d = 2  regular hexagon      d < 1  RE-ENTRANT      d -> 0  bow-tie      d -> 3  flat (y -> 0)

Pull PERPENDICULAR to the diameter (along y), read the response ALONG it (x):  nu = -eps_xx/eps_yy.

Two independent readings of the SAME 7-node network, no periodicity anywhere:
  * SOLVER  — `build_open_mesh` + the Phase 2 forward solve (the open/finite path), reduced to nu.
  * TRUSS   — a direct central-force spring solve written here in ~15 lines: apply a uniaxial
              force along y, remove the 3 rigid-body modes by least-squares, read the strains.
              It shares NO code with the solver, so it is an independent check by construction.

`d` is swept to **2.99**: at d = 3 exactly, y = 0 and the hexagon is FLAT (all seven points
collinear) — a genuinely degenerate limit, not a numerical one. An earlier sweep of mine stopped at
2.8 without saying so; the approach to 3 is where the interesting divergence lives.

Run:  python single_hexagon.py
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'Phase 3', 'verifications'))
import _common as C                                                    # noqa: E402  (wires sys.path)
sys.path.insert(0, os.path.join(REPO, 'Phase 2'))
import mesh_build as MB                                                # noqa: E402
import solver_build as SB                                              # noqa: E402

torch.set_default_dtype(torch.float64)
EPS_SOFT = 1e-3                       # soft radial spokes; perimeter is k = 1


class _Tri:
    """Minimal scipy-triangulation stand-in: `build_open_mesh` needs only .points/.simplices."""

    def __init__(self, points, simplices):
        self.points = np.asarray(points, float)
        self.simplices = np.asarray(simplices, np.int64)


def hexagon(d):
    """7 points (6 rim + centre) and the 6-triangle fan. Returns (_Tri, k_per_bond_builder)."""
    y = np.sqrt(max(1.0 - ((1.0 - d) / 2.0) ** 2, 0.0))
    rim = np.array([(d / 2, 0.0), (0.5, y), (-0.5, y), (-d / 2, 0.0), (-0.5, -y), (0.5, -y)])
    pts = np.vstack([rim, [[0.0, 0.0]]])                   # index 6 = centre
    simp = np.array([[6, a, (a + 1) % 6] for a in range(6)], np.int64)
    return _Tri(pts, simp), y


def k_of(mesh):
    """Perimeter (rim-rim, length 1) HARD = 1; radial spokes (touching the centre, index 6) SOFT."""
    u, v = np.asarray(mesh['bond_u']), np.asarray(mesh['bond_v'])
    radial = (u == 6) | (v == 6)
    return np.where(radial, EPS_SOFT, 1.0)


def truss_nu(pts, bu, bv, k):
    """INDEPENDENT reference: linear central-force truss, uniaxial force along y, no periodicity.

    K u = f with f = +1 on the two top rim nodes and -1 on the two bottom ones (net zero). The
    3 rigid-body modes are removed by lstsq rather than by clamping, so no boundary condition is
    imposed on the transverse direction — which is what makes -eps_xx/eps_yy a clean Poisson ratio."""
    n = len(pts)
    K = np.zeros((2 * n, 2 * n))
    for a, b, kk in zip(bu, bv, k):
        d = pts[b] - pts[a]
        L = np.linalg.norm(d)
        if L < 1e-14:
            continue
        t = d / L                                          # central force: stiffness along the bond
        blk = kk * np.outer(t, t)
        for (i, j, s) in ((a, a, +1), (b, b, +1), (a, b, -1), (b, a, -1)):
            K[2 * i:2 * i + 2, 2 * j:2 * j + 2] += s * blk
    f = np.zeros(2 * n)
    top = [1, 2]; bot = [4, 5]                             # rim indices at +y and -y
    for i in top:
        f[2 * i + 1] += 1.0
    for i in bot:
        f[2 * i + 1] -= 1.0
    u = np.linalg.lstsq(K, f, rcond=None)[0].reshape(n, 2)  # lstsq: rigid modes are the null space
    e_yy = (u[top, 1].mean() - u[bot, 1].mean()) / (pts[top, 1].mean() - pts[bot, 1].mean())
    dx = pts[0, 0] - pts[3, 0]                             # the diameter, V0 - V3
    e_xx = (u[0, 0] - u[3, 0]) / dx if abs(dx) > 1e-12 else np.nan
    return -e_xx / e_yy if abs(e_yy) > 1e-30 else np.nan


def solver_nu(tri, mesh, k):
    """The Phase 2 forward solve on the OPEN (non-periodic) mesh, reduced the same way."""
    mesh = dict(mesh)
    mesh['bond_k'] = k
    mesh['tri_k'] = k[mesh['tri_bond']]
    sv = SB.make_solver(mesh, MB.kkt_from_tri_bond(mesh['tri_bond'], mesh['edge_vecs']))
    out = sv.forward(torch.as_tensor(mesh['tri_k']),
                     rest_lengths=torch.as_tensor(np.sqrt(mesh['actual_len2'])),
                     method='intrinsic', physical_units=True)
    c6 = out['elastic_tensor'].detach().numpy()
    Cv = np.array([[c6[0], c6[2], c6[1]], [c6[2], c6[5], c6[4]], [c6[1], c6[4], c6[3]]])
    try:
        S = np.linalg.inv(Cv)
    except np.linalg.LinAlgError:
        return np.nan
    return float(-S[0, 1] / S[1, 1]) if abs(S[1, 1]) > 1e-300 else np.nan


def main():
    print('ONE hexagon, 7 nodes, 6 triangles, NO periodicity.  perimeter k=1, spokes k=%g' % EPS_SOFT)
    print('nu = pull along y, read along x  (d=2 regular, d<1 re-entrant, d->3 flat)\n')
    print('%6s %8s %14s %14s %12s' % ('d', 'y', 'SOLVER nu', 'TRUSS nu', 'gap'))
    for d in (0.05, 0.2, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 2.8, 2.9, 2.95, 2.99):
        tri, y = hexagon(d)
        mesh = MB.build_open_mesh(tri)
        k = k_of(mesh)
        try:
            s = solver_nu(tri, mesh, k)
        except Exception as e:
            s = np.nan
        try:
            t = truss_nu(tri.points, mesh['bond_u'], mesh['bond_v'], k)
        except Exception:
            t = np.nan
        gap = abs(s - t) if np.isfinite(s) and np.isfinite(t) else np.nan
        print('%6.2f %8.4f %+14.5f %+14.5f %12.5f' % (d, y, s, t, gap))
    print('\nd=3 exactly is FLAT (y=0, all 7 points collinear) — a real degeneracy, not numerical.')


if __name__ == '__main__':
    main()
