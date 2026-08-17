r"""Where does the SOLVER work? — the hexagon diameter family, both triangulations.

**Question (user, 2026-08-17):** does the solver run correctly, on cases where we know what to
expect? Audit **A-17** showed it is wrong on MALFORMED triangulations (rotating squares: solver
-0.685 where the textbook/sim value is -0.974) while being exact on a valid mesh across eight orders
of stiffness contrast. This maps out *where* the boundary is, on a family with a controlled knob.

**The construction (user's spec).** One hexagon, all six PERIMETER edges hard (k=1) and length 1.
Pick two opposite vertices; their separation `d` is the control. Squeeze/expand d while every
perimeter edge stays length 1, so the other vertices follow:

    y = sqrt(1 - ((1-d)/2)^2),  rim = [(+d/2,0), (+1/2,+y), (-1/2,+y), (-d/2,0), (-1/2,-y), (+1/2,-y)]

    d = 2  regular hexagon       d < 1  RE-ENTRANT        d -> 0  bow-tie
    d = 3  fully expanded (y=0, degenerate)

A hexagon is not a triangle, so it must be triangulated by SOFT fictional springs. **Two variants,
both tested** (the point of this script):

    (a) CHORDS   — 3 chords inside the polygon, V0-V2, V2-V4, V4-V0: a central triangle plus three
                   corner triangles. 4 triangles per hexagon, no new vertex. 3-fold symmetric.
    (b) CENTRE   — a fictitious vertex at the hexagon centre with 6 radial spokes: a fan of 6
                   triangles. This is what `dhex_family.py` already builds; reused via `build_dhex`.

**The measurement.** Pull PERPENDICULAR to the diameter (along y) and read the response ALONG the
diameter (x). From the compliance S = C^-1 in Voigt [xx,yy,xy] that is

    nu = -S[0,1] / S[1,1]          (uniaxial stress along y -> -eps_xx/eps_yy)

computed IDENTICALLY from the solver's tensor and from the INDEPENDENT sim's tensor, so the
comparison cannot be contaminated by a reduction-convention mismatch (audit A-10).

Mesh validity is reported per point: a closed periodic triangulation needs every bond in exactly 2
triangles and V-E+F = 0 (A-17). That is the suspected discriminator.

Run:  python hex_solver_validation.py [n_d]
Out:  Phase 5/results/hex_validation/
"""
import os
import sys
from collections import Counter

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'Phase 3', 'verifications'))
import _common as C                                                    # noqa: E402
sys.path.insert(0, os.path.join(REPO, 'Phase 5'))
sys.path.insert(0, HERE)
import physical_homog as PH                                            # noqa: E402
import triangulation                                                   # noqa: E402
from inverse_design import DesignProblem                               # noqa: E402
from dhex_family import build_dhex                                     # noqa: E402  (variant b)

torch.set_default_dtype(torch.float64)
OUT = os.path.join(REPO, 'Phase 5', 'results', 'hex_validation')


def rim_of(d):
    """The 6 rim vertices at x-diameter d, all perimeter edges length 1 (user's spec)."""
    y = np.sqrt(max(1.0 - ((1.0 - d) / 2.0) ** 2, 0.0))
    return np.array([(d / 2, 0.0), (0.5, y), (-0.5, y), (-d / 2, 0.0), (-0.5, -y), (0.5, -y)]), y


def build_chords(nx, ny, d, eps=1e-3):
    """Variant (a): hexagons triangulated by 3 INTERNAL CHORDS (V0-V2, V2-V4, V4-V0), no centre.

    4 triangles per hexagon: the central (V0,V2,V4) plus corners (V0,V1,V2), (V2,V3,V4), (V4,V5,V0).
    Chord springs are SOFT (k=eps); the 6 perimeter edges are HARD (k=1)."""
    rim, y = rim_of(d)
    a2 = np.array([(1.0 + d) / 2.0, y])
    Lx, Ly = nx * (1.0 + d), ny * (2.0 * y)
    box = np.array([Lx, Ly])

    centers = []
    for i in range(nx):
        for j in range(ny):
            base = np.array([i * (1.0 + d), j * 2.0 * y])
            centers.append(base)
            centers.append(base + a2)

    key2idx, pts = {}, []

    def vid(real):
        w = np.mod(real, box)
        k = (round(float(w[0]), 5), round(float(w[1]), 5))
        if k not in key2idx:
            key2idx[k] = len(pts); pts.append(w)
        return key2idx[k], np.round((real - w) / box).astype(int)

    tris = []
    for c in centers:
        idx, sh = zip(*[vid(c + r) for r in rim])            # the 6 rim vertices of this hexagon
        def V(a):
            return [idx[a], sh[a][0], sh[a][1]]
        for (a, b, cc) in ((0, 1, 2), (2, 3, 4), (4, 5, 0), (0, 2, 4)):
            tris.append([V(a), V(b), V(cc)])

    pts = np.array(pts)
    geo = triangulation.geo_from_simplices(pts, np.array(tris, np.int64), Lx, Ly)
    # perimeter edges connect ADJACENT rim vertices (length 1); chords are the rest -> soft
    L = np.linalg.norm(np.asarray(geo['bond_R'], float), axis=1)
    is_soft = np.abs(L - 1.0) > 1e-6
    return geo, np.where(is_soft, eps, 1.0), is_soft


def mesh_validity(geo):
    """(is_valid, bond->#triangles histogram, V-E+F). A-17's discriminator."""
    cnt = Counter(np.asarray(geo['tri_bond']).ravel().tolist())
    hist = dict(sorted(Counter(cnt.values()).items()))
    V, E, F = len(geo['pts']), len(geo['bond_u']), len(geo['simplices'])
    return (hist == {2: E} and V - E + F == 0), hist, V - E + F


def nu_pull_y(c6):
    """nu measured by pulling along y and reading x: -S[0,1]/S[1,1], S = C^-1 in Voigt [xx,yy,xy].

    The SAME reduction is applied to the solver tensor and the sim tensor, so a convention
    difference cannot masquerade as disagreement (audit A-10)."""
    c6 = np.asarray(c6, float)
    Cv = np.array([[c6[0], c6[2], c6[1]], [c6[2], c6[5], c6[4]], [c6[1], c6[4], c6[3]]])
    try:
        S = np.linalg.inv(Cv)
    except np.linalg.LinAlgError:
        return float('nan')
    return float(-S[0, 1] / S[1, 1]) if abs(S[1, 1]) > 1e-300 else float('nan')


def main():
    n_d = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    os.makedirs(OUT, exist_ok=True)
    ds = np.linspace(0.2, 2.8, n_d)

    for label, builder in (('(a) CHORDS  ', build_chords), ('(b) CENTRE  ', build_dhex)):
        print(f'\n=== variant {label} ===')
        print('%6s %11s %11s %10s %10s %-22s %s'
              % ('d', 'solver nu', 'sim nu', 'gap', 'mesh ok', 'bond->#tri', 'V-E+F'))
        for d in ds:
            try:
                geo, k0, _ = builder(2, 2, float(d))
            except Exception as e:
                print('%6.2f  build EXC %s' % (d, type(e).__name__)); continue
            ok, hist, chi = mesh_validity(geo)
            C.apply_k_to_geo(geo, k0)
            try:
                cs = C.solver_region_C6(DesignProblem.from_geo(geo), torch.as_tensor(k0))
                s = nu_pull_y(cs)
            except Exception as e:
                s = float('nan')
            try:
                m = nu_pull_y(C.sim_bulk_C6(geo))
            except PH.UnhealthyGeometryError:
                m = float('nan')
            except Exception:
                m = float('nan')
            gap = abs(s - m) if np.isfinite(s) and np.isfinite(m) else float('nan')
            print('%6.2f %+11.5f %+11.5f %10.5f %10s %-22s %+d'
                  % (d, s, m, gap, 'YES' if ok else 'no', str(hist)[:22], chi))
    print(f'\nd=2 is the regular hexagon; d<1 re-entrant; d->0 bow-tie; d->3 flat.')
    print('nu is pull-along-y / read-along-x, IDENTICAL reduction on both sides.')


if __name__ == '__main__':
    main()
