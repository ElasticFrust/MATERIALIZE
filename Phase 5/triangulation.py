"""Phase 5 M1b — explicit triangulation + periodic edge-flips.

FOUNDATIONAL module. Imported by seeds.py (tilings, via `geo_from_simplices`) and
designer.py (non-Delaunay candidates, via `random_flipped_geo`).

It factors the geometry-building block of `_common._periodic_delaunay` (everything AFTER
the `Delaunay(tiled).simplices` call) into a PURE function `geo_from_simplices` that builds
the same `geo` dict from an EXPLICIT triangle list, and adds a periodic edge-flip API.

Triangle representation used throughout ("tris"): an int array of shape (nt, 3, 3), where
`tris[t, k] == [base_index, sx, sy]`.  The real coordinate of that vertex is
`pts[base_index] + [sx*Lx, sy*Ly]`.  `base_index` indexes into `pts` (n points in
[0,Lx)x[0,Ly)); (sx,sy) is the integer periodic-image shift.

Public API
----------
    geo_from_simplices(pts, tris, Lx, Ly)        -> geo dict
    delaunay_tris(pts, Lx, Ly)                    -> tris  (canonical Delaunay start)
    legal_flips(tris, pts, Lx, Ly)               -> list[edge dict]
    apply_flip(tris, edge)                        -> tris_new
    random_flipped_geo(geo_or_pts, n_flips, seed, Lx=None, Ly=None) -> geo dict
"""
# ---- §0 preamble (verbatim; triangulation.py lives directly in Phase 5/) ----------------------
import os, sys
import numpy as np
import torch
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))   # repo root from Phase 5/
sys.path.insert(0, os.path.join(REPO, 'Phase 3', 'verifications'))
import _common as C                 # sets up all other sys.path and imports the solver stack
from inverse_design import (DesignProblem, Objective, optimize, validate, ANG,
                            c6_to_nuE, c6_to_nuE_theta)
torch.set_default_dtype(torch.float64)   # REQUIRED — the whole stack is float64

from scipy.spatial import Delaunay


# ---- 1. geo_from_simplices — the factored _periodic_delaunay tail ----------------------------
def geo_from_simplices(pts, tris, Lx, Ly):
    """Build the SAME `geo` dict `_common._periodic_delaunay` returns, but from an EXPLICIT
    triangle list `tris` (shape (nt,3,3): [base, sx, sy] per vertex) instead of computing a
    Delaunay.  Pure function of (pts, tris, Lx, Ly).  Reproduces _common's canonicalization
    exactly so bond_u/bond_v/bond_R/tri_bond match its conventions."""
    pts = np.asarray(pts, float)
    tris = np.asarray(tris, np.int64)
    n = len(pts); nt = len(tris); box = np.array([Lx, Ly])

    canon = tris[:, :, 0]                       # (nt,3) base indices  (== simp_t % n)
    sft = tris[:, :, 1:3]                       # (nt,3,2) integer image shifts (== shift_of[simp_t])
    P = pts[canon] + sft * box                  # (nt,3,2) real, image-correct coordinates
    p0, p1, p2 = P[:, 0], P[:, 1], P[:, 2]
    centroids = (p0 + p1 + p2) / 3.0            # TRUE centroids (image-correct)
    tri_verts = np.stack([p0, p1, p2], 1)       # (nt,3,2) image-correct vertices (for fills)
    edge_vecs = np.stack([p1 - p0, p2 - p0, p2 - p1], 1)
    l2 = (edge_vecs ** 2).sum(2)
    areas = 0.5 * np.abs(edge_vecs[:, 0, 0] * edge_vecs[:, 1, 1]
                         - edge_vecs[:, 0, 1] * edge_vecs[:, 1, 0])
    pairs = [(0, 1, 0), (0, 2, 1), (1, 2, 2)]
    keymap = {}; bonds = []; tri_bond = np.zeros((nt, 3), np.int64)
    for ti in range(nt):
        for ka, kb, ei in pairs:
            ca, cb = int(canon[ti, ka]), int(canon[ti, kb])
            d = sft[ti, kb] - sft[ti, ka]; dp = (int(d[0]), int(d[1])); R = edge_vecs[ti, ei]
            if (ca, dp[0], dp[1]) <= (cb, -dp[0], -dp[1]):
                key, Rk = (ca, cb, dp[0], dp[1]), R
            else:
                key, Rk = (cb, ca, -dp[0], -dp[1]), -R
            if key not in keymap:
                keymap[key] = len(bonds); bonds.append((key[0], key[1], Rk))
            tri_bond[ti, ei] = keymap[key]
    return dict(pts=pts, simplices=canon, edge_vecs=edge_vecs, actual_len2=l2,
                bond_u=np.array([b[0] for b in bonds], np.int64),
                bond_v=np.array([b[1] for b in bonds], np.int64),
                bond_R=np.array([b[2] for b in bonds], float),
                tri_bond=tri_bond, areas=areas, centroids=centroids, tri_verts=tri_verts,
                BL1=np.array([Lx, 0.0]), BL2=np.array([0.0, Ly]))


# ---- 2. delaunay_tris — the canonical starting triangulation for flips ------------------------
def delaunay_tris(pts, Lx, Ly):
    """Explicit `tris` list (shape (nt,3,3): [base, sx, sy] per vertex) for the periodic Delaunay
    of `pts`, using the 3x3-tile trick from `_common._periodic_delaunay`.  `geo_from_simplices`
    fed with this reproduces `_periodic_delaunay(pts, Lx, Ly)` exactly."""
    pts = np.asarray(pts, float)
    n = len(pts); box = np.array([Lx, Ly])
    shifts = np.array([(i, j) for i in (-1, 0, 1) for j in (-1, 0, 1)])
    tiled = np.concatenate([pts + s * box for s in shifts], axis=0)
    shift_of = np.repeat(shifts, n, axis=0)
    simp_t = Delaunay(tiled).simplices
    cen = tiled[simp_t].mean(1)
    keep = (cen[:, 0] >= 0) & (cen[:, 0] < Lx) & (cen[:, 1] >= 0) & (cen[:, 1] < Ly)
    simp_t = simp_t[keep]
    base = simp_t % n                           # (nt,3)
    sft = shift_of[simp_t]                       # (nt,3,2)
    return np.concatenate([base[:, :, None], sft], axis=2).astype(np.int64)   # (nt,3,3)


# ---- 3. periodic edge-flip API ---------------------------------------------------------------
def _rc(pts, v, box):
    """Real coordinate of a vertex v=[base,sx,sy]."""
    return pts[int(v[0])] + np.array([v[1] * box[0], v[2] * box[1]])


def _cross(o, p, q):
    return (p[0] - o[0]) * (q[1] - o[1]) - (p[1] - o[1]) * (q[0] - o[0])


def _edge_key(va, vb):
    """Canonical translation-invariant key for the edge between vertices va, vb (each [base,sx,sy]).
    Mirrors the bond canonicalization in _common (base indices + shift difference)."""
    ca, cb = int(va[0]), int(vb[0])
    d0, d1 = int(vb[1] - va[1]), int(vb[2] - va[2])
    if (ca, d0, d1) <= (cb, -d0, -d1):
        return (ca, cb, d0, d1)
    return (cb, ca, -d0, -d1)


def legal_flips(tris, pts, Lx, Ly):
    """Flippable interior edges of `tris`.  Each interior edge is shared by exactly 2 triangles
    (matched via the canonical periodic edge key).  A flip is LEGAL iff the quad formed by the
    edge's two endpoints (a,b) and the two opposite vertices (c,d) is STRICTLY CONVEX (c,d on
    opposite sides of line ab AND a,b on opposite sides of line cd — equivalently both resulting
    triangles have strictly positive signed area).  Returns a list of `edge` dicts consumable by
    `apply_flip`; each holds t1,t2 (rows of `tris`) and a,b,c,d as [base,sx,sy] in ONE common
    image frame (t2 shifted onto t1 so the shared edge coincides)."""
    tris = np.asarray(tris, np.int64)
    pts = np.asarray(pts, float); box = np.array([Lx, Ly])

    # group edge occurrences by canonical key
    occ = {}                                    # key -> list of (tri_idx, i_local, j_local)
    lpairs = [(0, 1), (1, 2), (2, 0)]
    for t in range(len(tris)):
        for i, j in lpairs:
            occ.setdefault(_edge_key(tris[t, i], tris[t, j]), []).append((t, i, j))

    flips = []
    for key, lst in occ.items():
        if len(lst) != 2:                       # boundary/degenerate — skip (torus: expect 2)
            continue
        (t1, i1, j1), (t2, i2, j2) = lst
        A1 = tris[t1, i1]; B1 = tris[t1, j1]
        c = tris[t1, 3 - i1 - j1]               # opposite vertex of t1
        # align t2 onto t1: find correspondence of t2's endpoints to (A1,B1) + integer shift delta
        e2 = [tris[t2, i2], tris[t2, j2]]
        d_local = tris[t2, 3 - i2 - j2]
        aligned = None
        for m, q in ((0, 1), (1, 0)):
            vm, vq = e2[m], e2[q]               # vm should map to A1, vq to B1
            if int(vm[0]) != int(A1[0]) or int(vq[0]) != int(B1[0]):
                continue
            delta = np.array([A1[1] - vm[1], A1[2] - vm[2]])     # shift applied to all of t2
            if vq[1] + delta[0] == B1[1] and vq[2] + delta[1] == B1[2]:
                aligned = delta
                break
        if aligned is None:
            continue
        d = d_local.copy()
        d[1] += aligned[0]; d[2] += aligned[1]  # opposite vertex of t2, in t1's frame

        A = _rc(pts, A1, box); B = _rc(pts, B1, box)
        Cc = _rc(pts, c, box); D = _rc(pts, d, box)
        # strict convexity of quad a-c-b-d (diagonals ab and cd cross in the interior)
        if _cross(A, B, Cc) * _cross(A, B, D) < 0 and _cross(Cc, D, A) * _cross(Cc, D, B) < 0:
            flips.append(dict(t1=t1, t2=t2, a=A1.copy(), b=B1.copy(), c=c.copy(), d=d))
    return flips


def _orient_ccw(pts, v0, v1, v2, box):
    """Return (v0,v1,v2) reordered so the triangle has positive (CCW) signed area."""
    if _cross(_rc(pts, v0, box), _rc(pts, v1, box), _rc(pts, v2, box)) < 0:
        return v0, v2, v1
    return v0, v1, v2


def apply_flip(tris, edge, pts=None, Lx=None, Ly=None):
    """Replace the two triangles sharing edge (a,b) — {a,b,c} and {a,b,d} — with {a,c,d} and
    {c,b,d}, in the common image frame carried by `edge`.  Returns a new tris array (positions
    fixed).  `pts,Lx,Ly` are optional and used only to orient the new triangles CCW; if omitted
    the triangles are stored in the given order (geo_from_simplices uses |area|, so orientation
    does not affect correctness)."""
    tris = np.asarray(tris, np.int64).copy()
    a, b, c, d = edge['a'], edge['b'], edge['c'], edge['d']
    if pts is not None:
        box = np.array([Lx, Ly]); pts = np.asarray(pts, float)
        n1 = np.stack(_orient_ccw(pts, a, c, d, box))
        n2 = np.stack(_orient_ccw(pts, c, b, d, box))
    else:
        n1 = np.stack([a, c, d]); n2 = np.stack([c, b, d])
    tris[edge['t1']] = n1
    tris[edge['t2']] = n2
    return tris


def random_flipped_geo(geo_or_pts, n_flips, seed, Lx=None, Ly=None):
    """Start from the Delaunay tris of the points, apply up to `n_flips` random LEGAL flips, and
    return `geo_from_simplices(...)`.  This is what designer.py calls for non-Delaunay candidates.
    `geo_or_pts` may be a `geo` dict (Lx,Ly taken from its BL1/BL2 if not given) or a raw (n,2)
    point array (then Lx,Ly are required)."""
    if isinstance(geo_or_pts, dict):
        pts = np.asarray(geo_or_pts['pts'], float)
        if Lx is None:
            Lx = float(geo_or_pts['BL1'][0])
        if Ly is None:
            Ly = float(geo_or_pts['BL2'][1])
    else:
        pts = np.asarray(geo_or_pts, float)
    if Lx is None or Ly is None:
        raise ValueError("Lx and Ly must be given when passing a raw point array")
    rng = np.random.default_rng(seed)
    tris = delaunay_tris(pts, Lx, Ly)
    for _ in range(n_flips):
        fl = legal_flips(tris, pts, Lx, Ly)
        if not fl:
            break
        edge = fl[int(rng.integers(len(fl)))]
        tris = apply_flip(tris, edge, pts, Lx, Ly)
    return geo_from_simplices(pts, tris, Lx, Ly)


# ---- self-test -------------------------------------------------------------------------------
def _solver_nuE(geo):
    prob = DesignProblem.from_geo(geo)
    out = prob.forward(torch.ones(prob.n_bond))
    nu, E = c6_to_nuE(prob.region_tensor(out['per_triangle'], None))
    return float(nu), float(E)


def _self_test():
    # (a) round-trip: geo_from_simplices + delaunay_tris reproduce _periodic_delaunay
    geo0 = C.make_lattice(1.0, 1.0, half=6)
    Lx, Ly = float(geo0['BL1'][0]), float(geo0['BL2'][1])
    tris = delaunay_tris(geo0['pts'], Lx, Ly)
    geo1 = geo_from_simplices(geo0['pts'], tris, Lx, Ly)

    assert len(geo1['simplices']) == len(geo0['simplices']), \
        f"tri count {len(geo1['simplices'])} != {len(geo0['simplices'])}"
    assert len(geo1['bond_u']) == len(geo0['bond_u']), \
        f"bond count {len(geo1['bond_u'])} != {len(geo0['bond_u'])}"
    nu0, E0 = _solver_nuE(geo0)
    nu1, E1 = _solver_nuE(geo1)
    assert abs(nu0 - nu1) < 1e-8 and abs(E0 - E1) < 1e-8, \
        f"round-trip nu,E mismatch: geo0=({nu0},{E0}) geo1=({nu1},{E1})"
    print(f"(a) round-trip OK: nt={len(geo1['simplices'])} nbond={len(geo1['bond_u'])} "
          f"nu={nu1:.6f} E={E1:.6f}  (|dnu|={abs(nu0-nu1):.1e} |dE|={abs(E0-E1):.1e})")

    # (b) flip validity: flipped net is a VALID network and solver ~ independent sim
    geoF = random_flipped_geo(geo0, n_flips=5, seed=3, Lx=Lx, Ly=Ly)
    assert (geoF['areas'] > 0).all(), "flipped net has non-positive triangle areas"
    used = np.unique(geoF['tri_bond'])
    assert len(used) == len(geoF['bond_u']), \
        f"some bonds unused by triangles: {len(used)} used of {len(geoF['bond_u'])}"
    assert set(used.tolist()) == set(range(len(geoF['bond_u']))), "bond indexing not contiguous"

    nuF, EF = _solver_nuE(geoF)
    C.apply_k_to_geo(geoF, np.ones(len(geoF['bond_u'])))
    nu_s, E_s = C.sim_region_nuE(geoF)
    assert abs(nuF - nu_s) < 0.02, f"solver vs sim nu gap too big: {nuF} vs {nu_s}"
    assert abs(EF - E_s) < 0.05, f"solver vs sim E gap too big: {EF} vs {E_s}"
    print(f"(b) flip validity OK: nt={len(geoF['simplices'])} nbond={len(geoF['bond_u'])}  "
          f"solver nu={nuF:.6f} E={EF:.6f} | sim nu={nu_s:.6f} E={E_s:.6f}  "
          f"(|dnu|={abs(nuF-nu_s):.1e} |dE|={abs(EF-E_s):.1e})")
    print(f"    (un-flipped lattice nu={nu0:.6f} E={E0:.6f} — flip changes connectivity, so these differ)")

    print("TRIANGULATION SELF-TEST PASSED")


if __name__ == '__main__':
    _self_test()
