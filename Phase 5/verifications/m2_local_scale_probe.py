r"""Is the single GLOBAL displacement scale throttling vertices? (user, 2026-09-16)

The user: *"The overall range can only be a function of the maximal distance of the neighbouring
triangles. It's not, and cannot be globally constrained."* Right on the first half, and this measures
how much the global scale costs.

THREE SCHEMES, all using the same exact per-triangle inversion algebra:

    global        s_v = t*, the min over ALL triangles                 <- what `displace_safe` does
    naive local   s_v = min over triangles INCIDENT to v of that triangle's own limit
    local fixed   start at naive local, then repeatedly shrink ONLY the vertices of triangles that
                  are actually violated, until none are

The naive local scheme is the obvious reading of "limit each vertex by its own neighbourhood", and it
is NOT safe: a triangle inverts from its three vertices moving TOGETHER, so three vertices each at
their individual limit can still flip it. Measured: it inverts in 9 of 9 cases.

The fixed point repairs that while staying local -- each update touches one triangle's three
vertices, so information travels only between triangles that share a vertex. Nothing global is
computed, which is the user's point.

Run:
    python "Phase 5/verifications/m2_local_scale_probe.py"
"""
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'Phase 2'))
sys.path.insert(0, os.path.join(REPO, 'Phase 5', 'm2'))
import mesh_build as MB                                                   # noqa: E402
import fields as F                                                        # noqa: E402


def add_box(g):
    """Recover the periodic box from the bonds' own integer image offsets."""
    bu = np.asarray(g['bond_u'], np.int64); bv = np.asarray(g['bond_v'], np.int64)
    p = np.asarray(g['pts'], float)
    sh = np.abs(np.asarray(g['bond_R'], float) - (p[bv] - p[bu]))
    g['BL1'] = np.array([sh[:, 0][sh[:, 0] > 1e-9].min(), 0.0])
    g['BL2'] = np.array([0.0, sh[:, 1][sh[:, 1] > 1e-9].min()])
    return g


def signed_areas(geo, pts):
    bu = np.asarray(geo['bond_u'], np.int64); bv = np.asarray(geo['bond_v'], np.int64)
    p0 = np.asarray(geo['pts'], float)
    shift = np.asarray(geo['bond_R'], float) - (p0[bv] - p0[bu])
    bR = np.asarray(pts, float)[bv] - np.asarray(pts, float)[bu] + shift
    ei, sg = MB.edge_vec_orientation(geo['tri_bond'], geo['simplices'], bu, bv)
    sg = np.where(sg == 0.0, 1.0, sg)
    ev = bR[ei] * sg[..., None]
    return 0.5 * (ev[:, 0, 0] * ev[:, 1, 1] - ev[:, 0, 1] * ev[:, 1, 0])


def per_triangle_limit(geo, d):
    """Exact first-inversion scale for EACH triangle on its own -- its three vertices only."""
    ei, sg = MB.edge_vec_orientation(geo['tri_bond'], geo['simplices'],
                                     geo['bond_u'], geo['bond_v'])
    sg = np.where(sg == 0.0, 1.0, sg)
    ev = np.asarray(geo['bond_R'], float)[ei] * sg[..., None]
    sm = np.asarray(geo['simplices'], np.int64)
    d = np.asarray(d, float)
    de1 = d[sm[:, 1]] - d[sm[:, 0]]
    de2 = d[sm[:, 2]] - d[sm[:, 0]]
    cr = lambda a, b: a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]               # noqa: E731
    c0, c2 = cr(ev[:, 0], ev[:, 1]), cr(de1, de2)
    c1 = cr(ev[:, 0], de2) + cr(de1, ev[:, 1])
    s = np.sign(c0)
    a, b, c = s * c2, s * c1, s * c0
    t = np.full(len(sm), np.inf)
    lin = np.abs(a) < 1e-300
    with np.errstate(invalid='ignore', divide='ignore'):
        tl = np.where(lin & (b < 0), -c / b, np.inf)
        disc = b * b - 4 * a * c
        rt = np.sqrt(np.where(disc > 0, disc, 0.0))
        real = (~lin) & (disc > 0)
        r1 = np.where(real, (-b - rt) / (2 * a), np.inf)
        r2 = np.where(real, (-b + rt) / (2 * a), np.inf)
    for r in (tl, r1, r2):
        r = np.where(np.isfinite(r) & (r > 0), r, np.inf)
        t = np.minimum(t, r)
    return t


def local_scale_field(geo, d, margin=0.05, shrink=0.85, iters=600):
    """Per-vertex scale by LOCAL fixed point. No global minimum is ever taken.

    Start optimistic -- each vertex at the min over its own incident triangles -- then repeatedly
    shrink only the vertices of triangles that are actually violated. Each update touches one
    triangle's three vertices, so information travels only between triangles sharing a vertex.
    `margin` keeps a triangle from being driven arbitrarily thin, not merely un-inverted."""
    sm = np.asarray(geo['simplices'], np.int64)
    p0 = np.asarray(geo['pts'], float)
    tt = per_triangle_limit(geo, d)
    s = np.full(len(p0), np.inf)
    np.minimum.at(s, sm.ravel(), np.repeat(tt, 3))
    s = np.minimum(s, 1e6)
    a0 = signed_areas(geo, p0)
    sgn = np.sign(a0)
    for it in range(iters):
        a = signed_areas(geo, p0 + s[:, None] * d)
        bad = (np.sign(a) != sgn) | (np.abs(a) < margin * np.abs(a0))
        if not bad.any():
            return s, it
        np.multiply.at(s, sm[bad].ravel(), shrink)
    return s, iters


def main():
    print('Does the single GLOBAL scale throttle vertices their own neighbourhood would allow?')
    print('  global      s_v = t*, min over ALL triangles        (what displace_safe does)')
    print('  naive local s_v = min over INCIDENT triangles       (unsafe -- shown below)')
    print('  local fixed local fixed point, nothing global       (safe -- shown below)')
    print()
    print('%-9s %-5s %10s %11s %8s %11s %9s %9s %6s'
          % ('mesh', 'seed', 'global t*', 'naive med', 'naive?', 'localfix med', 'gain med',
             'gain max', 'iters'))
    bad_naive = bad_fix = 0
    for eta, tag in ((0.0, 'regular'), (0.2, 'eta0.2'), (0.35, 'eta0.35')):
        g = add_box(MB.build_geometry(10, eta, seed=1))
        MB.set_VD(g, 0)
        p0 = np.asarray(g['pts'], float)
        sgn = np.sign(signed_areas(g, p0))
        sm = np.asarray(g['simplices'], np.int64)
        for sd in (0, 1, 2):
            rng = np.random.default_rng(sd)
            ang = rng.uniform(0, 2 * np.pi, len(p0))
            d = np.stack([np.cos(ang), np.sin(ang)], 1)

            tg = F.first_inversion_scale(g, d)
            tt = per_triangle_limit(g, d)
            sn = np.full(len(p0), np.inf)
            np.minimum.at(sn, sm.ravel(), np.repeat(tt, 3))
            inv_n = bool(np.any(np.sign(signed_areas(g, p0 + 0.99 * sn[:, None] * d)) != sgn))
            bad_naive += inv_n

            s, it = local_scale_field(g, d)
            inv_f = bool(np.any(np.sign(signed_areas(g, p0 + s[:, None] * d)) != sgn))
            bad_fix += inv_f
            print('%-9s %-5d %10.4f %11.4f %8s %11.4f %8.2fx %8.2fx %6d'
                  % (tag, sd, tg, np.median(sn), 'INVERTS' if inv_n else 'ok',
                     np.median(s), np.median(s) / tg, s.max() / tg, it))
    print()
    print('naive local inverted in %d of 9 -- three vertices each at their own limit still flip the '
          'triangle they share' % bad_naive)
    print('local fixed point inverted in %d of 9' % bad_fix)
    return 1 if bad_fix else 0


if __name__ == '__main__':
    raise SystemExit(main())
