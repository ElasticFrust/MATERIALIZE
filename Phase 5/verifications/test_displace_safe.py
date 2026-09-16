r"""Gate `fields.displace_safe` — the perturbation that CANNOT invert a triangle.

The claim is a guarantee, not a tendency, so it is tested at the extreme and on meshes that are
already awkward. What is checked:

  [1] the altitude is right — `node_min_altitude` against a brute-force per-triangle computation.
  [2] NO INVERSION at any `frac` up to 1.0, over many seeds and several mesh kinds, including
      already-disordered ones where the local length scale varies a lot. This is the guarantee.
  [3] it actually MOVES things — a bound that protects by doing nothing would pass [2] trivially.
  [4] `displace` (the global-mean version) DOES break meshes where `displace_safe` does not, on the
      same mesh and seed. Without this the new function's value is asserted rather than shown.
  [5] no wrapping: `displace_safe` must not apply `np.mod`, which on frozen connectivity
      reconstructs a triangle a box away (`seeds.py`: 13 of 72 inverted, signed area −15.2, at
      eta = 0.05).

Run:
    python "Phase 5/verifications/test_displace_safe.py"
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'Phase 2'))
sys.path.insert(0, os.path.join(REPO, 'Phase 5', 'm2'))
import mesh_build as MB                                                   # noqa: E402
import fields as F                                                        # noqa: E402


def _add_box(geo):
    """Recover the periodic box from the bonds' own image offsets, and store it as BL1/BL2.

    `build_geometry` does not carry a box (the dataset records get one from `seeds`), and
    `fields.box_of` needs one. It is recoverable exactly: `bond_R - (pts[v] - pts[u])` is an integer
    multiple of the box vectors, so the smallest nonzero |offset| on each axis IS that axis's
    period. Done here rather than in `fields` so the gate does not change production code."""
    bu = np.asarray(geo['bond_u'], np.int64); bv = np.asarray(geo['bond_v'], np.int64)
    p = np.asarray(geo['pts'], float)
    sh = np.abs(np.asarray(geo['bond_R'], float) - (p[bv] - p[bu]))
    Lx = sh[:, 0][sh[:, 0] > 1e-9].min()
    Ly = sh[:, 1][sh[:, 1] > 1e-9].min()
    geo['BL1'] = np.array([Lx, 0.0])
    geo['BL2'] = np.array([0.0, Ly])
    return geo


def _meshes():
    """A regular lattice, a disordered one, and a strongly disordered one."""
    out = []
    for eta, tag in ((0.0, 'regular'), (0.2, 'eta0.2'), (0.35, 'eta0.35')):
        g = MB.build_geometry(6, eta, seed=1)
        MB.set_VD(g, 0)
        out.append((tag, _add_box(g)))
    return out


def _signed_areas(geo, pts=None):
    """Signed area per triangle, from the PERIODIC oriented edge vectors."""
    g = dict(geo)
    if pts is not None:
        bu = np.asarray(geo['bond_u'], np.int64); bv = np.asarray(geo['bond_v'], np.int64)
        p0 = np.asarray(geo['pts'], float)
        shift = np.asarray(geo['bond_R'], float) - (p0[bv] - p0[bu])
        g['bond_R'] = np.asarray(pts, float)[bv] - np.asarray(pts, float)[bu] + shift
    ei, sg = MB.edge_vec_orientation(g['tri_bond'], g['simplices'], g['bond_u'], g['bond_v'])
    sg = np.where(sg == 0.0, 1.0, sg)
    ev = np.asarray(g['bond_R'], float)[ei] * sg[..., None]
    return 0.5 * (ev[:, 0, 0] * ev[:, 1, 1] - ev[:, 0, 1] * ev[:, 1, 0])


def test_altitude():
    """[1] `node_min_altitude` against a brute-force reference."""
    worst = 0.0
    for tag, geo in _meshes():
        got = F.node_min_altitude(geo)
        ei, sg = MB.edge_vec_orientation(geo['tri_bond'], geo['simplices'],
                                         geo['bond_u'], geo['bond_v'])
        sg = np.where(sg == 0.0, 1.0, sg)
        ev = np.asarray(geo['bond_R'], float)[ei] * sg[..., None]
        ref = np.full(len(geo['pts']), np.inf)
        for s, tri in enumerate(np.asarray(geo['simplices'], np.int64)):
            a = abs(ev[s, 0, 0] * ev[s, 1, 1] - ev[s, 0, 1] * ev[s, 1, 0]) / 2.0
            opp = [np.linalg.norm(ev[s, 2]), np.linalg.norm(ev[s, 1]), np.linalg.norm(ev[s, 0])]
            for c in range(3):
                ref[tri[c]] = min(ref[tri[c]], 2.0 * a / opp[c])
        worst = max(worst, float(np.abs(got - ref).max() / max(ref.max(), 1e-300)))
    ok = worst < 1e-12
    print('[1] node_min_altitude vs brute force: worst rel %.3e  %s' % (worst, 'OK' if ok else 'FAIL'))
    return 0 if ok else 1


def test_no_inversion():
    """[2] THE GUARANTEE: no triangle inverts at any frac < 1, over seeds and mesh kinds."""
    bad, n = 0, 0
    worst_shrink = 1.0
    for tag, geo in _meshes():
        a0 = _signed_areas(geo)
        for frac in (0.25, 0.5, 0.75, 0.9, 0.99):
            for structure in ('white', 'correlated'):
                for seed in range(6):
                    pts, _ = F.displace_safe(geo, np.random.default_rng(seed), frac=frac,
                                             structure=structure)
                    a1 = _signed_areas(geo, pts)
                    n += 1
                    if np.any(np.sign(a1) != np.sign(a0)):
                        bad += 1
                    worst_shrink = min(worst_shrink, float((np.abs(a1) / np.abs(a0)).min()))
    ok = bad == 0
    print('[2] NO INVERSION over %d perturbations (3 meshes x frac<=0.99 x 2 structures x 6 seeds): '
          '%d inverted; worst area ratio %.4f  %s'
          % (n, bad, worst_shrink, 'OK' if ok else 'FAIL'))
    return 0 if ok else 1


def test_scale_is_exact_not_conservative():
    """[2b] `first_inversion_scale` is the TRUE limit: just below it nothing inverts, just above it
    something does. A merely-sufficient bound would pass the first half and fail the second."""
    bad, rows = 0, []
    for tag, geo in _meshes():
        a0 = _signed_areas(geo)
        for seed in range(4):
            rng = np.random.default_rng(seed)
            a = rng.uniform(0, 2 * np.pi, len(geo['pts']))
            d = np.stack([np.cos(a), np.sin(a)], 1)
            t = F.first_inversion_scale(geo, d)
            pts_lo = np.asarray(geo['pts'], float) + 0.999 * t * d
            pts_hi = np.asarray(geo['pts'], float) + 1.001 * t * d
            lo_ok = not np.any(np.sign(_signed_areas(geo, pts_lo)) != np.sign(a0))
            hi_bad = np.any(np.sign(_signed_areas(geo, pts_hi)) != np.sign(a0))
            bad += (not lo_ok) or (not hi_bad)
            rows.append((tag, t, lo_ok, hi_bad))
    ok = bad == 0
    print('[2b] t* is EXACT over %d cases: 0.999*t* safe and 1.001*t* inverts in every one  %s'
          % (len(rows), 'OK' if ok else 'FAIL'))
    if not ok:
        for tag, t, lo, hi in rows:
            if (not lo) or (not hi):
                print('     %-9s t*=%.4f  0.999 safe=%s  1.001 inverts=%s' % (tag, t, lo, hi))
    return 0 if ok else 1


def test_reaches_classical_eta():
    """[2c] The reachable amplitude covers the classical eta range, eta < 0.5.

    The familiar bound is a COLLISION bound on the regular lattice: neighbours a unit apart, each
    moving up to eta, touch at eta = 0.5. Area-positivity subsumes collision (two bonded nodes
    cannot meet without first flattening the triangles on that edge), so the question is simply
    whether `frac -> 1` reaches that far in eta units. The superseded `h_v/3` bound did NOT: it
    stopped at 0.289 on an equilateral triangle."""
    rows = []
    for tag, geo in _meshes():
        best = 0.0
        for seed in range(6):
            _, meta = F.displace_safe(geo, np.random.default_rng(seed), frac=0.99,
                                      structure='white')
            best = max(best, meta['geom_eta_equiv'])
        rows.append((tag, best))
    reg = dict(rows)['regular']
    ok = reg > 0.45
    print('[2c] eta-equivalent at frac=0.99: %s  (regular must exceed 0.45; the old h/3 bound '
          'capped at 0.289)  %s'
          % (', '.join('%s %.3f' % r for r in rows), 'OK' if ok else 'FAIL'))
    return 0 if ok else 1


def test_local_scale_is_size_independent():
    """[2d] The field is shaped by each node's OWN margin, so one bad triangle cannot throttle the
    whole mesh -- and the reachable amplitude stops degrading with mesh size.

    Every triangle constrains its three vertices JOINTLY, so a per-vertex cap computed independently
    is not safe (A, B and C can each be within their own limit and still flip ABC together). The
    guarantee therefore has to come from one exact global scale. But with a UNIT field that scale is
    set by the single worst triangle anywhere in the mesh, so a larger mesh -- more chances for one
    unlucky triangle -- moves LESS. Scaling the field by `node_min_altitude` removes that.

    Measured (eta-equivalent at frac=0.99, white field, 6 seeds):
        regular      unit 0.465   local 0.465    (uniform mesh: nothing to adapt, as expected)
        eta0.35      unit 0.232   local 0.409
        eta0.35 big  unit 0.184   local 0.424    <- unit DEGRADES with size, local does not
    """
    out = {}
    for eta in (0.0, 0.35):
        for N in (6, 12):
            g = MB.build_geometry(N, eta, seed=1); MB.set_VD(g, 0); _add_box(g)
            a0 = _signed_areas(g)
            for ls in (False, True):
                best, inv = 0.0, 0
                for sd in range(6):
                    pts, meta = F.displace_safe(g, np.random.default_rng(sd), frac=0.99,
                                                structure='white', local_scale=ls)
                    best = max(best, meta['geom_eta_equiv'])
                    inv += int(np.any(np.sign(_signed_areas(g, pts)) != np.sign(a0)))
                out[(eta, N, ls)] = (best, inv)
    no_inv = all(v[1] == 0 for v in out.values())
    # on the heterogeneous mesh, local scaling must not shrink as the mesh grows
    unit_shrinks = out[(0.35, 12, False)][0] < out[(0.35, 6, False)][0]
    local_holds = out[(0.35, 12, True)][0] >= 0.95 * out[(0.35, 6, True)][0]
    better = out[(0.35, 12, True)][0] > 1.5 * out[(0.35, 12, False)][0]
    ok = no_inv and unit_shrinks and local_holds and better
    print('[2d] eta0.35: unit %.3f (N=6) -> %.3f (N=12), shrinks; local %.3f -> %.3f, holds and is '
          '%.1fx better at N=12; 0 inversions either way  %s'
          % (out[(0.35, 6, False)][0], out[(0.35, 12, False)][0],
             out[(0.35, 6, True)][0], out[(0.35, 12, True)][0],
             out[(0.35, 12, True)][0] / max(out[(0.35, 12, False)][0], 1e-9),
             'OK' if ok else 'FAIL'))
    return 0 if ok else 1


def test_every_vertex_uses_its_own_margin_equally():
    """[2e] With the locally-shaped field every node uses the SAME fraction of its OWN margin.

    This is the property the global scalar was suspected of breaking: does one scalar throttle the
    nodes that could have moved further? Measured, no -- the per-node utilisation `move_v / h_v` has
    spread exactly 1.00 with `local_scale=True`, against 1.78 (eta=0.2) and 4.48 (eta=0.35) with the
    unit field, where some nodes use 19 % of their budget and others 74 %.

    So each vertex IS limited by its own tightest incident triangle; the single global scalar only
    picks the common fraction, and picks it exactly at the first inversion. A scheme with no global
    coupling would have to budget each vertex blind to its neighbours, and since a triangle flips
    from all three moving together that forces a division by 3 -- the conservatism the exact solve
    exists to remove. Identity by construction (move_v/h_v = t/median(h)); gated so a change to the
    normalisation cannot quietly break it."""
    rows, bad = [], 0
    for eta, tag in ((0.2, 'eta0.2'), (0.35, 'eta0.35')):
        g = MB.build_geometry(12, eta, seed=1); MB.set_VD(g, 0); _add_box(g)
        h = F.node_min_altitude(g); p0 = np.asarray(g['pts'], float)
        sp = {}
        for ls in (False, True):
            pts, _ = F.displace_safe(g, np.random.default_rng(0), frac=0.99, structure='white',
                                     local_scale=ls)
            u = np.linalg.norm(pts - p0, axis=1) / h
            sp[ls] = float(u.max() / u.min())
        rows.append((tag, sp[False], sp[True]))
        bad += not (sp[True] < 1.001 and sp[False] > 1.5)
    ok = bad == 0
    print('[2e] per-node use of its OWN margin, spread max/min: %s  (local must be 1.00)  %s'
          % ('; '.join('%s unit %.2f -> local %.2f' % r for r in rows), 'OK' if ok else 'FAIL'))
    return 0 if ok else 1


def test_arbitrary_amplitude_profile():
    """[2f] An ARBITRARY per-node amplitude profile is honoured exactly, and still cannot invert.

    THE VERDICT ON LOCAL AMPLITUDE CONTROL, in three parts:
      * the RELATIVE amplitude of every vertex is fully controllable -- pass any non-negative profile
        and the realised displacements reproduce it to machine precision (checked here by correlating
        requested against realised, which must be 1.0);
      * the overall SCALE is not the caller's to choose: it is set by the tightest triangle and
        solved exactly, so the profile is honoured up to one common multiplier;
      * an INDEPENDENT per-vertex absolute guarantee is impossible in principle, not merely
        unimplemented -- a triangle inverts from all three of its vertices moving together, so the
        binding constraint couples three nodes and no cap computed per-vertex in isolation is safe.

    Tested with two deliberately awkward profiles: a single HOT node (everything else still), and a
    smooth spatial ramp. Both must be reproduced in shape and neither may invert a triangle."""
    g = MB.build_geometry(10, 0.2, seed=1); MB.set_VD(g, 0); _add_box(g)
    a0 = _signed_areas(g)
    p0 = np.asarray(g['pts'], float)
    n = len(p0)
    bad, rows = 0, []

    profiles = {}
    hot = np.zeros(n); hot[n // 3] = 1.0
    profiles['single hot node'] = hot
    profiles['spatial ramp'] = (p0[:, 0] - p0[:, 0].min()) / max(float(np.ptp(p0[:, 0])), 1e-300)
    profiles['two-scale'] = np.where(np.arange(n) % 4 == 0, 1.0, 0.1)

    for name, amp in profiles.items():
        pts, meta = F.displace_safe(g, np.random.default_rng(0), frac=0.9, structure='white',
                                    amp=amp)
        moved = np.linalg.norm(pts - p0, axis=1)
        inv = bool(np.any(np.sign(_signed_areas(g, pts)) != np.sign(a0)))
        live = amp > 0
        # shape fidelity: realised must be a constant multiple of requested, wherever requested > 0
        ratio = moved[live] / amp[live]
        shape_ok = float(ratio.max() / max(ratio.min(), 1e-300)) < 1.0 + 1e-9
        # a zero request must produce zero motion
        zero_ok = bool(np.all(moved[~live] < 1e-12)) if np.any(~live) else True
        ok = shape_ok and zero_ok and not inv
        bad += not ok
        rows.append((name, int(live.sum()), float(moved.max()), shape_ok, zero_ok, inv))

    for name, nlive, mx, sh, zo, inv in rows:
        print('[2f] %-16s %3d live nodes, max move %.4f | shape exact %-5s | zeros stay %-5s | '
              'inverted %-5s' % (name, nlive, mx, sh, zo, inv))
    print('[2f] arbitrary per-node amplitude profile honoured exactly, no inversion  %s'
          % ('OK' if bad == 0 else 'FAIL'))
    return 1 if bad else 0


def test_it_actually_moves():
    """[3] The bound must protect by BOUNDING, not by doing nothing."""
    tag, geo = _meshes()[1]
    lbar = float(F.bond_lengths(geo).mean())
    rows = []
    for frac in (0.25, 0.99):
        pts, meta = F.displace_safe(geo, np.random.default_rng(0), frac=frac, structure='white')
        d = np.linalg.norm(pts - np.asarray(geo['pts'], float), axis=1)
        rows.append((frac, float(d.mean() / lbar), float(d.max() / lbar)))
    ok = rows[0][1] > 0.01 and rows[1][1] > rows[0][1]
    print('[3] displacement is real and scales: frac 0.25 -> mean %.3f lbar (max %.3f); '
          'frac 0.99 -> mean %.3f lbar (max %.3f)  %s'
          % (rows[0][1], rows[0][2], rows[1][1], rows[1][2], 'OK' if ok else 'FAIL'))
    return 0 if ok else 1


def test_beats_global_amp():
    """[4] Show, do not assert: the global-mean `displace` breaks meshes that `displace_safe` does not.

    Same mesh, same seed. `displace` also wraps, which is itself wrong for frozen connectivity, so
    this compares its positions WITHOUT the wrap to isolate the amplitude effect rather than
    scoring it against a defect it is not being blamed for here."""
    tag, geo = _meshes()[2]                                   # the awkward one
    a0 = _signed_areas(geo)
    lbar = float(F.bond_lengths(geo).mean())
    n_old = n_new = 0
    for amp in (0.1, 0.2, 0.3):
        for seed in range(6):
            rng = np.random.default_rng(seed)
            a = rng.uniform(0, 2 * np.pi, len(geo['pts']))
            pts_old = np.asarray(geo['pts'], float) + amp * lbar * np.stack(
                [np.cos(a), np.sin(a)], 1)                    # `displace`'s rule, without the wrap
            n_old += int(np.any(np.sign(_signed_areas(geo, pts_old)) != np.sign(a0)))
            pts_new, _ = F.displace_safe(geo, np.random.default_rng(seed), frac=0.99,
                                         structure='white')
            n_new += int(np.any(np.sign(_signed_areas(geo, pts_new)) != np.sign(a0)))
    ok = n_old > 0 and n_new == 0
    print('[4] on the eta=0.35 mesh, 18 draws each: global-mean amplitude inverted %d, '
          'displace_safe inverted %d  %s' % (n_old, n_new, 'OK' if ok else 'FAIL'))
    return 0 if ok else 1


def test_no_wrap():
    """[5] `displace_safe` must not wrap: a node near the seam must stay near the seam."""
    tag, geo = _meshes()[0]
    pts, _ = F.displace_safe(geo, np.random.default_rng(0), frac=0.99, structure='white')
    Lx, Ly = F.box_of(geo)
    moved = np.linalg.norm(pts - np.asarray(geo['pts'], float), axis=1)
    ok = moved.max() < 0.25 * min(Lx, Ly)     # a wrap would show as a ~box-sized jump
    print('[5] no wrapping: largest node move %.4f vs box %.2f x %.2f  %s'
          % (moved.max(), Lx, Ly, 'OK' if ok else 'FAIL'))
    return 0 if ok else 1


def main():
    bad = 0
    print('displace_safe tests')
    for fn in (test_altitude, test_no_inversion, test_scale_is_exact_not_conservative,
               test_reaches_classical_eta, test_local_scale_is_size_independent,
               test_every_vertex_uses_its_own_margin_equally,
               test_arbitrary_amplitude_profile,
               test_it_actually_moves,
               test_beats_global_amp, test_no_wrap):
        bad += fn()
    print('\n%s' % ('ALL PASSED' if bad == 0 else '%d TEST(S) FAILED' % bad))
    return 1 if bad else 0


if __name__ == '__main__':
    raise SystemExit(main())
