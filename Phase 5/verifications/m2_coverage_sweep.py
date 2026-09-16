r"""A1/d — how much of RESPONSE SPACE do deformation + stiffness contrast actually reach?

THE QUESTION, and whose it is. The response-space audit (`m2_response_audit.py`) found five empty
(nu x anisotropy) cells in `dataset_v2_s0`, among them **isotropic auxetic** — nu in [-0.5,-0.2] with
anisotropy < 1.2. I presented that as a coverage hole. The user's objection: eta plus VD / alpha
contrast should reach a large range of isotropic negative Poisson ratios, and taking OTHER crystal
topologies and deforming them with and without stiffness contrast should fill most of the rest. If
that is right, the "hole" is a SAMPLING CHOICE of the existing builder, not a reach limit — and the
fix is to sample differently, not to invent a new generator.

This settles it by measurement rather than argument. It sweeps

    topology   Bravais (phi, psi) crystals, both diagonals, plus random patches
    geometry   `fields.displace_safe` frac x {white, correlated}
    stiffness  VD contrast `k = 1 + tanh(a(|R|-1))` over a, plus `fields.k_field` structures

and maps which (nu x anisotropy) cells fill.

WHY `displace_safe` AND NOT `eta`. A global eta is an ABSOLUTE displacement, so the same value is
harmless on one mesh and collapses a triangle on another, and the classical eta < 0.5 bound is a
regular-lattice measurement that does not transfer (§3.1f). `displace_safe` moves each node by a
fraction of the exactly-computed distance to the FIRST INVERSION, shaped by that node's own margin,
so no triangle can break, the amplitude does not degrade with cell size, and `geom_eta_equiv` still
reports the familiar eta scale. Every sample is nevertheless checked for inversion — a guarantee
that is not also verified is just a claim.

PRIOR ART, checked (`CLAUDE.md` §3 requires saying so): `verification_tools/recheck_sweep_nu_E_eta.py`
covers exactly "nu(eta) and E(eta), disordered networks and VD rigidity contrasts". It calls the
TOMBSTONED `Ceff_nuE`, so it sits in the dead legacy island and its numbers are stale (area-weighted
nu, internal-unit E). Re-measuring is warranted, not duplicated.

Run:
    python "Phase 5/verifications/m2_coverage_sweep.py"
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'Phase 3', 'verifications'))
import _common as C                                                       # noqa: E402,F401
from inverse_design import (DesignProblem, ANG, c6_to_nuE_theta)          # noqa: E402
import mesh_build as MB                                                   # noqa: E402
sys.path.insert(0, os.path.join(REPO, 'Phase 5'))
sys.path.insert(0, os.path.join(REPO, 'Phase 5', 'm2'))
import seeds as S                                                         # noqa: E402
import fields as F                                                        # noqa: E402

torch.set_default_dtype(torch.float64)
RESULTS = os.path.join(REPO, 'Phase 5', 'results', 'coverage_sweep')

NU_EDGES = [-np.inf, -1.0, -0.5, -0.2, 0.0, 0.2, 1.0 / 3.0, 0.5, 1.0, np.inf]
NU_LABEL = ['<-1', '-1..-.5', '-.5..-.2', '-.2..0', '0..0.2', '0.2..1/3', '1/3..0.5', '0.5..1', '>1']
AN_EDGES = [1.0, 1.2, 2.0, 5.0, 20.0, np.inf]
AN_LABEL = ['iso<1.2', '1.2-2', '2-5', '5-20', '>20']


def geo_moved(geo, pts):
    """The same mesh at new node positions: only the coordinate-carried fields change.

    `bond_R` is rebuilt through the bonds' constant periodic image offset (never re-derived by
    minimum image, which is what flips an offset and reconstructs a triangle a box away), and
    `edge_vecs` through `mesh_build.edge_vec_orientation` so it keeps the TRIANGLE's corner order —
    `bond_R[tri_bond]` would be sign-wrong on about half the rows and would silently corrupt the
    curvature constraint."""
    g = dict(geo)
    bu = np.asarray(geo['bond_u'], np.int64); bv = np.asarray(geo['bond_v'], np.int64)
    p0 = np.asarray(geo['pts'], float)
    shift = np.asarray(geo['bond_R'], float) - (p0[bv] - p0[bu])
    pts = np.asarray(pts, float)
    bR = pts[bv] - pts[bu] + shift
    ei, sg = MB.edge_vec_orientation(geo['tri_bond'], geo['simplices'], bu, bv)
    sg = np.where(sg == 0.0, 1.0, sg)
    ev = bR[ei] * sg[..., None]
    g['pts'] = pts
    g['bond_R'] = bR
    g['edge_vecs'] = ev
    g['actual_len2'] = (ev ** 2).sum(-1)
    g['areas'] = 0.5 * np.abs(ev[:, 0, 0] * ev[:, 1, 1] - ev[:, 0, 1] * ev[:, 1, 0])
    return g, float((0.5 * (ev[:, 0, 0] * ev[:, 1, 1] - ev[:, 0, 1] * ev[:, 1, 0])).min())


def response(geo, k):
    """nu(theta), E(theta) -> (mean nu, anisotropy, mean E, max|W|). One exact solver forward."""
    g = dict(geo)
    g['bond_k'] = np.asarray(k, float)
    g['tri_k'] = np.asarray(k, float)[np.asarray(geo['tri_bond'], np.int64)]
    prob = DesignProblem.from_geo(g)
    with torch.no_grad():
        out = prob.forward(torch.as_tensor(np.asarray(k, float)), physical_units=True)
        c6 = prob.region_tensor(out['per_triangle'], None)
        nt, Et = c6_to_nuE_theta(c6, ANG)
        wm = float(np.abs(np.asarray(out['W'], float)).max())
    nt, Et = nt.numpy(), Et.numpy()
    if not (np.isfinite(nt).all() and np.isfinite(Et).all() and Et.min() > 0):
        return None
    return dict(nu=float(nt.mean()), nu_min=float(nt.min()), nu_max=float(nt.max()),
                aniso=float(Et.max() / max(Et.min(), 1e-300)), E=float(Et.mean()), w_max=wm)


def topologies(n_random, seed, phipsi=None, reps=5, n_nodes=80):
    """Bravais crystals across the five planar types, both diagonals, plus random patches.

    `phipsi` restricts the (phi, psi) list -- e.g. to the TRIANGULAR lattice alone, which is the only
    intrinsically ISOTROPIC one. The others are anisotropic by construction, and length-keyed VD
    amplifies that: in a non-equilateral cell bond length correlates with orientation, so VD becomes
    an orientation field and drives anisotropy up rather than leaving it alone."""
    out = []
    for phi, psi in (phipsi or ((1.0, 1.0), (0.0, 2 / np.sqrt(3)), (0.0, 1.0), (1.0, 0.7), (0.6, 1.3))):
        for diag in S.BRAVAIS_DIAGONALS:
            try:
                r = S.bravais_lattice(phi, psi, reps=reps, diagonal=diag, eta=0.0, seed=seed)
                out.append(('bravais_p%.1f_%.1f_%s' % (phi, psi, diag), r['geo']))
            except Exception:                                             # noqa: BLE001
                pass
    for i in range(n_random):
        try:
            r = S.random_patch(n_nodes=n_nodes, process='poisson_disk', seed=seed + 100 + i)
            out.append(('random_%d' % i, r['geo']))
        except Exception:                                                 # noqa: BLE001
            pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fracs', default='0,0.3,0.6,0.9')
    ap.add_argument('--vd', default='-10,-5,-2,0,2,5,10')
    ap.add_argument('--kstruct', default='uniform,correlated,orientation,length')
    ap.add_argument('--n_random', type=int, default=3)
    ap.add_argument('--reps', type=int, default=5,
                    help='Bravais cell size: disorder needs a big enough cell to self-average '
                         'toward isotropy, so a small cell can look anisotropic by fluctuation')
    ap.add_argument('--n_nodes', type=int, default=80)
    ap.add_argument('--triangular_only', action='store_true',
                    help='the only intrinsically ISOTROPIC Bravais lattice')
    ap.add_argument('--seeds', type=int, default=2)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--tag', default='')
    a = ap.parse_args()

    fracs = [float(x) for x in a.fracs.split(',')]
    vds = [float(x) for x in a.vd.split(',')]
    ks = a.kstruct.split(',')
    topos = topologies(a.n_random, a.seed,
                       phipsi=((1.0, 1.0),) if a.triangular_only else None,
                       reps=a.reps, n_nodes=a.n_nodes)
    print('%d topologies x %d fracs x 2 structures x %d VD x %d k-structures x %d seeds'
          % (len(topos), len(fracs), len(vds), len(ks), a.seeds))

    rows, t0, inverted, failed = [], time.time(), 0, 0
    for tname, geo0 in topos:
        for frac in fracs:
            for gstruct in (('white', 'correlated') if frac > 0 else ('none',)):
                for sd in range(a.seeds if frac > 0 else 1):
                    rng = np.random.default_rng(1000 * sd + a.seed)
                    if frac > 0:
                        pts, meta = F.displace_safe(geo0, rng, frac=frac, structure=gstruct)
                    else:
                        pts, meta = np.asarray(geo0['pts'], float), dict(geom_eta_equiv=0.0)
                    geo, min_area = geo_moved(geo0, pts)
                    if min_area <= 0:
                        inverted += 1
                        continue
                    for vd in vds:
                        MB.set_VD(geo, vd)
                        k_vd = np.asarray(geo['bond_k'], float)
                        for kst in ks:
                            if kst == 'uniform':
                                k = k_vd
                            else:
                                kf, _ = F.k_field(geo, np.random.default_rng(7 * sd + 3),
                                                  structure=kst, marginal='lognormal')
                                k = k_vd * np.asarray(kf, float)
                            k = k / max(k.mean(), 1e-300)
                            try:
                                r = response(geo, k)
                            except Exception:                             # noqa: BLE001
                                r = None
                            if r is None:
                                failed += 1
                                continue
                            r.update(topo=tname, frac=frac, gstruct=gstruct, vd=vd, kstruct=kst,
                                     eta_equiv=float(meta.get('geom_eta_equiv', 0.0)),
                                     n_tri=len(geo['tri_bond']))
                            rows.append(r)
        print('  %-28s %5d samples  (%.0fs)' % (tname, len(rows), time.time() - t0))

    print('\n%d samples; %d geometries rejected as inverted (must be 0); %d solver failures'
          % (len(rows), inverted, failed))

    nu = np.array([r['nu'] for r in rows]); an = np.array([r['aniso'] for r in rows])
    nb = np.digitize(nu, NU_EDGES[1:-1]); ab = np.digitize(an, AN_EDGES[1:-1])
    n = len(rows)
    print('\n--- (nu x anisotropy) occupancy, %% of the sweep ---')
    print('    %-10s %s' % ('nu \\ aniso', ' '.join('%8s' % l for l in AN_LABEL)))
    filled = set()
    for i, nl in enumerate(NU_LABEL):
        row = []
        for j in range(len(AN_LABEL)):
            c = int(((nb == i) & (ab == j)).sum())
            row.append('%8.2f' % (100 * c / n) if c else '       .')
            if c:
                filled.add((nl, AN_LABEL[j]))
        print('    %-10s %s' % (nl, ' '.join(row)))

    # the five cells the audit found EMPTY in dataset_v2_s0
    audit_empty = [('<-1', 'iso<1.2'), ('<-1', '1.2-2'), ('-.5..-.2', 'iso<1.2'),
                   ('>1', 'iso<1.2'), ('>1', '1.2-2')]
    print('\n--- the five cells EMPTY in dataset_v2_s0: does this sweep reach them? ---')
    for cell in audit_empty:
        hit = cell in filled
        cnt = int(((nb == NU_LABEL.index(cell[0])) & (ab == AN_LABEL.index(cell[1]))).sum())
        print('    %-10s x %-8s  %s  (%d samples)'
              % (cell[0], cell[1], 'REACHED' if hit else 'still empty', cnt))

    # what reaches the isotropic-auxetic corner, if anything
    key = (nu < -0.2) & (nu > -0.5) & (an < 1.2)
    print('\n--- isotropic auxetic (nu in [-0.5,-0.2], anisotropy < 1.2): %d samples ---' % key.sum())
    if key.sum():
        for r in [rows[i] for i in np.flatnonzero(key)][:10]:
            print('    %-26s frac %.2f %-10s VD %+5.1f %-11s -> nu %+.3f aniso %.2f eta~%.2f'
                  % (r['topo'], r['frac'], r['gstruct'], r['vd'], r['kstruct'], r['nu'],
                     r['aniso'], r['eta_equiv']))

    print('\n--- what each axis alone buys (median nu, and the nu range) ---')
    for name, vals in (('frac', [r['frac'] for r in rows]), ('vd', [r['vd'] for r in rows])):
        v = np.array(vals)
        print('  %s:' % name)
        for x in sorted(set(v.tolist())):
            m = v == x
            print('    %-6s n=%4d  nu median %+.3f  range [%+.3f, %+.3f]  aniso median %.2f'
                  % (x, m.sum(), np.median(nu[m]), nu[m].min(), nu[m].max(), np.median(an[m])))

    os.makedirs(RESULTS, exist_ok=True)
    dst = os.path.join(RESULTS, 'coverage_sweep%s.json' % (('_' + a.tag) if a.tag else ''))
    with open(dst, 'w') as f:
        json.dump(dict(config=vars(a), n=len(rows), inverted=inverted, failed=failed, rows=rows), f)
    print('\n-> %s' % dst)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
