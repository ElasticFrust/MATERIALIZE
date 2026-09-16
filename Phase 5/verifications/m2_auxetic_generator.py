r"""A1/d — generate AUXETIC networks from arbitrary base topologies, by eta / VD / eta+alpha.

WHAT THIS IS FOR. The response audit found the auxetic corner thinly sampled in `dataset_v2_s0`
(nu < 0 is 19.3 % overall but concentrated in two generator families, and the isotropic-auxetic cell
was empty). The measured conclusion of A1/d is that this is a SAMPLING gap, not a reach limit -- the
existing machinery gets there once the axes are crossed. This script does the crossing.

THE THREE MECHANISMS, in the user's definitions (`COVERAGE_SWEEP.md`):

    eta         the network is really distorted; k stays uniform. Geometry alone.
    VD          virtual displacements on top of the real network: an eta variation decides the
                spring rigidities, but the initial configuration is NOT distorted.
    eta+alpha   the network is really distorted, and the distortion is enhanced through the alpha
                stiffness mechanism, k = 1 + tanh(alpha (l - l0)) off the DISTORTED lengths.

AMPLITUDE: POSITION-DEPENDENT MAX, ONE UNIFORM KNOB (the user's spec). `displace_safe(scale='local')`
gives every vertex its own maximum -- the exact distance to the first inversion in its own
neighbourhood, by local fixed point, nothing global -- and `frac` is the single scalar saying what
portion of that maximum to take, uniform everywhere. So the amplitude adapts to the mesh while the
caller turns one dial. Worth 2.3x median amplitude over a single global scale on a disordered mesh,
and up to 12x on individual vertices.

RANGES, not operating points. Earlier I quoted eta = 0.35 / alpha = 5 as a "recipe"; that is the
point that happens to fill one cell, not a sampling prescription. `frac` and `alpha` are swept, and
alpha is capped at |alpha| <= 10 (the user's range; the alpha = 30/60 probes elsewhere in this
session are outside it and should not be read as guidance).

EVERY sample is checked for triangle inversion and the count is reported. It must be zero: at
eta >~ 0.44 the classical fixed-amplitude disorder folds meshes (10/10 seeds at eta = 0.48), and on a
folded mesh solver-vs-sim agreement degrades from 1e-13 to 1.6e-2 -- fine for a qualitative claim,
NOT fine for labels whose must-tier is 0.02. `displace_safe` cannot fold a mesh by construction,
which is the point of using it rather than a fixed eta.

Run:
    python "Phase 5/verifications/m2_auxetic_generator.py" --out_dir <dir>
"""
import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import torch

warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'Phase 3', 'verifications'))
import _common as C                                                       # noqa: E402,F401
from inverse_design import DesignProblem, ANG, c6_to_nuE_theta            # noqa: E402
import mesh_build as MB                                                   # noqa: E402
sys.path.insert(0, os.path.join(REPO, 'Phase 5'))
sys.path.insert(0, os.path.join(REPO, 'Phase 5', 'm2'))
import seeds as S                                                         # noqa: E402
import fields as F                                                        # noqa: E402
sys.path.insert(0, HERE)
from m2_coverage_sweep import geo_moved                                   # noqa: E402

torch.set_default_dtype(torch.float64)
RESULTS = os.path.join(REPO, 'Phase 5', 'results', 'auxetic_generation')


def base_networks(n_random, reps, n_nodes, seed):
    """Base topologies to deform: Bravais crystals across the planar types, plus random patches."""
    out = []
    for phi, psi in ((1.0, 1.0), (0.0, 2 / np.sqrt(3)), (0.0, 1.0), (1.2, 0.8), (0.6, 1.3)):
        for diag in S.BRAVAIS_DIAGONALS:
            try:
                r = S.bravais_lattice(phi, psi, reps=reps, diagonal=diag, eta=0.0, seed=seed)
                out.append(('bravais_p%.1f_%.1f_%s' % (phi, psi, diag), r['geo']))
            except Exception:                                             # noqa: BLE001
                pass
    for i in range(n_random):
        for proc in ('poisson_disk', 'blue_noise'):
            try:
                r = S.random_patch(n_nodes=n_nodes, process=proc, seed=seed + 100 + i)
                out.append(('random_%s_%d' % (proc, i), r['geo']))
            except Exception:                                             # noqa: BLE001
                pass
    return out


def response(geo, k):
    """One exact solver forward -> nu(theta) mean, anisotropy, E, max|W|."""
    g = dict(geo)
    k = np.asarray(k, float)
    k = k / max(k.mean(), 1e-300)
    g['bond_k'] = k
    g['tri_k'] = k[np.asarray(geo['tri_bond'], np.int64)]
    prob = DesignProblem.from_geo(g)
    with torch.no_grad():
        out = prob.forward(torch.as_tensor(k), physical_units=True)
        nt, Et = c6_to_nuE_theta(prob.region_tensor(out['per_triangle'], None), ANG)
        wm = float(np.abs(np.asarray(out['W'], float)).max())
        c6 = np.asarray(prob.region_tensor(out['per_triangle'], None), float)
    nt, Et = nt.numpy(), Et.numpy()
    if not (np.isfinite(nt).all() and np.isfinite(Et).all() and Et.min() > 0):
        return None
    return dict(nu=float(nt.mean()), nu_min=float(nt.min()), nu_max=float(nt.max()),
                aniso=float(Et.max() / max(Et.min(), 1e-300)), E=float(Et.mean()),
                w_max=wm, C6=c6.tolist())


def alpha_k(lengths, l_ref, alpha):
    """k = 1 + tanh(alpha (l - l_ref)), normalised to mean 1. `l_ref` is the bond's own reference.

    `mesh_build.set_VD` hardcodes `l_ref = 1`, which is wrong on any lattice whose natural spacing is
    not 1 -- the Bravais cells with phi, psi != 1 among them. Passing the reference explicitly is the
    generalisation that makes the mechanism well-defined on an arbitrary base network."""
    k = 1.0 + np.tanh(alpha * (np.asarray(lengths, float) - np.asarray(l_ref, float)))
    return k / max(k.mean(), 1e-300)


def bond_len(geo, pts=None):
    if pts is None:
        R = np.asarray(geo['bond_R'], float)
    else:
        bu = np.asarray(geo['bond_u'], np.int64); bv = np.asarray(geo['bond_v'], np.int64)
        p0 = np.asarray(geo['pts'], float)
        shift = np.asarray(geo['bond_R'], float) - (p0[bv] - p0[bu])
        R = np.asarray(pts, float)[bv] - np.asarray(pts, float)[bu] + shift
    return np.hypot(R[:, 0], R[:, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fracs', default='0.2,0.4,0.6,0.8',
                    help='portion of each vertex own local maximum -- ONE uniform knob')
    ap.add_argument('--alphas', default='-10,-5,-2,2,5,10', help='|alpha| <= 10 (user range)')
    ap.add_argument('--mechanisms', default='eta,vd,eta_alpha')
    ap.add_argument('--reps', type=int, default=8)
    ap.add_argument('--n_nodes', type=int, default=120)
    ap.add_argument('--n_random', type=int, default=2)
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out_dir', default=RESULTS)
    ap.add_argument('--tag', default='')
    a = ap.parse_args()

    fracs = [float(x) for x in a.fracs.split(',')]
    alphas = [float(x) for x in a.alphas.split(',')]
    mechs = a.mechanisms.split(',')
    assert max(abs(x) for x in alphas) <= 10.0 + 1e-9, 'alpha is capped at |alpha| <= 10'
    bases = base_networks(a.n_random, a.reps, a.n_nodes, a.seed)
    print('%d base networks x %d mechanisms x %d fracs x %d alphas x %d seeds'
          % (len(bases), len(mechs), len(fracs), len(alphas), a.seeds))
    print('amplitude: per-vertex local maximum (scale="local"), frac = uniform portion of it')

    rows, t0, inverted, failed = [], time.time(), 0, 0
    for bname, geo0 in bases:
        l0 = bond_len(geo0)                       # each bond's own reference length
        for mech in mechs:
            for frac in fracs:
                for sd in range(a.seeds):
                    rng = np.random.default_rng(1000 * sd + a.seed)
                    pts, meta = F.displace_safe(geo0, rng, frac=frac, structure='white',
                                                scale='local')
                    al_list = [0.0] if mech == 'eta' else alphas
                    for al in al_list:
                        if mech == 'eta':
                            geo, k = geo_moved(geo0, pts)[0], np.ones(len(l0))
                        elif mech == 'vd':
                            # geometry UNTOUCHED; k decided by the virtual displacement
                            geo, k = geo0, alpha_k(bond_len(geo0, pts), l0, al)
                        elif mech == 'eta_alpha':
                            geo = geo_moved(geo0, pts)[0]
                            k = alpha_k(bond_len(geo), l0, al)
                        else:
                            raise ValueError('unknown mechanism %r' % mech)
                        sa = MB.signed_areas(geo)
                        if (sa < 0).any():
                            inverted += 1
                            continue
                        try:
                            r = response(geo, k)
                        except Exception:                                 # noqa: BLE001
                            r = None
                        if r is None:
                            failed += 1
                            continue
                        r.update(base=bname, mech=mech, frac=frac, alpha=al, seed=sd,
                                 n_tri=len(geo['tri_bond']),
                                 eta_equiv=float(meta.get('geom_eta_equiv', 0.0)))
                        r.pop('C6')
                        rows.append(r)
        print('  %-30s %6d samples  (%.0fs)' % (bname, len(rows), time.time() - t0))

    nu = np.array([r['nu'] for r in rows]); an = np.array([r['aniso'] for r in rows])
    print('\n%d networks generated; %d rejected as inverted (MUST be 0); %d solver failures'
          % (len(rows), inverted, failed))
    print('AUXETIC yield: %d of %d (%.1f %%) have nu < 0;  %d below -0.2;  %d below -0.5'
          % ((nu < 0).sum(), len(nu), 100 * (nu < 0).mean(), (nu < -0.2).sum(), (nu < -0.5).sum()))
    print('   isotropic auxetic (nu < -0.2 AND aniso < 1.2): %d' % ((nu < -0.2) & (an < 1.2)).sum())

    print('\nauxetic yield BY MECHANISM:')
    print('   %-11s %7s %9s %11s %11s' % ('mech', 'n', 'nu<0', 'nu median', 'nu min'))
    for m in mechs:
        s = np.array([r['mech'] == m for r in rows])
        if not s.any():
            continue
        print('   %-11s %7d %8.1f%% %11.4f %11.4f'
              % (m, s.sum(), 100 * (nu[s] < 0).mean(), np.median(nu[s]), nu[s].min()))

    print('\nauxetic yield BY BASE TOPOLOGY (the point: not just the triangular lattice):')
    print('   %-30s %7s %9s %11s' % ('base', 'n', 'nu<0', 'nu min'))
    for b in sorted({r['base'] for r in rows}):
        s = np.array([r['base'] == b for r in rows])
        print('   %-30s %7d %8.1f%% %11.4f' % (b, s.sum(), 100 * (nu[s] < 0).mean(), nu[s].min()))

    os.makedirs(a.out_dir, exist_ok=True)
    dst = os.path.join(a.out_dir, 'auxetic_generation%s.json' % (('_' + a.tag) if a.tag else ''))
    with open(dst, 'w') as f:
        json.dump(dict(config=vars(a), n=len(rows), inverted=inverted, failed=failed, rows=rows), f)
    print('\n-> %s' % dst)
    return 1 if inverted else 0


if __name__ == '__main__':
    raise SystemExit(main())
