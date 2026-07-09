"""
LARGE (~16k-triangle) inverse designs — the most promising result from each design TYPE, on two
topologies (regular + disordered). Types:
  ISO    : isotropise to a flat auxetic nu0 = -0.5 (the ordered isotropic-auxetic win).
  ANISO  : realise a REALIZABLE anisotropic response (target = the aniso_str tensor).
  INDEP  : independent E/nu — E directional (2-fold) with nu held flat (the working direction).
  PATCH  : decoupled E-region (stiff) + nu-region (auxetic) at large scale.
Each design is saved as a network (.npz) and rendered as a local nu/E map. HALF=42 -> ~16000 triangles
(adjoint path). n_iter kept modest for tractability.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import _common as C

HALF, NITER, REG = 42, 55, 1e-4
TH = C.ANG
TOPOS = ['regular', 'disorder_hi']
E0, EAMP = 1.0, 0.40
csv = []


def objs_for(kind, prob, geo):
    """Return (objectives, regions_to_mark, focus_field)."""
    if kind == 'iso':
        return [C.Objective('nu', -0.5)], None, 'nu'
    if kind == 'aniso':
        return [C.Objective('tensor', C.reference_C6('aniso'))], None, 'nu'
    if kind == 'indep':
        return [C.Objective('nu_theta', 0.20, weight=10.0),
                C.Objective('E_theta', E0 * (1 + EAMP * np.cos(2 * TH)), weight=1.0)], None, 'E'
    if kind == 'patch':
        Lx, Ly = C.box(geo)
        RE = {'kind': 'circle', 'center': (0.30 * Lx, 0.5 * Ly), 'radius': 0.14 * Lx, 'color': 'cyan'}
        RN = {'kind': 'circle', 'center': (0.70 * Lx, 0.5 * Ly), 'radius': 0.14 * Lx, 'color': 'lime'}
        R_E, _ = C.region_shape(prob, RE); R_N, _ = C.region_shape(prob, RN)
        oE = np.setdiff1d(np.arange(prob.n_tri), R_E); oN = np.setdiff1d(np.arange(prob.n_tri), R_N)
        objs = [C.Objective('nu', 0.20, region=oN, weight=3.0), C.Objective('nu', -0.30, region=R_N, weight=4.0),
                C.Objective('E', 1.0, region=oE, weight=1.0), C.Objective('E', 1.8, region=R_E, weight=1.5)]
        return objs, [RE, RN], 'both'
    raise ValueError(kind)


def run(kind, topo, nd):
    prob, geo = C.make_case(topo, HALF)
    objs, region, focus = objs_for(kind, prob, geo)
    r = C.optimize(prob, objs, mode='k', n_iter=NITER, reg=REG, verbose=False)
    C.apply_k_to_geo(geo, r['k']); C6 = C.sim_per_triangle_C6(geo)
    nu_g, E_g = C.c6_nuE(C.region_phys_C6(geo, C6, None))
    csv.append((kind, topo, prob.n_tri, f'{nu_g:+.3f}', f'{E_g:.3f}'))
    C.save_network(os.path.join(nd, f'{kind}__{topo}.npz'), geo, r['k'], C6, kind=kind, topo=topo,
                   n_tri=int(prob.n_tri), region=region, global_nu=float(nu_g), global_E=float(E_g))
    print(f"  {kind:6s} {topo:12s} tri={prob.n_tri} | global nu={nu_g:+.3f} E={E_g:.3f}", flush=True)
    return region


def maps_figure(kind, nd):
    entries = []
    for topo in TOPOS:
        geo, k, C6, meta = C.load_network(os.path.join(nd, f'{kind}__{topo}.npz'))
        t0 = f"{topo}  (global ν={meta['global_nu']:+.3f})"
        t1 = f"global E={meta['global_E']:.3f}"
        entries.append((t0, t1, geo, C6, meta.get('region')))
    C.nuE_row_grid(os.path.join(HERE, f'large16k_{kind}.png'),
                   f'LARGE 16k-triangle design — {kind} (regular vs disordered)', entries,
                   figsize=(3.2, 6.4))
    print('saved map', kind)


def main():
    nd = os.path.join(HERE, 'networks'); os.makedirs(nd, exist_ok=True)
    for kind in ['iso', 'aniso', 'indep', 'patch']:
        for topo in TOPOS:
            run(kind, topo, nd)
        maps_figure(kind, nd)
    C.write_csv(os.path.join(HERE, 'large16k.csv'),
                ['kind', 'topology', 'n_tri', 'global_nu', 'global_E'], csv)
    print('done')


if __name__ == '__main__':
    main()
