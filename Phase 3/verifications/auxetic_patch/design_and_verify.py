"""
Case auxetic_patch (MIXED / LOCAL). For each topology x size, design k so the network is normal
(nu ~ +0.3) everywhere EXCEPT a central circular patch that is auxetic (nu ~ -0.3). Then simulate
the designed network and check, region by region, that the patch is auxetic while the surroundings
are not — and draw a spatial local-nu map.

Note on verification strength: the regional nu here is the region-averaged per-triangle physical
tensor; the sim uses the same definition, so this validates the design->realise->simulate loop and
the SPATIAL PATTERN (it is not a fully independent measurement of a sub-region's modulus).
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _common as C

CASE = 'auxetic_patch'
NU_OUT, NU_PATCH = 0.30, -0.30
REG = 0.002


def regions(prob):
    cen = prob.centroids
    span = cen[:, 0].max() - cen[:, 0].min()
    patch = prob.region_in_circle(cen.mean(0), radius=0.20 * span)
    outside = np.setdiff1d(np.arange(prob.n_tri), patch)
    return patch, outside


def main():
    rows = []                       # (topo, N, out_nu, patch_nu, out_solver, patch_solver)
    reps = []                       # detail-figure designs
    rep = None                      # representative for the spatial map
    for topo, label, _ in C.TOPOS:
        for N in C.SIZES:
            prob, geo = C.make_case(topo, N)
            patch, outside = regions(prob)
            objs = [C.Objective('nu', NU_OUT, region=outside, weight=1.0),
                    C.Objective('nu', NU_PATCH, region=patch, weight=1.5)]
            res = C.optimize(prob, objs, mode='k', n_iter=120, reg=REG, verbose=False)
            oo = C.solver_region_nuE(prob, res['k'], outside)[0]     # solver prediction
            pp = C.solver_region_nuE(prob, res['k'], patch)[0]
            C.apply_k_to_geo(geo, res['k'])
            C6_per = C.sim_per_triangle_C6(geo)
            out_nu = C.c6_nuE(C.region_phys_C6(geo, C6_per, outside))[0]
            pat_nu = C.c6_nuE(C.region_phys_C6(geo, C6_per, patch))[0]
            rows.append((topo, N, out_nu, pat_nu, oo, pp))
            print(f"  {topo:12s} N={N} | outside nu solver/sim={oo:+.3f}/{out_nu:+.3f}  "
                  f"patch nu solver/sim={pp:+.3f}/{pat_nu:+.3f}", flush=True)
            if topo == 'disorder_lo' and N == C.SIZES[-1]:
                rep = dict(geo=geo, prob=prob, C6_per=C6_per, patch=patch)
            if N == C.SIZES[-1] and topo in ('regular', 'aniso_shr', 'disorder_lo'):
                reps.append((label, geo, res['k'], C6_per))

    d = C.savedir(CASE)
    np.savez(os.path.join(d, 'results.npz'),
             topo=[r[0] for r in rows], N=[r[1] for r in rows],
             out_nu=[r[2] for r in rows], patch_nu=[r[3] for r in rows],
             out_solver=[r[4] for r in rows], patch_solver=[r[5] for r in rows],
             target_out=NU_OUT, target_patch=NU_PATCH)

    fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.8))
    # panel 0: outside nu vs patch nu, all topo x size; targets marked
    ax[0].axvline(NU_OUT, color='gray', ls='--', lw=1); ax[0].axhline(NU_PATCH, color='gray', ls='--', lw=1)
    ax[0].scatter([NU_OUT], [NU_PATCH], marker='*', s=260, c='gold', edgecolor='k',
                  zorder=5, label='target')
    for topo, label, _ in C.TOPOS:
        for N in C.SIZES:
            m = [r for r in rows if r[0] == topo and r[1] == N]
            if m:
                o, p, so, sp = m[0][2], m[0][3], m[0][4], m[0][5]
                ax[0].scatter([o], [p], c=C.TOPO_COLORS[topo], marker=C.SIZE_MARKERS[N], s=70,
                              edgecolor='k', linewidth=0.4, zorder=3,
                              label=label if N == C.SIZES[0] else None)
                ax[0].scatter([so], [sp], c='k', marker='+', s=30, alpha=0.6, zorder=2)
    ax[0].axhline(0, color='gray', lw=0.4, ls=':')
    ax[0].set_xlabel('outside ν  (colored=sim, + =solver)'); ax[0].set_ylabel('patch ν')
    ax[0].set_title(f'Mixed design: outside → {NU_OUT:+.2f}, patch → {NU_PATCH:+.2f}\n'
                    f'(each point = one topology×size, simulated)')
    ax[0].legend(fontsize=7); ax[0].grid(alpha=0.3)

    # panel 1: spatial local-nu map for the representative case
    if rep is not None:
        geo, prob, C6_per, patch = rep['geo'], rep['prob'], rep['C6_per'], rep['patch']
        cen = prob.centroids
        ncell = 16
        xg = np.linspace(cen[:, 0].min(), cen[:, 0].max(), ncell + 1)
        yg = np.linspace(cen[:, 1].min(), cen[:, 1].max(), ncell + 1)
        tri_nu = np.full(prob.n_tri, np.nan)
        for i in range(ncell):
            for j in range(ncell):
                sel = np.where((cen[:, 0] >= xg[i]) & (cen[:, 0] < xg[i + 1]) &
                               (cen[:, 1] >= yg[j]) & (cen[:, 1] < yg[j + 1]))[0]
                if len(sel) >= 3:
                    tri_nu[sel] = C.c6_nuE(C.region_phys_C6(geo, C6_per, sel))[0]
        good = ~np.isnan(tri_nu)
        vmax = np.nanpercentile(np.abs(tri_nu[good]), 95)
        sc = ax[1].scatter(cen[good, 0], cen[good, 1], c=tri_nu[good], cmap='RdBu_r',
                           vmin=-vmax, vmax=vmax, s=14)
        pc = prob.centroids[patch].mean(0)
        r = 0.20 * (cen[:, 0].max() - cen[:, 0].min())
        ax[1].add_patch(plt.Circle(pc, r, facecolor='none', edgecolor='lime', lw=2))
        ax[1].set_aspect('equal')
        ax[1].set_title('Local ν map — designed disorder_lo (simulated)\n(blue = auxetic patch, red = normal matrix)')
        plt.colorbar(sc, ax=ax[1], fraction=0.046, label='local ν')
    fig.suptitle(f'{CASE}: local/mixed design verified against simulation — '
                 f'{len(C.TOPOS)} topologies × {len(C.SIZES)} sizes', fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    p = os.path.join(d, f'{CASE}.png')
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print('saved', p)

    if reps:
        C.design_detail_figure(os.path.join(d, f'{CASE}_detail.png'), reps,
                               f'{CASE}: designed networks (auxetic patch in a positive matrix, '
                               f'largest size) — rigidity k, local ν, local E')
        print('saved', os.path.join(d, f'{CASE}_detail.png'))


if __name__ == '__main__':
    main()
