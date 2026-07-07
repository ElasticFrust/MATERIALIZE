"""
Case graded_nu (SPATIAL GRADIENT). For each topology x size, design k so the Poisson ratio varies
smoothly across x — a stack of vertical-strip objectives from nu ~ +0.3 (left) to nu ~ -0.3
(right). Then simulate the designed network and check the realised nu(x) profile per strip.

Same verification caveat as auxetic_patch: strip nu is the region-averaged per-triangle physical
tensor (validates the loop + the spatial gradient, not a fully independent sub-region measurement).
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _common as C

CASE = 'graded_nu'
NSTRIP = 6
NU_LEFT, NU_RIGHT = 0.30, -0.30
REG = 0.002


def strips(prob):
    x = prob.centroids[:, 0]
    xg = np.linspace(x.min(), x.max() + 1e-9, NSTRIP + 1)
    idx = [np.where((x >= xg[i]) & (x < xg[i + 1]))[0] for i in range(NSTRIP)]
    xc = 0.5 * (xg[:-1] + xg[1:])
    tgt = np.linspace(NU_LEFT, NU_RIGHT, NSTRIP)
    return idx, xc, tgt


def main():
    results = {}                    # (topo,N) -> (xc, tgt, sim_nu, solver_nu) per strip
    reps = []
    rep = None
    for topo, label, _ in C.TOPOS:
        for N in C.SIZES:
            prob, geo = C.make_case(topo, N)
            idx, xc, tgt = strips(prob)
            objs = [C.Objective('nu', float(tgt[i]), region=idx[i], weight=1.0)
                    for i in range(NSTRIP) if len(idx[i]) >= 3]
            res = C.optimize(prob, objs, mode='k', n_iter=150, reg=REG, verbose=False)
            ssolv = np.array([C.solver_region_nuE(prob, res['k'], idx[i])[0]
                              if len(idx[i]) >= 3 else np.nan for i in range(NSTRIP)])
            C.apply_k_to_geo(geo, res['k'])
            C6_per = C.sim_per_triangle_C6(geo)
            snu = np.array([C.c6_nuE(C.region_phys_C6(geo, C6_per, idx[i]))[0]
                            if len(idx[i]) >= 3 else np.nan for i in range(NSTRIP)])
            results[(topo, N)] = (xc, tgt, snu, ssolv)
            print(f"  {topo:12s} N={N} | sim nu(x) = "
                  + " ".join(f"{v:+.2f}" for v in snu), flush=True)
            if topo == 'disorder_lo' and N == C.SIZES[-1]:
                rep = dict(prob=prob, geo=geo, C6_per=C6_per, idx=idx, xc=xc, tgt=tgt)
            if N == C.SIZES[-1] and topo in ('regular', 'aniso_shr', 'disorder_lo'):
                reps.append((label, geo, res['k'], C6_per))

    d = C.savedir(CASE)
    np.savez(os.path.join(d, 'results.npz'),
             keys=[f"{t}_N{n}" for (t, n) in results],
             xc=[v[0] for v in results.values()], tgt=[v[1] for v in results.values()],
             sim_nu=[v[2] for v in results.values()], solver_nu=[v[3] for v in results.values()])

    fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.8))
    # panel 0: nu(x) target vs simulated (solid) and solver (thin dashed), all topo x size
    xc0, tgt0 = next(iter(results.values()))[:2]
    ax[0].plot(np.linspace(0, 1, NSTRIP), tgt0, 'k--', lw=2.5, label='prescribed ν(x)')
    lbl = {t: l for t, l, _ in C.TOPOS}
    for (topo, N), (xc, tgt, snu, ssolv) in results.items():
        xn = (xc - xc.min()) / (xc.max() - xc.min())
        ax[0].plot(xn, snu, '-', color=C.TOPO_COLORS[topo], marker=C.SIZE_MARKERS[N], ms=5,
                   lw=1.2, alpha=0.85, label=lbl[topo] if N == C.SIZES[0] else None)
        ax[0].plot(xn, ssolv, ':', color=C.TOPO_COLORS[topo], lw=0.8, alpha=0.6)
    ax[0].axhline(0, color='gray', lw=0.5, ls=':')
    ax[0].set_xlabel('normalised x'); ax[0].set_ylabel('ν (solid=sim, dotted=solver)')
    ax[0].set_title('Prescribed vs simulated ν(x) gradient\n(all topologies × sizes; solver dotted)')
    ax[0].legend(fontsize=7); ax[0].grid(alpha=0.3)

    # panel 1: spatial nu map (representative) showing the left-to-right gradient
    if rep is not None:
        prob, geo, C6_per = rep['prob'], rep['geo'], rep['C6_per']
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
        ax[1].set_aspect('equal')
        ax[1].set_title('Local ν map — designed disorder_lo (simulated)\n(left positive → right auxetic)')
        plt.colorbar(sc, ax=ax[1], fraction=0.046, label='local ν')
    fig.suptitle(f'{CASE}: graded ν(x) design verified against simulation — '
                 f'{len(C.TOPOS)} topologies × {len(C.SIZES)} sizes', fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    p = os.path.join(d, f'{CASE}.png')
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print('saved', p)

    if reps:
        C.design_detail_figure(os.path.join(d, f'{CASE}_detail.png'), reps,
                               f'{CASE}: designed networks (ν(x) gradient, largest size) — '
                               f'rigidity k, local ν, local E')
        print('saved', os.path.join(d, f'{CASE}_detail.png'))


if __name__ == '__main__':
    main()
