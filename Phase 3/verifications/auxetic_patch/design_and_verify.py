"""
Case auxetic_patch (LOCAL / MIXED). For each topology x size, design k so the network is normal
(nu~+0.3) everywhere EXCEPT a central circular patch that is auxetic (nu~-0.3). Simulate the
designed network and check, region by region, that the patch is auxetic while the surroundings
are not. Outputs:
  - auxetic_patch.csv            : topology, size, outside solver/sim, patch solver/sim
  - auxetic_patch_summary.png    : overlaid (outside nu vs patch nu), all topos; gold star = target
  - auxetic_patch_bytopo.png     : small multiples, per-topology bars (target/solver/sim, out & patch)
  - auxetic_patch_detail.png     : 5-topology grid [rigidity k | local nu | local E], patch marked
  - auxetic_patch_<topology>.png : per-topology detail

Regional nu is the region-averaged per-triangle physical tensor (validates the loop + the spatial
pattern; not a fully independent sub-region modulus).
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _common as C

CASE, REG = 'auxetic_patch', 0.002
NU_OUT, NU_PATCH = 0.30, -0.30


def regions(prob):
    cen = prob.centroids
    center = cen.mean(0)
    span = cen[:, 0].max() - cen[:, 0].min()
    radius = 0.20 * span
    patch = prob.region_in_circle(center, radius)
    outside = np.setdiff1d(np.arange(prob.n_tri), patch)
    return patch, outside, center, radius


def main():
    d = C.savedir(CASE)
    rows = []                       # (topo,label,N, out_solv,out_sim, pat_solv,pat_sim)
    detail = {}
    for topo, label, _ in C.TOPOS:
        for N in C.SIZES:
            prob, geo = C.make_case(topo, N)
            patch, outside, center, radius = regions(prob)
            objs = [C.Objective('nu', NU_OUT, region=outside, weight=1.0),
                    C.Objective('nu', NU_PATCH, region=patch, weight=1.5)]
            res = C.optimize(prob, objs, mode='k', n_iter=120, reg=REG, verbose=False)
            os_ = C.solver_region_nuE(prob, res['k'], outside)[0]
            ps_ = C.solver_region_nuE(prob, res['k'], patch)[0]
            C.apply_k_to_geo(geo, res['k'])
            C6 = C.sim_per_triangle_C6(geo)
            om = C.c6_nuE(C.region_phys_C6(geo, C6, outside))[0]
            pm = C.c6_nuE(C.region_phys_C6(geo, C6, patch))[0]
            rows.append((topo, label, N, os_, om, ps_, pm))
            print(f"  {topo:12s} N={N} | outside solver/sim={os_:+.3f}/{om:+.3f}  "
                  f"patch solver/sim={ps_:+.3f}/{pm:+.3f}", flush=True)
            if N == C.SIZES[-1]:
                detail[topo] = (label, geo, res['k'], C6,
                                {'kind': 'circle', 'center': center, 'radius': radius})

    C.write_csv(os.path.join(d, f'{CASE}.csv'),
                ['topology', 'half_size', 'out_target', 'out_solver', 'out_sim',
                 'patch_target', 'patch_solver', 'patch_sim'],
                [(r[0], r[2], NU_OUT, f'{r[3]:.4f}', f'{r[4]:.4f}', NU_PATCH,
                  f'{r[5]:.4f}', f'{r[6]:.4f}') for r in rows])
    print(f"\n{'topology':12s} {'N':>3} | {'out sim':>8} {'patch sim':>9}   (targets +0.30 / -0.30)")
    for r in rows:
        print(f"{r[0]:12s} {r[2]:>3} | {r[4]:>+8.3f} {r[6]:>+9.3f}")

    # ---- overlaid summary: outside nu vs patch nu ----
    fig, ax = plt.subplots(figsize=(7.5, 7))
    ax.axvline(NU_OUT, color='gray', ls='--', lw=1); ax.axhline(NU_PATCH, color='gray', ls='--', lw=1)
    ax.scatter([NU_OUT], [NU_PATCH], marker='*', s=320, c='gold', edgecolor='k', zorder=6, label='target')
    for topo, label, _ in C.TOPOS:
        for N in C.SIZES:
            m = [r for r in rows if r[0] == topo and r[2] == N]
            if m:
                r = m[0]
                ax.scatter([r[4]], [r[6]], c=C.TOPO_COLORS[topo], marker=C.SIZE_MARKERS[N], s=80,
                           edgecolor='k', lw=0.4, zorder=4, label=label if N == C.SIZES[0] else None)
                ax.scatter([r[3]], [r[5]], c='k', marker='+', s=40, alpha=0.6, zorder=3)
    ax.axhline(0, color='gray', lw=0.4, ls=':')
    ax.set_xlabel('SIMULATED outside ν  (+ = solver)'); ax.set_ylabel('SIMULATED patch ν')
    ax.set_title(f'{CASE}: outside → +0.30, patch → −0.30 (each point = one topology×size)')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(d, f'{CASE}_summary.png'), dpi=150, bbox_inches='tight')
    plt.close(); print('saved summary')

    # ---- small multiples: per-topology grouped bars (target / solver / sim, out & patch) ----
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, (topo, label, _) in zip(axes.ravel(), C.TOPOS):
        r = [x for x in rows if x[0] == topo and x[2] == C.SIZES[-1]][0]
        xs = np.arange(2); w = 0.26
        ax.bar(xs - w, [NU_OUT, NU_PATCH], w, label='target', color='0.7')
        ax.bar(xs, [r[3], r[5]], w, label='solver', color=C.TOPO_COLORS[topo], alpha=0.6)
        ax.bar(xs + w, [r[4], r[6]], w, label='sim', color=C.TOPO_COLORS[topo])
        ax.axhline(0, color='k', lw=0.5); ax.set_xticks(xs); ax.set_xticklabels(['outside', 'patch'])
        ax.set_title(label, fontsize=10); ax.set_ylabel('ν'); ax.legend(fontsize=7); ax.grid(alpha=0.3, axis='y')
    axes.ravel()[-1].axis('off')
    fig.suptitle(f'{CASE}: per-topology — target vs solver vs SIMULATION (outside & patch ν)', fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(d, f'{CASE}_bytopo.png'), dpi=150, bbox_inches='tight')
    plt.close(); print('saved bytopo')

    entries = [(*detail[t[0]][:4], detail[t[0]][4]) for t in C.TOPOS if t[0] in detail]
    C.design_detail_figure(os.path.join(d, f'{CASE}_detail.png'), entries,
                           f'{CASE}: designed networks (auxetic patch in a positive matrix, largest '
                           f'size) — rigidity k, local ν (lime = patch), local E')
    C.design_detail_per_topology(d, CASE, entries, f'{CASE} — {{name}}: auxetic patch (lime) in a positive matrix')
    print('saved detail grid + per-topology')


if __name__ == '__main__':
    main()
