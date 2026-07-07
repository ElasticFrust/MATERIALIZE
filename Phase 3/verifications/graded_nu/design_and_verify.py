"""
Case graded_nu (SPATIAL GRADIENT). For each topology x size, design k so nu varies across x — a
stack of vertical-strip objectives from nu~+0.3 (left) to nu~-0.3 (right). Simulate and check the
realised nu(x) profile per strip. Outputs:
  - graded_nu.csv            : topology, size, strip x, target, solver, sim
  - graded_nu_summary.png    : overlaid nu(x), all topos (sim solid, solver dotted, target dashed)
  - graded_nu_bytopo.png     : small multiples, one nu(x) panel per topology
  - graded_nu_detail.png     : 5-topology grid [rigidity k | local nu | local E], strip lines marked
  - graded_nu_<topology>.png : per-topology detail
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _common as C

CASE, REG, NSTRIP = 'graded_nu', 0.002, 6
NU_L, NU_R = 0.30, -0.30


def strips(prob):
    x = prob.centroids[:, 0]
    xg = np.linspace(x.min(), x.max() + 1e-9, NSTRIP + 1)
    idx = [np.where((x >= xg[i]) & (x < xg[i + 1]))[0] for i in range(NSTRIP)]
    xc = 0.5 * (xg[:-1] + xg[1:])
    tgt = np.linspace(NU_L, NU_R, NSTRIP)
    return idx, xc, tgt, xg


def main():
    d = C.savedir(CASE)
    res_all = {}                    # (topo,N) -> (xc, tgt, sim, solver)
    detail = {}
    csv_rows = []
    for topo, label, _ in C.TOPOS:
        for N in C.SIZES:
            prob, geo = C.make_case(topo, N)
            idx, xc, tgt, xg = strips(prob)
            objs = [C.Objective('nu', float(tgt[i]), region=idx[i]) for i in range(NSTRIP)
                    if len(idx[i]) >= 3]
            r = C.optimize(prob, objs, mode='k', n_iter=150, reg=REG, verbose=False)
            solv = np.array([C.solver_region_nuE(prob, r['k'], idx[i])[0] if len(idx[i]) >= 3
                             else np.nan for i in range(NSTRIP)])
            C.apply_k_to_geo(geo, r['k'])
            C6 = C.sim_per_triangle_C6(geo)
            sim = np.array([C.c6_nuE(C.region_phys_C6(geo, C6, idx[i]))[0] if len(idx[i]) >= 3
                            else np.nan for i in range(NSTRIP)])
            res_all[(topo, N)] = (xc, tgt, sim, solv)
            for i in range(NSTRIP):
                csv_rows.append((topo, N, i, f'{tgt[i]:.3f}', f'{solv[i]:.4f}', f'{sim[i]:.4f}'))
            print(f"  {topo:12s} N={N} | sim nu(x)=" + " ".join(f"{v:+.2f}" for v in sim), flush=True)
            if N == C.SIZES[-1]:
                detail[topo] = (label, geo, r['k'], C6, {'kind': 'vlines', 'xs': xg[1:-1]})

    C.write_csv(os.path.join(d, f'{CASE}.csv'),
                ['topology', 'half_size', 'strip', 'target_nu', 'solver_nu', 'sim_nu'], csv_rows)
    print(f"\nprescribed nu(x): " + " ".join(f"{v:+.2f}" for v in np.linspace(NU_L, NU_R, NSTRIP)))

    xn = np.linspace(0, 1, NSTRIP)
    lbl = {t: l for t, l, _ in C.TOPOS}

    # ---- overlaid summary ----
    fig, ax = plt.subplots(figsize=(8, 6.5))
    ax.plot(xn, np.linspace(NU_L, NU_R, NSTRIP), 'k--', lw=2.5, label='prescribed ν(x)')
    for (topo, N), (xc, tgt, sim, solv) in res_all.items():
        if N != C.SIZES[-1]:
            continue
        ax.plot(xn, sim, '-o', color=C.TOPO_COLORS[topo], ms=4, lw=1.6, label=lbl[topo])
        ax.plot(xn, solv, ':', color=C.TOPO_COLORS[topo], lw=0.9, alpha=0.6)
    ax.axhline(0, color='gray', lw=0.5, ls=':')
    ax.set_xlabel('normalised x'); ax.set_ylabel('ν  (solid = SIMULATION, dotted = solver)')
    ax.set_title(f'{CASE}: prescribed vs simulated ν(x) gradient (largest size, all topologies)')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(d, f'{CASE}_summary.png'), dpi=150, bbox_inches='tight')
    plt.close(); print('saved summary')

    # ---- small multiples ----
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, (topo, label, _) in zip(axes.ravel(), C.TOPOS):
        xc, tgt, sim, solv = res_all[(topo, C.SIZES[-1])]
        ax.plot(xn, np.linspace(NU_L, NU_R, NSTRIP), 'k--', lw=2, label='target')
        ax.plot(xn, sim, '-o', color=C.TOPO_COLORS[topo], ms=4, lw=1.6, label='sim')
        ax.plot(xn, solv, ':', color='k', lw=1.0, alpha=0.6, label='solver')
        ax.axhline(0, color='gray', lw=0.5, ls=':')
        ax.set_title(label, fontsize=10); ax.set_xlabel('normalised x'); ax.set_ylabel('ν')
        ax.legend(fontsize=7); ax.grid(alpha=0.3)
    axes.ravel()[-1].axis('off')
    fig.suptitle(f'{CASE}: per-topology — prescribed vs SIMULATED ν(x)', fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(d, f'{CASE}_bytopo.png'), dpi=150, bbox_inches='tight')
    plt.close(); print('saved bytopo')

    entries = [(*detail[t[0]][:4], detail[t[0]][4]) for t in C.TOPOS if t[0] in detail]
    C.design_detail_figure(os.path.join(d, f'{CASE}_detail.png'), entries,
                           f'{CASE}: designed networks (ν(x) gradient, largest size) — '
                           f'rigidity k, local ν (lime = strip boundaries), local E')
    C.design_detail_per_topology(d, CASE, entries, f'{CASE} — {{name}}: graded ν(x) left(+)→right(−)')
    print('saved detail grid + per-topology')


if __name__ == '__main__':
    main()
