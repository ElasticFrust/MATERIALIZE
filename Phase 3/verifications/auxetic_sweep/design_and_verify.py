"""
Case auxetic_sweep (GLOBAL Poisson ratio). For each topology x size, design k for a sweep of
global targets nu*, then INDEPENDENTLY simulate the designed network and compare the simulated
global nu to the target (and to the forward solver's own prediction). Outputs:
  - auxetic_sweep.csv                : topology, size, target, solver, sim, error, stable
  - auxetic_sweep_summary.png        : overlaid achievable-range (sim solid, solver dotted), all topos
  - auxetic_sweep_bytopo.png         : small multiples, one panel per topology
  - auxetic_sweep_detail.png         : 5-topology grid [rigidity k | local nu | local E] at nu*=-0.3
  - auxetic_sweep_<topology>.png     : per-topology detail

Global nu from the sim is the energy/virial response (independent of the solver's homogenisation).
Saturation = the topology's physical auxetic limit; extreme targets can give floppy designs the
simulation flags as unstable (E<0/huge or |nu|>1.2).
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _common as C

TARGETS = [0.30, 0.15, 0.00, -0.15, -0.30, -0.45, -0.60]
DETAIL_TOPOS = C.TOPO_IDS                                    # per-topo detail across ALL targets, EVERY topology
CASE, REG = 'auxetic_sweep', 0.003


def is_stable(dnu, snu, sE):
    return (0.0 < sE < 5.0) and (abs(snu) < 1.2) and (abs(snu - dnu) < 0.10)


def main():
    d = C.savedir(CASE)
    ndir = C.networks_dir(CASE)
    rows = []                       # (topo, label, N, target, solver, sim, simE, stable)
    detail = {t: [] for t in DETAIL_TOPOS}      # topo -> [(name, geo, k, C6_per, None) per target]
    for topo, label, _ in C.TOPOS:
        for N in C.SIZES:
            for tgt in TARGETS:
                prob, geo = C.make_case(topo, N)
                res = C.optimize(prob, [C.Objective('nu', tgt)], mode='k', n_iter=80,
                                 reg=REG, verbose=False)
                dnu = C.solver_region_nuE(prob, res['k'])[0]
                C.apply_k_to_geo(geo, res['k'])
                C6 = C.sim_per_triangle_C6(geo)                    # one relaxation, reused below
                snu, sE = C.c6_nuE(C.region_phys_C6(geo, C6, None))
                st = is_stable(dnu, snu, sE)
                rows.append((topo, label, N, tgt, dnu, snu, sE, st))
                C.save_network(os.path.join(ndir, f'{topo}_N{N}_nu{tgt:+.2f}.npz'), geo, res['k'], C6,
                               topo=topo, N=N, target=float(tgt), solver_nu=float(dnu),
                               sim_nu=float(snu), sim_E=float(sE), stable=bool(st))
                print(f"  {topo:12s} N={N} tgt={tgt:+.2f} | solver={dnu:+.3f} sim={snu:+.3f} "
                      f"E={sE:8.3g} {'ok' if st else 'UNSTABLE'}", flush=True)
                if N == C.SIZES[-1] and topo in DETAIL_TOPOS:      # capture EVERY target for detail
                    detail[topo].append((f'nu*={tgt:+.2f}', geo, res['k'], C6, None))

    # ---- CSV + printed table ----
    C.write_csv(os.path.join(d, f'{CASE}.csv'),
                ['topology', 'half_size', 'target_nu', 'solver_nu', 'sim_nu', 'sim_E', 'stable'],
                [(r[0], r[2], r[3], f'{r[4]:.4f}', f'{r[5]:.4f}', f'{r[6]:.4f}', r[7]) for r in rows])
    print(f"\n{'topology':12s} {'N':>3} {'tgt':>6} {'solver':>7} {'sim':>7} {'stable':>7}")
    for r in rows:
        print(f"{r[0]:12s} {r[2]:>3} {r[3]:>+6.2f} {r[4]:>+7.3f} {r[5]:>+7.3f} {str(r[7]):>7}")

    lo, hi = min(TARGETS) - 0.08, max(TARGETS) + 0.08

    def panel(ax, topo):
        ax.plot([lo, hi], [lo, hi], 'k--', lw=1, alpha=0.5)
        ax.axhline(0, color='gray', lw=0.4, ls=':')
        for N, mk in C.SIZE_MARKERS.items():
            stab = sorted([(r[3], r[5], r[4]) for r in rows if r[0] == topo and r[2] == N and r[7]])
            unst = [r[3] for r in rows if r[0] == topo and r[2] == N and not r[7]]
            if stab:
                t = [x[0] for x in stab]; s = [x[1] for x in stab]; v = [x[2] for x in stab]
                ax.plot(t, s, '-', color=C.TOPO_COLORS[topo], marker=mk, ms=5, lw=1.5,
                        label=f'sim N={N}')
                ax.plot(t, v, ':', color='k', lw=0.8, alpha=0.5, label='solver' if N == C.SIZES[0] else None)
            if unst:
                ax.scatter(unst, [hi - 0.03] * len(unst), c='red', marker='x', s=30)

    # ---- overlaid summary (all topologies, sim solid + solver dotted) ----
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot([lo, hi], [lo, hi], 'k--', lw=1, alpha=0.5, label='ideal (sim = target)')
    ax.axhline(0, color='gray', lw=0.4, ls=':')
    for topo, label, _ in C.TOPOS:
        st = sorted([(r[3], r[5]) for r in rows if r[0] == topo and r[2] == C.SIZES[-1] and r[7]])
        sv = sorted([(r[3], r[4]) for r in rows if r[0] == topo and r[2] == C.SIZES[-1] and r[7]])
        if st:
            t, s = zip(*st); ax.plot(t, s, '-o', color=C.TOPO_COLORS[topo], ms=5, lw=1.6, label=label)
        if sv:
            t, v = zip(*sv); ax.plot(t, v, ':', color=C.TOPO_COLORS[topo], lw=1.0, alpha=0.6)
    ax.scatter([], [], c='red', marker='x', label='unstable (not shown)')
    ax.set_xlabel('target ν'); ax.set_ylabel('achieved ν  (line = SIMULATION, dotted = solver)')
    ax.set_title(f'{CASE}: global ν design vs simulation (largest size)\n'
                 'curve leaving the diagonal = that topology\'s physical auxetic limit')
    ax.legend(fontsize=9, loc='upper left'); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(d, f'{CASE}_summary.png'), dpi=150, bbox_inches='tight')
    plt.close(); print('saved', f'{CASE}_summary.png')

    # ---- small multiples (one panel per topology) ----
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, (topo, label, _) in zip(axes.ravel(), C.TOPOS):
        panel(ax, topo)
        ax.set_title(label, fontsize=10); ax.set_xlabel('target ν'); ax.set_ylabel('ν')
        ax.legend(fontsize=7); ax.grid(alpha=0.3)
    axes.ravel()[-1].axis('off')
    fig.suptitle(f'{CASE}: per-topology — SIMULATED ν vs target (red x = unstable design)', fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(d, f'{CASE}_bytopo.png'), dpi=150, bbox_inches='tight')
    plt.close(); print('saved', f'{CASE}_bytopo.png')

    # ---- detail: for each detail topology, ALL targets as rows [k | local ν | local E] ----
    for topo in DETAIL_TOPOS:
        if detail.get(topo):
            C.design_detail_figure(os.path.join(d, f'{CASE}_alltargets_{topo}.png'), detail[topo],
                                   f'{CASE}: {topo} — designs across ALL global targets ν* (rows, '
                                   f'largest size) — rigidity k, local ν, local E')
    print('saved per-topology all-target detail')


if __name__ == '__main__':
    main()
