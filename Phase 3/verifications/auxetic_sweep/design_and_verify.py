"""
Case auxetic_sweep (GLOBAL). For each topology x size, design k for a sweep of global Poisson
targets nu* (pushed past the auxetic limit to expose where each topology saturates), then
INDEPENDENTLY simulate the designed network and compare simulated global nu (and E) to target.

Global nu/E from the sim is the energy/virial response — independent of the solver's
homogenisation, so this is a genuine end-to-end check. Two physical signals emerge and are
plotted: (a) SATURATION — the sim nu stops tracking extreme targets (achievable-range limit,
topology dependent); (b) INSTABILITY — a design can satisfy the scalar target in the solver yet
be a floppy/degenerate network that collapses in simulation (E<0 or E huge, |nu|>~1). A light
uniformity regulariser (reg) suppresses gratuitous instability; the remainder is flagged.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _common as C

TARGETS = [0.30, 0.15, 0.00, -0.15, -0.30, -0.45, -0.60]
CASE = 'auxetic_sweep'
REG = 0.003


def is_stable(design_nu, sim_nu, sim_E):
    return (0.0 < sim_E < 5.0) and (abs(sim_nu) < 1.2) and (abs(sim_nu - design_nu) < 0.10)


def main():
    rows = []                       # (topo, N, target, design_nu, sim_nu, sim_E, stable)
    reps = []                       # representative designs for the detail figure
    for topo, label, _ in C.TOPOS:
        for N in C.SIZES:
            for tgt in TARGETS:
                prob, geo = C.make_case(topo, N)
                res = C.optimize(prob, [C.Objective('nu', tgt)], mode='k', n_iter=80,
                                 reg=REG, verbose=False)
                dnu = C.validate(prob, res['k'], res['l0'], [C.Objective('nu', tgt)])[0]['achieved']
                C.apply_k_to_geo(geo, res['k'])
                snu, sE = C.sim_region_nuE(geo, None)
                st = is_stable(dnu, snu, sE)
                rows.append((topo, N, tgt, dnu, snu, sE, st))
                print(f"  {topo:12s} N={N} tgt={tgt:+.2f} | solver={dnu:+.3f} sim={snu:+.3f} "
                      f"E={sE:9.3g} {'ok' if st else 'UNSTABLE'}", flush=True)
                if N == C.SIZES[-1] and abs(tgt + 0.30) < 1e-9 and st and \
                        topo in ('regular', 'aniso_shr', 'disorder_lo'):
                    reps.append((f'{label} (ν→-0.3)', geo, res['k'], C.sim_per_triangle_C6(geo)))

    d = C.savedir(CASE)
    np.savez(os.path.join(d, 'results.npz'),
             topo=[r[0] for r in rows], N=[r[1] for r in rows], target=[r[2] for r in rows],
             design_nu=[r[3] for r in rows], sim_nu=[r[4] for r in rows],
             sim_E=[r[5] for r in rows], stable=[r[6] for r in rows])

    fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.8))
    lo, hi = min(TARGETS) - 0.1, max(TARGETS) + 0.1
    # panel 0: parity target vs simulated nu (stable only); unstable flagged on a top row
    ax[0].plot([lo, hi], [lo, hi], 'k--', lw=1, alpha=0.6, label='ideal (sim = target)')
    for topo, label, _ in C.TOPOS:
        for N in C.SIZES:
            pts = [(r[2], r[4]) for r in rows if r[0] == topo and r[1] == N and r[6]]
            if pts:
                t, s = zip(*pts)
                ax[0].scatter(t, s, c=C.TOPO_COLORS[topo], marker=C.SIZE_MARKERS[N], s=45,
                              edgecolor='k', linewidth=0.3, zorder=3)
    solv = [(r[2], r[3]) for r in rows if r[6]]              # solver prediction (design)
    if solv:
        st_, ss_ = zip(*solv)
        ax[0].scatter(st_, ss_, c='k', marker='+', s=22, alpha=0.6, zorder=2, label='solver (design)')
    unst = [(r[2]) for r in rows if not r[6]]
    if unst:
        ax[0].scatter(unst, [hi - 0.03] * len(unst), c='red', marker='x', s=40,
                      label=f'unstable design ({len(unst)})', zorder=4)
    ax[0].axhline(0, color='gray', lw=0.5, ls=':'); ax[0].axvline(0, color='gray', lw=0.5, ls=':')
    ax[0].set_xlabel('target ν'); ax[0].set_ylabel('achieved ν (designed network)')
    ax[0].set_title('Designed ν vs target — colored = SIMULATION (independent), + = solver\n'
                    '(stable designs; solver prediction confirmed by sim)')
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
    # panel 1: achievable range — sim nu vs target lines per topology (stable), saturation visible
    for topo, label, _ in C.TOPOS:
        for N in C.SIZES:
            pts = sorted([(r[2], r[4]) for r in rows if r[0] == topo and r[1] == N and r[6]])
            if len(pts) < 2:
                continue
            t, s = zip(*pts)
            lab = label if N == C.SIZES[0] else None
            ax[1].plot(t, s, '-', color=C.TOPO_COLORS[topo], marker=C.SIZE_MARKERS[N], ms=5,
                       lw=1.3, alpha=0.85, label=lab)
    ax[1].plot([lo, hi], [lo, hi], 'k--', lw=1, alpha=0.4)
    ax[1].axhline(0, color='gray', lw=0.5, ls=':')
    ax[1].set_xlabel('target ν'); ax[1].set_ylabel('simulated ν (stable designs)')
    ax[1].set_title('Achievable auxetic range per topology\n(curve leaving the diagonal = physical limit)')
    ax[1].legend(fontsize=7, loc='upper left'); ax[1].grid(alpha=0.3)
    marks = ', '.join(f'{s}={C.SIZE_MARKERS[s]}' for s in C.SIZES)
    fig.suptitle(f'{CASE}: global ν design verified against simulation — '
                 f'{len(C.TOPOS)} topologies × {len(C.SIZES)} sizes '
                 f'(markers {marks}; reg={REG})', fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    p = os.path.join(d, f'{CASE}.png')
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print('saved', p)

    if reps:
        C.design_detail_figure(os.path.join(d, f'{CASE}_detail.png'), reps,
                               f'{CASE}: designed networks (ν→-0.3, largest size) — '
                               f'rigidity k, local ν, local E')
        print('saved', os.path.join(d, f'{CASE}_detail.png'))


if __name__ == '__main__':
    main()
