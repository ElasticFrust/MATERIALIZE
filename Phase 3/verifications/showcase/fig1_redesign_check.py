r"""fig1's frozen network is STALE — redesign it, and show what the staleness cost.

Purpose. `fig1_recreate_pointy.py` caches its design in `fig1_pointy_network.npz`, frozen
**2026-07-12**, and renders whatever is in that file. The August audit changed the forward map
(A-0 shear contraction, A-10 ν convention), so the committed figure shows a JULY design read by
AUGUST instruments: ν(θ) swings to [-18.1, +19.1] against a target of [-3.63, +7.18]. That is not
an optimiser failure and not a rendering failure — it is `Phase 5/PLAN.md`'s "saved designs are
stale, re-run don't re-analyse", applying to Phase 3's frozen showcase networks, where it had not
been noticed.

What this script establishes.
  1. The same frozen k evaluated by today's solver gives C6 ~16x larger than July recorded, with a
     ~10% spread BETWEEN components. A uniform factor would leave ν untouched; the spread is what
     moves ν. (July's own record shows it never met the target either: peak +2.80 vs +7.18.)
  2. A fresh design at HEAD meets the target and is confirmed by the INDEPENDENT sim:
     rms(sim - target) = 1.7e-04, max|ν_solver - ν_sim| = 4.8e-09.
  3. The redesign is not a lucky basin: all five restarts converge onto the target
     (rms 1.1e-03 .. 1.8e-03; best 1.7e-04).

Honest limits. The design is FLOPPY — ~27% of bonds end below 0.02x median k. It hits the target
exactly and the independent sim agrees to 5e-09, so it is sound as a solution, but `reg=5e-3` (the
value fig1 uses) is below the 0.01-0.05 that `CLAUDE.md` §3 recommends, and that is the knob for a
less degenerate design at the same target.

Plot policy note. The network is drawn with a 5-95 PERCENTILE colour norm, not the linear
[kmin,kmax] norm in `plotting.py`. With a single stiff outlier (k_max/median ~ 17 here, ~150 for the
July network) the linear norm maps every bond to the bottom of viridis and the panel reads as a flat
dark mass — which is what made the regenerated figures look like they had "missing parts". This
deviation is deliberate, marked in the panel title, and is the open question behind audit B-4.

Run:  python fig1_redesign_check.py            (anaconda python; ~38 min if the npz is absent)
Out:  fig1_pointy_network_redesigned.npz, fig1_redesign_restarts.npz,
      fig1_stale_vs_fresh.png, fig1_redesigned_response.png, fig1_redesigned_network.png

Leaves the July artifacts (`fig1_pointy_network.npz`, `fig1_recreate_pointy.png`) UNTOUCHED.
"""
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
VER = os.path.dirname(HERE)
REPO = os.path.abspath(os.path.join(VER, '..', '..'))
sys.path.insert(0, VER)
import _common as C                                                   # noqa: E402
sys.path.insert(0, REPO)
import matplotlib                                                     # noqa: E402
matplotlib.use('Agg')
import matplotlib.colors as mcolors                                   # noqa: E402
import matplotlib.pyplot as plt                                       # noqa: E402
import plotting as P                                                  # noqa: E402

torch.set_default_dtype(torch.float64)

TOPO, N, REG, NITER, NREST, SEED = 'disorder_hi', 16, 5e-3, 500, 5, 0   # as fig1_recreate_pointy.py
TH, DEG = C.ANG, np.degrees(C.ANG)
JULY = os.path.join(HERE, 'fig1_pointy_network.npz')
FRESH = os.path.join(HERE, 'fig1_pointy_network_redesigned.npz')
RESTARTS = os.path.join(HERE, 'fig1_redesign_restarts.npz')


def crystal_target():
    """fig1's target: ν(θ) of the φ=4, ψ=1 crystal at uniform k, via the INDEPENDENT sim."""
    cg = C.make_crystal(4.0, 1.0, half=4.0)
    cg['bond_k'] = np.ones(len(cg['bond_R']))
    cg['tri_k'] = cg['bond_k'][cg['tri_bond']]
    return C.nu_E_theta(C.sim_region_C6(cg), TH)[0]


def design_or_load(target_nu):
    """Load the redesigned network, or run fig1's five restarts ONE AT A TIME and save the best.

    Running them individually (seed=SEED+r, n_restarts=1) is equivalent to optimize's own loop --
    it calls _init_raw(prob, mode, seed+r) -- but lets each restart be timed and scored, which is
    what shows the redesign is reproducible rather than one lucky basin."""
    prob, geo = C.make_case(TOPO, N)
    if os.path.exists(FRESH):
        g, k, C6_per, meta = C.load_network(FRESH)
        return prob, g, np.asarray(k, float), np.asarray(C6_per, float), meta
    rows, best = [], None
    for r in range(NREST):
        t0 = time.perf_counter()
        res = C.optimize(prob, [C.Objective('nu_theta', target_nu)], mode='k',
                         n_iter=NITER, n_restarts=1, seed=SEED + r, reg=REG, verbose=False)
        dt = time.perf_counter() - t0
        k = np.asarray(res['k'].detach().numpy(), float)
        nu = C.nu_E_theta(C.solver_region_C6(prob, res['k']), TH)[0]
        rms = float(np.sqrt(np.mean((nu - target_nu) ** 2)))
        rows.append((r, float(res['loss']), rms, float(np.median(k)), float(k.min()),
                     int((k < 0.02 * np.median(k)).sum()), dt))
        print(f'  restart {r}: {dt:6.1f} s  loss={res["loss"]:.6g}  rms={rms:.4g}', flush=True)
        if best is None or res['loss'] < best[0]:
            best = (float(res['loss']), k)
    np.savez(RESTARTS, rows=np.array(rows, float),
             cols=np.array(['restart', 'loss', 'rms', 'k_median', 'k_min', 'n_dashed', 'seconds']))
    k = best[1]
    C.apply_k_to_geo(geo, k)
    C6_per = C.sim_per_triangle_C6(geo)          # as fig1 stores it (spatial pattern, not a check)
    C.save_network(FRESH, geo, k, C6_per=C6_per, seed=SEED, topo=TOPO, N=N, reg=REG,
                   n_iter=NITER, n_restarts=NREST, target='crystal_phi4_psi1',
                   nu_target=list(map(float, target_nu)), loss=best[0],
                   note='redesign of the STALE 2026-07-12 fig1_pointy_network.npz')
    return prob, geo, k, C6_per, {}


def main():
    target_nu = crystal_target()
    print(f'target nu in [{target_nu.min():+.4f}, {target_nu.max():+.4f}]', flush=True)
    prob, geo, k, _, _ = design_or_load(target_nu)

    nu_slv, E_slv = C.nu_E_theta(C.solver_region_C6(prob, torch.as_tensor(k)), TH)
    geo_s = dict(geo); C.apply_k_to_geo(geo_s, k)
    nu_sim, E_sim = C.nu_E_theta(C.sim_bulk_C6(geo_s), TH)
    gap = float(np.abs(nu_slv - nu_sim).max())
    rms = float(np.sqrt(np.mean((nu_sim - target_nu) ** 2)))
    print(f'FRESH: rms(sim-target)={rms:.3e}   max|nu_solver-nu_sim|={gap:.3e}', flush=True)

    # the July network, as July recorded it and as today's solver reads it
    geo_j, k_j, C6per_j, _ = C.load_network(JULY)
    k_j = np.asarray(k_j, float)
    nu_j_stored = C.nu_E_theta(np.asarray(C6per_j, float).reshape(-1, 6).mean(0), TH)[0]
    C6_j_now = np.asarray(C.solver_region_C6(C.DesignProblem.from_geo(geo_j),
                                             torch.as_tensor(k_j)), float).ravel()
    nu_j_now = C.nu_E_theta(C6_j_now, TH)[0]
    C6_j_stored = np.asarray(C6per_j, float).reshape(-1, 6).mean(0)
    print('C6 stored (July)  :', np.array2string(C6_j_stored, precision=5), flush=True)
    print('C6 today, same k  :', np.array2string(C6_j_now, precision=5), flush=True)
    print('component ratio   :', np.array2string(C6_j_now / C6_j_stored, precision=2), flush=True)

    # ---- 1. stale vs fresh ---------------------------------------------------------------------
    fig, (ax, axz) = plt.subplots(1, 2, figsize=(14.0, 5.2), width_ratios=[1.7, 1.0])
    ax.plot(DEG, target_nu, 'k--', lw=2.6, label='target (crystal φ4,ψ1)')
    ax.plot(DEG, nu_j_now, color='tab:red', lw=2.0,
            label="JULY design, read by TODAY's solver  ← the committed figure")
    ax.plot(DEG, nu_j_stored, color='tab:orange', lw=1.8, ls='-.',
            label='JULY design, as JULY recorded it')
    ax.plot(DEG, nu_sim, color='tab:blue', lw=2.2, label='REDESIGN at HEAD (independent sim)')
    ax.axhline(0, color='0.7', lw=0.8)
    ax.set_xlim(0, 180); ax.set_xticks(range(0, 181, 45)); ax.grid(alpha=0.25)
    ax.set_xlabel('loading angle θ (deg)'); ax.set_ylabel('ν(θ)'); ax.legend(fontsize=8.5)
    ax.set_title('the committed curve is a July design read by August instruments', fontsize=10.5)
    for a, arr in ((axz, None),):
        a.plot(DEG, target_nu, 'k--', lw=2.6); a.plot(DEG, nu_j_stored, color='tab:orange', lw=1.8, ls='-.')
        a.plot(DEG, nu_sim, color='tab:blue', lw=2.2); a.axhline(0, color='0.7', lw=0.8)
        a.set_xlim(0, 180); a.set_xticks(range(0, 181, 45)); a.set_ylim(-4.5, 8); a.grid(alpha=0.25)
        a.set_xlabel('θ (deg)')
        a.set_title("on July's axis range:\nredesign = target, July falls short", fontsize=10)
    fig.suptitle(f'fig1 case — {TOPO} N={N} ({len(k)} bonds), reg={REG}   ·   '
                 f'redesign rms vs target {rms:.1e}', fontsize=12)
    fig.tight_layout()
    P.save_fig(fig, os.path.join(HERE, 'fig1_stale_vs_fresh.png'))

    # ---- 2. the redesign's response (solver and sim SHARE the panels, per policy) ---------------
    f2 = P.plot_directional(TH, [(nu_slv, E_slv), (nu_sim, E_sim)], target_nu=target_nu,
                            labels=['solver (design path)', 'independent sim'],
                            suptitle=f'fig1 REDESIGN — sim vs target rms {rms:.1e}   ·   '
                                     f'solver vs sim max|Δν| {gap:.1e}\n'
                                     '(the three curves coincide; the solver line is under the sim)')
    P.save_fig(f2, os.path.join(HERE, 'fig1_redesigned_response.png'))

    # ---- 3. the network ------------------------------------------------------------------------
    f3, ax3 = plt.subplots(figsize=(6.4, 6.4))
    lc = P.draw_network(ax3, geo_s, k)                    # policy norm: top cut at K_HI_PCT, not max
    f3.colorbar(lc, ax=ax3, fraction=0.046, label='bond rigidity k', extend='max')
    n_d = int((k < 0.02 * np.median(k)).sum())
    ax3.set_title(f'fig1 redesign — k median {np.median(k):.3f}, TRUE max {k.max():.2f} '
                  f'(colour cut at the {P.STYLE.K_HI_PCT}th pct); '
                  f'{n_d}/{len(k)} bonds dashed (k < 0.02·median)', fontsize=9.5)
    P.save_fig(f3, os.path.join(HERE, 'fig1_redesigned_network.png'))
    print('wrote fig1_stale_vs_fresh.png, fig1_redesigned_response.png, '
          'fig1_redesigned_network.png', flush=True)


if __name__ == '__main__':
    main()
