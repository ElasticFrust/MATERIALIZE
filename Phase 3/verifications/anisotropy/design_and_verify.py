"""
Case anisotropy — directional Poisson ratio ν(θ) and Young's modulus E(θ), θ∈[0,π]. Three tests,
all driven by the differentiable directional objectives (inverse_design.Objective kind='nu_theta'
and 'E_theta'):

  TEST A — PROGRAM ν(θ): design k so the network realises a PRESCRIBED angular profile ν*(θ)
           (magnitude + direction), on every topology. Profiles: a 2-fold lobe, a 4-fold pattern,
           and a sign-flipping 'directional-auxetic' profile (ν>0 one way, ν<0 across it).

  TEST B — ISOTROPIZE: design k so an ANISOTROPIC base becomes ISOTROPIC at a CHOSEN level
           ν(θ)=ν0 (isotropic is not only 1/3): ν0 ∈ {0.8,0.5,1/3,0,-0.2,-0.5,-0.8}.

  TEST C — INDEPENDENT E / ν ANISOTROPY: make E(θ) directional while ν(θ) stays FLAT, and the
           reverse (ν(θ) directional while E(θ) stays flat) — decoupling the two moduli's angular
           dependence.

ν(θ), E(θ) read from the FULL tensor: solver prediction (objective) AND the independent PBC
simulation of the designed network. Outputs:
  - anisotropy.csv                    : test, topology/base, target, solver err, sim err
  - anisotropy_program.png            : per profile; target (black) + each topology (sim solid, solver dotted)
  - anisotropy_isotropize.png         : per base; base ν(θ) + designed ν(θ) flattened to each ν0
  - anisotropy_isotropize_parity.png  : achieved mean ν vs target ν0 (flatness = error bar), per base
  - anisotropy_independent_EnuC.png   : Test C — ν(θ) and E(θ), one flat while the other is directional
  - anisotropy_detail.png             : representative designs [rigidity k | local ν | local E]
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _common as C

CASE, REG, N, NITER = 'anisotropy', 1e-4, 10, 110
TH = C.ANG                      # θ∈[0,π] grid (shared with the design objectives)
DEG = np.degrees(TH)

PROFILES = [                    # (key, label, ν*(θ))
    ('two_fold',   '2-fold  0.2+0.35cos2θ',         0.20 + 0.35 * np.cos(2 * TH)),
    ('four_fold',  '4-fold  0.25+0.25cos4θ',        0.25 + 0.25 * np.cos(4 * TH)),
    ('dir_aux',    'directional-auxetic 0.45cos2θ', 0.45 * np.cos(2 * TH)),
]
ISO_TGTS = [0.8, 0.5, 1 / 3., 0.0, -0.2, -0.5, -0.8]
ISO_BASES = ['aniso_str', 'aniso_shr', 'disorder_hi']
# Test C: flat one modulus, directional the other
E0, EAMP, NU0, NUAMP = 1.0, 0.40, 0.20, 0.30
C_TOPOS = ['regular', 'aniso_str']
C_CASES = [
    ('E-anisotropic / nu-isotropic', np.full_like(TH, NU0), E0 * (1 + EAMP * np.cos(2 * TH))),
    ('nu-anisotropic / E-isotropic', NU0 + NUAMP * np.cos(2 * TH), np.full_like(TH, E0)),
]
LBL = {t[0]: t[1] for t in C.TOPOS}


def design(topo, objectives):
    """Design k on `topo` for `objectives`; return (k, geo, C6_per, dict of solver/sim ν(θ) & E(θ))."""
    prob, geo = C.make_case(topo, N)
    r = C.optimize(prob, objectives, mode='k', n_iter=NITER, reg=REG, verbose=False)
    snu, sE = C.nu_E_theta(C.solver_region_C6(prob, r['k']), TH)
    C.apply_k_to_geo(geo, r['k'])
    C6_per = C.sim_per_triangle_C6(geo)
    mnu, mE = C.nu_E_theta(C.region_phys_C6(geo, C6_per, None), TH)
    return r['k'], geo, C6_per, dict(solv_nu=snu, solv_E=sE, sim_nu=mnu, sim_E=mE)


def main():
    d = C.savedir(CASE); ndir = C.networks_dir(CASE)
    csv_rows = []
    detail = []

    # ---------------- TEST A : program ν(θ) profiles ----------------
    progA = {}
    for pkey, plabel, pvals in PROFILES:
        for topo, _, _ in C.TOPOS:
            k, geo, C6, res = design(topo, [C.Objective('nu_theta', pvals)])
            progA[(pkey, topo)] = res
            se = np.abs(res['sim_nu'] - pvals).max(); ve = np.abs(res['solv_nu'] - pvals).max()
            csv_rows.append(('program', topo, pkey, f'{ve:.3f}', f'{se:.3f}'))
            C.save_network(os.path.join(ndir, f'A_{pkey}_{topo}.npz'), geo, k, C6,
                           test='A_program', topo=topo, N=N, profile=pkey, target=pvals.tolist(),
                           sim_nu=res['sim_nu'].tolist(), sim_maxerr=float(se))
            print(f"  [A] {topo:12s} {pkey:10s} | solver maxerr={ve:.3f} sim maxerr={se:.3f}", flush=True)
            if pkey == 'dir_aux' and topo == 'regular':
                detail.append(('regular → directional-auxetic', geo, k, C6, None))

    # ---------------- TEST B : isotropize to chosen ν0 ----------------
    progB = {}
    for base in ISO_BASES:
        _, geo0 = C.make_case(base, N)
        C.apply_k_to_geo(geo0, np.ones(len(geo0['bond_u'])))
        progB[base] = {'base': C.nu_E_theta(C.sim_region_C6(geo0, None), TH)[0]}
        for nu0 in ISO_TGTS:
            k, geo, C6, res = design(base, [C.Objective('nu_theta', np.full_like(TH, nu0))])
            sim = res['sim_nu']; flat = np.abs(sim - sim.mean()).max()
            progB[base][nu0] = (sim, float(sim.mean()), flat)
            csv_rows.append(('isotropize', base, f'nu0={nu0:+.2f}',
                             f'{np.abs(res["solv_nu"]-nu0).max():.3f}', f'{np.abs(sim-nu0).max():.3f}'))
            C.save_network(os.path.join(ndir, f'B_{base}_nu{nu0:+.2f}.npz'), geo, k, C6,
                           test='B_isotropize', base=base, N=N, nu0=float(nu0),
                           sim_nu=sim.tolist(), sim_mean=float(sim.mean()), flatness=float(flat))
            print(f"  [B] {base:12s} nu0={nu0:+.2f} | sim mean={sim.mean():+.3f} flatness={flat:.3f}", flush=True)
            if base == 'aniso_str' and abs(nu0 - 1 / 3.) < 1e-6:
                detail.append(('aniso_str → isotropic 1/3', geo, k, C6, None))

    # ---------------- TEST C : independent E / ν anisotropy ----------------
    progC = {}                                          # (case_label, topo) -> res
    for clabel, nu_tgt, E_tgt in C_CASES:
        tag = clabel.replace(' / ', '_').replace(' ', '')
        for topo in C_TOPOS:
            k, geo, C6, res = design(topo, [C.Objective('nu_theta', nu_tgt, weight=4.0),
                                            C.Objective('E_theta', E_tgt, weight=1.0)])
            progC[(clabel, topo)] = res
            csv_rows.append(('independent', topo, clabel,
                             f'{np.abs(res["solv_nu"]-nu_tgt).max():.3f}',
                             f'{np.abs(res["sim_nu"]-nu_tgt).max():.3f}'))
            C.save_network(os.path.join(ndir, f'C_{tag}_{topo}.npz'), geo, k, C6,
                           test='C_independent', topo=topo, N=N, case=clabel,
                           nu_target=nu_tgt.tolist(), E_target=E_tgt.tolist(),
                           sim_nu=res['sim_nu'].tolist(), sim_E=res['sim_E'].tolist())
            print(f"  [C] {topo:12s} {clabel:30s} | nu span {np.ptp(res['sim_nu']):.3f} "
                  f"E span {np.ptp(res['sim_E']):.3f}", flush=True)

    C.write_csv(os.path.join(d, f'{CASE}.csv'),
                ['test', 'topology_or_base', 'target', 'solver_maxerr', 'sim_maxerr'], csv_rows)

    # ---------------- plots : Test A ----------------
    fig, axes = plt.subplots(1, len(PROFILES), figsize=(6 * len(PROFILES), 5.4), squeeze=False)
    for ax, (pkey, plabel, pvals) in zip(axes[0], PROFILES):
        ax.plot(DEG, pvals, 'k--', lw=3, label='target ν*(θ)', zorder=5)
        ax.axhline(0, color='gray', lw=0.5, ls=':')
        for topo, _, _ in C.TOPOS:
            ax.plot(DEG, progA[(pkey, topo)]['sim_nu'], '-', color=C.TOPO_COLORS[topo], lw=1.8, label=LBL[topo])
            ax.plot(DEG, progA[(pkey, topo)]['solv_nu'], ':', color=C.TOPO_COLORS[topo], lw=1.0, alpha=0.7)
        ax.set_title(plabel, fontsize=10); ax.set_xlabel('loading angle θ (deg)')
        ax.set_ylabel('ν(θ)'); ax.set_xlim(0, 180); ax.grid(alpha=0.3); ax.legend(fontsize=7)
    fig.suptitle(f'{CASE} — TEST A: PROGRAM ν(θ) (solid = SIMULATION, dotted = solver, dashed = target); N={N}',
                 fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(d, f'{CASE}_program.png'), dpi=150, bbox_inches='tight'); plt.close()
    print('saved program')

    # ---------------- plots : Test B ----------------
    cmap = plt.cm.coolwarm_r; norm = plt.Normalize(min(ISO_TGTS), max(ISO_TGTS))
    fig, axes = plt.subplots(1, len(ISO_BASES), figsize=(6 * len(ISO_BASES), 5.4), squeeze=False)
    for ax, base in zip(axes[0], ISO_BASES):
        ax.plot(DEG, progB[base]['base'], color='0.4', ls=':', lw=2.5, label='base (uniform k)')
        for nu0 in ISO_TGTS:
            sim, mean, flat = progB[base][nu0]
            ax.axhline(nu0, color=cmap(norm(nu0)), lw=0.8, ls='--', alpha=0.5)
            ax.plot(DEG, sim, '-', color=cmap(norm(nu0)), lw=1.8, label=f'→ ν0={nu0:+.2f}')
        ax.set_title(f'{LBL[base]}', fontsize=10); ax.set_xlabel('loading angle θ (deg)')
        ax.set_ylabel('ν(θ) (simulated)'); ax.set_xlim(0, 180); ax.set_ylim(-1, 1)
        ax.grid(alpha=0.3); ax.legend(fontsize=7, ncol=2)
    fig.suptitle(f'{CASE} — TEST B: ISOTROPIZE to a chosen flat ν0 (dashed = target level); N={N}', fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(d, f'{CASE}_isotropize.png'), dpi=150, bbox_inches='tight'); plt.close()
    print('saved isotropize')

    # ---------------- plots : Test B parity ----------------
    fig, ax = plt.subplots(figsize=(6.5, 6.2))
    ax.plot([-1, 1], [-1, 1], 'k--', lw=1, alpha=0.5, label='ideal (isotropic at ν0)')
    for base in ISO_BASES:
        means = [progB[base][n][1] for n in ISO_TGTS]; flats = [progB[base][n][2] for n in ISO_TGTS]
        ax.errorbar(ISO_TGTS, means, yerr=flats, marker='o', ms=5, lw=1.4, capsize=3, label=LBL[base])
    ax.set_xlabel('target ν0'); ax.set_ylabel('achieved mean ν  (bar = angular spread / non-flatness)')
    ax.set_title(f'{CASE} — TEST B: closeness to isotropic-ν0 per base (N={N})')
    ax.grid(alpha=0.3); ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(d, f'{CASE}_isotropize_parity.png'), dpi=150, bbox_inches='tight'); plt.close()
    print('saved isotropize_parity')

    # ---------------- plots : Test C (independent E / ν) ----------------
    fig, axes = plt.subplots(len(C_CASES), 2, figsize=(13, 5.4 * len(C_CASES)), squeeze=False)
    for row, (clabel, nu_tgt, E_tgt) in enumerate(C_CASES):
        an, aE = axes[row]
        an.plot(DEG, nu_tgt, 'k--', lw=2.5, label='target ν(θ)'); an.axhline(0, color='gray', lw=0.5, ls=':')
        aE.plot(DEG, E_tgt, 'k--', lw=2.5, label='target E(θ)')
        for topo in C_TOPOS:
            res = progC[(clabel, topo)]
            an.plot(DEG, res['sim_nu'], '-', color=C.TOPO_COLORS[topo], lw=1.8, label=LBL[topo])
            aE.plot(DEG, res['sim_E'], '-', color=C.TOPO_COLORS[topo], lw=1.8, label=LBL[topo])
        for a, ttl in ((an, f'{clabel}: ν(θ)'), (aE, f'{clabel}: E(θ)')):
            a.set_title(ttl, fontsize=10); a.set_xlabel('loading angle θ (deg)'); a.set_xlim(0, 180)
            a.grid(alpha=0.3); a.legend(fontsize=8)
        an.set_ylabel('ν(θ)'); aE.set_ylabel('E(θ)')
    fig.suptitle(f'{CASE} — TEST C: INDEPENDENT E / ν anisotropy (one modulus flat, the other directional); '
                 f'solid = SIMULATION, dashed = target; N={N}', fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(d, f'{CASE}_independent_Enu.png'), dpi=150, bbox_inches='tight'); plt.close()
    print('saved independent_Enu')

    # ---------------- detail ----------------
    if detail:
        C.design_detail_figure(os.path.join(d, f'{CASE}_detail.png'), detail,
                               f'{CASE}: representative designs — rigidity k, local ν, local E')
    print('done')


if __name__ == '__main__':
    main()
