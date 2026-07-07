"""
Case anisotropy (DIRECTIONAL). For each topology x size we run TWO cross-target designs and verify
the full directional Poisson ratio nu(theta) against simulation:

  ISOTROPIZE  — design k toward an ISOTROPIC reference tensor (the regular triangular lattice,
                nu=1/3 at every angle). On the anisotropic (stretched / sheared) bases this
                CANCELS the geometric anisotropy with rigidity: nu(theta) should flatten from a
                strongly angle-dependent curve to ~1/3.
  ANISOTROPIZE — design k toward an ANISOTROPIC reference tensor (the stretched lattice). On the
                isotropic bases (regular / disordered) this INDUCES the prescribed anisotropy:
                the simulated nu(theta) should match the anisotropic target curve.

nu(theta) comes from the full simulated tensor, so shape (not just a scalar) is checked.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _common as C

CASE = 'anisotropy'
REG = 0.002
THETA = np.linspace(0, np.pi, 61)
DEG = np.degrees(THETA)


def main():
    iso_ref = C.reference_C6('iso')                          # regular lattice tensor (isotropic)
    ani_ref = C.reference_C6('aniso')                        # stretched lattice tensor (anisotropic)
    iso_nu = C.nu_E_theta(iso_ref, THETA)[0]
    ani_nu = C.nu_E_theta(ani_ref, THETA)[0]

    rec = {}                        # (topo,N) -> dict(base, iso, ani)  nu(theta) arrays
    reps = []
    Nplot = C.SIZES[-1]
    for topo, label, _ in C.TOPOS:
        for N in C.SIZES:
            prob, geo = C.make_case(topo, N)
            base = C.nu_E_theta(C.sim_region_C6(geo, None), THETA)[0]     # undesigned (uniform k)
            r_iso = C.optimize(prob, [C.Objective('tensor', iso_ref)], mode='k', n_iter=150,
                               reg=REG, verbose=False)
            C.apply_k_to_geo(geo, r_iso['k'])
            iso = C.nu_E_theta(C.sim_region_C6(geo, None), THETA)[0]
            prob2, geo2 = C.make_case(topo, N)
            r_ani = C.optimize(prob2, [C.Objective('tensor', ani_ref)], mode='k', n_iter=150,
                               reg=REG, verbose=False)
            C.apply_k_to_geo(geo2, r_ani['k'])
            ani = C.nu_E_theta(C.sim_region_C6(geo2, None), THETA)[0]
            rec[(topo, N)] = dict(base=base, iso=iso, ani=ani)
            if N == Nplot and topo in ('aniso_str', 'aniso_shr'):
                reps.append((f'{label} → ISO', geo, r_iso['k'], C.sim_per_triangle_C6(geo)))
            if N == Nplot and topo == 'regular':
                reps.append((f'{label} → ANISO', geo2, r_ani['k'], C.sim_per_triangle_C6(geo2)))
            print(f"  {topo:12s} N={N} | base nu range=[{base.min():+.2f},{base.max():+.2f}] "
                  f"iso-designed range=[{iso.min():+.2f},{iso.max():+.2f}] "
                  f"ani err(max|dnu|)={np.abs(ani - ani_nu).max():.3f}", flush=True)

    d = C.savedir(CASE)
    np.savez(os.path.join(d, 'results.npz'), theta=THETA, iso_nu=iso_nu, ani_nu=ani_nu,
             keys=[f"{t}_N{n}" for (t, n) in rec],
             base=[v['base'] for v in rec.values()], iso=[v['iso'] for v in rec.values()],
             ani=[v['ani'] for v in rec.values()])

    fig, ax = plt.subplots(1, 2, figsize=(14, 5.8))
    # panel 0: ISOTROPIZE — base (dashed) vs designed-to-iso (solid); target = flat 1/3
    ax[0].axhline(1/3, color='k', ls='--', lw=2, label='isotropic target (ν=1/3)')
    for topo, label, _ in C.TOPOS:
        v = rec[(topo, Nplot)]
        ax[0].plot(DEG, v['iso'], '-', color=C.TOPO_COLORS[topo], lw=2, label=f'{label} → iso')
        if topo in ('aniso_str', 'aniso_shr'):
            ax[0].plot(DEG, v['base'], ':', color=C.TOPO_COLORS[topo], lw=1.5, alpha=0.8)
    ax[0].set_xlabel('loading angle θ (deg)'); ax[0].set_ylabel('ν(θ) (simulated)')
    ax[0].set_title('ISOTROPIZE: design cancels geometric anisotropy\n'
                    '(dotted = anisotropic base before; solid = designed → flat)')
    ax[0].legend(fontsize=7); ax[0].grid(alpha=0.3)

    # panel 1: ANISOTROPIZE — target aniso nu(theta) vs designed-to-aniso for all topologies
    ax[1].plot(DEG, ani_nu, 'k--', lw=2.5, label='anisotropic target ν(θ)')
    for topo, label, _ in C.TOPOS:
        v = rec[(topo, Nplot)]
        ax[1].plot(DEG, v['ani'], '-', color=C.TOPO_COLORS[topo], lw=1.6, alpha=0.9, label=label)
    ax[1].set_xlabel('loading angle θ (deg)'); ax[1].set_ylabel('ν(θ) (simulated)')
    ax[1].set_title('ANISOTROPIZE: design induces the prescribed anisotropy\n'
                    '(all topologies → the same anisotropic target)')
    ax[1].legend(fontsize=7); ax[1].grid(alpha=0.3)
    fig.suptitle(f'{CASE}: directional ν(θ) design verified against simulation (N={Nplot}) — '
                 f'incl. isotropizing anisotropic networks', fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    p = os.path.join(d, f'{CASE}.png')
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print('saved', p)

    if reps:
        C.design_detail_figure(os.path.join(d, f'{CASE}_detail.png'), reps,
                               f'{CASE}: designed networks (isotropized / anisotropized, largest size) — '
                               f'rigidity k, local ν, local E')
        print('saved', os.path.join(d, f'{CASE}_detail.png'))


if __name__ == '__main__':
    main()
