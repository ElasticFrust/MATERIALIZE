"""
Case anisotropy (DIRECTIONAL nu(theta)). Two cross-target designs per topology x size:
  ISOTROPIZE   — design k toward an ISOTROPIC reference (regular lattice, nu=1/3 at all angles);
                 on the anisotropic bases this cancels the geometric anisotropy (nu(theta) -> flat).
  ANISOTROPIZE — design k toward an ANISOTROPIC reference; on any base this induces that anisotropy.
nu(theta) comes from the full simulated tensor (shape, not just a scalar). Outputs:
  - anisotropy.csv            : topology, size, isotropize max|dnu|, anisotropize max|dnu|
  - anisotropy_summary.png    : overlaid ISOTROPIZE (base dotted -> flat) + ANISOTROPIZE (-> target)
  - anisotropy_bytopo.png     : small multiples, per-topology nu(theta): base / iso-designed / ani-designed
  - anisotropy_detail.png     : 5-topology grid [rigidity k | local nu | local E] of the ISOTROPIZED design
  - anisotropy_<topology>.png : per-topology detail
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _common as C

CASE, REG = 'anisotropy', 0.002
THETA = np.linspace(0, np.pi, 61); DEG = np.degrees(THETA)


def main():
    d = C.savedir(CASE)
    iso_ref, ani_ref = C.reference_C6('iso'), C.reference_C6('aniso')
    iso_nu = C.nu_E_theta(iso_ref, THETA)[0]
    ani_nu = C.nu_E_theta(ani_ref, THETA)[0]
    rec = {}                        # (topo,N) -> base, iso, ani  (sim nu(theta))
    detail = {}
    csv_rows = []
    Nplot = C.SIZES[-1]
    for topo, label, _ in C.TOPOS:
        for N in C.SIZES:
            prob, geo = C.make_case(topo, N)
            base = C.nu_E_theta(C.sim_region_C6(geo, None), THETA)[0]
            r_iso = C.optimize(prob, [C.Objective('tensor', iso_ref)], mode='k', n_iter=150,
                               reg=REG, verbose=False)
            C.apply_k_to_geo(geo, r_iso['k'])
            iso = C.nu_E_theta(C.sim_region_C6(geo, None), THETA)[0]
            C6_iso = C.sim_per_triangle_C6(geo)
            prob2, geo2 = C.make_case(topo, N)
            r_ani = C.optimize(prob2, [C.Objective('tensor', ani_ref)], mode='k', n_iter=150,
                               reg=REG, verbose=False)
            C.apply_k_to_geo(geo2, r_ani['k'])
            ani = C.nu_E_theta(C.sim_region_C6(geo2, None), THETA)[0]
            rec[(topo, N)] = dict(base=base, iso=iso, ani=ani)
            csv_rows.append((topo, N, f'{np.abs(iso - iso_nu).max():.4f}',
                             f'{np.abs(ani - ani_nu).max():.4f}'))
            print(f"  {topo:12s} N={N} | iso->flat err={np.abs(iso-iso_nu).max():.3f}  "
                  f"ani->target err={np.abs(ani-ani_nu).max():.3f}", flush=True)
            if N == Nplot:
                detail[topo] = (f'{label} → ISO', geo, r_iso['k'], C6_iso, None)

    C.write_csv(os.path.join(d, f'{CASE}.csv'),
                ['topology', 'half_size', 'isotropize_max_dnu(theta)', 'anisotropize_max_dnu(theta)'],
                csv_rows)
    print(f"\n{'topology':12s} {'N':>3} | {'iso err':>8} {'ani err':>8}  (max|dnu(theta)| vs target)")
    for r in csv_rows:
        print(f"{r[0]:12s} {r[1]:>3} | {r[2]:>8} {r[3]:>8}")

    # ---- overlaid summary (2 panels) ----
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.8))
    ax[0].axhline(1/3, color='k', ls='--', lw=2, label='isotropic target (ν=1/3)')
    for topo, label, _ in C.TOPOS:
        v = rec[(topo, Nplot)]
        ax[0].plot(DEG, v['iso'], '-', color=C.TOPO_COLORS[topo], lw=2, label=f'{label} → iso')
        if topo in ('aniso_str', 'aniso_shr'):
            ax[0].plot(DEG, v['base'], ':', color=C.TOPO_COLORS[topo], lw=1.5, alpha=0.8)
    ax[0].set_xlabel('loading angle θ (deg)'); ax[0].set_ylabel('ν(θ) (simulated)')
    ax[0].set_title('ISOTROPIZE: dotted = anisotropic base, solid = designed → flat')
    ax[0].legend(fontsize=7); ax[0].grid(alpha=0.3)
    ax[1].plot(DEG, ani_nu, 'k--', lw=2.5, label='anisotropic target ν(θ)')
    for topo, label, _ in C.TOPOS:
        ax[1].plot(DEG, rec[(topo, Nplot)]['ani'], '-', color=C.TOPO_COLORS[topo], lw=1.5, label=label)
    ax[1].set_xlabel('loading angle θ (deg)'); ax[1].set_ylabel('ν(θ) (simulated)')
    ax[1].set_title('ANISOTROPIZE: all topologies → the same anisotropic target')
    ax[1].legend(fontsize=7); ax[1].grid(alpha=0.3)
    fig.suptitle(f'{CASE}: directional ν(θ) design verified against simulation (N={Nplot})', fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(d, f'{CASE}_summary.png'), dpi=150, bbox_inches='tight')
    plt.close(); print('saved summary')

    # ---- small multiples: per-topology nu(theta) (base / iso-designed / ani-designed) ----
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, (topo, label, _) in zip(axes.ravel(), C.TOPOS):
        v = rec[(topo, Nplot)]
        ax.axhline(1/3, color='gray', ls=':', lw=1)
        ax.plot(DEG, ani_nu, 'k--', lw=1.2, alpha=0.7, label='aniso target')
        ax.plot(DEG, v['base'], ':', color='0.4', lw=1.5, label='base (uniform k)')
        ax.plot(DEG, v['iso'], '-', color='#1f77b4', lw=2, label='→ isotropic')
        ax.plot(DEG, v['ani'], '-', color='#d62728', lw=1.6, label='→ anisotropic')
        ax.set_title(label, fontsize=10); ax.set_xlabel('θ (deg)'); ax.set_ylabel('ν(θ)')
        ax.legend(fontsize=6); ax.grid(alpha=0.3)
    axes.ravel()[-1].axis('off')
    fig.suptitle(f'{CASE}: per-topology ν(θ) — base, isotropized, anisotropized (simulated)', fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(d, f'{CASE}_bytopo.png'), dpi=150, bbox_inches='tight')
    plt.close(); print('saved bytopo')

    entries = [(*detail[t[0]][:4], detail[t[0]][4]) for t in C.TOPOS if t[0] in detail]
    C.design_detail_figure(os.path.join(d, f'{CASE}_detail.png'), entries,
                           f'{CASE}: ISOTROPIZED designs (largest size) — rigidity k, local ν, local E')
    C.design_detail_per_topology(d, CASE, entries, f'{CASE} — {{name}} (designed toward isotropic)')
    print('saved detail grid + per-topology')


if __name__ == '__main__':
    main()
