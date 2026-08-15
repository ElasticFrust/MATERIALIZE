"""
Good working regimes for the directional-design tests, focused on the ORDERED (regular) case.
Three parts:
  ISO   — isotropise to a chosen flat nu0: which nu0 each topology can reach (mean + flatness).
          Reuses the saved Test-B networks (aniso_str/shr, disorder_hi) and runs the two missing
          topologies (regular, disorder_lo).
  ANISO — program a REALIZABLE anisotropic response (target = the full tensor of the aniso_str
          lattice) on every topology: how close each gets (ν(θ) error). Realizable by construction.
  INDEP — independent E/nu on separate directions: find a good working case by sweeping the weight
          of the 'flat' modulus, on regular + disorder_hi (both directions of the decoupling).
Saves plots + networks + a CSV of the regime numbers.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import _common as C

TH = C.ANG; DEG = np.degrees(TH)
N, NITER, REG = 10, 130, 1e-4
ISO_TGTS = [0.80, 0.50, 1 / 3., 0.00, -0.20, -0.50, -0.80]
B_ND = os.path.join(HERE, '..', 'anisotropy', 'networks')
csv = []


def nd():
    d = os.path.join(HERE, 'networks'); os.makedirs(d, exist_ok=True); return d


def iso_regime():
    """achieved mean nu + flatness vs target nu0, per topology."""
    data = {}
    for base in ['aniso_str', 'aniso_shr', 'disorder_hi']:                 # from saved Test B
        row = []
        for n in ISO_TGTS:
            _, _, _, meta = C.load_network(os.path.join(B_ND, f'B_{base}_nu{n:+.2f}.npz'))
            row.append((n, meta['sim_mean'], meta['flatness']))
        data[base] = row
    for topo in ['regular', 'disorder_lo']:                               # run the two missing
        row = []
        for n in ISO_TGTS:
            prob, geo = C.make_case(topo, N)
            r = C.optimize(prob, [C.Objective('nu', n)], mode='k', n_iter=NITER, reg=REG, verbose=False)
            C.apply_k_to_geo(geo, r['k']); C6 = C.sim_per_triangle_C6(geo)
            nu = C.nu_E_theta(C.sim_bulk_C6(geo), TH)[0]
            row.append((n, float(nu.mean()), float(np.abs(nu - nu.mean()).max())))
            C.save_network(os.path.join(nd(), f'iso_{topo}_nu{n:+.2f}.npz'), geo, r['k'], C6,
                           regime='isotropize', topo=topo, nu0=float(n), mean=float(nu.mean()))
            print(f"  ISO {topo:12s} nu0={n:+.2f} mean={nu.mean():+.3f} flat={row[-1][2]:.3f}", flush=True)
        data[topo] = row
    for t, row in data.items():
        for (n, m, f) in row:
            csv.append(('isotropize', t, f'{n:+.2f}', f'{m:+.3f}', f'{f:.3f}'))

    order = ['regular', 'disorder_lo', 'disorder_hi', 'aniso_shr', 'aniso_str']
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.4))
    ax[0].plot([-.85, .85], [-.85, .85], 'k--', lw=1, alpha=.5, label='ideal')
    for t in order:
        n0 = [x[0] for x in data[t]]; mn = [x[1] for x in data[t]]; fl = [x[2] for x in data[t]]
        ax[0].errorbar(n0, mn, yerr=fl, marker='o', ms=4, lw=1.3, capsize=3, label=t)
        ax[1].plot(n0, fl, '-o', ms=4, label=t)
    ax[0].set_xlabel('target ν0'); ax[0].set_ylabel('achieved mean ν (bar = non-flatness)')
    ax[0].set_title('ISOTROPISE regime — reachability per topology'); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    ax[1].axhline(0.05, color='gray', ls=':', label='flat threshold 0.05')
    ax[1].set_xlabel('target ν0'); ax[1].set_ylabel('angular spread (flatness)'); ax[1].set_yscale('log')
    ax[1].set_title('how ISOTROPIC (lower = better)'); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(HERE, 'regime_isotropize.png'), dpi=150, bbox_inches='tight')
    plt.close(); print('saved regime_isotropize.png')


def aniso_regime():
    """Program a REALIZABLE anisotropy (target = aniso_str tensor) on every topology."""
    tgt = C.reference_C6('aniso')                                          # realizable anisotropic tensor
    tgt_nu = C.nu_E_theta(tgt, TH)[0]
    fig, ax = plt.subplots(figsize=(8, 5.6))
    ax.plot(DEG, tgt_nu, 'k--', lw=3, label='target ν(θ) (aniso_str tensor)', zorder=5)
    for topo in C.TOPO_IDS:
        prob, geo = C.make_case(topo, N)
        r = C.optimize(prob, [C.Objective('tensor', tgt)], mode='k', n_iter=NITER, reg=REG, verbose=False)
        C.apply_k_to_geo(geo, r['k']); C6 = C.sim_per_triangle_C6(geo)
        nu = C.nu_E_theta(C.sim_bulk_C6(geo), TH)[0]
        err = np.abs(nu - tgt_nu).max()
        csv.append(('anisotropic', topo, 'aniso_tensor', f'{err:.3f}', ''))
        C.save_network(os.path.join(nd(), f'aniso_{topo}.npz'), geo, r['k'], C6, regime='anisotropic',
                       topo=topo, maxerr=float(err))
        ax.plot(DEG, nu, '-', color=C.TOPO_COLORS[topo], lw=1.8, label=f'{topo} (err={err:.2f})')
        print(f"  ANISO {topo:12s} max|dnu(theta)|={err:.3f}", flush=True)
    ax.set_xlabel('θ (deg)'); ax.set_ylabel('ν(θ) (simulated)'); ax.set_xlim(0, 180)
    ax.set_title('ANISOTROPIC regime — realise a prescribed (realizable) ν(θ) per topology')
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(HERE, 'regime_anisotropic.png'), dpi=150, bbox_inches='tight')
    plt.close(); print('saved regime_anisotropic.png')


def indep_regime():
    """Independent E/nu: sweep the weight of the FLAT modulus to find a good decoupling case."""
    NU0, NUAMP, E0, EAMP = 0.20, 0.30, 1.0, 0.40
    setups = [('nu-directional / E-flat', NU0 + NUAMP * np.cos(2 * TH), np.full_like(TH, E0), 'E'),
              ('E-directional / nu-flat', np.full_like(TH, NU0), E0 * (1 + EAMP * np.cos(2 * TH)), 'nu')]
    weights = [1.0, 4.0, 10.0]                                            # weight on the FLAT modulus
    fig, axes = plt.subplots(len(setups), 2, figsize=(13, 5.4 * len(setups)), squeeze=False)
    for row, (label, nu_t, E_t, flatq) in enumerate(setups):
        an, aE = axes[row]
        an.plot(DEG, nu_t, 'k--', lw=2, label='target ν'); aE.plot(DEG, E_t, 'k--', lw=2, label='target E')
        for w in weights:
            wn = w if flatq == 'nu' else 1.0; we = w if flatq == 'E' else 1.0
            prob, geo = C.make_case('regular', N)
            r = C.optimize(prob, [C.Objective('nu_theta', nu_t, weight=4.0 if flatq == 'E' else wn),
                                  C.Objective('E_theta', E_t, weight=4.0 if flatq == 'nu' else we)],
                           mode='k', n_iter=NITER, reg=REG, verbose=False)
            C.apply_k_to_geo(geo, r['k']); C6 = C.sim_per_triangle_C6(geo)
            snu, sE = C.nu_E_theta(C.sim_bulk_C6(geo), TH)
            flat = np.ptp(snu) if flatq == 'nu' else np.ptp(sE)
            csv.append(('independent', f'regular:{label}', f'wflat={w}', f'{flat:.3f}', ''))
            an.plot(DEG, snu, '-', lw=1.5, label=f'wflat={w:g} (νspan {np.ptp(snu):.2f})')
            aE.plot(DEG, sE, '-', lw=1.5, label=f'wflat={w:g} (Espan {np.ptp(sE):.2f})')
            C.save_network(os.path.join(nd(), f'indep_{flatq}flat_w{w:g}.npz'), geo, r['k'], C6,
                           regime='independent', setup=label, wflat=float(w))
            print(f"  INDEP {label:26s} wflat={w:g} | nu span={np.ptp(snu):.2f} E span={np.ptp(sE):.2f}", flush=True)
        for a, t in ((an, f'{label}: ν(θ)'), (aE, f'{label}: E(θ)')):
            a.set_title(t, fontsize=10); a.set_xlabel('θ (deg)'); a.set_xlim(0, 180); a.grid(alpha=.3); a.legend(fontsize=7)
    fig.suptitle('INDEPENDENT E/ν regime (regular) — sweep the FLAT modulus weight for clean decoupling', fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(HERE, 'regime_independent.png'), dpi=150, bbox_inches='tight')
    plt.close(); print('saved regime_independent.png')


def main():
    iso_regime(); aniso_regime(); indep_regime()
    C.write_csv(os.path.join(HERE, 'working_regimes.csv'),
                ['regime', 'topology', 'setting', 'metric1', 'metric2'], csv)
    print('done')


if __name__ == '__main__':
    main()
