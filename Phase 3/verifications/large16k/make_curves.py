"""
Target-vs-achieved DIRECTIONAL curves for the large-16k designs (loaded from saved networks).
Shows the prescribed ν(θ)/E(θ) target (dashed) against the simulated response (solid, regular vs
disordered) for the directional design types: iso (flat ν0=-0.5), aniso (realizable anisotropic
tensor), indep (flat ν + 2-fold E).
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
ND = os.path.join(HERE, 'networks')
TOPOS = ['regular', 'disorder_hi']
COL = {'regular': '#000000', 'disorder_hi': '#d62728'}


def ach(kind, topo):
    geo, k, C6, meta = C.load_network(os.path.join(ND, f'{kind}__{topo}.npz'))
    return C.nu_E_theta(C.region_phys_C6(geo, C6, None), TH)      # (nu(θ), E(θ))


def main():
    iso_tgt = np.full_like(TH, -0.5)
    aniso_tgt = C.nu_E_theta(C.reference_C6('aniso'), TH)[0]
    indep_nu_tgt = np.full_like(TH, 0.20)
    indep_E_tgt = 1.0 * (1 + 0.40 * np.cos(2 * TH))

    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    # iso: nu(θ) target flat -0.5
    ax[0].plot(DEG, iso_tgt, 'k--', lw=2.5, label='target ν0=-0.5')
    for t in TOPOS:
        ax[0].plot(DEG, ach('iso', t)[0], '-', color=COL[t], lw=1.8, label=f'{t}')
    ax[0].set_title('ISO — ν(θ): isotropic auxetic'); ax[0].axhline(0, color='gray', lw=.5, ls=':')
    # aniso: nu(θ) target = realizable anisotropic
    ax[1].plot(DEG, aniso_tgt, 'k--', lw=2.5, label='target ν(θ) (aniso tensor)')
    for t in TOPOS:
        ax[1].plot(DEG, ach('aniso', t)[0], '-', color=COL[t], lw=1.8, label=f'{t}')
    ax[1].set_title('ANISO — ν(θ): realizable anisotropy')
    # indep: nu(θ) flat + E(θ) 2-fold
    ax[2].plot(DEG, indep_nu_tgt, 'k--', lw=2.5, label='target ν flat')
    ax[3].plot(DEG, indep_E_tgt, 'k--', lw=2.5, label='target E 2-fold')
    for t in TOPOS:
        nu, E = ach('indep', t)
        ax[2].plot(DEG, nu, '-', color=COL[t], lw=1.8, label=f'{t}')
        ax[3].plot(DEG, E, '-', color=COL[t], lw=1.8, label=f'{t}')
    ax[2].set_title('INDEP — ν(θ) (target flat)'); ax[3].set_title('INDEP — E(θ) (target 2-fold)')
    for a in ax:
        a.set_xlabel('θ (deg)'); a.set_xlim(0, 180); a.grid(alpha=.3); a.legend(fontsize=8)
    ax[0].set_ylabel('ν(θ)'); ax[3].set_ylabel('E(θ)')
    fig.suptitle('LARGE 16k designs — target (dashed) vs achieved (solid, SIMULATION) directional response',
                 fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(HERE, 'large16k_curves.png'), dpi=150, bbox_inches='tight'); plt.close()
    print('saved large16k_curves.png')


if __name__ == '__main__':
    main()
