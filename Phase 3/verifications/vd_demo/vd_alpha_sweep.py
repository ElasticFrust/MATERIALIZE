"""
Regular-lattice VD: homogenised ν and E vs the VD contrast alpha (0..30), at fixed virtual-distortion
eta = 0.1 and 0.15. Same construction as vd_sweep_sim (regular geometry kept, connectivity fixed, no
re-triangulation): a virtual copy is displaced by eta*(cos,sin) per vertex, its edge lengths l set the
rigidities k = 1 + tanh(alpha*(l - 1)) on the REAL regular lattice, and a PBC simulation gives ν, E.
Here eta is fixed and alpha is swept (alpha->inf makes k near-binary {0,2}).
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', '..', 'verification_tools'))
sys.path.insert(0, os.path.join(HERE, '..', '..', '..', 'Phase 2'))
import test_cluster_VD as VD
import test_cluster_rigidity as TR
import physical_homog as PH

N = 14
ETAS = [0.10, 0.15]
ALPHAS = np.linspace(0.0, 30.0, 31)
SEEDS = [0, 1, 2]


def main():
    res = {}
    for eta in ETAS:
        NU = np.zeros((len(ALPHAS), len(SEEDS))); EE = np.zeros_like(NU)
        for j, s in enumerate(SEEDS):
            reg = VD.build_geometry(N, 0.0, s)                 # real regular lattice (l0 = 1)
            virt = VD.build_geometry(N, eta, s)                # virtual copy, SAME connectivity
            Lv = np.sqrt((virt['bond_R'] ** 2).sum(1))        # virtual edge lengths
            free = np.arange(2, 2 * len(reg['pts']))
            for i, a in enumerate(ALPHAS):
                k = 1.0 + np.tanh(a * (Lv - 1.0))             # VD rigidity on the regular lattice
                g = dict(reg); g['bond_k'] = k; g['tri_k'] = k[reg['tri_bond']]
                NU[i, j], EE[i, j] = PH.sim_nuE(g, free, TR.assemble_K_faff)
        res[eta] = (NU, EE)
        print(f"eta={eta}: nu(alpha=0)={NU[0].mean():+.3f}  nu(alpha=30)={NU[-1].mean():+.3f}  "
              f"min nu={NU.mean(1).min():+.3f}", flush=True)
    np.savez(os.path.join(HERE, 'vd_alpha_sweep.npz'), alphas=ALPHAS, etas=ETAS, seeds=SEEDS,
             **{f'nu_{e}': res[e][0] for e in ETAS}, **{f'E_{e}': res[e][1] for e in ETAS})

    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
    cols = {0.10: '#1f77b4', 0.15: '#d62728'}
    for eta in ETAS:
        NU, EE = res[eta]
        m, sd = NU.mean(1), NU.std(1)
        ax[0].fill_between(ALPHAS, m - sd, m + sd, color=cols[eta], alpha=0.2)
        ax[0].plot(ALPHAS, m, '-o', color=cols[eta], ms=3, label=f'η={eta}')
        m, sd = EE.mean(1), EE.std(1)
        ax[1].fill_between(ALPHAS, m - sd, m + sd, color=cols[eta], alpha=0.2)
        ax[1].plot(ALPHAS, m, '-o', color=cols[eta], ms=3, label=f'η={eta}')
    ax[0].axhline(0, color='k', lw=0.6); ax[0].axhline(1 / 3, color='gray', ls=':', lw=1, label='ν=1/3')
    ax[0].set_xlabel('VD contrast α'); ax[0].set_ylabel('homogenised ν (PBC simulation)')
    ax[0].set_title('ν vs α'); ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[1].set_xlabel('VD contrast α'); ax[1].set_ylabel('homogenised E'); ax[1].set_title('E vs α')
    ax[1].legend(); ax[1].grid(alpha=0.3)
    fig.suptitle(f'Regular-lattice VD (k on regular geometry) — ν, E vs contrast α  (N={N}, {len(SEEDS)} seeds)',
                 fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(HERE, 'vd_alpha_sweep.png'), dpi=150, bbox_inches='tight'); plt.close()
    print('saved vd_alpha_sweep.png + vd_alpha_sweep.npz')


if __name__ == '__main__':
    main()
