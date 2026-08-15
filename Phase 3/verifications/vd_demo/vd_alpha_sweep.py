"""
Regular-lattice VD: homogenised ν and E vs the VD contrast alpha (0..30), at fixed virtual-distortion
eta = 0.1 and 0.15. Construction (regular geometry kept, connectivity fixed, no re-triangulation): a
virtual copy is displaced by eta*(cos,sin) per vertex, its edge lengths l set the rigidities
k = 1 + tanh(alpha*(l - 1)) on the REAL regular lattice; homogenised nu, E from BOTH:
  - the PHYSICAL PBC SIMULATION (physical_homog.sim_nuE, energy=virial ground truth), and
  - the FORWARD SOLVER (forward(method='intrinsic', physical_units=True)),
averaged over several seeds. Shows the regular lattice becomes auxetic at strong contrast, and that
the solver reproduces the simulation on this construction.
"""
import os, sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', '..', 'verification_tools'))
sys.path.insert(0, os.path.join(HERE, '..', '..', '..', 'Phase 2'))
import physical_homog as PH
from mesh_build import kkt_from_tri_bond
from solver_build import make_solver
import mesh_build as MB
import sim_assembly as SA
torch.set_default_dtype(torch.float64)

N = 14
ETAS = [0.10, 0.15]
ALPHAS = np.linspace(0.0, 30.0, 25)
SEEDS = list(range(10))


def main():
    res = {}
    for eta in ETAS:
        nus = np.zeros((len(ALPHAS), len(SEEDS))); Es = np.zeros_like(nus)         # simulation
        nuo = np.zeros_like(nus); Eo = np.zeros_like(nus)                          # solver
        for j, s in enumerate(SEEDS):
            reg = MB.build_geometry(N, 0.0, s)                # real regular lattice (l0 = 1)
            virt = MB.build_geometry(N, eta, s)              # virtual copy, SAME connectivity
            Lv = np.sqrt((virt['bond_R'] ** 2).sum(1))
            free = np.arange(2, 2 * len(reg['pts']))
            sv = make_solver(reg, kkt_from_tri_bond(reg['tri_bond'], reg['edge_vecs']))
            rl = torch.as_tensor(np.sqrt(reg['actual_len2']))
            for i, a in enumerate(ALPHAS):
                k = 1.0 + np.tanh(a * (Lv - 1.0))
                g = dict(reg); g['bond_k'] = k; g['tri_k'] = k[reg['tri_bond']]
                nus[i, j], Es[i, j] = PH.sim_nuE(g, free, SA.assemble_K_faff)
                out = sv.forward(torch.as_tensor(k[reg['tri_bond']]), rest_lengths=rl,
                                 method='intrinsic', physical_units=True)
                nuo[i, j], Eo[i, j] = float(out['poisson']), float(out['young'])
        res[eta] = dict(nus=nus, Es=Es, nuo=nuo, Eo=Eo)
        dmax = np.abs(nus.mean(1) - nuo.mean(1)).max()
        print(f"eta={eta}: sim nu(a=30)={nus[-1].mean():+.3f}  solver nu(a=30)={nuo[-1].mean():+.3f}  "
              f"max|sim-solver| nu over alpha = {dmax:.3f}", flush=True)
    np.savez(os.path.join(HERE, 'vd_alpha_sweep.npz'), alphas=ALPHAS, etas=ETAS, seeds=SEEDS,
             **{f'{q}_{e}': res[e][q] for e in ETAS for q in ('nus', 'Es', 'nuo', 'Eo')})

    fig, ax = plt.subplots(1, 2, figsize=(13, 5.4))
    cols = {0.10: '#1f77b4', 0.15: '#d62728'}
    for eta in ETAS:
        r = res[eta]
        m, sd = r['nus'].mean(1), r['nus'].std(1)
        ax[0].fill_between(ALPHAS, m - sd, m + sd, color=cols[eta], alpha=0.18)
        ax[0].plot(ALPHAS, m, '-', color=cols[eta], lw=2, label=f'η={eta} SIM')
        ax[0].plot(ALPHAS, r['nuo'].mean(1), '--o', color=cols[eta], ms=3, lw=1.3, label=f'η={eta} solver')
        me, sde = r['Es'].mean(1), r['Es'].std(1)
        ax[1].fill_between(ALPHAS, me - sde, me + sde, color=cols[eta], alpha=0.18)
        ax[1].plot(ALPHAS, me, '-', color=cols[eta], lw=2, label=f'η={eta} SIM')
        ax[1].plot(ALPHAS, r['Eo'].mean(1), '--o', color=cols[eta], ms=3, lw=1.3, label=f'η={eta} solver')
    ax[0].axhline(0, color='k', lw=0.6); ax[0].axhline(1 / 3, color='gray', ls=':', lw=1, label='ν=1/3')
    ax[0].set_xlabel('VD contrast α'); ax[0].set_ylabel('homogenised ν'); ax[0].set_title('ν vs α')
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
    ax[1].set_xlabel('VD contrast α'); ax[1].set_ylabel('homogenised E'); ax[1].set_title('E vs α')
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    fig.suptitle(f'Regular-lattice VD — ν, E vs contrast α: SIMULATION (solid+band) vs forward SOLVER '
                 f'(dashed)  (N={N}, {len(SEEDS)} seeds)', fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(HERE, 'vd_alpha_sweep.png'), dpi=150, bbox_inches='tight'); plt.close()
    print('saved vd_alpha_sweep.png + vd_alpha_sweep.npz')


if __name__ == '__main__':
    main()
