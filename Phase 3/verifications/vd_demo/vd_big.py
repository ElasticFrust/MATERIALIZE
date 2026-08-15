"""
LARGE regular-lattice VD run (~16000 triangles).
  Part A: nu, E vs contrast alpha (eta=0.1, 0.15), SIMULATION vs forward SOLVER (3 seeds).
  Part B: local nu and local E maps of one strongly-contrasted design (alpha=30, eta=0.15).
Regular geometry kept, connectivity fixed, no re-triangulation; k = 1 + tanh(alpha*(l_virt - 1)).
"""
import os, sys, time
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

N = 90                                  # 2*N^2 = 16200 triangles
ETAS = [0.10, 0.15]
ALPHAS = np.linspace(0.0, 30.0, 11)
SEEDS = [0, 1, 2]


def sweep():
    res = {}
    ntri = None
    for eta in ETAS:
        nus = np.zeros((len(ALPHAS), len(SEEDS))); Es = np.zeros_like(nus)
        nuo = np.zeros_like(nus); Eo = np.zeros_like(nus)
        for j, s in enumerate(SEEDS):
            reg = MB.build_geometry(N, 0.0, s); virt = MB.build_geometry(N, eta, s)
            ntri = len(reg['simplices'])
            Lv = np.sqrt((virt['bond_R'] ** 2).sum(1)); free = np.arange(2, 2 * len(reg['pts']))
            sv = make_solver(reg, kkt_from_tri_bond(reg['tri_bond'], reg['edge_vecs']))
            rl = torch.as_tensor(np.sqrt(reg['actual_len2']))
            for i, a in enumerate(ALPHAS):
                k = 1.0 + np.tanh(a * (Lv - 1.0))
                g = dict(reg); g['bond_k'] = k; g['tri_k'] = k[reg['tri_bond']]
                nus[i, j], Es[i, j] = PH.sim_nuE(g, free, SA.assemble_K_faff)
                out = sv.forward(torch.as_tensor(k[reg['tri_bond']]), rest_lengths=rl,
                                 method='intrinsic', physical_units=True)
                nuo[i, j], Eo[i, j] = float(out['poisson']), float(out['young'])
            print(f"  eta={eta} seed={s} done nu(a=30) sim={nus[-1,j]:+.3f} sol={nuo[-1,j]:+.3f}", flush=True)
        res[eta] = dict(nus=nus, Es=Es, nuo=nuo, Eo=Eo)
    np.savez(os.path.join(HERE, 'vd_big_sweep.npz'), alphas=ALPHAS, etas=ETAS, seeds=SEEDS, ntri=ntri,
             **{f'{q}_{e}': res[e][q] for e in ETAS for q in ('nus', 'Es', 'nuo', 'Eo')})
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.4)); cols = {0.10: '#1f77b4', 0.15: '#d62728'}
    for eta in ETAS:
        r = res[eta]
        m, sd = r['nus'].mean(1), r['nus'].std(1)
        ax[0].fill_between(ALPHAS, m - sd, m + sd, color=cols[eta], alpha=0.18)
        ax[0].plot(ALPHAS, m, '-', color=cols[eta], lw=2, label=f'η={eta} SIM')
        ax[0].plot(ALPHAS, r['nuo'].mean(1), '--o', color=cols[eta], ms=3, lw=1.3, label=f'η={eta} solver')
        ax[1].plot(ALPHAS, r['Es'].mean(1), '-', color=cols[eta], lw=2, label=f'η={eta} SIM')
        ax[1].plot(ALPHAS, r['Eo'].mean(1), '--o', color=cols[eta], ms=3, lw=1.3, label=f'η={eta} solver')
    ax[0].axhline(0, color='k', lw=0.6); ax[0].axhline(1 / 3, color='gray', ls=':', lw=1, label='ν=1/3')
    ax[0].set_xlabel('VD contrast α'); ax[0].set_ylabel('homogenised ν'); ax[0].set_title('ν vs α')
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
    ax[1].set_xlabel('VD contrast α'); ax[1].set_ylabel('homogenised E'); ax[1].set_title('E vs α')
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    fig.suptitle(f'LARGE regular-lattice VD ({ntri} triangles) — ν, E vs α: SIMULATION vs forward SOLVER '
                 f'({len(SEEDS)} seeds)', fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(HERE, 'vd_big_sweep.png'), dpi=150, bbox_inches='tight'); plt.close()
    print('saved vd_big_sweep.png')


def maps(alpha=30.0, eta=0.15, half=42, seed=0):
    sys.path.insert(0, os.path.join(HERE, '..'))
    import _common as C
    geo = C.make_lattice(1.0, 1.0, half=half, eta=0.0)          # regular geometry (tri_verts for fills)
    rng = np.random.default_rng(seed); ang = rng.uniform(0, 2 * np.pi, len(geo['pts']))
    pert = eta * np.stack([np.cos(ang), np.sin(ang)], 1)
    Lv = np.sqrt(((geo['bond_R'] + pert[geo['bond_v']] - pert[geo['bond_u']]) ** 2).sum(1))
    k = 1.0 + np.tanh(alpha * (Lv - 1.0))
    C.apply_k_to_geo(geo, k); C6 = C.sim_per_triangle_C6(geo)
    nu_g, E_g = C.c6_nuE(C.sim_bulk_C6(geo))
    print(f"  maps: {geo['tri_bond'].shape[0]} tri  global nu={nu_g:+.3f} E={E_g:.3f}", flush=True)
    C.save_network(os.path.join(HERE, 'vd_big.npz'), geo, k, C6, kind='VD_regular_large',
                   alpha=alpha, eta=eta, half=half, nu=float(nu_g), E=float(E_g))
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 6.6))
    nu_loc = C.local_field_smooth(geo, C6, 'nu'); pnu = C.fill_local_map(ax[0], geo, nu_loc, cmap='RdBu_r', sym=True)
    C.draw_box(ax[0], geo); plt.colorbar(pnu, ax=ax[0], fraction=0.046); ax[0].set_title(f'local ν  (global ν={nu_g:+.3f})')
    E_loc = C.local_field_smooth(geo, C6, 'E'); pE = C.fill_local_map(ax[1], geo, E_loc, cmap='viridis')
    C.draw_box(ax[1], geo); plt.colorbar(pE, ax=ax[1], fraction=0.046); ax[1].set_title(f'local E  (global E={E_g:.3f})')
    fig.suptitle(f'LARGE regular-lattice VD ({geo["tri_bond"].shape[0]} triangles) — local ν, local E '
                 f'(α={alpha:.0f}, η={eta})', fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(HERE, 'vd_big_maps.png'), dpi=150, bbox_inches='tight'); plt.close()
    print('saved vd_big_maps.png + vd_big.npz')


if __name__ == '__main__':
    t = time.time(); sweep(); print(f"[sweep {time.time()-t:.0f}s]")
    t = time.time(); maps(); print(f"[maps {time.time()-t:.0f}s]")
