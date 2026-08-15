"""
Single VD (virtual-distortion) run on a REGULAR lattice — the geometry stays a perfect equilateral
triangular lattice (every actual bond length = 1); only the RIGIDITIES carry the disorder. A virtual
copy of the nodes is displaced by eta (η=0.45), the resulting virtual bond lengths |R| are read off,
and k = 1 + tanh(a·(|R|−1)) (a=5) is put on the UNCHANGED regular lattice. We then simulate and map
the local ν and local E.

Outputs (kept): vd_regular.png  [VD rigidity k | local ν | local E]  and  vd_regular.npz (network).

CORRECTION: at a=5 the regular-lattice VD is only mildly ν-reducing (this file, nu=+0.10) — but that
is because a=5 is BELOW threshold. Sweeping the contrast (vd_alpha_sweep.py, 10-seed mean) shows the
regular lattice DOES become auxetic at strong contrast (nu crosses 0 near a~15-21, reaching ~-0.07
(eta=0.1) / -0.10 (eta=0.15) at a=30; large seed variance). So rigidity contrast alone on a regular
lattice is auxetic given enough contrast; the forward solver matches the simulation there to <0.01.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _common as C

HERE = os.path.dirname(os.path.abspath(__file__))
ALPHA, ETA, HALF, SEED = 5.0, 0.45, 12, 0


def main():
    geo = C.make_lattice(1.0, 1.0, half=HALF, eta=0.0)          # REGULAR geometry (unchanged)
    Lreg = np.sqrt((geo['bond_R'] ** 2).sum(1))
    # virtual distortion: displace a copy of the nodes by eta, read the virtual bond lengths
    rng = np.random.default_rng(SEED)
    ang = rng.uniform(0, 2 * np.pi, len(geo['pts']))
    pert = ETA * np.stack([np.cos(ang), np.sin(ang)], 1)
    Rvirt = geo['bond_R'] + pert[geo['bond_v']] - pert[geo['bond_u']]
    Lvirt = np.sqrt((Rvirt ** 2).sum(1))
    k = 1.0 + np.tanh(ALPHA * (Lvirt - 1.0))                    # VD rigidity on the regular lattice
    print(f"regular bond length: {Lreg.min():.3f}..{Lreg.max():.3f} (all ~1)")
    print(f"virtual (eta={ETA}) bond length: {Lvirt.min():.3f}..{Lvirt.max():.3f}")
    print(f"VD rigidity k=1+tanh({ALPHA}*(|R|-1)): {k.min():.3f}..{k.max():.3f}")

    C.apply_k_to_geo(geo, k)
    C6 = C.sim_per_triangle_C6(geo)
    nu_g, E_g = C.c6_nuE(C.sim_bulk_C6(geo))
    print(f"homogenised (regular geometry + VD rigidity): nu={nu_g:+.3f}  E={E_g:.3f}")

    C.save_network(os.path.join(HERE, 'vd_regular.npz'), geo, k, C6, kind='VD_on_regular',
                   alpha=ALPHA, eta=ETA, half=HALF, seed=SEED, nu=float(nu_g), E=float(E_g))

    fig, ax = plt.subplots(1, 3, figsize=(19, 6.4))
    lc = C.draw_network(ax[0], geo, k, cmap='viridis')
    plt.colorbar(lc, ax=ax[0], fraction=0.046, label='k')
    ax[0].set_title(f'VD rigidity k = 1+tanh({ALPHA:.0f}·(|R|−1))', fontsize=11)
    nu_loc = C.local_field_smooth(geo, C6, 'nu')
    pnu = C.fill_local_map(ax[1], geo, nu_loc, cmap='RdBu_r', sym=True)
    C.draw_box(ax[1], geo); plt.colorbar(pnu, ax=ax[1], fraction=0.046)
    ax[1].set_title(f'local ν  (global ν={nu_g:+.3f})', fontsize=11)
    E_loc = C.local_field_smooth(geo, C6, 'E')
    pE = C.fill_local_map(ax[2], geo, E_loc, cmap='viridis')
    C.draw_box(ax[2], geo); plt.colorbar(pE, ax=ax[2], fraction=0.046)
    ax[2].set_title(f'local E  (global E={E_g:.3f})', fontsize=11)
    fig.suptitle(f'VD on a REGULAR lattice (geometry unchanged) — a={ALPHA:.0f}, η={ETA}, '
                 f'{geo["tri_bond"].shape[0]} triangles: rigidity contrast alone → local ν, local E',
                 fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(HERE, 'vd_regular.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('saved vd_regular.png + vd_regular.npz')


if __name__ == '__main__':
    main()
