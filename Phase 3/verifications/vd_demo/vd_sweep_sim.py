"""
VD stiffness on a REGULAR lattice, swept over the virtual-distortion eta — by SIMULATION (physical
PBC energy relaxation, NOT the forward solver).

Exact recipe (per spec):
  - Regular periodic triangular lattice (l0 = 1), fixed connectivity.
  - Virtual copy: displace each vertex by v -> v + eta*(cos th, sin th), th random per vertex,
    0 <= eta < 0.5. DO NOT re-triangulate — keep the original connectivity.
  - Virtual edge length l = |edge| of the deformed copy (same bond identified by connectivity).
  - Set the REAL (regular) lattice rigidities to k(l) = 1 + tanh(alpha*(l - l0)), alpha = 5.
  - Run the PBC simulation (physical_homog.sim_nuE) -> homogenised nu, E.
Sweep eta in [0, 0.49], average over seeds.

test_cluster_VD.build_geometry(N, eta, seed) does exactly this deformation (pts = ref + eta*(cos,sin))
with eta-independent connectivity, so build_geometry(N,0) is the regular lattice and build_geometry(N,eta)
is its virtual copy on the SAME bonds; set_VD reads l from the copy and writes k=1+tanh(alpha*(l-1)).
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

N, ALPHA = 14, 5.0
ETAS = np.round(np.arange(0.0, 0.491, 0.02), 3)
SEEDS = [0, 1, 2, 3, 4]


def sim_nu_E(eta, seed):
    """Return nu,E for the SAME VD k on (regular geometry [spec], deformed geometry [original])."""
    reg = VD.build_geometry(N, 0.0, seed)                  # real regular lattice (l0 = 1)
    free = np.arange(2, 2 * len(reg['pts']))
    if eta == 0.0:
        k = np.ones(len(reg['bond_u'])); virt = reg
    else:
        virt = VD.build_geometry(N, eta, seed)             # virtual copy, SAME connectivity (no retriangulation)
        VD.set_VD(virt, ALPHA)                             # k = 1 + tanh(alpha*(l_virt - 1))
        k = virt['bond_k']
    gr = dict(reg); gr['bond_k'] = k; gr['tri_k'] = k[reg['tri_bond']]     # k on REGULAR geometry (spec)
    gd = dict(virt); gd['bond_k'] = k; gd['tri_k'] = k[virt['tri_bond']]   # k on DEFORMED geometry (original)
    return (PH.sim_nuE(gr, free, TR.assemble_K_faff), PH.sim_nuE(gd, free, TR.assemble_K_faff))


def main():
    NUr = np.zeros((len(ETAS), len(SEEDS))); EEr = np.zeros_like(NUr)
    NUd = np.zeros_like(NUr); EEd = np.zeros_like(NUr)
    for i, eta in enumerate(ETAS):
        for j, s in enumerate(SEEDS):
            (NUr[i, j], EEr[i, j]), (NUd[i, j], EEd[i, j]) = sim_nu_E(eta, s)
        print(f"  eta={eta:.2f} | REGULAR nu={NUr[i].mean():+.3f}  |  DEFORMED nu={NUd[i].mean():+.3f}", flush=True)
    np.savez(os.path.join(HERE, 'vd_sweep_sim.npz'), etas=ETAS, nu_reg=NUr, E_reg=EEr,
             nu_def=NUd, E_def=EEd, alpha=ALPHA, N=N, seeds=SEEDS)

    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
    for arr, col, lab in [(NUr, '#3182bd', 'k on REGULAR geometry (this spec)'),
                          (NUd, '#e6550d', 'k on DEFORMED geometry (original a=5 plot)')]:
        m, s = arr.mean(1), arr.std(1)
        ax[0].fill_between(ETAS, m - s, m + s, color=col, alpha=0.2); ax[0].plot(ETAS, m, '-o', color=col, ms=4, label=lab)
    ax[0].axhline(0, color='k', lw=0.6); ax[0].axhline(1 / 3, color='gray', ls=':', lw=1, label='ν=1/3')
    ax[0].set_xlabel('virtual-distortion η'); ax[0].set_ylabel('homogenised ν (PBC simulation)')
    ax[0].set_title('Same VD rigidity, two geometries — where auxeticity comes from')
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
    for arr, col, lab in [(EEr, '#3182bd', 'regular geometry'), (EEd, '#e6550d', 'deformed geometry')]:
        m, s = arr.mean(1), arr.std(1)
        ax[1].fill_between(ETAS, m - s, m + s, color=col, alpha=0.2); ax[1].plot(ETAS, m, '-o', color=col, ms=4, label=lab)
    ax[1].set_xlabel('virtual-distortion η'); ax[1].set_ylabel('homogenised E'); ax[1].set_title('E vs η')
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    fig.suptitle(f'VD rigidity k=1+tanh({ALPHA:.0f}·(l_virt−1)) — SIMULATION, PBC, connectivity fixed '
                 f'(N={N}, {len(SEEDS)} seeds). REGULAR geometry stays ν>0; only the DEFORMED geometry is auxetic.',
                 fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(HERE, 'vd_sweep_sim.png'), dpi=150, bbox_inches='tight'); plt.close()
    print('saved vd_sweep_sim.png + vd_sweep_sim.npz')


if __name__ == '__main__':
    main()
