"""
What actually makes the VD networks auxetic — the GEOMETRY disorder, the RIGIDITY contrast, or their
correlation? A clean 2x2 on ONE fixed connectivity (test_cluster_VD.build_geometry has eta-independent
connectivity, so the regular and disordered meshes share bond indices and a VD k-field transfers
between them):

  geometry ∈ {regular (eta=0), disordered (eta=0.45)}  ×  rigidity ∈ {uniform k=1, VD k=1+tanh(a(|R|-1))}

The VD k is computed from the DISORDERED bond lengths (a=5) and then applied to BOTH geometries.
Homogenised ν/E from the physical PBC simulation (ground truth). This isolates whether "VD on a
regular lattice" is auxetic (rigidity alone) or whether the disordered geometry is required.

CORRECTION: the numbers below are at a=5 ONLY (regular+VD -> +0.02), which wrongly suggested the
disordered geometry is REQUIRED. It is NOT — a=5 is simply below threshold. Sweeping the contrast
(vd_alpha_sweep.py) shows regular+VD becomes auxetic at strong contrast (nu<0 for a>~12-18). So
rigidity contrast alone on a regular lattice IS auxetic given enough contrast; the disordered
geometry only lowers the contrast threshold.
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

N, ETA, A, SEED = 14, 0.45, 5.0, 0


def homog(geo, k):
    g = dict(geo); g['bond_k'] = k; g['tri_k'] = k[g['tri_bond']]
    free = np.arange(2, 2 * len(g['pts']))
    return PH.sim_nuE(g, free, TR.assemble_K_faff)


def main():
    geo_reg = VD.build_geometry(N, 0.0, seed=SEED)              # regular geometry (|R|=1)
    geo_dis = VD.build_geometry(N, ETA, seed=SEED)             # disordered geometry, SAME connectivity
    VD.set_VD(geo_dis, A)                                       # k = 1+tanh(A*(|R_dis|-1))
    k_vd = geo_dis['bond_k'].copy()
    k_1 = np.ones_like(k_vd)
    Lreg = np.sqrt((geo_reg['bond_R'] ** 2).sum(1))
    Ldis = np.sqrt((geo_dis['bond_R'] ** 2).sum(1))
    print(f"regular |R|={Lreg.min():.3f}..{Lreg.max():.3f}  disordered |R|={Ldis.min():.3f}..{Ldis.max():.3f}  "
          f"VD k={k_vd.min():.3f}..{k_vd.max():.3f}")

    rows = []
    for gname, geo in [('regular', geo_reg), ('disordered', geo_dis)]:
        for rname, k in [('uniform k=1', k_1), ('VD k', k_vd)]:
            nu, E = homog(geo, k)
            rows.append((gname, rname, nu, E))
            print(f"  geometry={gname:11s} rigidity={rname:12s} | nu={nu:+.3f}  E={E:.3f}", flush=True)

    # bar plot of nu for the 4 combos
    labels = [f'{g}\n{r}' for g, r, _, _ in rows]
    nus = [x[2] for x in rows]
    cols = ['#9ecae1', '#3182bd', '#fdae6b', '#e6550d']
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.bar(range(4), nus, color=cols)
    ax.axhline(0, color='k', lw=0.6); ax.axhline(1 / 3, color='gray', ls=':', lw=1, label='ν=1/3')
    for i, v in enumerate(nus):
        ax.text(i, v + (0.02 if v >= 0 else -0.05), f'{v:+.3f}', ha='center', fontsize=10)
    ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('homogenised ν (physical sim)')
    ax.set_title(f'What makes VD auxetic? geometry × rigidity (a={A:.0f}, η={ETA}, N={N})\n'
                 'VD-on-REGULAR (rigidity contrast alone) vs disorder+VD')
    ax.legend(); plt.tight_layout()
    plt.savefig(os.path.join(HERE, 'vd_decompose.png'), dpi=150, bbox_inches='tight'); plt.close()
    print('saved vd_decompose.png')


if __name__ == '__main__':
    main()
