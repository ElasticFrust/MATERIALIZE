"""
Extension of verify_soft_circles_50x50.py: per-triangle response W3(s) in the OTHER
macroscopic strain directions (not just dilation response to e_xx), for the same three
50x50 networks (perfect lattice eta=0, disordered eta=0.3, periodic Poisson-Delaunay),
each with the same two soft-bond circles (k=0.05).

For each network (row) and each response direction (column-group):
  - dilation response (g_xx+g_yy) to the e_xx macro mode
  - dilation response (g_xx+g_yy) to the e_yy macro mode
  - shear response (g_xy) to the e_xy macro mode
we plot the spatial maps (sim, intrinsic solver, sim-intrinsic) and the value distribution
(sim vs solver histogram) across all triangles.
"""
import os, sys, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..', 'Phase 2'))
import test_cluster_VD as VD
from verify_soft_circles_50x50 import (
    N, SOFT_K, build_poisson_geometry, soft_circles_mask, sim_W3, solver_W3)

RESPONSES = [
    ('dilation resp. to e_xx (g_xx+g_yy)', lambda W: W[:, 0, 0] + W[:, 2, 0]),
    ('dilation resp. to e_yy (g_xx+g_yy)', lambda W: W[:, 0, 1] + W[:, 2, 1]),
    ('shear resp. to e_xy (g_xy)',         lambda W: W[:, 1, 2]),
]


def main():
    cases = [
        ('perfect lattice (eta=0.0)', lambda: VD.build_geometry(N, 0.0, seed=0)),
        ('disordered lattice (eta=0.3)', lambda: VD.build_geometry(N, 0.3, seed=0)),
        ('Poisson Delaunay (N*N pts)', lambda: build_poisson_geometry(N, seed=0)),
    ]

    fig, axes = plt.subplots(3, 12, figsize=(34, 11))
    for row, (label, builder) in enumerate(cases):
        t0 = time.time()
        geo = builder()
        mask, centres, r = soft_circles_mask(geo)
        bond_k = np.ones(len(geo['bond_R'])); bond_k[mask] = SOFT_K
        geo['bond_k'] = bond_k; geo['tri_k'] = bond_k[geo['tri_bond']]

        Wsim = sim_W3(geo)
        Wint = solver_W3(geo)
        cen = geo['pts'][geo['simplices']].mean(axis=1)
        print(f"{label}  [{time.time()-t0:.1f}s]", flush=True)

        for m, (ttl, f) in enumerate(RESPONSES):
            vals_s, vals_i = f(Wsim), f(Wint)
            corr = np.corrcoef(vals_s, vals_i)[0, 1]
            rmse = np.sqrt(np.mean((vals_s - vals_i) ** 2))
            print(f"    {ttl:<32} corr={corr:.5f}  rmse={rmse:.4e}", flush=True)
            vmax = max(np.abs(vals_s).max(), np.abs(vals_i).max())
            c0 = m * 4
            for col, (vals, sub) in enumerate([(vals_s, 'sim'), (vals_i, 'intrinsic'),
                                                (vals_s - vals_i, 'sim-intrinsic')]):
                ax = axes[row, c0 + col]
                vm = vmax if col < 2 else np.abs(vals).max()
                sc = ax.scatter(cen[:, 0], cen[:, 1], c=vals, cmap='RdBu_r', vmin=-vm, vmax=vm, s=5)
                for c in centres:
                    ax.add_patch(plt.Circle(c, r, facecolor='none', edgecolor='lime', lw=1.2))
                ax.set_title(f'{sub}: {ttl}' if col < 2 else 'sim - intrinsic',
                             fontsize=8)
                ax.set_aspect('equal'); plt.colorbar(sc, ax=ax, fraction=0.046)
                if col == 0:
                    ax.set_ylabel(label, fontsize=10)

            ax = axes[row, c0 + 3]
            bins = np.linspace(-vmax, vmax, 60) if vmax > 1e-12 else 60
            ax.hist(vals_s, bins=bins, alpha=0.5, color='k', label='sim')
            ax.hist(vals_i, bins=bins, alpha=0.5, color='#d62728', label='intrinsic')
            ax.set_title(f'distribution: {ttl}', fontsize=8)
            ax.legend(fontsize=7)

    fig.suptitle(f'Per-triangle response W3(s): sim vs forward solver (intrinsic), other '
                 f'macro directions, two soft circles (k={SOFT_K}), N={N}', fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    p = os.path.join(HERE, 'plots', 'dg_soft_circles_50x50_directions.png')
    plt.savefig(p, dpi=110, bbox_inches='tight'); plt.close()
    print('saved', p)


if __name__ == '__main__':
    main()
