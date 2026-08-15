"""
showcase / fig6 — ANY Poisson ratio, ANY topology. Design k so the isotropic Poisson ratio hits a range
of targets (auxetic → strongly positive) on regular, disordered and anisotropic networks, and show the
achieved ⟨ν⟩ vs target (points on the diagonal = success; bars = angular spread). Small insets show the
three base topologies. Self-contained: designs freshly + independent-sim readback.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

HERE = os.path.dirname(os.path.abspath(__file__))
VER = os.path.dirname(HERE)
sys.path.insert(0, VER)
import _common as C

TOPOS = [('regular', 'regular', '#1f77b4'), ('disorder_hi', 'disordered', '#2ca02c'),
         ('aniso_str', 'anisotropic', '#d62728')]
TARGETS = [-0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7]
N, REG, NITER, NREST = 12, 2e-3, 200, 1
TH = C.ANG


def main():
    plt.rcParams.update({'font.size': 12})
    fig, ax = plt.subplots(figsize=(8.2, 7.6))
    ax.plot([-0.7, 0.9], [-0.7, 0.9], 'k--', lw=1.2, alpha=0.6, label='ideal (achieved = target)')
    ax.axhline(0, color='0.8', lw=.6); ax.axvline(0, color='0.8', lw=.6)

    for topo, label, col in TOPOS:
        means, spreads = [], []
        for nu0 in TARGETS:
            prob, geo = C.make_case(topo, N)
            r = C.optimize(prob, [C.Objective('nu_theta', np.full_like(TH, nu0))],
                           mode='k', n_iter=NITER, n_restarts=NREST, reg=REG, verbose=False)
            C.apply_k_to_geo(geo, r['k'])
            nu = C.nu_E_theta(C.sim_bulk_C6(geo), TH)[0]
            means.append(nu.mean()); spreads.append(nu.std())
        ax.errorbar(TARGETS, means, yerr=spreads, marker='o', ms=7, lw=1.8, capsize=4,
                    color=col, label=label)
        print(f"  {topo}: max|achieved-target|={max(abs(m-t) for m,t in zip(means,TARGETS)):.3f}", flush=True)

    ax.set_xlabel('target Poisson ratio ν*'); ax.set_ylabel('achieved ⟨ν⟩  (bar = angular spread)')
    ax.set_xlim(-0.7, 0.9); ax.set_ylim(-0.7, 0.9); ax.set_aspect('equal')
    ax.grid(alpha=0.3); ax.legend(fontsize=11, loc='upper left')
    ax.set_title('Any Poisson ratio, any topology — auxetic → strongly positive,\n'
                 'designed on regular, disordered & anisotropic networks', fontsize=12.5)

    # topology insets (bare mesh, fixed colour) stacked in the empty lower-right corner
    from matplotlib.collections import LineCollection
    for i, (topo, label, col) in enumerate(TOPOS):
        _, geo = C.make_case(topo, 8)
        iax = ax.inset_axes([0.70, 0.04 + 0.235 * i, 0.24, 0.215])
        u = geo['pts'][geo['bond_u']]
        iax.add_collection(LineCollection(np.stack([u, u + geo['bond_R']], 1), colors=col, linewidths=0.4, alpha=0.85))
        C.square_frame(iax, geo)
        iax.set_title(label, fontsize=7.5, color=col, pad=1)

    out = os.path.join(HERE, 'fig6_different_nu.png')
    plt.savefig(out, dpi=200, bbox_inches='tight'); plt.close()
    print(f'saved {os.path.basename(out)}')


if __name__ == '__main__':
    main()
