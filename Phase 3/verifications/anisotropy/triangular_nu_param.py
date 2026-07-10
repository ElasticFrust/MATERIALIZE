"""
anisotropy / triangular_nu_param — PARAMETRIC E(θ) vs ν(θ) plot for the designed triangular-sawtooth
patch. Reads triangular_nu.npz (no re-optimise), computes the directional ν(θ) and E(θ) from the
region-homogenised physical tensor (C.nu_E_theta), and traces the curve {(ν(θ), E(θ)) : θ∈[0,π]} in
the ν–E plane, coloured by θ. Shows how stiffness and Poisson ratio co-vary with load direction:
each point is one direction; the loop closes over the π period. Output: triangular_nu_param.png.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _common as C

CASE = 'anisotropy'


def main():
    geo, k_bond, C6_per, meta = C.load_network(os.path.join(C.savedir(CASE), 'triangular_nu.npz'))
    th = C.ANG                                                   # θ grid ∈ [0,π]
    nu, E = C.nu_E_theta(C.region_phys_C6(geo, C6_per, None), th)
    deg = np.degrees(th)

    fig, ax = plt.subplots(figsize=(8, 7))
    # θ-coloured parametric curve (segments) + scatter of the sample directions
    pts = np.column_stack([nu, E])
    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, cmap='twilight', array=deg[:-1], lw=2.4)
    ax.add_collection(lc)
    sc = ax.scatter(nu, E, c=deg, cmap='twilight', s=22, zorder=3, edgecolors='k', linewidths=.3)
    cb = plt.colorbar(sc, ax=ax); cb.set_label('load direction θ (deg)')

    # annotate the extremes (ν trough/peak, softest/stiffest directions)
    for i, tag in ((int(np.argmin(nu)), 'ν min'), (int(np.argmax(nu)), 'ν max'),
                   (int(np.argmin(E)), 'E min'), (int(np.argmax(E)), 'E max')):
        ax.annotate(f'{tag}\nθ={deg[i]:.0f}°', (nu[i], E[i]), fontsize=8,
                    xytext=(6, 6), textcoords='offset points', color='#333')

    ax.set_xlabel('Poisson ratio ν(θ)'); ax.set_ylabel('Young modulus E(θ)')
    ax.set_title('triangular_nu — parametric E(θ) vs ν(θ)  (colour = θ)', fontsize=12)
    ax.grid(alpha=.3); ax.set_ylim(bottom=0)
    ax.autoscale_view()
    plt.tight_layout()
    path = os.path.join(C.savedir(CASE), 'triangular_nu_param.png')
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  [triangular_nu_param] ν∈[{nu.min():+.3f},{nu.max():+.3f}]  E∈[{E.min():.3f},{E.max():.3f}]")
    print(f'saved {os.path.basename(path)}')


if __name__ == '__main__':
    main()
