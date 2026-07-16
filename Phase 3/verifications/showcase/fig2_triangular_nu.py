"""
showcase / fig2 — designed anisotropic Poisson ratio ν(θ) (triangular_nu), WITHOUT the ν spatial map.
Self-contained: reads the saved triangular_nu.{npz,csv} and renders three panels — the designed
network (bond rigidity k), the target-vs-achieved ν(θ) line, and the ν(θ) polar. High quality.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
VER = os.path.dirname(HERE)
sys.path.insert(0, VER)
import _common as C

ANI = os.path.join(VER, 'anisotropy')


def main():
    geo, k_bond, C6, meta = C.load_network(os.path.join(ANI, 'triangular_nu.npz'))
    k = np.asarray(k_bond, float)
    d = np.genfromtxt(os.path.join(ANI, 'triangular_nu.csv'), delimiter=',', names=True)
    th = np.radians(d['theta_deg']); deg = d['theta_deg']
    target, achieved = d['nu_target'], d['nu_achieved_independent']

    plt.rcParams.update({'font.size': 12, 'axes.titlesize': 13})
    fig = plt.figure(figsize=(16, 5.4))

    a0 = fig.add_subplot(1, 3, 1)
    lc = C.draw_network(a0, geo, k, cmap='viridis', lw_scale=4.5)
    cb = plt.colorbar(lc, ax=a0, fraction=0.046, pad=0.04); cb.set_label('bond rigidity k')
    a0.set_title(f'designed network  (k median {np.median(k):.2f})', fontsize=12)

    a1 = fig.add_subplot(1, 3, 2)
    a1.plot(deg, target, 'k--', lw=2.4, label='target ν*(θ)')
    a1.plot(deg, achieved, color='#d62728', lw=2.2, label='achieved (independent sim)')
    a1.axhline(0, color='0.7', lw=.6); a1.set_xlim(0, 180); a1.set_xticks(range(0, 181, 45))
    a1.set_xlabel('loading angle θ (deg)'); a1.set_ylabel('Poisson ratio ν(θ)')
    a1.grid(alpha=.3); a1.legend(fontsize=11, loc='best')
    a1.set_title('designed anisotropic ν(θ): target vs achieved', fontsize=12)

    a2 = fig.add_subplot(1, 3, 3, projection='polar')
    th2 = np.concatenate([th, th + np.pi])
    a2.plot(th2, np.concatenate([target, target]), 'k--', lw=2.2, label='target')
    a2.plot(th2, np.concatenate([achieved, achieved]), color='#d62728', lw=2.2, label='achieved')
    a2.plot(th2, np.zeros_like(th2), color='0.6', lw=.5)
    a2.set_rorigin(min(0.0, float(np.min(achieved))) * 1.05)
    a2.set_title('ν(θ) polar', fontsize=12, pad=16); a2.legend(fontsize=10, loc='upper right')

    fig.suptitle('Inverse-designed anisotropic Poisson ratio ν(θ) on a disordered network',
                 fontsize=15, y=1.02)
    plt.tight_layout()
    out = os.path.join(HERE, 'fig2_triangular_nu.png')
    plt.savefig(out, dpi=200, bbox_inches='tight'); plt.close()
    print(f'saved {os.path.basename(out)}  ν∈[{achieved.min():+.2f},{achieved.max():+.2f}]')


if __name__ == '__main__':
    main()
