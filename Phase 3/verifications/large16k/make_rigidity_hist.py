"""
For the large decoupled-PATCH design: is the overall elastic response isotropic, and is the spring
rigidity distributed evenly over edge DIRECTIONS (per patch and system-wide)?

For each topology we (a) compute the global directional response ν(θ) (flat ⇒ isotropic), and
(b) bin every edge by its orientation angle and show, for the whole system, the stiff-E region R_E,
and the auxetic-ν region R_ν: the MEAN k vs direction (polar; a circle ⇒ no directional bias) and
the k distribution (histogram). Loaded from the saved networks — no re-solve.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import _common as C

ND = os.path.join(HERE, 'networks')
TOPOS = ['regular', 'disorder_hi']
NBIN = 18
GCOL = {'whole system': '#555555', 'R_E (stiff-E)': '#00b7c2', 'R_ν (auxetic)': '#2ca02c'}


def bond_dir(geo):
    R = np.asarray(geo['bond_R'])
    return np.mod(np.arctan2(R[:, 1], R[:, 0]), np.pi)               # edge orientation in [0,π)


def bond_mid(geo):
    return np.asarray(geo['pts'])[geo['bond_u']] + 0.5 * np.asarray(geo['bond_R'])


def in_circle(mid, spec):
    c = np.asarray(spec['center'])
    return ((mid - c) ** 2).sum(1) < spec['radius'] ** 2


def main():
    fig = plt.figure(figsize=(15, 5.6 * len(TOPOS)))
    edges_deg = np.degrees(np.linspace(0, np.pi, NBIN + 1))
    centers = np.mod(np.linspace(0, np.pi, NBIN + 1)[:-1] + np.pi / (2 * NBIN), np.pi)
    for row, topo in enumerate(TOPOS):
        geo, k, C6, meta = C.load_network(os.path.join(ND, f'patch__{topo}.npz'))
        k = np.asarray(k); RE, RN = meta['region']
        ang = bond_dir(geo); mid = bond_mid(geo)
        groups = {'whole system': np.ones(len(k), bool),
                  'R_E (stiff-E)': in_circle(mid, RE), 'R_ν (auxetic)': in_circle(mid, RN)}
        nu_th = C.nu_E_theta(C.region_phys_C6(geo, C6, None), C.ANG)[0]
        iso = np.ptp(nu_th)

        # polar: mean k vs edge direction (mirrored to [0,2π))
        axp = fig.add_subplot(len(TOPOS), 3, 3 * row + 1, projection='polar')
        for name, sel in groups.items():
            mk = np.array([k[sel & (ang >= np.radians(edges_deg[i])) & (ang < np.radians(edges_deg[i + 1]))].mean()
                           if (sel & (ang >= np.radians(edges_deg[i])) & (ang < np.radians(edges_deg[i + 1]))).any()
                           else np.nan for i in range(NBIN)])
            th = np.concatenate([centers, centers + np.pi, centers[:1]])
            r = np.concatenate([mk, mk, mk[:1]])
            axp.plot(th, r, '-o', ms=3, color=GCOL[name], label=name)
        axp.set_title(f'{topo}: mean k vs edge direction\n(circle ⇒ isotropic rigidity)', fontsize=9)
        axp.legend(fontsize=7, loc='upper right', bbox_to_anchor=(1.35, 1.15))

        # mean k vs direction, linear
        axl = fig.add_subplot(len(TOPOS), 3, 3 * row + 2)
        for name, sel in groups.items():
            mk = np.array([k[sel & (ang >= np.radians(edges_deg[i])) & (ang < np.radians(edges_deg[i + 1]))].mean()
                           if (sel & (ang >= np.radians(edges_deg[i])) & (ang < np.radians(edges_deg[i + 1]))).any()
                           else np.nan for i in range(NBIN)])
            axl.plot(np.degrees(centers), mk, '-o', ms=3, color=GCOL[name], label=name)
        axl.set_xlabel('edge orientation (deg)'); axl.set_ylabel('mean k'); axl.set_xlim(0, 180)
        axl.set_title(f'mean k vs direction  (global ν spread={iso:.3f} ⇒ '
                      f'{"isotropic" if iso < 0.05 else "anisotropic"})', fontsize=9)
        axl.grid(alpha=.3); axl.legend(fontsize=7)

        # k distribution histograms per group
        axh = fig.add_subplot(len(TOPOS), 3, 3 * row + 3)
        for name, sel in groups.items():
            axh.hist(k[sel], bins=40, range=(0, np.percentile(k, 99)), histtype='step', density=True,
                     lw=1.8, color=GCOL[name], label=f'{name} (n={sel.sum()})')
        axh.set_xlabel('k'); axh.set_ylabel('density'); axh.set_title('rigidity distribution', fontsize=9)
        axh.grid(alpha=.3); axh.legend(fontsize=7)
        print(f"  {topo}: global nu spread={iso:.3f}  mean k whole={k.mean():.2f} "
              f"R_E={k[groups['R_E (stiff-E)']].mean():.2f} R_nu={k[groups['R_ν (auxetic)']].mean():.2f}", flush=True)

    fig.suptitle('Decoupled-PATCH design (16k) — global response isotropy & rigidity-vs-direction '
                 '(whole system, R_E, R_ν)', fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(os.path.join(HERE, 'large16k_rigidity_hist.png'), dpi=150, bbox_inches='tight')
    plt.close(); print('saved large16k_rigidity_hist.png')


if __name__ == '__main__':
    main()
