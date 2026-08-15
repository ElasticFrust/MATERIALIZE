"""
For the large decoupled-PATCH design: is the overall elastic response isotropic, and is the spring
rigidity distributed evenly over edge DIRECTIONS (per patch and system-wide)?

For each topology we (a) compute the global directional response ν(θ) (flat ⇒ isotropic), and
(b) bin every edge by its orientation angle and show, for the whole system, the stiff-E region R_E,
and the auxetic-ν region R_ν: the MEAN k vs direction WITH ±1σ / ±2σ distribution bands (polar +
linear) and the k distribution (histogram). Loaded from the saved networks — no re-solve.
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
EDG = np.linspace(0, np.pi, NBIN + 1)
CEN = 0.5 * (EDG[:-1] + EDG[1:])


def bond_dir(geo):
    R = np.asarray(geo['bond_R']); return np.mod(np.arctan2(R[:, 1], R[:, 0]), np.pi)


def bond_mid(geo):
    return np.asarray(geo['pts'])[geo['bond_u']] + 0.5 * np.asarray(geo['bond_R'])


def in_circle(mid, spec):
    c = np.asarray(spec['center']); return ((mid - c) ** 2).sum(1) < spec['radius'] ** 2


def binned(k, ang, sel):
    """mean and std of k per direction bin (restricted to `sel`)."""
    m = np.full(NBIN, np.nan); s = np.full(NBIN, np.nan)
    for i in range(NBIN):
        b = sel & (ang >= EDG[i]) & (ang < EDG[i + 1])
        if b.any():
            m[i] = k[b].mean(); s[i] = k[b].std()
    return m, s


def main():
    fig = plt.figure(figsize=(15, 5.6 * len(TOPOS)))
    for row, topo in enumerate(TOPOS):
        geo, k, C6, meta = C.load_network(os.path.join(ND, f'patch__{topo}.npz'))
        k = np.asarray(k); RE, RN = meta['region']
        ang = bond_dir(geo); mid = bond_mid(geo)
        groups = {'whole system': np.ones(len(k), bool),
                  'R_E (stiff-E)': in_circle(mid, RE), 'R_ν (auxetic)': in_circle(mid, RN)}
        stats = {n: binned(k, ang, sel) for n, sel in groups.items()}
        iso = np.ptp(C.nu_E_theta(C.sim_bulk_C6(geo), C.ANG)[0])
        deg = np.degrees(CEN)
        ncol = 1 + len(groups)
        axes = [fig.add_subplot(len(TOPOS), ncol, ncol * row + i + 1) for i in range(ncol)]

        # ONE plot: all means together (no bands)
        for name, (m, s) in stats.items():
            axes[0].plot(deg, m, '-o', ms=3, color=GCOL[name], label=name)
        axes[0].set_ylabel('mean k'); axes[0].set_title(
            f'{topo}: all means (global ν spread={iso:.3f} ⇒ '
            f'{"isotropic" if iso < 0.05 else "anisotropic"})', fontsize=9)
        axes[0].legend(fontsize=7)

        # SEPARATE plot per group: that group's mean + ±1σ / ±2σ spread
        for ax, (name, (m, s)) in zip(axes[1:], stats.items()):
            ax.fill_between(deg, m - 2 * s, m + 2 * s, color=GCOL[name], alpha=0.15, label='±2σ')
            ax.fill_between(deg, m - s, m + s, color=GCOL[name], alpha=0.30, label='±1σ')
            ax.plot(deg, m, '-o', ms=3, color=GCOL[name])
            ax.set_title(f'{name}  (mean ±σ)', fontsize=9); ax.legend(fontsize=7)
        for ax in axes:
            ax.set_xlabel('edge orientation (deg)'); ax.set_xlim(0, 180); ax.grid(alpha=.3)
        lo = min(np.nanmin(m - 2 * s) for (m, s) in stats.values())
        hi = max(np.nanmax(m + 2 * s) for (m, s) in stats.values())
        for ax in axes:
            ax.set_ylim(min(0, lo), hi * 1.05)
        print(f"  {topo}: nu spread={iso:.3f}  mean±std k whole={k.mean():.2f}±{k.std():.2f} "
              f"R_E={k[groups['R_E (stiff-E)']].mean():.2f}±{k[groups['R_E (stiff-E)']].std():.2f} "
              f"R_nu={k[groups['R_ν (auxetic)']].mean():.2f}±{k[groups['R_ν (auxetic)']].std():.2f}", flush=True)

    fig.suptitle('Decoupled-PATCH design (16k) — mean k vs edge direction: all means together (col 1), '
                 'each group\'s mean±σ separately (cols 2–4)', fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(os.path.join(HERE, 'large16k_rigidity_hist.png'), dpi=150, bbox_inches='tight')
    plt.close(); print('saved large16k_rigidity_hist.png')


if __name__ == '__main__':
    main()
