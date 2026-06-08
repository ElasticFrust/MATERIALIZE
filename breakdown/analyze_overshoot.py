"""
Overshoot characterization + high-eta breakdown, stratified by TRIANGLE QUALITY.

For each triangle compare MF delta_g (Std) to the simulation:
  ratio = ||dg_MF|| / ||dg_sim||         (magnitude overshoot; >1 = MF too big)
  cos   = <dg_MF, dg_sim>_F / (||.||.||) (direction agreement; 1 = aligned)
Binned by the triangle's minimum interior angle, for each eta. Also the global overshoot
scalar c for well-shaped triangles (min angle > 30 deg) and the sliver fractions.

Goal: understand the high-eta (->0.5) regime -- is the breakdown uniform, or concentrated
on distorted triangles, and is the overshoot a stable scalar we could renormalize away?
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'dg_analysis_data')
PLOTS = os.path.join(HERE, 'plots')
ETAS = [0.1, 0.2, 0.3, 0.4, 0.5]
NTRIAL = 10
BINS = np.array([0, 5, 10, 15, 20, 25, 30, 40, 60])
CENT = 0.5 * (BINS[:-1] + BINS[1:])
CMAP = plt.cm.viridis(np.linspace(0, 0.95, len(ETAS)))


def minang(ev):
    e01, e02, e12 = ev[:, 0], ev[:, 1], ev[:, 2]
    def a(x, y):
        c = (x * y).sum(1) / np.sqrt(np.maximum((x**2).sum(1) * (y**2).sum(1), 1e-30))
        return np.degrees(np.arccos(np.clip(c, -1, 1)))
    a0, a1 = a(e01, e02), a(-e01, e12)
    return np.minimum(np.minimum(a0, a1), np.abs(180 - a0 - a1))


def fro(a, b):
    return (a * b).sum((1, 2))


def main():
    ratio_by_eta, cos_by_eta = {}, {}
    fr20, fr10, fr5 = [], [], []
    c_good, corr_good = [], []     # well-shaped (>30 deg)

    for eta in ETAS:
        ratios = [[] for _ in CENT]
        coss = [[] for _ in CENT]
        f20, f10, f5 = [], [], []
        cg_num, cg_den, good_mf, good_sim = 0.0, 0.0, [], []
        for t in range(NTRIAL):
            f = os.path.join(DATA, f'sample_eta{eta:.2f}_trial{t}.npz')
            if not os.path.exists(f):
                continue
            s = np.load(f, allow_pickle=True)
            sim, mf = s['dg_sim'], s['dg_Std']
            ang = minang(s['edge_vecs'])
            nsim = np.sqrt(fro(sim, sim)); nmf = np.sqrt(fro(mf, mf))
            ratio = nmf / np.maximum(nsim, 1e-30)
            cos = fro(mf, sim) / np.maximum(nmf * nsim, 1e-30)
            for i in range(len(CENT)):
                m = (ang >= BINS[i]) & (ang < BINS[i + 1]) & (nsim > 1e-12)
                if m.any():
                    ratios[i].extend(ratio[m]); coss[i].extend(cos[m])
            f20.append(np.mean(ang < 20)); f10.append(np.mean(ang < 10)); f5.append(np.mean(ang < 5))
            g = ang > 30
            cg_num += fro(mf[g], sim[g]).sum(); cg_den += fro(sim[g], sim[g]).sum()
            good_mf.append(mf[g]); good_sim.append(sim[g])
        ratio_by_eta[eta] = [np.median(r) if r else np.nan for r in ratios]
        cos_by_eta[eta] = [np.median(c) if c else np.nan for c in coss]
        fr20.append(np.mean(f20)); fr10.append(np.mean(f10)); fr5.append(np.mean(f5))
        c_good.append(cg_num / max(cg_den, 1e-300))
        gm = np.concatenate(good_mf); gs = np.concatenate(good_sim)
        a = np.stack([gm[:, 0, 0], gm[:, 0, 1], gm[:, 1, 1]], 1).ravel()
        b = np.stack([gs[:, 0, 0], gs[:, 0, 1], gs[:, 1, 1]], 1).ravel()
        corr_good.append(np.corrcoef(a, b)[0, 1])

    print("Well-shaped triangles (min angle > 30 deg):")
    print(f"{'eta':>5} {'overshoot c':>12} {'corr(MF,sim)':>13} {'%<20':>7} {'%<10':>7} {'%<5':>6}")
    for i, eta in enumerate(ETAS):
        print(f"{eta:>5.1f} {c_good[i]:>12.2f} {corr_good[i]:>13.2f} "
              f"{100*fr20[i]:>7.1f} {100*fr10[i]:>7.1f} {100*fr5[i]:>6.1f}")

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    ax = axes[0]
    for k, eta in enumerate(ETAS):
        ax.plot(CENT, cos_by_eta[eta], '-o', ms=4, color=CMAP[k], label=f'η={eta}')
    ax.axhline(1, color='gray', lw=0.5, ls=':'); ax.axhline(0, color='gray', lw=0.5)
    ax.set_xlabel('min triangle angle (deg)'); ax.set_ylabel('median cos(δg_MF, δg_sim)')
    ax.set_title('Direction agreement vs triangle quality'); ax.legend(fontsize=9); ax.grid(alpha=0.3)

    ax = axes[1]
    for k, eta in enumerate(ETAS):
        ax.plot(CENT, ratio_by_eta[eta], '-o', ms=4, color=CMAP[k], label=f'η={eta}')
    ax.axhline(1, color='gray', lw=0.5, ls=':')
    ax.set_xlabel('min triangle angle (deg)'); ax.set_ylabel('median ||δg_MF|| / ||δg_sim||')
    ax.set_title('Magnitude overshoot vs triangle quality'); ax.set_ylim(0, 2.5)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(ETAS, np.array(fr20) * 100, '-o', label='min∠ < 20°')
    ax.plot(ETAS, np.array(fr10) * 100, '-s', label='min∠ < 10°')
    ax.plot(ETAS, np.array(fr5) * 100, '-^', label='min∠ < 5°')
    ax.set_xlabel('η'); ax.set_ylabel('% of triangles')
    ax.set_title('Sliver fraction grows with disorder'); ax.legend(fontsize=9); ax.grid(alpha=0.3)

    fig.suptitle('Overshoot & high-η breakdown, by triangle quality (Std method)', fontsize=12)
    plt.tight_layout()
    out = os.path.join(PLOTS, 'dg_overshoot_sliver.png')
    plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    print('\nsaved', out)


if __name__ == '__main__':
    main()
