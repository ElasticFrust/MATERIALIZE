"""
Empirical test (no claim of principle): rescale the MF delta_g by the overshoot factor and
see how much closer it gets to the simulation.

A scalar rescale leaves corr and principal-axis angle unchanged; it only removes the
magnitude overshoot. So this isolates: of the MF error, how much is 'just the ~1.4x
overshoot' vs irreducible direction/shape error (= sqrt(1-corr^2))?

Compares: raw, rescaled by per-eta optimal c (well-shaped LS), rescaled by universal c=1.4.
Reports relative RMS error  ||c'^-1 dg_MF - dg_sim|| / ||dg_sim||  for all triangles and for
well-shaped (min angle>30) triangles, for Std and Std+edge.
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
CUNIV = 1.4


def minang(ev):
    e01, e02, e12 = ev[:, 0], ev[:, 1], ev[:, 2]
    def a(x, y):
        c = (x * y).sum(1) / np.sqrt(np.maximum((x**2).sum(1) * (y**2).sum(1), 1e-30))
        return np.degrees(np.arccos(np.clip(c, -1, 1)))
    a0, a1 = a(e01, e02), a(-e01, e12)
    return np.minimum(np.minimum(a0, a1), np.abs(180 - a0 - a1))


def fro(a, b):
    return (a * b).sum((1, 2))


def relerr(mf, sim, mask):
    return np.sqrt(fro((mf - sim)[mask], (mf - sim)[mask]).sum()
                   / max(fro(sim[mask], sim[mask]).sum(), 1e-300))


def main():
    out = {m: {k: [] for k in ['raw', 'reN', 'r14', 'rawG', 'reNG', 'r14G', 'corrG']}
           for m in ('Std', 'Std+edge')}
    for eta in ETAS:
        acc = {m: {k: [] for k in out[m]} for m in out}
        for t in range(10):
            f = os.path.join(DATA, f'sample_eta{eta:.2f}_trial{t}.npz')
            if not os.path.exists(f):
                continue
            s = np.load(f, allow_pickle=True)
            sim = s['dg_sim']; ang = minang(s['edge_vecs'])
            allm = np.ones(len(sim), bool); good = ang > 30
            for m in out:
                mf = s[f'dg_{m}']
                c = fro(mf[good], sim[good]).sum() / max(fro(sim[good], sim[good]).sum(), 1e-300)
                acc[m]['raw'].append(relerr(mf, sim, allm))
                acc[m]['reN'].append(relerr(mf / c, sim, allm))
                acc[m]['r14'].append(relerr(mf / CUNIV, sim, allm))
                acc[m]['rawG'].append(relerr(mf, sim, good))
                acc[m]['reNG'].append(relerr(mf / c, sim, good))
                acc[m]['r14G'].append(relerr(mf / CUNIV, sim, good))
                a = np.stack([mf[good, 0, 0], mf[good, 0, 1], mf[good, 1, 1]], 1).ravel()
                b = np.stack([sim[good, 0, 0], sim[good, 0, 1], sim[good, 1, 1]], 1).ravel()
                acc[m]['corrG'].append(np.corrcoef(a, b)[0, 1])
        for m in out:
            for k in out[m]:
                out[m][k].append(np.nanmean(acc[m][k]))

    print("Relative RMS error ||dg_MF(/c) - dg_sim|| / ||dg_sim||  (Std):")
    print(f"{'eta':>5} {'raw all':>8} {'÷c all':>8} {'÷1.4 all':>9} | "
          f"{'raw>30':>8} {'÷c >30':>8} {'÷1.4>30':>9} {'corr>30':>8}")
    for i, eta in enumerate(ETAS):
        o = out['Std']
        print(f"{eta:>5.1f} {o['raw'][i]:>8.2f} {o['reN'][i]:>8.2f} {o['r14'][i]:>9.2f} | "
              f"{o['rawG'][i]:>8.2f} {o['reNG'][i]:>8.2f} {o['r14G'][i]:>9.2f} {o['corrG'][i]:>8.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, scope, ttl in [(axes[0], ('raw', 'reN', 'r14'), 'all triangles'),
                           (axes[1], ('rawG', 'reNG', 'r14G'), 'well-shaped (min∠>30°)')]:
        for m, ls in [('Std', '-'), ('Std+edge', '--')]:
            o = out[m]
            ax.plot(ETAS, o[scope[0]], ls + 'o', color='#888', label=f'{m} raw')
            ax.plot(ETAS, o[scope[1]], ls + 's', color='#1f77b4', label=f'{m} ÷ c(η)')
            ax.plot(ETAS, o[scope[2]], ls + '^', color='#d62728', label=f'{m} ÷ 1.4')
        # irreducible floor sqrt(1-corr^2) for Std (corr on >30 set)
        floor = np.sqrt(np.clip(1 - np.array(out['Std']['corrG'])**2, 0, 1))
        ax.plot(ETAS, floor, ':', color='k', label='dir. floor √(1−corr²)')
        ax.set_xlabel('η'); ax.set_ylabel('relative RMS error'); ax.set_title(ttl)
        ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3); ax.set_ylim(0, None)
    fig.suptitle('Renormalizing δg_MF by the overshoot: how much error is "just the scale"?',
                 fontsize=12)
    plt.tight_layout()
    p = os.path.join(PLOTS, 'dg_renormalize_test.png')
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print('\nsaved', p)


if __name__ == '__main__':
    main()
