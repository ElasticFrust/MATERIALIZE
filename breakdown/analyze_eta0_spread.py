"""
What is going on at eta=0 (perfect crystal): the 'orientation spread' of delta_g.

At eta=0 there is no disorder and the equilibrium is exactly affine (zero relaxation,
u_fluct ~ 0). Yet delta_g_sim = g_s - g is NONZERO, because the unit cell has TWO
inequivalent triangle orientations (T1, T2). Under a uniaxial strain they acquire
different affine metric changes, so each deviates from the macroscopic mean g.

To first order in the strain amplitude delta,
    delta_g_aff(tri) = 2*delta * [[ax^2, ax*bx],[ax*bx, bx^2]]
where (a, b) = (e01, e02) are the triangle's two basis edges and ax, bx their
x-components. It depends only on how the edges project onto the loading (x) axis.
T1 has (a,b)=(L1,L2), T2 has (a,b)=(L2, L2-L1) -> different x-projections -> different
delta_g_aff. delta_g_sim = delta_g_aff - mean, so on the two sublattices it is equal and
opposite. The mean-field gives W=0 here (identical bare tensors -> no contrast), hence
delta_g_MF = 0: it misses the spread entirely.

Run:  python analyze_eta0_spread.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'dg_analysis_data')
PLOTS = os.path.join(HERE, 'plots')


def principal(g):
    w, V = np.linalg.eigh(g)
    th = np.degrees(np.arctan2(V[1, 1], V[0, 1]))
    return w[1], w[0], (th + 90) % 180 - 90


def main():
    s = np.load(os.path.join(DATA, 'sample_eta0.00_trial0.npz'), allow_pickle=True)
    pts, sx = s['pts'], s['simplices']
    dg_sim, dg_mf, delta = s['dg_sim'], s['dg_Std'], float(s['delta'])
    n = len(dg_sim)
    T1, T2 = np.arange(0, n, 2), np.arange(1, n, 2)        # appended T1 then T2 per cell
    Dg = np.linalg.norm(np.array([[6.25e-1, 1.25e-1], [1.25e-1, 2.5e-1]]) * 2 * delta)

    print(f"distinct dg_sim tensors: {len(np.unique(np.round(dg_sim.reshape(n,4),9),axis=0))}")
    for lab, ti in [('T1', T1[0]), ('T2', T2[0])]:
        l1, l2, th = principal(dg_sim[ti])
        print(f"  {lab}: lam1={l1:+.3e} ({l1/Dg:+.2f}|Dg|)  lam2={l2:+.3e}  theta={th:+.1f}deg")
    print(f"  MF |dg| max = {np.abs(dg_mf).max():.2e}  -> MF predicts 0")

    # ── figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel A: a small mesh patch, triangles by sublattice + major-axis of dg_sim
    cent = pts[sx].mean(1)
    sel = np.where((cent[:, 0] >= 0) & (cent[:, 0] <= 4.5) &
                   (cent[:, 1] >= 0) & (cent[:, 1] <= 4.5))[0]
    ax = axes[0]
    colT = np.array(['#1f77b4' if (t % 2 == 0) else '#ff7f0e' for t in sel])
    pc = PolyCollection(pts[sx[sel]], facecolors=colT, edgecolors='w', alpha=0.35)
    ax.add_collection(pc)
    for t in sel:
        l1, l2, th = principal(dg_sim[t])
        d = np.array([np.cos(np.radians(th)), np.sin(np.radians(th))]) * 0.34
        c = cent[t]
        ax.plot([c[0]-d[0], c[0]+d[0]], [c[1]-d[1], c[1]+d[1]],
                color=('#d62728' if l1 > 0 else '#2ca02c'), lw=2.2)
    ax.set_aspect('equal'); ax.autoscale_view()
    ax.set_title('η=0 patch: sublattice (blue=T1, orange=T2)\n'
                 'lines = major principal axis of δg_sim', fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])

    # Panel B: the two distinct strain crosses (principal axes), T1 vs T2 vs MF
    ax = axes[1]
    for lab, ti, off in [('T1', T1[0], (-1.4, 0)), ('T2', T2[0], (1.4, 0))]:
        w, V = np.linalg.eigh(dg_sim[ti])
        c = np.array(off)
        for k in range(2):
            v = V[:, k] * (w[k] / Dg)            # scale by lambda/|Dg|
            col = '#d62728' if w[k] > 0 else '#2ca02c'
            ax.annotate('', xy=c + v, xytext=c - v,
                        arrowprops=dict(arrowstyle='<->', color=col, lw=2.2))
        ax.text(off[0], -1.25, lab, ha='center', fontsize=11)
    ax.plot(0, 0, 'ko', ms=6); ax.text(0, -0.2, 'MF\n(δg=0)', ha='center', fontsize=9)
    ax.set_xlim(-2.6, 2.6); ax.set_ylim(-1.5, 1.5); ax.set_aspect('equal')
    ax.set_title('δg_sim principal axes (÷‖Δg‖)\nred=stretch (λ>0), green=compress (λ<0)',
                 fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])

    # Panel C: every triangle's (dg[0,0], dg[0,1]) -> two points; MF at origin
    ax = axes[2]
    ax.scatter(dg_sim[:, 0, 0]/Dg, dg_sim[:, 0, 1]/Dg, s=10, c='#1f77b4',
               label='sim (every triangle)')
    ax.scatter(dg_mf[:, 0, 0]/Dg, dg_mf[:, 0, 1]/Dg, s=10, c='k', label='MF')
    ax.axhline(0, color='gray', lw=.5); ax.axvline(0, color='gray', lw=.5)
    ax.set_xlabel(r'$\delta g_{xx}/\|\Delta g\|$'); ax.set_ylabel(r'$\delta g_{xy}/\|\Delta g\|$')
    ax.set_title('All 3200 triangles collapse to ±one value\n(MF sits at 0)', fontsize=10)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    fig.suptitle('η=0 orientation spread: a perfect crystal already breaks the mean field '
                 '(zero relaxation, yet δg_sim ≠ 0; MF = 0)', fontsize=12)
    plt.tight_layout()
    out = os.path.join(PLOTS, 'dg_eta0_orientation_spread.png')
    plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    print('saved', out)


if __name__ == '__main__':
    main()
