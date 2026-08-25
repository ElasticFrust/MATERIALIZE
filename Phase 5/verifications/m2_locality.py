r"""How LOCAL is the elastic response? -- the receptive-field question, and what to label.

Answers two questions the M2 v2 design rests on (`Phase 5/m2/M2_V2_PLAN.md` sections 2.1, 3.1c, 3.3):

Q1  CAN A FINITE-HOP GNN CAPTURE `W` AT ALL?
    3.1c flags this as the plan's biggest architectural risk, and states it as a prediction rather
    than a measurement: `W` comes from a GLOBAL constrained solve (edge compatibility + zero discrete
    Gaussian curvature couple the whole cell, Poisson-like), while message passing sees only
    `n_layers` hops.  The surrogate can therefore capture `W` only insofar as the response is
    SCREENED over a finite correlation length -- and the prediction is that accuracy degrades with
    cell size, fastest near a mechanism, where that length diverges.  Nothing had measured it.

Q2  HOW MUCH SIGNAL DOES A BULK-ONLY LABEL THROW AWAY?
    The head predicts per-triangle `C(s)` and averages to `C_eff`, but 3.3 labels only the bulk
    `C6`.  Supervising the average lets local errors CANCEL: one triangle too stiff and another too
    soft gives zero bulk loss, so many wrong local fields produce the right mean.

METHOD
    Q1: build the triangle adjacency graph (triangles adjacent iff they share a bond), BFS the hop
        distance between every pair, and correlate the standardised per-triangle `C(s)` components
        against hop distance.  Averaged over the 6 components and over seeds.
    Q2: count numbers, and measure how much structure the bulk mean removes.

    The correlation length also converts into the quantity that matters for training: the number of
    EFFECTIVELY INDEPENDENT local samples a single network supplies, ~ n_tri / |k-hop ball|.

Run:  C:\Users\doron\anaconda3\python.exe "Phase 5/verifications/m2_locality.py"
Outputs -> Phase 5/results/m2_locality/  (locality.csv, locality.png); doc M2_LOCALITY.md.
"""
import os
import sys
import warnings
from collections import deque

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'Phase 3', 'verifications'))
import _common as C                                                       # noqa: E402
sys.path.insert(0, os.path.join(REPO, 'Phase 5'))
sys.path.insert(0, os.path.join(REPO, 'Phase 5', 'm2'))
sys.path.insert(0, REPO)
import matplotlib                                                         # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                           # noqa: E402
import torch                                                              # noqa: E402
import fields as F                                                        # noqa: E402
import plotting as P                                                      # noqa: E402
import seeds as S                                                         # noqa: E402
from inverse_design import DesignProblem                                  # noqa: E402

warnings.simplefilter('ignore')
torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)                       # D4: pinned, so the numbers are reproducible

OUT = os.path.join(REPO, 'Phase 5', 'results', 'm2_locality')
MAX_HOP = 8
SEEDS = (0, 1, 2)


def tri_adjacency(geo):
    """Triangle adjacency: two triangles are neighbours iff they share a bond."""
    tb = np.asarray(geo['tri_bond'])
    owner = {}
    for t, bonds in enumerate(tb):
        for b in bonds:
            owner.setdefault(int(b), []).append(t)
    adj = {t: set() for t in range(len(tb))}
    for ts in owner.values():
        for i in ts:
            adj[i].update(t for t in ts if t != i)
    return {t: sorted(v) for t, v in adj.items()}


def hop_matrix(adj, n, cap=MAX_HOP):
    """All-pairs BFS hop distance, capped (anything beyond `cap` is lumped at cap+1)."""
    D = np.full((n, n), cap + 1, np.int16)
    for src in range(n):
        D[src, src] = 0
        q = deque([src])
        while q:
            u = q.popleft()
            if D[src, u] >= cap:
                continue
            for v in adj[u]:
                if D[src, v] > D[src, u] + 1:
                    D[src, v] = D[src, u] + 1
                    q.append(v)
    return D


def per_triangle_C(geo, k):
    geo = dict(geo)
    geo['bond_k'] = k
    geo['tri_k'] = k[geo['tri_bond']]
    prob = DesignProblem.from_geo(geo)
    with torch.no_grad():
        out = prob.forward(torch.as_tensor(k), physical_units=True)
    return (np.asarray(out['per_triangle'], float),
            float(np.abs(np.asarray(out['W'], float)).max()))


def correlation_vs_hop(geo, k, cap=MAX_HOP, var_tol=1e-9):
    """Mean correlation of standardised per-triangle C(s) against hop distance.

    Returns all-NaN when the per-triangle field is CONSTANT: a perfect crystal at uniform k has
    W == 0 and identical C(s) on every triangle, so there is no variation to correlate and
    standardising divides by float noise.  Left unguarded that produced a hop-0 correlation of 0.91
    where it must be exactly 1 -- a meaningless row that looked like a measurement."""
    Cs, wmax = per_triangle_C(geo, k)
    n = len(Cs)
    sd = Cs.std(0)
    if float(np.max(sd / (np.abs(Cs).mean(0) + 1e-300))) < var_tol:
        return np.full(cap + 1, np.nan), wmax, n
    D = hop_matrix(tri_adjacency(geo), n, cap=cap)
    x = Cs - Cs.mean(0)
    x = x / np.where(sd > 0, sd, 1.0)
    cors = []
    for h in range(cap + 1):
        ii, jj = (np.where(np.eye(n, dtype=bool)) if h == 0
                  else np.where(np.triu(D == h, 1)))
        if len(ii) < 20:
            cors.append(np.nan)
            continue
        cors.append(float(np.mean([np.mean(x[ii, d] * x[jj, d]) for d in range(6)])))
    return np.array(cors), wmax, n


def ball_size(geo, hops=2):
    """Mean number of triangles within `hops` of a triangle -- the size of one correlated patch."""
    adj = tri_adjacency(geo)
    n = len(adj)
    D = hop_matrix(adj, n, cap=hops)
    return float(np.mean((D <= hops).sum(1)))


def regimes(seed):
    """The regimes to compare, spanning uniform -> disordered -> near-mechanism."""
    rng = np.random.default_rng(seed)
    out = []

    geo = S.bravais_lattice(1.0, 1.0, reps=6)['geo']
    # NB the perfect crystal at UNIFORM k is deliberately not in this set: W == 0 there and every
    # C(s) is identical, so there is no per-triangle variation to correlate -- see the var_tol
    # guard in `correlation_vs_hop`.
    out.append(('crystal, iid lognormal k', geo, F.k_field(geo, rng, structure='iid',
                                                           marginal='lognormal')[0]))
    out.append(('crystal, correlated k', geo, F.k_field(geo, rng, structure='correlated',
                                                        marginal='lognormal')[0]))
    geo = S.random_patch(120, seed=seed)['geo']
    out.append(('random, iid k', geo, F.k_field(geo, rng, structure='iid',
                                                marginal='lognormal')[0]))
    out.append(('random, correlated k', geo, F.k_field(geo, rng, structure='correlated',
                                                       marginal='lognormal')[0]))
    out.append(('random, diluted f=0.35', geo, F.dilute(geo, rng, frac=0.35, k_soft=1e-6)[0]))

    rec = S._reentrant_honeycomb(reps=4, v=1.15)
    out.append(('re-entrant honeycomb', rec['geo'], rec['k0']))
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    names, curves, wmaxes, ntris, balls = [], {}, {}, {}, {}
    for seed in SEEDS:
        for name, geo, k in regimes(seed):
            cors, wmax, n = correlation_vs_hop(geo, k)
            curves.setdefault(name, []).append(cors)
            wmaxes.setdefault(name, []).append(wmax)
            ntris.setdefault(name, []).append(n)
            balls.setdefault(name, []).append(ball_size(geo, hops=2))
            if name not in names:
                names.append(name)

    rows = ['regime,seeds,n_tri,max_absW,ball_2hop,eff_samples,' +
            ','.join('corr_hop%d' % h for h in range(MAX_HOP + 1))]
    print('%-26s %6s %9s %8s %11s   correlation vs hop'
          % ('regime', 'n_tri', 'max|W|', 'ball(2)', 'eff samples'))
    print('%-26s %6s %9s %8s %11s   %s'
          % ('', '', '', '', '', '  '.join('%5d' % h for h in range(MAX_HOP + 1))))
    for name in names:
        m = np.nanmean(np.stack(curves[name]), 0)
        nt = float(np.mean(ntris[name]))
        bl = float(np.mean(balls[name]))
        eff = nt / max(bl, 1.0)
        print('%-26s %6.0f %9.1f %8.1f %11.1f   %s'
              % (name, nt, np.mean(wmaxes[name]), bl, eff,
                 '  '.join('%5.2f' % c if np.isfinite(c) else '    -' for c in m)))
        rows.append('%s,%d,%.0f,%.3f,%.2f,%.1f,%s'
                    % (name, len(SEEDS), nt, np.mean(wmaxes[name]), bl, eff,
                       ','.join('%.4f' % c for c in m)))
    with open(os.path.join(OUT, 'locality.csv'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(rows) + '\n')

    fig, axes = plt.subplots(1, 2, figsize=(P.STYLE.PANEL[0] * 2.2, P.STYLE.PANEL[1]))
    cmap = plt.get_cmap('viridis')
    for i, name in enumerate(names):
        m = np.nanmean(np.stack(curves[name]), 0)
        s = np.nanstd(np.stack(curves[name]), 0)
        h = np.arange(MAX_HOP + 1)
        col = cmap(i / max(1, len(names) - 1))
        axes[0].plot(h, m, 'o-', color=col, ms=4, label=name)
        axes[0].fill_between(h, m - s, m + s, color=col, alpha=0.15, linewidth=0)
    axes[0].axhline(0, color='k', lw=0.8)
    axes[0].axhspan(-0.1, 0.1, color='0.85', zorder=0)
    axes[0].set_xlabel('hop distance between triangles')
    axes[0].set_ylabel(r'correlation of $C(s)$')
    axes[0].set_title('the response is SCREENED\n(grey band = noise level)')
    axes[0].grid(alpha=0.25)

    for i, name in enumerate(names):
        col = cmap(i / max(1, len(names) - 1))
        m1 = np.nanmean(np.stack(curves[name]), 0)[1]
        # ORDERED motifs are excluded from the trend: their C(s) is periodic, so the hop-1
        # correlation measures the motif's own alternation (the re-entrant honeycomb reads 0.00
        # because rib and spoke triangles alternate) and NOT a screening length. Drawn hollow so it
        # is visible but cannot be misread as contradicting the trend.
        ordered = 'honeycomb' in name
        axes[1].scatter(np.mean(wmaxes[name]), m1, s=70, label=name,
                        facecolors='none' if ordered else col, edgecolors=col,
                        linewidths=1.8 if ordered else 0.5)
        if ordered:
            axes[1].annotate('periodic:\nnot a screening length',
                             (np.mean(wmaxes[name]), m1), textcoords='offset points',
                             xytext=(12, 6), fontsize=7, color=col)
    axes[1].set_xscale('log')
    axes[1].set_xlabel(r'max$|W|$  (non-affine content $\rightarrow$ mechanism)')
    axes[1].set_ylabel('correlation at hop 1')
    axes[1].set_title('correlation length grows toward a mechanism\n'
                      '(hollow = ordered motif, excluded)')
    axes[1].grid(alpha=0.25)

    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.13))
    fig.suptitle('How local is the elastic response? — %d seeds per regime, threads pinned'
                 % len(SEEDS))
    fig.tight_layout()
    P.save_fig(fig, os.path.join(OUT, 'locality.png'))
    print('\nwrote', OUT)


if __name__ == '__main__':
    main()
