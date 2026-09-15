r"""A1.a — audit the dataset in RESPONSE space, and cost the re-split.

WHY.  The dataset is organised, split and reported by GENERATOR FAMILY. But a family is how a mesh
was MADE, not what it DOES, and the two came apart under measurement:

  * `auxetic` as a generator is 299 samples (0.7 %); networks with nu < 0 number **8003 (19.3 %)**,
    of which the `auxetic` generator supplies only 2.9 % and `disordered` supplies 64 %.
  * leave-one-family-out on `auxetic` therefore holds out 0.7 % of the data and almost none of the
    physics it is named for.
  * A0.3's single worst failure was `tiling` at `max|W| = 0.8` -- the EASIEST regime -- and `tiling`
    is 0.2 % of the training set. A coverage failure, not a near-mechanism one.

So this script bins by what the network DOES and asks three things: where is the data actually
concentrated, what would a response-region holdout cost, and what would capping the dominant bin
cost. It GENERATES NOTHING and TRAINS NOTHING -- every descriptor it uses is already stored per
sample (`nu`, `E`, `anisotropy`, `w_max`, `min_eig`, `contrast`, `min_quality`), which is the point:
the fix is in how the data is split, not in building more of it.

Run:
    python "Phase 5/verifications/m2_response_audit.py"
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
RESULTS = os.path.join(REPO, 'Phase 5', 'results', 'm2_s1')

NU_EDGES = [-np.inf, -1.0, -0.5, -0.2, 0.0, 0.2, 1.0 / 3.0, 0.5, 1.0, np.inf]
NU_LABEL = ['<-1', '-1..-.5', '-.5..-.2', '-.2..0', '0..0.2', '0.2..1/3', '1/3..0.5', '0.5..1', '>1']
AN_EDGES = [1.0, 1.2, 2.0, 5.0, 20.0, np.inf]
AN_LABEL = ['iso<1.2', '1.2-2', '2-5', '5-20', '>20']
W_EDGES = [0.0, 1.0, 3.0, 10.0, 100.0, np.inf]
W_LABEL = ['<1', '1-3', '3-10', '10-100', '>100']


def load(path):
    d = np.load(path, allow_pickle=True)
    n = len(d['nu'])
    out = dict(nu=np.asarray(d['nu'], float), E=np.asarray(d['E'], float),
               an=np.asarray(d['anisotropy'], float), w=np.asarray(d['w_max'], float),
               fam=np.array([str(x) for x in d['family']]), n=n)
    # self-loops: not reconstructable into a solver, and slated for supercell re-representation
    bu, bv, pu, pv = d['bond_u'], d['bond_v'], d['bond_u_ptr'], d['bond_v_ptr']
    out['loop'] = np.array([bool((bu[pu[i]:pu[i + 1]] == bv[pv[i]:pv[i + 1]]).any())
                            for i in range(n)])
    return out


def occupancy(x, edges, labels):
    b = np.digitize(x, edges[1:-1])
    return b, np.array([int((b == i).sum()) for i in range(len(labels))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.join(REPO, 'Phase 5', 'm2', 'data',
                                                   'dataset_v2_s0.npz'))
    ap.add_argument('--holdout_nu', type=float, default=-0.2,
                    help='response-region holdout: every sample with nu below this')
    ap.add_argument('--cap', type=float, default=0.15, help='max share of any single nu bin')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    D = load(a.data)
    n = D['n']
    print('%s: %d samples' % (os.path.basename(a.data), n))
    print('self-loop samples (unreconstructable; supercell re-representation pending): %d (%.1f %%)'
          % (D['loop'].sum(), 100 * D['loop'].mean()))

    # ---- 1. marginal occupancy in response space -------------------------------------------
    nb, nc = occupancy(D['nu'], NU_EDGES, NU_LABEL)
    ab, ac = occupancy(D['an'], AN_EDGES, AN_LABEL)
    wb, wc = occupancy(D['w'], W_EDGES, W_LABEL)
    print('\n--- 1. MARGINAL occupancy (what the networks DO) ---')
    for name, lab, cnt in (('nu', NU_LABEL, nc), ('anisotropy', AN_LABEL, ac),
                           ('max|W|', W_LABEL, wc)):
        print('  %-11s %s' % (name, '  '.join('%s:%.1f%%' % (l, 100 * c / n)
                                              for l, c in zip(lab, cnt))))

    # ---- 2. the JOINT cells, and the empty ones --------------------------------------------
    print('\n--- 2. JOINT (nu x anisotropy) occupancy, %% of the set. EMPTY CELLS ARE THE GAPS ---')
    print('    %-10s %s' % ('nu \\ aniso', ' '.join('%8s' % l for l in AN_LABEL)))
    empty = []
    for i, nl in enumerate(NU_LABEL):
        row = []
        for j, al in enumerate(AN_LABEL):
            c = int(((nb == i) & (ab == j)).sum())
            row.append('%8.2f' % (100 * c / n))
            if c == 0:
                empty.append('%s x %s' % (nl, al))
        print('    %-10s %s' % (nl, ' '.join(row)))
    print('    EMPTY: %s' % (', '.join(empty) if empty else 'none'))

    # ---- 3. what a RESPONSE-REGION holdout costs, vs a family holdout ----------------------
    hold = D['nu'] < a.holdout_nu
    print('\n--- 3. RESPONSE-REGION holdout: nu < %+.2f ---' % a.holdout_nu)
    print('  held out: %d samples (%.1f %%), drawn from %d generator families'
          % (hold.sum(), 100 * hold.mean(), len(set(D['fam'][hold]))))
    u, c = np.unique(D['fam'][hold], return_counts=True)
    for i in np.argsort(-c)[:6]:
        print('     %-14s %5d  (%.0f %% of the held-out region)' % (u[i], c[i], 100 * c[i] / c.sum()))
    print('  FOR CONTRAST, leave-one-family-out removes:')
    for f in ('auxetic', 'tiling', 'basis', 'bravais'):
        m = D['fam'] == f
        print('     %-14s %5d samples (%.1f %%) -- of which nu<0: %d'
              % (f, m.sum(), 100 * m.mean(), int((D['nu'][m] < 0).sum())))

    # ---- 4. what CAPPING the dominant bin costs ---------------------------------------------
    # The cap must be SELF-CONSISTENT: capping at `cap * n` and then dropping shrinks the
    # denominator, so the capped bin ends up ABOVE the cap as a share of what remains (measured:
    # 15 % of the original became 21.5 % of the kept set). Solve m = cap * sum_i min(c_i, m) by
    # iteration -- it is monotone in m and converges in a few passes.
    rng = np.random.default_rng(a.seed)
    counts = np.array([int((nb == i).sum()) for i in range(len(NU_LABEL))])
    m = a.cap * n
    for _ in range(100):
        kept_tot = float(np.minimum(counts, m).sum())
        m_new = a.cap * kept_tot
        if abs(m_new - m) < 0.5:
            break
        m = m_new
    cap_n = int(m)
    keep = np.ones(n, bool)
    for i in range(len(NU_LABEL)):
        idx = np.flatnonzero(nb == i)
        if len(idx) > cap_n:
            drop = rng.choice(idx, len(idx) - cap_n, replace=False)
            keep[drop] = False
    print('\n--- 4. CAPPING any nu bin at %.0f %% of the set (subsample only, ZERO generation) ---'
          % (100 * a.cap))
    print('  kept %d of %d (%.1f %%); dropped %d' % (keep.sum(), n, 100 * keep.mean(), (~keep).sum()))
    _, nc2 = occupancy(D['nu'][keep], NU_EDGES, NU_LABEL)
    print('  nu occupancy after: %s'
          % '  '.join('%s:%.1f%%' % (l, 100 * c / keep.sum()) for l, c in zip(NU_LABEL, nc2)))
    worst = max(100 * c / keep.sum() for c in nc2)
    print('  worst bin share: %.1f %% (gate: <= %.0f %%)  %s'
          % (worst, 100 * a.cap + 1, 'OK' if worst <= 100 * a.cap + 1 else 'STILL OVER'))
    u, c = np.unique(D['fam'][~keep], return_counts=True)
    print('  dropped samples come from: %s'
          % ', '.join('%s %d' % (u[i], c[i]) for i in np.argsort(-c)[:5]))

    dst = os.path.join(RESULTS, 'response_audit.json')
    os.makedirs(RESULTS, exist_ok=True)
    with open(dst, 'w') as f:
        json.dump(dict(config=vars(a), n=n,
                       nu_counts=dict(zip(NU_LABEL, map(int, nc))),
                       aniso_counts=dict(zip(AN_LABEL, map(int, ac))),
                       w_counts=dict(zip(W_LABEL, map(int, wc))),
                       empty_cells=empty, n_selfloop=int(D['loop'].sum()),
                       holdout_n=int(hold.sum()), capped_keep=int(keep.sum())), f, indent=1)
    print('\n-> %s' % dst)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
