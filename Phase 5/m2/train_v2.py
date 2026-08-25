r"""Train the M2 v2 surrogate on PER-TRIANGLE C(s).  S1 of `Phase 5/m2/M2_V2_PLAN.md`.

Replaces `train.py` (v1), which trained on bulk C6 with a random split scored against the solver
labels it had trained on -- i.e. it measured memorisation (`M2.md` records this).

WHAT IS OPTIMISED
    Per-triangle C(s), the target the section 2.1 head natively produces, each of the 6 components
    normalised by its TRAIN-split std so no component dominates by scale.  The bulk C_eff falls out
    as the unweighted mean and is reported, never fitted separately -- it cannot drift, since the
    head assembles it from the same per-triangle field.

HOW IT IS SPLIT -- both leakage traps of section 3.5
    * TRAJECTORY leakage: steps within one optimisation are near-duplicates, so a per-SAMPLE split
      puts near-copies on both sides.  Split on `traj_id`.
    * FAMILY holdout: the headline metric is leave-one-family-out, so `--holdout <family>` trains on
      every other family.  `--holdout random` (the default) reports the WITHIN-family split instead,
      which is NOT a result -- it is the memorisation baseline, and the gap between the two is the
      number the July validation did not have.

S1 GATES (the parametrisation gates are in `Phase 5/verifications/test_m2_head.py`, and pass before
training).  Here: does it LEARN -- per-triangle MAE falling, SPD rate 100 %, and the analytic cells
reproduced.

Run:  C:\Users\doron\anaconda3\python.exe "Phase 5/m2/train_v2.py" --epochs 200
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, HERE)
import model_v2 as M                                                      # noqa: E402

torch.set_default_dtype(torch.float64)


def load(path):
    """npz (pointer scheme) -> list of per-sample dicts ready for the model."""
    d = np.load(path)
    n = len(d['C6'])
    out = []
    for i in range(n):
        g = {}
        for key in ('pts', 'bond_u', 'bond_v', 'bond_R', 'tri_bond', 'tri_verts', 'areas', 'k',
                    'is_fictional', 'C6_per'):
            p = d[key + '_ptr']
            g[key] = d[key][p[i]:p[i + 1]]
        g['n_nodes'] = len(g['pts'])
        g['family'] = str(d['family'][i])
        g['traj_id'] = str(d['traj_id'][i])
        g['C6'] = d['C6'][i]
        out.append(g)
    return out


def prepare(g):
    """Per-sample tensors: invariant features, the equivariant basis Q, and the target."""
    k = torch.as_tensor(g['k'])
    kbar = k.mean().clamp_min(1e-12)
    ev = tri_edge_vectors(g)
    t = dict(
        bond_u=torch.as_tensor(g['bond_u'].astype(np.int64)),
        bond_v=torch.as_tensor(g['bond_v'].astype(np.int64)),
        tri_bond=torch.as_tensor(g['tri_bond'].astype(np.int64)),
        tri_verts=torch.as_tensor(g['tri_verts'].astype(np.int64)),
        edge_feat=M.edge_features(k, g['bond_R']),
        n_nodes=int(g['n_nodes']),
        Q=M.edge_carriers(ev),
        kbar=kbar,                          # the scaling symmetry: feed k/kbar, rescale C by kbar
        target=torch.as_tensor(g['C6_per']),
    )
    deg = torch.zeros(t['n_nodes'], 1)
    for idx in (t['bond_u'], t['bond_v']):
        deg = deg.index_add(0, idx, torch.ones(len(idx), 1))
    t['node_feat'] = deg / deg.mean().clamp_min(1.0)
    # physical factor, so predictions land in the same units as the stored labels
    areas = torch.as_tensor(g['areas'])
    t['phys'] = 8.0 * len(g['tri_verts']) / areas.sum()
    return t


def tri_edge_vectors(g):
    """Per-triangle edge vectors, taken straight from `bond_R` via `tri_bond`.

    Rebuilt from `bond_R` rather than by differencing node positions: a bond crossing the periodic
    seam has its stored endpoint on the far side of the box, so `pts[v] - pts[u]` would be a vector
    a whole box long.  `bond_R` already carries the wrap.

    ORIENTATION AND ORDER DO NOT MATTER HERE, which is worth stating because it looks as though they
    should.  The carrier `q_e = vec3(dx dx^T)` is QUADRATIC in `dx`, so it is invariant under
    `dx -> -dx`; and a permutation of a triangle's three carriers just permutes the columns of `Q`,
    which the learned `G` absorbs -- the correspondence that must hold is between column `i` of `Q`
    and the edge features of bond `i`, and both are gathered in `tri_bond` order.  Verified against
    `geo['edge_vecs']` on four mesh kinds: identical carrier sets, `max|QQ^T diff| <= 1e-14`."""
    return g['bond_R'][g['tri_bond'].astype(np.int64)]          # (n_tri, 3, 2)


def predict(net, t):
    G = net(t)
    C_per, C_eff = M.assemble(t['Q'], G, physical_factor=t['phys'] * t['kbar'])
    return M.sym3_to_c6(C_per), M.sym3_to_c6(C_eff)


def evaluate(net, samples, mu, sd):
    net.eval()
    per_ae, bulk_ae, spd_bad, n_tri = [], [], 0, 0
    with torch.no_grad():
        for t in samples:
            pc6, bc6 = predict(net, t)
            per_ae.append((pc6 - t['target']).abs().mean(0))
            bulk_ae.append((bc6 - t['target'].mean(0)).abs())
            eig = torch.linalg.eigvalsh(M.c6_to_sym3(pc6))
            spd_bad += int((eig.min(-1).values < -1e-10).sum())
            n_tri += len(pc6)
    return (torch.stack(per_ae).mean(0), torch.stack(bulk_ae).mean(0),
            spd_bad / max(n_tri, 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', default=os.path.join(HERE, 'data', 'dataset_smoke.npz'))
    ap.add_argument('--epochs', type=int, default=150)
    ap.add_argument('--hidden', type=int, default=64)
    ap.add_argument('--layers', type=int, default=4)
    ap.add_argument('--lr', type=float, default=3e-3)
    ap.add_argument('--holdout', default='random',
                    help="family name for leave-one-family-out, or 'random' for the "
                         "WITHIN-family memorisation baseline")
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    raw = load(a.data)
    if a.limit:
        raw = raw[:a.limit]
    print('%d samples from %s' % (len(raw), os.path.basename(a.data)))

    if a.holdout != 'random':
        tr_i = [i for i, g in enumerate(raw) if g['family'] != a.holdout]
        va_i = [i for i, g in enumerate(raw) if g['family'] == a.holdout]
        split = 'LEAVE-ONE-FAMILY-OUT: %s' % a.holdout
    else:
        # split on TRAJECTORY id, not sample id (section 3.5 trap 1)
        trajs = sorted({g['traj_id'] for g in raw})
        rng = np.random.default_rng(a.seed)
        rng.shuffle(trajs)
        val = set(trajs[:max(1, len(trajs) // 5)])
        tr_i = [i for i, g in enumerate(raw) if g['traj_id'] not in val]
        va_i = [i for i, g in enumerate(raw) if g['traj_id'] in val]
        split = 'within-family (trajectory split) -- MEMORISATION BASELINE, not a result'
    if not va_i:
        print('empty validation split for holdout=%r' % a.holdout)
        return 1
    print('%s\n  train %d / val %d' % (split, len(tr_i), len(va_i)))

    train = [prepare(raw[i]) for i in tr_i]
    valid = [prepare(raw[i]) for i in va_i]
    allt = torch.cat([t['target'] for t in train])
    mu, sd = allt.mean(0), allt.std(0).clamp_min(1e-12)

    net = M.ForwardGNNv2(hidden=a.hidden, n_layers=a.layers, passive=True)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    print('%d parameters' % sum(p.numel() for p in net.parameters()))

    t0 = time.time()
    order = np.arange(len(train))
    for ep in range(a.epochs):
        net.train()
        np.random.default_rng(ep).shuffle(order)
        tot = 0.0
        for j in order:
            t = train[j]
            pc6, _ = predict(net, t)
            loss = (((pc6 - t['target']) / sd) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            tot += float(loss)
        sched.step()
        if ep % max(1, a.epochs // 10) == 0 or ep == a.epochs - 1:
            per, bulk, bad = evaluate(net, valid, mu, sd)
            print('  ep %4d  train %.4f   val MAE/std per-tri %.4f  bulk %.4f   SPD viol %.4f  (%.0fs)'
                  % (ep, tot / len(train), float((per / sd).mean()), float((bulk / sd).mean()),
                     bad, time.time() - t0))

    per, bulk, bad = evaluate(net, valid, mu, sd)
    print('\nFINAL  (%s)' % split)
    names = ('xxxx', 'xxxy', 'xxyy', 'xyxy', 'xyyy', 'yyyy')
    print('  component   per-triangle MAE      bulk MAE        label std')
    for i, nm in enumerate(names):
        print('  %-9s   %14.5f  %13.5f  %13.5f' % (nm, per[i], bulk[i], sd[i]))
    print('  normalised per-triangle MAE/std = %.4f     bulk MAE/std = %.4f'
          % (float((per / sd).mean()), float((bulk / sd).mean())))
    print('  SPD violation rate = %.5f   (must be 0 -- structural in passive mode)' % bad)
    ck = os.path.join(HERE, 'checkpoint_v2.pt')
    torch.save(dict(state=net.state_dict(), hidden=a.hidden, layers=a.layers,
                    mu=mu, sd=sd, holdout=a.holdout, data=os.path.basename(a.data)), ck)
    print('  saved', ck)
    return 0


if __name__ == '__main__':
    sys.exit(main())
