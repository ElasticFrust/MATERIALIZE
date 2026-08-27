r"""Train the v3 TENSOR-message model.  Reuses train_v2's data handling; own prepare/collate.

v2 is left untouched as the working scalar-message reference, so the two can be compared on the same
data, split and seed -- the difference between them is then the answer to "does passing the geometry
as a tensor, on triangle adjacency, actually matter".

Run:
  python "Phase 5/m2/train_v3.py" --holdout bravais --epochs 300
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, HERE)
import model_v2 as M2                                                    # noqa: E402
import model_v3 as M3                                                    # noqa: E402
import train_v2 as T2                                                    # noqa: E402

torch.set_default_dtype(torch.float64)
RESULTS = os.path.join(REPO, 'Phase 5', 'results', 'm2_s1')


def prepare(g):
    """Everything v3 consumes, from (edge vectors, k, l0) ALONE.

    No lengths, angles, degrees or gap statistics: those are derived and the network forms them if it
    wants them.  Lengths are normalised by `lbar` and `k` by its mean, which is what makes the
    prediction scale-invariant (see `train_v2.prepare` for why an unnormalised Q is unlearnable)."""
    bR = np.asarray(g['bond_R'], float)
    tb = g['tri_bond'].astype(np.int64)
    lbar = max(float(np.hypot(bR[:, 0], bR[:, 1]).mean()), 1e-12)
    Q = M2.edge_carriers(bR[tb] / lbar)                       # (n_tri, 3, 3) carriers as columns
    ell = np.hypot(bR[:, 0], bR[:, 1]) / lbar
    k = np.asarray(g['k'], float)
    kk = k / max(k.mean(), 1e-30)
    # l0 == l in all current data (forward(rest_lengths=None) -> zero prestress). Passed explicitly
    # anyway: it is a real input of A(s), and the residual-stress programme makes it independent.
    l0 = ell
    src, dst, bnd = M3.triangle_adjacency(tb, len(tb))
    bq = M2.edge_carriers((bR / lbar)[:, None, :])[:, :, 0]   # (n_bond, 3) per-bond carrier

    t = dict(Q=Q, tri_src=src, tri_dst=dst,
             tri_scalars=torch.as_tensor(np.concatenate([kk[tb], l0[tb], np.log(kk[tb] + 1e-12)], 1)),
             bond_q=bq[bnd],
             bond_feat=torch.as_tensor(np.stack([kk[bnd], l0[bnd]], 1)),
             target=torch.as_tensor(g['C6_per']),
             n_tri=len(tb), n_bond=len(bR))
    t['phys'] = 8.0 * len(tb) / torch.as_tensor(g['areas']).sum() * (lbar ** 2)
    t['kbar'] = torch.as_tensor(k).mean()
    return t


def collate(ts):
    """Block-diagonal batch.  Triangle indices offset by triangle count, not node count."""
    to = 0
    Q, src, dst, tsc, bq, bf, tg, sc = [], [], [], [], [], [], [], []
    for t in ts:
        Q.append(t['Q']); tsc.append(t['tri_scalars']); tg.append(t['target'])
        src.append(t['tri_src'] + to); dst.append(t['tri_dst'] + to)
        bq.append(t['bond_q']); bf.append(t['bond_feat'])
        sc.append(torch.full((t['n_tri'],), float(t['phys'] * t['kbar'])))
        to += t['n_tri']
    return dict(Q=torch.cat(Q), tri_src=torch.cat(src), tri_dst=torch.cat(dst),
                tri_scalars=torch.cat(tsc), bond_q=torch.cat(bq), bond_feat=torch.cat(bf),
                target=torch.cat(tg), scale=torch.cat(sc).reshape(-1, 1, 1))


def predict(net, t):
    G = net(t)
    scale = t['scale'] if 'scale' in t else t['phys'] * t['kbar']
    C_per, _ = M2.assemble(t['Q'], G, physical_factor=scale)
    return M2.sym3_to_c6(C_per)


def oracle_check(prepped, raws, tol=1e-9):
    """Same hard gate as v2: on W=0 samples the analytic G = diag(k/16 l^2) must reproduce the
    stored target to machine precision, or the units/scale/geometry are inconsistent."""
    n = 0
    for t, g in zip(prepped, raws):
        if g.get('w_max', 1.0) > 1e-9:
            continue
        l2 = t['Q'][:, 0, :] + t['Q'][:, 2, :]
        ktri = torch.as_tensor(g['k'][g['tri_bond'].astype(np.int64)])
        G = torch.diag_embed(ktri / (16.0 * l2))
        Cp, _ = M2.assemble(t['Q'], G, physical_factor=t['phys'] * t['kbar'])
        rel = float((M2.sym3_to_c6(Cp) - t['target']).abs().max()
                    / t['target'].abs().max().clamp_min(1e-300))
        if rel > tol:
            raise SystemExit('ORACLE CHECK FAILED (%s): rel %.3e -- units/scale/geometry '
                             'inconsistent, training would fit an unreachable target'
                             % (g.get('family', '?'), rel))
        n += 1
    return n


def evaluate(net, samples, sd):
    net.eval()
    ae, spd_bad, n_tri = [], 0, 0
    with torch.no_grad():
        for t in samples:
            p = predict(net, t)
            ae.append((p - t['target']).abs().mean(0))
            eig = torch.linalg.eigvalsh(M2.c6_to_sym3(p))
            spd_bad += int((eig.min(-1).values < -1e-10).sum())
            n_tri += len(p)
    return torch.stack(ae).mean(0), spd_bad / max(n_tri, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', default=os.path.join(HERE, 'data', 'dataset.npz'))
    ap.add_argument('--epochs', type=int, default=300)
    ap.add_argument('--ns', type=int, default=32)
    ap.add_argument('--nt', type=int, default=8)
    ap.add_argument('--hidden', type=int, default=64)
    ap.add_argument('--layers', type=int, default=3)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--lr', type=float, default=3e-3)
    ap.add_argument('--holdout', default='bravais')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    raw = [g for g in T2.load(a.data) if g['sim_ok'] and g['size_bin'] != 'large_holdout']
    if a.limit:
        # RANDOM subsample, not raw[:n]. The builder yields families in order (cells first), so a
        # head slice is a biased subset -- with --limit 400 the holdout family vanished entirely,
        # and the same trap produced a 10x benchmark misestimate earlier in this project.
        sel = np.random.default_rng(a.seed).choice(len(raw), min(a.limit, len(raw)), replace=False)
        raw = [raw[i] for i in sorted(sel)]
    if a.holdout != 'random':
        tr = [g for g in raw if g['family'] != a.holdout]
        va = [g for g in raw if g['family'] == a.holdout]
        split = 'LEAVE-ONE-FAMILY-OUT: %s' % a.holdout
    else:
        trajs = sorted({g['traj_id'] for g in raw})
        rng = np.random.default_rng(a.seed); rng.shuffle(trajs)
        val = set(trajs[:max(1, len(trajs) // 5)])
        tr = [g for g in raw if g['traj_id'] not in val]
        va = [g for g in raw if g['traj_id'] in val]
        split = 'within-family (trajectory split) -- MEMORISATION BASELINE'
    if not tr or not va:
        raise SystemExit('empty split: train %d, val %d (holdout=%r). With --limit the holdout '
                         'family may not have survived the subsample.' % (len(tr), len(va), a.holdout))
    print('%s\n  train %d / val %d' % (split, len(tr), len(va)))

    train = [prepare(g) for g in tr]
    valid = [prepare(g) for g in va]
    print('oracle check: %d W=0 samples exact' % oracle_check(train + valid, tr + va))
    sd = torch.cat([t['target'] for t in train]).std(0).clamp_min(1e-12)

    net = M3.ForwardGNNv3(ns=a.ns, nt=a.nt, hidden=a.hidden, n_layers=a.layers)
    print('%d parameters  (ns=%d nt=%d hidden=%d layers=%d)'
          % (sum(p.numel() for p in net.parameters()), a.ns, a.nt, a.hidden, a.layers))
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)

    t0, hist, order = time.time(), [], np.arange(len(train))
    for ep in range(a.epochs):
        net.train()
        np.random.default_rng(ep).shuffle(order)
        tot = nb = 0.0
        for b0 in range(0, len(order), a.batch):
            t = collate([train[j] for j in order[b0:b0 + a.batch]])
            loss = (((predict(net, t) - t['target']) / sd) ** 2).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step(); tot += float(loss); nb += 1
        sch.step()
        if ep % max(1, a.epochs // 10) == 0 or ep == a.epochs - 1:
            per, bad = evaluate(net, valid, sd)
            hist.append(dict(epoch=ep, train=tot / max(nb, 1),
                             val=float((per / sd).mean()), spd=bad))
            print('  ep %4d  train %.4f   val MAE/std %.4f   SPD viol %.4f  (%.0fs)'
                  % (ep, tot / max(nb, 1), float((per / sd).mean()), bad, time.time() - t0))

    per, bad = evaluate(net, valid, sd)
    print('\nFINAL (%s)   per-triangle MAE/std = %.4f   SPD viol %.5f'
          % (split, float((per / sd).mean()), bad))
    print('  for reference: bulk baseline 0.5144 ;  v2 scalar-message 0.509-0.511')
    os.makedirs(RESULTS, exist_ok=True)
    tag = a.holdout if a.holdout != 'random' else 'within'
    torch.save(dict(state=net.state_dict(), ns=a.ns, nt=a.nt, hidden=a.hidden, layers=a.layers),
               os.path.join(HERE, 'checkpoint_v3_%s.pt' % tag))
    with open(os.path.join(RESULTS, 'run_v3_%s.json' % tag), 'w', encoding='utf-8') as fh:
        json.dump(dict(model='v3', split=split, epochs=a.epochs, ns=a.ns, nt=a.nt,
                       hidden=a.hidden, layers=a.layers, n_train=len(train), n_val=len(valid),
                       params=int(sum(p.numel() for p in net.parameters())),
                       per_triangle_mae=[float(x) for x in per],
                       label_std=[float(x) for x in sd],
                       mae_over_std=float((per / sd).mean()), spd_violation=float(bad),
                       history=hist, wall_seconds=round(time.time() - t0, 1)), fh, indent=2)
    return 0


if __name__ == '__main__':
    sys.exit(main())
