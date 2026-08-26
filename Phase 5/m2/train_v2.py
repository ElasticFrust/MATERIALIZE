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
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, HERE)
import model_v2 as M                                                      # noqa: E402

#: Every run writes its metrics here, in-repo.  Results that live only in a terminal or a scratch
#: file are the project's documented failure mode ("findings live in prose, not data"), and the
#: first two S1 runs hit it -- their only record was a temp log, and a fixed checkpoint name meant
#: the second run silently overwrote the first one's weights.
RESULTS = os.path.join(REPO, 'Phase 5', 'results', 'm2_s1')

torch.set_default_dtype(torch.float64)


#: variable-length per-sample arrays, stored flat with a companion `<key>_ptr` offset array
VAR_KEYS = ('pts', 'bond_u', 'bond_v', 'bond_R', 'tri_bond', 'tri_verts', 'areas', 'k',
            'is_fictional', 'C6_per')


def load(path):
    """npz (pointer scheme) -> list of per-sample dicts ready for the model.

    Every array is pulled out of the archive ONCE, before the per-sample loop.  `np.load` on an npz
    returns a lazy `NpzFile` whose `__getitem__` DECOMPRESSES THE WHOLE ARRAY on each access, so
    indexing it inside the loop meant ~100 000 full decompressions of a 65 MB archive -- which both
    crawled and died with `Unable to allocate 13.0 MiB` from the churn, on a machine with plenty of
    memory free.  Hoisting the reads is the entire fix."""
    with np.load(path) as d:
        arrs = {k: d[k] for k in VAR_KEYS}
        ptrs = {k: d[k + '_ptr'] for k in VAR_KEYS}
        C6 = d['C6']
        family, traj = d['family'], d['traj_id']
        kpat = d['k_pattern'] if 'k_pattern' in d else np.array([''] * len(C6))
        # `w_max` is what `oracle_check` selects on: W == 0 means C(s) = A(s) is a closed form
        wmax = d['w_max'] if 'w_max' in d else np.full(len(C6), np.inf)
        sim_ok = d['sim_ok'] if 'sim_ok' in d else np.ones(len(C6), bool)
        size_bin = d['size_bin'] if 'size_bin' in d else np.array(['train'] * len(C6))

    out = []
    for i in range(len(C6)):
        g = {k: arrs[k][ptrs[k][i]:ptrs[k][i + 1]] for k in VAR_KEYS}
        g.update(n_nodes=len(g['pts']), C6=C6[i], family=str(family[i]), traj_id=str(traj[i]),
                 sim_ok=bool(sim_ok[i]), size_bin=str(size_bin[i]),
                 k_pattern=str(kpat[i]), w_max=float(wmax[i]))
        out.append(g)
    return out


def prepare(g, angles=True):
    """Per-sample tensors: invariant features, the equivariant basis Q, and the target."""
    k = torch.as_tensor(g['k'])
    kbar = k.mean().clamp_min(1e-12)
    # LENGTH SCALE. The carriers q_e = vec3(dx dx^T) are dimension length^2, so an unnormalised Q
    # makes C_pred scale as lbar^2 -- while the TARGET (physical C) is scale-INVARIANT and every
    # network input (k/kbar, l/lbar, angles, degree) is scale-invariant too, so the head would have
    # to guess a per-network factor it cannot see. Measured spread across the dataset: lbar 0.752 to
    # 1.600, i.e. a factor ~4 in C. Building Q from dx/lbar makes it dimensionless, and lbar^2 moves
    # into the physical factor, where `phys * lbar^2` is itself scale-invariant (phys = 8N/sum(areas)
    # and sum(areas) ~ N lbar^2).
    ev = tri_edge_vectors(g)
    lbar = float(np.hypot(g['bond_R'][:, 0], g['bond_R'][:, 1]).mean())
    lbar = max(lbar, 1e-12)
    ev = ev / lbar
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
    # ANGULAR features (section 2.3) -- degree alone leaves the scalar path blind to geometry.
    # `angles=False` reproduces the degree-only feature set exactly, so the two can be run as a
    # controlled ABLATION on identical data and seed: comparing against an earlier run would confound
    # the feature change with a different dataset and different code.
    if angles:
        t['node_feat'] = M.node_angle_features(g['bond_u'], g['bond_v'], g['bond_R'], t['n_nodes'])
        t['tri_feat'] = M.triangle_angle_features(t['Q'])
    else:
        deg = torch.zeros(t['n_nodes'], 1)
        for idx in (t['bond_u'], t['bond_v']):
            deg = deg.index_add(0, idx, torch.ones(len(idx), 1))
        t['node_feat'] = deg / deg.mean().clamp_min(1.0)
        t['tri_feat'] = torch.zeros(len(t['Q']), 0)
    # physical factor, so predictions land in the same units as the stored labels
    areas = torch.as_tensor(g['areas'])
    t['phys'] = 8.0 * len(g['tri_verts']) / areas.sum() * (lbar ** 2)   # lbar^2 from the hatted Q
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


def oracle_check(prepped, raws, tol=1e-9):
    """Push the ANALYTIC answer through the exact pipeline and demand machine precision.

    THIS IS THE CHECK THAT CATCHES THE BUG CLASS NO STRUCTURAL GATE CAN.  Twice on 2026-08-26 the
    model "failed to learn" because the target was unreachable from its inputs -- once because `Q`
    was unnormalised so `C_pred` scaled as `lbar^2` (a ~4x per-sample factor), once because the
    target sat in INTERNAL units while the prediction was PHYSICAL (18..220x).  SPD, equivariance,
    intensivity and expressiveness all passed in both cases, because they check the FORM of the head
    rather than whether the target is REACHABLE.

    On any sample with `W == 0` the answer is closed form: `C(s) = A(s)`, i.e. `G = diag(k_e/16 l_e^2)`
    in the same hatted units `prepare` builds.  If assembling that does not reproduce the stored
    target to machine precision, the units, the scale or the geometry are inconsistent and training
    is pointless -- so this raises rather than warns.

    Returns the number of samples checked (0 if the dataset contains no W == 0 sample)."""
    n = 0
    for t, g in zip(prepped, raws):
        if g.get('w_max', 1.0) > 1e-9:
            continue
        l2 = t['Q'][:, 0, :] + t['Q'][:, 2, :]
        ktri = torch.as_tensor(g['k'][g['tri_bond'].astype(np.int64)])
        G = torch.diag_embed(ktri / (16.0 * l2))
        Cp, _ = M.assemble(t['Q'], G, physical_factor=t['phys'] * t['kbar'])
        rel = float((M.sym3_to_c6(Cp) - t['target']).abs().max()
                    / t['target'].abs().max().clamp_min(1e-300))
        if rel > tol:
            raise SystemExit(
                'ORACLE CHECK FAILED on a W=0 sample (%s): the analytic C(s) = A(s) reproduces the '
                'stored target only to rel %.3e. The units, scale or geometry are inconsistent -- '
                'training would fit an unreachable target. See `oracle_check`.'
                % (g.get('family', '?'), rel))
        n += 1
    return n


def collate(ts):
    """Merge prepared samples into ONE block-diagonal graph.

    Graphs of different sizes batch by concatenation with index offsets -- there is no padding and
    no masking, because message passing only ever follows edges and the blocks share none.  Without
    this the trainer takes one optimiser step per graph: measured 0.014 s/graph, i.e. 136 s per epoch
    over 9715 graphs and ~3.8 h for 100 epochs, which is also 9715 very noisy gradient steps.

    The physical scale is PER SAMPLE (`8*n_tri/sum(areas)` and `mean(k)` differ between graphs), so
    it is carried as a per-TRIANGLE vector rather than a scalar."""
    nb = bo = to = 0
    bu, bv, tb, tv, ef, nf, tf, Q, tg, sc = [], [], [], [], [], [], [], [], [], []
    for t in ts:
        bu.append(t['bond_u'] + nb)
        bv.append(t['bond_v'] + nb)
        tb.append(t['tri_bond'] + bo)
        tv.append(t['tri_verts'] + nb)
        ef.append(t['edge_feat'])
        nf.append(t['node_feat'])
        tf.append(t['tri_feat'])
        Q.append(t['Q'])
        tg.append(t['target'])
        sc.append(torch.full((len(t['Q']),), float(t['phys'] * t['kbar'])))
        nb += t['n_nodes']
        bo += len(t['edge_feat'])
        to += len(t['Q'])
    return dict(bond_u=torch.cat(bu), bond_v=torch.cat(bv), tri_bond=torch.cat(tb),
                tri_verts=torch.cat(tv), edge_feat=torch.cat(ef), node_feat=torch.cat(nf),
                tri_feat=torch.cat(tf), n_nodes=nb, Q=torch.cat(Q), target=torch.cat(tg),
                scale=torch.cat(sc).reshape(-1, 1, 1))


def predict(net, t):
    """Per-triangle C6 (and, for a SINGLE graph, the bulk mean).

    For a batch the bulk is meaningless -- `C_eff` is the mean over ONE network's triangles -- so it
    is returned only when the input is a single prepared sample, and `evaluate` uses that path."""
    G = net(t)
    scale = t['scale'] if 'scale' in t else t['phys'] * t['kbar']
    C_per, C_eff = M.assemble(t['Q'], G, physical_factor=scale)
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
    ap.add_argument('--batch', type=int, default=32, help='graphs per optimiser step')
    ap.add_argument('--no_angles', dest='angles', action='store_false',
                    help='ABLATION: degree-only node features, no triangle angles')
    ap.add_argument('--only_family', default=None, help='restrict to one family')
    ap.add_argument('--only_kpattern', default=None, help='restrict to one k-pattern')
    ap.add_argument('--keep_untrusted', action='store_true',
                    help='train on labels the independent sim disagrees with (default: exclude)')
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    raw = load(a.data)
    n_all = len(raw)
    # Untrusted labels are EXCLUDED, not silently absent: they are flagged in the dataset (the
    # builder keeps every sample, audit A-11/A-12) and filtered here, with the count reported so the
    # denominator stays honest.
    # Optional restriction to a sub-problem. `--only_family bravais --only_kpattern iid_uniform`
    # selects the ORDERED CRYSTALS at k = 1, where W == 0 (measured 1.3e-14) so C(s) = A(s) is a
    # closed form in the triangle's OWN three edges -- a purely LOCAL target needing no message
    # passing, and therefore the sharpest simple test of head plus features.
    if a.only_family:
        raw = [g for g in raw if g['family'] == a.only_family]
    if a.only_kpattern:
        raw = [g for g in raw if g['k_pattern'] == a.only_kpattern]
    if not raw:
        raise SystemExit('no samples left after --only_family/--only_kpattern')
    n_untrusted = sum(1 for g in raw if not g['sim_ok'])
    if not a.keep_untrusted:
        raw = [g for g in raw if g['sim_ok']]
    # The large-size bin is held out for size generalisation and is never trained on (section 3.1c).
    large = [g for g in raw if g['size_bin'] == 'large_holdout']
    raw = [g for g in raw if g['size_bin'] != 'large_holdout']
    # Counted BEFORE --limit truncates, or the limit is reported as exclusions: a 200-sample limit
    # on an 848-sample set claimed "648 untrusted excluded" when the true number was 12.
    n_limited = max(0, len(raw) - a.limit) if a.limit else 0
    if a.limit:
        raw = raw[:a.limit]
    print('%d samples from %s  (%d total, %d untrusted excluded, %d held-out large, %d cut by --limit)'
          % (len(raw), os.path.basename(a.data), n_all,
             0 if a.keep_untrusted else n_untrusted, len(large), n_limited))

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

    train = [prepare(raw[i], a.angles) for i in tr_i]
    valid = [prepare(raw[i], a.angles) for i in va_i]
    allt = torch.cat([t['target'] for t in train])
    mu, sd = allt.mean(0), allt.std(0).clamp_min(1e-12)

    n_ok = oracle_check(train + valid, [raw[i] for i in tr_i] + [raw[i] for i in va_i])
    print('oracle check: %d W=0 samples reproduce the stored target to machine precision'
          % n_ok if n_ok else 'oracle check: no W=0 sample in this split -- NOT verified')

    net = M.ForwardGNNv2(hidden=a.hidden, n_layers=a.layers, passive=True,
                         n_node_feat=train[0]['node_feat'].shape[1],
                         n_tri_feat=train[0]['tri_feat'].shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    print('%d parameters' % sum(p.numel() for p in net.parameters()))

    t0 = time.time()
    history = []
    order = np.arange(len(train))
    for ep in range(a.epochs):
        net.train()
        np.random.default_rng(ep).shuffle(order)
        tot = nb = 0.0
        for b0 in range(0, len(order), a.batch):
            t = collate([train[j] for j in order[b0:b0 + a.batch]])
            pc6, _ = predict(net, t)
            loss = (((pc6 - t['target']) / sd) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            tot += float(loss); nb += 1
        tot /= max(nb, 1)
        sched.step()
        if ep % max(1, a.epochs // 10) == 0 or ep == a.epochs - 1:
            per, bulk, bad = evaluate(net, valid, mu, sd)
            history.append(dict(epoch=ep, train_loss=float(tot),
                                val_per_tri=float((per / sd).mean()),
                                val_bulk=float((bulk / sd).mean()), spd=float(bad),
                                seconds=round(time.time() - t0, 1)))
            print('  ep %4d  train %.4f   val MAE/std per-tri %.4f  bulk %.4f   SPD viol %.4f  (%.0fs)'
                  % (ep, tot, float((per / sd).mean()), float((bulk / sd).mean()),
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
    # checkpoint name carries the SPLIT, so sequential runs cannot overwrite each other
    tag = (a.holdout if a.holdout != 'random' else 'within') + ('' if a.angles else '_noang')
    ck = os.path.join(HERE, 'checkpoint_v2_%s.pt' % tag)
    torch.save(dict(state=net.state_dict(), hidden=a.hidden, layers=a.layers,
                    mu=mu, sd=sd, holdout=a.holdout, data=os.path.basename(a.data),
                    n_node_feat=int(train[0]['node_feat'].shape[1]),
                    n_tri_feat=int(train[0]['tri_feat'].shape[1])), ck)
    print('  saved', ck)

    os.makedirs(RESULTS, exist_ok=True)
    rec = dict(split=split, holdout=a.holdout, data=os.path.basename(a.data),
               epochs=a.epochs, batch=a.batch, hidden=a.hidden, layers=a.layers, lr=a.lr,
               seed=a.seed, params=int(sum(p.numel() for p in net.parameters())),
               n_train=len(train), n_val=len(valid), n_total=n_all,
               n_untrusted_excluded=int(n_all - len(raw) - len(large)), n_large_holdout=len(large),
               per_triangle_mae=[float(x) for x in per], bulk_mae=[float(x) for x in bulk],
               label_std=[float(x) for x in sd],
               per_triangle_mae_over_std=float((per / sd).mean()),
               bulk_mae_over_std=float((bulk / sd).mean()),
               spd_violation_rate=float(bad), history=history,
               wall_seconds=round(time.time() - t0, 1), commit=_commit())
    out_json = os.path.join(RESULTS, 'run_%s.json' % tag)
    with open(out_json, 'w', encoding='utf-8') as fh:
        json.dump(rec, fh, indent=2)
    print('  metrics ->', out_json)
    return 0


def _commit():
    import subprocess
    try:
        return subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=REPO,
                              capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:                                                    # noqa: BLE001
        return 'unknown'


if __name__ == '__main__':
    sys.exit(main())
