r"""Score a trained M2 v2 surrogate the way `M2_V2_PLAN.md` section 1 asks -- against the SIM.

The trainer reports normalised `C6` error against the SOLVER labels it trained on.  That is the
wrong yardstick for the headline: section 1's must-tier is

    family-held-out  MAE(nu) <= 0.02  and  MAE(E)/E <= 5 %   against the INDEPENDENT SIM

and the kill criterion is MAE(nu) > 0.05.  `M2.md` records that the July validation scored a random
split against its own training labels, i.e. measured memorisation; this script exists so that cannot
happen again.

WHAT IT COMPARES, and against what
    model -> C_eff -> nu, E  (the solver's own `c6_to_nuE`, so no new convention enters)
    sim   -> C_eff -> nu, E  via `_common.sim_bulk_C6` -> `physical_homog.virial_C`, a DIFFERENT
             code path.  NOT `sim_region_C6`, which routes the sim's relaxation back through the
             solver's contraction and would be self-verification (`CLAUDE.md` section 3).

It also reports the two baselines the model must actually beat, because "MAE looks smallish" is not
a result:
    * predict the dataset's mean tensor everywhere;
    * predict each network's OWN bulk value at every triangle (the local-structure bar).

And it reports the SOLVER-vs-SIM gap on the same networks, which bounds how well ANY surrogate
trained on solver labels could score -- if the solver itself differs from the sim by more than the
target tolerance on a family, no model trained on it can meet the tolerance there.

Run:
    python "Phase 5/m2/evaluate_v2.py" --ckpt checkpoint_v2_bravais.pt --family bravais
"""
import argparse
import json
import os
import sys
import time
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'Phase 3', 'verifications'))
import _common as C                                                       # noqa: E402
sys.path.insert(0, os.path.join(REPO, 'Phase 5'))
sys.path.insert(0, HERE)
import torch                                                              # noqa: E402
import model_v2 as M                                                      # noqa: E402
import train_v2 as T                                                      # noqa: E402
from inverse_design import c6_to_nuE                                      # noqa: E402

warnings.simplefilter('ignore')
torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)
RESULTS = os.path.join(REPO, 'Phase 5', 'results', 'm2_s1')


def nuE(c6):
    nu, E = c6_to_nuE(torch.as_tensor(np.asarray(c6, float)))
    return float(nu), float(E)


def geo_of(g):
    """Rebuild a solver-ready geo dict from a stored sample (the dataset keeps no `edge_vecs`)."""
    ev = g['bond_R'][g['tri_bond'].astype(np.int64)]
    return dict(pts=g['pts'], simplices=g['tri_verts'].astype(np.int64),
                bond_u=g['bond_u'].astype(np.int64), bond_v=g['bond_v'].astype(np.int64),
                bond_R=g['bond_R'], tri_bond=g['tri_bond'].astype(np.int64),
                areas=g['areas'], edge_vecs=ev, actual_len2=(ev ** 2).sum(-1),
                centroids=np.zeros((len(g['tri_verts']), 2)),
                tri_verts=np.zeros((len(g['tri_verts']), 3, 2)),
                BL1=np.array([g['Lx'], 0.0]), BL2=np.array([0.0, g['Ly']]),
                # the sim reads its stiffness off the geo -- without this it would silently
                # relax a UNIFORM-k network and the comparison would be against the wrong material
                bond_k=np.asarray(g['k'], float),
                tri_k=np.asarray(g['k'], float)[g['tri_bond'].astype(np.int64)])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--ckpt', default='checkpoint_v2_bravais.pt')
    ap.add_argument('--data', default=os.path.join(HERE, 'data', 'dataset.npz'))
    ap.add_argument('--family', default='bravais', help='the held-out family to score')
    ap.add_argument('--max_n', type=int, default=250, help='cap on networks scored (sim is slow)')
    a = ap.parse_args()

    ck = torch.load(os.path.join(HERE, a.ckpt), weights_only=False)
    # A checkpoint is only loadable if it was trained with the CURRENT feature set. The angle
    # features (section 2.3) were added on 2026-08-26, changing node_embed 1 -> 4 inputs and the
    # readout 201 -> 204; earlier checkpoints are architecturally obsolete, not merely stale, and
    # saying so beats a torch size-mismatch dump.
    n_node = int(ck['state']['node_embed.weight'].shape[1])
    have = M.N_NODE_FEAT
    if n_node != have:
        raise SystemExit(
            'checkpoint %r expects %d node features but the current model builds %d. '
            'It predates the angle features of section 2.3 and is architecturally obsolete, not '
            'merely stale -- retrain, or check out the commit it was produced at.'
            % (a.ckpt, n_node, have))
    net = M.ForwardGNNv2(hidden=ck['hidden'], n_layers=ck['layers'],
                         n_node_feat=n_node, n_tri_feat=ck.get('n_tri_feat', 3))
    net.load_state_dict(ck['state'])
    net.eval()

    raw = [g for g in T.load(a.data) if g['sim_ok'] and g['size_bin'] != 'large_holdout']
    held = [g for g in raw if g['family'] == a.family]
    rng = np.random.default_rng(0)
    if len(held) > a.max_n:
        held = [held[i] for i in rng.choice(len(held), a.max_n, replace=False)]
    print('checkpoint %s (holdout=%s)  scoring %d networks of family %r'
          % (a.ckpt, ck.get('holdout'), len(held), a.family))

    rows, n_fail, t0 = [], 0, time.time()
    for i, g in enumerate(held):
        t = T.prepare(g)
        with torch.no_grad():
            _, bulk = T.predict(net, t)
        nu_m, E_m = nuE(bulk)
        nu_s, E_s = nuE(g['C6'])                       # solver label (what it trained on)
        try:
            nu_p, E_p = nuE(C.sim_bulk_C6(geo_of(g)))  # INDEPENDENT sim
        except Exception as e:                                            # noqa: BLE001
            # Report, never swallow. A bare `continue` here hid THREE bugs at once (missing box,
            # unapplied k, and its own silence) and produced an empty result array that failed far
            # downstream with an unrelated IndexError.
            n_fail += 1
            if n_fail <= 3:
                print('   sim failed on %s: %s: %s' % (g.get('family', '?'), type(e).__name__,
                                                       str(e)[:80]))
            continue
        rows.append((nu_m, E_m, nu_s, E_s, nu_p, E_p))
        if (i + 1) % 50 == 0:
            print('   %d/%d  (%.0fs)' % (i + 1, len(held), time.time() - t0))

    if not rows:
        raise SystemExit('every sim call failed (%d) -- see the messages above' % n_fail)
    if n_fail:
        print('   %d of %d sim calls failed' % (n_fail, n_fail + len(rows)))
    r = np.array(rows)
    nu_m, E_m, nu_s, E_s, nu_p, E_p = (r[:, j] for j in range(6))

    def rep(tag, dnu, dE_rel):
        print('  %-28s MAE(nu) %8.4f   median %8.4f   MAE(E)/E %7.2f %%   median %6.2f %%'
              % (tag, np.mean(np.abs(dnu)), np.median(np.abs(dnu)),
                 100 * np.mean(np.abs(dE_rel)), 100 * np.median(np.abs(dE_rel))))

    print('\nn = %d networks\n' % len(r))
    rep('MODEL vs SIM  (headline)', nu_m - nu_p, (E_m - E_p) / np.maximum(np.abs(E_p), 1e-30))
    rep('model vs solver labels', nu_m - nu_s, (E_m - E_s) / np.maximum(np.abs(E_s), 1e-30))
    rep('SOLVER vs SIM (the floor)', nu_s - nu_p, (E_s - E_p) / np.maximum(np.abs(E_s), 1e-30))
    rep('baseline: dataset mean nu', np.mean(nu_s) - nu_p,
        (np.mean(E_s) - E_p) / np.maximum(np.abs(E_p), 1e-30))

    must = np.mean(np.abs(nu_m - nu_p)) <= 0.02 and \
        np.mean(np.abs((E_m - E_p) / np.maximum(np.abs(E_p), 1e-30))) <= 0.05
    kill = np.mean(np.abs(nu_m - nu_p)) > 0.05
    print('\n  section 1 MUST tier (MAE(nu) <= 0.02 and MAE(E)/E <= 5 %%): %s' % ('MET' if must else 'NOT met'))
    print('  section 1 KILL criterion (MAE(nu) > 0.05): %s' % ('TRIGGERED' if kill else 'not triggered'))
    print('  NOTE the floor above: no model trained on solver labels can beat SOLVER-vs-SIM.')

    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, 'eval_%s.json' % a.family)
    with open(out, 'w', encoding='utf-8') as fh:
        json.dump(dict(ckpt=a.ckpt, family=a.family, n=len(r),
                       mae_nu_model_vs_sim=float(np.mean(np.abs(nu_m - nu_p))),
                       mae_nu_solver_vs_sim=float(np.mean(np.abs(nu_s - nu_p))),
                       mae_E_rel_model_vs_sim=float(np.mean(np.abs((E_m - E_p) / np.maximum(np.abs(E_p), 1e-30)))),
                       must_tier_met=bool(must), kill_triggered=bool(kill),
                       nu_model=nu_m.tolist(), nu_sim=nu_p.tolist(), nu_solver=nu_s.tolist(),
                       E_model=E_m.tolist(), E_sim=E_p.tolist(), E_solver=E_s.tolist()), fh, indent=2)
    print('  ->', out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
