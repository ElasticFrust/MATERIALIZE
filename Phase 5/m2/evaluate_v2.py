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
import model_v3 as M3                                                     # noqa: E402
import train_v2 as T                                                      # noqa: E402
import train_v3 as T3                                                     # noqa: E402
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
    ap.add_argument('--E_eps_frac', type=float, default=0.05, help="regularise the relative-E denominator as |E_sim| + eps_E, with eps_E = this fraction of the MEDIAN E_sim -- the convention CLAUDE.md section 3 already uses for nu (eps_nu = 0.05). Without it the metric is unusable: E_sim is BIMODAL, ~4 %% of networks at ~1e-5 against a median of 0.68, and those alone drove the mean relative error to 7648 %% against a median of 8.86 %%. Approved 2026-08-27.")
    ap.add_argument('--size_bin', default=None, choices=(None, 'train', 'large_holdout'),
                    help="score a SIZE bin instead of a family; 'large_holdout' is the ~1000-triangle "
                         "set that is never trained on (section 3.1c)")
    a = ap.parse_args()

    ck = torch.load(os.path.join(HERE, a.ckpt), weights_only=False)
    # v3 checkpoints carry 'ns'/'nt' (tensor channels); v2's do not. Detecting from the checkpoint
    # rather than a flag means a v2/v3 mix-up cannot be made silently on the command line.
    kind = 'v3' if 'ns' in ck else 'v2'
    if kind == 'v3':
        # ARCHITECTURE COMES FROM THE CHECKPOINT, never from this script's defaults -- the star and
        # M_S channels are optional, and a default-built model silently mismatches a run trained
        # with --no_star/--no_global. One shared builder, so the rule cannot drift between scripts.
        net = M3.from_checkpoint(ck)

        def predict_bulk(g):
            """C_eff = the UNWEIGHTED mean of the per-triangle C(s) -- the project's homogenisation
            (CLAUDE.md section 3); area weighting would bias nu on unequal-area meshes."""
            with torch.no_grad():
                return T3.predict(net, T3.prepare(g)).mean(0)
    else:
        net, predict_bulk = _load_v2(a, ck)

    _run(a, ck, kind, predict_bulk)
    return 0


def _load_v2(a, ck):
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

    def predict_bulk(g):
        with torch.no_grad():
            return T.predict(net, T.prepare(g, angles=n_node == M.N_NODE_FEAT))[1]

    return net, predict_bulk


def _run(a, ck, kind, predict_bulk):
    raw = T.load(a.data)
    if a.size_bin:
        # SIZE GENERALISATION (section 3.1c). `M2_LOCALITY.md` measured C(s) decorrelating in 2-3
        # hops and concluded a 4-5 layer receptive field suffices; the honest test of that claim is
        # a model trained only on 60-360-node graphs, scored on ~1000-triangle ones it never saw.
        # The large bin is excluded from training by the trainer, so this is a clean holdout.
        held = [g for g in raw if g['sim_ok'] and g['size_bin'] == a.size_bin]
        label = 'size bin %r' % a.size_bin
    else:
        held = [g for g in raw
                if g['sim_ok'] and g['size_bin'] != 'large_holdout' and g['family'] == a.family]
        label = 'family %r' % a.family
    rng = np.random.default_rng(0)
    if len(held) > a.max_n:
        held = [held[i] for i in rng.choice(len(held), a.max_n, replace=False)]
    if not held:
        raise SystemExit('no networks selected (%s)' % label)
    print('checkpoint %s [%s] (trained holdout=%s)  scoring %d networks -- %s'
          % (a.ckpt, kind, ck.get('holdout'), len(held), label))
    ntri = [len(g['C6_per']) for g in held]
    print('   n_tri  min %d  median %d  max %d' % (min(ntri), int(np.median(ntri)), max(ntri)))

    rows, n_fail, t0 = [], 0, time.time()
    for i, g in enumerate(held):
        nu_m, E_m = nuE(predict_bulk(g))
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

    eps_E = a.E_eps_frac * float(np.median(np.abs(E_p)))
    print('   relative-E denominator floored: eps_E = %.4g  (%.0f %% of median E_sim = %.4g)'
          % (eps_E, 100 * a.E_eps_frac, float(np.median(np.abs(E_p)))))

    def relE(dE, ref):
        return dE / (np.abs(ref) + eps_E)

    def rep(tag, dnu, dE_rel):
        print('  %-28s MAE(nu) %8.4f   median %8.4f   MAE(E)/E %7.2f %%   median %6.2f %%'
              % (tag, np.mean(np.abs(dnu)), np.median(np.abs(dnu)),
                 100 * np.mean(np.abs(dE_rel)), 100 * np.median(np.abs(dE_rel))))

    print('\nn = %d networks\n' % len(r))
    rep('MODEL vs SIM  (headline)', nu_m - nu_p, relE(E_m - E_p, E_p))
    rep('model vs solver labels', nu_m - nu_s, relE(E_m - E_s, E_s))
    rep('SOLVER vs SIM (the floor)', nu_s - nu_p, relE(E_s - E_p, E_s))
    rep('baseline: dataset mean nu', np.mean(nu_s) - nu_p, relE(np.mean(E_s) - E_p, E_p))
    # the UNFLOORED mean too, so the floor can never quietly manufacture a pass
    print('  %-28s %.4g   (unfloored, reference only)' % ('raw mean |dE|/|E_sim|:',
          float(np.mean(np.abs((E_m - E_p) / np.maximum(np.abs(E_p), 1e-30))))))

    must = np.mean(np.abs(nu_m - nu_p)) <= 0.02 and \
        np.mean(np.abs(relE(E_m - E_p, E_p))) <= 0.05
    kill = np.mean(np.abs(nu_m - nu_p)) > 0.05
    print('\n  section 1 MUST tier (MAE(nu) <= 0.02 and MAE(E)/E <= 5 %%): %s' % ('MET' if must else 'NOT met'))
    print('  section 1 KILL criterion (MAE(nu) > 0.05): %s' % ('TRIGGERED' if kill else 'not triggered'))
    print('  NOTE the floor above: no model trained on solver labels can beat SOLVER-vs-SIM.')

    os.makedirs(RESULTS, exist_ok=True)
    # Tag by CHECKPOINT, not just by model kind. Tagging by kind alone was not enough: the
    # depth-5 run overwrote the depth-3 result it was meant to be compared against (recovered
    # from git, but the next one might not be). The checkpoint stem is what actually
    # identifies a run.
    stem = os.path.splitext(os.path.basename(a.ckpt))[0].replace('checkpoint_', '')
    out = os.path.join(RESULTS, 'eval_%s_%s.json' % (stem, a.size_bin or a.family))
    with open(out, 'w', encoding='utf-8') as fh:
        json.dump(dict(ckpt=a.ckpt, model=kind, family=a.family, size_bin=a.size_bin,
                       scored=label, n=len(r),
                       mae_nu_model_vs_sim=float(np.mean(np.abs(nu_m - nu_p))),
                       mae_nu_solver_vs_sim=float(np.mean(np.abs(nu_s - nu_p))),
                       eps_E=float(eps_E), E_eps_frac=float(a.E_eps_frac),
                       mae_E_rel_model_vs_sim=float(np.mean(np.abs(relE(E_m - E_p, E_p)))),
                       mae_E_rel_unfloored=float(np.mean(np.abs(
                           (E_m - E_p) / np.maximum(np.abs(E_p), 1e-30)))),
                       must_tier_met=bool(must), kill_triggered=bool(kill),
                       nu_model=nu_m.tolist(), nu_sim=nu_p.tolist(), nu_solver=nu_s.tolist(),
                       E_model=E_m.tolist(), E_sim=E_p.tolist(), E_solver=E_s.tolist()), fh, indent=2)
    print('  ->', out)


if __name__ == '__main__':
    sys.exit(main())
