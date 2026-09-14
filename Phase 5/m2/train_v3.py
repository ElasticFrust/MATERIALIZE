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
    # CURVATURE constraint coupling (C_curv): the vertex stars and their dtheta/dg weights, verified
    # identical to the solver's operator by `test_m2_constraints.py`. Lengths are normalised by
    # `lbar` here too, so the weights are computed on the SAME geometry the carriers use.
    st_t, st_v, st_w, n_vert = M3.vertex_stars(g['tri_verts'], tb, g['bond_u'], g['bond_v'],
                                               bR / lbar)
    bq = M2.edge_carriers((bR / lbar)[:, None, :])[:, :, 0]   # (n_bond, 3) per-bond carrier

    t = dict(Q=Q, tri_src=src, tri_dst=dst,
             tri_scalars=torch.as_tensor(np.concatenate([kk[tb], l0[tb], np.log(kk[tb] + 1e-12)], 1)),
             bond_q=bq[bnd],
             bond_feat=torch.as_tensor(np.stack([kk[bnd], l0[bnd]], 1)),
             star_tri=torch.as_tensor(st_t), star_vert=torch.as_tensor(st_v),
             star_w=torch.as_tensor(st_w), n_vert=int(n_vert),
             # M_S weights: areas normalised by their own sum, so the channel is scale-free and
             # only the RELATIVE area distribution enters -- which is all the constraint uses.
             area_w=torch.as_tensor(np.asarray(g['areas'], float)
                                    / max(float(np.sum(g['areas'])), 1e-30)),
             # A SINGLE sample is a batch of one, and it must say so. `forward` enables the M_S
             # channel on `'tri_batch' in g`; without these two keys the channel silently switched
             # OFF for one-at-a-time evaluation and ON for batched training -- so the model would
             # have been scored under a different architecture than it was trained with. Caught by
             # comparing batched against one-at-a-time output (7.8e-02 apart).
             tri_batch=torch.zeros(len(tb), dtype=torch.long), n_graphs=1,
             target=torch.as_tensor(g['C6_per']),
             n_tri=len(tb), n_bond=len(bR))
    t['phys'] = 8.0 * len(tb) / torch.as_tensor(g['areas']).sum() * (lbar ** 2)
    t['kbar'] = torch.as_tensor(k).mean()
    return t


def collate(ts):
    """Block-diagonal batch.  Triangle indices offset by triangle count, not node count."""
    to = vo = 0
    Q, src, dst, tsc, bq, bf, tg, sc = [], [], [], [], [], [], [], []
    stt, stv, stw, aw, tb = [], [], [], [], []
    for gi, t in enumerate(ts):
        Q.append(t['Q']); tsc.append(t['tri_scalars']); tg.append(t['target'])
        src.append(t['tri_src'] + to); dst.append(t['tri_dst'] + to)
        bq.append(t['bond_q']); bf.append(t['bond_feat'])
        sc.append(torch.full((t['n_tri'],), float(t['phys'] * t['kbar'])))
        # VERTEX indices need their OWN offset -- they index a per-sample vertex array, not the
        # triangle array, so reusing the triangle offset would silently fuse stars across graphs.
        stt.append(t['star_tri'] + to); stv.append(t['star_vert'] + vo); stw.append(t['star_w'])
        aw.append(t['area_w'])
        tb.append(torch.full((t['n_tri'],), gi, dtype=torch.long))   # M_S pools PER GRAPH
        to += t['n_tri']; vo += t['n_vert']
    return dict(Q=torch.cat(Q), tri_src=torch.cat(src), tri_dst=torch.cat(dst),
                tri_scalars=torch.cat(tsc), bond_q=torch.cat(bq), bond_feat=torch.cat(bf),
                star_tri=torch.cat(stt), star_vert=torch.cat(stv), star_w=torch.cat(stw),
                n_vert=vo, area_w=torch.cat(aw), tri_batch=torch.cat(tb), n_graphs=len(ts),
                target=torch.cat(tg), scale=torch.cat(sc).reshape(-1, 1, 1))


def bulk_mean(x, tri_batch, n_graphs):
    """UNWEIGHTED per-graph mean of a per-triangle quantity -- the project's homogenisation.

    `C_eff = (1/N) sum_s C(s)` (CLAUDE.md section 3); area weighting is used ONLY in the M_S
    constraint and would bias nu on unequal-area meshes. Pooled per graph via `tri_batch`, never
    across the batch, for the same reason `GlobalMS` is: collate packs graphs block-diagonally."""
    cnt = torch.zeros(n_graphs, 1, dtype=x.dtype, device=x.device).index_add(
        0, tri_batch, torch.ones(len(tri_batch), 1, dtype=x.dtype, device=x.device))
    tot = torch.zeros(n_graphs, x.shape[1], dtype=x.dtype, device=x.device).index_add(
        0, tri_batch, x)
    return tot / cnt.clamp_min(1.0)


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
        # LOAD-BEARING AND PREVIOUSLY UNSTATED: this check uses the RAW `k` while `assemble` is given
        # `phys * kbar`, so the two agree only because the builder normalises every sample to
        # mean(k) = 1 (verified: 4000/4000 samples exactly 1). `model_v3.analytic_G` uses the
        # NORMALISED k instead, which is correct either way -- but if this ever fires, that
        # discrepancy is live and both paths need revisiting.
        kbar = float(np.mean(np.asarray(g['k'], float)))
        if abs(kbar - 1.0) > 1e-9:
            raise SystemExit('oracle_check assumes mean(k) == 1 (got %.6g for family %r). The raw-k '
                             'convention here and the kbar factor in `assemble` would disagree.'
                             % (kbar, g.get('family', '?')))
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


def check_init_compat(a, ck):
    """FAIL FAST if a `--init_from` checkpoint's architecture differs from this run's.

    Raises `SystemExit` on the first difference; returns None when the two match.

    `load_state_dict` raises on a SHAPE mismatch by itself, so that is not what this is for. It is
    for the differences that keep shapes and load SILENTLY: `use_star` / `use_global` toggle whole
    sub-modules, so a checkpoint trained with a channel can be loaded into a model built without it
    (or the reverse) and will train, and look plausible, on half-loaded weights. `head` is checked
    for the same reason -- a different head version means the tensor being predicted is not the same
    object.

    Lives here, rather than inline in `main`, so `test_m2_train_surface.py [2]` exercises THE SHIPPED
    GUARD instead of a copy of it. A test that reimplements the logic it is testing cannot catch a
    regression in the original.
    """
    for k in ('ns', 'nt', 'hidden', 'layers'):
        if int(ck.get(k, -1)) != int(getattr(a, k)):
            raise SystemExit('--init_from architecture mismatch: checkpoint %s=%s, this run %s=%s'
                             % (k, ck.get(k), k, getattr(a, k)))
    for k, want in (('use_star', not a.no_star), ('use_global', not a.no_global)):
        if bool(ck.get(k, True)) != bool(want):
            raise SystemExit('--init_from channel mismatch: checkpoint %s=%s, this run %s=%s'
                             % (k, ck.get(k), k, want))
    if ck.get('head') != M3.HEAD_VERSION:
        raise SystemExit('--init_from head mismatch: checkpoint head=%s, this run %s'
                         % (ck.get('head'), M3.HEAD_VERSION))


def _make_sched(a, opt):
    """The LR schedule. One definition, because `--restart_lr` has to rebuild it mid-run.

    `plateau` decays ON EVIDENCE OF STALLING, not on a clock; `threshold_mode='rel'` makes
    `min_delta` a relative improvement so the bar scales with the current score. `cosine` is retained
    ONLY so the pre-2026-08-28 runs stay reproducible -- it reaches ~0 lr at `T_max` and so flattens
    the loss curve BY CONSTRUCTION, which once manufactured a "converged" result (CLAUDE.md section 3).
    """
    if a.schedule == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.3, patience=a.patience,
        threshold=a.min_delta, threshold_mode='rel')


def run_tag(a):
    """The run's identity, and the name of its checkpoint.

    Must encode ARCHITECTURE and EPOCHS, not just the data flags: an earlier version keyed only on
    holdout+filter+huber, so a 2-epoch throughput benchmark sharing those flags silently overwrote a
    220-epoch checkpoint -- 11.5 h of compute lost, and the junk committed in its place because the
    metadata was never checked.

    The DATASET is part of the identity too (added 2026-09-12). `--resume` looks up `<tag>.resume` by
    name alone, so without it a resume against a different `--data` silently continues a trajectory
    on DIFFERENT SAMPLES: `--overfit_probe n` draws its n by `rng(seed).choice(len(raw))`, and `raw`
    depends on the file. Measured that morning -- resuming the 200-sample probe against
    `dataset.npz` (9 715 usable) instead of `dataset_v2_s0.npz` (40 775) changed which 200 samples
    were drawn, and only the `keeps 1956/9715` log line gave it away. The basename was already
    recorded INSIDE the saved `.pt`, which makes a finished run traceable but cannot stop the
    filename collision or the wrong-dataset resume."""
    tag = '%s_%s' % (M3.HEAD_VERSION, a.holdout if a.holdout != 'random' else 'within')
    # dataset: `dataset_v2_s0.npz` -> `_dv2s0`, plain `dataset.npz` -> `_dbase`. Alphanumerics only,
    # so the tag stays a legal filename on every platform.
    _stem = os.path.splitext(os.path.basename(a.data))[0]
    _stem = _stem[len('dataset'):].lstrip('_') if _stem.startswith('dataset') else _stem
    tag += '_d%s' % (''.join(c for c in _stem if c.isalnum()) or 'base')
    if a.w_max_cut or a.huber:
        tag += '_w%g_h%g' % (a.w_max_cut, a.huber)
    tag += '_L%d_ns%d_h%d_e%d' % (a.layers, a.ns, a.hidden, a.epochs)
    if not getattr(a, 'no_star', False):
        tag += '_star'                                        # architecture, so part of the identity
    if not getattr(a, 'no_global', False):
        tag += '_ms'
    if getattr(a, 'overfit_probe', 0):
        tag += '_probe%d_c%g' % (a.overfit_probe, a.contrast_min)
    if getattr(a, 'tie', False):
        # a tied model is a DIFFERENT architecture from a stacked one of the same depth
        tag += '_tie%d' % (a.n_iter or a.layers)
        if getattr(a, 'n_iter_hi', 0):
            tag += '-%d' % a.n_iter_hi
    if getattr(a, 'init_from', ''):
        # WEIGHTS-ONLY warm start: a different starting point is a different run. The tag carries a
        # short digest of the source basename rather than the basename itself (which is ~70 chars);
        # the full path goes in the checkpoint metadata and the run JSON, so it stays traceable.
        import hashlib
        tag += '_init%s' % hashlib.sha1(
            os.path.basename(a.init_from).encode('utf-8')).hexdigest()[:6]
    if getattr(a, 'restart_lr', 0.0):
        # a restarted trajectory is a DIFFERENT run from its parent, and two restarts at different
        # RATES are different runs from each other
        tag += '_rlr%g' % a.restart_lr
    if getattr(a, 'graph_balance', False):
        tag += '_gb'                           # a different objective is a different run
    if getattr(a, 'bulk_weight', 0):
        tag += '_b%g' % a.bulk_weight           # a different objective is a different run
    if a.opt != 'adam' or a.wd:
        tag += '_%s%g' % (a.opt, a.wd)                        # optimiser is part of the identity
    if a.holdout == 'random' and getattr(a, 'mesh_frac', 0):
        tag += '_m%g' % a.mesh_frac            # a second holdout changes the TRAIN set: part of id
    if a.limit:
        tag += '_lim%d' % a.limit                             # probes can never look like real runs
    if getattr(a, 'run_suffix', ''):
        # For runs the other fields CANNOT separate -- successive `--restart_lr` rounds at the SAME
        # rate being the case this exists for. Without it round 2 wears round 1's name, and saving it
        # needs `--force`, i.e. overwriting the very result it is trying to improve on.
        tag += '_%s' % ''.join(c for c in a.run_suffix if c.isalnum())
    return tag


def evaluate(net, samples, sd):
    """-> (per-triangle MAE per component, SPD violation rate, BULK MAE per component).

    The bulk figure is reported because `--bulk_weight` trades the two against each other: diverting
    gradient to `C_eff` makes the per-triangle metric WORSE (measured, 0.6841 -> 0.7326 after one
    epoch at weight 1.0). Reporting only the per-triangle number would make that read as a
    regression, when it is the intended exchange -- and `nu`/`E`, the quantities the tiers are
    defined on, are read off the BULK."""
    net.eval()
    ae, be, spd_bad, n_tri = [], [], 0, 0
    with torch.no_grad():
        for t in samples:
            p = predict(net, t)
            ae.append((p - t['target']).abs().mean(0))
            be.append((p.mean(0) - t['target'].mean(0)).abs())
            eig = torch.linalg.eigvalsh(M2.c6_to_sym3(p))
            spd_bad += int((eig.min(-1).values < -1e-10).sum())
            n_tri += len(p)
    if not ae:
        z = torch.zeros(6, dtype=sd.dtype)
        return z, 0.0, z
    return torch.stack(ae).mean(0), spd_bad / max(n_tri, 1), torch.stack(be).mean(0)


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
    ap.add_argument('--opt', default='adam', choices=('adam', 'adamw'),
                    help="'adamw' DECOUPLES weight decay from the adaptive scaling: Adam adds "
                         "wd*theta to the gradient, which is then divided by sqrt(v_hat), so "
                         "parameters with small gradients get MORE effective decay -- not what "
                         "anyone intends. At --wd 0 the two are IDENTICAL, so this flag is really "
                         "the question 'should we use weight decay'.")
    ap.add_argument('--wd', type=float, default=0.0,
                    help='weight decay. DEFAULT 0, and that is a measured choice, not a default '
                         'reached for: weight decay on an UNDERFIT model makes it worse, and at '
                         'train 0.3921 vs val 0.3906 there was no gap to close. The deep run then '
                         'showed a 21 %% gap, so this is worth testing -- but note that with '
                         'LayerNorm/tensor_rms_norm downstream, decay on a pre-normalisation weight '
                         'leaves the output unchanged and acts as an EFFECTIVE-LR modifier rather '
                         'than regularisation. Separate those before calling a gain regularisation.')
    ap.add_argument('--schedule', default='plateau', choices=('plateau', 'cosine'),
                    help="'plateau' (default) decays the LR only when the VALIDATION SCORE STOPS "
                         "IMPROVING, and stops when decaying no longer helps -- so the run length is "
                         "set by the data, not by a clock. 'cosine' anneals to zero at --epochs, "
                         "which MANUFACTURES a flat tail: the curve levels off because the learning "
                         "rate reached zero, not because the model stopped learning, and calling "
                         "that 'converged' is an artefact of the schedule (user, 2026-08-28).")
    ap.add_argument('--eval_every', type=int, default=2,
                    help='epochs between validation passes; the plateau scheduler steps on these')
    ap.add_argument('--patience', type=int, default=4,
                    help='evaluations without improvement before the LR is cut (plateau only)')
    ap.add_argument('--stop_patience', type=int, default=12,
                    help='evaluations without improvement before STOPPING (plateau only)')
    ap.add_argument('--min_delta', type=float, default=2e-2,
                    help='relative val improvement that counts as progress. DEFAULT 2 %%, raised '
                         'from 0.2 %% after measuring the evaluation-to-evaluation scatter at ~9 %% '
                         'early and ~0.5 %% late: a threshold 40x below the noise means random '
                         'fluctuation reads as progress and the run never stops.')
    ap.add_argument('--no_global', action='store_true',
                    help='disable the M_S (area-weighted global mean) channel -- rank 3, and the '
                         'quantity it zeroes is exactly the mean the channel computes.')
    ap.add_argument('--no_star', action='store_true',
                    help='disable the VERTEX-STAR (curvature-constraint) message channel. The model '
                         "already had J_edge's coupling for free -- `triangle_adjacency` IS the "
                         'edge-compatibility pairing -- but C_curv, which couples the ~6 triangles '
                         'of a vertex star simultaneously, had no channel at all. This flag exists '
                         'so its effect can be measured as a single variable.')
    ap.add_argument('--tie', action='store_true',
                    help='WEIGHT-TIED ITERATION: one message-passing block applied --n_iter times, '
                         'instead of --layers distinct blocks used once each. `W(s)` is an operator '
                         'INVERSE, so a fixed-depth net is a fixed number of relaxation sweeps; '
                         'plain depth is the wrong way to buy more, since it adds parameters and '
                         'does not train (depth 8 DIVERGED, depth 10 STALLED). Tying decouples the '
                         'iteration count from both.')
    ap.add_argument('--n_iter', type=int, default=0,
                    help='block applications per forward pass (0 = --layers). With --tie this is '
                         'free, and can be changed AFTER training to sweep it at inference -- which '
                         'is the direct test of the iteration hypothesis.')
    ap.add_argument('--n_iter_hi', type=int, default=0,
                    help='if > 0, JITTER the iteration count per batch, uniform in '
                         '[--n_iter, --n_iter_hi]. Without this a later inference-time sweep over '
                         'n_iter is an EXTRAPOLATION; with it the sweep is in-distribution and the '
                         'test means something. Evaluation always uses the nominal --n_iter.')
    ap.add_argument('--init_from', default='',
                    help='WEIGHTS-ONLY warm start from a FINAL checkpoint (.pt), with a fresh '
                         'optimiser and scheduler. Distinct from --resume, which continues a '
                         'trajectory and needs the optimiser state in a <tag>.resume whose name '
                         'matches this run -- so --resume cannot cross a config change, and a change '
                         'of --data or --w_max_cut is exactly such a change. Use this to fine-tune an '
                         'existing arm on a different training set: the measured alternative is '
                         'retraining from scratch, 129 h vs 18.5 h for the unfiltered set. The source '
                         "checkpoint's architecture must match this run's exactly -- checked and "
                         'FAIL-FAST, because load_state_dict would otherwise either raise deep in the '
                         'stack or, on a compatible-but-wrong config, load silently. Mutually '
                         'exclusive with --resume.')
    ap.add_argument('--run_suffix', default='',
                    help='appended to the run tag, for runs the other fields cannot tell apart -- '
                         'successive --restart_lr rounds at the SAME rate above all. Alphanumerics '
                         'are kept, everything else dropped, so the tag stays a legal filename.')
    ap.add_argument('--restart_lr', type=float, default=0.0,
                    help='LR-RESTART PROBE (CLAUDE.md section 3), as a RATE; 0 = off. With --resume, '
                         'set the lr to this value and train on -- the decisive test of whether a '
                         'flat tail is the model finding the bottom or the schedule no longer '
                         'letting it walk, which a decay-driven ending cannot tell apart. Needed as '
                         'a flag because --resume alone cannot do it: opt.load_state_dict restores '
                         'the DECAYED lr (measured: 2.19e-06), so even a fresh scheduler keeps it. '
                         'A RATE rather than a switch because restoring the INITIAL lr was measured '
                         'DESTRUCTIVE on the 8-sample probe: 3e-3 threw a converged 0.1936 up to '
                         '0.73 and it stalled at 0.705, never re-entering the basin, so the test '
                         'could not interrogate the floor at all. Pick a rate at which the run was '
                         'observably still making progress (2.7e-4 there), not the initial one. '
                         'Resets the optimiser lr, the scheduler (fresh, so an inherited `best` '
                         'cannot re-cut at once) and the stop clock; `best` is KEPT, since beating '
                         'it is the measurement.')
    ap.add_argument('--resume', action='store_true',
                    help='continue from the last per-evaluation snapshot (<checkpoint>.resume) if '
                         'one exists: model, optimiser, scheduler, best-so-far and history, so the '
                         'run continues its trajectory instead of restarting.')
    ap.add_argument('--force', action='store_true',
                    help='allow overwriting an existing checkpoint of the same tag')
    ap.add_argument('--overfit_probe', type=int, default=0,
                    help='CAPACITY TEST, not a training run. Train on this many samples and '
                         'VALIDATE ON THE SAME ONES, so the reported score is the error on data the '
                         'model may see unlimited times. The labels are a DETERMINISTIC function of '
                         'the inputs (the solver computes C(s) from exactly the geometry and k the '
                         'model receives), so there is no label noise and no irreducible floor: any '
                         'residual error is the architecture. Error -> 0 means the architecture CAN '
                         'represent the target and the shortfall is data or optimisation; error '
                         'stuck means it cannot, and running it at two DEPTHS separates reach from '
                         'expressivity.')
    ap.add_argument('--contrast_min', type=float, default=0.0,
                    help='keep only samples with k-contrast (max k / min k) at or above this. The '
                         'error rises 26x with contrast, so a capacity test on an unselected sample '
                         'would be dominated by the easy end and answer the wrong question.')
    ap.add_argument('--graph_balance', action='store_true',
                    help='average the per-triangle loss PER GRAPH before averaging over the '
                         'batch, so every mesh contributes equally regardless of triangle '
                         'count (default: average over triangles, which weights a '
                         '700-triangle mesh ~44x a 16-triangle one). The cheaper alternative '
                         'to --bulk_weight for the same problem; they can be combined but '
                         'then the effect of neither is attributable.')
    ap.add_argument('--bulk_weight', type=float, default=0.0,
                    help='weight on a BULK loss term -- the per-graph mean C6 (i.e. C_eff, what nu '
                         'and E are read from) against its own sigma, added to the per-triangle '
                         'term. DEFAULT 0, which reproduces every run before 2026-09-11 exactly; '
                         'it is NOT a claim that 0 is right. The per-triangle loss averages over '
                         'TRIANGLES, so a 700-triangle mesh contributes ~60x the gradient of a '
                         '12-triangle one, and small meshes also have the fewest triangles for '
                         'errors to cancel over -- measured, `cells` reaches MAE(nu) 0.1613 against '
                         '`random` 0.0256 at near-identical per-triangle error. The right value '
                         'must be SWEPT: too large and the model fits C_eff while the per-triangle '
                         'field (which the edit-policy needs) degrades, which would look like a win '
                         'on every number currently reported.')
    ap.add_argument('--mesh_frac', type=float, default=0.10,
                    help='fraction of MESHES reserved as a SECOND, mesh-disjoint holdout when '
                         "--holdout random. A random sample split cannot answer \"a new mesh\": "
                         'each mesh carries ~26 samples differing only in the k-field, so ~21 land '
                         'in train and ~5 in val. Reserving whole meshes costs one extra evaluation '
                         'per cycle and measures the gap between "new k, known mesh" and "new mesh". '
                         'Never used for selection or stopping -- that would fit it too. 0 disables.')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--w_max_cut', type=float, default=0.0,
                    help='drop TRAINING samples with max|W| above this (0 = keep all). Near-mechanism '
                         'networks are where the solver is least trustworthy, and measured 7x worse: '
                         'MAE(nu) 0.12 at max|W|<3 rising to 0.86 above 200. They are rare, so this '
                         'is a ~20 %% effect on the total error, NOT a fix on its own. The HOLDOUT is '
                         'never filtered -- the domain restriction has to stay visible.')
    ap.add_argument('--huber', type=float, default=0.0,
                    help='Huber delta on the NORMALISED residual (0 = plain squared error). The label '
                         'distribution is heavy-tailed, so a few triangles dominate the squared-error '
                         'gradient; Huber bounds their influence. This is NOT regularisation -- the '
                         'model underfits (train 0.3921 vs val 0.3906), so weight decay or dropout '
                         'would push the wrong way.')
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    raw = [g for g in T2.load(a.data) if g['sim_ok'] and g['size_bin'] != 'large_holdout']
    if a.limit:
        # RANDOM subsample, not raw[:n]. The builder yields families in order (cells first), so a
        # head slice is a biased subset -- with --limit 400 the holdout family vanished entirely,
        # and the same trap produced a 10x benchmark misestimate earlier in this project.
        sel = np.random.default_rng(a.seed).choice(len(raw), min(a.limit, len(raw)), replace=False)
        raw = [raw[i] for i in sorted(sel)]
    if a.contrast_min:
        n0 = len(raw)
        raw = [g for g in raw if float(g.get('contrast', 0.0)) >= a.contrast_min]
        print('contrast filter: >= %g keeps %d/%d samples' % (a.contrast_min, len(raw), n0))
        if not raw:
            raise SystemExit('no samples at or above contrast %g' % a.contrast_min)
    if a.overfit_probe:
        # TRAIN AND VALIDATE ON THE SAME SAMPLES. Deliberate, and the whole point of the mode:
        # the question is not "does it generalise" but "can this architecture represent the target
        # at all", and the labels carry no noise for it to hide behind.
        sel = np.random.default_rng(a.seed).choice(len(raw), min(a.overfit_probe, len(raw)),
                                                   replace=False)
        tr = va = [raw[i] for i in sorted(sel)]
        va_mesh = []
        split = ('OVERFIT PROBE: %d samples, train == val (CAPACITY test, not a generalisation '
                 'measurement)' % len(tr))
    elif a.holdout != 'random':
        tr = [g for g in raw if g['family'] != a.holdout]
        va = [g for g in raw if g['family'] == a.holdout]
        va_mesh = []          # no second holdout: the family IS mesh-disjoint
        split = 'LEAVE-ONE-FAMILY-OUT: %s' % a.holdout
    else:
        # TWO HOLDOUTS FROM ONE RUN, because they answer DIFFERENT questions and the gap
        # between them is itself the measurement (user, 2026-09-11).
        #
        #   (a) MESH-DISJOINT  -- whole meshes reserved, so no k-field of that mesh is ever seen.
        #       Question: "a new MESH". This is the deployment number.
        #   (b) RANDOM SAMPLE  -- a seeded fraction of what remains. Each mesh carries ~26 samples
        #       differing only in the k-field, so ~21 of a mesh's samples land in train and ~5 in
        #       val: essentially every val sample sits on a mesh seen in training.
        #       Question: "a new k-FIELD on a KNOWN mesh" -- exactly Phase 3's inner loop, where
        #       `optimize` designs k on a fixed mesh.
        #
        # (b) is NOT merely memorisation: the mesh is an INPUT (q_e, areas, star weights are passed
        # every time), so there is no hidden lookup. But it IS mildly optimistic vs (a), and the
        # (a)-(b) gap is the first measurement of by how much.
        #
        # SELECTION AND STOPPING KEY ON (b) ONLY. (a) is therefore never used to choose weights or
        # to stop, which is what keeps it a clean holdout rather than a second thing being fitted.
        #
        # (b) is keyed on `traj_id`, intended as a trajectory grouping -- but MEASURED 2026-09-11,
        # every traj_id holds EXACTLY ONE sample (40775 / 40775), so the grouping currently does
        # nothing and this is a plain random sample split. Left keyed on it so it becomes a real
        # group split the moment multi-step trajectories are stored.
        rng = np.random.default_rng(a.seed)
        meshes = sorted({g['mesh_id'] for g in raw})
        n_mesh = int(round(a.mesh_frac * len(meshes)))
        held = set(np.asarray(meshes)[rng.permutation(len(meshes))[:n_mesh]].tolist()) if n_mesh else set()
        va_mesh = [g for g in raw if g['mesh_id'] in held]
        rest = [g for g in raw if g['mesh_id'] not in held]
        trajs = np.asarray(sorted({g['traj_id'] for g in rest}))
        val = set(trajs[rng.permutation(len(trajs))[:max(1, len(trajs) // 5)]].tolist())
        tr = [g for g in rest if g['traj_id'] not in val]
        va = [g for g in rest if g['traj_id'] in val]
        split = ('RANDOM 20 %% sample split (seed %d) -- new k on a KNOWN mesh; '
                 'plus %d/%d MESHES (%d samples) held out entirely'
                 % (a.seed, len(held), len(meshes), len(va_mesh)))
    if a.w_max_cut and not a.overfit_probe:
        n0 = len(tr)
        tr = [g for g in tr if float(g['w_max']) <= a.w_max_cut]
        print('near-mechanism filter: max|W| <= %g keeps %d/%d training samples (%.1f %% dropped); '
              'holdout left UNFILTERED' % (a.w_max_cut, len(tr), n0, 100 * (1 - len(tr) / max(n0, 1))))
    if not tr or not va:
        raise SystemExit('empty split: train %d, val %d (holdout=%r). With --limit the holdout '
                         'family may not have survived the subsample.' % (len(tr), len(va), a.holdout))
    print('%s\n  train %d / val %d' % (split, len(tr), len(va)))

    train = [prepare(g) for g in tr]
    valid = [prepare(g) for g in va]
    # the SECOND holdout, scored every cycle but NEVER used for selection or stopping
    valid_mesh = [prepare(g) for g in va_mesh]
    if valid_mesh:
        print('  + mesh-disjoint holdout: %d samples on %d unseen meshes'
              % (len(valid_mesh), len({g['mesh_id'] for g in va_mesh})))
    print('oracle check: %d W=0 samples exact' % oracle_check(train + valid, tr + va))
    sd = torch.cat([t['target'] for t in train]).std(0).clamp_min(1e-12)
    # separate sigma for the BULK term: the per-graph mean C6 is a much narrower
    # distribution than the per-triangle one, and normalising it by `sd` would make the
    # term negligible by construction rather than by choice.
    sd_bulk = torch.stack([t['target'].mean(0) for t in train]).std(0).clamp_min(1e-12)

    net = M3.ForwardGNNv3(ns=a.ns, nt=a.nt, hidden=a.hidden, n_layers=a.layers,
                          use_star=not a.no_star, use_global=not a.no_global,
                          tie=a.tie, n_iter=a.n_iter)
    if a.tie:
        print('WEIGHT-TIED: 1 block applied %d times%s (untied depth-%d would be %d distinct blocks)'
              % (net.n_iter,
                 ' , jittered per batch in [%d, %d]' % (net.n_iter, a.n_iter_hi)
                 if a.n_iter_hi else '', a.layers, a.layers))
    print('%d parameters  (ns=%d nt=%d hidden=%d layers=%d)'
          % (sum(p.numel() for p in net.parameters()), a.ns, a.nt, a.hidden, a.layers))
    if a.init_from:
        if a.resume:
            raise SystemExit('--init_from and --resume are mutually exclusive: one starts a NEW '
                             'trajectory from borrowed weights, the other continues an existing one.')
        ck0 = torch.load(a.init_from, weights_only=False)
        check_init_compat(a, ck0)
        net.load_state_dict(ck0['state'])
        print('INIT FROM %s  (trained on %s, w_max_cut=%s, n_train=%s) -- WEIGHTS ONLY, fresh '
              'optimiser and scheduler' % (os.path.basename(a.init_from), ck0.get('data'),
                                           ck0.get('w_max_cut'), ck0.get('n_train')))
    opt = (torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd) if a.opt == 'adamw'
           else torch.optim.Adam(net.parameters(), lr=a.lr, weight_decay=a.wd))
    sch = _make_sched(a, opt)

    # Track the BEST epoch's weights, not just the last. Training is NOT monotone here: the
    # depth-5 run jumped from val 0.4017 to 0.6522 in eight epochs before recovering, and a run
    # ending on such a spike would save a bad model while the history JSON showed a good epoch
    # existed. Costs one state_dict copy per evaluation.
    best = dict(val=float('inf'), epoch=-1, state=None)
    last_sig = dict(val=float('inf'), epoch=0)      # last SIGNIFICANT improvement; drives stopping
    t0, hist, order = time.time(), [], np.arange(len(train))

    ckpt_path = os.path.join(HERE, 'checkpoint_v3_%s.pt' % run_tag(a))
    resume_path = ckpt_path + '.resume'
    start_ep = 0
    # RESUME. A run is many hours; a power cut, a sleep or an interrupted session used to cost all of
    # it, because the checkpoint was only written after the loop. `--resume` picks up the last
    # snapshot: model, optimiser, scheduler, best-so-far and history, so the continuation is the same
    # trajectory rather than a fresh run wearing the same name.
    if a.resume and os.path.exists(resume_path):
        ck = torch.load(resume_path, weights_only=False)
        net.load_state_dict(ck['state']); opt.load_state_dict(ck['opt'])
        try:
            sch.load_state_dict(ck['sch'])
        except Exception:                                                # noqa: BLE001
            print('  (scheduler state not restorable -- continuing with a fresh scheduler)')
        best, last_sig, hist = ck['best'], ck['last_sig'], ck['hist']
        start_ep = int(ck['epoch']) + 1
        t0 = time.time() - float(ck.get('elapsed', 0.0))
        print('RESUMED from %s at epoch %d (best %.4f at ep %d)'
              % (resume_path, start_ep, best['val'], best['epoch']))
        if a.restart_lr:
            for grp in opt.param_groups:
                grp['lr'] = a.restart_lr
            sch = _make_sched(a, opt)          # fresh: an inherited `best` would re-cut at once
            # The stop clock is reset to INFINITY, not to the inherited `best`. Seeding it with
            # `best` makes the criterion "improve on the old best within stop_patience evaluations",
            # which a restart that first LOSES ground can essentially never satisfy -- the 3e-3
            # restart was killed 209 epochs in while still descending from 0.73, judged against a
            # 0.1936 it had been thrown far away from. The restart must be judged on its OWN
            # trajectory; whether it beat the inherited best is read off `best` at the end.
            last_sig = dict(val=float('inf'), epoch=start_ep)
            print('  LR RESTART: lr -> %.2e, fresh scheduler, stop clock reset (judged on its OWN '
                  'trajectory). Going below the inherited best %.4f means the flat tail was the '
                  'SCHEDULE, not the model.' % (a.restart_lr, best['val']))
    elif a.resume:
        print('  --resume given but %s does not exist; starting fresh' % resume_path)
    for ep in range(start_ep, a.epochs):
        net.train()
        np.random.default_rng(ep).shuffle(order)
        tot = nb = 0.0
        nominal_iter = net.n_iter
        for b0 in range(0, len(order), a.batch):
            if a.tie and a.n_iter_hi:
                # jitter the sweep count per BATCH so an inference-time n_iter sweep is
                # in-distribution. Seeded by epoch so a resumed run repeats the same schedule.
                net.n_iter = int(np.random.default_rng(10007 * ep + b0).integers(
                    nominal_iter, a.n_iter_hi + 1))
            t = collate([train[j] for j in order[b0:b0 + a.batch]])
            pred = predict(net, t)
            resid = (pred - t['target']) / sd

            def _el(r):
                """ELEMENTWISE loss, so the reduction can be chosen by the caller."""
                return (torch.nn.functional.huber_loss(r, torch.zeros_like(r), delta=a.huber,
                                                       reduction='none')
                        if a.huber else r ** 2)

            def _obj(r):
                return _el(r).mean()

            # GRAPH BALANCE. The plain per-triangle loss averages over TRIANGLES, so a 700-triangle
            # mesh contributes ~44x the gradient of a 16-triangle one. This averages per GRAPH
            # first, giving every mesh an equal say -- the same re-weighting `--bulk_weight` applies
            # as a side effect, but WITHOUT adding a second objective and without the per-triangle/
            # bulk trade it forces (measured 0.6841 -> 0.7326 at weight 1.0).
            #
            # It is the cheaper hypothesis, and the cancellation measurement is why it is tried
            # first: `m2_error_cancellation.py` found R = mean|err|/|mean err| tracking sqrt(N) with
            # a CONSTANT prefactor 0.435 across every family, i.e. errors that behave as independent
            # over blocks of ~5 triangles and carry NO large systematic per-network bias. So the
            # bulk term is not suppressing a bias (there is none to suppress); the `cells` failure
            # is just weak sqrt(N) averaging on a small mesh, and re-weighting is the direct fix.
            if a.graph_balance and 'tri_batch' in t:
                loss = bulk_mean(_el(resid), t['tri_batch'], int(t['n_graphs'])).mean()
            else:
                loss = _obj(resid)
            # BULK TERM. The per-triangle loss is an average over TRIANGLES, so it implicitly
            # weights big meshes: a 700-triangle `random` network contributes 60x the gradient of a
            # 12-triangle `cells` one. But nu and E are contractions of the per-graph MEAN C(s), so
            # what a designer actually reads off is a quantity the loss never looks at directly --
            # and small meshes have the fewest triangles to average over. Measured consequence
            # (2026-09-11, fresh unseen meshes vs the independent sim): `cells` and `random` have
            # near-identical per-triangle error (0.211 vs 0.206) and MAE(nu) 0.1613 vs 0.0256, a
            # 6x spread that the training objective is blind to. This term puts C_eff in the loss.
            # Normalised by its OWN sigma: bulk C6 is much narrower than per-triangle C6 (errors
            # cancel in the mean), so reusing `sd` here would silently down-weight it to noise.
            if a.bulk_weight and 'tri_batch' in t:
                bp = bulk_mean(pred, t['tri_batch'], int(t['n_graphs']))
                bt = bulk_mean(t['target'], t['tri_batch'], int(t['n_graphs']))
                loss = loss + a.bulk_weight * _obj((bp - bt) / sd_bulk)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step(); tot += float(loss.detach()); nb += 1
        if a.schedule == 'cosine':
            sch.step()
        net.n_iter = nominal_iter          # evaluate at the NOMINAL count, never a jittered one
        if ep % max(1, a.eval_every) == 0 or ep == a.epochs - 1:
            per, bad, blk = evaluate(net, valid, sd)
            vnorm = float((per / sd).mean())
            vbulk = float((blk / sd).mean())
            # mesh-disjoint score: REPORTED, never acted on (see the split comment)
            vmesh = (float((evaluate(net, valid_mesh, sd)[0] / sd).mean())
                     if valid_mesh else float('nan'))
            if vnorm < best['val']:
                best = dict(val=vnorm, epoch=ep,
                            state={k: v.detach().clone() for k, v in net.state_dict().items()})
            # SIGNIFICANT improvement is tracked SEPARATELY from `best`, and the stop criterion uses
            # this one. `best` is a strict `<`, so a 0.05 % fluctuation counted as a "new best" and
            # reset the stop window -- the previous run kept going on noise for that reason. The
            # scheduler already judged those same points insignificant, so the two now agree.
            if vnorm < last_sig['val'] * (1.0 - a.min_delta):
                last_sig = dict(val=vnorm, epoch=ep)
            lr_now = opt.param_groups[0]['lr']
            hist.append(dict(epoch=ep, train=tot / max(nb, 1), val=vnorm, val_mesh=vmesh,
                             val_bulk=vbulk, spd=bad, lr=lr_now))
            print('  ep %4d  train %.4f   val MAE/std %.4f   bulk %.4f   val(new MESH) %.4f   '
                  'SPD viol %.4f   lr %.2e  (%.0fs)'
                  % (ep, tot / max(nb, 1), vnorm, vbulk, vmesh, bad, lr_now, time.time() - t0))
            # SNAPSHOT at every evaluation. Cheap next to an epoch, and it makes the run
            # restartable from here instead of from zero.
            torch.save(dict(state=net.state_dict(), opt=opt.state_dict(), sch=sch.state_dict(),
                            best=best, last_sig=last_sig, hist=hist, epoch=ep,
                            elapsed=time.time() - t0,
                            ns=a.ns, nt=a.nt, hidden=a.hidden, layers=a.layers),
                       resume_path + '.tmp')
            os.replace(resume_path + '.tmp', resume_path)   # atomic: never a half-written snapshot
            if a.schedule == 'plateau':
                sch.step(vnorm)
                # STOP when more epochs demonstrably buy nothing -- which is what "as many epochs as
                # needed" actually means. Counts EVALUATIONS since the last SIGNIFICANT improvement
                # (> min_delta relative), not since the last strict best.
                since_sig = (ep - last_sig['epoch']) // max(1, a.eval_every)
                if since_sig >= a.stop_patience:
                    print('  STOPPING: %d evaluations (%d epochs) without a >%.1f%% improvement '
                          '(last significant %.4f at ep %d; best seen %.4f at ep %d). '
                          'Convergence measured, not scheduled.'
                          % (since_sig, ep - last_sig['epoch'], 100 * a.min_delta,
                             last_sig['val'], last_sig['epoch'], best['val'], best['epoch']))
                    break
                if lr_now < a.lr * 1e-3:
                    print('  STOPPING: lr decayed to %.2e (<1e-3 of initial); further decay cannot '
                          'help.' % lr_now)
                    break

    per, bad, blk = evaluate(net, valid, sd)
    print('\nFINAL (%s)   per-triangle MAE/std = %.4f   SPD viol %.5f'
          % (split, float((per / sd).mean()), bad))
    # Reference points for the `bravais` holdout ONLY -- these are SPLIT-DEPENDENT, and quoting
    # the wrong split's number is exactly the error that made v3's 5 % margin look like 24 %:
    # the own-bulk oracle is 0.4114 on the bravais VAL split but 0.5414 on TRAIN, and the
    # previously printed 0.5144 was the train-split figure. Recompute both with
    # `Phase 5/verifications/m2_v3_report.py` for any other holdout.
    if a.holdout == 'bravais':
        print('  bravais reference: own-bulk oracle 0.4114 (val) ;  v2 0.5111 ;  '
              'v3 d3 0.3906 ;  v3 d5 0.3561')
    # keep whichever of {final, best} actually scores better, and SAY which -- silently saving
    # the best would make the reported number and the saved weights disagree
    final_val = float((per / sd).mean())
    print('  best checkpointed epoch %d at %.4f  (final %.4f) -> saving %s'
          % (best['epoch'], best['val'], final_val,
             'BEST' if best['val'] < final_val else 'FINAL'))
    if best['state'] is not None and best['val'] < final_val:
        net.load_state_dict(best['state'])
        per, bad, blk = evaluate(net, valid, sd)

    os.makedirs(RESULTS, exist_ok=True)
    # The tag must identify the RUN, not just its data flags. It previously encoded only
    # holdout+filter+huber, so a 2-epoch throughput benchmark sharing those flags silently
    # overwrote a 220-epoch checkpoint -- 11.5 h of compute lost, and the junk was committed in its
    # place because the metadata was never checked. Architecture and epoch count now go in the name,
    # and an existing file is never overwritten without --force.
    tag = run_tag(a)
    if os.path.exists(ckpt_path) and not a.force:
        raise SystemExit('refusing to overwrite %s (pass --force). A checkpoint is the only '
                         'irreplaceable output of a run -- everything else can be recomputed from '
                         'it, and it cannot be recomputed from anything.' % ckpt_path)
    torch.save(dict(state=net.state_dict(), ns=a.ns, nt=a.nt, hidden=a.hidden, layers=a.layers,
                    use_star=not a.no_star, use_global=not a.no_global, head=M3.HEAD_VERSION,
                    holdout=a.holdout, w_max_cut=a.w_max_cut, huber=a.huber,
                    epochs=a.epochs, n_train=len(train), seed=a.seed,
                    data=os.path.basename(a.data), tie=a.tie, n_iter=net.n_iter,
                    n_iter_hi=a.n_iter_hi,
                    init_from=os.path.basename(a.init_from) if a.init_from else None),
               ckpt_path)
    with open(os.path.join(RESULTS, 'run_v3_%s.json' % tag), 'w', encoding='utf-8') as fh:
        json.dump(dict(model='v3', split=split, epochs=a.epochs, ns=a.ns, nt=a.nt,
                       w_max_cut=a.w_max_cut, huber=a.huber, batch=a.batch, lr=a.lr,
                       bulk_weight=a.bulk_weight, graph_balance=bool(a.graph_balance),
                       opt=a.opt, wd=a.wd, init_from=a.init_from or None,
                       tie=bool(a.tie), n_iter=int(net.n_iter), n_iter_hi=int(a.n_iter_hi),
                       schedule=a.schedule, patience=a.patience, stop_patience=a.stop_patience,
                       min_delta=a.min_delta, eval_every=a.eval_every,
                       hidden=a.hidden, layers=a.layers, n_train=len(train), n_val=len(valid),
                       params=int(sum(p.numel() for p in net.parameters())),
                       per_triangle_mae=[float(x) for x in per],
                       label_std=[float(x) for x in sd],
                       mae_over_std=float((per / sd).mean()), spd_violation=float(bad),
                       bulk_mae_over_std=float((blk / sd).mean()),
                       n_val_mesh=len(valid_mesh), mesh_frac=a.mesh_frac,
                       mae_over_std_new_mesh=(float((evaluate(net, valid_mesh, sd)[0] / sd).mean())
                                             if valid_mesh else None),
                       final_epoch_mae_over_std=final_val, best_epoch=int(best['epoch']),
                       saved=('best' if best['val'] < final_val else 'final'),
                       history=hist, wall_seconds=round(time.time() - t0, 1)), fh, indent=2)
    return 0


if __name__ == '__main__':
    sys.exit(main())
