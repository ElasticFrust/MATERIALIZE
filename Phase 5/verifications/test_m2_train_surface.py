"""Gate for the M2 TRAINING surface: run identity, warm start, LR restart, and weight-tied iteration.

WHY THIS EXISTS. Five flags and a `run_tag` change were added to `train_v3.py` / `model_v3.py` over
2026-09-12..14 (`--restart_lr`, `--run_suffix`, `--init_from`, `--tie`/`--n_iter`/`--n_iter_hi`), and
every one was verified ONLY ad hoc in a terminal. `CLAUDE.md` requires that any new capability get a
structured test alongside the existing regressions, and this session supplied the reason: one standing
gate had been RED FOR 19 DAYS and another crashed whenever stdout was piped. Checks that are not run,
rot. These are those verifications, made repeatable.

What each test protects, and what it would catch:

  [1] RUN IDENTITY -- `run_tag` names both the checkpoint and the `.resume` that `--resume` matches by
      NAME. If two different runs can share a tag, one silently continues or overwrites the other.
      Measured twice in this project: a 2-epoch benchmark overwrote a 220-epoch checkpoint (11.5 h
      lost), and a resume against the wrong `--data` continued a trajectory on a DIFFERENT sample draw.
  [2] WARM-START GUARDS -- `--init_from` must refuse every architecture difference. The channel
      toggles are the dangerous ones: `use_star`/`use_global` switch whole sub-modules while staying
      SHAPE-COMPATIBLE, so `load_state_dict` accepts them silently and trains a half-loaded model.
  [3] TIED ITERATION -- the iteration count must actually reach the output, survive being changed
      after construction, and stay finite when raised. NOTE THE TRAP: the readout's last layer is
      zero-initialised BY DESIGN so an untrained model returns exactly `A(s)`, which makes the output
      independent of `n_iter` -- the first version of this check passed VACUOUSLY for that reason.
  [4] THE ANALYTIC ANSWER SURVIVES TYING -- with a zero readout the model must return `A(s)` exactly,
      at ANY `n_iter`, tied or not. This is the known-value check (`CLAUDE.md`'s verification ladder):
      `C(s) -> A(s)` wherever the non-affine correction vanishes.
  [5] BACKWARD COMPATIBILITY -- a checkpoint written before `tie` existed must still rebuild untied
      with `n_iter == layers`, inferred from its own state dict.
  [6] SCHEDULE + CONDITIONING HELPERS -- `_make_sched` honours `--schedule`, and `solve_probe`'s
      effective conditioning matches a matrix with KNOWN singular values.

Run:  C:\\Users\\doron\\anaconda3\\python.exe "Phase 5/verifications/test_m2_train_surface.py"
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
M2DIR = os.path.join(REPO, 'Phase 5', 'm2')
sys.path.insert(0, os.path.join(REPO, 'Phase 3', 'verifications'))
import _common as C                                                       # noqa: E402,F401
sys.path.insert(0, os.path.join(REPO, 'Phase 5'))
sys.path.insert(0, M2DIR)

import torch                                                              # noqa: E402
torch.set_default_dtype(torch.float64)

import model_v3 as M3                                                     # noqa: E402
import train_v3 as T3                                                     # noqa: E402
import train_v2 as T2                                                     # noqa: E402
import solve_probe as SP                                                  # noqa: E402
sys.path.insert(0, HERE)
import m2_probe_vs_trained as PVT                                         # noqa: E402

CKPT = os.path.join(M2DIR, 'checkpoint_v3_res_bravais_w10_h1_L5_ns48_h64_e400_star_ms.pt')
SMOKE = os.path.join(M2DIR, 'data', 'dataset_smoke.npz')

BASE = dict(data='data/dataset_v2_s0.npz', holdout='bravais', w_max_cut=10.0, huber=1.0, layers=5,
            ns=48, nt=10, hidden=64, epochs=400, no_star=False, no_global=False, overfit_probe=0,
            contrast_min=0.0, graph_balance=False, bulk_weight=0.0, opt='adam', wd=0.0,
            mesh_frac=0.10, limit=0, restart_lr=0.0, run_suffix='', init_from='', tie=False,
            n_iter=0, n_iter_hi=0, resume=False, lr=3e-3, schedule='plateau', patience=4,
            min_delta=2e-2)


def _tag(**kw):
    return T3.run_tag(argparse.Namespace(**dict(BASE, **kw)))


def test_run_identity():
    """Anything that changes WHAT A RUN IS must change its tag -- see the module docstring."""
    variants = {
        'baseline': _tag(),
        'other dataset': _tag(data='data/dataset_fresh_s4321.npz'),
        'unfiltered': _tag(w_max_cut=0.0),
        'deeper': _tag(layers=8),
        'more epochs': _tag(epochs=110),
        'restart 8.1e-5': _tag(restart_lr=8.1e-5),
        'restart 2.7e-4': _tag(restart_lr=2.7e-4),
        'restart + suffix': _tag(restart_lr=2.7e-4, run_suffix='rs2'),
        'warm start': _tag(init_from=CKPT),
        'tied': _tag(tie=True, n_iter=6),
        'tied, jittered': _tag(tie=True, n_iter=6, n_iter_hi=14),
        'no star': _tag(no_star=True),
        'no global': _tag(no_global=True),
    }
    seen = {}
    for name, tg in variants.items():
        assert tg not in seen, f'tag COLLISION: {name!r} and {seen[tg]!r} both -> {tg}'
        seen[tg] = name
    # the historical format must be unchanged for an ordinary untied run, or every existing
    # checkpoint is orphaned from the runs that would resume it
    assert variants['baseline'].endswith('_star_ms'), variants['baseline']
    assert '_dv2s0_' in variants['baseline'], variants['baseline']
    assert '_tie' not in variants['baseline'] and '_rlr' not in variants['baseline']
    print('  [1] run identity: %d variants, all tags distinct; untied format unchanged  OK'
          % len(variants))


def test_init_from_guards():
    """Every architecture difference must be REFUSED -- the channel ones load silently otherwise."""
    ck = torch.load(CKPT, weights_only=False)
    ref = M3.from_checkpoint(ck)
    n_ref = sum(p.numel() for p in ref.parameters())

    def build_and_load(**kw):
        """Calls THE SHIPPED GUARD (`train_v3.check_init_compat`), not a copy of it.

        The guard was extracted from `main` precisely so this test exercises the real code: a test
        that reimplements the logic it is testing cannot catch a regression in the original.
        """
        a = argparse.Namespace(**dict(BASE, **kw))
        net = M3.ForwardGNNv3(ns=a.ns, nt=a.nt, hidden=a.hidden, n_layers=a.layers,
                              use_star=not a.no_star, use_global=not a.no_global)
        T3.check_init_compat(a, ck)
        net.load_state_dict(ck['state'])
        return net

    for label, kw in (('layers', dict(layers=4)), ('ns', dict(ns=32)), ('nt', dict(nt=8)),
                      ('hidden', dict(hidden=32)), ('use_star', dict(no_star=True)),
                      ('use_global', dict(no_global=True))):
        try:
            build_and_load(**kw)
            raise AssertionError('--init_from accepted a %s mismatch -- silent wrong load' % label)
        except SystemExit:
            pass
    # HEAD VERSION: a different head means the tensor being predicted is not the same object, and
    # nothing about the shapes would reveal it.
    try:
        T3.check_init_compat(argparse.Namespace(**BASE), dict(ck, head='some_other_head'))
        raise AssertionError('--init_from accepted a head-version mismatch')
    except SystemExit:
        pass

    ok = build_and_load()
    same = all(torch.equal(a, b) for a, b in zip(ok.state_dict().values(),
                                                 ref.state_dict().values()))
    assert same, 'matching config did not reproduce the reference weights'
    print('  [2] --init_from: 7 mismatches refused (incl. both SHAPE-COMPATIBLE channel toggles); '
          'matching config loads %d params bit-identically  OK' % n_ref)


def _smoke_graph():
    return T3.prepare([g for g in T2.load(SMOKE) if g['sim_ok']][0])


def test_tied_iteration():
    """n_iter must reach the output, be changeable after construction, and stay finite."""
    t = _smoke_graph()
    torch.manual_seed(0)
    net = M3.ForwardGNNv3(ns=48, nt=10, hidden=64, n_layers=5, tie=True, n_iter=8)
    tied_params = sum(p.numel() for p in net.parameters())
    untied_net = M3.ForwardGNNv3(ns=48, nt=10, hidden=64, n_layers=5)
    untied = sum(p.numel() for p in untied_net.parameters())
    assert tied_params < untied, (tied_params, untied)

    # EVERY channel must be tied, not just `layers`. Checking only the parameter COUNT is too weak:
    # a version of this where two of three replacements had silently failed still had FEWER params
    # than untied and passed, while carrying 5 `stars` and 5 `globals` blocks of which 4 each were
    # allocated, saved and NEVER reached by the forward pass -- 65 % of the checkpoint was dead
    # weight, and the reported model size was wrong by 3x. Count the blocks.
    import re as _re
    def _blocks(m):
        ks = list(m.state_dict())
        return {p: len({int(x.group(1)) for k in ks for x in [_re.match(p + r'\.(\d+)\.', k)] if x})
                for p in ('layers', 'stars', 'globals')}
    tb, ub = _blocks(net), _blocks(untied_net)
    assert set(tb.values()) == {1}, 'tied model is not tied in every channel: %s' % tb
    assert set(ub.values()) == {5}, 'untied model lost its distinct blocks: %s' % ub

    # THE TRAP: the readout's last layer is zero-init BY DESIGN, so X = 0 and the output is exactly
    # A(s) regardless of s, T and hence of n_iter. Without perturbing it this test passes vacuously.
    with torch.no_grad():
        net.readout[-1].weight.normal_(0, 0.05)
        net.readout[-1].bias.normal_(0, 0.05)
    outs = {}
    with torch.no_grad():
        for n in (1, 2, 8, 30):
            net.n_iter = n
            outs[n] = net(t).clone()
    assert not torch.equal(outs[1], outs[8]), 'n_iter does not reach the output'
    assert not torch.equal(outs[30], outs[8]), 'raising n_iter after construction changed nothing'
    assert torch.isfinite(outs[30]).all(), 'output blew up at n_iter=30'

    try:
        M3.ForwardGNNv3(ns=8, nt=4, hidden=8, n_layers=5, tie=False, n_iter=9)
        raise AssertionError('untied n_iter != n_layers was accepted')
    except ValueError:
        pass
    print('  [3] tied iteration: ALL channels tied (1 block each) %d params vs untied (5 each) %d; '
          'n_iter reaches the output, is changeable post-construction, finite at 30; untied mismatch '
          'refused  OK' % (tied_params, untied))


def test_analytic_survives_tying():
    """KNOWN VALUE: a zero readout must return A(s) EXACTLY, at any n_iter, tied or not.

    This is the head's founding guarantee (`X = 0` => `G = G_an = A(s)`), and it is what makes the
    residual head exact wherever the non-affine correction vanishes. Tying changes how `s`, `T` are
    produced; it must not touch that.
    """
    t = _smoke_graph()
    worst = 0.0
    for tie, n_iter in ((False, 0), (True, 1), (True, 6), (True, 25)):
        torch.manual_seed(1)
        net = M3.ForwardGNNv3(ns=48, nt=10, hidden=64, n_layers=5, tie=tie, n_iter=n_iter)
        # the readout's last layer is zero-initialised by the constructor, so X = 0 and `forward`
        # must return `G_an` itself. Both are (n_tri, 3, 3) -- `forward` returns G, not a 6-vector.
        with torch.no_grad():
            worst = max(worst, float((net(t) - net.analytic_G(t)).abs().max()))
    assert worst == 0.0, 'zero readout no longer returns A(s) exactly: %.3e' % worst
    print('  [4] analytic answer survives tying: zero readout returns A(s) EXACTLY (worst %.1e) '
          'untied and at n_iter 1/6/25  OK' % worst)


def test_backward_compatible_checkpoint():
    """A pre-`tie` checkpoint must rebuild UNTIED with n_iter == layers, inferred from its own keys."""
    ck = torch.load(CKPT, weights_only=False)
    net = M3.from_checkpoint(ck)
    assert net.tie is False, 'shipped untied checkpoint rebuilt as TIED'
    assert net.n_iter == int(ck['layers']), (net.n_iter, ck['layers'])
    stripped = {k: v for k, v in ck.items() if k not in ('tie', 'n_iter')}
    net2 = M3.from_checkpoint(stripped)
    assert net2.tie is False and net2.n_iter == int(ck['layers'])
    print('  [5] backward compatibility: shipped checkpoint rebuilds untied n_iter=%d, with and '
          'without the tie keys  OK' % net.n_iter)


def test_helpers():
    """`_make_sched` honours the flag; `solve_probe` matches KNOWN singular values."""
    net = M3.ForwardGNNv3(ns=8, nt=4, hidden=8, n_layers=1)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    p = T3._make_sched(argparse.Namespace(**dict(BASE, schedule='plateau')), opt)
    c = T3._make_sched(argparse.Namespace(**dict(BASE, schedule='cosine')), opt)
    assert type(p).__name__ == 'ReduceLROnPlateau', type(p).__name__
    assert type(c).__name__ == 'CosineAnnealingLR', type(c).__name__

    # a matrix with singular values we CHOSE: cond must be the ratio of the live ones, and the
    # rank drop must count exactly those placed below the lstsq cutoff.
    rng = np.random.default_rng(0)
    q1, _ = np.linalg.qr(rng.normal(size=(6, 6)))
    q2, _ = np.linalg.qr(rng.normal(size=(6, 6)))
    sv = np.array([1.0, 1e-1, 1e-2, 1e-3, 1e-4, 0.0])          # one exactly-zero direction
    cond, drop = SP.effective_cond(q1 @ np.diag(sv) @ q2.T)
    assert drop == 1, 'expected exactly one direction below the cutoff, got %d' % drop
    assert abs(cond / 1e4 - 1) < 1e-6, 'effective cond %.6e != 1e4' % cond
    print('  [6] helpers: _make_sched honours --schedule; effective_cond = %.4e on a matrix built '
          'with cond 1e4 and one null direction (rank drop %d)  OK' % (cond, drop))


def test_draw_is_reproducible():
    """A probe's SAMPLE DRAW must be reproducible from its checkpoint alone, or not at all.

    The probe's samples are never stored as a list -- they are drawn by
    `rng(seed).choice(len(raw), n)` -- so a consumer has to regenerate them. FOUR things determine the
    result: seed, dataset, contrast filter (which shrinks `raw` BEFORE the draw, changing what every
    index means) and count. Hardcoding any of them encodes an assumption about how a different
    program was invoked, and a mismatch yields a normal-looking table for the WRONG population.
    """
    for path in PVT.PROBES.values():
        ck = torch.load(path, weights_only=False)
        draw = PVT.draw_from_checkpoint(ck, os.path.basename(path))
        assert set(draw) == set(PVT.DRAW_KEYS), draw
        assert int(draw['overfit_probe']) > 0 and float(draw['contrast_min']) > 0, draw

    # a checkpoint missing ANY of the four must be refused, not guessed at
    full = torch.load(PVT.PROBES[8], weights_only=False)
    for key in PVT.DRAW_KEYS:
        try:
            PVT.draw_from_checkpoint({k: v for k, v in full.items() if k != key}, 'stripped')
            raise AssertionError('a checkpoint missing %r was accepted' % key)
        except SystemExit:
            pass
    # and asking for a count the checkpoint was not trained on must be refused
    try:
        PVT.draw_from_checkpoint(full, 'probe8', expect_n=200)
        raise AssertionError('mismatched --n was accepted')
    except SystemExit:
        pass
    print('  [7] sample draw: %d probe checkpoints record all of %s; each missing key refused, and '
          'a mismatched --n refused  OK' % (len(PVT.PROBES), ', '.join(PVT.DRAW_KEYS)))


def main():
    print('M2 training-surface tests (run identity, warm start, tied iteration)')
    failed = 0
    for t in (test_run_identity, test_init_from_guards, test_tied_iteration,
              test_analytic_survives_tying, test_backward_compatible_checkpoint, test_helpers,
              test_draw_is_reproducible):
        try:
            t()
        except Exception as e:                                             # noqa: BLE001
            failed += 1
            print('  FAIL %s: %s' % (t.__name__, e))
    print('ALL PASSED' if failed == 0 else '%d TEST(S) FAILED' % failed)
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
