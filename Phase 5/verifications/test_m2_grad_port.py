r"""Gate the A0.1 TORCH PORT of the M2 input pipeline: faithful forward, and a real gradient.

WHAT THE PORT IS.  `train_v3.prepare()` built `Q`, the star weights and the areas in NumPy and
handed them to `torch.as_tensor`, which produces leaves with NO history.  So the surrogate was
differentiable only in its own weights: `dC/dpts` and `dC/dk` did not exist, despite
`model_v3.py:36` stating that the chain rule flows.  That matters because the surrogate's whole
reason to exist is that geometry is differentiable through it while the exact solver's positions are
"fixed inputs, set at construction, not differentiated" (`ElasticSolver`; hence SPSA, CLAUDE.md S3).

TWO INDEPENDENT THINGS ARE CHECKED, and the separation is the point:

  [1][2][3]  the FORWARD answer is unchanged -- the port must not be a silent retune.  If it is,
             every measurement taken against the existing checkpoint is invalidated.
  [4][5]     the GRADIENT is real and correct -- autograd against central finite differences on the
             MODEL ITSELF.  This says nothing about whether the model is any good; it says the
             derivative it reports is the derivative of the function it computes.  Keeping these
             apart is what lets A0.2 blame a bad cosine on the model rather than on the wiring.

ON THE ANGLE GRADIENT.  `model_v3.angle_gradient_vec` is now ONE type-preserving function (numpy in
-> numpy out, torch in -> torch out), deliberately not a torch twin: the solver's version and this
one are the only two implementations of the formula.  `test_m2_constraints.py [1]` already pins the
numpy path to the solver elementwise; [3] below pins the torch path to the numpy one.  The two
compose, so the torch path is pinned to the solver without a third copy to keep in step.

Run:
    python "Phase 5/verifications/test_m2_grad_port.py"
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'Phase 5', 'm2'))
torch.set_default_dtype(torch.float64)

import model_v3 as M3                                                     # noqa: E402
import train_v2 as T2                                                     # noqa: E402
import train_v3 as T3                                                     # noqa: E402

SMOKE = os.path.join(REPO, 'Phase 5', 'm2', 'data', 'dataset_smoke.npz')
REF = os.path.join(HERE, 'data', 'm2_prepare_reference.npz')
CKPT = os.path.join(REPO, 'Phase 5', 'm2',
                    'checkpoint_v3_res_bravais_w10_h1_L5_ns48_h64_e400_star_ms.pt')
# one mesh per family, spanning 4 -> 120 triangles; the gradient gates are O(n_probe) forwards, so
# they run on the small end and the equivalence gates cover the rest
FAMILY_IDX = dict(cells=0, anchor=420, bravais=434, disordered=542, random=560, tiling=776,
                  auxetic=791)


def _load():
    raw = [g for g in T2.load(SMOKE) if g['sim_ok']]
    net = M3.from_checkpoint(torch.load(CKPT, map_location='cpu', weights_only=False))
    return raw, net


def _scalar(net, g, pts=None, k=None, proj=None):
    """A scalar readout of the prediction, so autograd and finite differences compare one number.

    A random projection rather than a plain sum: a sum would be blind to any error that cancels
    across the six tensor components, which is exactly the kind of mistake a carrier or area
    orientation bug produces."""
    c6 = T3.predict(net, T3.prepare(g, pts=pts, k=k))
    return (c6 * proj).sum()


# ---------------------------------------------------------------------------------------------
def test_forward_unchanged(raw, net):
    """[1] The ported pipeline reproduces the PRE-PORT prediction, from a frozen reference.

    The reference was generated from the modules as they stood in git HEAD before the port
    (`scratchpad/gen_ref.py`), so this survives the commit -- comparing against HEAD would stop
    meaning anything the moment the port is committed."""
    if not os.path.exists(REF):
        print('[1] SKIP  reference missing: %s' % REF)
        return 1
    ref = np.load(REF)
    sel = ref['sel']
    worst, worst_i = 0.0, -1
    for j, i in enumerate(sel):
        with torch.no_grad():
            got = T3.predict(net, T3.prepare(raw[int(i)])).numpy()
        exp = ref['pred_%d' % j]
        rel = np.abs(got - exp).max() / max(np.abs(exp).max(), 1e-300)
        if rel > worst:
            worst, worst_i = rel, int(i)
    ok = worst <= 1e-12
    print('[1] forward unchanged vs pre-port: %d samples, worst rel %.3e (sample %d)  %s'
          % (len(sel), worst, worst_i, 'OK' if ok else 'FAIL'))
    return 0 if ok else 1


def test_areas(raw):
    """[2] Areas recomputed from the edge vectors match the STORED areas.

    `prepare` no longer reads `g['areas']` -- a constant would truncate the position gradient
    through `area_w` and `phys`.  This is the gate that catches a wrong cross-product orientation,
    which is otherwise silent: a sign error still yields positive areas after `abs`."""
    worst, worst_f = 0.0, ''
    for fam, i in FAMILY_IDX.items():
        g = raw[i]
        a = M3.triangle_areas(g['tri_verts'], g['tri_bond'].astype(np.int64),
                              g['bond_u'], g['bond_v'], g['bond_R']).numpy()
        stored = np.asarray(g['areas'], float)
        rel = float((np.abs(a - stored) / np.abs(stored)).max())
        if rel > worst:
            worst, worst_f = rel, fam
    ok = worst <= 1e-13
    print('[2] areas vs stored, %d families: worst rel %.3e (%s)  %s'
          % (len(FAMILY_IDX), worst, worst_f, 'OK' if ok else 'FAIL'))
    return 0 if ok else 1


def test_angle_gradient_types(raw):
    """[3] The torch path of `angle_gradient_vec` equals the numpy path, bit for bit where it can.

    Composed with `test_m2_constraints.py [1]` (numpy vs the SOLVER, elementwise), this pins the
    torch path to the solver's convention without adding a third implementation of the formula."""
    rng = np.random.default_rng(0)
    A = rng.normal(size=(500, 2)); B = rng.normal(size=(500, 2))
    w_np = M3.angle_gradient_vec(A, B)
    w_t = M3.angle_gradient_vec(torch.as_tensor(A), torch.as_tensor(B)).numpy()
    d1 = float(np.abs(w_np - w_t).max())

    # and through `vertex_stars`, where the type is what decides whether dC/dpts survives
    g = raw[FAMILY_IDX['bravais']]
    tb = g['tri_bond'].astype(np.int64)
    _, _, w_a, _ = M3.vertex_stars(g['tri_verts'], tb, g['bond_u'], g['bond_v'], g['bond_R'])
    _, _, w_b, _ = M3.vertex_stars(g['tri_verts'], tb, g['bond_u'], g['bond_v'],
                                   torch.as_tensor(np.asarray(g['bond_R'], float)))
    d2 = float(np.abs(np.asarray(w_a) - w_b.numpy()).max())
    ok = d1 == 0.0 and d2 == 0.0
    print('[3] angle gradient torch vs numpy: raw %.3e, via vertex_stars %.3e  %s'
          % (d1, d2, 'OK' if ok else 'FAIL'))
    return 0 if ok else 1


def test_grad_vs_fd(raw, net, n_probe=8, seed=0):
    """[4] Autograd vs CENTRAL finite differences, for dC/dpts and dC/dk, on three families.

    Directional derivatives along random unit directions, not the full Jacobian: the same estimator
    A0.2 uses against the solver, so if the step size is wrong it is wrong here first, where the
    reference is exact.  `h` is scaled to the mesh (by mean bond length for positions, by mean k for
    stiffness) because an absolute step means something different on every mesh."""
    rng = np.random.default_rng(seed)
    bad, rows = 0, []
    for fam in ('cells', 'bravais', 'random'):
        g = raw[FAMILY_IDX[fam]]
        p0 = torch.as_tensor(np.asarray(g['pts'], float))
        k0 = torch.as_tensor(np.asarray(g['k'], float))
        proj = torch.as_tensor(rng.normal(size=(len(g['tri_bond']), 6)))
        lbar = float(np.linalg.norm(np.asarray(g['bond_R'], float), axis=1).mean())

        for chan, x0, h in (('pts', p0, 1e-6 * lbar), ('k', k0, 1e-6 * float(k0.mean()))):
            x = x0.clone().requires_grad_(True)
            val = _scalar(net, g, pts=x if chan == 'pts' else None,
                          k=x if chan == 'k' else None, proj=proj)
            val.backward()
            grad = x.grad.detach().clone()

            worst = 0.0
            for _ in range(n_probe):
                d = torch.as_tensor(rng.normal(size=tuple(x0.shape)))
                d = d / torch.linalg.norm(d)
                with torch.no_grad():
                    fp = _scalar(net, g, pts=(x0 + h * d) if chan == 'pts' else None,
                                 k=(x0 + h * d) if chan == 'k' else None, proj=proj)
                    fm = _scalar(net, g, pts=(x0 - h * d) if chan == 'pts' else None,
                                 k=(x0 - h * d) if chan == 'k' else None, proj=proj)
                fd = float((fp - fm) / (2 * h))
                an = float((grad * d).sum())
                worst = max(worst, abs(fd - an) / max(abs(fd), abs(an), 1e-12))
            rows.append((fam, chan, worst))
            bad += worst > 1e-6
    for fam, chan, w in rows:
        print('[4] d/d%-4s %-10s worst rel |autograd - FD| = %.3e  %s'
              % (chan, fam, w, 'OK' if w <= 1e-6 else 'FAIL'))
    return 1 if bad else 0


def test_channels_independent(raw, net):
    """[5] The two design channels switch on and off independently (D6).

    D6 requires `k` alone, geometry alone, or both -- they are not a fused action space, because the
    two are not symmetric (`k` has cheap solver gradients, positions do not).  A channel that is not
    requested must receive NO gradient, and one that is must receive a nonzero one."""
    g = raw[FAMILY_IDX['bravais']]
    proj = torch.as_tensor(np.random.default_rng(1).normal(size=(len(g['tri_bond']), 6)))
    out, bad = [], 0
    for want_p, want_k in ((True, False), (False, True), (True, True)):
        p = torch.as_tensor(np.asarray(g['pts'], float)).requires_grad_(want_p)
        k = torch.as_tensor(np.asarray(g['k'], float)).requires_grad_(want_k)
        _scalar(net, g, pts=p if want_p else None, k=k if want_k else None, proj=proj).backward()
        gp = 0.0 if p.grad is None else float(p.grad.abs().max())
        gk = 0.0 if k.grad is None else float(k.grad.abs().max())
        ok = (gp > 0) == want_p and (gk > 0) == want_k
        bad += not ok
        out.append('pts=%-5s k=%-5s -> |dpts|=%.2e |dk|=%.2e %s'
                   % (want_p, want_k, gp, gk, 'OK' if ok else 'FAIL'))
    for o in out:
        print('[5] %s' % o)
    return 1 if bad else 0


def main():
    raw, net = _load()
    bad = 0
    bad += test_forward_unchanged(raw, net)
    bad += test_areas(raw)
    bad += test_angle_gradient_types(raw)
    bad += test_grad_vs_fd(raw, net)
    bad += test_channels_independent(raw, net)
    print('\n%s' % ('ALL GRAD-PORT GATES PASSED' if bad == 0 else 'FAILURES: %d' % bad))
    return 1 if bad else 0


if __name__ == '__main__':
    raise SystemExit(main())
