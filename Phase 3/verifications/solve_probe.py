"""Instrumentation for the intrinsic solve's linear algebra, shared by its consumers.

WHY THIS EXISTS. Two verification scripts need the same two things — capture the matrix the
constrained solve ends in, and judge its conditioning the way `lstsq` itself does — and they had two
copies of the arithmetic:

  * `Phase 3/verifications/b1_excursion_analysis.py` (audit B-1: is the anomaly rank truncation?);
  * `Phase 5/verifications/m2_conditioning_vs_error.py` (does cond(G) predict the surrogate's error?).

A second consumer means one shared, tested routine (`CLAUDE.md`, architecture). The cutoff convention
in particular is easy to get subtly wrong, and a wrong cutoff silently changes what "rank" and
"condition number" mean in a result.

WHAT THE CUTOFF IS. `torch.linalg.lstsq(G, r)` with `rcond=None` on CPU is LAPACK `gelsy`, which
truncates by effective rank against `eps * max(G.shape) * sigma_max`. Any statement about G's rank or
conditioning has to use that same threshold to be a statement about the solve that actually ran.

WHY EFFECTIVE, NOT RAW, CONDITIONING. `G = J3 @ PinvJt` is SINGULAR BY CONSTRUCTION on some meshes —
the constraint rows are redundant (B-1: rank 671/672, cond ~3e16 on the regular lattice) — so
`np.linalg.cond` returns ~1e16 there regardless of the physics and discriminates nothing. The
meaningful quantity is the spread over the spectrum the solve actually uses.

This module is pure NumPy/torch instrumentation: it computes nothing physical and is not on any
design path. The PROTECTED CORE is never modified — `capture_lstsq_lhs` patches `torch.linalg.lstsq`
for the duration of one call and restores it in a `finally`.
"""
import numpy as np
import torch


def lstsq_cutoff(sv, shape):
    """The `rcond=None` threshold LAPACK applies: `eps * max(shape) * sigma_max`.

    `sv` must be the singular values in DESCENDING order (as `np.linalg.svd` returns them).
    """
    return float(np.finfo(np.float64).eps * max(shape) * sv[0])


def spectrum(M):
    """-> (singular values descending, the cutoff `lstsq` would apply to them).

    Use this when you need the spectrum itself (rank counts, tail gaps); use `effective_cond` when
    you only need the conditioning number.
    """
    M = np.asarray(M, float)
    sv = np.linalg.svd(M, compute_uv=False)
    return sv, lstsq_cutoff(sv, M.shape)


def effective_cond(M):
    """-> (sigma_max / smallest sigma ABOVE the cutoff, how many fell below it).

    The second return value is the numerical rank deficiency, and it is worth reporting alongside the
    first: a large effective condition number on a full-rank matrix and one on a rank-deficient matrix
    are different statements about the solve.
    """
    sv, cut = spectrum(M)
    live = sv[sv > cut]
    return (float(sv[0] / live[-1]) if len(live) else float('inf')), int(len(sv) - len(live))


def capture_lstsq_lhs(call, which='first'):
    """Run `call()` with `torch.linalg.lstsq` patched to record a left-hand side; -> (result, A).

    `which='first'` keeps the FIRST matrix passed (the KKT solve in the intrinsic path); `'last'`
    keeps the most recent. `A` is a detached NumPy copy, or None if `lstsq` was never reached.

    The patch is always undone, including on an exception, so a failed solve cannot leave the process
    with a patched `torch.linalg.lstsq` — which would silently corrupt every later solve in the same
    session.
    """
    if which not in ('first', 'last'):
        raise ValueError("which must be 'first' or 'last', got %r" % (which,))
    grabbed = {}
    real = torch.linalg.lstsq

    def spy(A, B, *args, **kw):
        if which == 'last' or 'A' not in grabbed:
            grabbed['A'] = A.detach().cpu().numpy().copy()
        return real(A, B, *args, **kw)

    torch.linalg.lstsq = spy
    try:
        out = call()
    finally:
        torch.linalg.lstsq = real
    return out, grabbed.get('A')
