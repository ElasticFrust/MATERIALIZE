r"""Does a rank-deficient A(s) actually mean a wrong answer? The hexagon says no.

THE CLAIM UNDER TEST
--------------------
`CLAUDE.md` §3 states that the solver's validity condition is that `A(s)` stays INVERTIBLE PER
TRIANGLE, that "low k -- soft/dead edges" is one of two routes to losing that, and that where `A(s)`
is rank-deficient "the inverse is set by that regulariser, not by physics, and `C(s)` for those
triangles can be arbitrarily wrong".

The hexagon diameter family is a direct counterexample, and it is already a GATE
(`test_hex_closed_form.py`). Every one of its six triangles is [centre, V_i, V_i+1]: TWO soft spokes
plus one hard perimeter edge. As `k_spoke -> 0` two of the three rank-1 terms in
`A(s) = sum_e (k_e/4l_e^2) q_e q_e^T` vanish, so `A(s)` becomes numerically rank-1 by construction --
yet the solver reproduces the CLOSED FORM nu(r) = (4r^2-1)/(3+4r-4r^2) to 4.4e-06 in that same limit.

This script puts the two numbers side by side: per-triangle conditioning of `A(s)` against accuracy
versus an ANALYTIC reference that is independent of both the solver and the sim.

The prediction, if the CLAUDE.md wording were right as stated: accuracy degrades as conditioning
degrades. The hexagon's O(k_spoke) residual scaling (asserted by the gate) predicts the OPPOSITE --
accuracy improves as k_spoke falls, i.e. as A(s) becomes more singular.

Note the distinction this is really probing: `k_spoke = 1e-8` is SOFT (small but finite, the correct
free-hinge limit approached smoothly), which is not the same as DEAD (k underflowing to exactly 0,
removing the term outright). The measured 2026-08-16 failure had 11% of bonds DEAD.

Determinism: closed-form geometry, no RNG -- no seed to take or emit.

Run:  C:\\Users\\doron\\anaconda3\\python.exe "Phase 5/verifications/hex_conditioning_check.py"
Writes: Phase 5/results/hex_validation/hex_conditioning.csv
"""
# ---- §0 preamble (verbatim; this file lives in Phase 5/verifications/) ------------------------
import os, sys, csv
import numpy as np
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'Phase 3', 'verifications'))
import _common as C                                        # noqa: F401,E402  (wires sys.path)
import torch                                               # noqa: E402
torch.set_default_dtype(torch.float64)

sys.path.insert(0, os.path.join(REPO, 'Phase 5'))
sys.path.insert(0, os.path.join(REPO, 'Phase 2'))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mesh_build as MB                                    # noqa: E402
import single_hexagon as S                                 # noqa: E402
from hex_nu_linear import nu_closed_form, nu_solver_single  # noqa: E402
from conditioning_probe import a3_rcond                    # noqa: E402  — ONE A(s) implementation
from persistence import _provenance                        # noqa: E402

OUT = os.path.join(REPO, 'Phase 5', 'results', 'hex_validation')
DS = [0.5, 1.0, 1.5, 2.0]                    # the gate's landmarks: nu = -0.2, 0, ~0.6, +1
K_SPOKES = [1e-2, 1e-3, 1e-4, 1e-6, 1e-8]    # 1e-3 is the designer default; 1e-8 the gate's limit


def hex_mesh(d, k_spoke):
    """The gate's own construction: 7 nodes, 6 triangles, spokes soft, perimeter k=1."""
    tri, _ = S.hexagon(d)
    mesh = MB.build_open_mesh(tri)
    u, v = np.asarray(mesh['bond_u']), np.asarray(mesh['bond_v'])
    k = np.where((u == 6) | (v == 6), k_spoke, 1.0)        # index 6 = the centre vertex
    m = dict(mesh); m['bond_k'] = k; m['tri_k'] = k[m['tri_bond']]
    return m, k


def main():
    rows = []
    print('A(s) conditioning vs accuracy against the CLOSED FORM (independent of solver AND sim)\n')
    print(f"{'d':>5} {'k_spoke':>9} {'rcond_min':>11} {'rcond_med':>11} "
          f"{'nu_solver':>10} {'nu_closed':>10} {'|dnu|':>9}")
    for d in DS:
        for ks in K_SPOKES:
            m, _k = hex_mesh(d, ks)
            rc = a3_rcond(m, m['bond_k'])
            nu_s = nu_solver_single(d, ks)
            nu_c = float(nu_closed_form(d))
            err = abs(nu_s - nu_c)
            rows.append(dict(d=d, k_spoke=ks, rcond_min=float(rc.min()),
                             rcond_median=float(np.median(rc)), nu_solver=nu_s,
                             nu_closed_form=nu_c, abs_err=err))
            print(f'{d:5.2f} {ks:9.0e} {rc.min():11.3e} {np.median(rc):11.3e} '
                  f'{nu_s:+10.5f} {nu_c:+10.5f} {err:9.2e}')

    commit, dirty, saved = _provenance()
    os.makedirs(OUT, exist_ok=True)
    dest = os.path.join(OUT, 'hex_conditioning.csv')
    with open(dest, 'w', newline='') as fh:
        fh.write(f'# commit={commit} dirty={dirty} saved_utc={saved}\n')
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    rc = np.array([r['rcond_min'] for r in rows])
    er = np.array([r['abs_err'] for r in rows])
    good = (rc > 0) & (er > 0)
    r = float(np.corrcoef(np.log10(rc[good]), np.log10(er[good]))[0, 1])
    print(f'\ncorr(log10 rcond_min, log10 |dnu|) = {r:+.3f}')
    print('  POSITIVE => accuracy IMPROVES as A(s) becomes more singular, i.e. rank-deficiency here'
          '\n  does NOT imply a wrong answer -- the opposite of what a conditioning gate assumes.')
    worst = max(rows, key=lambda x: x['abs_err'])
    best = min(rows, key=lambda x: x['abs_err'])
    print(f"\n  worst: d={worst['d']} k_spoke={worst['k_spoke']:.0e} "
          f"rcond={worst['rcond_min']:.2e} |dnu|={worst['abs_err']:.2e}")
    print(f"  best : d={best['d']} k_spoke={best['k_spoke']:.0e} "
          f"rcond={best['rcond_min']:.2e} |dnu|={best['abs_err']:.2e}")
    print(f'\n-> {dest}   [commit {commit[:7]} dirty={dirty}]')


if __name__ == '__main__':
    main()
