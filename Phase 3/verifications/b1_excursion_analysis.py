r"""B-1 post-mortem: what STRUCTURE do the captured [15] excursions have, and what causes them?

Implements no physics -- it reads the `b1_dumps/b1_anomaly_*.json` records written by
`test_inverse_design._b1_dump_if_anomalous` (audit B-1; VERIFICATION_CAMPAIGN.md 5) and then
probes the solver path those excursions come from (`forward_solver_torch._intrinsic_dense_W`,
Phase 2 SOLVER_GUIDE 4).

THE DISCRIMINATING QUESTION, posed in VERIFICATION_CAMPAIGN.md 5.4 and unanswered until
2026-08-23: is the deviation ONE tensor component or DIFFUSE? A single component -- especially
shear-shear -- points at the contraction (the A-0 channel). Diffuse points at the solve.

Three sections, run in order:
  [A] structure of the captured excursions   -- pure JSON, no solving
  [B] rank-truncation hypothesis             -- REFUTED (see the results doc)
  [C] conditioning across the three [15] cases -- REFUTED (case 1 is the BEST conditioned)

IMPORTANT LIMITATION, and the reason [B]/[C] could only refute and not explain: every probe here
runs in a CLEAN, ISOLATED process, and B-1 is documented as requiring suite context (0 in 30
isolated processes). So these sections characterise the HEALTHY state and infer backwards. The
dumps record the wrong OUTPUT but none of the upstream intermediates, so nothing here can say
which quantity actually moved during an excursion.

Run:  C:\Users\doron\anaconda3\python.exe "Phase 3/verifications/b1_excursion_analysis.py"
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
DUMPS = os.path.join(HERE, 'b1_dumps')

sys.path.insert(0, HERE)
import _common as C                                              # wires the rest of sys.path
import torch
from inverse_design import DesignProblem

# The three cases [15] compares solver against the independent oracle on. Case 1 is the regular
# lattice and is the ONLY one that has ever produced an excursion.
CASES = [('case1 REGULAR (excursions)', 1.0, 1.0, 0.0, 0),
         ('case2 eta=0.35  (clean)   ', 1.0, 1.0, 0.35, 1),
         ('case3 psi=0.6   (clean)   ', 1.0, 0.6, 0.0, 2)]


def build(phi, psi, eta, seed):
    """Rebuild one [15] case exactly as the test does (same geometry, same k draw)."""
    geo = C.make_lattice(phi, psi, half=6.0, eta=eta, seed=seed)
    rng = np.random.default_rng(seed)
    k = 0.5 + rng.random(len(geo['bond_u']))
    geo['bond_k'] = k
    geo['tri_k'] = k[geo['tri_bond']]
    return DesignProblem.from_geo(geo), k


def c33(c6):
    """Solver 6-vector -> Voigt [xx,yy,xy] 3x3, matching the dump's convention."""
    return np.array([[c6[0], c6[2], c6[1]], [c6[2], c6[5], c6[4]], [c6[1], c6[4], c6[3]]])


def solve(prob, k, **lstsq_kw):
    """Forward solve; `lstsq_kw` overrides the arguments of the single internal lstsq call."""
    real = torch.linalg.lstsq

    def patched(A, B, *a, **kw):
        return real(A, B, **lstsq_kw) if lstsq_kw else real(A, B, *a, **kw)

    torch.linalg.lstsq = patched
    try:
        with torch.no_grad():
            out = prob.solver.forward(torch.as_tensor(k)[prob.tri_bond],
                                      rest_lengths=prob.rl_ref, method='intrinsic',
                                      physical_units=True)
        return c33(prob.region_tensor(out['per_triangle'], None).numpy())
    finally:
        torch.linalg.lstsq = real


# ---- [A] structure of the captured excursions -------------------------------------------------
def section_a():
    """Diffuse or one component? Same direction? Reproducible? (pure JSON, no solving)"""
    files = sorted(f for f in os.listdir(DUMPS) if f.startswith('b1_anomaly'))
    ds = [json.load(open(os.path.join(DUMPS, f))) for f in files]
    print('[A] captured excursions')
    for f, d in zip(files, ds):
        print(f'    {f:48} vs={d["vs"]:.6e} thr={d["torch_threads"]} commit={d["commit"]}')

    P = np.array(ds[0]['c_phys'])
    same = all(np.abs(np.array(d['c_phys']) - P).max() == 0 for d in ds)
    print(f'    oracle c_phys bit-identical across ALL dumps: {same}')

    scale = np.abs(P).max()
    lab = [('xx', 0, 0), ('xx-yy', 0, 1), ('yy', 1, 1), ('xyxy', 2, 2)]
    print('    component deficit % (large entries), solver vs oracle:')
    for f, d in zip(files, ds):
        S = np.array(d['c_solver'])
        r = np.array([S[i, j] / P[i, j] for _, i, j in lab])
        print(f'      {f[11:26]}  ' + '  '.join(f'{n}={v:+.3f}' for (n, _, _), v
                                                in zip(lab, (1 - r) * 100))
              + f'   all softer: {bool((r < 1).all())}')

    dev = [np.array(d['c_solver']) - P for d in ds]
    print('    pairwise cos(direction), |d| ratio, and bit-identity:')
    for i in range(len(dev)):
        for j in range(i + 1, len(dev)):
            cos = float((dev[i] * dev[j]).sum()
                        / np.linalg.norm(dev[i]) / np.linalg.norm(dev[j]))
            md = np.abs(dev[i] - dev[j]).max()
            print(f'      dump{i} vs dump{j}: cos={cos:.6f}  ratio={np.linalg.norm(dev[j])/np.linalg.norm(dev[i]):.4f}'
                  f'  max|dC|={md:.3e}' + ('   <<< BIT-IDENTICAL' if md == 0 else ''))
    return {round(d['vs'], 12): np.array(d['c_solver']) for d in ds}, P, scale


# ---- [B] rank-truncation hypothesis -- REFUTED -------------------------------------------------
def section_b(targets, P, scale):
    """`lstsq(G, r)` on CPU is LAPACK gelsy, which truncates by effective rank against rcond.
    If a singular value sat AT the cutoff, ulp-level threading jitter would flip the rank and
    give discrete, bit-reproducible alternative answers -- exactly the observed signature."""
    print('\n[B] rank-truncation hypothesis at the lstsq cutoff')
    prob, k = build(1.0, 1.0, 0.0, 0)

    cap = {}
    real = torch.linalg.lstsq

    def spy(A, B, *a, **kw):
        cap['G'] = A.detach().clone()
        return real(A, B, *a, **kw)

    torch.linalg.lstsq = spy
    try:
        solve(prob, k)
    finally:
        torch.linalg.lstsq = real

    G = cap['G'].numpy()
    sv = np.linalg.svd(G, compute_uv=False)
    cut = np.finfo(np.float64).eps * max(G.shape) * sv[0]          # the rcond=None convention
    print(f'    G {G.shape}  sigma_max={sv[0]:.4e}  cutoff={cut:.4e}  sigma_min={sv[-1]:.4e}'
          f'  ({sv[-1]/cut:.3e} x cutoff)')
    print(f'    numerical rank {(sv > cut).sum()}/{G.shape[0]};  '
          f'sigma within 1e3x of cutoff: {((sv > cut*1e-3) & (sv < cut*1e3)).sum()}')
    print(f'    tail gap: sv[-2]={sv[-2]:.3e} -> sv[-1]={sv[-1]:.3e}')
    print('    rcond sweep (does the solve jump to a captured anomalous state?):')
    for rc in [None, 1e-16, 1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-5]:
        S = solve(prob, k, rcond=rc)
        hit = next((f' <<< matches vs={vs:.3e}' for vs, T in targets.items()
                    if np.abs(S - T).max() / scale < 1e-6), '')
        print(f'      rcond={str(rc):>6}  |S-c_phys|/scale = {np.abs(S-P).max()/scale:.4e}{hit}')


# ---- [C] conditioning across the three cases -- REFUTED ---------------------------------------
def section_c():
    """If a ulp-level BLAS difference were amplified to ~1%, the failing case should be the
    ill-conditioned one. Instrument every solve/inv/lstsq in the dense intrinsic path."""
    print('\n[C] conditioning of every linear solve, per case')
    _solve, _inv, _lstsq = torch.linalg.solve, torch.linalg.inv, torch.linalg.lstsq
    for nm, phi, psi, eta, seed in CASES:
        prob, k = build(phi, psi, eta, seed)
        rec = {'solve': [], 'inv': [], 'lstsq': []}

        def s(A, B, *a, **kw):
            rec['solve'].append(np.linalg.cond(A.detach().numpy())); return _solve(A, B, *a, **kw)

        def iv(A, *a, **kw):
            rec['inv'].append(float(np.max(np.linalg.cond(A.detach().numpy())))); return _inv(A, *a, **kw)

        def ls(A, B, *a, **kw):
            rec['lstsq'].append(np.linalg.cond(A.detach().numpy())); return _lstsq(A, B, *a, **kw)

        torch.linalg.solve, torch.linalg.inv, torch.linalg.lstsq = s, iv, ls
        try:
            with torch.no_grad():
                prob.solver.forward(torch.as_tensor(k)[prob.tri_bond], rest_lengths=prob.rl_ref,
                                    method='intrinsic', physical_units=True)
        finally:
            torch.linalg.solve, torch.linalg.inv, torch.linalg.lstsq = _solve, _inv, _lstsq
        print(f'    {nm} n_tri={prob.n_tri:4d}  cond(I3-Sw)={rec["solve"][0]:.3e}  '
              f'cond(A3)max={max(rec["inv"]):.3e}  cond(G)={rec["lstsq"][0]:.3e}')


if __name__ == '__main__':
    tg, P, scale = section_a()
    section_b(tg, P, scale)
    section_c()
