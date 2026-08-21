r"""Phase 5 / G1.2 — recover the SOLVER side of every design, so untrustworthy runs can be plotted.

WHY this exists
---------------
`run_g1_2.py` records only the SIM's achieved nu (`nu_achieved_sim`) and the composite honesty
scalar `solver_sim_gap`. It does NOT record the solver's own nu(theta) — so nothing downstream can
show *how far apart* the two code paths actually are, only whether they cleared `gap < 0.05`.

That turned out to matter (2026-08-21). G1_2.md reported "reachable nu" over trustworthy rows only
and therefore read as "distortion never reaches negative nu". Recomputation showed the opposite:
**43 designs reach nu < 0 in BOTH code paths** (median |nu_solver - nu_sim| = 0.008, most auxetic
-0.436 sim / -0.488 solver, angle-averaged as this script reports them), all with k identically 1. They were filtered out because the gap is a
RELATIVE DIRECTIONAL criterion — `max_th|dnu|/(|nu_sim|+0.05) + max_th|dE|/|E_sim|`, CLAUDE.md §3 —
so at |nu| ~ 0.44 a 0.09 disagreement at ONE angle scores 0.19 and fails, while the bulk values agree.

So: a rejected design is DATA (`run_g1_2.py` says as much where it saves `UNTRUSTED_*`), and the
plots must show it. This script supplies the missing solver column.

What it does
------------
Loads every saved G1.2 design named by `results.csv:design_path`, re-runs `designer.verify` (ONE
sim relaxation + the solver readout, the same routine that produced the original gap) and stores
both directional responses, per design, in `Phase 5/results/g1_2/solver_recheck.npz`.

`results.csv` is NOT modified: it is byte-identical to the collected copy in
`validation_2026-08/phase5/run_g1_2/`, and that identity is the artifact's provenance.

Determinism / seeds: this is a pure recompute of saved geometry — `verify` draws no random numbers,
so there is no seed to take or emit. The designs' own seeds live in their `save_network` stamps.

Output: `Phase 5/results/g1_2/solver_recheck.npz`
    run_id, topo, nu_target, trustworthy, gap_csv          (per design)
    nu_sim, nu_solver, E_sim, E_solver                     (n_design, 37) on ANG
    ok                                                     False where the sim refused the geometry
    geo_sha                                                sha1 of the node positions -- runs that
                                                           SHARE one are the same configuration
                                                           reported for several targets (a stalled
                                                           search, or saturation at a real limit)
  + commit / dirty / saved_utc provenance (audit B-3), via the ONE stamp implementation.

Run:  C:\\Users\\doron\\anaconda3\\python.exe "Phase 5/verifications/g1_2_solver_recheck.py"
"""
# ---- §0 preamble (verbatim; this file lives in Phase 5/verifications/) ------------------------
import os, sys, csv, hashlib
import numpy as np
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'Phase 3', 'verifications'))
import _common as C                 # wires the rest of sys.path + the solver stack
from inverse_design import ANG      # noqa: E402  — canonical 37-angle grid
import torch                        # noqa: E402
torch.set_default_dtype(torch.float64)   # REQUIRED — the whole stack is float64

sys.path.insert(0, os.path.join(REPO, 'Phase 5'))
import designer                                        # noqa: E402  — verify() is the canonical path
import physical_homog as PH                            # noqa: E402  — UnhealthyGeometryError
sys.path.insert(0, os.path.join(REPO, 'Phase 2'))
from persistence import _provenance                    # noqa: E402  — ONE B-3 stamp implementation

RESDIR = os.path.join(REPO, 'Phase 5', 'results', 'g1_2')


def recheck():
    """Re-verify every G1.2 design; return the columns to save. Prints a per-design line."""
    with open(os.path.join(RESDIR, 'results.csv')) as fh:
        rows = list(csv.DictReader(fh))

    n, nth = len(rows), len(ANG)
    out = dict(run_id=np.zeros(n, int), topo=np.empty(n, object),
               nu_target=np.zeros(n), trustworthy=np.zeros(n, bool), gap_csv=np.zeros(n),
               nu_sim=np.full((n, nth), np.nan), nu_solver=np.full((n, nth), np.nan),
               E_sim=np.full((n, nth), np.nan), E_solver=np.full((n, nth), np.nan),
               ok=np.zeros(n, bool), geo_sha=np.empty(n, object))

    for i, r in enumerate(rows):
        out['run_id'][i] = int(float(r['run_id']))
        out['topo'][i] = r['topo']
        out['nu_target'][i] = float(r['nu_target'])
        out['trustworthy'][i] = float(r['trustworthy']) >= 0.5
        out['gap_csv'][i] = float(r['solver_sim_gap'])
        path = r['design_path']
        if not path or not os.path.exists(path):
            print(f"  [{i:3d}] {r['topo']:20} MISSING {os.path.basename(path or '-')}")
            continue
        geo, bond_k, _C6_per, _meta = C.load_network(path)
        # Identical positions across several targets = ONE configuration reported many times. For
        # `triangular` all five negative targets share one geometry: SPSA never moved off the perfect
        # lattice (a stationary point), so those rows are a failed search, not a reach measurement.
        out['geo_sha'][i] = hashlib.sha1(
            np.ascontiguousarray(np.asarray(geo['pts'], float)).tobytes()).hexdigest()[:12]
        try:
            # The sim raises this BY DESIGN on near-singular geometry (it would otherwise segfault
            # scipy/LAPACK natively). A refused design is recorded, not fatal — CLAUDE.md §3.
            rep = designer.verify(geo, bond_k)
        except PH.UnhealthyGeometryError as e:
            print(f"  [{i:3d}] {r['topo']:20} UNHEALTHY: {e}")
            continue
        for key in ('nu_sim', 'nu_solver', 'E_sim', 'E_solver'):
            out[key][i] = rep[key]
        out['ok'][i] = True
        flag = 'T' if out['trustworthy'][i] else 'u'
        print(f"  [{i:3d}] {r['topo']:20} {flag} nu*={out['nu_target'][i]:+.2f} "
              f"sim={rep['nu_sim'].mean():+.4f} solver={rep['nu_solver'].mean():+.4f} "
              f"|d|={abs(rep['nu_sim'].mean() - rep['nu_solver'].mean()):.4f}")
    return out


def main():
    print('G1.2 solver recheck — recomputing the solver side of every saved design\n')
    out = recheck()
    commit, dirty, saved = _provenance()
    dest = os.path.join(RESDIR, 'solver_recheck.npz')
    np.savez(dest, ang=ANG, commit=commit, dirty=dirty, saved_utc=saved,
             **{k: (v.astype(str) if v.dtype == object else v) for k, v in out.items()})

    ok = out['ok']
    n_ok, n_tr = int(ok.sum()), int((out['trustworthy'] & ok).sum())
    nu_s = np.nanmean(out['nu_sim'][ok], axis=1)
    nu_v = np.nanmean(out['nu_solver'][ok], axis=1)
    both_neg = int(((nu_s < 0) & (nu_v < 0)).sum())
    print(f'\n{n_ok}/{len(ok)} designs re-verified ({len(ok) - n_ok} unreadable/unhealthy); '
          f'{n_tr} of them trustworthy')
    print(f'  |nu_solver - nu_sim|: median {np.median(np.abs(nu_s - nu_v)):.4f}  '
          f'max {np.abs(nu_s - nu_v).max():.4f}')
    print(f'  auxetic (nu<0) in BOTH code paths: {both_neg}   '
          f'most auxetic: sim {nu_s.min():+.4f} / solver {nu_v.min():+.4f}')
    print(f'  -> {dest}   [commit {commit[:7]} dirty={dirty}]')


if __name__ == '__main__':
    main()
