"""Blast radius of the 2026-08 shear-channel core fix, over every SAVED Phase 5 design.

Context. `_compute_actual_elastic_tensor` lifted the vec3 strain-concentration operator W into 4
indices without the 1/2 on a shear INPUT pair (and with an unsymmetrised identity in (1+W)), which
over-stiffened C_xyxy on any network with W != 0 -- invisible on the regular lattice, where W == 0
identically. Every saved design was produced and verified under that contraction, so its recorded
response and its recorded solver-vs-sim gap are both suspect.

What this does. For each `Phase 5/networks/*.npz`: reload (geo, k, stored meta), recompute the
directional response with the CORRECTED code on all three routes --
  solver   : the differentiable forward solve            (design path)
  sim      : the PBC relaxation via `_common.sim_per_triangle_C6`  (designer's honesty check)
  physical : `physical_homog.energy_C`, the INDEPENDENT energy-Hessian tensor (ground truth)
-- and compare against the target stored with the design. Reports, per design, how far the stored
numbers moved and whether the design still meets its target.

Writes results/shear_fix/blast_radius.csv (+ a printed summary). Pure measurement: it re-runs no
optimisation and writes no networks, so the saved designs are left exactly as they are.

Run: python "Phase 5/verifications/blast_radius_shear_fix.py"
"""
import os, sys, glob, csv
import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'Phase 3', 'verifications'))
import _common as C                      # wires the rest of sys.path + the solver stack
from inverse_design import DesignProblem, ANG
import physical_homog as PH
import sim_assembly as SA
torch.set_default_dtype(torch.float64)

EPS_NU = 0.05                            # designer.verify's nu floor (CLAUDE.md 3)
OUT = os.path.join(REPO, 'Phase 5', 'results', 'shear_fix')


def rel_gap(nu_a, E_a, nu_b, E_b):
    """designer.verify()'s relative solver-vs-oracle gap, theta-resolved."""
    return (float((np.abs(nu_a - nu_b) / (np.abs(nu_b) + EPS_NU)).max())
            + float((np.abs(E_a - E_b) / np.maximum(np.abs(E_b), 1e-12)).max()))


def voigt_to_c6(CV):
    """Inverse of the [[c0,c2,c1],[c2,c5,c4],[c1,c4,c3]] Voigt [xx,yy,xy] assembly."""
    return np.array([CV[0, 0], CV[0, 2], CV[0, 1], CV[2, 2], CV[1, 2], CV[1, 1]])


def one(path):
    """Recompute the three routes for one saved design; None if the mesh is unsimulable."""
    geo, k, _c6_stored, meta = C.load_network(path)
    nt = len(geo['simplices']); free = np.arange(2, 2 * len(geo['pts']))

    prob = DesignProblem.from_geo(geo)
    with torch.no_grad():
        out = prob.forward(torch.as_tensor(np.asarray(k, float)), physical_units=True)
    nu_slv, E_slv = C.nu_E_theta(prob.region_tensor(out['per_triangle'], None).numpy(), ANG)

    try:                                             # the sim self-screens near-singular geometry
        nu_sim, E_sim = C.nu_E_theta(C.sim_bulk_C6(geo), ANG)
        c6_phys = voigt_to_c6(PH.energy_C(geo, free, SA.assemble_K_faff))
        nu_phy, E_phy = C.nu_E_theta(c6_phys, ANG)
    except PH.UnhealthyGeometryError:
        return None

    tgt_nu = np.asarray(meta.get('target_nu', []), float)
    tgt_E = np.asarray(meta.get('target_E', []), float)
    has_tgt = tgt_nu.size == len(ANG) and tgt_E.size == len(ANG)
    err_new = (float(max(np.abs(nu_phy - tgt_nu).max(), np.abs(E_phy - tgt_E).max()))
               if has_tgt else float('nan'))
    return dict(
        name=os.path.relpath(path, os.path.join(REPO, 'Phase 5', 'networks')).replace('\\', '/'),
        seed_name=str(meta.get('seed_name', '?')), n_tri=nt,
        gap_stored=float(meta.get('solver_sim_gap', float('nan'))),
        gap_new=rel_gap(nu_slv, E_slv, nu_sim, E_sim),
        gap_physical=rel_gap(nu_slv, E_slv, nu_phy, E_phy),
        err_stored=float(meta.get('target_err_sim', float('nan'))), err_new=err_new,
        nu_min=float(nu_phy.min()), nu_max=float(nu_phy.max()),
        nu_min_sim=float(nu_sim.min()), nu_max_sim=float(nu_sim.max()))


def main():
    os.makedirs(OUT, exist_ok=True)
    # RECURSIVE: the per-experiment designs live in networks/<exp>/ (goal1, g1_2, goal2,
    # goal2_attempts, reentrant) — those are the ones backing the results docs, so a top-level
    # glob would assess only a fraction of the saved designs.
    paths = sorted(glob.glob(os.path.join(REPO, 'Phase 5', 'networks', '**', '*.npz'),
                             recursive=True))
    rows, skipped = [], []
    for i, p in enumerate(paths):
        r = one(p)
        (rows if r is not None else skipped).append(
            r if r is not None else os.path.relpath(p, REPO))
        if i % 25 == 0:
            print(f"  {i}/{len(paths)}", flush=True)
    print(f"  {len(paths)}/{len(paths)}", flush=True)

    with open(os.path.join(OUT, 'blast_radius.csv'), 'w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)

    print(f"\n{len(rows)} designs recomputed ({len(skipped)} unsimulable: {skipped})\n")
    print(f"{'design':30s} {'gap stored':>10} {'gap now':>8} {'gap vs phys':>11} "
          f"{'tgt err stored':>14} {'tgt err now':>11}")
    print('-' * 92)
    for r in rows:
        print(f"{r['name']:30s} {r['gap_stored']:10.4f} {r['gap_new']:8.4f} {r['gap_physical']:11.4f} "
              f"{r['err_stored']:14.4f} {r['err_new']:11.4f}")
    print('-' * 92)
    g0 = np.array([r['gap_stored'] for r in rows]); g1 = np.array([r['gap_new'] for r in rows])
    gp = np.array([r['gap_physical'] for r in rows])
    e0 = np.array([r['err_stored'] for r in rows]); e1 = np.array([r['err_new'] for r in rows])
    ok = np.isfinite(e0) & np.isfinite(e1)
    print(f"solver-vs-sim gap   : stored mean {np.nanmean(g0):.4f} -> now {np.nanmean(g1):.4f}; "
          f"vs INDEPENDENT physical {np.nanmean(gp):.4f}")
    print(f"designs over gap_tol=0.05 : stored {int(np.nansum(g0 > .05))}/{len(rows)} -> "
          f"now {int(np.nansum(g1 > .05))}/{len(rows)} (physical {int(np.nansum(gp > .05))}/{len(rows)})")
    print(f"target error        : stored mean {np.nanmean(e0[ok]):.4f} -> corrected "
          f"{np.nanmean(e1[ok]):.4f}   (worsened on {int((e1[ok] > e0[ok] + 1e-9).sum())}/{int(ok.sum())})")
    print(f"\nwrote {os.path.join(OUT, 'blast_radius.csv')}")


if __name__ == '__main__':
    main()
