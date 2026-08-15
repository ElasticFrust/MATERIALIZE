"""
Re-check ν(η) and E(η) — disordered networks and VD rigidity contrasts — against stored results.

Purpose: confirm that after the A-7b re-layering (and the earlier shear-channel core fix) the
framework still reproduces the recorded physics: the disorder-driven auxetic band (ν: +1/3 at η=0
falling toward ≈ −0.1 by η=0.5) and the ν/E response under virtual-distortion rigidity contrast
k = 1 + tanh(a·(|R|−1)). Two things are checked, and they are DIFFERENT questions:

  (A) SOLVER vs SIM, now. The honest accuracy measure (CLAUDE.md §3): the differentiable metric
      solve vs the INDEPENDENT nodal relaxation + virial/energy homogenisation, on the SAME
      realisation. This is the physics check, and it does not depend on any stored file.

  (B) KNOWN VALUE at η=0. The regular triangular lattice must give ν = 1/3 and E = 2/√3 on BOTH
      paths (CLAUDE.md §3 sanity gate). Independent of any stored file or previous run.

  (C) NOW vs a REFERENCE COMMIT, matched realisation by realisation (`--ref-json`). This is what
      isolates a refactor: same N, same η grid, same seeds ⇒ compare elementwise. Produce the
      reference by running this script in a worktree of the older commit.

**Why NOT against `verification_tools/plots/dg_solver_sweep_20x20.npz`.** That stored sweep is
**superseded and must not be used as a reference** — measured 2026-08-15, not assumed:

  - Its E is in the solver's INTERNAL units: E(η=0) = 0.0625, where the physical value is
    2/√3 = 1.154701. The ratio is 18.4752086 = (2/√3)/0.0625 to 12 digits. It predates
    `physical_units`.
  - Its ν at η>0 came from the legacy AREA-WEIGHTED metric average (`Ceff_nuE`), tombstoned
    2026-08-10 as physically wrong precisely because it biases ν on disordered meshes — which is
    the regime this sweep covers. ν(η=0) still matches exactly (0.333333), because at η=0 all
    triangles have equal area and the weighting is a no-op.
  - It also predates the shear-channel fix (65734b0).

So its numbers are not a target to reproduce; reproducing them would mean the corrections had been
undone. It is kept only as provenance for the superseded figures.

Reuses `verify_solver_sweep.one_realisation` rather than reimplementing the sweep (DRY).

Run:  python recheck_sweep_nu_E_eta.py [--N 20] [--seeds 10] [--etas 11] [--ref-json PATH]
Outputs → verification_tools/plots/recheck_nu_E_eta/
"""
import os, sys, json, time, argparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, 'Phase 2'))
sys.path.insert(0, ROOT)

import verify_solver_sweep as svs           # one_realisation, CASES — the recorded code path
import physical_homog as PH                 # for UnhealthyGeometryError
import plotting as P                        # the single source of truth for figures

OUTDIR = os.path.join(HERE, 'plots', 'recheck_nu_E_eta')

NU_CRYSTAL, E_CRYSTAL = 1.0 / 3.0, 2.0 / np.sqrt(3.0)     # the known value at eta=0
TOL_KNOWN = 1e-6          # eta=0 is the exact crystal on both paths
TOL_REF = 1e-10           # vs a reference commit: a pure refactor must not move the physics
                          # (well above the measured ~1e-16 solver nondeterminism, audit B-1)


def run(N, etas, seeds):
    """results[case][quantity] -> (n_eta, n_seed), via the same call the stored file used.

    Realisations the SIM refuses are recorded as NaN, not fatal. At large η a perturbed lattice can
    produce a sliver/inverted triangle; `physical_homog` screens for exactly that and raises a
    catchable `UnhealthyGeometryError` instead of letting scipy/LAPACK hard-crash (CLAUDE.md §3 —
    callers just try/except). Skipping the realisation is the correct response: the geometry, not
    the solver, is the problem, and averaging is over the seeds that are physical."""
    keys = ['nu_s', 'E_s', 'nu_i', 'E_i']
    res = {c: {k: np.full((len(etas), len(seeds)), np.nan) for k in keys} for c in svs.CASES}
    skipped = []
    t0 = time.time()
    for ie, eta in enumerate(etas):
        n_bad = 0
        for js, seed in enumerate(seeds):
            try:
                r = svs.one_realisation(N, float(eta), int(seed))
            except PH.UnhealthyGeometryError as e:
                skipped.append(dict(eta=float(eta), seed=int(seed), why=str(e)[:120]))
                n_bad += 1
                continue                                  # leave NaN
            for c in svs.CASES:
                ns, Es, ni, Ei = r[c]
                res[c]['nu_s'][ie, js] = ns; res[c]['E_s'][ie, js] = Es
                res[c]['nu_i'][ie, js] = ni; res[c]['E_i'][ie, js] = Ei
        tag = f"  ({n_bad} seed(s) skipped: near-singular geometry)" if n_bad else ""
        print(f"  eta={eta:.3f}  ({len(seeds)} seeds){tag}  [{time.time()-t0:.0f}s]", flush=True)
    return res, skipped


def compare_to_reference(res, etas, ref_path):
    """Elementwise vs a reference run of THIS script (e.g. from a pre-refactor worktree)."""
    ref = np.load(ref_path, allow_pickle=True)
    if not np.allclose(ref['etas'], etas):
        raise SystemExit(f"reference eta grid {ref['etas']} != {etas} — rerun with matching --etas")
    rows, worst = [], 0.0
    for c in svs.CASES:
        for q in ('nu_s', 'E_s', 'nu_i', 'E_i'):
            old, new = np.asarray(ref[f'{c}__{q}']), res[c][q]
            n = min(old.shape[1], new.shape[1])
            dev = np.abs(new[:, :n] - old[:, :n])
            scale = np.maximum(np.abs(old[:, :n]), 0.05 if q.startswith('nu') else 1e-12)
            rel = float(np.nanmax(dev / scale))
            rows.append((c, q, float(np.nanmax(dev)), rel, rel <= TOL_REF))
            worst = max(worst, rel)
    return rows, worst


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--N', type=int, default=20)
    ap.add_argument('--seeds', type=int, default=10)
    ap.add_argument('--eta-step', type=float, default=0.02,
                    help='eta grid step over [0, 0.5]; CLAUDE.md §3 asks for <=0.02 (>=26 points)')
    ap.add_argument('--eta-max', type=float, default=0.5)
    ap.add_argument('--ref-json', metavar='NPZ',
                    help='recheck_raw.npz from a reference commit, for the matched elementwise check')
    ap.add_argument('--tag', default='', help='suffix for the output directory')
    a = ap.parse_args()

    etas = np.round(np.arange(0.0, a.eta_max + 1e-9, a.eta_step), 6)
    seeds = list(range(a.seeds))
    outdir = OUTDIR + (('_' + a.tag) if a.tag else '')
    os.makedirs(outdir, exist_ok=True)
    print(f"re-check nu(eta), E(eta): N={a.N}, {len(etas)} etas, {len(seeds)} seeds, "
          f"{len(svs.CASES)} cases (disordered + VD)")
    res, skipped = run(a.N, etas, seeds)

    # ---- (A) solver vs sim, now — the physics check, independent of any stored file ----------
    print("\n(A) SOLVER vs INDEPENDENT SIM (same realisation, averaged over seeds)")
    print(f"    {'case':<12} {'nu_sim':>8} {'nu_slv':>8} {'dnu':>8} | "
          f"{'E_sim':>8} {'E_slv':>8} {'dE/E':>8}")
    acc = {}
    for c in svs.CASES:
        ns, ni = np.nanmean(res[c]['nu_s']), np.nanmean(res[c]['nu_i'])
        Es, Ei = np.nanmean(res[c]['E_s']), np.nanmean(res[c]['E_i'])
        dnu = float(np.nanmax(np.abs(res[c]['nu_i'] - res[c]['nu_s'])))
        dEr = float(np.nanmax(np.abs(res[c]['E_i'] - res[c]['E_s']) /
                              np.maximum(np.abs(res[c]['E_s']), 1e-12)))
        acc[c] = dict(nu_sim=ns, nu_slv=ni, max_dnu=dnu, E_sim=Es, E_slv=Ei, max_dE_rel=dEr)
        print(f"    {c:<12} {ns:>8.4f} {ni:>8.4f} {dnu:>8.4f} | {Es:>8.4f} {Ei:>8.4f} {dEr:>8.4f}")

    # the headline physical result: disorder drives the lattice auxetic
    dis = res['disordered']
    nu0 = float(np.nanmean(dis['nu_s'][0])); nu1 = float(np.nanmean(dis['nu_s'][-1]))
    print(f"\n    auxetic band (sim): nu(eta={etas[0]:.2f})={nu0:+.4f} -> "
          f"nu(eta={etas[-1]:.2f})={nu1:+.4f}")

    # ---- (B) the KNOWN value at eta=0 --------------------------------------------------------
    known_ok = True
    if abs(etas[0]) < 1e-12:
        print(f"\n(B) KNOWN VALUE at eta=0 (regular lattice): nu=1/3, E=2/sqrt(3)={E_CRYSTAL:.6f}")
        for lbl, q in [('sim', '_s'), ('solver', '_i')]:
            nuk = float(np.nanmean(dis['nu' + q][0])); Ek = float(np.nanmean(dis['E' + q][0]))
            dn, dE = abs(nuk - NU_CRYSTAL), abs(Ek - E_CRYSTAL)
            ok = dn < TOL_KNOWN and dE < TOL_KNOWN
            known_ok &= ok
            print(f"    {lbl:<7} nu={nuk:.6f} (d={dn:.2e})   E={Ek:.6f} (d={dE:.2e})   "
                  f"{'OK' if ok else 'FAIL'}")
    else:
        print("\n(B) skipped — eta grid does not include 0")

    # ---- (C) now vs a reference commit, matched realisations ---------------------------------
    rows, worst_ref = [], None
    if a.ref_json:
        print(f"\n(C) NOW vs REFERENCE {os.path.basename(a.ref_json)} (matched eta+seed)")
        rows, worst_ref = compare_to_reference(res, etas, a.ref_json)
        for c, q, absd, rel, ok in rows:
            if not ok:
                print(f"    FLAG {c:<12} {q:<5} max|d|={absd:.3e}  rel={rel:.3e}")
        print(f"    worst relative deviation: {worst_ref:.3e}   (tol {TOL_REF:.0e})   "
              f"{'OK' if worst_ref <= TOL_REF else 'FAIL'}")
    else:
        print("\n(C) skipped — no --ref-json given")

    # ---- figures (project convention: plotting.py primitives only) ---------------------------
    # Convention (CLAUDE.md §3): the two METHODS share a panel — the comparison IS the point.
    for q, lab, fn, h0 in [('nu', 'Poisson ratio ν', 'recheck_nu_eta.png', True),
                           ('E', "Young's modulus E", 'recheck_E_eta.png', False)]:
        panels = {c: {'independent sim': (etas, res[c][f'{q}_s'].T),
                      'solver': (etas, res[c][f'{q}_i'].T)} for c in svs.CASES}
        fig = P.plot_overlay_grid(panels, xlabel='η', ylabel=lab, ncols=3, hline0=h0,
                                  suptitle=f'{lab} vs disorder η — solver vs INDEPENDENT sim '
                                           f'(N={a.N}, {len(seeds)} seeds, mean±σ)')
        P.save_fig(fig, os.path.join(outdir, fn))
        print('saved', os.path.join(outdir, fn))

    out = dict(N=a.N, etas=[float(e) for e in etas], seeds=seeds, cases=list(svs.CASES),
               solver_vs_sim=acc, known_value_ok=bool(known_ok),
               ref_json=a.ref_json, worst_rel_vs_ref=worst_ref,
               ref_check=[dict(case=c, quantity=q, max_abs=absd, max_rel=rel, ok=bool(ok))
                          for c, q, absd, rel, ok in rows],
               auxetic_band=dict(nu_eta_min=nu0, nu_eta_max=nu1),
               skipped_unhealthy=skipped)
    with open(os.path.join(outdir, 'recheck.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=1)
    np.savez(os.path.join(outdir, 'recheck_raw.npz'), etas=np.asarray(etas),
             cases=np.array(svs.CASES, object),
             **{f'{c}__{k}': res[c][k] for c in svs.CASES for k in res[c]})
    print(f"\nwrote {outdir}/recheck.json + recheck_raw.npz")


if __name__ == '__main__':
    main()
