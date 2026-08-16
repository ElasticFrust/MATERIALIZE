"""Phase 5 — GOAL 1 sweep: rich ISOTROPIC inverse-design test.

Target a SCALAR (isotropic) Poisson ratio nu with E=1, sweeping BOTH a nu grid and a per-bond
k-CONTRAST band, over >=100 DIFFERENT topologies, optimising BOTH k and vertex positions, each
design INDEPENDENTLY sim-verified, recording the target-error at three stages
(initial -> k-only -> k+positions) so the optimisation's improvement can be proven.

    nu grid  : 11 values spanning the realizable 2D range.
    k-bands  : 5 contrast floors f = min(k)/avg(k) in {0.0, 0.1, 0.5, 0.9, 0.99}.
    topology : >=110 DISTINCT topologies (foam / bravais / tiling / auxetic / flipped), one per run.

Outputs (Phase 5/results/goal1/): results.csv, results.npz, and (via plot_goal1.py) the figures.
Each final design is saved to Phase 5/networks/goal1/design_g1_<i>.npz with full metadata.

Run:  C:\\Users\\doron\\anaconda3\\python.exe "Phase 5/verifications/run_goal1.py"
"""
# ---- §0 preamble (run_goal1.py lives in Phase 5/verifications/ -> two levels up to REPO) -------
import os, sys, time, csv, traceback
import numpy as np
import torch
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'Phase 3', 'verifications'))
import _common as C
from inverse_design import (DesignProblem, Objective, optimize, validate, ANG,
                            c6_to_nuE, c6_to_nuE_theta)
torch.set_default_dtype(torch.float64)

sys.path.insert(0, os.path.join(REPO, 'Phase 5'))          # so seeds/designer/design_iso import
import seeds, designer, design_iso, triangulation

# ---- the sweep grid --------------------------------------------------------------------------
NU_GRID = np.array([-0.9, -0.7, -0.5, -0.3, -0.15, 0.0, 0.1, 0.2, 0.3, 0.4, 0.45])
BANDS = [('soft', 0.0), ('large', 0.1), ('medium', 0.5), ('small', 0.9), ('none', 0.99)]
REPS_PER_COMBO = 2                                          # distinct topologies per (nu,band)
GAP_TOL = 0.05
# Seeds as NAMED constants, not literals buried in main() — the charter requires every artifact to be
# traceable to (code version, config, SEED), and `save_network` now stamps these into each design
# (audit B-3; the shakedown found `seed: None` because they were inline and never recorded).
SEED_TOPO = 0                                               # topology generation (build_topologies)
SEED_SHUFFLE = 12345                                        # run-order shuffle across classes
RESDIR = os.path.join(REPO, 'Phase 5', 'results', 'goal1')
NETDIR = os.path.join(REPO, 'Phase 5', 'networks', 'goal1')

# design budget (modest so the whole sweep finishes in <~3h; n_iter trimmed from 150 to keep all
# 110 runs under the ~3h guideline — Adam converges well before 120 iters on these small cells)
KW = dict(n_outer=2, spsa_steps=20, n_iter=120, n_restarts=2)


# ---- build a big zoo of DISTINCT small topologies (40-70 nodes preferred) ---------------------
def build_topologies(n_needed, seed0=0):
    """A list of distinct (name, topo_class, geo) across the five classes.  Small cells; each
    topology appears once in the sweep so every run is a different network."""
    topos = []

    def add(name, cls, geo):
        if geo is None:
            return
        if len(geo['pts']) < 24:                           # skip tiny cells (target ~40-70 nodes)
            return
        if not (np.asarray(geo['areas']) > 0).all():
            return
        topos.append((name, cls, geo))

    # --- bravais (small half so ~40-70 nodes) ---
    for name, geo in seeds.seed_bravais(phi_vals=(0.8, 1.0, 1.2, 1.4), psi_vals=(0.8, 1.0, 1.2),
                                        etas=(0.0, 0.2), seeds=(0, 1), half=3):
        add('bravais_' + name.split('bravais_')[-1], 'bravais', geo)

    # --- tilings (a few reps each) ---
    for tname, reps in (('square', 6), ('square', 7), ('honeycomb', 3), ('honeycomb', 4),
                        ('kagome', 2), ('kagome', 3), ('square_octagon', 2), ('square_octagon', 3)):
        try:
            rec = seeds.seed_tiling(tname, reps)
            add(rec['name'], 'tiling', rec['geo'])
        except Exception:
            pass

    # --- auxetic motifs (sizes + rotating-square angle variants) ---
    for reps in (2, 4, 6):
        for rec in seeds.auxetic_motifs(reps=reps):
            add(rec['name'] + f'_r{reps}', 'auxetic', rec['geo'])
    for theta in (15.0, 20.0, 30.0, 35.0):
        try:
            rec = seeds._rotating_squares(reps=4, theta_deg=theta)
            add(rec['name'] + '_r4', 'auxetic', rec['geo'])
        except Exception:
            pass

    # --- flipped (non-Delaunay) variants off small foam/bravais bases ---
    fb = [C.make_lattice(1.0, 1.0, half=3)]
    for s in range(6):
        fb.append(seeds.random_patch(55, seed=200 + s, process='poisson_disk')['geo'])
    for i in range(16):
        base = fb[i % len(fb)]
        s = seed0 + 300 + i
        try:
            gf = triangulation.random_flipped_geo(base, n_flips=8, seed=s)
            add(f'flipped_s{s}_f8', 'flipped', gf)
        except Exception:
            pass

    # --- foam (random patches; the bulk) — vary process, size, seed for distinctness ---
    procs = ('poisson_disk', 'blue_noise', 'uniform', 'graded')
    s = seed0
    while len(topos) < n_needed:
        n_nodes = int(40 + (s * 7) % 31)                   # 40..70
        proc = procs[s % len(procs)]
        try:
            rec = seeds.random_patch(n_nodes, seed=1000 + s, process=proc)
            add(rec['name'], 'foam', rec['geo'])
        except Exception:
            pass
        s += 1
        if s > seed0 + 5000:
            break
    return topos


# ---- one design run --------------------------------------------------------------------------
def run_one(run_id, nu_target, band_name, f, topo):
    """Design (k-only + k+positions) on ONE topology for one (nu,band); sim-verify all three
    stages; save the final design.  Returns a result-row dict (or None on failure)."""
    name, cls, geo = topo
    nbond = len(geo['bond_u'])
    n_nodes = len(geo['pts'])

    # (i) INITIAL — uniform k=1, no optimisation
    rep_i = designer.verify(geo, np.ones(nbond), nu_target, 1.0)
    nu_i = float(rep_i['nu_sim'].mean()); E_i = float(rep_i['E_sim'].mean())
    err_i = abs(nu_i - nu_target)

    # (ii)+(iii) design: k-only stage returned alongside full k+positions
    geoF, kF, nuF_slv, EF_slv, konly = design_iso.design_iso(
        geo, nu_target, f, return_stages=True, seed=run_id, **KW)
    geo0, k0, _, _ = konly

    # (ii) k-only sim verify
    rep_ii = designer.verify(geo0, k0, nu_target, 1.0)
    nu_ii = float(rep_ii['nu_sim'].mean()); E_ii = float(rep_ii['E_sim'].mean())
    err_ii = abs(nu_ii - nu_target)

    # (iii) k+positions sim verify (final; honesty gap gate)
    rep_iii = designer.verify(geoF, kF, nu_target, 1.0)
    nu_iii = float(rep_iii['nu_sim'].mean()); E_iii = float(rep_iii['E_sim'].mean())
    err_iii = abs(nu_iii - nu_target)
    gap = float(rep_iii['solver_sim_gap'])
    kmin_avg = float(np.min(kF) / np.mean(kF))

    # save final design
    os.makedirs(NETDIR, exist_ok=True)
    path = os.path.join(NETDIR, f'design_g1_{run_id}.npz')
    C.apply_k_to_geo(geoF, kF)
    C.save_network(path, geoF, kF, C6_per=rep_iii['C6_per'],
                   seed=dict(topo=SEED_TOPO, shuffle=SEED_SHUFFLE),   # B-3 traceability
                   target_nu=float(nu_target), target_E=1.0, band=band_name, band_f=f,
                   topo_class=cls, seed_name=name, n_nodes=n_nodes,
                   err_initial=err_i, err_konly=err_ii, err_full=err_iii,
                   nu_sim_initial=nu_i, nu_sim_konly=nu_ii, nu_sim_full=nu_iii,
                   E_sim_full=E_iii, solver_sim_gap=gap, nu_solver_full=float(nuF_slv),
                   trustworthy=bool(gap < GAP_TOL))    # was computed for the row but never saved

    return dict(run_id=run_id, topo_name=name, topo_class=cls, n_nodes=n_nodes, n_bond=nbond,
                nu_target=float(nu_target), band=band_name, band_f=f,
                nu_sim_initial=nu_i, E_sim_initial=E_i, err_initial=err_i,
                nu_sim_konly=nu_ii, E_sim_konly=E_ii, err_konly=err_ii,
                nu_sim_full=nu_iii, E_sim_full=E_iii, err_full=err_iii,
                nu_solver_full=float(nuF_slv), kmin_avg=kmin_avg,
                solver_sim_gap=gap, trustworthy=int(gap < GAP_TOL),
                design_path=path)


# ---- the sweep -------------------------------------------------------------------------------
def _keep_awake():
    """Prevent Windows from sleeping while this long sweep runs (a prior overnight run was killed
    by system sleep at ~15/110). ES_CONTINUOUS | ES_SYSTEM_REQUIRED — cleared automatically when
    the process exits. No admin needed."""
    if sys.platform == 'win32':
        try:
            import ctypes
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
            print("[goal1] sleep-prevention enabled (ES_SYSTEM_REQUIRED)", flush=True)
        except Exception as e:                             # noqa: BLE001
            print(f"[goal1] could not set keep-awake: {e}", flush=True)


def main():
    t_start = time.time()
    _keep_awake()
    os.makedirs(RESDIR, exist_ok=True)
    combos = [(nu, bn, f) for nu in NU_GRID for (bn, f) in BANDS]
    n_runs = len(combos) * REPS_PER_COMBO
    print(f"[goal1] {len(NU_GRID)} nu x {len(BANDS)} bands x {REPS_PER_COMBO} = {n_runs} runs",
          flush=True)

    topos = build_topologies(n_runs + 5, seed0=SEED_TOPO)
    rng = np.random.default_rng(SEED_SHUFFLE)
    rng.shuffle(topos)                                     # spread classes across the grid
    print(f"[goal1] built {len(topos)} distinct topologies; classes: "
          f"{ {c: sum(1 for t in topos if t[1]==c) for c in set(t[1] for t in topos)} }",
          flush=True)

    # assign a distinct topology to each run: rep-major so each (nu,band) gets varied classes
    runs = []
    ti = 0
    for rep in range(REPS_PER_COMBO):
        for (nu, bn, f) in combos:
            runs.append((nu, bn, f, topos[ti % len(topos)]))
            ti += 1

    rows = []
    for run_id, (nu, bn, f, topo) in enumerate(runs):
        t0 = time.time()
        try:
            row = run_one(run_id, nu, bn, f, topo)
            row['status'], row['error'] = 'ok', ''
            rows.append(row)
            print(f"[{run_id+1:3d}/{n_runs}] {topo[1]:8s} nu*={nu:+.2f} band={bn:6s} "
                  f"| err i={row['err_initial']:.3f} k={row['err_konly']:.3f} "
                  f"k+p={row['err_full']:.3f} | gap={row['solver_sim_gap']:.3f} "
                  f"| {time.time()-t0:.0f}s", flush=True)
        except Exception as e:                             # noqa: BLE001
            # RECORD the failure instead of dropping it (audit A-11): a dropped run left the success
            # rate with a survivorship-biased denominator. The bare `except` is retained on purpose
            # — an unattended campaign must survive one bad topology — but it no longer hides the
            # attempt. NB it would also absorb UnhealthyGeometryError, which `status` now names.
            print(f"[{run_id+1:3d}/{n_runs}] FAILED {topo[0]} nu*={nu} band={bn}: {e}", flush=True)
            traceback.print_exc()
            rows.append(dict(run_id=run_id, topo=topo[0], topo_class=topo[1],
                             nu_target=float(nu), band=bn,
                             status=f'FAILED:{type(e).__name__}', error=str(e)[:200]))
        # incremental save so a crash keeps progress
        if rows and (run_id % 5 == 0 or run_id == n_runs - 1):
            _save(rows)

    _save(rows)
    dt = time.time() - t_start
    print(f"[goal1] DONE {len(rows)} runs in {dt/60:.1f} min "
          f"({sum(r['trustworthy'] for r in rows)} trustworthy)", flush=True)
    _summary(rows)


def _save(rows):
    if not rows:
        return
    os.makedirs(RESDIR, exist_ok=True)
    # UNION of keys, not rows[0]'s: since A-11 a FAILED run is recorded too and carries fewer
    # fields, so keying off the first row would raise (DictWriter on extra keys, KeyError in the
    # npz comprehension) the moment run 1 failed. Missing entries are written blank.
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(os.path.join(RESDIR, 'results.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction='ignore')
        w.writeheader()
        w.writerows([{k: r.get(k, '') for k in keys} for r in rows])
    np.savez(os.path.join(RESDIR, 'results.npz'),
             **{k: np.array([r.get(k, '') for r in rows]) for k in keys})


def _summary(rows):
    trust = [r for r in rows if r['trustworthy']]
    if not trust:
        print("[goal1] no trustworthy runs!", flush=True)
        return
    errs = np.array([r['err_full'] for r in trust])
    print(f"\n[goal1] trustworthy runs: {len(trust)}/{len(rows)}", flush=True)
    print(f"  median nu-error (k+pos) = {np.median(errs):.4f}", flush=True)
    print(f"  success rate |err|<0.05 = {np.mean(errs < 0.05):.2%}", flush=True)
    for bn, f in BANDS:
        b = [r for r in trust if r['band'] == bn]
        if not b:
            continue
        ei = np.mean([r['err_initial'] for r in b])
        ek = np.mean([r['err_konly'] for r in b])
        ep = np.mean([r['err_full'] for r in b])
        nus = [r['nu_sim_full'] for r in b]
        print(f"  band {bn:6s} (f={f}): err init={ei:.3f} -> k={ek:.3f} -> k+pos={ep:.3f} "
              f"| nu reach [{min(nus):+.2f},{max(nus):+.2f}] (n={len(b)})", flush=True)


if __name__ == '__main__':
    main()
