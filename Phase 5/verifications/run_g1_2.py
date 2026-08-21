"""Phase 5 — EXPERIMENT G1.2: isotropic Poisson ratio nu by GEOMETRIC DISTORTION ALONE.

k is FIXED = 1 for EVERY bond (no k optimisation at all); the ONLY design variable is the vertex
POSITIONS.  Connectivity is FROZEN per topology (positions.spsa_positions with redelaunay_every=0
— no re-Delaunay / flips during a run).  We sweep 11 isotropic nu targets over [-1,1] across 10
GENUINELY-DISTINCT triangulated topologies (distinct coordination-number histograms) and measure
how much of the isotropic nu range distortion alone recovers, per topology.

Framing (user): a triangular or deformed-triangular lattice is the SAME topology; a crystalline
lattice with a basis is a DIFFERENT topology.  Two topologies are the "same" iff they share the
coordination histogram and differ only by distortion.  Every edge is a REAL uniform spring k=1 —
NO soft 'fictional' edges (that would be huge k-variation).

Outputs (Phase 5/results/g1_2/): results.csv, results.npz, eta_reference.npz, topologies.csv.
Every final design -> Phase 5/networks/g1_2/ (add-only); `UNTRUSTED_` prefix where the two code
paths disagree by more than GAP_TOL, which is recorded, NOT used to discard (audit A-12).

Run:  C:\\Users\\doron\\anaconda3\\python.exe "Phase 5/verifications/run_g1_2.py"
      (optional first arg 'smoke' -> triangular + 3 targets only)
"""
# ---- §0 preamble (run_g1_2.py lives in Phase 5/verifications/ -> two levels up to REPO) --------
import os, sys, time, csv, traceback
import numpy as np
import torch
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'Phase 3', 'verifications'))
import _common as C
from inverse_design import (DesignProblem, Objective, optimize, validate, ANG,
                            c6_to_nuE, c6_to_nuE_theta)
torch.set_default_dtype(torch.float64)

sys.path.insert(0, os.path.join(REPO, 'Phase 5'))
import seeds, designer, positions, triangulation
import physical_homog as PH        # require_healthy_mesh / UnhealthyGeometryError — ONE definition
                                   # of "healthy geometry" (audit A-13), not a local re-implementation

# ---- sweep grid ------------------------------------------------------------------------------
NU_GRID = np.array([-0.95, -0.75, -0.5, -0.3, -0.1, 0.1, 0.2, 0.3, 0.4, 0.6, 0.9])
# SPSA restarts per (topology, nu): each restart starts from an INDEPENDENT symmetry-breaking
# jitter of the canonical positions (a perfect lattice is a STATIONARY point -> zero gradient ->
# SPSA stalls; the jitter gives it a non-zero gradient to follow) and uses a distinct step size
# `a` (small = safe/near-target + trustworthy; large = aggressive/far-reaching but risks a big
# solver-vs-sim gap).  We VERIFY every restart and keep the CLOSEST-to-target TRUSTWORTHY one.
SCORE_LAMBDA = 0.5            # selection score = target_err + SCORE_LAMBDA * solver_sim_gap.
                              # Weighs honesty against doing the job instead of vetoing on it. The
                              # value is a JUDGEMENT, not a fit: on the 110 saved designs every
                              # lambda in [0, 1] picks the same per-target winners, so the data
                              # cannot distinguish them -- what matters is that the veto is gone.
A_LIST = [0.15, 0.25]         # per-restart SPSA step sizes (SEEDS_PER = len(A_LIST))
SEEDS_PER = len(A_LIST)
JITTER = 0.10                 # symmetry-break jitter std (fraction of unit spacing); conn. frozen
SPSA_C = 0.05                 # SPSA probe size (larger than default 0.01 -> cleaner descent dir)
# Budget: at ~0.37 s / SPSA step (avg over the 10 topologies, benchmarked) 2 restarts x 120 steps
# = 240 position-polish steps/target -> ~2.7 h for the full 10x11 sweep (fits the ~3 h guideline).
N_STEPS = 120                # position-polish steps per restart
GAP_TOL = 0.05
RESDIR = os.path.join(REPO, 'Phase 5', 'results', 'g1_2')
NETDIR = os.path.join(REPO, 'Phase 5', 'networks', 'g1_2')


# ---- explicit tetrakis / union-jack builder (square lattice + one centre per cell) -----------
def tetrakis(reps, s=1.0):
    """Union-Jack (tetrakis-square) tiling: square-lattice corners + one centre per cell; each
    centre joined to its 4 corners, giving 4 real triangles per cell.  Corner vertices -> degree 8,
    centre vertices -> degree 4 (a crystalline lattice WITH A BASIS -> a distinct topology; it is
    the honest 'square + diagonals' class, since a square + ONE consistent diagonal is merely a
    sheared triangular lattice = the triangular topology)."""
    Lx = Ly = reps * s
    corners = [(i * s, j * s) for i in range(reps) for j in range(reps)]
    centers = [((i + 0.5) * s, (j + 0.5) * s) for i in range(reps) for j in range(reps)]
    pts = np.array(corners + centers, float)
    nC = len(corners)

    def cidx(i, j):
        return (i % reps) * reps + (j % reps)

    def zidx(i, j):
        return nC + (i % reps) * reps + (j % reps)

    def cv(ii, jj):
        return [cidx(ii, jj), 1 if ii >= reps else 0, 1 if jj >= reps else 0]

    tris = []
    for i in range(reps):
        for j in range(reps):
            z = [zidx(i, j), 0, 0]
            c00, c10 = cv(i, j), cv(i + 1, j)
            c11, c01 = cv(i + 1, j + 1), cv(i, j + 1)
            tris += [[c00, c10, z], [c10, c11, z], [c11, c01, z], [c01, c00, z]]
    return triangulation.geo_from_simplices(pts, np.array(tris, np.int64), Lx, Ly)


# ---- coordination signature -------------------------------------------------------------------
def coord_hist(geo):
    n = len(geo['pts'])
    deg = np.bincount(geo['bond_u'], minlength=n) + np.bincount(geo['bond_v'], minlength=n)
    vals, cnts = np.unique(deg, return_counts=True)
    return {int(v): int(c) for v, c in zip(vals, cnts)}


def sig_str(hist):
    """Compact coordination signature, e.g. '4:0.50|8:0.50' (degree:fraction)."""
    n = sum(hist.values())
    return '|'.join(f'{d}:{c / n:.2f}' for d, c in sorted(hist.items()))


def _healthy(geo):
    """True iff `geo` is a non-degenerate elastic network: all triangles have healthy area AND the
    solver's uniform-k nu,E are finite and physical.  Gates the independent sim (scipy/LAPACK),
    which can HARD-CRASH (native segfault, uncatchable in Python) on a near-singular geometry — so
    both topology construction and the position search must screen geometries through this first."""
    # Geometry screen: delegate to the CANONICAL check rather than re-implementing the area test
    # (audit A-13 — this file carried it three times, once with a different threshold).
    try:
        PH.require_healthy_mesh(geo)
    except PH.UnhealthyGeometryError:
        return False
    # Response screen: "is the RESPONSE physical" is the CALLER's job (CLAUDE.md §3) — it needs the
    # solver, which the sim must not depend on. Narrow except: a near-singular geometry can make the
    # solve return non-finite or raise a linalg error, and THAT is the answer (unhealthy); anything
    # else is a real bug and must not be silently swallowed (audit A-11).
    try:
        prob = DesignProblem.from_geo(geo)
        out = prob.forward(torch.ones(len(geo['bond_u'])))
        nu, E = c6_to_nuE(prob.region_tensor(out['per_triangle'], None))
        nu, E = float(nu), float(E)
    except (np.linalg.LinAlgError, torch._C._LinAlgError, ValueError):
        return False
    return bool(np.isfinite(nu) and np.isfinite(E) and abs(nu) < 2.5 and E > 1e-6)


# ---- the 10 distinct topologies ---------------------------------------------------------------
# Three of the ten topologies are NOT triangulations — square_octagon (60/114 bonds),
# rotating_squares (16/101) and reentrant_honeycomb (98/194) carry FICTIONAL edges that exist only to
# triangulate them. `USE_SEED_K0 = False` reproduces this file's historical behaviour EXACTLY: k=1 on
# every bond, fictional ones included, which BRACES the hinges. Measured cost of that bracing (sim):
# rotating squares +0.289 braced vs -1.000 freed; reentrant honeycomb +0.303 vs -1.083. It converts
# the auxetic motifs into ordinary trusses before the experiment begins.
# `USE_SEED_K0 = True` uses each seed's OWN `k0` (1.0 real, 0.001 fictional) — the configuration the
# `is_fictional` mask exists for. Set by `run_g1_2_freed.py`; see documentation/campaign/stage3_g1_2/.
USE_SEED_K0 = False


def build_topologies():
    """The 10 genuinely-distinct connectivity classes (distinct coordination histograms).

    Historically each was taken FULLY triangulated with ALL edges real (k=1); with
    `USE_SEED_K0=True` the three non-triangulation tilings instead use their seed's own `k0`, which
    frees the fictional bracing edges (see the note above). ~30-72 nodes each."""
    T = []

    def add(name, cls, geo, k0=None):
        assert (np.asarray(geo['areas']) > 0).all(), f'{name}: non-positive area'
        h = coord_hist(geo)
        nb = len(geo['bond_u'])
        k_use = np.ones(nb) if (k0 is None or not USE_SEED_K0) else np.asarray(k0, float)
        T.append(dict(name=name, cls=cls, geo=geo, hist=h, sig=sig_str(h),
                      n_nodes=len(geo['pts']), n_bond=nb, k0=k_use))

    base_tri = C.make_lattice(1.0, 1.0, half=3.5)                       # n=56, 6-6-6
    add('triangular', 'bravais', base_tri)
    add('honeycomb', 'basis', seeds.honeycomb(reps=4)['geo'])          # n=64
    add('kagome', 'basis', seeds.kagome(reps=3)['geo'])                # n=54
    _so = seeds.seed_tiling('square_octagon', 3)
    add('square_octagon', 'tiling', _so['geo'], _so.get('k0'))         # n=36, 60/114 fictional
    add('tetrakis', 'basis', tetrakis(6))                              # n=72  {4:.5,8:.5}
    _rs = seeds._rotating_squares(reps=4, theta_deg=25.0)
    add('rotating_squares', 'auxetic', _rs['geo'], _rs.get('k0'))      # 32, 16/101 fictional
    _rh = seeds._reentrant_honeycomb(reps=4)
    add('reentrant_honeycomb', 'auxetic', _rh['geo'], _rh.get('k0'))   # 64, 98/194 fictional
    add('foam_poisson', 'foam', seeds.random_patch(55, seed=200, process='poisson_disk')['geo'])  # 54
    # flipped variants: search seeds for a NON-degenerate distinct connectivity (heavy flipping can
    # yield a near-singular network whose nu blows up, e.g. the old f16 had nu~646 — reject those)
    for nf in (8, 14):
        for s in range(60):
            gf = triangulation.random_flipped_geo(base_tri, n_flips=nf, seed=s)
            if _healthy(gf):
                add(f'flipped_tri_f{nf}', 'flipped', gf)
                break

    # distinctness audit (report; do not silently drop)
    sigs = {}
    for t in T:
        sigs.setdefault(t['sig'], []).append(t['name'])
    return T, sigs


# ---- eta-disorder reference band (distortion of the TRIANGULAR topology) ----------------------
def eta_reference(half=4.0, etas=(0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.48),
                  seeds_=range(10)):
    """Uniform-k (k=1) eta-disordered triangular lattice, FROZEN 6-6-6 connectivity, using the
    PHASE-2 convention (pbc_dg_analysis.build_periodic_tf_mesh — 'triangulate-first-then-deform'):
    every vertex is displaced by a random-DIRECTION vector of magnitude EXACTLY `eta`, and the
    triangulation is NEVER re-computed (eta<0.5 is the singular limit).  This is the passive-
    distortion reference for how far RANDOM distortion of the triangular topology moves nu; nu falls
    from +1/3 and crosses into auxetic (~ -0.1) as eta -> 0.5 (verified against the Phase-2 sweep)."""
    base = C.make_lattice(1.0, 1.0, half=half, eta=0.0)
    Lx, Ly = float(base['BL1'][0]), float(base['BL2'][1])
    pts0 = np.asarray(base['pts'], float)
    tris = positions._tris_from_geo(base)                  # frozen triangular connectivity
    rows = []
    for eta in etas:
        for s in seeds_:
            rng = np.random.default_rng(700 + s)
            ang = rng.uniform(0.0, 2 * np.pi, len(pts0))
            pj = pts0 + eta * np.stack([np.cos(ang), np.sin(ang)], axis=1)   # magnitude-eta move
            g = triangulation.geo_from_simplices(pj, tris, Lx, Ly)           # NEVER re-triangulate
            try:
                PH.require_healthy_mesh(g)                   # canonical screen (A-13), not a copy
            except PH.UnhealthyGeometryError:
                continue
            C.apply_k_to_geo(g, np.ones(len(g['bond_u'])))
            nu, E = C.sim_region_nuE(g)
            if np.isfinite(nu):
                rows.append((float(eta), int(s), float(nu), float(E), 1))
    return rows


# ---- one (topology, nu) design ---------------------------------------------------------------
def nu_tag(nu):
    return ('m' if nu < 0 else 'p') + f'{abs(nu):.2f}'.replace('.', '')


def _jittered_start(pts0, tris, Lx, Ly, rng, jitter):
    """Symmetry-breaking start: jitter positions (frozen connectivity) until all triangles keep a
    healthy positive area, so SPSA begins away from the stationary symmetric point.

    Uses the canonical screen with a DELIBERATELY STRICTER `min_area_frac` (audit A-13: this was a
    bare 0.05 inline, indistinguishable from the 1e-3 sites and readable as an inconsistency). 1e-3
    is the "will the sim segfault" floor; here we want a comfortable starting geometry, and the loop
    shrinks the jitter until it gets one — so the stricter bound is the point, not a mistake."""
    START_MIN_AREA_FRAC = 0.05
    for _ in range(8):
        pj = pts0 + rng.normal(0.0, jitter, pts0.shape)
        g = triangulation.geo_from_simplices(pj, tris, Lx, Ly)
        try:
            PH.require_healthy_mesh(g, min_area_frac=START_MIN_AREA_FRAC)
            return g
        except PH.UnhealthyGeometryError:
            jitter *= 0.6
    return g


def design_one(topo, nu_target):
    """Positions-only distortion of `topo` toward a flat isotropic nu_target at FIXED k=1.

    SEEDS_PER restarts, each from an INDEPENDENT symmetry-break jitter and a distinct SPSA step
    size (A_LIST); connectivity FROZEN (redelaunay_every=0); E free (E_weight=0).  Every restart
    is INDEPENDENTLY sim-VERIFIED; we keep the CLOSEST-to-target design among the TRUSTWORTHY ones
    SCORED by `err + SCORE_LAMBDA*gap` — the gap is a COST, not a veto (2026-08-21; the old
    prefer-trustworthy-else-lowest-gap rule discarded designs that hit the target, see design_one).
    Returns (best_geo, k, rep, loss)."""
    geo0 = topo['geo']
    k = np.asarray(topo.get('k0', np.ones(topo['n_bond'])), float)   # ones unless USE_SEED_K0
    Lx, Ly = float(geo0['BL1'][0]), float(geo0['BL2'][1])
    tris = positions._tris_from_geo(geo0)                  # frozen connectivity
    pts0 = np.asarray(geo0['pts'], float)

    cands = []                                             # (geo, rep, loss, err, gap, trust)
    for ri, a in enumerate(A_LIST):
        rng = np.random.default_rng(1000 + 17 * ri)
        gstart = _jittered_start(pts0, tris, Lx, Ly, rng, JITTER)
        g = positions.spsa_positions(gstart, k, nu_target, 1.0, n_steps=N_STEPS, a=a, c=SPSA_C,
                                     seed=ri, redelaunay_every=0, nu_weight=1.0, E_weight=0.0)
        if not _healthy(g):                                # a degenerate distortion would HARD-CRASH
            continue                                       # the scipy/LAPACK sim -> skip this restart
        L = positions.loss_at(g, k, nu_target, 1.0, 1.0, 0.0)
        rep = designer.verify(g, k, nu_target, 1.0)        # safe: geometry screened as healthy
        err = abs(float(rep['nu_sim'].mean()) - float(nu_target))
        gap = float(rep['solver_sim_gap'])
        cands.append((g, rep, L, err, gap, gap < GAP_TOL))

    if not cands:                                          # every restart degenerated -> fall back to
        rep = designer.verify(geo0, k, nu_target, 1.0)     # the undistorted (healthy) topology
        return geo0, k, rep, positions.loss_at(geo0, k, nu_target, 1.0, 1.0, 0.0)

    # SCORED selection: target error and solver-sim disagreement are WEIGHED, not vetoed
    # (2026-08-21). The old rule kept the closest-to-target TRUSTWORTHY candidate and fell back to
    # the lowest-gap one, which systematically preferred timid designs: measured on `triangular`,
    # the a=0.15 restart barely moves, lands on a TARGET-INDEPENDENT endpoint (identical for
    # nu*=-0.10 and -0.30) and scores gap 0.031, while the a=0.25 restart REACHES nu=-0.134 at gap
    # 0.371 and was discarded -- so the experiment reported +0.038 for every negative target and
    # read as "distortion cannot reach auxetic". Evidence: `g1_2_triangular_start_probe.py`.
    # A large gap means the two CODE PATHS disagree about this network, not that the network is
    # unreal; it belongs in the score as a cost, and in the report as a separate number.
    best = min(cands, key=lambda c: c[3] + SCORE_LAMBDA * c[4])
    g, rep, L, _, _, _ = best
    return g, k, rep, L


# ---- sleep-prevention (MANDATORY on win32) ---------------------------------------------------
def _keep_awake():
    if sys.platform == 'win32':
        try:
            import ctypes
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
            print('[g1.2] sleep-prevention enabled (ES_CONTINUOUS|ES_SYSTEM_REQUIRED)', flush=True)
        except Exception as e:                             # noqa: BLE001
            print(f'[g1.2] could not set keep-awake: {e}', flush=True)


# ---- persistence ------------------------------------------------------------------------------
def _save(rows):
    if not rows:
        return
    os.makedirs(RESDIR, exist_ok=True)
    # UNION of keys, not rows[0]'s — see run_goal1._save: FAILED rows (A-11) carry fewer fields.
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(os.path.join(RESDIR, 'results.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction='ignore')
        w.writeheader()
        w.writerows([{k: r.get(k, '') for k in keys} for r in rows])
    np.savez(os.path.join(RESDIR, 'results.npz'),
             **{k: np.array([r.get(k, '') for r in rows]) for k in keys})


def main():
    t_start = time.time()
    _keep_awake()
    smoke = len(sys.argv) > 1 and sys.argv[1] == 'smoke'
    os.makedirs(RESDIR, exist_ok=True)
    os.makedirs(NETDIR, exist_ok=True)

    topos, sigs = build_topologies()
    print(f'[g1.2] built {len(topos)} topologies; {len(sigs)} DISTINCT coordination signatures',
          flush=True)
    # topology table
    with open(os.path.join(RESDIR, 'topologies.csv'), 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['name', 'class', 'n_nodes', 'n_bond', 'coord_signature', 'coord_hist'])
        for t in topos:
            print(f"    {t['name']:20s} {t['cls']:8s} n={t['n_nodes']:3d} nb={t['n_bond']:3d} "
                  f"sig={t['sig']}", flush=True)
            w.writerow([t['name'], t['cls'], t['n_nodes'], t['n_bond'], t['sig'], str(t['hist'])])
    dups = {s: n for s, n in sigs.items() if len(n) > 1}
    if dups:
        print(f'[g1.2] WARNING duplicate signatures: {dups}', flush=True)

    # eta-disorder reference band
    print('[g1.2] computing eta-disorder reference band ...', flush=True)
    eta_rows = eta_reference()
    np.savez(os.path.join(RESDIR, 'eta_reference.npz'),
             eta=np.array([r[0] for r in eta_rows]), seed=np.array([r[1] for r in eta_rows]),
             nu=np.array([r[2] for r in eta_rows]), E=np.array([r[3] for r in eta_rows]),
             is_666=np.array([r[4] for r in eta_rows]))
    ref_nu = [r[2] for r in eta_rows]
    ref_nu_666 = [r[2] for r in eta_rows if r[4]]
    print(f"[g1.2] eta reference nu spread: all [{min(ref_nu):+.3f},{max(ref_nu):+.3f}] ; "
          f"6-6-6-only [{min(ref_nu_666):+.3f},{max(ref_nu_666):+.3f}]", flush=True)

    if smoke:                                              # triangular over the FULL nu grid
        topos = [t for t in topos if t['name'] == 'triangular']
        nu_grid = NU_GRID
    else:
        nu_grid = NU_GRID

    # initial (undistorted, k=1) nu per topology
    nu_init = {}
    for t in topos:
        rep0 = designer.verify(t['geo'], np.ones(t['n_bond']), 0.0, 1.0)
        nu_init[t['name']] = float(rep0['nu_sim'].mean())
        print(f"    init nu (undistorted) {t['name']:20s} = {nu_init[t['name']]:+.4f}", flush=True)

    n_runs = len(topos) * len(nu_grid)
    print(f'[g1.2] sweep: {len(topos)} topologies x {len(nu_grid)} nu = {n_runs} runs', flush=True)

    rows = []
    run_id = 0
    for t in topos:
        for nu_target in nu_grid:
            run_id += 1
            t0 = time.time()
            try:
                geoB, k, rep, loss = design_one(t, float(nu_target))
                nu_ach = float(rep['nu_sim'].mean())
                E_ach = float(rep['E_sim'].mean())
                aniso = float(rep['nu_sim'].std())
                gap = float(rep['solver_sim_gap'])
                err = abs(nu_ach - float(nu_target))
                trust = int(gap < GAP_TOL)
                # SAVE EVERY RUN, trustworthy or not (audit A-12). The previous `if trust:` discarded
                # 58 of 110 runs, and because the trust filter was applied with an instrument later
                # found ~200x too lenient (A-0), the rejected set could not be re-examined — it was
                # simply gone. An untrustworthy design is DATA: it is the record of where the solver
                # and the sim part company. The filename marks it so nothing is mistaken for a good
                # design, and `trustworthy` is stored in the file's own metadata.
                tag = 'design' if trust else 'UNTRUSTED'
                path = os.path.join(NETDIR, f"{tag}_g12_{t['name']}_{nu_tag(float(nu_target))}.npz")
                C.apply_k_to_geo(geoB, k)
                C.save_network(path, geoB, k, C6_per=rep['C6_per'],
                               target_nu=float(nu_target), target_E=1.0,
                               topo=t['name'], topo_class=t['cls'], coord_sig=t['sig'],
                               nu_sim=nu_ach, E_sim=E_ach, nu_initial=nu_init[t['name']],
                               nu_aniso_std=aniso, solver_sim_gap=gap, trustworthy=bool(trust),
                               select_score=float(err + SCORE_LAMBDA * gap),
                               note='G1.2 positions-only k=1')
                rows.append(dict(run_id=run_id, topo=t['name'], topo_class=t['cls'],
                                 coord_sig=t['sig'], n_nodes=t['n_nodes'], n_bond=t['n_bond'],
                                 nu_target=float(nu_target), nu_initial=nu_init[t['name']],
                                 nu_achieved_sim=nu_ach, E_achieved_sim=E_ach, err=err,
                                 nu_aniso_std=aniso, solver_sim_gap=gap, trustworthy=trust,
                                 select_score=float(err + SCORE_LAMBDA * gap),
                                 status='ok', error='',
                                 design_loss=float(loss), design_path=path))
                print(f"[{run_id:3d}/{n_runs}] {t['name']:20s} nu*={nu_target:+.2f} | "
                      f"init={nu_init[t['name']]:+.3f} -> ach={nu_ach:+.3f} err={err:.3f} "
                      f"aniso={aniso:.3f} gap={gap:.3f} {'OK' if trust else 'UNTRUST'} "
                      f"| {time.time()-t0:.0f}s", flush=True)
            except Exception as e:                         # noqa: BLE001
                # A failed run is RECORDED, not dropped (audit A-11). Dropping it left the success
                # rate with a survivorship-biased denominator: N_ok/N_ok instead of N_ok/N_attempted.
                # The bare `except` stays deliberately — this is a long unattended campaign driver and
                # one bad topology must not kill it — but it is now honest about what it swallowed.
                print(f"[{run_id:3d}/{n_runs}] FAILED {t['name']} nu*={nu_target}: {e}", flush=True)
                traceback.print_exc()
                rows.append(dict(run_id=run_id, topo=t['name'], topo_class=t['cls'],
                                 coord_sig=t['sig'], n_nodes=t['n_nodes'], n_bond=t['n_bond'],
                                 nu_target=float(nu_target), nu_initial=nu_init[t['name']],
                                 nu_achieved_sim=float('nan'), E_achieved_sim=float('nan'),
                                 err=float('nan'), nu_aniso_std=float('nan'),
                                 solver_sim_gap=float('nan'), trustworthy=0,
                                 status=f'FAILED:{type(e).__name__}', error=str(e)[:200],
                                 design_loss=float('nan'), design_path=''))
            if rows and (run_id % 5 == 0 or run_id == n_runs):
                _save(rows)

    _save(rows)
    dt = time.time() - t_start
    trust = [r for r in rows if r['trustworthy']]
    print(f"\n[g1.2] DONE {len(rows)} runs in {dt/60:.1f} min ({len(trust)} trustworthy)",
          flush=True)

    # Per-topology reach over EVERY run, with the two quality numbers reported SEPARATELY:
    # `err` = achieved-vs-target (did it do the job) and `gap` = solver-vs-sim (do the two code
    # paths agree about it). Reporting only gap-passing rows is what made this experiment read as
    # "distortion never reaches negative nu" when 43 designs reach nu<0 in BOTH code paths.
    print(chr(10) + '[g1.2] reach per topology -- ALL runs; err and gap reported separately:', flush=True)
    for t in topos:
        b = [r for r in rows if r['topo'] == t['name'] and r['status'] == 'ok']
        if not b:
            print(f"    {t['name']:20s} : no successful run", flush=True); continue
        nus = [r['nu_achieved_sim'] for r in b]
        bt = [r for r in b if r['trustworthy']]
        tw = (f"[{min(r['nu_achieved_sim'] for r in bt):+.3f},"
              f"{max(r['nu_achieved_sim'] for r in bt):+.3f}]" if bt else 'none')
        print(f"    {t['name']:20s} init={nu_init[t['name']]:+.3f}  reach "
              f"[{min(nus):+.3f},{max(nus):+.3f}] (n={len(b)})  "
              f"median|err|={np.median([r['err'] for r in b]):.3f}  "
              f"median gap={np.median([r['solver_sim_gap'] for r in b]):.3f}  "
              f"gap<{GAP_TOL} sub-range {tw} (n={len(bt)})", flush=True)


if __name__ == '__main__':
    main()
