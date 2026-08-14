r"""Phase 5 / M2 — SCALED, COVERAGE-DRIVEN dataset builder for the GNN forward surrogate.

Problem with plain forward-random sampling (Phase 5/dataset.py): it piles up in the
trivial nu~0.3, E~1.15 blob (see Phase 5/dataset/coverage.png -> only ~16 occupied
cells).  This builder improves coverage by COMBINING two sources:

  (A) FORWARD SCAN  — many seed-zoo topologies (all point processes + tilings +
      Bravais + complex-basis + auxetic motifs) x several k-PATTERNS
      (uniform / lognormal / graded / soft-edge k0), each labelled by the SOLVER
      forward pass with its homogenised C6 -> nu(theta),E(theta).

  (B) INGEST DESIGNED NETWORKS — load ALL Phase 5/networks/**/*.npz (including any
      goal1/ goal2/ produced by sibling runs).  These carry an independent-sim
      per-triangle tensor C6_per -> region C6 and populate the RARE, interesting
      regions (auxetic / anisotropic / off-blob) the forward scan under-samples.

Each sample stores the graph (pts, bond_u, bond_v, bond_R, k, tri_bond, areas,
BL1, BL2), the label C6 (6,), the derived nu(theta),E(theta) and descriptors
(mean nu, mean E, anisotropy).  Everything is written to ONE consolidated file
Phase 5/m2/data/dataset.npz (concatenated graphs + labels), and a coverage.png is
produced so the improved coverage is visible.  The occupied-cell count is reported
against the old 16.

The build is PARAMETERIZED (n_random topologies, k_patterns, tiling reps) with a
modest default that finishes in minutes, but is structured to scale up overnight
(see --scale / the README).

Run:  "C:\Users\doron\anaconda3\python.exe" "Phase 5\m2\build_dataset.py"
      "C:\Users\doron\anaconda3\python.exe" "Phase 5\m2\build_dataset.py" --scale
"""
import os, sys, json, glob, argparse, time
import numpy as np
import torch
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'Phase 3', 'verifications'))
import _common as C
from inverse_design import DesignProblem, ANG, c6_to_nuE_theta
torch.set_default_dtype(torch.float64)

PHASE5 = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PHASE5)                       # so `import seeds` (Phase 5/seeds.py) works
import seeds

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
NET_GLOB = os.path.join(PHASE5, 'networks', '**', '*.npz')

# coarse (nu, E) grid used for the coverage / occupied-cell metric (same as Phase 5/dataset.py)
NU_BINS = np.linspace(-0.6, 0.6, 13)             # 12 columns
E_BINS = np.linspace(0.0, 2.5, 11)               # 10 rows


# ---- solver label ----------------------------------------------------------------------------
def solver_response(geo, k):
    """SOLVER directional response for a (geo,k): (nu_theta(37,), E_theta(37,), C6(6,)) numpy."""
    prob = DesignProblem.from_geo(geo)
    out = prob.forward(torch.as_tensor(np.asarray(k, float)))
    C6 = prob.region_tensor(out['per_triangle'], None)
    nu_th, E_th = (t.detach().numpy() for t in c6_to_nuE_theta(C6, ANG))
    return nu_th, E_th, C6.detach().numpy()


def _descriptors(nu_th, E_th):
    """(mean nu, mean E, anisotropy) — anisotropy = normalised spread of E(theta)."""
    mnu, mE = float(np.mean(nu_th)), float(np.mean(E_th))
    Emax, Emin = float(np.max(E_th)), float(np.min(E_th))
    aniso = (Emax - Emin) / (Emax + Emin + 1e-12)
    return mnu, mE, aniso


# ---- k-patterns (parameterized) --------------------------------------------------------------
def k_patterns(geo, k0, seed, which=('k0', 'uniform', 'lognormal', 'graded')):
    """Yield (name, k) stiffness fields to place on ONE topology.

        k0        — the seed's own reference field (native=1 / fictional=EPS for tilings).
        uniform   — all k=1.
        lognormal — structured multiplicative disorder exp(N(0,sigma)).
        graded    — smooth spatial gradient in k across x (soft one side, stiff the other).
    """
    nbond = len(geo['bond_R'])
    rng = np.random.default_rng(seed)
    if 'k0' in which:
        yield 'k0', np.asarray(k0, float)
    if 'uniform' in which:
        yield 'uniform', np.ones(nbond)
    if 'lognormal' in which:
        for i, sig in enumerate((0.4, 0.8)):
            yield f'lognormal{i}', np.exp(rng.normal(0.0, sig, nbond))
    if 'graded' in which:
        # bond midpoint x (minimal-image midpoint = pts[u] + bond_R/2), wrapped into [0,Lx)
        mid = geo['pts'][geo['bond_u']] + 0.5 * geo['bond_R']
        Lx = float(geo['BL1'][0])
        xf = np.mod(mid[:, 0], Lx) / max(Lx, 1e-9)
        for i, amp in enumerate((1.5, 3.0)):
            yield f'graded{i}', np.exp(amp * (xf - 0.5))


# ---- (A) forward scan ------------------------------------------------------------------------
def forward_scan(n_random, n_nodes, reps_mult, which_k, seed=0):
    """Seed-zoo topologies x k-patterns, solver-labelled.  Yields sample dicts."""
    pool = list(seeds.seed_pool(n_random=n_random, n_nodes=n_nodes,
                                include=('bravais', 'random', 'tiling', 'basis', 'auxetic')))
    print(f"      seed pool: {len(pool)} topologies")
    for j, rec in enumerate(pool):
        geo, k0 = rec['geo'], rec['k0']
        for kname, k in k_patterns(geo, k0, seed + 1000 * j, which=which_k):
            try:
                nu_th, E_th, C6 = solver_response(geo, k)
            except Exception as e:                                # skip degenerate/unstable
                print(f"        [skip] {rec['name']}/{kname}: {e}")
                continue
            if not (np.all(np.isfinite(nu_th)) and np.all(np.isfinite(E_th)) and np.all(np.isfinite(C6))):
                continue
            mnu, mE, aniso = _descriptors(nu_th, E_th)
            yield dict(name=f"{rec['name']}::{kname}", source='forward', geo=geo,
                       k=np.asarray(k, float), C6=C6, nu_theta=nu_th, E_theta=E_th,
                       mean_nu=mnu, mean_E=mE, aniso=aniso)


# ---- (B) ingest designed networks ------------------------------------------------------------
def ingest_designs():
    """Load ALL Phase 5/networks/**/*.npz as labelled samples.  Where a sim per-triangle tensor
    C6_per is present, the label is the INDEPENDENT-SIM region C6 (populates interesting regions);
    otherwise fall back to the solver forward pass on the stored k."""
    files = sorted(glob.glob(NET_GLOB, recursive=True))
    print(f"      found {len(files)} saved networks under Phase 5/networks/")
    for f in files:
        try:
            geo, kb, C6_per, meta = C.load_network(f)
            if C6_per is not None and np.size(C6_per) > 0:
                C6 = C.region_phys_C6(geo, np.asarray(C6_per), None)
                nu_th, E_th = C.nu_E_theta(C6, ANG)               # sim directional response
                src = 'design_sim'
            else:
                nu_th, E_th, C6 = solver_response(geo, kb)        # seeds w/o C6_per
                src = 'design_solver'
            if not (np.all(np.isfinite(nu_th)) and np.all(np.isfinite(E_th)) and np.all(np.isfinite(C6))):
                continue
            mnu, mE, aniso = _descriptors(nu_th, E_th)
            yield dict(name=f"ingest::{os.path.splitext(os.path.basename(f))[0]}", source=src,
                       geo=geo, k=np.asarray(kb, float), C6=np.asarray(C6, float),
                       nu_theta=np.asarray(nu_th), E_theta=np.asarray(E_th),
                       mean_nu=mnu, mean_E=mE, aniso=aniso)
        except Exception as e:                                    # noqa: BLE001
            print(f"        [skip ingest] {os.path.basename(f)}: {e}")
            continue


# ---- occupied-cell coverage metric -----------------------------------------------------------
def occupied_cells(mean_nu, mean_E):
    cells = set()
    for a, b in zip(mean_nu, mean_E):
        cells.add((int(np.digitize(a, NU_BINS)), int(np.digitize(b, E_BINS))))
    return cells


# ---- persist consolidated dataset ------------------------------------------------------------
def save_consolidated(samples, path):
    """Concatenate all graphs into one file (batch-of-graphs ptr scheme) + labels + descriptors."""
    node_ptr = [0]; edge_ptr = [0]; tri_ptr = [0]
    pts, bu, bv, bR, kk, tri_bond, areas = [], [], [], [], [], [], []
    BL1, BL2 = [], []
    C6, nu_all, E_all = [], [], []
    mnu, mE, aniso, names, sources = [], [], [], [], []
    for s in samples:
        g = s['geo']
        pts.append(np.asarray(g['pts'], float))
        bu.append(np.asarray(g['bond_u'], np.int64))
        bv.append(np.asarray(g['bond_v'], np.int64))
        bR.append(np.asarray(g['bond_R'], float))
        kk.append(np.asarray(s['k'], float))
        tri_bond.append(np.asarray(g['tri_bond'], np.int64))
        areas.append(np.asarray(g['areas'], float))
        BL1.append(np.asarray(g['BL1'], float)); BL2.append(np.asarray(g['BL2'], float))
        node_ptr.append(node_ptr[-1] + len(g['pts']))
        edge_ptr.append(edge_ptr[-1] + len(g['bond_u']))
        tri_ptr.append(tri_ptr[-1] + len(g['tri_bond']))
        C6.append(np.asarray(s['C6'], float))
        nu_all.append(np.asarray(s['nu_theta'], float)); E_all.append(np.asarray(s['E_theta'], float))
        mnu.append(s['mean_nu']); mE.append(s['mean_E']); aniso.append(s['aniso'])
        names.append(s['name']); sources.append(s['source'])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(
        path,
        node_ptr=np.array(node_ptr, np.int64), edge_ptr=np.array(edge_ptr, np.int64),
        tri_ptr=np.array(tri_ptr, np.int64),
        pts=np.concatenate(pts), bond_u=np.concatenate(bu), bond_v=np.concatenate(bv),
        bond_R=np.concatenate(bR), k=np.concatenate(kk),
        tri_bond=np.concatenate(tri_bond), areas=np.concatenate(areas),
        BL1=np.stack(BL1), BL2=np.stack(BL2),
        C6=np.stack(C6), nu_theta=np.stack(nu_all), E_theta=np.stack(E_all),
        mean_nu=np.array(mnu), mean_E=np.array(mE), aniso=np.array(aniso),
        thetas=ANG, names=np.array(names), source=np.array(sources),
        meta=json.dumps(dict(n=len(names), n_theta=len(ANG),
                             nu_bins=NU_BINS.tolist(), e_bins=E_BINS.tolist())))
    return len(names)


# ---- coverage figure -------------------------------------------------------------------------
def coverage_plot(samples, occ, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    mnu = np.array([s['mean_nu'] for s in samples])
    mE = np.array([s['mean_E'] for s in samples])
    src = np.array([s['source'] for s in samples])
    cmap = {'forward': '#4C78A8', 'design_sim': '#E45756', 'design_solver': '#F58518'}
    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    for grp, col in cmap.items():
        m = src == grp
        if m.any():
            ax.scatter(mnu[m], mE[m], s=20, alpha=0.65, c=col, edgecolors='none',
                       label=f'{grp} (n={int(m.sum())})')
    # draw the coarse coverage grid
    for x in NU_BINS:
        ax.axvline(x, color='0.85', lw=0.6, zorder=0)
    for y in E_BINS:
        ax.axhline(y, color='0.85', lw=0.6, zorder=0)
    ax.set_xlim(NU_BINS[0], NU_BINS[-1]); ax.set_ylim(E_BINS[0], E_BINS[-1])
    ax.set_aspect(1.0 / ax.get_data_ratio())                      # square plot region
    ax.set_xlabel('mean ν'); ax.set_ylabel('mean E')
    ax.set_title(f'M2 dataset coverage — {len(samples)} samples, '
                 f'{len(occ)} occupied cells (old: 16)')
    ax.legend(loc='upper left', fontsize=8, framealpha=0.9)
    fig.tight_layout(); fig.savefig(path, dpi=170); plt.close(fig)


# ---- main ------------------------------------------------------------------------------------
def build(n_random=24, n_nodes=90, reps_mult=1, do_ingest=True,
          which_k=('k0', 'uniform', 'lognormal', 'graded'), seed=0):
    t0 = time.time()
    os.makedirs(DATA_DIR, exist_ok=True)
    print("=" * 88)
    print("BUILDING M2 DATASET (scaled, coverage-driven)")
    print("=" * 88)

    print(f"\n[A] forward scan  (n_random={n_random}, n_nodes={n_nodes}, k_patterns={which_k})")
    samples = list(forward_scan(n_random, n_nodes, reps_mult, which_k, seed=seed))
    print(f"      -> {len(samples)} forward samples")

    if do_ingest:
        print("\n[B] ingest designed networks (interesting-region coverage)")
        dsamples = list(ingest_designs())
        print(f"      -> {len(dsamples)} ingested samples")
        samples += dsamples

    print("\n[C] saving consolidated dataset + coverage")
    path = os.path.join(DATA_DIR, 'dataset.npz')
    n = save_consolidated(samples, path)
    mnu = [s['mean_nu'] for s in samples]; mE = [s['mean_E'] for s in samples]
    occ = occupied_cells(mnu, mE)
    cov_png = os.path.join(DATA_DIR, 'coverage.png')
    coverage_plot(samples, occ, cov_png)

    print("-" * 88)
    print(f"  dataset      : {n} samples -> {path}")
    print(f"  coverage     : {len(occ)} occupied cells on the coarse (nu,E) grid  (OLD: 16)")
    print(f"  coverage png : {cov_png}")
    print(f"  nu range     : [{min(mnu):+.3f}, {max(mnu):+.3f}]   "
          f"E range: [{min(mE):.3f}, {max(mE):.3f}]")
    print(f"  elapsed      : {time.time() - t0:.1f}s")
    print("DATASET BUILD DONE")
    return path, n, len(occ)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--scale', action='store_true',
                    help='overnight-scale build (many more topologies)')
    ap.add_argument('--n_random', type=int, default=None)
    ap.add_argument('--n_nodes', type=int, default=None)
    ap.add_argument('--no_ingest', action='store_true')
    args = ap.parse_args()
    if args.scale:
        build(n_random=args.n_random or 200, n_nodes=args.n_nodes or 140,
              do_ingest=not args.no_ingest)
    else:
        build(n_random=args.n_random or 24, n_nodes=args.n_nodes or 90,
              do_ingest=not args.no_ingest)
