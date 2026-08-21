r"""Phase 5 / G1.2 — can the position optimiser reach what RANDOM eta-disorder reaches?

THE QUESTION
------------
`g1_2` reports `triangular` as reaching only nu >= +0.038: all five of its negative targets saved one
byte-identical geometry (sha1 3b5ca445f57d), i.e. the search stopped and reported the same
configuration five times. But **random eta-disorder of the very same topology reaches nu = -0.109**
(eta=0.48, k = 1, frozen 6-6-6 connectivity, never re-triangulated) — the same design space this row
searches. Random displacement therefore beats the directed optimiser by ~0.15 in nu.

Until an optimiser at least matches random sampling of its own search space, its "reachable" numbers
cannot be trusted. This script tests exactly that, changing ONE thing: the START.

ARMS (identical topology, k = 1, frozen connectivity, identical SPSA budget from `run_g1_2.py`:
n_steps=120, c=0.05, a in {0.15, 0.25}, redelaunay_every=0, E_weight=0)
  - `jitter_0.10`  the g1_2 default symmetry-break jitter  -> expected to reproduce the failure
  - `jitter_0.30`  a larger jitter, same KIND of start
  - `eta_0.45`     start FROM an eta-disordered configuration (magnitude-exact random-direction
                   displacement, the eta_reference construction) -- i.e. hand the optimiser the
                   thing random sampling already found, and see whether it holds or improves on it

Success criterion, stated up front: an arm passes if it reaches **nu <= -0.109** (the eta-disorder
result) on this topology. Reported on the INDEPENDENT sim, with the solver-vs-sim gap beside it --
noting that the eta configuration itself scores gap 0.111 while solver and sim agree on bulk nu to
0.001, so `trustworthy` is reported but is NOT the success criterion here.

Seeds are explicit inputs (SEEDS below) and echoed in the output; the eta start uses the
`eta_reference` seeding (700 + s) so it is comparable with that reference.

Run:  C:\\Users\\doron\\anaconda3\\python.exe "Phase 5/verifications/g1_2_triangular_start_probe.py"
Writes: Phase 5/results/g1_2/triangular_start_probe.csv  (+ a provenance stamp in the header)
"""
# ---- §0 preamble (verbatim; this file lives in Phase 5/verifications/) ------------------------
import os, sys, csv
import numpy as np
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'Phase 3', 'verifications'))
import _common as C                 # wires the rest of sys.path + the solver stack
import torch                        # noqa: E402
torch.set_default_dtype(torch.float64)   # REQUIRED — the whole stack is float64

sys.path.insert(0, os.path.join(REPO, 'Phase 5'))
import positions, triangulation, designer                  # noqa: E402
import physical_homog as PH                                # noqa: E402
sys.path.insert(0, os.path.join(REPO, 'Phase 2'))
from persistence import _provenance                        # noqa: E402

RESDIR = os.path.join(REPO, 'Phase 5', 'results', 'g1_2')

# --- g1_2's budget, copied so the ONLY difference between arms is the start -------------------
N_STEPS, SPSA_C, A_LIST, GAP_TOL = 120, 0.05, [0.15, 0.25], 0.05
TARGETS = [-0.10, -0.30]
SEEDS = [0, 1, 2]
ETA_BEST = -0.109               # what random eta-disorder reaches on this topology (eta=0.48, s=0)


def triangular():
    """The g1_2 `triangular` topology: perfect lattice, k=1, connectivity frozen."""
    base = C.make_lattice(1.0, 1.0, half=4.0, eta=0.0)
    Lx, Ly = float(base['BL1'][0]), float(base['BL2'][1])
    return base, np.asarray(base['pts'], float), positions._tris_from_geo(base), Lx, Ly


def start_geo(kind, pts0, tris, Lx, Ly, rng):
    """Build a starting geometry. `jitter_*` = gaussian std (the g1_2 kind of start);
    `eta_*` = magnitude-EXACT random-direction displacement (the eta_reference construction)."""
    amt = float(kind.split('_')[1])
    for _ in range(8):
        if kind.startswith('eta'):
            ang = rng.uniform(0.0, 2 * np.pi, len(pts0))
            pj = pts0 + amt * np.stack([np.cos(ang), np.sin(ang)], axis=1)
        else:
            pj = pts0 + rng.normal(0.0, amt, pts0.shape)
        g = triangulation.geo_from_simplices(pj, tris, Lx, Ly)   # NEVER re-triangulate
        try:
            PH.require_healthy_mesh(g, min_area_frac=0.05)
            return g
        except PH.UnhealthyGeometryError:
            amt *= 0.6
    return None


def main():
    _base, pts0, tris, Lx, Ly = triangular()
    k = np.ones(len(_base['bond_u']))
    rows = []
    print(f'triangular start probe — target: beat eta-disorder at nu = {ETA_BEST:+.3f}')
    print(f'seeds={SEEDS} (explicit), budget n_steps={N_STEPS} c={SPSA_C} a={A_LIST}\n')
    print(f"{'start':12} {'nu*':>6} {'seed':>4} {'a':>5} {'nu_sim':>9} {'gap':>7} {'trust':>5}")
    for kind in ('jitter_0.10', 'jitter_0.30', 'eta_0.45'):
        for nu_t in TARGETS:
            for s in SEEDS:
                for a in A_LIST:
                    rng = np.random.default_rng(700 + s)
                    g0 = start_geo(kind, pts0, tris, Lx, Ly, rng)
                    if g0 is None:
                        print(f'{kind:12} {nu_t:+6.2f} {s:4d} {a:5.2f}   no healthy start')
                        continue
                    g = positions.spsa_positions(g0, k, nu_t, 1.0, n_steps=N_STEPS, a=a,
                                                 c=SPSA_C, seed=s, redelaunay_every=0,
                                                 nu_weight=1.0, E_weight=0.0)
                    try:
                        PH.require_healthy_mesh(g)
                        rep = designer.verify(g, k, nu_t, 1.0)
                    except PH.UnhealthyGeometryError:
                        print(f'{kind:12} {nu_t:+6.2f} {s:4d} {a:5.2f}   unhealthy after SPSA')
                        continue
                    nu = float(rep['nu_sim'].mean())
                    gap = float(rep['solver_sim_gap'])
                    tr = gap < GAP_TOL
                    rows.append(dict(start=kind, nu_target=nu_t, seed=s, a=a, nu_sim=nu,
                                     nu_solver=float(rep['nu_solver'].mean()), gap=gap,
                                     trustworthy=int(tr)))
                    print(f'{kind:12} {nu_t:+6.2f} {s:4d} {a:5.2f} {nu:+9.4f} {gap:7.3f} '
                          f'{"YES" if tr else "no":>5}')

    commit, dirty, saved = _provenance()
    dest = os.path.join(RESDIR, 'triangular_start_probe.csv')
    with open(dest, 'w', newline='') as fh:
        fh.write(f'# commit={commit} dirty={dirty} saved_utc={saved} seeds={SEEDS}\n')
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print('\nBEST nu reached per start (lower is better):')
    for kind in ('jitter_0.10', 'jitter_0.30', 'eta_0.45'):
        g = [r for r in rows if r['start'] == kind]
        if not g:
            continue
        b = min(g, key=lambda r: r['nu_sim'])
        verdict = 'REACHES eta limit' if b['nu_sim'] <= ETA_BEST else 'does NOT reach it'
        print(f"  {kind:12} nu={b['nu_sim']:+.4f} (gap {b['gap']:.3f}, "
              f"nu*={b['nu_target']:+.2f}, seed {b['seed']}, a={b['a']})  -> {verdict}")
    print(f'\n-> {dest}   [commit {commit[:7]} dirty={dirty}]')


if __name__ == '__main__':
    main()
