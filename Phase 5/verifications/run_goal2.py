r"""Phase 5 GOAL 2 — design DIRECTIONAL cases with PHYSICALLY-REALIZABLE targets.

For a set of physical directional targets nu(theta), E(theta) (all DERIVED from valid elastic
tensors by `phys_targets`, so none is a plain impossible cos(2 theta)), this script:

  1. builds ONE shared, small, anisotropy-rich topology pool (reused across cases);
  2. per target: designs per-bond k on every pool topology (differentiable solver), verifies each
     candidate with the INDEPENDENT full-PBC simulation, drops untrustworthy ones
     (solver-vs-sim gap >= GAP_TOL), and keeps the best few by INDEPENDENT-SIM target error;
  3. runs VERTEX-POSITION optimization (positions.design_with_positions) on the best topology with
     weights MATCHING the k-design (the real bug we fixed: nu_weight/E_weight must be passed into
     the position polish too), re-verifies, and records target error k-only vs k+positions;
  4. saves every kept design to Phase 5/networks/goal2/design_g2_<case>_<rank>.npz with full
     metadata (provenance, target arrays, generating tensor, errors, gap, topo_class);
  5. renders per-case nu(theta)/E(theta) overlays (+ polar), aggregated achieved-vs-target scatter
     (coloured by provenance and by topology class), and a best-per-case network montage — all
     under Phase 5/results/goal2/ with high-DPI elements/ saved separately.

Run:  "C:\Users\doron\anaconda3\python.exe" "Phase 5\verifications\run_goal2.py"
Quick smoke test (2 cases, tiny budget):  ... "Phase 5\verifications\run_goal2.py" --smoke
"""
import os, sys, json, glob, time, argparse
import numpy as np
import torch
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'Phase 3', 'verifications'))
import _common as C
from inverse_design import ANG
torch.set_default_dtype(torch.float64)
sys.path.insert(0, os.path.join(REPO, 'Phase 5'))
import seeds, designer, positions, triangulation, gallery
import phys_targets as PT
from plot_responses import response_from_npz

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

P5 = os.path.join(REPO, 'Phase 5')
NETDIR = os.path.join(P5, 'networks', 'goal2')
RESDIR = os.path.join(P5, 'results', 'goal2')
ELEMDIR = os.path.join(RESDIR, 'elements')
for d in (NETDIR, RESDIR, ELEMDIR):
    os.makedirs(d, exist_ok=True)

# ---- design budget (small cells so the full run finishes overnight) --------------------------
NU_W, E_W = 3.0, 1.0                       # prioritise the (harder) directional nu anisotropy
GAP_TOL = 0.05                             # keep only trustworthy designs (solver-vs-sim honesty)
KEEP = 3                                   # designs kept & plotted per case
KDESIGN = dict(n_iter=70, n_restarts=1, reg=0.02, nu_weight=NU_W, E_weight=E_W)
# position polish budget — weights MATCH the k-design (the fixed bug)
POS_BUDGET = dict(n_outer=2, spsa_steps=20, n_iter=60, n_restarts=1, nu_weight=NU_W, E_weight=E_W)

PROV_COLOR = {'crystal': '#1f77b4', 'random': '#d62728', 'hand': '#2ca02c'}


# ---- shared topology pool --------------------------------------------------------------------
def _tag(geo, name):
    geo['seed_name'] = name
    geo['is_fictional'] = np.zeros(len(geo['bond_R']), bool)
    return geo


def build_pool(seed=0):
    """A small, anisotropy-rich pool reused across all cases: stretched/sheared Bravais lattices
    (span the crystal-like targets), random patches over several point processes, and a couple of
    NON-Delaunay edge-flip variants (M1b flavour).  Kept small so the overnight run finishes."""
    pool = []
    for phi, psi in [(1.0, 1.0), (1.3, 1.0), (1.5, 0.8), (0.8, 1.3), (1.2, 1.2), (0.7, 1.3)]:
        pool.append(_tag(C.make_lattice(phi, psi, half=3), f'bravais_phi{phi}_psi{psi}'))
    for i, proc in enumerate(['poisson_disk', 'blue_noise', 'poisson_disk', 'uniform']):
        rec = seeds.random_patch(48, seed=seed + i, process=proc)
        pool.append(_tag(rec['geo'], rec['name']))
    # non-Delaunay flips of the regular lattice (reach topologies Delaunay cannot).
    # SCREEN them: `random_flipped_geo` can collapse a triangle, and an unscreened degenerate
    # topology in this SHARED pool poisons every case that reaches it — the sim refuses it
    # (UnhealthyGeometryError) and the case dies. Measured 2026-08-17: `flipped_reg_s50` had min
    # triangle area 1.1e-16 (machine epsilon) against a mean of 0.433, and killed 10 of 13 cases.
    # `run_goal1` and `run_g1_2` already search for a healthy seed like this; goal2 did not.
    base = C.make_lattice(1.0, 1.0, half=3)
    found = 0
    for s in range(seed + 50, seed + 200):
        if found == 2:
            break
        gf = triangulation.random_flipped_geo(base, 6, seed=s)
        a = np.asarray(gf['areas'], float)
        if a.min() > 1e-3 * a.mean():                    # same criterion as require_healthy_mesh
            pool.append(_tag(gf, f'flipped_reg_s{s}'))
            found += 1
    return pool


# ---- one case --------------------------------------------------------------------------------
def design_case(t, pool):
    """Design one physical target `t` (a phys_targets Target dict).  Returns a result dict with the
    kept designs, k-only vs k+positions errors, gap, and the saved .npz paths."""
    nu_t, E_t = t['nu'], t['E']
    case = t['label']
    print(f"\n{'=' * 96}\nCASE {case}  ({t['provenance']} / {t['topo_class']})")
    print(f"  target nu in [{nu_t.min():+.3f},{nu_t.max():+.3f}]  "
          f"E in [{E_t.min():.3f},{E_t.max():.3f}]")

    # (i) k-only design on every pool topology, ranked by solver loss
    ranked = designer.search(nu_t, E_t, pool, keep=len(pool), **KDESIGN)

    # (ii) independent-sim verification of a generous candidate set; drop untrustworthy
    verified = []
    for geo, k, loss in ranked[:max(2 * KEEP, 6)]:
        rep = designer.verify(geo, k, nu_t, E_t)
        verified.append((geo, k, loss, rep))
    trust = [v for v in verified if v[3]['solver_sim_gap'] < GAP_TOL]
    usable = trust if trust else verified
    usable.sort(key=lambda v: v[3]['target_err_sim'])
    kept = usable[:KEEP]
    konly_best = kept[0]
    konly_err = konly_best[3]['target_err_sim']

    # (iii) vertex-position polish of the best topology (weights matched to the k-design)
    geo0, k0, _loss0, _rep0 = konly_best
    geoP, kP, _hist = positions.design_with_positions(nu_t, E_t, geo0, verbose=False, **POS_BUDGET)
    repP = designer.verify(geoP, kP, nu_t, E_t)
    kpos_err = repP['target_err_sim']
    kpos_ok = (repP['solver_sim_gap'] < GAP_TOL)
    improved = kpos_ok and (kpos_err < konly_err)
    print(f"  k-only best err {konly_err:.4f} (gap {konly_best[3]['solver_sim_gap']:.3f}) "
          f"[{geo0['seed_name']}]  ->  k+pos err {kpos_err:.4f} (gap {repP['solver_sim_gap']:.3f})"
          f"{'  IMPROVED' if improved else '  (kept k-only)'}")

    # rank0 = polished design if it improved AND is trustworthy, else the k-only best
    if improved:
        _tag(geoP, str(geo0['seed_name']) + '+pos')
        kept = [(geoP, kP, _loss0, repP)] + kept[1:]

    # (iv) save each kept design with full metadata
    paths = []
    for rank, (geo, k, loss, rep) in enumerate(kept):
        path = os.path.join(NETDIR, f'design_g2_{case}_{rank}.npz')
        C.apply_k_to_geo(geo, k)
        C.save_network(path, geo, k, C6_per=rep['C6_per'],
                       target_nu=nu_t.tolist(), target_E=E_t.tolist(),
                       c6_target=t['c6'].tolist(), provenance=t['provenance'],
                       topo_class=t['topo_class'], case=case, rank=rank,
                       seed_name=str(geo.get('seed_name', 'unknown')),
                       loss=float(loss), solver_sim_gap=rep['solver_sim_gap'],
                       target_err_sim=rep['target_err_sim'],
                       konly_err=float(konly_err), kpos_err=float(kpos_err),
                       note=f"{case}\n{t['provenance']}")
        paths.append(path)

    return dict(case=case, provenance=t['provenance'], topo_class=t['topo_class'],
                nu_target=nu_t, E_target=E_t,
                konly_err=konly_err, kpos_err=kpos_err, improved=improved,
                best_gap=kept[0][3]['solver_sim_gap'],
                best_seed=str(kept[0][0].get('seed_name', 'unknown')),
                best_nu_sim=kept[0][3]['nu_sim'], best_E_sim=kept[0][3]['E_sim'],
                paths=paths, n_kept=len(kept),
                trustworthy=bool(kept[0][3]['solver_sim_gap'] < GAP_TOL))


# ---- plotting --------------------------------------------------------------------------------
def plot_case_responses(case):
    """nu(theta) & E(theta): all kept designs' INDEPENDENT-sim curves vs the dashed target
    (Cartesian + polar).  Saves results/goal2/response_<case>.png and polar_<case>.png."""
    paths = sorted(glob.glob(os.path.join(NETDIR, f'design_g2_{case}_*.npz')))
    if not paths:
        return
    cmap = plt.get_cmap('viridis')
    curves, meta = [], {}
    for i, p in enumerate(paths):
        nu, E, meta = response_from_npz(p)
        curves.append((nu, E))
    tnu = np.broadcast_to(np.asarray(meta['target_nu'], float), ANG.shape)
    tE = np.broadcast_to(np.asarray(meta['target_E'], float), ANG.shape)

    # Cartesian
    fig, (ax_nu, ax_E) = plt.subplots(1, 2, figsize=(11.5, 4.8))
    for i, (nu, E) in enumerate(curves):
        col = cmap(i / max(len(curves) - 1, 1))
        ax_nu.plot(ANG, nu, color=col, lw=1.6, label=f"#{i}")
        ax_E.plot(ANG, E, color=col, lw=1.6)
    ax_nu.plot(ANG, tnu, 'k--', lw=2.4, label='target')
    ax_E.plot(ANG, tE, 'k--', lw=2.4)
    for ax, ylab in ((ax_nu, r'$\nu(\theta)$'), (ax_E, r'$E(\theta)$')):
        ax.set_xlabel(r'$\theta$'); ax.set_ylabel(ylab); ax.set_xlim(0, np.pi)
        ax.set_xticks([0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi])
        ax.set_xticklabels(['0', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$'])
        ax.grid(alpha=0.25)
    ax_nu.set_title(f"{case} — sim " + r'$\nu(\theta)$'); ax_E.set_title(f"{case} — sim " + r'$E(\theta)$')
    ax_nu.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(RESDIR, f'response_{case}.png'), dpi=200)
    fig.savefig(os.path.join(ELEMDIR, f'response_{case}.png'), dpi=220)
    plt.close(fig)

    # Polar (full 0..2pi by pi-periodic mirroring) — square regions by construction
    th2 = np.concatenate([ANG, ANG + np.pi])
    fig = plt.figure(figsize=(10, 5))
    for j, (arr_t, arrs, ttl) in enumerate(
            [(tnu, [c[0] for c in curves], r'$\nu(\theta)$'),
             (tE, [c[1] for c in curves], r'$E(\theta)$')]):
        ax = fig.add_subplot(1, 2, j + 1, projection='polar')
        for i, a in enumerate(arrs):
            col = cmap(i / max(len(arrs) - 1, 1))
            ax.plot(th2, np.concatenate([a, a]), color=col, lw=1.4)
        ax.plot(th2, np.concatenate([arr_t, arr_t]), 'k--', lw=2.2)
        ax.set_title(f"{case}  {ttl}", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(RESDIR, f'polar_{case}.png'), dpi=200)
    plt.close(fig)


def plot_scatter(results, color_by='provenance'):
    """Aggregated achieved(sim)-vs-target scatter over all cases & angles for nu and E, coloured by
    provenance or by topology class.  Saves results/goal2/scatter_by_<color_by>.png."""
    keyset = sorted({r[color_by] for r in results})
    if color_by == 'provenance':
        cmap = PROV_COLOR
    else:
        base = plt.get_cmap('tab10')
        cmap = {k: base(i % 10) for i, k in enumerate(keyset)}

    fig, (ax_nu, ax_E) = plt.subplots(1, 2, figsize=(11, 5.4))
    for r in results:
        col = cmap[r[color_by]]
        ax_nu.scatter(r['nu_target'], r['best_nu_sim'], s=10, color=col, alpha=0.6, edgecolors='none')
        ax_E.scatter(r['E_target'], r['best_E_sim'], s=10, color=col, alpha=0.6, edgecolors='none')
    for ax, lab in ((ax_nu, r'$\nu$'), (ax_E, r'$E$')):
        lo = min(ax.get_xlim()[0], ax.get_ylim()[0]); hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([lo, hi], [lo, hi], 'k--', lw=1.2, alpha=0.7)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect('equal')
        ax.set_xlabel(f'target {lab}'); ax.set_ylabel(f'achieved (sim) {lab}'); ax.grid(alpha=0.25)
    handles = [plt.Line2D([0], [0], marker='o', ls='', color=cmap[k], label=k) for k in keyset]
    ax_nu.legend(handles=handles, fontsize=8, title=color_by)
    ax_nu.set_title(r'achieved vs target $\nu(\theta)$'); ax_E.set_title(r'achieved vs target $E(\theta)$')
    fig.suptitle(f'Goal 2 — achieved (independent sim) vs target, all cases & angles '
                 f'(colour = {color_by})')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(RESDIR, f'scatter_by_{color_by}.png')
    fig.savefig(out, dpi=200); fig.savefig(os.path.join(ELEMDIR, f'scatter_by_{color_by}.png'), dpi=220)
    plt.close(fig)
    return out


# ---- main ------------------------------------------------------------------------------------
def main(smoke=False):
    t_start = time.time()
    targets, rejects = PT.all_targets(seed=0)
    if smoke:
        targets = [targets[0], targets[5], targets[9]]        # 1 crystal, 1 random, 1 hand
        global POS_BUDGET
        POS_BUDGET = dict(n_outer=1, spsa_steps=8, n_iter=40, n_restarts=1, nu_weight=NU_W, E_weight=E_W)

    print("=" * 96)
    print(f"GOAL 2 — {len(targets)} PHYSICAL targets to design; "
          f"{len(rejects)} candidates REJECTED as non-physical")
    print("REJECTIONS (why non-physical):")
    for r in rejects:
        print(f"    [{r['provenance']:7s}] {r['label']}: {r['reason']}")
    print("=" * 96)

    pool = build_pool(seed=0)
    print(f"shared pool: {len(pool)} topologies "
          f"(bond counts {[len(g['bond_R']) for g in pool]})")

    results = []
    for t in targets:
        try:
            results.append(design_case(t, pool))
        except Exception as e:                                # noqa: BLE001
            print(f"  !! CASE {t['label']} FAILED: {type(e).__name__}: {e}")

    # per-case response figures
    for r in results:
        plot_case_responses(r['case'])

    # aggregated scatters (by provenance and by topology class)
    s1 = plot_scatter(results, 'provenance')
    s2 = plot_scatter(results, 'topo_class')

    # best-per-case montage (rank0 of each case)
    best_paths = [r['paths'][0] for r in results if r['paths']]
    montage = os.path.join(RESDIR, 'gallery_g2.png')
    gallery.gallery(best_paths, montage, ncols=min(4, max(1, len(best_paths))))

    # ---- summary table -----------------------------------------------------------------------
    print("\n" + "=" * 110)
    print(f"{'case':30s} {'prov':7s} {'topo':16s} {'k-only':>8} {'k+pos':>8} "
          f"{'d(help)':>8} {'gap':>7} {'trust':>6}")
    print("-" * 110)
    for r in results:
        print(f"{r['case'][:30]:30s} {r['provenance']:7s} {r['topo_class'][:16]:16s} "
              f"{r['konly_err']:>8.4f} {r['kpos_err']:>8.4f} "
              f"{r['konly_err'] - r['kpos_err']:>+8.4f} {r['best_gap']:>7.3f} "
              f"{str(r['trustworthy']):>6s}")
    print("-" * 110)
    n_trust = sum(r['trustworthy'] for r in results)
    n_helped = sum(r['improved'] for r in results)
    print(f"designed {len(results)} physical cases; {n_trust} trustworthy (gap<{GAP_TOL}); "
          f"positions helped in {n_helped}/{len(results)} cases")
    mean_help = np.mean([r['konly_err'] - r['kpos_err'] for r in results]) if results else 0.0
    print(f"mean target-error improvement (k-only - k+pos): {mean_help:+.4f}  "
          f"(positive = positions HELPED on average)")

    # ---- machine-readable summary ------------------------------------------------------------
    summary = dict(
        n_targets=len(targets), n_rejected=len(rejects),
        rejections=rejects, gap_tol=GAP_TOL, nu_weight=NU_W, E_weight=E_W,
        cases=[dict(case=r['case'], provenance=r['provenance'], topo_class=r['topo_class'],
                    konly_err=r['konly_err'], kpos_err=r['kpos_err'], improved=r['improved'],
                    best_gap=r['best_gap'], best_seed=r['best_seed'], trustworthy=r['trustworthy'],
                    n_kept=r['n_kept'], paths=[os.path.relpath(p, P5) for p in r['paths']])
               for r in results],
        figures=dict(scatter_provenance=os.path.relpath(s1, P5),
                     scatter_topo=os.path.relpath(s2, P5),
                     montage=os.path.relpath(montage, P5)))
    with open(os.path.join(RESDIR, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=lambda o: o.tolist() if hasattr(o, 'tolist') else o)

    print(f"\nfigures + summary under {RESDIR}")
    print(f"GOAL 2 DONE in {time.time() - t_start:.0f}s "
          f"({len(results)} cases, {n_trust} trustworthy)")
    return results


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true', help='quick 3-case smoke test, tiny budget')
    args = ap.parse_args()
    main(smoke=args.smoke)
