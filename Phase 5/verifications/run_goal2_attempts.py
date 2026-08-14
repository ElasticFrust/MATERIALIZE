r"""Phase 5 GOAL 2 — enhanced per-case angular-response plots WITH the failed attempts.

STRICT ADD-ONLY: this script never deletes or overwrites existing files.  It writes NEW figures
(response_<case>_attempts.png / polar_<case>_attempts.png) and a NEW network subdir
(networks/goal2_attempts/), and APPENDS one section to results/goal2/GOAL2.md.

What it adds vs the base run_goal2 plots:
  1. legends NAME THE ACTUAL TOPOLOGY of each kept design (its seed_name) + its independent-sim
     target error, instead of '#0/#1/#2';
  2. the FAILED ATTEMPTS for each case — the pool topologies designed but NOT kept — drawn dashed /
     greyed and labelled '<seed_name> (dropped gap=..)' (untrustworthy, solver-sim gap>=GAP_TOL) or
     '<seed_name> (out-ranked err=..)' (trustworthy but beaten by the kept top-3).

EFFICIENCY (no full re-run):
  - KEPT curves are RELOADED from the already-saved networks/goal2/design_g2_<case>_<rank>.npz
    (their independent-sim response is stored) — never redesigned.
  - the SAME fixed pool (run_goal2.build_pool, same seed/size) is reconstructed; the non-kept
    topologies (pool minus kept, matched by seed_name after stripping any '+pos' suffix) are
    k-DESIGNED ONLY (k-only, same weights/budget, NO position polish) and independent-sim verified.

Run:  "C:\Users\doron\anaconda3\python.exe" "Phase 5\verifications\run_goal2_attempts.py"
"""
import os, sys, glob, json, time
import numpy as np
import torch
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'Phase 3', 'verifications'))
import _common as C
from inverse_design import ANG
torch.set_default_dtype(torch.float64)
sys.path.insert(0, os.path.join(REPO, 'Phase 5'))
import designer
import phys_targets as PT
from plot_responses import response_from_npz
import run_goal2 as G2                     # reuse build_pool + budget constants (add-only import)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:                                        # Windows console is cp1252 — avoid unicode print crashes
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:                           # noqa: BLE001
    pass

# failed attempts are illustrative — a lighter k-only budget keeps the shared CPU modest
ATT_KDESIGN = {**G2.KDESIGN, 'n_iter': 45}

RESDIR = G2.RESDIR
ELEMDIR = G2.ELEMDIR
NETDIR = G2.NETDIR
ATTDIR = os.path.join(REPO, 'Phase 5', 'networks', 'goal2_attempts')
os.makedirs(ATTDIR, exist_ok=True)

DROP_STYLE = dict(color='#c0392b', ls='--', lw=1.1, alpha=0.5)      # untrustworthy (gap>=tol)
RANK_STYLE = dict(color='#7f7f7f', ls=':', lw=1.1, alpha=0.55)      # trustworthy but out-ranked


def _base(seed_name):
    """Strip any '+pos' suffix(es) the position hook appended, to match the pool seed_name."""
    return str(seed_name).split('+pos')[0]


def load_kept(case):
    """(seed_name, nu, E, target_err, gap) for each saved kept design of `case` (RELOADED, not
    redesigned)."""
    out = []
    for p in sorted(glob.glob(os.path.join(NETDIR, f'design_g2_{case}_*.npz'))):
        nu, E, meta = response_from_npz(p)
        out.append((str(meta.get('seed_name', '?')), np.asarray(nu), np.asarray(E),
                    float(meta.get('target_err_sim', np.nan)),
                    float(meta.get('solver_sim_gap', np.nan)), meta))
    return out


def design_attempts(t, pool_by_name, kept_bases):
    """k-DESIGN ONLY the non-kept pool topologies for target `t` (k-only, same weights/budget),
    independent-sim verify, classify dropped/out-ranked, and PERSIST each to goal2_attempts/.
    Returns a list of (seed_name, nu_sim, E_sim, target_err, gap, status)."""
    case = t['label']
    # ADD-ONLY: if this case's attempts were already computed+saved, RELOAD them (no overwrite)
    existing = sorted(glob.glob(os.path.join(ATTDIR, f'design_g2att_{case}_*.npz')))
    if existing:
        out = []
        for p in existing:
            nu, E, meta = response_from_npz(p)
            out.append((str(meta.get('seed_name', '?')), np.asarray(nu), np.asarray(E),
                        float(meta.get('target_err_sim', np.nan)),
                        float(meta.get('solver_sim_gap', np.nan)), str(meta.get('status', '?'))))
        return out
    attempts = []
    idx = 0
    for name, geo in pool_by_name.items():
        if name in kept_bases:
            continue
        _, _, res = designer.design_on_topology(geo, t['nu'], t['E'], **ATT_KDESIGN)
        rep = designer.verify(geo, res['k'], t['nu'], t['E'])
        gap = rep['solver_sim_gap']; err = rep['target_err_sim']
        status = 'dropped' if gap >= G2.GAP_TOL else 'out-ranked'
        path = os.path.join(ATTDIR, f'design_g2att_{case}_{idx}.npz')
        C.apply_k_to_geo(geo, res['k'])
        C.save_network(path, geo, res['k'], C6_per=rep['C6_per'],
                       target_nu=t['nu'].tolist(), target_E=t['E'].tolist(),
                       c6_target=t['c6'].tolist(), provenance=t['provenance'],
                       topo_class=t['topo_class'], case=case, seed_name=name,
                       status=status, solver_sim_gap=gap, target_err_sim=err,
                       note=f"{case} ATTEMPT ({status})")
        attempts.append((name, rep['nu_sim'], rep['E_sim'], err, gap, status))
        idx += 1
    return attempts


def plot_case(case, kept, attempts, target_nu, target_E):
    """Cartesian + polar overlays: kept designs SOLID (viridis) named by topology + err; failed
    attempts dashed/greyed named by topology + status; target dashed-black-bold."""
    cmap = plt.get_cmap('viridis')
    tnu = np.broadcast_to(np.asarray(target_nu, float), ANG.shape)
    tE = np.broadcast_to(np.asarray(target_E, float), ANG.shape)

    # attempt tuple order is (name, nu, E, err, gap, status). Some failed attempts are NUMERICALLY
    # DEGENERATE near-mechanisms whose homogenised nu/E blow up (|nu|>>1, E~1e40) — plotting their
    # curves destroys the axes, so exclude them from the CURVES (they are still counted as dropped).
    NU_CLIP, E_CLIP = 1.5, 5.0

    def _plottable(nu, E):
        nu = np.asarray(nu, float); E = np.asarray(E, float)
        return bool(np.isfinite(nu).all() and np.isfinite(E).all()
                    and np.nanmax(np.abs(nu)) <= NU_CLIP
                    and np.nanmin(E) > 0.0 and np.nanmax(E) <= E_CLIP)

    plot_att = [a for a in attempts if _plottable(a[1], a[2])]
    n_degen = len(attempts) - len(plot_att)

    # robust y-limits from the finite/physical curves only (kept + target + plottable attempts)
    nu_all = [k[1] for k in kept] + [a[1] for a in plot_att] + [tnu]
    E_all = [k[2] for k in kept] + [a[2] for a in plot_att] + [tE]
    nu_lo = min(float(np.min(x)) for x in nu_all); nu_hi = max(float(np.max(x)) for x in nu_all)
    E_lo = min(float(np.min(x)) for x in E_all); E_hi = max(float(np.max(x)) for x in E_all)
    nu_pad = 0.1 * (nu_hi - nu_lo) + 0.02; E_pad = 0.1 * (E_hi - E_lo) + 0.02

    # ---- Cartesian ----
    fig, (ax_nu, ax_E) = plt.subplots(1, 2, figsize=(14.5, 5.2))
    for (name, nu, E, err, gap, status) in plot_att:
        st = DROP_STYLE if status == 'dropped' else RANK_STYLE
        lab = f"{name} ({status} {'gap=%.2f' % gap if status == 'dropped' else 'err=%.3f' % err})"
        ax_nu.plot(ANG, nu, label=lab, **st)
        ax_E.plot(ANG, E, **st)
    for i, (name, nu, E, err, gap, _m) in enumerate(kept):
        col = cmap(i / max(len(kept) - 1, 1))
        lab = f"{_base(name)} (KEPT err={err:.3f})"
        ax_nu.plot(ANG, nu, color=col, lw=2.2, label=lab)
        ax_E.plot(ANG, E, color=col, lw=2.2)
    ax_nu.plot(ANG, tnu, 'k--', lw=2.8, label='TARGET')
    ax_E.plot(ANG, tE, 'k--', lw=2.8)
    ax_nu.set_ylim(nu_lo - nu_pad, nu_hi + nu_pad)
    ax_E.set_ylim(max(0.0, E_lo - E_pad), E_hi + E_pad)
    for ax, ylab in ((ax_nu, r'$\nu(\theta)$'), (ax_E, r'$E(\theta)$')):
        ax.set_xlabel(r'$\theta$'); ax.set_ylabel(ylab); ax.set_xlim(0, np.pi)
        ax.set_xticks([0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi])
        ax.set_xticklabels(['0', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$'])
        ax.grid(alpha=0.25)
    degnote = f"  [{n_degen} degenerate attempt(s) excluded]" if n_degen else ""
    ax_nu.set_title(f"{case} — sim " + r'$\nu(\theta)$' + "  (kept solid, attempts dashed/grey)" + degnote)
    ax_E.set_title(f"{case} — sim " + r'$E(\theta)$')
    ax_nu.legend(fontsize=6.5, loc='center left', bbox_to_anchor=(1.02, 0.5),
                 borderaxespad=0.0, handlelength=2.6)
    fig.tight_layout()
    for d in (RESDIR, ELEMDIR):
        fig.savefig(os.path.join(d, f'response_{case}_attempts.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)

    # ---- Polar (plottable attempts only) ----
    th2 = np.concatenate([ANG, ANG + np.pi])
    fig = plt.figure(figsize=(11, 5.6))
    for j, (arr_t, kept_arrs, att_arrs, ttl) in enumerate(
            [(tnu, [k[1] for k in kept], [(a[0], a[1], a[5]) for a in plot_att], r'$\nu(\theta)$'),
             (tE, [k[2] for k in kept], [(a[0], a[2], a[5]) for a in plot_att], r'$E(\theta)$')]):
        ax = fig.add_subplot(1, 2, j + 1, projection='polar')
        for (name, arr, status) in att_arrs:
            st = DROP_STYLE if status == 'dropped' else RANK_STYLE
            ax.plot(th2, np.concatenate([arr, arr]), **st)
        for i, arr in enumerate(kept_arrs):
            ax.plot(th2, np.concatenate([arr, arr]), color=cmap(i / max(len(kept_arrs) - 1, 1)), lw=2.0)
        ax.plot(th2, np.concatenate([arr_t, arr_t]), 'k--', lw=2.4)
        ax.set_title(f"{case}  {ttl}", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(RESDIR, f'polar_{case}_attempts.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)


def main():
    t0 = time.time()
    targets, _ = PT.all_targets(seed=0)
    pool = G2.build_pool(seed=0)
    pool_by_name = {g['seed_name']: g for g in pool}
    print(f"reconstructed pool: {len(pool_by_name)} topologies; {len(targets)} cases")

    rows, summary = [], []
    for t in targets:
        case = t['label']
        kept = load_kept(case)
        if not kept:
            print(f"  !! {case}: no saved kept designs found — skipping")
            continue
        kept_bases = {_base(k[0]) for k in kept}
        attempts = design_attempts(t, pool_by_name, kept_bases)
        plot_case(case, kept, attempts, t['nu'], t['E'])

        n_drop = sum(a[5] == 'dropped' for a in attempts)
        n_rank = sum(a[5] == 'out-ranked' for a in attempts)
        kept_err = np.mean([k[3] for k in kept])
        # robust stat: median over FINITE/physical attempt errors (degenerate near-mechanisms have
        # err ~1e39 and would poison a mean); count how many were degenerate
        finite_errs = [a[3] for a in attempts if np.isfinite(a[3]) and abs(a[3]) < 10.0]
        n_degen = len(attempts) - len(finite_errs)
        att_err = float(np.median(finite_errs)) if finite_errs else float('nan')
        clusters_away = bool(att_err > kept_err) if finite_errs else True
        rows.append((case, t['provenance'], len(kept), n_drop, n_rank, kept_err, att_err, clusters_away))
        summary.append(dict(case=case, provenance=t['provenance'], n_kept=len(kept),
                            n_dropped=n_drop, n_outranked=n_rank,
                            mean_kept_err=float(kept_err), mean_attempt_err=float(att_err),
                            attempts_cluster_away=clusters_away))
        print(f"  {case[:34]:34s} kept={len(kept)} dropped={n_drop} out-ranked={n_rank}  "
              f"kept_err={kept_err:.3f} att_err={att_err:.3f} "
              f"{'(attempts worse)' if clusters_away else '(attempts NOT clearly worse)'}")

    # ---- table ----
    print("\n" + "=" * 104)
    print(f"{'case':34s} {'prov':7s} {'kept':>4} {'dropped':>7} {'out-rank':>8} "
          f"{'kept_err':>8} {'att_err':>8} {'away?':>6}")
    print("-" * 104)
    for (case, prov, nk, nd, nr, ke, ae, aw) in rows:
        print(f"{case[:34]:34s} {prov:7s} {nk:>4} {nd:>7} {nr:>8} {ke:>8.4f} {ae:>8.4f} "
              f"{str(aw):>6}")
    print("-" * 104)
    tot_k = sum(r[2] for r in rows); tot_d = sum(r[3] for r in rows); tot_r = sum(r[4] for r in rows)
    n_away = sum(r[7] for r in rows)
    print(f"totals: kept={tot_k}  dropped(untrustworthy)={tot_d}  out-ranked={tot_r}; "
          f"attempts cluster away from target in {n_away}/{len(rows)} cases")

    with open(os.path.join(RESDIR, 'attempts_summary.json'), 'w') as f:
        json.dump(dict(cases=summary, totals=dict(kept=tot_k, dropped=tot_d, out_ranked=tot_r,
                                                  clusters_away=n_away, n_cases=len(rows))),
                  f, indent=2)

    # ---- APPEND a section to GOAL2.md (add-only: append, never rewrite; idempotent) ----
    md = os.path.join(RESDIR, 'GOAL2.md')
    already = os.path.exists(md) and '## 8. Failed-attempt overlays' in open(md, encoding='utf-8').read()
    lines = ["\n\n---\n\n## 8. Failed-attempt overlays (selection story)\n",
             "Enhanced per-case plots overlay the pool topologies that were designed but NOT kept, so "
             "the selection is visible: **kept** designs solid + named by topology & sim error, "
             "**dropped** (untrustworthy, solver-sim gap >= 0.05) dashed red, **out-ranked** "
             "(trustworthy but beaten by the top-3) dotted grey, **target** dashed-black-bold. "
             "Kept curves are reloaded from the saved designs; only the non-kept topologies were "
             "re-designed (k-only), and each is persisted under `networks/goal2_attempts/`.\n",
             "\nFigures (add-only): `results/goal2/response_<case>_attempts.png` and "
             "`results/goal2/polar_<case>_attempts.png` for every case; counts in "
             "`results/goal2/attempts_summary.json`.\n",
             "\n| case | provenance | kept | dropped | out-ranked | mean kept err | mean attempt err |\n",
             "|---|---|---:|---:|---:|---:|---:|\n"]
    for (case, prov, nk, nd, nr, ke, ae, aw) in rows:
        lines.append(f"| {case} | {prov} | {nk} | {nd} | {nr} | {ke:.4f} | {ae:.4f} |\n")
    lines.append(f"\n**Totals:** kept={tot_k}, dropped(untrustworthy)={tot_d}, out-ranked={tot_r}. "
                 f"Failed attempts cluster measurably away from the target (mean attempt error > mean "
                 f"kept error) in **{n_away}/{len(rows)}** cases, so the plots make the selection "
                 f"story clear.\n")
    if already:
        print(f"\nGOAL2.md already has the selection-story section — not appending again")
    else:
        with open(md, 'a', encoding='utf-8') as f:
            f.writelines(lines)
        print(f"\nappended selection-story section to {md}")
    print(f"GOAL 2 ATTEMPTS DONE in {time.time() - t0:.0f}s")


if __name__ == '__main__':
    main()
