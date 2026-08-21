r"""Phase 5 — EXPERIMENT G1.2 visualization.  Loads Phase 5/results/g1_2/{results.csv,
eta_reference.npz} and produces the four required figures (+ each panel as its own high-DPI
element in results/g1_2/elements/):

  1. reachable-nu per topology: solid bar = reached with solver and sim AGREEING (gap < 0.05),
     pale bar = full reach INCLUDING designs the honesty gate rejected; undistorted nu marked;
     eta-disordered-triangular reference band overlaid.
  2. achieved-vs-target nu scatter (y=x), coloured by topology.
  3. per-topology nu_target -> nu_achieved curves (one line per topology).

  EVERY design is drawn (2026-08-21). FILLED marker = trustworthy (solver and sim agree within
  `gap < 0.05`); HOLLOW marker = the two code paths disagree by more than that, with a bar spanning
  nu_sim..nu_solver so the disagreement is visible instead of hidden. Plotting only trustworthy rows
  is what made G1.2 read as "distortion never reaches negative nu" when 43 designs reach nu < 0 in
  BOTH code paths (median |dnu| = 0.008) — the gate is a RELATIVE DIRECTIONAL criterion, not a
  verdict on whether the design is real.
  4. montage (gallery.gallery) of distorted networks at their extreme reachable nu.

Square plot regions.  Run:  C:\\Users\\doron\\anaconda3\\python.exe "Phase 5/verifications/plot_g1_2.py"
"""
import os, sys, csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RESDIR = os.path.join(REPO, 'Phase 5', 'results', 'g1_2')
ELDIR = os.path.join(RESDIR, 'elements')
sys.path.insert(0, os.path.join(REPO, 'Phase 5'))

# ---- palette (project categorical hues extended to 10 topologies) ----------------------------
TOPO_ORDER = ['triangular', 'honeycomb', 'kagome', 'square_octagon', 'tetrakis',
              'rotating_squares', 'reentrant_honeycomb', 'foam_poisson',
              'flipped_tri_f8', 'flipped_tri_f14']   # f16 was dropped (near-singular, nu~646)
TOPO_COLOR = dict(zip(TOPO_ORDER, ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4',
                                   '#7b5cd6', '#d64550', '#20b2c4', '#8a8a2a', '#a9682f']))
TOPO_MARK = dict(zip(TOPO_ORDER, ['o', 's', '^', 'D', 'v', 'P', 'X', '*', '<', '>']))
INK, MUTED, GRID = '#0b0b0b', '#898781', '#e1e0d9'
ETA_BAND = '#c7c3b6'

NUM = {'run_id', 'n_nodes', 'n_bond', 'nu_target', 'nu_initial', 'nu_achieved_sim',
       'E_achieved_sim', 'err', 'nu_aniso_std', 'solver_sim_gap', 'trustworthy', 'design_loss'}


def load_rows(trustworthy_only=True):
    with open(os.path.join(RESDIR, 'results.csv')) as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k in list(r):
            if k in NUM:
                r[k] = float(r[k])
    if trustworthy_only:
        rows = [r for r in rows if r['trustworthy'] >= 0.5]
    return rows


def solver_recheck():
    """run_id -> (nu_sim, nu_solver), both angle-averaged, from `g1_2_solver_recheck.py`.

    The SOLVER side is absent from results.csv (only the composite gap is stored), so this is what
    lets a rejected design be drawn with its actual disagreement. Empty dict if not yet computed."""
    path = os.path.join(RESDIR, 'solver_recheck.npz')
    if not os.path.exists(path):
        return {}
    d = np.load(path, allow_pickle=True)
    return {int(rid): (float(np.mean(ns)), float(np.mean(nv)))
            for rid, ns, nv, ok in zip(d['run_id'], d['nu_sim'], d['nu_solver'], d['ok']) if ok}


def repeated_endpoints():
    """run_ids whose saved GEOMETRY is shared with another target and which failed the gate.

    One configuration reported for several targets means the search stopped there. At the positive
    end that is genuine saturation at the nu ~ +1/3 cap (those clusters are trustworthy); at the
    negative end it is a LOWER BOUND on what the topology can do, never an upper bound. `triangular`
    is the proven case (`g1_2_triangular_start_probe.py`): at SPSA step size a=0.15 the search barely
    moves and lands on a TARGET-INDEPENDENT endpoint -- identical for nu*=-0.10 and -0.30 -- which is
    why one geometry was saved for all five negative targets. At a=0.25 the same setup reaches
    nu=-0.134, but with a larger gap, so `design_one`'s prefer-trustworthy rule discards it."""
    path = os.path.join(RESDIR, 'solver_recheck.npz')
    if not os.path.exists(path):
        return set()
    d = np.load(path, allow_pickle=True)
    if 'geo_sha' not in d.files:
        return set()
    groups = {}
    for rid, topo, sha, tr in zip(d['run_id'], d['topo'], d['geo_sha'], d['trustworthy']):
        groups.setdefault((str(topo), str(sha)), []).append((int(rid), bool(tr)))
    return {rid for members in groups.values() if len(members) > 1
            for rid, tr in members if not tr}


def _face(row, color):
    """Marker fill by trust: solid where solver and sim agree, HOLLOW where they do not."""
    return color if row['trustworthy'] >= 0.5 else 'none'


def _trust_proxies(ax):
    """One legend entry for the fill convention, drawn off-axis (never covers data)."""
    ax.scatter([], [], s=42, color=INK, marker='o', edgecolor='white', linewidth=0.5,
               label='trustworthy (solver = sim)')
    ax.scatter([], [], s=42, facecolor='none', marker='o', edgecolor=INK, linewidth=1.2,
               label='solver-sim gap > 0.05 (bar = disagreement)')


def eta_band():
    d = np.load(os.path.join(RESDIR, 'eta_reference.npz'))
    nu = d['nu']
    nu666 = d['nu'][d['is_666'] > 0.5]
    return (float(nu.min()), float(nu.max()),
            float(nu666.min()) if len(nu666) else float(nu.min()),
            float(nu666.max()) if len(nu666) else float(nu.max()))


def _chrome(ax):
    ax.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_color(INK)


def _save_element(draw, name, figsize=(4.8, 4.8)):
    os.makedirs(ELDIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=figsize)
    draw(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(ELDIR, name), dpi=300)
    plt.close(fig)


def _present_topos(rows):
    return [t for t in TOPO_ORDER if any(r['topo'] == t for r in rows)]


# ---- 1. reachable-nu per topology ------------------------------------------------------------
# The reach bar is a RIBBON whose half-thickness at each achieved nu is that design's
# |nu_solver - nu_sim|. So it is thin where the two code paths agree and swells where they diverge --
# the uncertainty is resolved ALONG nu instead of collapsed to one number per topology.
# One scale for the whole figure, so thicknesses are comparable between rows.
U_HALF_MAX = 0.34       # row half-height given to the largest |dnu| in the figure, in row units


def _reach_axis(ax):
    rows = load_rows(trustworthy_only=False)
    rech = solver_recheck()
    topos = _present_topos(rows)
    lo6, hi6 = eta_band()[2], eta_band()[3]
    loA, hiA = eta_band()[0], eta_band()[1]
    ax.axvspan(loA, hiA, color=ETA_BAND, alpha=0.35, zorder=0,
               label='η-disorder triangular (all)')
    ax.axvspan(lo6, hi6, color=ETA_BAND, alpha=0.75, zorder=0,
               label='η-disorder triangular (6-6-6 only)')
    stalled = repeated_endpoints()
    u_all = [abs(a - b) for a, b in rech.values()]
    u_max = max(u_all) if u_all else 0.0
    u_scale = U_HALF_MAX / u_max if u_max > 0 else 0.0
    for i, t in enumerate(topos):
        g = [r for r in rows if r['topo'] == t]
        # FULL reach, including designs the honesty gate rejected. Dropping these is what hid the
        # auxetic half of this experiment (audit 2026-08-21).
        pts = sorted((rech[int(r['run_id'])][0],
                      abs(rech[int(r['run_id'])][0] - rech[int(r['run_id'])][1]),
                      r['trustworthy'] >= 0.5,
                      int(r['run_id']) in stalled)
                     for r in g if int(r['run_id']) in rech)
        nus_all = [r['nu_achieved_sim'] for r in g]
        lo_a, hi_a = min(nus_all), max(nus_all)
        if pts:
            x = np.array([q[0] for q in pts])
            half = np.array([q[1] for q in pts]) * u_scale
            ax.fill_between(x, i - half, i + half, color=TOPO_COLOR[t], alpha=0.45,
                            linewidth=0, zorder=1)
            # spine: the reach itself, so a zero-uncertainty stretch stays visible
            ax.plot([lo_a, hi_a], [i, i], color=TOPO_COLOR[t], lw=2.0, solid_capstyle='round',
                    alpha=0.95, zorder=2)
            for xv, _hv, tr, st in pts:      # filled = trustworthy, hollow = gap > 0.05
                if st:                       # search returned one design for several targets
                    ax.plot(xv, i, 'X', markersize=7, markerfacecolor=TOPO_COLOR[t],
                            markeredgecolor='white', markeredgewidth=0.8, zorder=3)
                    continue
                ax.plot(xv, i, 'o', markersize=4.5,
                        markerfacecolor=TOPO_COLOR[t] if tr else 'none',
                        markeredgecolor='white' if tr else TOPO_COLOR[t],
                        markeredgewidth=0.6 if tr else 1.1, zorder=3)
        if t == 'triangular':
            # eta-disorder IS distorted triangular at k=1 with frozen 6-6-6 connectivity -- the same
            # design space this row searches. It reaches nu = -0.109, which the search never found,
            # so this row's negative end is a SEARCH limit, not a property of the topology.
            ax.plot(loA, i, '*', markersize=13, markerfacecolor=INK, markeredgecolor='white',
                    markeredgewidth=0.8, zorder=5)
            ax.annotate('η-disorder reaches %+.3f' % loA + chr(10) + 'on THIS topology',
                        (loA, i), textcoords='offset points', xytext=(4, -22),
                        ha='left', va='top', fontsize=7, color=INK)
        nu0 = g[0]['nu_initial']
        ax.plot(nu0, i, 'o', color='white', markeredgecolor=TOPO_COLOR[t], markersize=9,
                markeredgewidth=2.2, zorder=4)
        ax.annotate(f'{lo_a:+.2f}', (lo_a, i), textcoords='offset points', xytext=(-6, 0),
                    ha='right', va='center', fontsize=7.5, color=INK)
        ax.annotate(f'{hi_a:+.2f}', (hi_a, i), textcoords='offset points', xytext=(6, 0),
                    ha='left', va='center', fontsize=7.5, color=INK)
    ax.axvline(0, color=MUTED, lw=1, ls=':')
    # Reserve empty margin on the LEFT for the legend and the width key (never cover the data), and
    # so the min-value annotations are not clipped at the axis edge.
    xs = [r['nu_achieved_sim'] for r in rows] + [r['nu_initial'] for r in rows]
    x0 = min(xs) - 0.36
    ax.set_xlim(x0, max(xs) + 0.06)
    ax.set_yticks(range(len(topos)))
    ax.set_yticklabels(topos, color=INK, fontsize=9)
    ax.set_ylim(-0.8, len(topos) - 0.3)
    ax.set_xlabel('reachable (independent-sim) ν  [min .. max];  ○ = undistorted (k=1);'
                  '  ribbon thickness = solver-vs-sim disagreement at that ν', color=INK)
    ax.set_title('G1.2 — isotropic ν reachable by GEOMETRIC DISTORTION alone (k=1 fixed)',
                 color=INK, fontsize=11)
    _reach_legend(ax, x0, u_scale, u_max)
    _chrome(ax)


def _reach_legend(ax, x0, u_scale, u_max):
    """Legend plus an explicit WIDTH KEY -- a reference ribbon of known |dnu| in the left margin, so
    the thickness can be read quantitatively and not merely compared row to row."""
    ax.plot([], [], 'o', markersize=5, markerfacecolor=INK, markeredgecolor='white',
            label='trustworthy (solver = sim)')
    ax.plot([], [], 'o', markersize=5, markerfacecolor='none', markeredgecolor=INK,
            label='solver-sim gap > 0.05')
    ax.fill_between([], [], [], color=INK, alpha=0.45,
                    label='ribbon half-width = |ν_solver − ν_sim|')
    ax.plot([], [], 'X', markersize=7, markerfacecolor=INK, markeredgecolor='white',
            label='search stopped here (lower bound only)')
    ax.legend(fontsize=7, loc='upper left', framealpha=0.95)
    yk, h = -0.52, u_max * u_scale
    xk0, xk1 = x0 + 0.03, x0 + 0.15
    ax.fill_between([xk0, xk1], [yk - h, yk - h], [yk + h, yk + h],
                    color=INK, alpha=0.35, linewidth=0, zorder=3)
    ax.annotate(f'= {u_max:.3f} in ν', (xk1 + 0.012, yk), va='center', ha='left',
                fontsize=7, color=INK)


def fig_reach():
    _save_element(_reach_axis, 'fig1_reachable_by_topology.png', figsize=(8.2, 5.6))
    fig, ax = plt.subplots(figsize=(9.0, 6.0))
    _reach_axis(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(RESDIR, 'fig1_reachable_by_topology.png'), dpi=200)
    plt.close(fig)


# ---- 2. achieved vs target -------------------------------------------------------------------
def _scatter_axis(ax):
    rows = load_rows(trustworthy_only=False)
    rech = solver_recheck()
    tv = [r['nu_target'] for r in rows] + [r['nu_achieved_sim'] for r in rows]
    lo, hi = min(tv) - 0.05, max(tv) + 0.05
    ax.plot([lo, hi], [lo, hi], '--', color=MUTED, lw=1.6, zorder=1, label='$y=x$ (ideal)')
    for t in _present_topos(rows):
        g = [r for r in rows if r['topo'] == t]
        for r in g:                    # solver-vs-sim disagreement bar (invisible where it is ~0)
            pair = rech.get(int(r['run_id']))
            if pair is not None:
                ax.plot([r['nu_target']] * 2, sorted(pair), '-', color=TOPO_COLOR[t], lw=1.2,
                        alpha=0.75, zorder=2)
        # Label whichever branch is drawn first, so a topology with NO trustworthy design
        # (square_octagon, reentrant_honeycomb) still appears in the legend.
        has_trust = any(r['trustworthy'] >= 0.5 for r in g)
        for trust in (True, False):
            gg = [r for r in g if (r['trustworthy'] >= 0.5) == trust]
            if not gg:
                continue
            ax.scatter([r['nu_target'] for r in gg], [r['nu_achieved_sim'] for r in gg],
                       s=42, marker=TOPO_MARK[t],
                       color=TOPO_COLOR[t] if trust else 'none',
                       edgecolor='white' if trust else TOPO_COLOR[t],
                       linewidth=0.5 if trust else 1.2,
                       label=t if trust == has_trust else None, zorder=3)
    _trust_proxies(ax)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect('equal', 'box')
    ax.set_xlabel(r'target $\nu$', color=INK)
    ax.set_ylabel(r'achieved (independent-sim) $\nu$', color=INK)
    ax.set_title('G1.2 — achieved vs target $\\nu$ (positions-only, k=1)', color=INK, fontsize=11)
    ax.legend(fontsize=7, loc='upper left', framealpha=0.9, ncol=2)
    _chrome(ax)


def fig_scatter():
    _save_element(_scatter_axis, 'fig2_achieved_vs_target.png', figsize=(5.6, 5.6))
    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    _scatter_axis(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(RESDIR, 'fig2_achieved_vs_target.png'), dpi=200)
    plt.close(fig)


# ---- 3. per-topology curves ------------------------------------------------------------------
def _curves_axis(ax):
    rows = load_rows(trustworthy_only=False)
    for t in _present_topos(rows):
        g = sorted([r for r in rows if r['topo'] == t], key=lambda r: r['nu_target'])
        ax.plot([r['nu_target'] for r in g], [r['nu_achieved_sim'] for r in g], '-',
                color=TOPO_COLOR[t], lw=1.6, label=t, zorder=2)
        for trust in (True, False):               # filled = solver agrees with sim; hollow = not
            gg = [r for r in g if (r['trustworthy'] >= 0.5) == trust]
            ax.plot([r['nu_target'] for r in gg], [r['nu_achieved_sim'] for r in gg],
                    linestyle='none', marker=TOPO_MARK[t], markersize=5,
                    markerfacecolor=TOPO_COLOR[t] if trust else 'none',
                    markeredgecolor='white' if trust else TOPO_COLOR[t],
                    markeredgewidth=0.4 if trust else 1.2, zorder=3)
    tv = [r['nu_target'] for r in rows]
    lo, hi = min(tv) - 0.05, max(tv) + 0.05
    ax.plot([lo, hi], [lo, hi], '--', color=MUTED, lw=1.3, zorder=0, label='$y=x$')
    ax.set_xlabel(r'target $\nu$', color=INK)
    ax.set_ylabel(r'achieved (sim) $\nu$', color=INK)
    ax.set_title('G1.2 — $\\nu_{\\rm target}\\!\\to\\!\\nu_{\\rm achieved}$ per topology',
                 color=INK, fontsize=11)
    ax.legend(fontsize=7, loc='upper left', framealpha=0.9, ncol=2)
    _chrome(ax)


def fig_curves():
    _save_element(_curves_axis, 'fig3_curves_per_topology.png', figsize=(6.0, 5.4))
    fig, ax = plt.subplots(figsize=(6.6, 5.8))
    _curves_axis(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(RESDIR, 'fig3_curves_per_topology.png'), dpi=200)
    plt.close(fig)


# ---- 4. montage of extreme distorted networks ------------------------------------------------
def fig_montage():
    import gallery
    # ALL rows: the most auxetic networks are exactly the ones the gate rejects, so a
    # trustworthy-only montage cannot show the extreme this experiment actually reaches.
    rows = load_rows(trustworthy_only=False)
    paths = []
    for t in _present_topos(rows):
        g = [r for r in rows if r['topo'] == t and r['design_path']]
        if not g:
            continue
        rmin = min(g, key=lambda r: r['nu_achieved_sim'])
        rmax = max(g, key=lambda r: r['nu_achieved_sim'])
        for r in ({rmin['design_path']: rmin, rmax['design_path']: rmax}).values():
            if r['design_path'] and os.path.exists(r['design_path']) and r['design_path'] not in paths:
                paths.append(r['design_path'])
    if not paths:
        print('[plot] no saved designs for montage')
        return
    out = os.path.join(RESDIR, 'fig4_montage.png')
    gallery.gallery(paths, out, ncols=4)
    print(f'[plot] montage -> {out} ({len(paths)} panels)')


def main():
    rows = load_rows(trustworthy_only=False)
    n_tr = sum(1 for r in rows if r['trustworthy'] >= 0.5)
    print(f'[plot] {len(rows)} runs loaded ({n_tr} trustworthy, {len(rows) - n_tr} '
          f'solver-sim gap > 0.05 — drawn hollow, not dropped)')
    fig_reach()
    fig_scatter()
    fig_curves()
    fig_montage()
    print(f'[plot] figures -> {RESDIR}')
    print(f'[plot] elements -> {ELDIR}')


if __name__ == '__main__':
    main()
