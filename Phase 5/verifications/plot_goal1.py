r"""Phase 5 — GOAL 1 visualization.  Loads Phase 5/results/goal1/results.csv and produces the
four required figures (+ every panel as its own high-DPI element in results/goal1/elements/):

  1. achieved-vs-target nu scatter (y=x), colored by k-band AND by topology class.
  2. coverage / frontier: reachable achieved-nu range vs k-band contrast floor f.
  3. improvement: per-run target-error at initial -> k-only -> k+positions
     (a) all band means in ONE plot; (b) each band's mean+-sigma in its OWN subplot.
  4. breakdown: mean nu-error by topology class and by k-band (with spread).

Palette: dataviz-skill validated categorical hues (fixed order) + a blue sequential ramp for the
ordinal k-bands; scatters carry a per-class marker as secondary (CVD-safe) encoding; scatter regions
are square.  Run:  C:\\Users\\doron\\anaconda3\\python.exe "Phase 5/verifications/plot_goal1.py"
"""
import os, sys, csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RESDIR = os.path.join(REPO, 'Phase 5', 'results', 'goal1')
ELDIR = os.path.join(RESDIR, 'elements')

# ---- palette (validated categorical + blue sequential ramp) ----------------------------------
BAND_ORDER = ['soft', 'large', 'medium', 'small', 'none']
BAND_F = {'soft': 0.0, 'large': 0.1, 'medium': 0.5, 'small': 0.9, 'none': 0.99}
BAND_COLOR = dict(zip(BAND_ORDER, ['#86b6ef', '#5598e7', '#2a78d6', '#1c5cab', '#0d366b']))
CLASS_ORDER = ['bravais', 'tiling', 'foam', 'auxetic', 'flipped']
CLASS_COLOR = dict(zip(CLASS_ORDER, ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']))
CLASS_MARK = dict(zip(CLASS_ORDER, ['o', 's', '^', 'D', 'v']))
STAGES = ['err_initial', 'err_konly', 'err_full']
STAGE_LAB = ['initial\n(uniform k)', 'k-only', 'k + positions']
STAGE_COLOR = ['#898781', '#2a78d6', '#1baf7a']
INK, MUTED, GRID = '#0b0b0b', '#898781', '#e1e0d9'

NUM = {'run_id', 'n_nodes', 'n_bond', 'nu_target', 'band_f', 'nu_sim_initial', 'E_sim_initial',
       'err_initial', 'nu_sim_konly', 'E_sim_konly', 'err_konly', 'nu_sim_full', 'E_sim_full',
       'err_full', 'nu_solver_full', 'kmin_avg', 'solver_sim_gap', 'trustworthy'}


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


def _trust_proxies(ax):
    """Legend entry for the fill convention. FILLED = trustworthy (solver-vs-sim gap < 0.05);
    HOLLOW = the two code paths disagree by more than that. A rejected run is DATA -- it records
    where solver and sim part company, NOT that the network is unreal (audit 2026-08-21)."""
    ax.scatter([], [], s=34, color=INK, marker='o', edgecolor='white', linewidth=0.5,
               label='trustworthy (solver = sim)')
    ax.scatter([], [], s=34, facecolor='none', marker='o', edgecolor=INK, linewidth=1.2,
               label='solver-sim gap > 0.05 (bar = disagreement)')


# ---- shared axis chrome ----------------------------------------------------------------------
def _chrome(ax):
    ax.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_color(INK)


def _save_element(draw, name, figsize=(4.6, 4.6)):
    """Render one panel as its own high-DPI element figure."""
    os.makedirs(ELDIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=figsize)
    draw(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(ELDIR, name), dpi=300)
    plt.close(fig)


# ---- 1. achieved vs target -------------------------------------------------------------------
def _scatter_axis(ax, rows, by):     # `rows` should be ALL runs; rejected ones are drawn hollow
    lo = min(min(r['nu_target'] for r in rows), min(r['nu_sim_full'] for r in rows)) - 0.05
    hi = max(max(r['nu_target'] for r in rows), max(r['nu_sim_full'] for r in rows)) + 0.05
    ax.plot([lo, hi], [lo, hi], '--', color=MUTED, lw=1.6, zorder=1, label='$y=x$ (ideal)')
    groups = ([(bn, [r for r in rows if r['band'] == bn], BAND_COLOR[bn], 'o',
                f'{bn} (f={BAND_F[bn]:g})') for bn in BAND_ORDER] if by == 'band' else
              [(cls, [r for r in rows if r['topo_class'] == cls], CLASS_COLOR[cls],
                CLASS_MARK[cls], cls) for cls in CLASS_ORDER])
    for _key, g, col, mark, lab in groups:
        if not g:
            continue
        for r in g:      # solver-vs-sim disagreement bar (invisible where the two agree)
            ax.plot([r['nu_target']] * 2, sorted((r['nu_sim_full'], r['nu_solver_full'])),
                    '-', color=col, lw=1.1, alpha=0.75, zorder=2)
        # Label whichever branch is drawn first, so a group with NO trustworthy run still appears.
        has_trust = any(r['trustworthy'] >= 0.5 for r in g)
        for trust in (True, False):
            gg = [r for r in g if (r['trustworthy'] >= 0.5) == trust]
            if not gg:
                continue
            ax.scatter([r['nu_target'] for r in gg], [r['nu_sim_full'] for r in gg],
                       s=34 if by == 'band' else 36, marker=mark,
                       color=col if trust else 'none',
                       edgecolor='white' if trust else col,
                       linewidth=0.5 if trust else 1.2,
                       label=lab if trust == has_trust else None, zorder=3)
    _trust_proxies(ax)
    ttl = ('achieved vs target  $\\nu$  —  by k-band' if by == 'band' else
           'achieved vs target  $\\nu$  —  by topology class')
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect('equal', 'box')
    ax.set_xlabel(r'target $\nu$', color=INK); ax.set_ylabel(r'achieved (sim) $\nu$', color=INK)
    ax.set_title(ttl, color=INK, fontsize=11)
    ax.legend(fontsize=7.5, loc='lower right', framealpha=0.95)   # never cover the data
    _chrome(ax)


def fig_achieved(rows):
    _save_element(lambda ax: _scatter_axis(ax, rows, 'band'), 'fig1a_achieved_by_band.png')
    _save_element(lambda ax: _scatter_axis(ax, rows, 'class'), 'fig1b_achieved_by_class.png')
    fig, axs = plt.subplots(1, 2, figsize=(10.4, 5.2))
    _scatter_axis(axs[0], rows, 'band'); _scatter_axis(axs[1], rows, 'class')
    fig.suptitle('Goal 1 — isotropic $\\nu$ target: achieved (independent-sim) vs requested',
                 fontsize=12.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(RESDIR, 'fig1_achieved_vs_target.png'), dpi=200)
    plt.close(fig)


# ---- 2. coverage / frontier ------------------------------------------------------------------
# Bar WIDTH encodes solver-vs-sim uncertainty: strictly proportional to the band's WORST
# |nu_solver - nu_sim|, so a razor-thin bar means the two code paths agree and a fat bar means they
# do not. Proportional (no offset) on purpose -- an offset would make "agrees exactly" look uncertain.
# Median disagreement is 0.0000 in every band, which is why the MAX is the informative statistic.
W_MAX_SLOT = 0.80       # the widest bar occupies this fraction of a band slot (slots are 1.0 apart)
# The scale is ADAPTIVE, not a fixed units-per-dnu: it was hardcoded at 2.4, calibrated when the
# worst band uncertainty was 0.311, and when the repaired position budget pushed that to 0.708 the
# medium bar became 1.7 slots wide and overlapped its neighbours. Proportionality (a bar twice as
# wide means twice the disagreement) is preserved within a figure; the absolute value is printed
# under each bar, so nothing quantitative depends on the scale factor.


def _frontier_axis(ax):
    rows = load_rows(trustworthy_only=False)
    unc_max = max((max(abs(r['nu_sim_full'] - r['nu_solver_full'])
                       for r in rows if r['band'] == bn)
                   for bn in BAND_ORDER if any(r['band'] == bn for r in rows)), default=1.0) or 1.0
    for i, bn in enumerate(BAND_ORDER):
        g = [r for r in rows if r['band'] == bn]
        if not g:
            continue
        unc = max(abs(r['nu_sim_full'] - r['nu_solver_full']) for r in g)
        w = W_MAX_SLOT * unc / unc_max
        # Pale = full reach INCLUDING runs the honesty gate rejected; solid = trustworthy only.
        # Showing only the solid range is what made G1.2 read as "never reaches negative nu".
        nus_all = [r['nu_sim_full'] for r in g]
        lo, hi = min(nus_all), max(nus_all)
        ax.bar(i, hi - lo, bottom=lo, width=w, color=BAND_COLOR[bn], alpha=0.28, zorder=1)
        gt = [r for r in g if r['trustworthy'] >= 0.5]
        if gt:
            nus = [r['nu_sim_full'] for r in gt]
            ax.bar(i, max(nus) - min(nus), bottom=min(nus), width=w,
                   color=BAND_COLOR[bn], alpha=0.9, zorder=2)
            ax.plot(i, float(np.median(nus)), 'o', color='white',
                    markeredgecolor=BAND_COLOR[bn], markersize=8, markeredgewidth=2, zorder=3)
        # the reach is a line even where the bar is too thin to read as an area
        ax.plot([i, i], [lo, hi], color=BAND_COLOR[bn], lw=1.2, alpha=0.9, zorder=2)
        ax.annotate(f'{lo:+.2f}', (i, lo), textcoords='offset points', xytext=(0, -12),
                    ha='center', fontsize=8, color=INK)
        ax.annotate(f'{hi:+.2f}', (i, hi), textcoords='offset points', xytext=(0, 8),
                    ha='center', fontsize=8, color=INK)
        ax.annotate(f'$\\pm${unc:.3f}', (i, lo), textcoords='offset points', xytext=(0, -24),
                    ha='center', fontsize=7.5, color=MUTED)
    ax.legend(handles=[
        Patch(facecolor=INK, alpha=0.9, label='reached, solver = sim'),
        Patch(facecolor=INK, alpha=0.28, label='reached, solver-sim gap > 0.05'),
        Patch(facecolor='none', edgecolor=MUTED,
              label='bar WIDTH proportional to the value printed below each bar'),
    ], fontsize=7.5, loc='lower right', framealpha=0.95)
    # Headroom under the lowest bar so the min / uncertainty annotations clear the ticks.
    ylo = min(r['nu_sim_full'] for r in rows)
    yhi = max(r['nu_sim_full'] for r in rows)
    ax.set_ylim(ylo - 0.16 * (yhi - ylo), yhi + 0.10 * (yhi - ylo))
    # (the reach itself is also drawn as a thin line, so a near-zero-width bar stays readable)

    ax.axhline(0, color=MUTED, lw=1, ls=':')
    ax.set_xticks(range(len(BAND_ORDER)))
    ax.set_xticklabels([f'{bn}\nf={BAND_F[bn]:g}' for bn in BAND_ORDER], color=INK)
    ax.set_xlabel(r'k-contrast band  (floor $f=\min k/\mathrm{avg}\,k$)', color=INK)
    ax.set_ylabel(r'reachable (sim) $\nu$  [min .. max];  width = solver-sim uncertainty',
                  color=INK)
    ax.set_title(r'Coverage / frontier: contrast limits achievable $\nu$', color=INK, fontsize=11)
    ax.set_xlim(-0.5, len(BAND_ORDER) - 0.5)
    _chrome(ax)


def fig_frontier():
    _save_element(_frontier_axis, 'fig2_frontier.png', figsize=(6.0, 4.8))
    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    _frontier_axis(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(RESDIR, 'fig2_frontier.png'), dpi=200)
    plt.close(fig)


# ---- 3. improvement --------------------------------------------------------------------------
def _improve_means_axis(ax):
    rows = load_rows()      # AGGREGATE error stats stay on trustworthy runs (GOAL1.md quotes them)
    x = np.arange(3)
    for bn in BAND_ORDER:
        g = [r for r in rows if r['band'] == bn]
        if not g:
            continue
        means = [np.mean([r[s] for r in g]) for s in STAGES]
        ax.plot(x, means, '-o', color=BAND_COLOR[bn], lw=2, markersize=7,
                markeredgecolor='white', label=f'{bn} (f={BAND_F[bn]:g})')
    ax.set_xticks(x); ax.set_xticklabels(STAGE_LAB, color=INK)
    ax.set_ylabel(r'mean target-error  $|\nu_{\rm sim}-\nu^*|$', color=INK)
    ax.set_title('Improvement across design stages (mean per band)', color=INK, fontsize=11)
    ax.legend(fontsize=8, title='k-band', title_fontsize=8)
    _chrome(ax)


def _improve_spread_axis(ax, bn):
    rows = [r for r in load_rows() if r['band'] == bn]   # trustworthy, as in _improve_means_axis
    x = np.arange(3)
    means = np.array([np.mean([r[s] for r in rows]) for s in STAGES])
    sig = np.array([np.std([r[s] for r in rows]) for s in STAGES])
    ax.fill_between(x, means - sig, means + sig, color=BAND_COLOR[bn], alpha=0.22, zorder=1)
    ax.plot(x, means, '-o', color=BAND_COLOR[bn], lw=2, markersize=7,
            markeredgecolor='white', zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(['init', 'k', 'k+pos'], color=INK, fontsize=8)
    ax.set_title(f'{bn} (f={BAND_F[bn]:g}, n={len(rows)})', color=INK, fontsize=10)
    _chrome(ax)


def fig_improvement():
    _save_element(_improve_means_axis, 'fig3a_improvement_means.png', figsize=(6.2, 4.8))
    for bn in BAND_ORDER:
        _save_element(lambda ax, b=bn: _improve_spread_axis(ax, b),
                      f'fig3b_spread_{bn}.png', figsize=(3.4, 3.2))
    # composed
    fig = plt.figure(figsize=(13.5, 5.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 2.2], wspace=0.28)
    _improve_means_axis(fig.add_subplot(gs[0, 0]))
    inner = gs[0, 1].subgridspec(1, len(BAND_ORDER), wspace=0.35)
    ymax = 0
    axes = []
    for i, bn in enumerate(BAND_ORDER):
        ax = fig.add_subplot(inner[0, i]); _improve_spread_axis(ax, bn); axes.append(ax)
        ymax = max(ymax, ax.get_ylim()[1])
    for i, ax in enumerate(axes):
        ax.set_ylim(0, ymax)
        if i == 0:
            ax.set_ylabel(r'mean$\pm\sigma$ error', color=INK)
    fig.suptitle('Goal 1 — does optimisation reduce the target error? '
                 '(means together, per-band spread separate)', fontsize=12.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(RESDIR, 'fig3_improvement.png'), dpi=200)
    plt.close(fig)


# ---- 4. breakdown ----------------------------------------------------------------------------
def _bar_axis(ax, keys, key_of, colors, labels, title):
    rows = load_rows()      # AGGREGATE error stats stay on trustworthy runs (GOAL1.md quotes them)
    xs, means, sigs, cols = [], [], [], []
    for i, k in enumerate(keys):
        g = [r['err_full'] for r in rows if key_of(r) == k]
        if not g:
            continue
        xs.append(len(xs)); means.append(np.mean(g)); sigs.append(np.std(g))
        cols.append(colors[k])
    ax.bar(xs, means, color=cols, width=0.66, zorder=2, edgecolor='white', linewidth=0.8)
    ax.errorbar(xs, means, yerr=sigs, fmt='none', ecolor=INK, elinewidth=1.3, capsize=4, zorder=3)
    present = [k for k in keys if any(key_of(r) == k for r in rows)]
    ax.set_xticks(range(len(present)))
    ax.set_xticklabels([labels[k] for k in present], color=INK, fontsize=9)
    ax.set_ylabel(r'mean target-error $|\nu_{\rm sim}-\nu^*|$ (k+pos)', color=INK)
    ax.set_title(title, color=INK, fontsize=11)
    _chrome(ax)


def fig_breakdown():
    class_lab = {c: c for c in CLASS_ORDER}
    band_lab = {b: f'{b}\nf={BAND_F[b]:g}' for b in BAND_ORDER}
    _save_element(lambda ax: _bar_axis(ax, CLASS_ORDER, lambda r: r['topo_class'], CLASS_COLOR,
                                       class_lab, 'nu-error by topology class'),
                  'fig4a_by_class.png', figsize=(5.2, 4.4))
    _save_element(lambda ax: _bar_axis(ax, BAND_ORDER, lambda r: r['band'], BAND_COLOR,
                                       band_lab, 'nu-error by k-band'),
                  'fig4b_by_band.png', figsize=(5.2, 4.4))
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.8))
    _bar_axis(axs[0], CLASS_ORDER, lambda r: r['topo_class'], CLASS_COLOR, class_lab,
              'mean nu-error by topology class')
    _bar_axis(axs[1], BAND_ORDER, lambda r: r['band'], BAND_COLOR, band_lab,
              'mean nu-error by k-band')
    fig.suptitle('Goal 1 — target-error breakdown (bar = mean, whisker = $\\pm\\sigma$)',
                 fontsize=12.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(RESDIR, 'fig4_breakdown.png'), dpi=200)
    plt.close(fig)


def main():
    rows = load_rows(trustworthy_only=False)     # scatter shows every run; rejected ones hollow
    n_tr = sum(1 for r in rows if r['trustworthy'] >= 0.5)
    print(f"[plot] {len(rows)} runs loaded ({n_tr} trustworthy, {len(rows) - n_tr} "
          f"solver-sim gap > 0.05 - drawn hollow, not dropped)")
    fig_achieved(rows)
    fig_frontier()
    fig_improvement()
    fig_breakdown()
    print(f"[plot] figures -> {RESDIR}")
    print(f"[plot] elements -> {ELDIR}")


if __name__ == '__main__':
    main()
