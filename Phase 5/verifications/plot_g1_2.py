r"""Phase 5 — EXPERIMENT G1.2 visualization.  Loads Phase 5/results/g1_2/{results.csv,
eta_reference.npz} and produces the four required figures (+ each panel as its own high-DPI
element in results/g1_2/elements/):

  1. reachable-nu per topology: for each topology the [min..max] sim nu it reaches (trustworthy),
     its undistorted nu marked; eta-disordered-triangular reference band overlaid.
  2. achieved-vs-target nu scatter (y=x), coloured by topology.
  3. per-topology nu_target -> nu_achieved curves (one line per topology).
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
              'flipped_tri_f8', 'flipped_tri_f16']
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
def _reach_axis(ax):
    rows = load_rows()
    topos = _present_topos(rows)
    lo6, hi6 = eta_band()[2], eta_band()[3]
    loA, hiA = eta_band()[0], eta_band()[1]
    # eta reference bands (vertical spans over the whole plot)
    ax.axvspan(loA, hiA, color=ETA_BAND, alpha=0.35, zorder=0,
               label='η-disorder triangular (all)')
    ax.axvspan(lo6, hi6, color=ETA_BAND, alpha=0.75, zorder=0,
               label='η-disorder triangular (6-6-6 only)')
    for i, t in enumerate(topos):
        g = [r for r in rows if r['topo'] == t]
        nus = [r['nu_achieved_sim'] for r in g]
        lo, hi = min(nus), max(nus)
        ax.plot([lo, hi], [i, i], color=TOPO_COLOR[t], lw=8, solid_capstyle='round',
                alpha=0.9, zorder=2)
        ax.plot([lo, hi], [i, i], '|', color=INK, markersize=9, markeredgewidth=1.4, zorder=3)
        nu0 = g[0]['nu_initial']
        ax.plot(nu0, i, 'o', color='white', markeredgecolor=TOPO_COLOR[t], markersize=9,
                markeredgewidth=2.2, zorder=4)
        ax.annotate(f'{lo:+.2f}', (lo, i), textcoords='offset points', xytext=(-6, 0),
                    ha='right', va='center', fontsize=7.5, color=INK)
        ax.annotate(f'{hi:+.2f}', (hi, i), textcoords='offset points', xytext=(6, 0),
                    ha='left', va='center', fontsize=7.5, color=INK)
    ax.axvline(0, color=MUTED, lw=1, ls=':')
    ax.set_yticks(range(len(topos)))
    ax.set_yticklabels(topos, color=INK, fontsize=9)
    ax.set_ylim(-0.6, len(topos) - 0.4)
    ax.set_xlabel(r'reachable (independent-sim) $\nu$  [min .. max];  '
                  r'$\circ$ = undistorted (k=1)', color=INK)
    ax.set_title('G1.2 — isotropic $\\nu$ reachable by GEOMETRIC DISTORTION alone (k=1 fixed)',
                 color=INK, fontsize=11)
    ax.legend(fontsize=7.5, loc='lower right', framealpha=0.9)
    _chrome(ax)


def fig_reach():
    _save_element(_reach_axis, 'fig1_reachable_by_topology.png', figsize=(8.2, 5.6))
    fig, ax = plt.subplots(figsize=(9.0, 6.0))
    _reach_axis(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(RESDIR, 'fig1_reachable_by_topology.png'), dpi=200)
    plt.close(fig)


# ---- 2. achieved vs target -------------------------------------------------------------------
def _scatter_axis(ax):
    rows = load_rows()
    tv = [r['nu_target'] for r in rows] + [r['nu_achieved_sim'] for r in rows]
    lo, hi = min(tv) - 0.05, max(tv) + 0.05
    ax.plot([lo, hi], [lo, hi], '--', color=MUTED, lw=1.6, zorder=1, label='$y=x$ (ideal)')
    for t in _present_topos(rows):
        g = [r for r in rows if r['topo'] == t]
        ax.scatter([r['nu_target'] for r in g], [r['nu_achieved_sim'] for r in g],
                   s=42, color=TOPO_COLOR[t], marker=TOPO_MARK[t], edgecolor='white',
                   linewidth=0.5, label=t, zorder=3)
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
    rows = load_rows()
    for t in _present_topos(rows):
        g = sorted([r for r in rows if r['topo'] == t], key=lambda r: r['nu_target'])
        ax.plot([r['nu_target'] for r in g], [r['nu_achieved_sim'] for r in g], '-',
                color=TOPO_COLOR[t], marker=TOPO_MARK[t], markersize=5, lw=1.6,
                markeredgecolor='white', markeredgewidth=0.4, label=t)
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
    rows = load_rows()
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
    rows = load_rows()
    print(f'[plot] {len(rows)} trustworthy runs loaded')
    fig_reach()
    fig_scatter()
    fig_curves()
    fig_montage()
    print(f'[plot] figures -> {RESDIR}')
    print(f'[plot] elements -> {ELDIR}')


if __name__ == '__main__':
    main()
