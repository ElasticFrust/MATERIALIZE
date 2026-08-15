r"""
plotting.py — the SINGLE source of truth for all MATERIALIZE figures.

Provenance: implements the plotting & output conventions settled in CLAUDE.md §3 (Plotting).
**Every plotting task MUST start from this module** — import it and use these primitives; never
roll a one-off `plot_*`/`draw_*`. If a needed primitive is missing, add it *here* (and update the
CLAUDE.md §3 policy), so the conventions stay in one place.

Pure render: depends only on numpy + matplotlib (NO compute/IO — callers pass already-loaded data:
`geo` dicts, per-bond `k`, per-triangle values). This keeps compute (`_common`) separate from
rendering (the charter's compute-vs-render split).

Conventions enforced here (see CLAUDE.md §3 for the why):
- square plot regions; periodic networks drawn TILED + CROPPED (bonds crossing the boundary render
  continuously — no wrap gaps);
- bond colour = k (viridis, sequential, k≥0), CONSTANT medium width, dashed only very near k=0;
- per-triangle field maps FILL each triangle: ν → diverging cmap centred at 0, E → sequential;
- directional response: cartesian (MAIN) + polar (|ν| radius, blue ν>0 / red ν<0; E direct);
- means-together / spread-separate (combine-all only for ≤3 groups whose overlap is the point);
- standalone reusable elements ≥300 DPI, montages 200; each element also saved as its own image;
- figures LOAD saved data, never re-optimise (that is the caller's contract).

`geo` keys used: `pts (n,2)`, `bond_u`, `bond_R`, `BL1`, `BL2` (cell vectors; `BL1[0],BL2[1]=Lx,Ly`),
`tri_verts (nt,3,2)` (image-correct triangle vertices, for fills).
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection, PolyCollection


# ---- STYLE: single source of truth for style values -----------------------------------------
class STYLE:
    K_CMAP      = 'viridis'      # bond stiffness k (sequential, k≥0)
    NU_CMAP     = 'coolwarm'     # Poisson field (diverging, centred at 0)
    E_CMAP      = 'viridis'      # Young field (sequential, E>0)
    LW          = 1.5            # constant medium bond line width (k NOT encoded by width)
    K0_FRAC     = 0.02           # k < K0_FRAC·median(k) ⇒ "very close to 0" ⇒ dashed (else solid)
    DPI_ELEMENT = 300            # standalone reusable elements (≥300; raise for publication)
    DPI_MONTAGE = 200            # montage working DPI (raise for production)
    TILE_REPS   = 3              # reps×reps tiling for the periodic-continuous network draw
    PAD         = 0.06           # crop padding, fraction of the cell
    POLAR_POS   = 'tab:blue'     # ν > 0 in the polar plot
    POLAR_NEG   = 'tab:red'      # ν < 0 in the polar plot
    TARGET_KW   = dict(color='k', ls='--', lw=2.2)   # dashed target overlay
    PANEL       = (4.4, 3.5)     # inches per grid panel — big enough that a legend never covers
                                 # the data; if the legend crowds the curves, the panel is too small


def square(ax):
    """Square axes frame, no ticks (project convention for spatial/network panels)."""
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])


def _k_norm(k):
    k = np.asarray(k, float)
    kmin, kmax = float(np.nanmin(k)), float(np.nanmax(k))
    if not np.isfinite(kmin) or kmax <= kmin:
        kmax = kmin + 1e-9
    return mcolors.Normalize(kmin, kmax)


# ---- 1. canonical network draw (tiled-continuous, cropped) ----------------------------------
def draw_network(ax, geo, bond_k, cmap=STYLE.K_CMAP, norm=None, reps=STYLE.TILE_REPS,
                 pad=STYLE.PAD, title=None):
    """THE canonical network draw. Tile the periodic cell reps×reps and crop to the central cell so
    bonds crossing the boundary render CONTINUOUSLY (no wrap gaps). Colour by k (viridis), CONSTANT
    medium width; bonds very close to k=0 drawn dashed (else solid). Square. Returns the (solid-bond)
    LineCollection so a colorbar can be attached. Supersedes the stub draws (fractional-wrap /
    real-coord-stub) that left boundary gaps."""
    Lx, Ly = float(geo['BL1'][0]), float(geo['BL2'][1])
    pts = np.asarray(geo['pts'], float)
    u = pts[geo['bond_u']]
    v = u + np.asarray(geo['bond_R'], float)
    k = np.asarray(bond_k, float)
    norm = norm or _k_norm(k)
    kmed = np.nanmedian(k)
    kmed = kmed if (np.isfinite(kmed) and kmed > 0) else 1.0
    solid = k >= STYLE.K0_FRAC * kmed                        # dashed only VERY near k=0

    r = reps // 2
    seg_s, k_s, seg_d, k_d = [], [], [], []
    for dx in range(-r, r + 1):
        for dy in range(-r, r + 1):
            off = np.array([dx * Lx, dy * Ly])
            s = np.stack([u + off, v + off], axis=1)         # (nbond,2,2)
            seg_s.append(s[solid]); k_s.append(k[solid])
            seg_d.append(s[~solid]); k_d.append(k[~solid])
    lc = LineCollection(np.concatenate(seg_s), array=np.concatenate(k_s), cmap=cmap, norm=norm,
                        linewidths=STYLE.LW, zorder=2)
    ax.add_collection(lc)
    if any(len(x) for x in seg_d):
        segd = np.concatenate(seg_d)
        if len(segd):
            ax.add_collection(LineCollection(segd, array=np.concatenate(k_d), cmap=cmap, norm=norm,
                                             linewidths=STYLE.LW, linestyles='dashed', zorder=1))
    ax.set_xlim(-pad * Lx, (1 + pad) * Lx)
    ax.set_ylim(-pad * Ly, (1 + pad) * Ly)
    square(ax)
    if title:
        ax.set_title(title, fontsize=8)
    return lc


# ---- 2. per-triangle scalar field (filled triangles) ----------------------------------------
def draw_field(ax, geo, values, kind='nu', cmap=None, vlim=None, colorbar=True, title=None,
               cbar_label=None):
    """Per-triangle scalar field as FILLED triangles (image-correct `tri_verts`, no tearing).
    kind='nu' → diverging cmap centred at 0 (auxetic ν<0 vs normal ν>0); kind='E' → sequential.
    Returns the PolyCollection. Pass `vlim=(lo,hi)` to fix the scale across panels, and
    `cbar_label` for any field that is neither ν nor E (kind only picks the colour scaling, so
    without it such a field is mislabelled on the colourbar)."""
    tv = list(np.asarray(geo['tri_verts'], float))          # (nt,3,2) image-correct
    vals = np.asarray(values, float)
    if cmap is None:
        cmap = STYLE.NU_CMAP if kind == 'nu' else STYLE.E_CMAP
    if kind == 'nu':
        m = vlim[1] if vlim else float(np.nanmax(np.abs(vals)))
        m = m if (np.isfinite(m) and m > 0) else 1.0
        norm = mcolors.TwoSlopeNorm(vcenter=0.0, vmin=-m, vmax=m)   # diverging, centred at 0
    else:
        lo, hi = vlim if vlim else (float(np.nanmin(vals)), float(np.nanmax(vals)))
        if hi <= lo:
            hi = lo + 1e-9
        norm = mcolors.Normalize(lo, hi)
    pc = PolyCollection(tv, array=vals, cmap=cmap, norm=norm, edgecolors='none')
    ax.add_collection(pc)
    Lx, Ly = float(geo['BL1'][0]), float(geo['BL2'][1])
    ax.set_xlim(0, Lx); ax.set_ylim(0, Ly)
    square(ax)
    if colorbar:
        ax.figure.colorbar(pc, ax=ax, fraction=0.046, pad=0.02,
                           label=(cbar_label if cbar_label is not None
                                  else (r'$\nu$' if kind == 'nu' else r'$E$')))
    if title:
        ax.set_title(title, fontsize=8)
    return pc


# ---- 3. directional response ν(θ), E(θ): cartesian (MAIN) + polar ---------------------------
_THETA_TICKS = ([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi],
                ['0', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$'])


def _cartesian_pair(ax_nu, ax_E, thetas, curves, target_nu, target_E, labels, cmap='viridis'):
    cm = plt.get_cmap(cmap)
    n = len(curves)
    for i, (nu, E) in enumerate(curves):
        col = cm(i / max(n - 1, 1)) if n > 1 else cm(0.5)
        lab = labels[i] if labels else (f'#{i}' if n > 1 else None)
        ax_nu.plot(thetas, nu, color=col, lw=1.5, label=lab)
        ax_E.plot(thetas, E, color=col, lw=1.5)
    for ax, tgt in ((ax_nu, target_nu), (ax_E, target_E)):
        if tgt is not None:
            ax.plot(thetas, np.broadcast_to(np.asarray(tgt, float), thetas.shape),
                    label='target', **STYLE.TARGET_KW)
    for ax, ylab in ((ax_nu, r'$\nu(\theta)$'), (ax_E, r'$E(\theta)$')):
        ax.set_xlabel(r'$\theta$'); ax.set_ylabel(ylab)
        ax.set_xlim(thetas[0], thetas[-1])
        ax.set_xticks(_THETA_TICKS[0]); ax.set_xticklabels(_THETA_TICKS[1])
        ax.grid(alpha=0.25)


def _polar_nu(ax, thetas, curves):
    """Polar ν: radius = |ν(θ)|, coloured blue where ν≥0, red where ν<0 (handles negative ν in polar).
    θ mirrored to [0,2π] (elasticity is π-periodic)."""
    th = np.concatenate([thetas, thetas + np.pi])
    for nu, _E in curves:
        nu2 = np.concatenate([nu, nu])
        r = np.abs(nu2)
        r_pos = np.where(nu2 >= 0, r, np.nan)
        r_neg = np.where(nu2 < 0, r, np.nan)
        ax.plot(th, r_pos, color=STYLE.POLAR_POS, lw=1.5)
        ax.plot(th, r_neg, color=STYLE.POLAR_NEG, lw=1.5)
    ax.set_title(r'$|\nu(\theta)|$  (blue $\nu{>}0$, red $\nu{<}0$)', fontsize=9)


def _polar_E(ax, thetas, curves, cmap='viridis'):
    cm = plt.get_cmap(cmap); n = len(curves)
    th = np.concatenate([thetas, thetas + np.pi])
    for i, (_nu, E) in enumerate(curves):
        col = cm(i / max(n - 1, 1)) if n > 1 else cm(0.5)
        ax.plot(th, np.concatenate([E, E]), color=col, lw=1.5)
    ax.set_title(r'$E(\theta)$', fontsize=9)


def plot_directional(thetas, curves, target_nu=None, target_E=None, polar=True,
                     labels=None, suptitle=None):
    """Directional response. `curves` = list of (ν(θ), E(θ)) arrays (one per design) — a single
    (ν,E) tuple is accepted too. Cartesian ν|E is the MAIN plot (target dashed); if `polar`, add a
    polar ν (|ν| radius, blue ν>0 / red ν<0) and a polar E. Returns the Figure (caller saves via
    `save_fig`/`save_element`)."""
    thetas = np.asarray(thetas, float)
    if isinstance(curves, tuple):
        curves = [curves]
    curves = [(np.asarray(nu, float), np.asarray(E, float)) for nu, E in curves]

    if polar:
        fig = plt.figure(figsize=(11.5, 8.4))
        ax_nu = fig.add_subplot(2, 2, 1); ax_E = fig.add_subplot(2, 2, 2)
        axp_nu = fig.add_subplot(2, 2, 3, projection='polar')
        axp_E = fig.add_subplot(2, 2, 4, projection='polar')
        _cartesian_pair(ax_nu, ax_E, thetas, curves, target_nu, target_E, labels)
        _polar_nu(axp_nu, thetas, curves)
        _polar_E(axp_E, thetas, curves)
    else:
        fig, (ax_nu, ax_E) = plt.subplots(1, 2, figsize=(11.5, 4.8))
        _cartesian_pair(ax_nu, ax_E, thetas, curves, target_nu, target_E, labels)
    if len(curves) > 1 or labels:
        ax_nu.legend(fontsize=7, ncol=2, loc='best')
    if suptitle:
        fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout()
    return fig


# ---- 4. means-together / spread-separate ----------------------------------------------------
def plot_means_spread(groups, xlabel='x', ylabel='y', suptitle=None, combine_all=False):
    """Group comparison, project convention. `groups` = dict name → (x, ys) with ys shape
    (nsamples, nx). DEFAULT: one clean panel of all group MEANS (no bands) + one subplot per group
    showing that group's mean±σ. `combine_all=True` (ALLOWED ONLY for ≤3 groups whose *overlap* is
    the point): a single panel with means + σ bands overlaid. Returns the Figure."""
    names = list(groups)
    cm = plt.get_cmap('tab10')
    col = {nm: cm(i % 10) for i, nm in enumerate(names)}

    if combine_all:
        if len(names) > 3:
            raise ValueError('combine_all is only for ≤3 groups (whose overlap is the point); '
                             'use the default means-together / spread-separate instead.')
        fig, ax = plt.subplots(figsize=(6.4, 4.6))
        for nm in names:
            x, ys = groups[nm]; ys = np.asarray(ys, float)
            mu, sd = ys.mean(0), ys.std(0)
            ax.plot(x, mu, color=col[nm], lw=1.8, label=nm)
            ax.fill_between(x, mu - sd, mu + sd, color=col[nm], alpha=0.2)
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.grid(alpha=0.25); ax.legend(fontsize=8)
        if suptitle:
            fig.suptitle(suptitle, fontsize=12)
        fig.tight_layout()
        return fig

    # default: means together (no bands) + one spread subplot per group
    ncol = len(names) + 1
    fig, axes = plt.subplots(1, ncol, figsize=(3.3 * ncol, 3.6), squeeze=False)
    axm = axes[0][0]
    for nm in names:
        x, ys = groups[nm]; ys = np.asarray(ys, float)
        axm.plot(x, ys.mean(0), color=col[nm], lw=1.8, label=nm)
    axm.set_title('group means'); axm.set_xlabel(xlabel); axm.set_ylabel(ylabel)
    axm.grid(alpha=0.25); axm.legend(fontsize=8)
    for j, nm in enumerate(names):
        ax = axes[0][j + 1]
        x, ys = groups[nm]; ys = np.asarray(ys, float)
        mu, sd = ys.mean(0), ys.std(0)
        ax.plot(x, mu, color=col[nm], lw=1.8)
        ax.fill_between(x, mu - sd, mu + sd, color=col[nm], alpha=0.25)
        ax.set_title(f'{nm}  (mean±σ)'); ax.set_xlabel(xlabel); ax.grid(alpha=0.25)
    if suptitle:
        fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout()
    return fig


# ---- 5. saving: standalone high-DPI element, and a montage ----------------------------------
def plot_overlay_grid(panels, xlabel='x', ylabel='y', suptitle=None, ncols=3,
                      panel=STYLE.PANEL, ylim=None, hline0=False):
    """One panel per case; the METHODS COMPARED ARE OVERLAID INSIDE each panel (mean ± σ).

    Use this whenever two ways of computing the same quantity are being compared — above all
    **solver vs independent sim**, which by project convention (CLAUDE.md §3) always share a panel:
    the comparison IS the point, and splitting them into separate subplots puts the two curves the
    reader must overlay at opposite ends of the figure.

    Args:
        panels: dict case name → dict method name → (x, ys), `ys` shape (nsamples, nx).
                A method may pass ys of shape (nx,) for a single sample.
        ncols:  panels per row; the grid wraps (never one long strip).
        panel:  (w, h) inches per panel — big enough that the legend does not cover the data.
        ylim:   shared y-limits, or None to autoscale per panel.
        hline0: draw a dotted zero line (use for ν, where the sign is the physics).
    Returns the Figure.
    """
    names = list(panels)
    ncols = max(1, min(ncols, len(names)))
    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(panel[0]*ncols, panel[1]*nrows),
                             squeeze=False, sharex=True, sharey=ylim is not None)
    cm = plt.get_cmap('tab10')
    methods = list({m for d in panels.values() for m in d})
    col = {m: cm(i % 10) for i, m in enumerate(methods)}
    mk = {m: ['o', 's', '^', 'v', 'D'][i % 5] for i, m in enumerate(methods)}

    for i, nm in enumerate(names):
        ax = axes[i // ncols][i % ncols]
        for m, (x, ys) in panels[nm].items():
            ys = np.atleast_2d(np.asarray(ys, float))
            mu, sd = np.nanmean(ys, 0), np.nanstd(ys, 0)
            ax.plot(x, mu, color=col[m], lw=1.9, marker=mk[m], ms=3.6, label=m)
            if ys.shape[0] > 1:
                ax.fill_between(x, mu - sd, mu + sd, color=col[m], alpha=0.20, lw=0)
        if hline0:
            ax.axhline(0, color='gray', lw=0.6, ls=':')
        if ylim:
            ax.set_ylim(*ylim)
        ax.set_title(nm, fontsize=10); ax.grid(alpha=0.25)
        if i % ncols == 0:
            ax.set_ylabel(ylabel)
        if i // ncols == nrows - 1:
            ax.set_xlabel(xlabel)
    for j in range(len(names), nrows * ncols):          # blank any unused cell
        axes[j // ncols][j % ncols].axis('off')

    # ONE figure-level legend: repeating it per panel is what crowds the data out.
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=len(methods), fontsize=10, frameon=False,
               bbox_to_anchor=(0.5, -0.01))
    if suptitle:
        fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    return fig


def save_fig(fig, path, dpi=STYLE.DPI_ELEMENT, close=True):
    """Save `fig` to `path` at `dpi` (default the ≥300 element DPI); make parent dirs."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    if close:
        plt.close(fig)
    return path


def save_element(draw, path, figsize=(4.6, 4.6), dpi=STYLE.DPI_ELEMENT):
    """Render ONE panel as its own standalone high-DPI element image. `draw(ax)` does the drawing.
    (Project convention: render each figure element as its own high-DPI image so it is reusable
    without re-running.) Returns `path`."""
    fig, ax = plt.subplots(figsize=figsize)
    draw(ax)
    fig.tight_layout()
    return save_fig(fig, path, dpi=dpi)


def montage(items, out_path, ncols=4, suptitle=None, cbar_label='k', dpi=STYLE.DPI_MONTAGE):
    """Grid of canonical network panels + each panel ALSO saved standalone (high-DPI) in an
    `elements/` dir next to `out_path`. `items` = list of `(geo, bond_k, title)` (already loaded —
    plotting.py does no IO). Returns the list of element image paths."""
    items = list(items)
    if not items:
        raise ValueError('montage: no items given')
    n = len(items)
    ncols = max(1, min(ncols, n))
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 3.7 * nrows), squeeze=False)
    for idx, (geo, k, title) in enumerate(items):
        ax = axes[idx // ncols][idx % ncols]
        lc = draw_network(ax, geo, k, title=title)
        fig.colorbar(lc, ax=ax, fraction=0.046, pad=0.02, label=cbar_label)
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis('off')
    if suptitle:
        fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save_fig(fig, out_path, dpi=dpi)

    elem_dir = os.path.join(os.path.dirname(os.path.abspath(out_path)), 'elements')
    elem_paths = []
    for idx, (geo, k, title) in enumerate(items):
        ep = os.path.join(elem_dir, f'panel_{idx:02d}.png')
        save_element(lambda ax, g=geo, kk=k, t=title: (draw_network(ax, g, kk, title=t)), ep)
        elem_paths.append(ep)
    return elem_paths
