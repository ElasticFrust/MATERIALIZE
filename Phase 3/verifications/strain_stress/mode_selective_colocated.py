"""
Case strain_stress / mode_selective_colocated — the SAME spot shows a DIFFERENT stress SHAPE
depending on the load direction (stretch +x vs compress −y), at ~16k triangles, on ordered (regular)
and disordered (disorder_hi) lattices. Two shape-sets, each a single co-located design:

  cross_bars    : a full-width HORIZONTAL bar (lights under +x) crossing a full-height VERTICAL bar
                  (lights under −y) -- a "+" whose arms are direction-selective. These render CLEANLY
                  because a bar aligned with the load IS the natural stress load path (see below).
  disc_triangle : an inner filled DISC (lights under +x) nested in a TRIANGULAR frame (lights under
                  −y). Striking direction-selectivity, but the disc keeps a horizontal feeder streak.

Why the two shape-sets differ in fidelity (an honest physical point this demo makes): stress under a
uniform load is NOT free-form -- equilibrium (div σ = 0) forces the load flux to be conserved across
every cut, so high stress forms a connected PATH spanning the cell in the load direction. A bar
aligned with the load is exactly such a path and renders crisply; an isolated blob (a disc) cannot
carry the flux alone -- the horizontal strip at its height must carry the same x-flux, so a feeder
streak necessarily lights up (Eshelby gives a uniform disc INTERIOR, but the flux still has to enter
and leave). So cross_bars is the clean-rendering, physics-aligned design; disc_triangle is the
recognizable-shapes design with the feeder as an honest, non-removable artifact.

Same "full response operator" point as mode_selective.py: one W, queried at two loads in a single
optimize() call (shape-A loud & shape-B quiet under +x; shape-B loud & shape-A quiet under −y; a
suppressed background moat under both). Verified two independent ways (validate() readback +
_common.unit_mode_response's separate NumPy PBC sim; +x and −y stresses are the mode-0 and negated
mode-1 responses). Outputs: mode_selective_colocated.csv, mode_selective_colocated_<kind>_<topo>.npz,
strain_stress_mode_selective_colocated_<kind>_<topo>.png.
"""
import os, sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _common as C

CASE, REG, HALF, NITER = 'strain_stress', 1e-4, 42, 130           # HALF=42 -> ~16k triangles
TOPOS = ['regular', 'disorder_hi']
KINDS = ['cross_bars', 'disc_triangle']
S, SBG = 0.5, 0.12                                               # active-shape target; suppressed background
I2 = np.eye(2)
LOAD_X = C.MO.vec3(C.PH.Fk[0].T @ C.PH.Fk[0] - I2) / C.PH.DELTA   # STRETCH +x
LOAD_Y = -C.MO.vec3(C.PH.Fk[1].T @ C.PH.Fk[1] - I2) / C.PH.DELTA  # COMPRESS -y


def make_shapes(kind, prob, geo):
    """Return (A_spec, A_label, A_idx, B_spec, B_label, B_idx) for a shape-set. A lights under +x,
    B under −y; both co-located at the cell centre. Any overlap is excluded from BOTH (it sits on
    both load paths and simply lights under both -- e.g. the centre of the '+')."""
    Lx, Ly = C.box(geo); cx, cy = 0.5 * Lx, 0.5 * Ly
    if kind == 'cross_bars':
        A_spec = {'kind': 'rect', 'center': (cx, cy), 'w': Lx, 'h': 0.11 * Ly, 'color': 'cyan'}
        B_spec = {'kind': 'rect', 'center': (cx, cy), 'w': 0.11 * Lx, 'h': Ly, 'color': 'lime'}
        A_full, _ = C.region_shape(prob, A_spec); B_full, _ = C.region_shape(prob, B_spec)
        overlap = np.intersect1d(A_full, B_full)
        return (A_spec, 'horizontal bar', np.setdiff1d(A_full, overlap),
                B_spec, 'vertical bar', np.setdiff1d(B_full, overlap))
    # disc_triangle: filled disc (A) nested in a triangular frame (B = triangle minus a keep-out disc)
    A_spec = {'kind': 'circle', 'center': (cx, cy), 'radius': 0.10 * Lx, 'color': 'cyan'}
    B_spec = {'kind': 'polygon', 'verts': C.triangle_verts(cx, cy, 0.26 * Lx), 'color': 'lime'}
    gap_spec = {'kind': 'circle', 'center': (cx, cy), 'radius': 0.14 * Lx}
    A_idx, _ = C.region_shape(prob, A_spec)
    tri_full, _ = C.region_shape(prob, B_spec); gap, _ = C.region_shape(prob, gap_spec)
    return A_spec, 'inner disc', A_idx, B_spec, 'tri frame', np.setdiff1d(tri_full, gap)


def mag3(v):
    return np.sqrt(v[:, 0] ** 2 + 2 * v[:, 1] ** 2 + v[:, 2] ** 2)


def independent_stress(geo):
    """+x stress = mode-0 stress, −y stress = -(mode-1 stress), by linearity (separate NumPy sim)."""
    _, sig = C.unit_mode_response(geo)
    return C.MO.vec3(sig[0]), -C.MO.vec3(sig[1])


def _finite_pct(a, q):
    f = np.abs(a)[np.isfinite(a)]
    return float(np.percentile(f, q)) if f.size else 1.0


def plot_colocated(geo, A_spec, B_spec, A_label, B_label, sig_x_full, sig_y_full, bars, kind, topo, path):
    fig = plt.figure(figsize=(18, 6))
    a0 = fig.add_subplot(1, 3, 1); a1 = fig.add_subplot(1, 3, 2); a2 = fig.add_subplot(1, 3, 3)
    mx, my = mag3(sig_x_full), mag3(sig_y_full)
    vmax = max(_finite_pct(mx, 99), _finite_pct(my, 99), S)
    for ax, m, title in ((a0, mx, f'STRETCH +x  →  {A_label} lights up'),
                         (a1, my, f'COMPRESS −y  →  {B_label} lights up')):
        pc = C.fill_local_map(ax, geo, np.nan_to_num(m, nan=0.0, posinf=vmax, neginf=0.0), cmap='magma')
        pc.set_clim(0, vmax)
        C.draw_box(ax, geo); C.mark_region(ax, A_spec); C.mark_region(ax, B_spec)
        plt.colorbar(pc, ax=ax, fraction=0.046)
        ax.set_title(f'local ‖σ‖   ({title})', fontsize=11)

    x = np.arange(2); w = 0.35
    a2.bar(x - w / 2, [bars['A_x'], bars['B_x']], w, label='under +x stretch', color='#1f77b4')
    a2.bar(x + w / 2, [bars['A_y'], bars['B_y']], w, label='under −y compress', color='#d62728')
    a2.set_xticks(x); a2.set_xticklabels([A_label, B_label])
    a2.set_ylabel('mean ‖σ‖ in shape (INDEPENDENT sim)'); a2.legend(fontsize=9); a2.grid(alpha=.3, axis='y')
    a2.set_title('same spot — each shape responds to ITS OWN load', fontsize=11)
    fig.suptitle(f'{CASE} — CO-LOCATED mode-selective stress @ ~16k tri ({kind}, {topo}): one spot, '
                 f'DIFFERENT shape by direction ({A_label}↔+x, {B_label}↔−y)', fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f'saved {os.path.basename(path)}')


def run(kind, topo, csv_rows):
    prob, geo = C.make_case(topo, HALF)
    A_spec, A_label, A, B_spec, B_label, B = make_shapes(kind, prob, geo)
    bg = np.setdiff1d(np.arange(prob.n_tri), np.concatenate([A, B]))   # everything else, incl. the moat

    # ONE optimize(), stress objectives spanning TWO loads at ONE spot: A loud/B quiet under +x, B
    # loud/A quiet under −y, background suppressed under both (dark moat -> shapes read, see docstring).
    objs = [C.Objective('stress', target=torch.tensor([S, 0.0, 0.0]), region=A, load=LOAD_X, weight=2.0),
            C.Objective('stress', target=torch.tensor([0.0, 0.0, 0.0]), region=B, load=LOAD_X, weight=2.0),
            C.Objective('stress', target=torch.tensor([SBG, 0.0, 0.0]), region=bg, load=LOAD_X, weight=0.6),
            C.Objective('stress', target=torch.tensor([0.0, 0.0, -S]), region=B, load=LOAD_Y, weight=2.0),
            C.Objective('stress', target=torch.tensor([0.0, 0.0, 0.0]), region=A, load=LOAD_Y, weight=2.0),
            C.Objective('stress', target=torch.tensor([0.0, 0.0, -SBG]), region=bg, load=LOAD_Y, weight=0.6)]
    r = C.optimize(prob, objs, mode='k', n_iter=NITER, reg=REG, verbose=False)
    C.apply_k_to_geo(geo, r['k'])

    sig_x_full, sig_y_full = independent_stress(geo)
    mx, my = mag3(sig_x_full), mag3(sig_y_full)

    def rmean(m, idx):
        v = m[idx]; v = v[np.isfinite(v)]
        return float(v.mean()) if v.size else float('nan')
    bars = {'A_x': rmean(mx, A), 'B_x': rmean(mx, B), 'A_y': rmean(my, A), 'B_y': rmean(my, B)}
    A_sel = bars['A_x'] / max(bars['A_y'], 1e-9); B_sel = bars['B_y'] / max(bars['B_x'], 1e-9)
    print(f"  [colocated {kind} {topo}] tri={prob.n_tri}", flush=True)
    print(f"    {A_label:14s} ‖σ‖: +x={bars['A_x']:.3f}  −y={bars['A_y']:.3f}  (x/y {A_sel:.2f}×)  |  "
          f"{B_label:14s} ‖σ‖: +x={bars['B_x']:.3f}  −y={bars['B_y']:.3f}  (y/x {B_sel:.2f}×)", flush=True)
    csv_rows.append((kind, topo, A_label, f'{bars["A_x"]:.4f}', f'{bars["A_y"]:.4f}', f'{A_sel:.3f}'))
    csv_rows.append((kind, topo, B_label, f'{bars["B_x"]:.4f}', f'{bars["B_y"]:.4f}', f'{B_sel:.3f}'))

    C6 = C.sim_per_triangle_C6(geo)
    C.save_network(os.path.join(C.savedir(CASE), f'mode_selective_colocated_{kind}_{topo}.npz'), geo,
                   r['k'], C6, A=A_spec, B=B_spec, kind=kind, S=float(S), n_tri=int(prob.n_tri))
    plot_colocated(geo, A_spec, B_spec, A_label, B_label, sig_x_full, sig_y_full, bars, kind, topo,
                   os.path.join(C.savedir(CASE), f'strain_stress_mode_selective_colocated_{kind}_{topo}.png'))


def main():
    csv_rows = []
    for kind in KINDS:
        for topo in TOPOS:
            run(kind, topo, csv_rows)
    C.write_csv(os.path.join(C.savedir(CASE), 'mode_selective_colocated.csv'),
                ['kind', 'topo', 'shape', 'mean_stressmag_under_+x', 'mean_stressmag_under_-y',
                 'own_load_selectivity'], csv_rows)
    print('done')


if __name__ == '__main__':
    main()
