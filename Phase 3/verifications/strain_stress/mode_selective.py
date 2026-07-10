"""
Case strain_stress / mode_selective — a SINGLE large (~16k-triangle) network whose stress pattern
DEPENDS ON THE LOAD DIRECTION: stretch along x lights up a localized filled DISC of high stress;
compress along y lights up a separate TRIANGLE of high stress; each shape stays quiet under the
other load, and the surrounding background is suppressed so the two shapes are crisply defined. Run
on BOTH an ordered (regular) and a disordered (disorder_hi) topology.

This is the cleanest demonstration of the framework's "full response operator" character (see the
README notes): because the differentiable solver returns the whole strain-concentration operator W
(not one load's answer), a single optimize() call targets the per-region stress under TWO different
applied loads at once -- disc-stress under +x stretch AND triangle-stress under -y compression, plus
a quiet background under both -- from ONE design. A displacement/FEM design loop would need a separate
solve (and adjoint) per load case; here every load is just another contraction of the same W.

Mechanism: directional stiffness contrast -- the disc is stiff along x / soft along y (carries
x-stress, ignores y), the triangle stiff along y / soft along x, the background compliant in both.
Verified two independent ways per the repo convention: validate()'s differentiable readback AND
_common.unit_mode_response's separate NumPy PBC simulation (the +x and -y stresses are the mode-0 and
negated mode-1 responses, by linearity). Outputs: mode_selective.csv, mode_selective_<topo>.npz,
strain_stress_mode_selective_<topo>.png.
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
S, SBG = 0.5, 0.15                                                # active-shape target; suppressed background
I2 = np.eye(2)
# Two DIFFERENT applied macro loads, same vec3=[xx,xy,yy] convention as the other demos:
LOAD_X = C.CE.vec3(C.PH.Fk[0].T @ C.PH.Fk[0] - I2) / C.PH.DELTA   # STRETCH +x  (xx unit mode)
LOAD_Y = -C.CE.vec3(C.PH.Fk[1].T @ C.PH.Fk[1] - I2) / C.PH.DELTA  # COMPRESS -y (negated yy unit mode)


def mag3(v):
    """Voigt stress magnitude of a per-triangle vec3=[xx,xy,yy] field (off-diagonal counted twice)."""
    return np.sqrt(v[:, 0] ** 2 + 2 * v[:, 1] ** 2 + v[:, 2] ** 2)


def independent_stress(geo):
    """Non-autograd ground-truth stress under the two loads, via _common.unit_mode_response (a
    separate NumPy PBC sim). By linearity, +x stress = mode-0 stress, -y stress = -(mode-1 stress).
    Returns (sig_x_vec3, sig_y_vec3), each (nt,3)."""
    _, sig = C.unit_mode_response(geo)
    return C.CE.vec3(sig[0]), -C.CE.vec3(sig[1])


def _finite_pct(a, q):
    f = np.abs(a)[np.isfinite(a)]
    return float(np.percentile(f, q)) if f.size else 1.0


def plot_mode_selective(geo, disc_spec, tri_spec, sig_x_full, sig_y_full, bars, topo, path):
    """Two local stress-magnitude maps (+x stretch, -y compress) + a bar chart of each shape's mean
    stress magnitude under each load. Color scale is explicit and finite-guarded (a near-mechanism
    outlier can't auto-range the map to white)."""
    fig = plt.figure(figsize=(18, 6))
    a0 = fig.add_subplot(1, 3, 1); a1 = fig.add_subplot(1, 3, 2); a2 = fig.add_subplot(1, 3, 3)
    mx, my = mag3(sig_x_full), mag3(sig_y_full)
    vmax = max(_finite_pct(mx, 99), _finite_pct(my, 99), S)          # shared scale, comparable + robust
    for ax, m, title in ((a0, mx, 'STRETCH +x  →  DISC lights up'),
                         (a1, my, 'COMPRESS −y  →  TRIANGLE lights up')):
        pc = C.fill_local_map(ax, geo, np.nan_to_num(m, nan=0.0, posinf=vmax, neginf=0.0), cmap='magma')
        pc.set_clim(0, vmax)
        C.draw_box(ax, geo); C.mark_region(ax, disc_spec); C.mark_region(ax, tri_spec)
        plt.colorbar(pc, ax=ax, fraction=0.046)
        ax.set_title(f'local ‖σ‖   ({title})', fontsize=11)

    x = np.arange(2); w = 0.35
    a2.bar(x - w / 2, [bars['disc_x'], bars['tri_x']], w, label='under +x stretch', color='#1f77b4')
    a2.bar(x + w / 2, [bars['disc_y'], bars['tri_y']], w, label='under −y compress', color='#d62728')
    a2.set_xticks(x); a2.set_xticklabels(['DISC', 'TRIANGLE'])
    a2.set_ylabel('mean ‖σ‖ in shape (INDEPENDENT sim)'); a2.legend(fontsize=9); a2.grid(alpha=.3, axis='y')
    a2.set_title('each shape responds to ITS OWN load', fontsize=11)
    fig.suptitle(f'{CASE} — MODE-SELECTIVE stress @ ~16k tri ({topo}): one network, load-dependent '
                 f'pattern (disc↔+x stretch, triangle↔−y compress)', fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f'saved {os.path.basename(path)}')


def run(topo, csv_rows):
    prob, geo = C.make_case(topo, HALF)
    Lx, Ly = C.box(geo)
    disc_spec = {'kind': 'circle', 'center': (0.3 * Lx, 0.5 * Ly), 'radius': 0.16 * Lx, 'color': 'cyan'}
    tri_spec = {'kind': 'polygon', 'verts': C.triangle_verts(0.72 * Lx, 0.5 * Ly, 0.20 * Lx),
                'color': 'lime'}
    disc, _ = C.region_shape(prob, disc_spec)
    tri, _ = C.region_shape(prob, tri_spec)
    bg = np.setdiff1d(np.arange(prob.n_tri), np.concatenate([disc, tri]))

    # ONE optimize() call, SIX stress objectives spanning TWO loads. Under +x: disc loud, triangle
    # quiet, background suppressed. Under −y: triangle loud, disc quiet, background suppressed. Same W,
    # queried at both loads. Background suppression (not zero -> not floppy) sharpens shape definition.
    objs = [C.Objective('stress', target=torch.tensor([S, 0.0, 0.0]), region=disc, load=LOAD_X, weight=2.0),
            C.Objective('stress', target=torch.tensor([0.0, 0.0, 0.0]), region=tri, load=LOAD_X, weight=2.0),
            C.Objective('stress', target=torch.tensor([SBG, 0.0, 0.0]), region=bg, load=LOAD_X, weight=0.6),
            C.Objective('stress', target=torch.tensor([0.0, 0.0, -S]), region=tri, load=LOAD_Y, weight=2.0),
            C.Objective('stress', target=torch.tensor([0.0, 0.0, 0.0]), region=disc, load=LOAD_Y, weight=2.0),
            C.Objective('stress', target=torch.tensor([0.0, 0.0, -SBG]), region=bg, load=LOAD_Y, weight=0.6)]
    r = C.optimize(prob, objs, mode='k', n_iter=NITER, reg=REG, verbose=False)
    C.apply_k_to_geo(geo, r['k'])

    sig_x_full, sig_y_full = independent_stress(geo)
    mx, my = mag3(sig_x_full), mag3(sig_y_full)

    def region_mean(m, idx):                                         # finite-safe (guards a rare blow-up)
        v = m[idx]; v = v[np.isfinite(v)]
        return float(v.mean()) if v.size else float('nan')
    bars = {'disc_x': region_mean(mx, disc), 'tri_x': region_mean(mx, tri),
            'disc_y': region_mean(my, disc), 'tri_y': region_mean(my, tri),
            'bg_x': region_mean(mx, bg), 'bg_y': region_mean(my, bg)}
    disc_sel = bars['disc_x'] / max(bars['disc_y'], 1e-9)
    tri_sel = bars['tri_y'] / max(bars['tri_x'], 1e-9)
    print(f"  [mode_selective {topo}] tri={prob.n_tri}", flush=True)
    print(f"    DISC ‖σ‖: +x={bars['disc_x']:.3f}  −y={bars['disc_y']:.3f}  (x/y {disc_sel:.2f}×)  "
          f"| TRI ‖σ‖: +x={bars['tri_x']:.3f}  −y={bars['tri_y']:.3f}  (y/x {tri_sel:.2f}×)  "
          f"| background ‖σ‖: +x={bars['bg_x']:.3f}  −y={bars['bg_y']:.3f}", flush=True)
    csv_rows.append((topo, 'disc', f'{bars["disc_x"]:.4f}', f'{bars["disc_y"]:.4f}', f'{disc_sel:.3f}'))
    csv_rows.append((topo, 'triangle', f'{bars["tri_x"]:.4f}', f'{bars["tri_y"]:.4f}', f'{tri_sel:.3f}'))
    csv_rows.append((topo, 'background', f'{bars["bg_x"]:.4f}', f'{bars["bg_y"]:.4f}', '1.0'))

    C6 = C.sim_per_triangle_C6(geo)
    C.save_network(os.path.join(C.savedir(CASE), f'mode_selective_{topo}.npz'), geo, r['k'], C6,
                   disc=disc_spec, triangle=tri_spec, S=float(S), n_tri=int(prob.n_tri))
    plot_mode_selective(geo, disc_spec, tri_spec, sig_x_full, sig_y_full, bars, topo,
                        os.path.join(C.savedir(CASE), f'strain_stress_mode_selective_{topo}.png'))


def main():
    csv_rows = []
    for topo in TOPOS:
        run(topo, csv_rows)
    C.write_csv(os.path.join(C.savedir(CASE), 'mode_selective.csv'),
                ['topo', 'shape', 'mean_stressmag_under_+x', 'mean_stressmag_under_-y',
                 'own_load_selectivity'], csv_rows)
    print('done')


if __name__ == '__main__':
    main()
