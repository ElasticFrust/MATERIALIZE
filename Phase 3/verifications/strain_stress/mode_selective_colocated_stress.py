"""
Case strain_stress / mode_selective_colocated_stress — a co-located, FILLED DISC and FILLED TRIANGLE
that respond to DIFFERENT forcing directions in their STRESS: pull along x (ε_xx) lights up the DISC;
pull along y (ε_yy) lights up the whole TRIANGLE. The disc is NESTED inside the filled triangle (same
spot). The trick to make BOTH read as filled shapes (two filled shapes cannot co-locate as separate
selective regions -- an earlier attempt made the triangle a mere frame/outline around the disc): the
disc is stress-active under BOTH pulls (it is part of the triangle), and it is the triangle-minus-disc
FRAME being held quiet under ε_xx that makes ONLY the disc show there; under ε_yy the whole triangle
(disc + frame) is active, so it fills. Unlike mode_selective_colocated.py this uses two PULLS (ε_xx
and ε_yy, not stretch-x vs compress-y) and leaves the BACKGROUND completely FREE (non-zero).

Stress (not strain), so the usual caveat applies (see mode_selective_colocated.py): stress under a
uniform load follows equilibrium load PATHS, so the disc under ε_xx grows a horizontal feeder and the
triangular frame under ε_yy channels vertically -- the direction-selectivity is clean, the shapes are
load-path-shaped. Same "full response operator" point: one W, two loads (disc-loud/frame-quiet under
ε_xx; frame-loud/disc-quiet under ε_yy), one optimize() call. Verified two independent ways (validate
readback + _common.unit_mode_response's separate NumPy sim; ε_xx/ε_yy stresses are the mode-0/mode-1
responses). Outputs: mode_selective_colocated_stress.csv, mode_selective_colocated_stress_<topo>.npz,
strain_stress_mode_selective_colocated_stress_<topo>.png (+ a no-guides *_clean.png).
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
from mode_selective_colocated import mag3, _finite_pct                # reuse (DRY)

CASE, REG, HALF, NITER = 'strain_stress', 1e-4, 42, 130               # HALF=42 -> ~16k triangles
TOPOS = ['regular', 'disorder_hi']
S = 0.5                                                               # active-shape stress target
I2 = np.eye(2)
LOAD_X = C.MO.vec3(C.PH.Fk[0].T @ C.PH.Fk[0] - I2) / C.PH.DELTA       # PULL +x  (ε_xx)
LOAD_Y = C.MO.vec3(C.PH.Fk[1].T @ C.PH.Fk[1] - I2) / C.PH.DELTA       # PULL +y  (ε_yy)


def independent_stress(geo):
    """Non-autograd ground-truth stress under the two PULLS (+x, +y) via _common.unit_mode_response:
    ε_xx stress = mode-0, ε_yy stress = mode-1 (both positive; a separate NumPy PBC sim)."""
    _, sig = C.unit_mode_response(geo)
    return C.MO.vec3(sig[0]), C.MO.vec3(sig[1])


def plot_stress(geo, disc_spec, tri_spec, sig_x_full, sig_y_full, bars, topo, path, mark=True):
    """Local stress-magnitude maps under the two pulls (+ optional bar chart). mark=False -> no guides."""
    mx, my = mag3(sig_x_full), mag3(sig_y_full)
    vmax = max(_finite_pct(mx, 99), _finite_pct(my, 99), S)
    ncol = 3 if bars is not None else 2
    fig = plt.figure(figsize=(6.5 * ncol, 6))
    axes = [fig.add_subplot(1, ncol, i + 1) for i in range(ncol)]
    for ax, m, t in ((axes[0], mx, 'PULL +x (ε_xx)  →  DISC'), (axes[1], my, 'PULL +y (ε_yy)  →  TRIANGLE')):
        pc = C.fill_local_map(ax, geo, np.nan_to_num(m, nan=0.0, posinf=vmax, neginf=0.0), cmap='magma')
        pc.set_clim(0, vmax)
        C.draw_box(ax, geo)
        if mark:
            C.mark_region(ax, tri_spec); C.mark_region(ax, disc_spec)
        plt.colorbar(pc, ax=ax, fraction=0.046)
        ax.set_title(f'local ‖σ‖   ({t} lights up)', fontsize=11)
    if bars is not None:
        a2 = axes[2]; x = np.arange(2); w = 0.35
        a2.bar(x - w / 2, [bars['disc_x'], bars['tri_y']], w, label='its own pull', color='#d62728')
        a2.bar(x + w / 2, [bars['frame_x'], bars['bg_y']], w, label='reference (frame@εxx / bg@εyy)', color='#8c8c8c')
        a2.set_xticks(x); a2.set_xticklabels(['DISC @ εxx', 'TRIANGLE @ εyy'])
        a2.set_ylabel('mean ‖σ‖ (INDEPENDENT sim)'); a2.legend(fontsize=8); a2.grid(alpha=.3, axis='y')
        a2.set_title('disc reads under εxx (vs quiet frame); triangle fills under εyy', fontsize=10)
    tag = '' if mark else ', no guides'
    fig.suptitle(f'{CASE} — CO-LOCATED STRESS mode-selective @ ~16k tri ({topo}{tag}): filled DISC↔ε_xx, '
                 f'filled TRIANGLE↔ε_yy, nested (free non-zero background)', fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f'saved {os.path.basename(path)}', flush=True)


def run(topo, csv_rows):
    prob, geo = C.make_case(topo, HALF)
    Lx, Ly = C.box(geo)
    cx, cy = 0.5 * Lx, 0.5 * Ly
    disc_spec = {'kind': 'circle', 'center': (cx, cy), 'radius': 0.11 * Lx, 'color': 'cyan'}
    tri_spec = {'kind': 'polygon', 'verts': C.triangle_verts(cx, cy, 0.28 * Lx), 'color': 'lime'}
    disc, _ = C.region_shape(prob, disc_spec)
    tri_full, _ = C.region_shape(prob, tri_spec)                     # FILLED triangle (contains the disc)
    frame = np.setdiff1d(tri_full, disc)                            # triangle minus the disc

    # ONE optimize(), THREE stress objectives, TWO pulls, at ONE spot. The disc is NESTED in the
    # filled triangle so that BOTH read as FILLED shapes: under ε_xx the DISC alone carries stress
    # (the frame is held quiet), so you see a DISC; under ε_yy the WHOLE triangle (disc + frame)
    # carries stress, so you see a filled TRIANGLE. The disc is thus stress-active under BOTH pulls
    # (it is part of the triangle); it is the FRAME being quiet under ε_xx that makes only the disc
    # show there. NO background objective -- the surround is left FREE (non-zero).
    # (homogeneity on the active objectives was tried to fill the disc uniformly -- it BACKFIRED,
    # spreading stress so the frame carried MORE than the disc; omitted. The disc under εxx is
    # fundamentally a horizontal load-PATH streak, and the frame around it sits on that path, so the
    # disc/frame contrast caps at ~1.3x -- an inherent stress limit. The εyy triangle fills fine. For
    # clean FILLED disc AND triangle, see mode_selective_strain.py -- strain localises, stress does not.)
    objs = [C.Objective('stress', target=torch.tensor([S, 0.0, 0.0]), region=disc, load=LOAD_X, weight=2.0),
            C.Objective('stress', target=torch.tensor([0.0, 0.0, S]), region=tri_full, load=LOAD_Y, weight=2.0),
            C.Objective('stress', target=torch.tensor([0.0, 0.0, 0.0]), region=frame, load=LOAD_X, weight=3.0)]
    r = C.optimize(prob, objs, mode='k', n_iter=NITER, reg=REG, verbose=False)
    C.apply_k_to_geo(geo, r['k'])

    sig_x_full, sig_y_full = independent_stress(geo)
    mx, my = mag3(sig_x_full), mag3(sig_y_full)

    def rmean(m, idx):
        v = m[idx]; v = v[np.isfinite(v)]
        return float(v.mean()) if v.size else float('nan')
    bg = np.setdiff1d(np.arange(prob.n_tri), tri_full)
    bars = {'disc_x': rmean(mx, disc), 'frame_x': rmean(mx, frame), 'tri_y': rmean(my, tri_full),
            'disc_y': rmean(my, disc), 'frame_y': rmean(my, frame), 'bg_x': rmean(mx, bg), 'bg_y': rmean(my, bg)}
    # under εxx: DISC lights, frame stays down -> disc reads. under εyy: whole triangle lights.
    disc_vs_frame_xx = bars['disc_x'] / max(bars['frame_x'], 1e-9)
    tri_vs_bg_yy = bars['tri_y'] / max(bars['bg_y'], 1e-9)
    print(f"  [colocated_stress {topo}] tri={prob.n_tri}  bg ‖σ‖: εxx={bars['bg_x']:.3f} εyy={bars['bg_y']:.3f}",
          flush=True)
    print(f"    εxx: DISC ‖σ‖={bars['disc_x']:.3f} vs FRAME ‖σ‖={bars['frame_x']:.3f} (disc/frame {disc_vs_frame_xx:.2f}× -> disc shows)  |  "
          f"εyy: whole TRIANGLE ‖σ‖={bars['tri_y']:.3f} vs bg {bars['bg_y']:.3f} ({tri_vs_bg_yy:.2f}× -> triangle shows)",
          flush=True)
    csv_rows.append((topo, 'disc_under_exx', f'{bars["disc_x"]:.4f}', f'{bars["frame_x"]:.4f}', f'{disc_vs_frame_xx:.3f}'))
    csv_rows.append((topo, 'triangle_under_eyy', f'{bars["tri_y"]:.4f}', f'{bars["bg_y"]:.4f}', f'{tri_vs_bg_yy:.3f}'))

    C6 = C.sim_per_triangle_C6(geo)
    C.save_network(os.path.join(C.savedir(CASE), f'mode_selective_colocated_stress_{topo}.npz'), geo, r['k'],
                   C6, disc=disc_spec, triangle=tri_spec, S=float(S), n_tri=int(prob.n_tri))
    plot_stress(geo, disc_spec, tri_spec, sig_x_full, sig_y_full, bars, topo,
                os.path.join(C.savedir(CASE), f'strain_stress_mode_selective_colocated_stress_{topo}.png'), mark=True)
    plot_stress(geo, disc_spec, tri_spec, sig_x_full, sig_y_full, None, topo,
                os.path.join(C.savedir(CASE), f'strain_stress_mode_selective_colocated_stress_{topo}_clean.png'), mark=False)


def main():
    csv_rows = []
    for topo in TOPOS:
        run(topo, csv_rows)
    C.write_csv(os.path.join(C.savedir(CASE), 'mode_selective_colocated_stress.csv'),
                ['topo', 'reads_as', 'active_mean_stressmag', 'reference_mean_stressmag',
                 'contrast_ratio'], csv_rows)
    print('done')


if __name__ == '__main__':
    main()
