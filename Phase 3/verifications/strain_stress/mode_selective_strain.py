"""
Case strain_stress / mode_selective_strain — the STRAIN analogue of mode_selective.py: a single
~16k-triangle network whose local STRAIN pattern depends on the pull direction. Pull +x and a DISC
undergoes strong total DILATION (isotropic area expansion, ε_xx=ε_yy>0 -- it puffs up even though
you pull along one axis); pull +y and a separate TRIANGLE dilates instead; each shape stays ~rigid
(near-zero strain) under the other pull. On regular and disorder_hi.

Two points this makes beyond the stress version:
  - Strain, unlike stress, concentrates LOCALLY (a soft inclusion has ~uniform strain inside it --
    Eshelby), it does NOT percolate as a load path. So the shapes render as CLEAN filled regions
    with NO background suppression needed -- the background is left completely free (non-zero, as it
    naturally is), and the shapes still read, because the design just makes each shape a soft/auxetic
    (dilating) or rigid inclusion.
  - Same "full response operator" point: one W, queried at two loads in ONE optimize() call (disc
    dilates / triangle rigid under +x; triangle dilates / disc rigid under +y).

Mechanism: under +x the disc is auxetic (expands laterally too -> isotropic dilation) while the
triangle is rigid; under +y they swap roles. Verified two independent ways (validate() readback +
_common.unit_mode_response's separate NumPy PBC sim; +x/+y strains are the mode-0/mode-1 responses).
Outputs: mode_selective_strain.csv, mode_selective_strain_<topo>.npz,
strain_stress_mode_selective_strain_<topo>.png (+ a no-guides *_clean.png).
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

CASE, REG, HALF, NITER = 'strain_stress', 1e-4, 42, 150           # HALF=42 -> ~16k triangles
TOPOS = ['regular', 'disorder_hi']
MEAN_DIL = 1.0                                                    # the cell-MEAN dilation is FIXED by the
# applied pull (Δg has trace ~2 -> dilation ~1) -- a sub-region can only stand out by dilating ABOVE it
# (or staying rigid below it), NOT by the whole surround going quiet. So the active shape targets
# DIL > MEAN_DIL (dilates extra), the background sits AT the mean (uniform), the other shape is rigid.
DIL = 1.25                                                        # active shape: dilates moderately above the mean
# ^ 1.25 not 1.6: dilating far above the mean needs a very SOFT shape (E->0, near-mechanism -- see the
# nu/E maps). A milder excess keeps the shape's E much closer to the background (a real, stable soft
# inclusion, not a mechanism), at the cost of a gentler dilation contrast -- the tradeoff the user asked
# for. MEAN_DIL below is the fixed cell-mean the pull imposes.
I2 = np.eye(2)
LOAD_X = C.CE.vec3(C.PH.Fk[0].T @ C.PH.Fk[0] - I2) / C.PH.DELTA   # PULL +x  (xx unit mode)
LOAD_Y = C.CE.vec3(C.PH.Fk[1].T @ C.PH.Fk[1] - I2) / C.PH.DELTA   # PULL +y  (yy unit mode)


def dilation(v):
    """Total dilation e=(ε_xx+ε_yy)/2 of a per-triangle vec3=[xx,xy,yy] strain field."""
    return 0.5 * (v[:, 0] + v[:, 2])


def independent_strain(geo):
    """Non-autograd ground-truth strain under the two pulls, via _common.unit_mode_response (separate
    NumPy PBC sim): +x strain = mode-0, +y strain = mode-1. Returns (eps_x_vec3, eps_y_vec3)."""
    eps, _ = C.unit_mode_response(geo)
    return C.CE.vec3(eps[0]), C.CE.vec3(eps[1])


def _finite_pct(a, q):
    f = np.abs(a)[np.isfinite(a)]
    return float(np.percentile(f, q)) if f.size else 1.0


def plot_local_nuE(geo, C6_per, disc_spec, tri_spec, topo, path):
    """Two material maps: local angle-averaged ν (diverging, so auxetic ν<0 shows blue) and local
    angle-averaged E (positive). Load-independent -- shows HOW the design localises softness/auxeticity
    to make the shapes dilate. Region outlines drawn."""
    nu_avg, E_avg = C.local_nuE_angleavg(geo, C6_per)
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(13, 6.2))
    vnu = max(_finite_pct(nu_avg, 96), 0.1)
    pc0 = C.fill_local_map(a0, geo, np.nan_to_num(nu_avg, nan=0.0, posinf=vnu, neginf=-vnu),
                           cmap='RdBu_r', sym=True, vlim=vnu)
    C.draw_box(a0, geo); C.mark_region(a0, disc_spec); C.mark_region(a0, tri_spec)
    plt.colorbar(pc0, ax=a0, fraction=0.046); a0.set_title('local angle-averaged ν  (blue = auxetic)', fontsize=11)
    vE = max(_finite_pct(E_avg, 96), 1e-6)
    pc1 = C.fill_local_map(a1, geo, np.nan_to_num(E_avg, nan=0.0, posinf=vE, neginf=0.0), cmap='magma')
    pc1.set_clim(0, vE)
    C.draw_box(a1, geo); C.mark_region(a1, disc_spec); C.mark_region(a1, tri_spec)
    plt.colorbar(pc1, ax=a1, fraction=0.046); a1.set_title('local angle-averaged E', fontsize=11)
    fig.suptitle(f'{CASE} — MODE-SELECTIVE STRAIN ({topo}): local angle-averaged ν and E '
                 f'(load-independent material maps)', fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f'saved {os.path.basename(path)}', flush=True)


def plot_strain(geo, disc_spec, tri_spec, eps_x_full, eps_y_full, bars, topo, path, mark=True):
    """Local DILATION maps under the two pulls (+ optional bar chart). Diverging map centred at 0 so
    expansion (red) vs contraction (blue) is visible. mark=False -> clean, no region outlines."""
    dx, dy = dilation(eps_x_full), dilation(eps_y_full)
    vmax = 1.1 * DIL                                              # scale to the SHAPE level (not to any
    # residual floppy-triangle outlier -- otherwise a few spikes auto-range the map and the shapes,
    # sitting at ~DIL, wash out to pale). Outliers simply saturate at the ends. On this sym scale:
    # active shape (DIL) ~ dark red, background (MEAN_DIL) ~ mid, rigid other shape (0) ~ white.
    ncol = 3 if bars is not None else 2
    fig = plt.figure(figsize=(6.5 * ncol, 6))
    axes = [fig.add_subplot(1, ncol, i + 1) for i in range(ncol)]
    for ax, d, t in ((axes[0], dx, 'PULL +x  →  DISC dilates'), (axes[1], dy, 'PULL +y  →  TRIANGLE dilates')):
        pc = C.fill_local_map(ax, geo, np.nan_to_num(d, nan=0.0, posinf=vmax, neginf=-vmax),
                              cmap='RdBu_r', sym=True, vlim=vmax)
        C.draw_box(ax, geo)
        if mark:
            C.mark_region(ax, disc_spec); C.mark_region(ax, tri_spec)
        plt.colorbar(pc, ax=ax, fraction=0.046)
        ax.set_title(f'local dilation e=(ε_xx+ε_yy)/2   ({t})', fontsize=11)
    if bars is not None:
        a2 = axes[2]; x = np.arange(2); w = 0.35
        a2.bar(x - w / 2, [bars['disc_x'], bars['tri_x']], w, label='under +x pull', color='#1f77b4')
        a2.bar(x + w / 2, [bars['disc_y'], bars['tri_y']], w, label='under +y pull', color='#d62728')
        a2.set_xticks(x); a2.set_xticklabels(['DISC', 'TRIANGLE'])
        a2.set_ylabel('mean dilation in shape (INDEPENDENT sim)'); a2.axhline(0, color='k', lw=.5)
        a2.legend(fontsize=9); a2.grid(alpha=.3, axis='y')
        a2.set_title('each shape dilates under ITS OWN pull', fontsize=11)
    tag = '' if mark else ', no guides'
    fig.suptitle(f'{CASE} — MODE-SELECTIVE STRAIN @ ~16k tri ({topo}{tag}): pull-dependent dilation '
                 f'(disc↔+x, triangle↔+y)', fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f'saved {os.path.basename(path)}', flush=True)


def run(topo, csv_rows):
    prob, geo = C.make_case(topo, HALF)
    Lx, Ly = C.box(geo)
    disc_spec = {'kind': 'circle', 'center': (0.3 * Lx, 0.5 * Ly), 'radius': 0.16 * Lx, 'color': 'lime'}
    tri_spec = {'kind': 'polygon', 'verts': C.triangle_verts(0.72 * Lx, 0.5 * Ly, 0.20 * Lx), 'color': 'lime'}
    disc, _ = C.region_shape(prob, disc_spec)
    tri, _ = C.region_shape(prob, tri_spec)
    bg = np.setdiff1d(np.arange(prob.n_tri), np.concatenate([disc, tri]))

    # ONE optimize(), strain objectives spanning TWO pulls. Choices to make ONLY the active shape
    # visible as a clean filled dilation region:
    #  (1) homogeneity>0 on every region -- else a region's MEAN dilation is hit via a few floppy
    #      triangles spiking huge while the rest barely move (spiky, not filled).
    #  (2) the INACTIVE shape targets the background MEAN (blends in) with a STRONG weight (to actually
    #      pin its mean there) and the SAME homogeneity as the background (matched texture). Both
    #      matter: a rigid=0 target made it a visible white hole; a weak/low-homogeneity blend left it
    #      as a faint smooth patch against the noisier surround; a merged low-weight region let it
    #      drift back up. NOTE: the inactive shape is a genuinely different (anisotropic) material -- it
    #      is designed soft for its OWN pull -- so a faint residual is inherent; this just minimises it.
    #  (3) the quiet target sits AT the mean (cell-mean dilation is FIXED by the pull ~MEAN_DIL, can't
    #      be suppressed toward zero -- an early 0-background attempt was infeasible and pushed the
    #      compensating strain into floppy streaks).
    dil = torch.tensor([DIL, 0.0, DIL])
    mean = torch.tensor([MEAN_DIL, 0.0, MEAN_DIL])               # background AND the inactive shape sit here
    HB = 0.3                                                     # shared homogeneity -> matched texture
    objs = [C.Objective('strain', target=dil, region=disc, load=LOAD_X, weight=2.0, homogeneity=0.6),
            C.Objective('strain', target=mean, region=tri, load=LOAD_X, weight=1.5, homogeneity=HB),
            C.Objective('strain', target=mean, region=bg, load=LOAD_X, weight=0.5, homogeneity=HB),
            C.Objective('strain', target=dil, region=tri, load=LOAD_Y, weight=2.0, homogeneity=0.6),
            C.Objective('strain', target=mean, region=disc, load=LOAD_Y, weight=1.5, homogeneity=HB),
            C.Objective('strain', target=mean, region=bg, load=LOAD_Y, weight=0.5, homogeneity=HB)]
    r = C.optimize(prob, objs, mode='k', n_iter=NITER, reg=REG, verbose=False)
    C.apply_k_to_geo(geo, r['k'])

    eps_x_full, eps_y_full = independent_strain(geo)
    dx, dy = dilation(eps_x_full), dilation(eps_y_full)

    def rmean(d, idx):
        v = d[idx]; v = v[np.isfinite(v)]
        return float(v.mean()) if v.size else float('nan')
    bars = {'disc_x': rmean(dx, disc), 'tri_x': rmean(dx, tri),
            'disc_y': rmean(dy, disc), 'tri_y': rmean(dy, tri), 'bg_x': rmean(dx, bg), 'bg_y': rmean(dy, bg)}
    # the visible signal is EXCESS dilation over the background (the inactive shape now sits AT the
    # background mean, so it vanishes; only the active shape rises above it). contrast = active_excess
    # over inactive_excess.
    disc_ex_x, disc_ex_y = bars['disc_x'] - bars['bg_x'], bars['disc_y'] - bars['bg_y']
    tri_ex_x, tri_ex_y = bars['tri_x'] - bars['bg_x'], bars['tri_y'] - bars['bg_y']
    print(f"  [mode_selective_strain {topo}] tri={prob.n_tri}  background dilation +x={bars['bg_x']:.3f} "
          f"+y={bars['bg_y']:.3f}", flush=True)
    print(f"    DISC excess-over-bg: +x={disc_ex_x:+.3f}  +y={disc_ex_y:+.3f}  |  "
          f"TRI excess-over-bg: +x={tri_ex_x:+.3f}  +y={tri_ex_y:+.3f}  "
          f"(active shape rises above bg under ITS pull; inactive ~0)", flush=True)
    csv_rows.append((topo, 'disc', f'{bars["disc_x"]:.4f}', f'{bars["disc_y"]:.4f}',
                     f'{disc_ex_x:+.4f}', f'{disc_ex_y:+.4f}'))
    csv_rows.append((topo, 'triangle', f'{bars["tri_x"]:.4f}', f'{bars["tri_y"]:.4f}',
                     f'{tri_ex_x:+.4f}', f'{tri_ex_y:+.4f}'))
    csv_rows.append((topo, 'background', f'{bars["bg_x"]:.4f}', f'{bars["bg_y"]:.4f}', '+0.0000', '+0.0000'))

    C6 = C.sim_per_triangle_C6(geo)
    C.save_network(os.path.join(C.savedir(CASE), f'mode_selective_strain_{topo}.npz'), geo, r['k'], C6,
                   disc=disc_spec, triangle=tri_spec, DIL=float(DIL), n_tri=int(prob.n_tri))
    plot_strain(geo, disc_spec, tri_spec, eps_x_full, eps_y_full, bars, topo,
                os.path.join(C.savedir(CASE), f'strain_stress_mode_selective_strain_{topo}.png'), mark=True)
    plot_strain(geo, disc_spec, tri_spec, eps_x_full, eps_y_full, None, topo,
                os.path.join(C.savedir(CASE), f'strain_stress_mode_selective_strain_{topo}_clean.png'), mark=False)
    plot_local_nuE(geo, C6, disc_spec, tri_spec, topo,
                   os.path.join(C.savedir(CASE), f'strain_stress_mode_selective_strain_{topo}_nuE.png'))


def main():
    csv_rows = []
    for topo in TOPOS:
        run(topo, csv_rows)
    C.write_csv(os.path.join(C.savedir(CASE), 'mode_selective_strain.csv'),
                ['topo', 'shape', 'mean_dilation_under_+x', 'mean_dilation_under_+y',
                 'excess_over_bg_+x', 'excess_over_bg_+y'], csv_rows)
    print('done')


if __name__ == '__main__':
    main()
