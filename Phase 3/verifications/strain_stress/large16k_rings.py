"""
Case strain_stress / large16k_rings — the concentric-rings "bullseye" scaled to a ~16k-triangle
network (HALF=42, matching ../large16k/large_designs.py's own convention), WITHOUT the alternating
SIGN target that design_and_verify.py's CONCENTRIC RINGS demo found thermodynamically forbidden
(negative mean stress under a globally imposed dilation needs a locally negative bulk modulus in a
passive sub-region -- not achievable by a stable material, only faked via a mechanism). Instead the
bullseye look comes from alternating MAGNITUDE at a fixed (positive) sign: STRONG / WEAK / STRONG /
WEAK concentric bands, all under the same isotropic (equi-biaxial) load as before -- same rotational
symmetry, but nothing forbidden this time.

Each band also gets a low-weight 'isotropy' objective (the local response should be direction-
independent, not just have the right mean) -- a nice-to-have, not the main target. Plots BOTH the
stress and strain response fields (not stress alone), plus each band's directional nu(theta)/E(theta)
average.
"""
import os, sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import _common as C
import design_and_verify as dv          # reuse LOAD_ISO / independent_check_iso -- no duplication

CASE, REG, HALF, NITER = 'strain_stress', 1e-4, 42, 70        # HALF=42 -> ~16k triangles
TOPOS = ['regular', 'disorder_hi']
# STRONG/WEAK are both genuinely STRESS-BEARING levels (a 2x contrast), NOT stiff-vs-nearly-floppy:
# an earlier 1.0/0.1 (10x) drove the WEAK bands so soft -- to carry near-zero stress under the
# dilation -- that they became near-MECHANISMS at 16k, and the independent nonlinear relaxation
# diverged (stress ~1e19, one triangle swamping the whole map). Keeping the weak bands stress-bearing
# keeps every band well-conditioned; the 2x contrast still reads clearly as alternating rings.
STRONG, WEAK = 0.8, 0.4
ISO_WEIGHT = 0.15                                              # low-weight, nice-to-have
BAND_COLORS = ['lime', 'cyan', 'orange', 'magenta']


def ring_defs(geo):
    """4 concentric bands, alternating STRONG/WEAK magnitude, same (positive) sign throughout."""
    Lx, Ly = C.box(geo)
    cx, cy = 0.5 * Lx, 0.5 * Ly
    edges = [0.0, 0.09, 0.16, 0.23, 0.30]
    mags = [STRONG, WEAK, STRONG, WEAK]
    defs = []
    for i, (r0, r1, s) in enumerate(zip(edges[:-1], edges[1:], mags)):
        if i == 0:
            spec = {'kind': 'circle', 'center': (cx, cy), 'radius': r1 * Lx, 'color': BAND_COLORS[i]}
        else:
            spec = {'kind': 'ring', 'center': (cx, cy), 'r_in': r0 * Lx, 'r_out': r1 * Lx,
                    'color': BAND_COLORS[i]}
        defs.append((f'band{i}', spec, s))
    return defs


def plot_fields(geo, specs, eps_ind_full, sig_ind_full, achieved_rows, topo, path):
    """Local mean STRESS and mean STRAIN maps (not stress alone) + per-band target/achieved bars.
    Color ranges are set EXPLICITLY (fill_local_map only honours vlim when sym=True; these are
    non-symmetric 'Reds' maps) and from FINITE percentiles, so no single near-mechanism outlier can
    auto-range the whole map to white -- a defensive guard even though the moderate STRONG/WEAK
    contrast is chosen to keep every band well-conditioned in the first place."""
    fig, (a0, a1, a2) = plt.subplots(1, 3, figsize=(20, 6))
    p_sig = 0.5 * (sig_ind_full[:, 0] + sig_ind_full[:, 2])
    p_eps = 0.5 * (eps_ind_full[:, 0] + eps_ind_full[:, 2])

    def finite_pct(a, q):
        f = np.abs(a)[np.isfinite(a)]
        return float(np.percentile(f, q)) if f.size else 1.0

    vmax_s = max(max(abs(row[1][0]) for row in achieved_rows) * 1.3, finite_pct(p_sig, 98))
    pc0 = C.fill_local_map(a0, geo, np.nan_to_num(p_sig, nan=0.0, posinf=vmax_s, neginf=0.0), cmap='Reds')
    pc0.set_clim(0, vmax_s)
    C.draw_box(a0, geo)
    for s in specs:
        C.mark_region(a0, s)
    plt.colorbar(pc0, ax=a0, fraction=0.046)
    a0.set_title(f'local mean stress p=(σ_xx+σ_yy)/2 ({topo})', fontsize=10)

    v_eps = finite_pct(p_eps, 97)
    pc1 = C.fill_local_map(a1, geo, np.nan_to_num(p_eps, nan=0.0, posinf=v_eps, neginf=0.0), cmap='Reds')
    pc1.set_clim(0, v_eps)
    C.draw_box(a1, geo)
    for s in specs:
        C.mark_region(a1, s)
    plt.colorbar(pc1, ax=a1, fraction=0.046)
    a1.set_title(f'local mean strain e=(ε_xx+ε_yy)/2 ({topo})', fontsize=10)

    names = [row[0] for row in achieved_rows]; x = np.arange(len(names)); w = 0.25
    tgt = [row[1][0] for row in achieved_rows]
    ach_stress = [0.5 * (row[3][0] + row[3][2]) for row in achieved_rows]
    ach_strain = [0.5 * (row[4][0] + row[4][2]) for row in achieved_rows]
    a2.bar(x - w, tgt, w, label='target p (stress)', color='#2c3e50')
    a2.bar(x, ach_stress, w, label='achieved p (stress, INDEPENDENT sim)', color='#d62728')
    a2b = a2.twinx()
    a2b.bar(x + w, ach_strain, w, label='achieved e (strain, INDEPENDENT sim)', color='#1f77b4', alpha=0.75)
    a2.axhline(0, color='k', lw=.5)
    a2.set_xticks(x); a2.set_xticklabels(names)
    a2.set_ylabel('p (stress)'); a2b.set_ylabel('e (strain)')
    a2.legend(loc='upper left', fontsize=7); a2b.legend(loc='upper right', fontsize=7)
    a2.set_title('per-band target vs achieved', fontsize=10)
    fig.suptitle(f'{CASE} — LARGE ~16k-triangle bullseye ({topo}): alternating-MAGNITUDE (same-sign) '
                 'bands under isotropic stretch', fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f'saved {os.path.basename(path)}')


def plot_directional(geo, C6, regions, topo, path):
    """Local directional average nu(theta)/E(theta) per band -- polar plot, mirrored to a full
    circle (centrosymmetric response: theta and theta+pi are equivalent)."""
    thetas_full = np.concatenate([C.ANG, C.ANG + np.pi])
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(12, 6.5), subplot_kw={'projection': 'polar'})
    for name, spec, idx in regions:
        C6_region = C.region_phys_C6(geo, C6, idx)
        nu_th, E_th = C.nu_E_theta(C6_region, C.ANG)
        col = spec.get('color')
        a0.plot(thetas_full, np.concatenate([nu_th, nu_th]), label=name, color=col)
        a1.plot(thetas_full, np.concatenate([E_th, E_th]), label=name, color=col)
    a0.set_title('local ν(θ) by band', fontsize=11, pad=20); a0.legend(fontsize=7, loc='upper right')
    a1.set_title('local E(θ) by band', fontsize=11, pad=20); a1.legend(fontsize=7, loc='upper right')
    fig.suptitle(f'{CASE} — LARGE ~16k bullseye ({topo}): directional ν/E average per band', fontsize=12)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f'saved {os.path.basename(path)}')


def run(topo, csv_rows):
    prob, geo = C.make_case(topo, HALF)
    defs = ring_defs(geo)
    regions = []; objs = []
    for name, spec, s in defs:
        idx, _ = C.region_shape(prob, spec)
        regions.append((name, spec, idx))
        objs.append(C.Objective('stress', target=torch.tensor([s, 0.0, s]), region=idx, load=dv.LOAD_ISO,
                                weight=1.0))
        objs.append(C.Objective('isotropy', region=idx, weight=ISO_WEIGHT))
    r = C.optimize(prob, objs, mode='k', n_iter=NITER, reg=REG, verbose=False)
    C.apply_k_to_geo(geo, r['k'])
    rep = C.validate(prob, r['k'], None, objs)

    eps_ind_full, sig_ind_full = dv.independent_check_iso(geo)
    print(f"  [large16k rings {topo}] tri={prob.n_tri}", flush=True)
    achieved_rows = []
    for i, (name, spec, idx) in enumerate(regions):
        stress_rep = rep[2 * i]                                  # isotropy report sits at rep[2*i+1]
        sig_region = sig_ind_full[idx].mean(0); eps_region = eps_ind_full[idx].mean(0)
        err_ind = float(np.abs(sig_region - stress_rep['target']).max())
        print(f"    {name:6s} target p={stress_rep['target'][0]:+.3f}  "
              f"achieved(diff-path)={stress_rep['achieved']}  achieved(INDEPENDENT sim)={sig_region}  "
              f"err_ind={err_ind:.4f}  isotropy_err={rep[2 * i + 1]['err']:.4f}", flush=True)
        csv_rows.append((topo, name, *stress_rep['target'], *stress_rep['achieved'], *sig_region,
                         *eps_region, f'{stress_rep["err"]:.4f}', f'{err_ind:.4f}'))
        achieved_rows.append((name, stress_rep['target'], stress_rep['achieved'], sig_region, eps_region))

    C6 = C.sim_per_triangle_C6(geo)
    C.save_network(os.path.join(C.savedir(CASE), f'large16k_rings_{topo}.npz'), geo, r['k'], C6,
                   regions=[s for _, s, _ in defs], n_tri=int(prob.n_tri))
    plot_fields(geo, [s for _, s, _ in defs], eps_ind_full, sig_ind_full, achieved_rows, topo,
               os.path.join(C.savedir(CASE), f'strain_stress_large16k_rings_{topo}.png'))
    plot_directional(geo, C6, regions, topo,
                     os.path.join(C.savedir(CASE), f'strain_stress_large16k_rings_{topo}_directional.png'))


def main():
    csv_rows = []
    for topo in TOPOS:
        run(topo, csv_rows)
    C.write_csv(os.path.join(C.savedir(CASE), 'large16k_rings.csv'),
                ['topo', 'band', 'target_xx', 'target_xy', 'target_yy',
                 'achieved_diffpath_xx', 'achieved_diffpath_xy', 'achieved_diffpath_yy',
                 'achieved_independent_stress_xx', 'achieved_independent_stress_xy',
                 'achieved_independent_stress_yy', 'achieved_independent_strain_xx',
                 'achieved_independent_strain_xy', 'achieved_independent_strain_yy',
                 'err_diffpath', 'err_independent'], csv_rows)
    print('done')


if __name__ == '__main__':
    main()
