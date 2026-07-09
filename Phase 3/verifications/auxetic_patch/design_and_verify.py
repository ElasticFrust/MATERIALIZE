"""
Case auxetic_patch — LOCAL / SPATIAL control of the elastic response. A region is designed to
differ from its surroundings; the designed network is then INDEPENDENTLY simulated and checked
region-by-region. Four studies:

  GROUP 1 — SHAPE GALLERY: an auxetic patch (ν≈−0.3) in a normal matrix (ν≈+0.3), for different
            region SHAPES and positions across topologies: disc, square, triangle, ring.

  GROUP 2 — CONTRAST VARIETY (same disc, regular lattice): the patch need not be auxetic, just
            DIFFERENT — auxetic-in-normal, normal-in-auxetic (surroundings auxetic), a stiff-E
            patch, and a soft-E patch.

  GROUP 3 — DECOUPLED E / ν: E is uniform everywhere EXCEPT one region R_E, while ν is uniform
            everywhere EXCEPT a DIFFERENT region R_ν — simultaneously. Independent spatial control
            of the two moduli.

  CHECK   — REGIONAL ISOTROPY: is ν actually isotropic inside the patch / outside? Measured as the
            angular spread of ν(θ) from the region tensor.

Regional ν/E is the region-averaged per-triangle PHYSICAL tensor (validates the design→simulate
loop and the spatial pattern). Outputs:
  - auxetic_patch.csv                    : group, name, region, quantity, target, sim
  - auxetic_patch_shapes.png             : GROUP 1  [rigidity k | local ν | local E], region marked
  - auxetic_patch_contrast.png           : GROUP 2  ditto, four contrasts
  - auxetic_patch_decoupled.png          : GROUP 3  R_E (cyan) vs R_ν (lime) — E contrast only in R_E,
                                           ν contrast only in R_ν
  - auxetic_patch_regional_nutheta.png   : CHECK    ν(θ) inside vs outside a patch
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _common as C

CASE, REG, N, NITER = 'auxetic_patch', 2e-3, 10, 120
csv_rows = []


def run(prob, geo, objectives):
    """Design k for the objectives, install on geo, return (k, per-triangle sim tensor)."""
    r = C.optimize(prob, objectives, mode='k', n_iter=NITER, reg=REG, verbose=False)
    C.apply_k_to_geo(geo, r['k'])
    return r['k'], C.sim_per_triangle_C6(geo)


def reg_nuE(geo, C6, idx):
    return C.c6_nuE(C.region_phys_C6(geo, C6, idx))


def _safe(s):
    return (s.replace(' ', '_').replace('·', '').replace('/', '_')
             .replace('(', '').replace(')', '').replace('__', '_'))


# ------------------------------------------------------------------ GROUP 1: shape gallery
def group1():
    global csv_rows
    demos = [                                           # (name, topo, shape-spec builder(Lx,Ly))
        ('disc @ center · regular', 'regular',
         lambda Lx, Ly: {'kind': 'circle', 'center': (0.50 * Lx, 0.50 * Ly), 'radius': 0.18 * Lx}),
        ('square @ upper-left · aniso_str', 'aniso_str',
         lambda Lx, Ly: {'kind': 'rect', 'center': (0.34 * Lx, 0.66 * Ly), 'w': 0.34 * Lx, 'h': 0.34 * Ly}),
        ('triangle @ right · disorder_lo', 'disorder_lo',
         lambda Lx, Ly: {'kind': 'polygon', 'verts': C.triangle_verts(0.66 * Lx, 0.50 * Ly, 0.20 * Lx)}),
        ('ring @ center · aniso_shr', 'aniso_shr',
         lambda Lx, Ly: {'kind': 'ring', 'center': (0.50 * Lx, 0.50 * Ly),
                         'r_in': 0.12 * Lx, 'r_out': 0.24 * Lx}),
    ]
    nd = C.networks_dir(CASE)
    entries = []; check = None
    for name, topo, shapefn in demos:
        prob, geo = C.make_case(topo, N)
        Lx, Ly = C.box(geo)
        patch, spec = C.region_shape(prob, shapefn(Lx, Ly))
        outside = np.setdiff1d(np.arange(prob.n_tri), patch)
        k, C6 = run(prob, geo, [C.Objective('nu', 0.30, region=outside, weight=1.0),
                                C.Objective('nu', -0.30, region=patch, weight=1.5)])
        pin = reg_nuE(geo, C6, patch)[0]; pout = reg_nuE(geo, C6, outside)[0]
        csv_rows += [('shapes', name, 'patch', 'nu', -0.30, f'{pin:.3f}'),
                     ('shapes', name, 'outside', 'nu', 0.30, f'{pout:.3f}')]
        C.save_network(os.path.join(nd, f'shape_{_safe(name)}.npz'), geo, k, C6, group='shapes',
                       name=name, topo=topo, N=N, region=spec, patch_nu=float(pin), outside_nu=float(pout))
        print(f"  [1] {name:34s} | patch nu={pin:+.3f} outside nu={pout:+.3f}", flush=True)
        entries.append((name, geo, k, C6, spec))
        if check is None:                               # regional isotropy on the first (disc) demo
            th = C.ANG
            check = dict(name=name, inside=C.nu_E_theta(C.region_phys_C6(geo, C6, patch), th)[0],
                         outside=C.nu_E_theta(C.region_phys_C6(geo, C6, outside), th)[0], th=th)
    C.design_detail_figure(os.path.join(C.savedir(CASE), f'{CASE}_shapes.png'), entries,
                           f'{CASE} GROUP 1 — auxetic patch (ν≈−0.3) in normal matrix (ν≈+0.3): '
                           f'shape gallery [rigidity k | local ν | local E], region in lime')
    print('saved shapes')
    return check


# ------------------------------------------------------------------ GROUP 2: contrast variety
def group2():
    global csv_rows
    disc = lambda Lx, Ly: {'kind': 'circle', 'center': (0.50 * Lx, 0.50 * Ly), 'radius': 0.20 * Lx}
    contrasts = [                                       # (name, [(kind,out_target),(kind,patch_target)])
        ('auxetic disc in normal (nu -0.3 in +0.3)', ('nu', 0.30), ('nu', -0.30)),
        ('normal disc in auxetic (nu +0.3 in -0.3)', ('nu', -0.30), ('nu', 0.30)),
        ('stiff-E disc in soft (E 1.6 in 0.7)',      ('E', 0.70),   ('E', 1.60)),
        ('soft-E disc in stiff (E 0.6 in 1.4)',      ('E', 1.40),   ('E', 0.60)),
    ]
    nd = C.networks_dir(CASE)
    entries = []
    for name, (okind, otgt), (pkind, ptgt) in contrasts:
        prob, geo = C.make_case('disorder_hi', N)       # reaches both +/-0.3 freely (auxetic matrix ok)
        Lx, Ly = C.box(geo)
        patch, spec = C.region_shape(prob, disc(Lx, Ly))
        outside = np.setdiff1d(np.arange(prob.n_tri), patch)
        k, C6 = run(prob, geo, [C.Objective(okind, otgt, region=outside, weight=1.0),
                                C.Objective(pkind, ptgt, region=patch, weight=1.5)])
        pin = reg_nuE(geo, C6, patch); pout = reg_nuE(geo, C6, outside)
        q = 0 if pkind == 'nu' else 1
        csv_rows += [('contrast', name, 'patch', pkind, ptgt, f'{pin[q]:.3f}'),
                     ('contrast', name, 'outside', okind, otgt, f'{pout[q]:.3f}')]
        C.save_network(os.path.join(nd, f'contrast_{_safe(name)}.npz'), geo, k, C6, group='contrast',
                       name=name, topo='disorder_hi', N=N, region=spec, patch_kind=pkind,
                       patch_val=float(pin[q]), outside_val=float(pout[q]))
        print(f"  [2] {name:42s} | patch={pin[q]:+.3f} outside={pout[q]:+.3f}", flush=True)
        entries.append((name, geo, k, C6, spec))
    C.design_detail_figure(os.path.join(C.savedir(CASE), f'{CASE}_contrast.png'), entries,
                           f'{CASE} GROUP 2 — contrast variety (patch ≠ necessarily auxetic): '
                           f'[rigidity k | local ν | local E], region in lime')
    print('saved contrast')


# ------------------------------------------------------------------ GROUP 3: decoupled E / ν
def group3():
    global csv_rows
    NU0, NU1, E0, E1 = 0.20, -0.30, 1.0, 1.8         # must match C.decoupled_ENu_design's recipe
    nd = C.networks_dir(CASE)
    entries = []
    for topo in ['regular', 'aniso_str']:
        prob, geo = C.make_case(topo, N)
        k, C6, RE_spec, RN_spec, R_E, R_N, out_E, out_N = C.decoupled_ENu_design(prob, geo, NITER, REG)
        nuE_RN = reg_nuE(geo, C6, R_N); nuE_RE = reg_nuE(geo, C6, R_E)
        nu_bg = reg_nuE(geo, C6, np.setdiff1d(out_N, R_E))[0]
        E_bg = reg_nuE(geo, C6, np.setdiff1d(out_E, R_N))[1]
        csv_rows += [('decoupled', topo, 'R_nu', 'nu', NU1, f'{nuE_RN[0]:.3f}'),
                     ('decoupled', topo, 'R_E', 'E', E1, f'{nuE_RE[1]:.3f}'),
                     ('decoupled', topo, 'background', 'nu', NU0, f'{nu_bg:.3f}'),
                     ('decoupled', topo, 'background', 'E', E0, f'{E_bg:.3f}')]
        C.save_network(os.path.join(nd, f'decoupled_{topo}.npz'), geo, k, C6, group='decoupled',
                       topo=topo, N=N, region=[RE_spec, RN_spec], R_nu_nu=float(nuE_RN[0]),
                       R_E_E=float(nuE_RE[1]), bg_nu=float(nu_bg), bg_E=float(E_bg))
        print(f"  [3] {topo:12s} | R_nu nu={nuE_RN[0]:+.3f}(tgt{NU1}) R_E E={nuE_RE[1]:.3f}(tgt{E1}) "
              f"| bg nu={nu_bg:+.3f} bg E={E_bg:.3f}", flush=True)
        entries.append((f'{topo}: E-patch (cyan) + ν-patch (lime)', geo, k, C6, [RE_spec, RN_spec]))
    C.design_detail_figure(os.path.join(C.savedir(CASE), f'{CASE}_decoupled.png'), entries,
                           f'{CASE} GROUP 3 — DECOUPLED: E differs only in R_E (cyan), ν differs only in '
                           f'R_ν (lime). local ν map shows contrast at lime; local E map at cyan')
    print('saved decoupled')


def main():
    check = group1()
    group2()
    group3()
    C.write_csv(os.path.join(C.savedir(CASE), f'{CASE}.csv'),
                ['group', 'name', 'region', 'quantity', 'target', 'sim'], csv_rows)

    # ---- regional isotropy check: ν(θ) inside vs outside a patch ----
    if check:
        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        deg = np.degrees(check['th'])
        ax.plot(deg, check['inside'], '-o', ms=3, color='#d62728', label='inside patch')
        ax.plot(deg, check['outside'], '-o', ms=3, color='#1f77b4', label='outside')
        ax.axhline(check['inside'].mean(), color='#d62728', ls=':', lw=1)
        ax.axhline(check['outside'].mean(), color='#1f77b4', ls=':', lw=1)
        ax.set_xlabel('θ (deg)'); ax.set_ylabel('ν(θ) (simulated, region tensor)'); ax.set_xlim(0, 180)
        ax.set_title(f'{CASE} — regional isotropy: ν(θ) in/out of "{check["name"]}"\n'
                     f'inside spread={np.ptp(check["inside"]):.3f}, outside spread={np.ptp(check["outside"]):.3f}')
        ax.grid(alpha=0.3); ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(C.savedir(CASE), f'{CASE}_regional_nutheta.png'), dpi=150,
                    bbox_inches='tight'); plt.close()
        print(f"regional isotropy: inside spread={np.ptp(check['inside']):.3f} "
              f"outside spread={np.ptp(check['outside']):.3f}")
    print('done')


if __name__ == '__main__':
    main()
