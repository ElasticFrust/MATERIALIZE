"""
Case strain_stress — design objectives that target the ACTUAL per-triangle strain/stress response
(not just derived ν/E), via inverse_design.py's 'strain'/'stress' Objective kinds (see
[[phase3-strain-stress-objective-todo]]). Four demos:

  STRESS CONCENTRATOR : (uniaxial x-pull) a sub-region designed to carry AMPLIFIED σ_xx relative to
                        the background matrix (a stress-concentrating stiff patch).
  STRAIN SHIELD        : (uniaxial x-pull) a (different) sub-region designed to have NEAR-ZERO local
                        strain under the same load (a rigid, strain-shielded inclusion). Strain
                        objectives only make sense on a sub-region — whole-cell strain is degenerate
                        (see the Objective docstring in inverse_design.py: region-mean strain over
                        the WHOLE cell equals the applied load exactly).
  BULGE (AUX. INCL.)   : an off-centre TOP rectangle of a REGULAR-topology cell designed as a strongly
                        AUXETIC material (ν<0) embedded in a flat ν=0 background. An auxetic inclusion
                        expands LATERALLY under an x-pull (that IS ν<0), so it bulges where the ν=0
                        surround stays neutral -- and because ν is a clean STABLE material property, the
                        bulge appears in the actual OPEN cut-and-stretch test WITHOUT a near-mechanism.
                        (History, kept as a lesson: a low-σ_xx 'stress' target under-constrained the
                        deformation SIGN; a raw 'strain' eyy target pinned the sign, but a STRONG one --
                        eyy~2.5, a 250% response -- physically REQUIRES a near-mechanism and rendered as
                        ugly floppy strain streaks. Designing the patch as an auxetic MATERIAL is the
                        clean, mechanism-free route; on a REGULAR lattice ν cleanly reaches only ~-0.4
                        (the auxetic_sweep limit), a moderate but honest bulge -- a disordered patch
                        could go stronger.) On top of the periodic design verification, the actual
                        OPEN-boundary cut-and-stretch (matching two_region/ribbon.py's convention)
                        checks the real physical pull at both the reference strain and a large (~30%)
                        extrapolated strain (open_stretch is one LINEAR solve -> exact rescaling).
  CONCENTRIC RINGS      : (ISOTROPIC stretch, not uniaxial) a stress "bullseye" -- three contiguous
                        concentric regions (center disc, mid ring, outer ring) SIMULTANEOUSLY
                        designed for alternating-sign mean stress p=(σ_xx+σ_yy)/2 (+A / -A / +A), one
                        'stress' objective per ring in a single joint optimize() call, under an
                        ISOTROPIC (equi-biaxial) applied load matching the rings' own rotational
                        symmetry. A uniaxial pull-along-x load was tried FIRST and the middle ring
                        (sandwiched between two same-sign neighbours) failed with magnitude/disorder;
                        switching to the isotropic load does NOT fix this -- it makes the middle ring
                        WORSE, including outright numerical blow-up of the independent check at high
                        magnitude+disorder. The reason is physical, not a symmetry mismatch: under a
                        globally imposed AREAL (dilational) strain, a passive stable sub-region's mean
                        stress must have the SAME sign as the imposed dilation (negative mean stress
                        under positive global dilation needs a locally negative bulk modulus, which is
                        thermodynamically forbidden for a stable linear-elastic material). The
                        optimizer can only fake it with a near-mechanism (many bonds -> 0), which is
                        why the independent nonlinear relaxation becomes ill-conditioned instead of
                        merely inaccurate. (By contrast the uniaxial sigma_xx case is NOT similarly
                        forbidden -- Poisson/anisotropic redistribution genuinely can flip its sign in
                        a sub-region -- which is why it "merely" underachieved rather than diverging.)
                        Run at two magnitudes (A=0.35 "regular", A=1.0 "large") on both a regular and
                        a disorder_hi topology (4 designs total); kept as a documented negative result.

The first two designs are checked TWO independent ways: (a) validate()'s own differentiable-path
readback (self-consistency of the optimiser's own forward pass), and (b) INDEPENDENTLY via
_common.unit_mode_response's non-autograd NumPy PBC simulation (physical_homog.relax) — a genuinely
separate code path from the solver's own forward(), not just re-running it. The bulge's background ν
and the rings demo are checked the same two ways. Outputs: strain_stress.csv, rings.csv,
strain_stress_{stress,strain,bulge_design,bulge_nu,bulge_stretch,bulge_stretch_large,rings_*}.png,
saved networks.
"""
import os, sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _common as C

CASE, REG, N, NITER = 'strain_stress', 1e-4, 14, 150
TOPO = 'disorder_hi'

# Applied macro load: pure uniaxial pull along x (the "xx" unit mode, in the same linear-normalised
# Delta_g vec3=[xx,xy,yy] convention physical_homog.Fk/per_triangle_strain_stress use). Choosing
# exactly this unit mode means the independent check can reuse unit_mode_response's mode-0 output
# directly, with no new ground-truth machinery needed.
_Dgt0 = C.PH.Fk[0].T @ C.PH.Fk[0] - np.eye(2)
LOAD = C.MO.vec3(_Dgt0) / C.PH.DELTA

# Isotropic (equi-biaxial) load for the CONCENTRIC RINGS demo -- xx unit mode + yy unit mode. By
# linearity of the underlying elastic solve (see unit_mode_response's docstring), this is exactly
# the response to a pure dilation, with no preferred x/y direction -- matching the ring pattern's
# own rotational symmetry, unlike a uniaxial pull.
_Dgt1 = C.PH.Fk[1].T @ C.PH.Fk[1] - np.eye(2)
LOAD_ISO = LOAD + C.MO.vec3(_Dgt1) / C.PH.DELTA


def independent_check(geo):
    """Non-autograd ground truth for the SAME xx unit mode, via _common.unit_mode_response -- a
    genuinely separate NumPy PBC simulation, not the differentiable forward() path. Returns
    (eps_vec3, sig_vec3), each (nt,3)."""
    eps_ref, sig_ref = C.unit_mode_response(geo)
    return C.MO.vec3(eps_ref[0]), C.MO.vec3(sig_ref[0])


def independent_check_iso(geo):
    """Non-autograd ground truth for the isotropic (xx+yy) load -- same linearity argument as
    LOAD_ISO, applied to the simulated per-mode fields instead of the applied Delta_g."""
    eps_ref, sig_ref = C.unit_mode_response(geo)
    return C.MO.vec3(eps_ref[0] + eps_ref[1]), C.MO.vec3(sig_ref[0] + sig_ref[1])


def mag3(v):
    """Voigt-style magnitude of a per-triangle vec3=[xx,xy,yy] field (matches C.tensor_mag's
    (nt,2,2) convention: off-diagonal counted twice)."""
    return np.sqrt(v[:, 0] ** 2 + 2 * v[:, 1] ** 2 + v[:, 2] ** 2)


def plot_demo(geo, patch, spec, field_full, target, achieved_diff, achieved_ind, label, path, title):
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(13, 5.6))
    m = mag3(field_full)
    pc = C.fill_local_map(a0, geo, m, cmap='magma')
    pc.set_clim(0, np.nanpercentile(m, 97))
    C.draw_box(a0, geo); C.mark_region(a0, spec)
    plt.colorbar(pc, ax=a0, fraction=0.046)
    a0.set_title(f'local ‖{label}‖ (region marked)', fontsize=11)

    comps = ['xx', 'xy', 'yy']; x = np.arange(3); w = 0.25
    a1.bar(x - w, target, w, label='target', color='#2c3e50')
    a1.bar(x, achieved_diff, w, label='achieved (diff-path)', color='#1f77b4')
    a1.bar(x + w, achieved_ind, w, label='achieved (INDEPENDENT sim)', color='#d62728')
    a1.axhline(0, color='k', lw=.5)
    a1.set_xticks(x); a1.set_xticklabels([f'{label}_{c}' for c in comps])
    a1.set_title('region-mean target vs achieved (two independent checks)', fontsize=11)
    a1.legend(fontsize=8); a1.grid(alpha=.3, axis='y')
    fig.suptitle(title, fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f'saved {os.path.basename(path)}')


def plot_bulge_nu(geo, C6_per, spec, nu_patch_ind, nu_bg_ind, path):
    """Local nu map -- the design intent made directly visible: an AUXETIC patch (deep blue, nu<0)
    embedded in a flat nu=0 background (white). Both achieved values (INDEPENDENT sim) in the title."""
    nu_local = C.local_field_smooth(geo, C6_per, quantity='nu')
    fig, ax = plt.subplots(figsize=(7.5, 7))
    pc = C.fill_local_map(ax, geo, nu_local, cmap='RdBu_r', sym=True, vlim=0.5)
    C.draw_box(ax, geo); C.mark_region(ax, spec)
    plt.colorbar(pc, ax=ax, fraction=0.046)
    ax.set_title(f'{CASE} — local ν: AUXETIC patch (ν<0) in a flat ν=0 background\n'
                 f'achieved ν (INDEPENDENT sim): patch = {nu_patch_ind:+.3f}, background = {nu_bg_ind:+.3f}',
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f'saved {os.path.basename(path)}')


def run_stress_concentrator(csv_rows):
    prob, geo = C.make_case(TOPO, N)
    Lx, Ly = C.box(geo)
    spec = {'kind': 'circle', 'center': (0.5 * Lx, 0.5 * Ly), 'radius': 0.18 * Lx, 'color': 'lime'}
    patch, _ = C.region_shape(prob, spec)
    target = torch.tensor([0.35, 0.0, 0.0])                       # amplified sigma_xx, no shear/yy
    objs = [C.Objective('stress', target=target, region=patch, load=LOAD, weight=1.0)]
    r = C.optimize(prob, objs, mode='k', n_iter=NITER, reg=REG, verbose=False)
    C.apply_k_to_geo(geo, r['k'])
    rep = C.validate(prob, r['k'], None, objs)[0]

    _, sig_ind_full = independent_check(geo)
    sig_ind_patch = sig_ind_full[patch].mean(0)
    err_ind = float(np.abs(sig_ind_patch - target.numpy()).max())
    print(f"  [stress] target={target.numpy()}  achieved(diff-path)={rep['achieved']} err={rep['err']:.4f}  "
          f"achieved(INDEPENDENT sim)={sig_ind_patch} err={err_ind:.4f}", flush=True)
    csv_rows.append(('stress_concentrator', *target.numpy(), *rep['achieved'], *sig_ind_patch,
                     f'{rep["err"]:.4f}', f'{err_ind:.4f}'))

    C6 = C.sim_per_triangle_C6(geo)
    C.save_network(os.path.join(C.savedir(CASE), 'stress_concentrator.npz'), geo, r['k'], C6,
                   region=spec, target=target.numpy().tolist())
    plot_demo(geo, patch, spec, sig_ind_full, target.numpy(), rep['achieved'], sig_ind_patch, 'sigma',
             os.path.join(C.savedir(CASE), 'strain_stress_stress.png'),
             f'{CASE} — STRESS CONCENTRATOR: designed σ_xx amplified in the patch (pull along x)')


def run_strain_shield(csv_rows):
    prob, geo = C.make_case(TOPO, N)
    Lx, Ly = C.box(geo)
    spec = {'kind': 'circle', 'center': (0.5 * Lx, 0.5 * Ly), 'radius': 0.16 * Lx, 'color': 'cyan'}
    patch, _ = C.region_shape(prob, spec)
    target = torch.tensor([0.02, 0.0, 0.02])                       # near-zero local strain (rigid inclusion)
    objs = [C.Objective('strain', target=target, region=patch, load=LOAD, weight=1.0)]
    r = C.optimize(prob, objs, mode='k', n_iter=NITER, reg=REG, verbose=False)
    C.apply_k_to_geo(geo, r['k'])
    rep = C.validate(prob, r['k'], None, objs)[0]

    eps_ind_full, _ = independent_check(geo)
    eps_ind_patch = eps_ind_full[patch].mean(0)
    err_ind = float(np.abs(eps_ind_patch - target.numpy()).max())
    print(f"  [strain] target={target.numpy()}  achieved(diff-path)={rep['achieved']} err={rep['err']:.4f}  "
          f"achieved(INDEPENDENT sim)={eps_ind_patch} err={err_ind:.4f}", flush=True)
    csv_rows.append(('strain_shield', *target.numpy(), *rep['achieved'], *eps_ind_patch,
                     f'{rep["err"]:.4f}', f'{err_ind:.4f}'))

    C6 = C.sim_per_triangle_C6(geo)
    C.save_network(os.path.join(C.savedir(CASE), 'strain_shield.npz'), geo, r['k'], C6,
                   region=spec, target=target.numpy().tolist())
    plot_demo(geo, patch, spec, eps_ind_full, target.numpy(), rep['achieved'], eps_ind_patch, 'epsilon',
             os.path.join(C.savedir(CASE), 'strain_stress_strain.png'),
             f'{CASE} — STRAIN SHIELD: designed near-zero local strain in the patch (pull along x)')


def strain_field(geo, u):
    """Per-triangle exx, eyy from a displacement field -- the same small idiom used identically in
    two_region/ribbon.py and dir_aux_ribbon/three_ribbon.py (each verification script owns this
    tiny helper rather than importing a sibling directory's module)."""
    tv = np.asarray(geo['tri_verts']); p0, p1, p2 = tv[:, 0], tv[:, 1], tv[:, 2]
    ev = np.stack([p1 - p0, p2 - p0, p2 - p1], 1)
    eps = C.MO.tri_metric_change(ev, np.asarray(geo['simplices']), np.eye(2), u)
    return eps[:, 0, 0], eps[:, 1, 1]


def coarse2d(geo, val, nwt, ncell=16):
    """Area-weighted 2-D block-average of a per-triangle scalar over the open (nwt) domain -- kills
    the floppy-bond outlier noise that otherwise makes a raw per-triangle deformed-shape plot look
    folded/wrinkled rather than a legible smooth field (same recipe as two_region/response_fields.py
    and dir_aux_ribbon's profile())."""
    cen = np.asarray(geo['centroids']); ar = np.asarray(geo['areas'])
    frac = C.to_square(geo, cen)
    ix = np.clip((frac[:, 0] * ncell).astype(int), 0, ncell - 1)
    iy = np.clip((frac[:, 1] * ncell).astype(int), 0, ncell - 1)
    b = ix * ncell + iy; out = np.full(len(val), np.nan)
    for bb in np.unique(b[nwt]):
        m = (b == bb) & nwt
        out[m] = (val[m] * ar[m]).sum() / ar[m].sum()
    return out


def plot_bulge(geo, u, nwt, spec, path, scale=2.0, applied_label=None):
    """Deformed shape under the ACTUAL open x-stretch (not the periodic homogenised check), coloured
    by COARSE-GRAINED lateral strain eyy (raw per-triangle eyy makes the plot look folded/wrinkled --
    see coarse2d) -- the ribbon.py/dir_aux_ribbon 'bulge' tell: positive eyy = local lateral
    expansion. Region outline drawn at its (undeformed) design position for reference.
    `applied_label` overrides the default 'deform x{scale}' title phrase -- use it when `u` itself
    already carries a large applied strain (see run_strain_bulge's *_large call) so `scale` can stay
    at 1 and the title doesn't misleadingly say the plot is visually exaggerated."""
    _, eyy = strain_field(geo, u)
    eyy_c = coarse2d(geo, eyy, nwt)
    tv = np.asarray(geo['tri_verts'])[nwt]; sx = np.asarray(geo['simplices'])[nwt]
    dtv = tv + scale * u[sx]
    Lx, Ly = C.box(geo); mg = 0.12 * max(Lx, Ly)
    v = np.nanpercentile(np.abs(eyy_c[nwt]), 95)
    cols = plt.cm.RdBu_r(0.5 + 0.5 * np.clip(np.nan_to_num(eyy_c[nwt]) / v, -1, 1))
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.add_collection(PolyCollection(list(tv), facecolors='none', edgecolors='0.85', lw=0.15))
    ax.add_collection(PolyCollection(list(dtv), facecolors=cols, edgecolors='0.5', lw=0.1, alpha=0.9))
    C.mark_region(ax, spec)
    ax.set_xlim(-mg, Lx + mg); ax.set_ylim(-mg, Ly + mg); ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    label = applied_label if applied_label is not None else f'deform ×{scale:.0f}, coarse-grained'
    ax.set_title(f'Actual open x-stretch ({label}) — colour = lateral '
                 'strain εyy [red = expands]\nAuxetic inclusion (lime, top) — the auxetic patch '
                 'bulges laterally where the ν=0 surround stays neutral', fontsize=11)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f'saved {os.path.basename(path)}')


def run_strain_bulge(csv_rows):
    """AUXETIC-INCLUSION bulge: an off-centre top patch designed as a strongly AUXETIC material
    (nu<0) embedded in a nu=0 background. Under a real x-pull an auxetic inclusion expands LATERALLY
    (that IS what nu<0 means), so it bulges out where a nu=0 surround stays neutral -- and unlike a
    raw strain target, nu is a clean STABLE material property, so the bulge shows up in the actual
    OPEN cut-and-stretch test WITHOUT a near-mechanism. (History: a low-sigma_xx 'stress' target
    under-constrained the deformation SIGN; a raw 'strain' eyy target pinned the sign but a STRONG one
    -- eyy~2.5, a 250% response -- physically requires a near-mechanism and rendered as ugly floppy
    streaks. Designing the patch as an auxetic MATERIAL and the background as nu=0 is the clean way to
    get a real, mechanism-free bulge; the strength is then set by how auxetic a REGULAR lattice can
    cleanly go, ~-0.4, per the auxetic_sweep finding.) homogeneity on both objectives keeps each
    region's response uniform rather than concentrated in a few floppy bonds."""
    prob, geo = C.make_case('regular', N)
    Lx, Ly = C.box(geo)
    spec = {'kind': 'rect', 'center': (0.5 * Lx, 0.91 * Ly), 'w': 0.5 * Lx, 'h': 0.16 * Ly, 'color': 'lime'}
    patch, _ = C.region_shape(prob, spec)
    out = np.setdiff1d(np.arange(prob.n_tri), patch)
    NU_PATCH, NU_BG = -0.4, 0.0                                      # auxetic inclusion in a flat nu=0 matrix
    objs = [C.Objective('nu', target=NU_PATCH, region=patch, weight=2.0, homogeneity=0.5),
            C.Objective('nu', target=NU_BG, region=out, weight=1.0, homogeneity=1.0)]
    r = C.optimize(prob, objs, mode='k', n_iter=NITER, reg=REG, verbose=False)
    C.apply_k_to_geo(geo, r['k'])
    rep_patch, rep_bg = C.validate(prob, r['k'], None, objs)

    C6 = C.sim_per_triangle_C6(geo)                                  # ONE relaxation, reused below
    nu_ind_patch, _ = C.c6_nuE(C.region_phys_C6(geo, C6, patch))     # INDEPENDENT sim checks (both regions)
    nu_ind_out, _ = C.c6_nuE(C.region_phys_C6(geo, C6, out))
    print(f"  [bulge] AUXETIC patch nu target={NU_PATCH:+.2f}: achieved(diff-path)={rep_patch['achieved']:+.3f} "
          f"achieved(INDEPENDENT sim)={nu_ind_patch:+.3f}  |  background nu=0: "
          f"achieved(diff-path)={rep_bg['achieved']:+.3f} achieved(INDEPENDENT sim)={nu_ind_out:+.3f}", flush=True)
    csv_rows.append(('bulge_patch_nu', f'{NU_PATCH:+.2f}', f'{rep_patch["achieved"]:+.4f}', f'{nu_ind_patch:+.4f}'))
    csv_rows.append(('bulge_background_nu', f'{NU_BG:+.2f}', f'{rep_bg["achieved"]:+.4f}', f'{nu_ind_out:+.4f}'))

    C.save_network(os.path.join(C.savedir(CASE), 'strain_bulge.npz'), geo, r['k'], C6,
                   region=spec, nu_patch_target=NU_PATCH, nu_bg_target=NU_BG,
                   nu_patch_achieved_independent=float(nu_ind_patch),
                   nu_bg_achieved_independent=float(nu_ind_out))
    plot_bulge_nu(geo, C6, spec, nu_ind_patch, nu_ind_out,
                  os.path.join(C.savedir(CASE), 'strain_stress_bulge_nu.png'))

    # the actual physical pull test: open-boundary cut-and-stretch, not the periodic homogenised check
    u, nwt = C.open_stretch(geo, axis=0, regularize=True)
    plot_bulge(geo, u, nwt, spec, os.path.join(C.savedir(CASE), 'strain_stress_bulge_stretch.png'))

    # LARGE applied strain: open_stretch solves ONE linear spring-truss problem per its own docstring
    # ("linear -> any other stretch is this scaled"), so the large-strain response is exactly u
    # rescaled -- no new solve needed. BIG_STRAIN is the applied engineering strain along x (the base
    # open_stretch call above applies only ~1/Lx, a few percent).
    BIG_STRAIN = 0.30
    eps0 = 1.0 / Lx                                        # engineering strain at the base unit-disp BC
    u_big = u * (BIG_STRAIN / eps0)
    plot_bulge(geo, u_big, nwt, spec, os.path.join(C.savedir(CASE), 'strain_stress_bulge_stretch_large.png'),
              scale=1.0, applied_label=f'applied engineering strain ~{BIG_STRAIN:.0%} along x, '
              'linear extrapolation, undeformed plot scale')


def plot_rings(geo, specs, sig_ind_full, achieved_rows, magnitude, topo, path):
    """Local MEAN stress p=(sigma_xx+sigma_yy)/2 map (diverging colormap, so the alternating sign is
    directly visible) + per-ring target/achieved bar chart. Mean stress, not sigma_xx alone, is the
    right field to show here: under the isotropic load and isotropic (sigma_xx=sigma_yy) targets,
    p is exactly the designed quantity."""
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(14, 5.8))
    p = 0.5 * (sig_ind_full[:, 0] + sig_ind_full[:, 2])
    pc = C.fill_local_map(a0, geo, p, cmap='RdBu_r', sym=True, vlim=magnitude * 1.3)
    C.draw_box(a0, geo)
    for s in specs:
        C.mark_region(a0, s)
    plt.colorbar(pc, ax=a0, fraction=0.046)
    a0.set_title(f'local mean stress p=(σ_xx+σ_yy)/2 ({topo}, alternating-sign rings)', fontsize=11)

    names = [row[0] for row in achieved_rows]; x = np.arange(len(names)); w = 0.25
    tgt = [row[1][0] for row in achieved_rows]
    ach_d = [0.5 * (row[2][0] + row[2][2]) for row in achieved_rows]
    ach_i = [0.5 * (row[3][0] + row[3][2]) for row in achieved_rows]
    a1.bar(x - w, tgt, w, label='target', color='#2c3e50')
    a1.bar(x, ach_d, w, label='achieved (diff-path)', color='#1f77b4')
    a1.bar(x + w, ach_i, w, label='achieved (INDEPENDENT sim)', color='#d62728')
    a1.axhline(0, color='k', lw=.5)
    a1.set_xticks(x); a1.set_xticklabels(names)
    a1.set_ylabel('p = (σ_xx+σ_yy)/2'); a1.legend(fontsize=8); a1.grid(alpha=.3, axis='y')
    a1.set_title('per-ring target vs achieved', fontsize=11)
    fig.suptitle(f'{CASE} — CONCENTRIC RINGS bullseye ({topo}, magnitude={magnitude:+.2f}): '
                 f'alternating-sign mean stress (isotropic stretch)', fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f'saved {os.path.basename(path)}')


def run_concentric_rings(csv_rows, topo, magnitude, tag):
    """Alternating-sign concentric-ring MEAN-STRESS pattern (a stress 'bullseye'): center disc +A,
    middle ring -A, outer ring +A, under an ISOTROPIC (equi-biaxial) stretch load -- not the
    uniaxial pull the other demos use. A uniaxial load has no rotational symmetry, which fights a
    rotationally-symmetric ring target (the middle ring, sandwiched between two same-sign
    neighbours, could not reach its target under uniaxial load -- see the earlier failed attempt).
    Under the isotropic load, targeting isotropic stress (sigma_xx=sigma_yy=A) matches the pattern's
    own symmetry. Called for both 'regular' (A~0.35) and 'large' (A~1.0) magnitude levels, on both
    regular and disorder_hi topologies -- 4 designs total, each with 3 SIMULTANEOUS stress
    objectives (one per ring)."""
    prob, geo = C.make_case(topo, N)
    Lx, Ly = C.box(geo)
    cx, cy = 0.5 * Lx, 0.5 * Ly
    ring_defs = [
        ('center', {'kind': 'circle', 'center': (cx, cy), 'radius': 0.12 * Lx, 'color': 'lime'}, magnitude),
        ('ring1', {'kind': 'ring', 'center': (cx, cy), 'r_in': 0.12 * Lx, 'r_out': 0.22 * Lx, 'color': 'cyan'},
        -magnitude),
        ('ring2', {'kind': 'ring', 'center': (cx, cy), 'r_in': 0.22 * Lx, 'r_out': 0.32 * Lx, 'color': 'orange'},
        magnitude),
    ]
    regions = []; objs = []
    for name, spec, s in ring_defs:
        idx, _ = C.region_shape(prob, spec)
        regions.append((name, spec, idx))
        objs.append(C.Objective('stress', target=torch.tensor([s, 0.0, s]), region=idx, load=LOAD_ISO,
                                weight=1.0))
    r = C.optimize(prob, objs, mode='k', n_iter=NITER, reg=REG, verbose=False)
    C.apply_k_to_geo(geo, r['k'])
    rep = C.validate(prob, r['k'], None, objs)

    _, sig_ind_full = independent_check_iso(geo)
    print(f"  [rings {tag}] topo={topo} magnitude={magnitude:+.2f}", flush=True)
    achieved_rows = []
    for (name, spec, idx), ob_rep in zip(regions, rep):
        sig_ind_region = sig_ind_full[idx].mean(0)
        err_ind = float(np.abs(sig_ind_region - ob_rep['target']).max())
        print(f"    {name:8s} target p={ob_rep['target'][0]:+.3f} (isotropic)  achieved(diff-path)={ob_rep['achieved']}  "
              f"achieved(INDEPENDENT sim)={sig_ind_region}  err_ind={err_ind:.4f}", flush=True)
        csv_rows.append((tag, name, *ob_rep['target'], *ob_rep['achieved'], *sig_ind_region,
                         f'{ob_rep["err"]:.4f}', f'{err_ind:.4f}'))
        achieved_rows.append((name, ob_rep['target'], ob_rep['achieved'], sig_ind_region))

    C6 = C.sim_per_triangle_C6(geo)
    C.save_network(os.path.join(C.savedir(CASE), f'rings_{tag}.npz'), geo, r['k'], C6,
                   regions=[s for _, s, _ in ring_defs], magnitude=float(magnitude))
    plot_rings(geo, [s for _, s, _ in ring_defs], sig_ind_full, achieved_rows, magnitude, topo,
              os.path.join(C.savedir(CASE), f'strain_stress_rings_{tag}.png'))


def main():
    csv_rows = []
    run_stress_concentrator(csv_rows)
    run_strain_shield(csv_rows)
    C.write_csv(os.path.join(C.savedir(CASE), f'{CASE}.csv'),
                ['demo', 'target_xx', 'target_xy', 'target_yy',
                 'achieved_diffpath_xx', 'achieved_diffpath_xy', 'achieved_diffpath_yy',
                 'achieved_independent_xx', 'achieved_independent_xy', 'achieved_independent_yy',
                 'err_diffpath', 'err_independent'], csv_rows)

    bulge_rows = []                                                  # nu-based (own schema, not vec3)
    run_strain_bulge(bulge_rows)
    C.write_csv(os.path.join(C.savedir(CASE), 'bulge.csv'),
                ['region', 'target_nu', 'achieved_nu_diffpath', 'achieved_nu_independent'], bulge_rows)

    rings_rows = []
    for topo in ('regular', 'disorder_hi'):
        for magnitude, mtag in ((0.35, 'regular'), (1.0, 'large')):
            run_concentric_rings(rings_rows, topo, magnitude, f'{topo}_{mtag}')
    C.write_csv(os.path.join(C.savedir(CASE), 'rings.csv'),
                ['tag', 'ring', 'target_xx', 'target_xy', 'target_yy',
                 'achieved_diffpath_xx', 'achieved_diffpath_xy', 'achieved_diffpath_yy',
                 'achieved_independent_xx', 'achieved_independent_xy', 'achieved_independent_yy',
                 'err_diffpath', 'err_independent'], rings_rows)
    print('done')


if __name__ == '__main__':
    main()
