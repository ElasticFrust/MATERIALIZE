"""
Case strain_stress — design objectives that target the ACTUAL per-triangle strain/stress response
(not just derived ν/E), via inverse_design.py's 'strain'/'stress' Objective kinds (see
[[phase3-strain-stress-objective-todo]]). Three demos, all under the SAME applied uniaxial macro load
(pull along x — the "xx" unit mode, in this framework's native metric-change convention):

  STRESS CONCENTRATOR : a sub-region designed to carry AMPLIFIED σ_xx relative to the background
                        matrix (a stress-concentrating stiff patch).
  STRAIN SHIELD        : a (different) sub-region designed to have NEAR-ZERO local strain under the
                        same load (a rigid, strain-shielded inclusion). Strain objectives only make
                        sense on a sub-region — whole-cell strain is degenerate (see the Objective
                        docstring in inverse_design.py: region-mean strain over the WHOLE cell equals
                        the applied load exactly).
  STRAIN BULGE         : a large rectangle offset toward the TOP of a REGULAR-topology patch (not
                        centred, not disordered) designed for a strongly POSITIVE local eyy (lateral
                        expansion) under the same load. A 'stress'-shielded (low sigma_xx) version of
                        this was tried first and DID produce a strong top-only asymmetry under the
                        real stretch test, but the deformation pattern came out mixed (mostly
                        contracting, not a clean bulge) — stress only constrains magnitude, not the
                        SIGN of the local strain, so it doesn't reliably pick out "expands". A direct
                        'strain' target pins the deformation itself. On top of the usual periodic
                        design verification, an actual OPEN-boundary cut-and-stretch test (matching
                        two_region/ribbon.py's convention) checks the real physical pull — distinct
                        from the other two demos, which only check the periodic homogenised response.

The first two designs are checked TWO independent ways: (a) validate()'s own differentiable-path
readback (self-consistency of the optimiser's own forward pass), and (b) INDEPENDENTLY via
_common.unit_mode_response's non-autograd NumPy PBC simulation (physical_homog.relax) — a genuinely
separate code path from the solver's own forward(), not just re-running it. Outputs:
strain_stress.csv, strain_stress_{stress,strain,bulge_design,bulge_stretch}.png, saved networks.
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
LOAD = C.CE.vec3(_Dgt0) / C.PH.DELTA


def independent_check(geo):
    """Non-autograd ground truth for the SAME xx unit mode, via _common.unit_mode_response -- a
    genuinely separate NumPy PBC simulation, not the differentiable forward() path. Returns
    (eps_vec3, sig_vec3), each (nt,3)."""
    eps_ref, sig_ref = C.unit_mode_response(geo)
    return C.CE.vec3(eps_ref[0]), C.CE.vec3(sig_ref[0])


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
    eps = C.CE.tri_metric_change(ev, np.asarray(geo['simplices']), np.eye(2), u)
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


def plot_bulge(geo, u, nwt, spec, path, scale=2.0):
    """Deformed shape under the ACTUAL open x-stretch (not the periodic homogenised check), coloured
    by COARSE-GRAINED lateral strain eyy (raw per-triangle eyy makes the plot look folded/wrinkled --
    see coarse2d) -- the ribbon.py/dir_aux_ribbon 'bulge' tell: positive eyy = local lateral
    expansion. Region outline drawn at its (undeformed) design position for reference."""
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
    ax.set_title(f'Actual open x-stretch (deform ×{scale:.0f}, coarse-grained) — colour = lateral '
                 'strain εyy [red = expands]\nOff-centre strain-bulge rectangle (lime, top) — '
                 'clean asymmetric lateral bulge?', fontsize=11)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f'saved {os.path.basename(path)}')


def run_strain_bulge(csv_rows):
    """A 'stress'-shielded (low sigma_xx) target was tried first here and DID produce a strong
    top-only asymmetry under the real stretch test, but the local deformation pattern came out mixed
    (mostly contracting, not a clean bulge) -- stress only constrains magnitude, not the SIGN of the
    local strain. A direct 'strain' target (strongly positive eyy = local lateral expansion) pins the
    deformation itself, so it reliably produces a clean visible bulge instead of an emergent one."""
    prob, geo = C.make_case('regular', N)
    Lx, Ly = C.box(geo)
    spec = {'kind': 'rect', 'center': (0.5 * Lx, 0.91 * Ly), 'w': 0.5 * Lx, 'h': 0.16 * Ly, 'color': 'lime'}
    patch, _ = C.region_shape(prob, spec)
    out = np.setdiff1d(np.arange(prob.n_tri), patch)
    target = torch.tensor([0.3, 0.0, 2.5])                          # strong local LATERAL EXPANSION
    objs = [C.Objective('strain', target=target, region=patch, load=LOAD, weight=1.0)]
    r = C.optimize(prob, objs, mode='k', n_iter=NITER, reg=REG, verbose=False)
    C.apply_k_to_geo(geo, r['k'])
    rep = C.validate(prob, r['k'], None, objs)[0]

    eps_ind_full, _ = independent_check(geo)
    eps_ind_patch = eps_ind_full[patch].mean(0); eps_ind_out = eps_ind_full[out].mean(0)
    err_ind = float(np.abs(eps_ind_patch - target.numpy()).max())
    print(f"  [bulge] target={target.numpy()}  achieved(diff-path)={rep['achieved']} err={rep['err']:.4f}  "
          f"achieved(INDEPENDENT sim)={eps_ind_patch} err={err_ind:.4f}  "
          f"(background outside patch: {eps_ind_out})", flush=True)
    csv_rows.append(('strain_bulge', *target.numpy(), *rep['achieved'], *eps_ind_patch,
                     f'{rep["err"]:.4f}', f'{err_ind:.4f}'))

    C6 = C.sim_per_triangle_C6(geo)
    C.save_network(os.path.join(C.savedir(CASE), 'strain_bulge.npz'), geo, r['k'], C6,
                   region=spec, target=target.numpy().tolist())
    plot_demo(geo, patch, spec, eps_ind_full, target.numpy(), rep['achieved'], eps_ind_patch, 'epsilon',
             os.path.join(C.savedir(CASE), 'strain_stress_bulge_design.png'),
             f'{CASE} — STRAIN BULGE (regular topology, off-centre rectangle): designed strong lateral '
             f'expansion in the patch (pull along x)')

    # the actual physical pull test: open-boundary cut-and-stretch, not the periodic homogenised check
    u, nwt = C.open_stretch(geo, axis=0, regularize=True)
    plot_bulge(geo, u, nwt, spec, os.path.join(C.savedir(CASE), 'strain_stress_bulge_stretch.png'))


def main():
    csv_rows = []
    run_stress_concentrator(csv_rows)
    run_strain_shield(csv_rows)
    run_strain_bulge(csv_rows)
    C.write_csv(os.path.join(C.savedir(CASE), f'{CASE}.csv'),
                ['demo', 'target_xx', 'target_xy', 'target_yy',
                 'achieved_diffpath_xx', 'achieved_diffpath_xy', 'achieved_diffpath_yy',
                 'achieved_independent_xx', 'achieved_independent_xy', 'achieved_independent_yy',
                 'err_diffpath', 'err_independent'], csv_rows)
    print('done')


if __name__ == '__main__':
    main()
