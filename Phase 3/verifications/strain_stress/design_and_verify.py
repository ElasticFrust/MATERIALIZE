"""
Case strain_stress — design objectives that target the ACTUAL per-triangle strain/stress response
(not just derived ν/E), via inverse_design.py's 'strain'/'stress' Objective kinds (see
[[phase3-strain-stress-objective-todo]]). Two demos, both under the SAME applied uniaxial macro load
(pull along x — the "xx" unit mode, in this framework's native metric-change convention):

  STRESS CONCENTRATOR : a sub-region designed to carry AMPLIFIED σ_xx relative to the background
                        matrix (a stress-concentrating stiff patch).
  STRAIN SHIELD        : a (different) sub-region designed to have NEAR-ZERO local strain under the
                        same load (a rigid, strain-shielded inclusion). Strain objectives only make
                        sense on a sub-region — whole-cell strain is degenerate (see the Objective
                        docstring in inverse_design.py: region-mean strain over the WHOLE cell equals
                        the applied load exactly).

Both designs are checked TWO independent ways: (a) validate()'s own differentiable-path readback
(self-consistency of the optimiser's own forward pass), and (b) INDEPENDENTLY via
_common.unit_mode_response's non-autograd NumPy PBC simulation (physical_homog.relax) — a genuinely
separate code path from the solver's own forward(), not just re-running it. Outputs:
strain_stress.csv, strain_stress_{stress,strain}.png, saved networks.
"""
import os, sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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


def main():
    csv_rows = []
    run_stress_concentrator(csv_rows)
    run_strain_shield(csv_rows)
    C.write_csv(os.path.join(C.savedir(CASE), f'{CASE}.csv'),
                ['demo', 'target_xx', 'target_xy', 'target_yy',
                 'achieved_diffpath_xx', 'achieved_diffpath_xy', 'achieved_diffpath_yy',
                 'achieved_independent_xx', 'achieved_independent_xy', 'achieved_independent_yy',
                 'err_diffpath', 'err_independent'], csv_rows)
    print('done')


if __name__ == '__main__':
    main()
