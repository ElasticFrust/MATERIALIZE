"""
Per-triangle C(s): the solver's contraction vs the INDEPENDENT energy-Hessian oracle — visualised.

`Phase 2/test_forward_solver.py` [8] reduces this comparison to one number per mesh. That number
says whether the gate passes; it does not show WHERE the two disagree, how the disagreement is
distributed over triangles, or how far it sits from what a real defect looks like. This script
produces that picture, across the same mesh cases [8] uses (crystal → disordered → VD contrast).

Three figures:
  (1) component-wise RESIDUAL, one panel per tensor component, all cases overlaid, against the
      gate's tolerance band. (A raw solver-vs-independent scatter was tried first and discarded:
      at ~2e-3 relative error every point sits on the diagonal and it shows nothing.)
  (2) error vs case — per-triangle and BULK on the same panel, so the penalty for not averaging is
      visible, together with the A-0 reference showing how far a real defect sits from the floor.
  (3) spatial field — the per-triangle relative difference next to ‖W(s)‖, on a make_lattice
      geometry (an axis-aligned periodic box, which draw_field needs).

**Reference case: the A-0 shear defect.** The historical bug (dropping the ½ on a shear input pair
when lifting W to 4 indices) is reproduced HERE, locally, purely to show what the same plots look
like when the contraction is wrong. The protected core is never touched: `_contract` reimplements
`_compute_actual_elastic_tensor` with a `half` switch and is ASSERTED to reproduce the real one
exactly when `half=True`, so the buggy curve is trustworthy as an illustration.

Run:  python per_triangle_C_comparison.py
Out:  verification_tools/plots/per_triangle_C/
"""
import os, sys, json

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, 'Phase 2'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'Phase 3', 'verifications'))
sys.path.insert(0, os.path.join(ROOT, 'Phase 3'))

import forward_solver_torch as fst
import physical_homog as PH
import sim_assembly as SA
import metric_ops as MO
import mesh_build as MB
import plotting as P
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)
OUTDIR = os.path.join(HERE, 'plots', 'per_triangle_C')

G_TO_DG = 16.0        # oracle (acts on g) → solver convention (acts on Δg); see test [8]
COMPS = ['C_xxxx', 'C_xxyy', 'C_xxxy', 'C_yyyy', 'C_yyxy', 'C_xyxy']   # of the [xx,yy,xy] matrix
IJ = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
CASES = [(8, 0.00, 0, None, 'crystal η=0 (W≡0)'), (8, 0.20, 1, None, 'η=0.20'),
         (8, 0.20, 1, 5, 'η=0.20, VD a=+5'), (8, 0.35, 2, None, 'η=0.35'),
         (12, 0.30, 3, 10, 'η=0.30, VD a=+10')]


def _contract(bare, W3, half=True):
    """`_compute_actual_elastic_tensor` re-expressed with a switch for the A-0 defect.

    `half=True` is the correct contraction and is asserted below to match the core bit-for-bit.
    `half=False` drops the ½ on a shear INPUT pair — the exact 2026-08 bug — and exists only so the
    figures can show what a wrong contraction looks like. The protected core is not modified."""
    N = bare.shape[0]
    a = torch.as_tensor(bare); w = torch.as_tensor(W3.reshape(-1, 9))
    e = torch.arange(2)
    I, J, K, L = torch.meshgrid(e, e, e, e, indexing='ij')
    a_idx = ((I + J) + (K + L)).reshape(-1)
    w_idx = (3 * (I + J) + (K + L)).reshape(-1)
    hf = (1.0 - 0.5 * (K != L).to(a.dtype)) if half else torch.ones_like((K != L).to(a.dtype))
    A4 = a[:, a_idx].reshape(N, 2, 2, 2, 2)
    W4 = w[:, w_idx].reshape(N, 2, 2, 2, 2) * hf
    I2 = torch.eye(2, dtype=a.dtype)
    Id = 0.5 * (torch.einsum('ik,jl->ijkl', I2, I2) + torch.einsum('il,jk->ijkl', I2, I2))
    T = Id + W4
    C = torch.einsum('tmnij,tmnpq,tpqkl->tijkl', T, A4, T)
    return torch.stack([C[:, 0, 0, 0, 0], C[:, 0, 0, 0, 1], C[:, 0, 0, 1, 1],
                        C[:, 1, 0, 0, 1], C[:, 1, 0, 1, 1], C[:, 1, 1, 1, 1]], 1).numpy()


def _as_mat(c6):
    """solver 6-vector → (…,3,3) Voigt [xx,yy,xy], the assembly used throughout."""
    return np.stack([np.stack([c6[:, 0], c6[:, 2], c6[:, 1]], -1),
                     np.stack([c6[:, 2], c6[:, 5], c6[:, 4]], -1),
                     np.stack([c6[:, 1], c6[:, 4], c6[:, 3]], -1)], -2)


def one_case(N, eta, seed, vd):
    """-> dict with the solver's per-triangle C(s) (correct and A-0-buggy), the independent C(s),
    the bulk pair, and |W| per triangle."""
    mesh = MB.build_geometry(N, eta, seed)
    if vd is None:
        mesh['bond_k'] = np.ones(len(mesh['bond_R']))
        mesh['tri_k'] = mesh['bond_k'][mesh['tri_bond']]
    else:
        MB.set_VD(mesh, vd)
    nn, nt = len(mesh['pts']), len(mesh['simplices'])
    free = np.arange(2, 2 * nn)

    Dgt = [F.T @ F - np.eye(2) for F in PH.Fk]
    Dinv = np.linalg.inv(np.stack([MO.vec3(g) for g in Dgt], 1))
    u_modes = PH.relax(mesh, free, SA.assemble_K_faff)            # the SIM's relaxation
    D = np.zeros((nt, 3, 3))
    for j, (F, u) in enumerate(zip(PH.Fk, u_modes)):
        D[:, :, j] = MO.vec3(MO.tri_metric_change(mesh['edge_vecs'], mesh['simplices'], F, u) - Dgt[j])
    W3 = D @ Dinv
    bare = MO.bare_tensor(mesh)

    c6_ok = _contract(bare, W3, half=True)
    ref = fst._compute_actual_elastic_tensor(torch.as_tensor(bare),
                                             torch.as_tensor(W3.reshape(-1, 9))).numpy()
    assert np.allclose(c6_ok, ref, rtol=0, atol=0), \
        "local _contract(half=True) does not reproduce the core bit-for-bit — figures untrustworthy"
    c6_bug = _contract(bare, W3, half=False)

    # keys plotting.draw_field needs; build_geometry meshes carry neither, but both follow from
    # the unwrapped edge_vecs (image-correct by construction) and the lattice vectors.
    p0 = mesh['pts'][mesh['simplices'][:, 0]]
    mesh['tri_verts'] = np.stack([p0, p0 + mesh['edge_vecs'][:, 0], p0 + mesh['edge_vecs'][:, 1]], 1)
    mesh['BL1'] = N * np.array([1.0, 0.0]); mesh['BL2'] = N * np.array([0.5, np.sqrt(3) / 2])

    C_ind = PH.energy_C_per_triangle(mesh, free, SA.assemble_K_faff) / G_TO_DG
    bulk_solver = _as_mat(c6_ok).mean(0) * (8.0 * nt / mesh['areas'].sum())
    bulk_ind = PH.energy_C(mesh, free, SA.assemble_K_faff)
    return dict(mesh=mesh, ind=C_ind, ok=_as_mat(c6_ok), bug=_as_mat(c6_bug),
                Wmag=np.linalg.norm(W3.reshape(nt, 9), axis=1),
                bulk_solver=bulk_solver, bulk_ind=bulk_ind, nt=nt)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    res = {}
    for N, eta, seed, vd, lab in CASES:
        res[lab] = one_case(N, eta, seed, vd)
        r = res[lab]
        print(f"  {lab:<22} {r['nt']:>4} tri   per-tri {np.abs(r['ok']-r['ind']).max()/np.abs(r['ind']).max():.2e}"
              f"   (A-0 bug would give {np.abs(r['bug']-r['ind']).max()/np.abs(r['ind']).max():.2e})",
              flush=True)

    cm = plt.get_cmap('viridis')
    cols = {lab: cm(i / max(1, len(CASES) - 1)) for i, (*_, lab) in enumerate(CASES)}

    # ---- (1) component-wise RESIDUAL --------------------------------------------------------
    # A raw solver-vs-independent scatter is useless here: at ~2e-3 relative error every point sits
    # on the diagonal and the plot shows nothing. Plot the RESIDUAL against the value instead, which
    # is where the structure is. Scale = max|C| of that case, matching how the gate normalises.
    fig, axes = plt.subplots(2, 3, figsize=(P.STYLE.PANEL[0] * 3, P.STYLE.PANEL[1] * 2), squeeze=False)
    for c, (name, (i, j)) in enumerate(zip(COMPS, IJ)):
        ax = axes[c // 3][c % 3]
        for lab in res:
            sc = np.abs(res[lab]['ind']).max()
            x = res[lab]['ind'][:, i, j] / sc
            d = (res[lab]['ok'][:, i, j] - res[lab]['ind'][:, i, j]) / sc
            ax.plot(x, d, '.', ms=3.5, alpha=0.6, color=cols[lab], label=lab)
        ax.axhline(0, color='k', lw=1.0, ls='--', zorder=0)
        ax.axhspan(-2e-2, 2e-2, color='0.85', zorder=-1)
        ax.set_title(name, fontsize=10); ax.grid(alpha=0.25)
        ax.set_xlabel('C(s) / max|C|   (independent)')
        ax.set_ylabel('(solver − independent) / max|C|')
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=len(CASES), fontsize=9, frameon=False,
               markerscale=4, bbox_to_anchor=(0.5, -0.012))
    fig.suptitle('Per-triangle C(s): RESIDUAL vs the independent energy Hessian '
                 '(grey band = gate [8] tolerance ±2e-2)', fontsize=12)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    P.save_fig(fig, os.path.join(OUTDIR, 'per_triangle_residual.png'))

    # ---- (2) error vs case: per-triangle and bulk on ONE panel, with the A-0 reference --------
    labs = list(res)
    x = np.arange(len(labs))
    per = [np.abs(res[l]['ok'] - res[l]['ind']).max() / np.abs(res[l]['ind']).max() for l in labs]
    bulk = [np.abs(res[l]['bulk_solver'] - res[l]['bulk_ind']).max() / np.abs(res[l]['bulk_ind']).max()
            for l in labs]
    bug = [np.abs(res[l]['bug'] - res[l]['ind']).max() / np.abs(res[l]['ind']).max() for l in labs]
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.semilogy(x, bug, 's--', color='tab:red', lw=2.0, ms=7,
                label='per-triangle WITH the A-0 shear defect (reference)')
    ax.semilogy(x, per, 'o-', color='tab:blue', lw=2.0, ms=7, label='per-triangle, correct')
    ax.semilogy(x, bulk, '^-', color='tab:green', lw=2.0, ms=7, label='bulk, correct (= test [7])')
    ax.axhline(2e-2, color='k', ls=':', lw=1.4)
    ax.text(0.02, 2.3e-2, 'gate [8] tolerance 2e-2', fontsize=9, transform=ax.get_yaxis_transform())
    ax.set_xticks(x); ax.set_xticklabels(labs, rotation=18, ha='right', fontsize=9)
    ax.set_ylabel('max relative difference vs the independent oracle')
    ax.grid(alpha=0.25, which='both'); ax.legend(fontsize=9, loc='center left')
    fig.suptitle('Agreement per case — and how far a real defect sits from it', fontsize=12)
    fig.tight_layout()
    P.save_fig(fig, os.path.join(OUTDIR, 'error_by_case.png'))

    # ---- (3) spatial field ------------------------------------------------------------------
    # Built on a make_lattice geometry, NOT build_geometry: draw_field crops to an axis-aligned box
    # (BL1=(Lx,0), BL2=(0,Ly)), which a rhombic torus does not have — rendering it there produces a
    # cut wedge rather than the periodic cell.
    import _common as _C
    fg = _C.make_lattice(1.0, 1.0, half=7.0, eta=0.30, seed=3)
    _C.apply_k_to_geo(fg, torch.ones(len(fg['bond_u'])))
    fg['bond_k'] = 1.0 + np.tanh(10.0 * (np.sqrt((fg['bond_R'] ** 2).sum(1)) - 1.0))
    fg['tri_k'] = fg['bond_k'][fg['tri_bond']]
    nn_f, nt_f = len(fg['pts']), len(fg['simplices'])
    free_f = np.arange(2, 2 * nn_f)
    Dgt = [F.T @ F - np.eye(2) for F in PH.Fk]
    Dinv = np.linalg.inv(np.stack([MO.vec3(g) for g in Dgt], 1))
    um = PH.relax(fg, free_f, SA.assemble_K_faff)
    Df = np.zeros((nt_f, 3, 3))
    for j, (F, u) in enumerate(zip(PH.Fk, um)):
        Df[:, :, j] = MO.vec3(MO.tri_metric_change(fg['edge_vecs'], fg['simplices'], F, u) - Dgt[j])
    W3f = Df @ Dinv
    ok_f = _as_mat(_contract(MO.bare_tensor(fg), W3f, half=True))
    ind_f = PH.energy_C_per_triangle(fg, free_f, SA.assemble_K_faff) / G_TO_DG
    rel = np.abs(ok_f - ind_f).reshape(nt_f, 9).max(1) / np.abs(ind_f).max()
    Wf = np.linalg.norm(W3f.reshape(nt_f, 9), axis=1)
    corr = float(np.corrcoef(rel, Wf)[0, 1])

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.0))
    for ax, val, ttl, cmap, lb in [(axes[0], rel, 'per-triangle |ΔC| / max|C|', 'magma',
                                    r'$|\Delta C|/\max|C|$'),
                                   (axes[1], Wf, '‖W(s)‖  (non-affine response)', 'viridis',
                                    r'$\|W(s)\|$')]:
        P.draw_field(ax, fg, val, kind='E', cmap=cmap, title=ttl, cbar_label=lb)
    fig.suptitle(f'Where solver and oracle differ — disordered η=0.30, VD a=+10.  '
                 f'corr(|ΔC|, ‖W‖) = {corr:+.2f}: related, but only weakly — the residual is '
                 f'not simply proportional to ‖W‖', fontsize=10.5)
    fig.tight_layout()
    P.save_fig(fig, os.path.join(OUTDIR, 'residual_field.png'))
    out = dict(cases=labs, per_triangle=per, bulk=bulk, with_A0_bug=bug,
               ratio_bug_to_correct=[b / p for b, p in zip(bug, per)],
               residual_vs_Wmag_corr=corr, G_TO_DG=G_TO_DG)
    with open(os.path.join(OUTDIR, 'per_triangle_C.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=1)
    print(f"\n  corr(|ΔC|, ‖W‖) on {lab} = {corr:+.3f}")
    print(f"  wrote {OUTDIR}")


if __name__ == '__main__':
    main()
