"""
How well does the solver do on DISORDERED networks? — tensor-level accuracy vs η.

The crystal is a degenerate case: W ≡ 0, so C(s) = A(s) and every route agrees to round-off. It
therefore says nothing about the homogenisation (`CLAUDE.md` §3, "the crystal gate is blind").
Disorder is where the non-affine response W is real and where the machinery is actually tested.
`test_forward_solver` [7]/[8] check only 2–3 η values; this resolves the whole range.

Four quantities per (η, seed), all at the TENSOR level (component-wise, never reduced ν,E — so
this is immune to the open A-10 convention question):

  (a) virial_C vs energy_C — a CONTROL. Two independent reductions of the SAME relaxed field
      (macroscopic stress vs energy Hessian), sharing no code with the solver. Flat round-off is
      the expected answer; anything else means the oracle itself is drifting with disorder.
  (b) solver BULK C_eff vs the independent oracle          — the quantity gated by [7].
  (c) solver PER-TRIANGLE C(s) vs the independent oracle   — the quantity gated by [8].
  (d) the same per-triangle comparison WITH the A-0 shear defect — the reference for what a real
      defect looks like, so (c) can be read against something rather than in the abstract.

NINE families, so the answer is not read off one kind of network. Rigidity contrast drives W
independently of geometry, and the two disorder INTENTS (CLAUDE.md §3) are physically different, so
both appear:

  - frozen-connectivity magnitude-η, uniform k — the canonical disorder (`mesh_build.build_geometry`:
    fixed topology, every node displaced by exactly η; this is the auxetic-band family);
  - the same geometry under five VD contrasts k = 1 + tanh(a(|R|−1)), a ∈ {−10, −2, +5, +10, +100};
  - the same geometry with RANDOM BINARY k (stiff/soft, ratio 10) — rigidity disorder uncorrelated
    with the geometry, which the VD families are not;
  - RE-TRIANGULATED η (`_common.make_lattice`: perturb then periodic-Delaunay) — the other disorder
    intent, a topology scan rather than a distortion;
  - an ANISOTROPIC base lattice (ψ=0.6) plus η, so the crystal it starts from is not isotropic.

Near-singular realisations are skipped, not fatal: at large η the perturbed lattice produces sliver
triangles and `physical_homog` refuses them (`UnhealthyGeometryError`). The surviving-seed count is
recorded and plotted — at large η the average is over a shrinking, non-random subset, and any claim
about that end must say so.

Run:  python accuracy_vs_disorder.py [--N 10] [--seeds 5] [--eta-step 0.02]
Out:  verification_tools/plots/accuracy_vs_disorder/
"""
import os, sys, json, time, argparse

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
# reuse rather than re-derive: the validated contraction (with the A-0 switch) and the 6→3×3 map
from per_triangle_C_comparison import _contract, _as_mat, G_TO_DG

torch.set_default_dtype(torch.float64)
OUTDIR = os.path.join(HERE, 'plots', 'accuracy_vs_disorder')
KEYS = ['oracle_control', 'bulk', 'per_tri', 'per_tri_A0']
EXTRA = ['Wmag', 'E_bulk']   # diagnostics, not agreements


def build(kind, N, eta, seed):
    """Geometry + stiffness for one family. `kind` selects the network family (see module docstring)."""
    if kind[0] == 'retri':                       # perturb THEN periodic-Delaunay (topology scan)
        import _common as _C
        mesh = _C.make_lattice(1.0, 1.0, half=float(N) / 2 + 2, eta=eta, seed=seed)
    elif kind[0] == 'aniso':                     # anisotropic base crystal + eta
        import _common as _C
        mesh = _C.make_lattice(1.0, 0.6, half=float(N) / 2 + 2, eta=eta, seed=seed)
    else:                                        # frozen connectivity, magnitude-eta (canonical)
        mesh = MB.build_geometry(N, eta, seed)

    if kind[0] == 'vd':
        MB.set_VD(mesh, kind[1])
    elif kind[0] == 'binary':                    # random stiff/soft, independent of geometry
        rng = np.random.default_rng(1000 + seed)
        kb = np.where(rng.random(len(mesh['bond_R'])) < 0.5, 1.0, 1.0 / kind[1])
        mesh['bond_k'] = kb; mesh['tri_k'] = kb[mesh['tri_bond']]
    else:
        mesh['bond_k'] = np.ones(len(mesh['bond_R']))
        mesh['tri_k'] = mesh['bond_k'][mesh['tri_bond']]
    return mesh


def one(N, eta, seed, kind):
    """Tensor-level agreements for one realisation, or None if the sim refuses the geometry."""
    mesh = build(kind, N, eta, seed)
    nn, nt = len(mesh['pts']), len(mesh['simplices'])
    free = np.arange(2, 2 * nn)

    try:
        u_modes = PH.relax(mesh, free, SA.assemble_K_faff)
        C_s_raw = PH.energy_C_per_triangle(mesh, free, SA.assemble_K_faff)
    except PH.UnhealthyGeometryError:
        return None

    # independent: per-triangle, and the bulk DERIVED from it (region=None) so the two are one
    # construction and no extra solves are spent
    C_ind = C_s_raw / G_TO_DG
    bulk_energy = PH.energy_C_region(mesh, free, SA.assemble_K_faff, None, C_s=C_s_raw)
    bulk_virial = PH.virial_C(mesh, u_modes)

    # solver side, fed the SIM's W (isolates the contraction, as [7]/[8] do)
    Dgt = [F.T @ F - np.eye(2) for F in PH.Fk]
    Dinv = np.linalg.inv(np.stack([MO.vec3(g) for g in Dgt], 1))
    D = np.zeros((nt, 3, 3))
    for j, (F, u) in enumerate(zip(PH.Fk, u_modes)):
        D[:, :, j] = MO.vec3(MO.tri_metric_change(mesh['edge_vecs'], mesh['simplices'], F, u) - Dgt[j])
    W3 = D @ Dinv
    bare = MO.bare_tensor(mesh)
    ok = _as_mat(_contract(bare, W3, half=True))
    bug = _as_mat(_contract(bare, W3, half=False))
    bulk_solver = ok.mean(0) * (8.0 * nt / mesh['areas'].sum())

    def rel(a, b):
        return float(np.abs(a - b).max() / np.abs(b).max())

    return dict(oracle_control=rel(bulk_virial, bulk_energy),
                bulk=rel(bulk_solver, bulk_energy),
                per_tri=rel(ok, C_ind),
                per_tri_A0=rel(bug, C_ind),
                Wmag=float(np.abs(W3).max()),
                E_bulk=float(PH.virial_nuE(mesh, u_modes)[1]))   # how close to a MECHANISM


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--N', type=int, default=8)
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--eta-step', type=float, default=0.02)
    a = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)

    etas = np.round(np.arange(0.0, 0.5 + 1e-9, a.eta_step), 6)
    fams = [('frozen η, uniform k', ('uniform',)),
            ('VD a=−10', ('vd', -10)), ('VD a=−2', ('vd', -2)), ('VD a=+5', ('vd', 5)),
            ('VD a=+10', ('vd', 10)), ('VD a=+100', ('vd', 100)),
            ('random binary k (×10)', ('binary', 10.0)),
            ('re-triangulated η', ('retri',)),
            ('anisotropic base ψ=0.6', ('aniso',))]
    res = {f: {k: np.full((len(etas), a.seeds), np.nan) for k in KEYS + EXTRA} for f, _ in fams}
    nseed = {f: np.zeros(len(etas), int) for f, _ in fams}

    t0 = time.time()
    print(f"N={a.N}, eta 0→0.5 step {a.eta_step} ({len(etas)} pts), {a.seeds} seeds, "
          f"{len(fams)} families")
    for ie, eta in enumerate(etas):
        for fam, vd in fams:
            for js in range(a.seeds):
                r = one(a.N, float(eta), js, vd)
                if r is None:
                    continue
                nseed[fam][ie] += 1
                for k in KEYS + EXTRA:
                    res[fam][k][ie, js] = r[k]
        print(f"  eta={eta:.2f}  healthy seeds: " +
              ", ".join(f"{f}={nseed[f][ie]}/{a.seeds}" for f, _ in fams) +
              f"   [{time.time()-t0:.0f}s]", flush=True)

    # ---- figure: one panel per family, all four quantities overlaid --------------------------
    import matplotlib.pyplot as plt
    style = [('oracle_control', 'tab:purple', 'o', 'ORACLE CONTROL: virial vs energy (no solver)'),
             ('bulk', 'tab:green', '^', 'solver BULK vs oracle  (gated by [7])'),
             ('per_tri', 'tab:blue', 'o', 'solver PER-TRIANGLE vs oracle  (gated by [8])'),
             ('per_tri_A0', 'tab:red', 's', 'per-triangle WITH the A-0 defect (reference)')]
    ncol = 3
    nrow = int(np.ceil(len(fams) / ncol))
    fig, axg = plt.subplots(nrow, ncol, figsize=(P.STYLE.PANEL[0] * ncol, P.STYLE.PANEL[1] * nrow),
                            squeeze=False, sharey=True, sharex=True)
    axes_flat = [axg[i // ncol][i % ncol] for i in range(nrow * ncol)]
    for j in range(len(fams), nrow * ncol):
        axes_flat[j].axis('off')
    for ax, (fam, _) in zip(axes_flat, fams):
        for k, c, mk, lab in style:
            y = res[fam][k]
            mu = np.nanmean(y, 1)
            ax.semilogy(etas, mu, marker=mk, color=c, lw=1.9, ms=4, ls='--' if k.endswith('A0') else '-',
                        label=lab)
            lo, hi = np.nanmin(y, 1), np.nanmax(y, 1)
            ax.fill_between(etas, lo, hi, color=c, alpha=0.15, lw=0)
        ax.axhline(2e-2, color='k', ls=':', lw=1.3)
        ax.text(0.01, 2.4e-2, 'gate tolerance 2e-2', fontsize=8)
        bad = etas[nseed[fam] < a.seeds]
        if len(bad):
            ax.axvspan(bad.min(), etas[-1], color='0.85', zorder=-2)
            ax.text(bad.min() + 0.005, 1e-11, 'sim rejects some\nrealisations →', fontsize=7.5)
        ax.set_title(fam, fontsize=10)
        ax.grid(alpha=0.25, which='both')
    for i in range(len(fams)):
        if i // ncol == nrow - 1 or i + ncol >= len(fams):
            axes_flat[i].set_xlabel('disorder η')
        if i % ncol == 0:
            axes_flat[i].set_ylabel('max relative difference')
    h, l = axes_flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=2, fontsize=10, frameon=False,
               bbox_to_anchor=(0.5, -0.005))
    fig.suptitle(f'Tensor-level accuracy vs disorder, {len(fams)} network families '
                 f'(N={a.N}, {a.seeds} seeds, band = min–max) — η=0 is the degenerate case, '
                 f'not the informative one', fontsize=12)
    fig.tight_layout(rect=(0, 0.075, 1, 1))
    P.save_fig(fig, os.path.join(OUTDIR, 'accuracy_vs_disorder.png'))

    # ---- figure 2: the real explanatory variable ---------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(7.4, 5.2))
    cm2 = plt.get_cmap('tab10')
    for i, (fam, _) in enumerate(fams):
        E = res[fam]['E_bulk'].ravel(); e = res[fam]['per_tri'].ravel()
        m = np.isfinite(E) & np.isfinite(e) & (E > 0)
        ax2.loglog(E[m], e[m], 'o', ms=4.5, alpha=0.65, color=cm2(i % 10), label=fam)
    ax2.axhline(2e-2, color='k', ls=':', lw=1.3)
    ax2.text(ax2.get_xlim()[0] * 1.4, 2.4e-2, 'gate tolerance 2e-2', fontsize=8)
    ax2.set_xlabel("bulk Young's modulus E of the realisation  (physical units)")
    ax2.set_ylabel('per-triangle max relative difference')
    ax2.grid(alpha=0.25, which='both'); ax2.legend(fontsize=8, loc='upper right')
    fig2.suptitle('Accuracy is governed by proximity to a MECHANISM, not by η itself — '
                  'the softer the network, the worse the linear read-back', fontsize=11)
    fig2.tight_layout()
    P.save_fig(fig2, os.path.join(OUTDIR, 'accuracy_vs_stiffness.png'))

    out = {'N': a.N, 'seeds': a.seeds, 'etas': [float(e) for e in etas],
           'families': {f: {k: [float(v) for v in np.nanmean(res[f][k], 1)] for k in KEYS + EXTRA}
                        | {'healthy_seeds': [int(v) for v in nseed[f]]} for f, _ in fams}}
    with open(os.path.join(OUTDIR, 'accuracy_vs_disorder.json'), 'w', encoding='utf-8') as fh:
        json.dump(out, fh, indent=1)
    np.savez(os.path.join(OUTDIR, 'raw.npz'), etas=etas,
             **{f'{f}__{k}': res[f][k] for f, _ in fams for k in KEYS + EXTRA})

    print('\n  quantity                         eta=0      eta=0.2    eta=0.4    eta=0.5')
    ie = [0, int(round(0.2 / a.eta_step)), int(round(0.4 / a.eta_step)), len(etas) - 1]
    for fam, _ in fams:
        print(f'  --- {fam} ---')
        for k, _c, _m, lab in style:
            m = np.nanmean(res[fam][k], 1)
            print(f'   {lab[:32]:<32} ' + '  '.join(f'{m[i]:.2e}' for i in ie))
    print(f'\n  wrote {OUTDIR}')


if __name__ == '__main__':
    main()
