"""
showcase / fig1c & fig1d — recreate the ORIGINAL anisotropic crystal ν(θ) (peak 25°) by INVERSE-DESIGNING
the bond rigidities of a substrate that is itself ROTATED 90°. The point: the response orientation is
set by the DESIGN, not by the substrate's orientation — a 90°-rotated lattice still reproduces the same
lab-frame ν(θ).

  base=regular      -> fig1c : substrate = 90°-rotated REGULAR equilateral triangular lattice
  base=disorder_hi  -> fig1d : substrate = 90°-rotated DISORDERED network  (= fig1 with the mesh rotated)

Run:  python fig1cd_rotated_substrate.py regular       (fig1c)
      python fig1cd_rotated_substrate.py disorder_hi   (fig1d)

Reproducible: designs ONCE, saves the network (npz + C6), reloads thereafter (delete npz to redesign).
Curves: crystal target (black) vs achieved solver (blue) + independent sim (red). Verified with the PBC sim.
"""
import os, sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
VER = os.path.dirname(HERE)
sys.path.insert(0, VER)
import _common as C

BASE = sys.argv[1] if len(sys.argv) > 1 else 'regular'
N, NITER, NREST = 16, 500, 8
# The REGULAR lattice has no quenched rigidity: below reg~1e-2 the optimiser's forward solve wanders into
# an EXACTLY singular (mechanism) k-config and the sparse factorisation crashes; at reg>=1.2e-2 it stays
# stable but k cannot build the anisotropy (flat ν). There is NO reg where it both stays stable AND
# reproduces the leaning target -> the regular equilateral lattice genuinely cannot do it.
REG = 1.2e-2 if BASE == 'regular' else 5e-3
KFLOOR = 1e-2 if BASE == 'regular' else 0.0      # floor only for regular; disorder runs floor-free
TH = C.ANG; DEG = np.degrees(TH)
CRY = 'k'; SOLV = '#1f77b4'; SIM = '#d62728'
_META = {'regular':     ('fig1c', '90°-rotated regular (equilateral) lattice', 'fig1c_rot90_regular'),
         'disorder_hi': ('fig1d', '90°-rotated disordered network',            'fig1d_rot90_disorder')}
TAG, SUBLABEL, STEM = _META[BASE]
NETPATH = os.path.join(HERE, STEM + '_network.npz')


def rotate_geo(geo, deg):
    a = np.radians(deg); R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    g = dict(geo)
    for key in ('pts', 'edge_vecs', 'bond_R', 'centroids', 'tri_verts'):
        g[key] = geo[key] @ R.T
    g['BL1'] = R @ geo['BL1']; g['BL2'] = R @ geo['BL2']
    g['actual_len2'] = (g['edge_vecs'] ** 2).sum(-1)
    return g


def design_or_load(target_nu):
    """Design k on the 90°-rotated `BASE` substrate to hit target_nu; save once, load thereafter."""
    if os.path.exists(NETPATH):
        geo, k, C6, _ = C.load_network(NETPATH)
        return geo, np.asarray(k, float), C6, C.DesignProblem.from_geo(geo)
    geo = rotate_geo(C.make_topology(BASE, N), 90.0)        # rotated design substrate
    prob = C.DesignProblem.from_geo(geo)
    r = C.optimize(prob, [C.Objective('nu_theta', target_nu)], mode='k',
                   n_iter=NITER, n_restarts=NREST, reg=REG, verbose=False)
    kraw = np.asarray(r['k'].detach().numpy(), float)
    k = np.maximum(kraw, KFLOOR) if KFLOOR > 0 else kraw   # no clamp at all when KFLOOR==0 (floor-free)
    print(f"    raw k range [{kraw.min():.3e}, {kraw.max():.2f}]  applied k min {k.min():.3e}  "
          f"(KFLOOR={KFLOOR}, floored {int((k != kraw).sum())} bonds)", flush=True)
    C.apply_k_to_geo(geo, k)
    C6 = C.sim_per_triangle_C6(geo)
    C.save_network(NETPATH, geo, k, C6, base=BASE, rot_deg=90, N=N, reg=REG,
                   target='crystal_phi4_psi1', nu_target=list(map(float, target_nu)))
    return geo, k, C6, prob


def polar(ax, vals, colors, labels, title, rmin=None):
    th2 = np.concatenate([TH, TH + np.pi])
    for v, c, (lab, ls) in zip(vals, colors, labels):
        ax.plot(th2, np.concatenate([v, v]), color=c, lw=2.2, ls=ls, label=lab)
    ax.plot(th2, np.zeros_like(th2), color='0.6', lw=.5)
    if rmin is not None:
        ax.set_rorigin(rmin)
    ax.set_title(title, fontsize=12, pad=16); ax.legend(fontsize=9, loc='upper right')


def main():
    cg = C.make_crystal(4.0, 1.0, half=4.0)
    cg['bond_k'] = np.ones(len(cg['bond_R'])); cg['tri_k'] = cg['bond_k'][cg['tri_bond']]
    nu_t, E_t = C.nu_E_theta(C.sim_region_C6(cg), TH)         # ORIGINAL crystal target (peak 25°)

    geo, k, C6, prob = design_or_load(nu_t)
    nu_d, E_d = C.nu_E_theta(C.solver_region_C6(prob, torch.as_tensor(k, dtype=torch.float64)), TH)
    nu_s, E_s = C.nu_E_theta(C.sim_bulk_C6(geo), TH)
    # undesigned baseline: the SAME substrate with UNIFORM k=1 (bare lattice, before inverse design)
    gb = dict(geo); gb['bond_k'] = np.ones(len(geo['bond_R'])); gb['tri_k'] = gb['bond_k'][geo['tri_bond']]
    nu_u = C.nu_E_theta(C.sim_bulk_C6(gb), TH)[0]
    print(f"  [{TAG} {BASE} rot90] target ν∈[{nu_t.min():+.2f},{nu_t.max():+.2f}]  solver "
          f"[{nu_d.min():+.2f},{nu_d.max():+.2f}]  sim [{nu_s.min():+.2f},{nu_s.max():+.2f}]  "
          f"kmed={np.median(k):.2f}", flush=True)

    plt.rcParams.update({'font.size': 12})
    fig = plt.figure(figsize=(13.5, 9.6))
    gs = fig.add_gridspec(2, 2)

    a_nu = fig.add_subplot(gs[0, 0])
    a_nu.plot(DEG, nu_t, color=CRY, ls='--', lw=2.4, label='target (crystal ν)')
    a_nu.plot(DEG, nu_u, color='0.55', lw=1.6, ls='-.', label='substrate k=1 (undesigned)')
    a_nu.plot(DEG, nu_d, color=SOLV, lw=2.0, label='achieved (solver)')
    a_nu.plot(DEG, nu_s, color=SIM, lw=1.8, ls=':', label='achieved (independent sim)')
    a_nu.axhline(0, color='0.7', lw=.6); a_nu.set_xlim(0, 180); a_nu.set_xticks(range(0, 181, 45))
    a_nu.set_xlabel('loading angle θ (deg)'); a_nu.set_ylabel('ν(θ)')
    a_nu.grid(alpha=.3); a_nu.legend(fontsize=9, loc='lower right')
    a_nu.set_title('Poisson ratio ν(θ): crystal target vs achieved', fontsize=12)
    axc = a_nu.inset_axes([0.60, 0.58, 0.37, 0.40])
    C.draw_lattice_zoom(axc, cg, color='k', cells=3.0, lw=1.0)
    axc.set_title('source crystal (φ4,ψ1)', fontsize=8, color='k', pad=2)

    a_np = fig.add_subplot(gs[0, 1], projection='polar')
    polar(a_np, [nu_t, nu_s], [CRY, SIM], [('target', '--'), ('achieved', '-')], 'ν(θ) polar',
          rmin=min(0.0, float(nu_t.min()), float(nu_s.min())) * 1.05)

    axn = fig.add_subplot(gs[1, 0])
    lc = C.draw_network(axn, geo, k, cmap='viridis', lw_scale=4.0)
    plt.colorbar(lc, ax=axn, fraction=0.046, label='bond rigidity k')
    axn.set_title(f'designed {SUBLABEL}\n(k median {np.median(k):.2f})', fontsize=11)

    a_E = fig.add_subplot(gs[1, 1])
    a_E.plot(DEG, E_t, color=CRY, ls='--', lw=2.4, label='crystal E (not a target)')
    a_E.plot(DEG, E_d, color=SOLV, lw=2.0, label='design E (solver)')
    a_E.plot(DEG, E_s, color=SIM, lw=1.8, ls=':', label='design E (sim)')
    a_E.set_xlim(0, 180); a_E.set_xticks(range(0, 181, 45)); a_E.set_ylim(bottom=0)
    a_E.set_xlabel('loading angle θ (deg)'); a_E.set_ylabel('E(θ)')
    a_E.grid(alpha=.3); a_E.legend(fontsize=9); a_E.set_title('Young modulus E(θ): NOT targeted', fontsize=11.5)

    if BASE == 'regular':
        sup = (f'{TAG}: the 90°-rotated REGULAR equilateral lattice CANNOT reproduce the crystal\'s '
               'anisotropic ν(θ) — it flattens (any reg low enough to build the anisotropy drives the '
               'optimiser singular). Contrast fig1d (disorder).')
    else:
        sup = (f'{TAG}: crystal ν(θ) inverse-designed on a {SUBLABEL} — the response orientation is set by '
               'the DESIGN, not the substrate (target unrotated, substrate rotated 90°)')
    fig.suptitle(sup, fontsize=12.5, y=1.005)
    plt.tight_layout()
    out = os.path.join(HERE, STEM + '.png')
    plt.savefig(out, dpi=200, bbox_inches='tight'); plt.close()
    print(f'saved {os.path.basename(out)}')


if __name__ == '__main__':
    main()
