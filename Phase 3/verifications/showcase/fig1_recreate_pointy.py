"""
showcase / fig1 — recreate the pointy leaning crystal ν(θ) on a disordered network (reg=5e-3).

Reproducibility: the design is run ONCE and its network is SAVED (fig1_pointy_network.npz, incl. the
per-triangle physical tensor); every re-render LOADS that frozen network — no re-optimisation, so the
figure is deterministic (optimize() uses random restarts, so re-running would give a different network —
this is why earlier renders drifted). Delete the npz to redesign.

2x3, self-contained:
  top:    ORIGINAL crystal topology (φ=4,ψ=1 leaning crystal we copy) | ν(θ) target-vs-achieved | ν polar
  bottom: designed disordered network (k)                             | E(θ) target-vs-achieved | E polar
ν(θ) is the only design target; the E(θ) panel is the honest counterpoint — same Poisson signature,
different stiffness. Curves are crystal target vs the independent PBC sim (matching the original CSV).
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

TOPO, N, REG, NITER, NREST = 'disorder_hi', 16, 5e-3, 500, 5   # match the original recreate_pointy_reg5e3
TH = C.ANG; DEG = np.degrees(TH)
CRY = 'k'; SOLV = '#1f77b4'; SIM = '#d62728'          # target/crystal in black
NETPATH = os.path.join(HERE, 'fig1_pointy_network.npz')


def design_or_load(target_nu, netpath):
    """LOAD the frozen designed network if it exists; else design ONCE and SAVE it (network + C6).
    Returns (geo, k, C6, prob) — prob (rebuilt from geo on load) gives the solver homogenisation."""
    if os.path.exists(netpath):
        geo, k, C6, _ = C.load_network(netpath)
        return geo, np.asarray(k, float), C6, C.DesignProblem.from_geo(geo)
    prob, geo = C.make_case(TOPO, N)
    r = C.optimize(prob, [C.Objective('nu_theta', target_nu)], mode='k',
                   n_iter=NITER, n_restarts=NREST, reg=REG, verbose=False)
    C.apply_k_to_geo(geo, r['k'])
    C6 = C.sim_per_triangle_C6(geo)
    k = np.asarray(r['k'].detach().numpy(), float)
    C.save_network(netpath, geo, k, C6, topo=TOPO, N=N, reg=REG, target='crystal_phi4_psi1',
                   nu_target=list(map(float, target_nu)))
    return geo, k, C6, prob


def polar(ax, vals, colors, labels, title, rmin=None):
    th2 = np.concatenate([TH, TH + np.pi])
    for v, c, (lab, ls) in zip(vals, colors, labels):
        ax.plot(th2, np.concatenate([v, v]), color=c, lw=2.0, ls=ls, label=lab)
    ax.plot(th2, np.zeros_like(th2), color='0.6', lw=.5)
    if rmin is not None:
        ax.set_rorigin(rmin)
    ax.set_title(title, fontsize=12, pad=16); ax.legend(fontsize=8.5, loc='upper right')


def main():
    cg = C.make_crystal(4.0, 1.0, half=4.0)                      # deterministic source crystal
    cg['bond_k'] = np.ones(len(cg['bond_R'])); cg['tri_k'] = cg['bond_k'][cg['tri_bond']]
    nu_t, E_t = C.nu_E_theta(C.sim_region_C6(cg), TH)

    geo, k, C6, prob = design_or_load(nu_t, NETPATH)
    nu_d, E_d = C.nu_E_theta(C.solver_region_C6(prob, torch.as_tensor(k, dtype=torch.float64)), TH)  # solver
    nu_s, E_s = C.nu_E_theta(C.region_phys_C6(geo, C6, None), TH)                                     # sim
    print(f"  {'loaded' if os.path.exists(NETPATH) else 'designed'} net: target ν∈[{nu_t.min():+.2f},"
          f"{nu_t.max():+.2f}]  solver ν[{nu_d.min():+.2f},{nu_d.max():+.2f}]  sim ν[{nu_s.min():+.2f},"
          f"{nu_s.max():+.2f}]  kmed={np.median(k):.2f}", flush=True)

    plt.rcParams.update({'font.size': 12})
    fig = plt.figure(figsize=(13.5, 9.6))
    gs = fig.add_gridspec(2, 2)

    a_nu = fig.add_subplot(gs[0, 0])
    a_nu.plot(DEG, nu_t, color=CRY, ls='--', lw=2.4, label='target (crystal ν)')
    a_nu.plot(DEG, nu_d, color=SOLV, lw=2.0, label='achieved (solver)')
    a_nu.plot(DEG, nu_s, color=SIM, lw=1.8, ls=':', label='achieved (independent sim)')
    a_nu.axhline(0, color='0.7', lw=.6); a_nu.set_xlim(0, 180); a_nu.set_xticks(range(0, 181, 45))
    a_nu.set_xlabel('loading angle θ (deg)'); a_nu.set_ylabel('ν(θ)')
    a_nu.grid(alpha=.3); a_nu.legend(fontsize=9, loc='lower right')
    a_nu.set_title('Poisson ratio ν(θ): target vs achieved', fontsize=12)
    axc = a_nu.inset_axes([0.60, 0.58, 0.37, 0.40])              # small source-crystal inset (3x3, edges cut)
    C.draw_lattice_zoom(axc, cg, color='k', cells=3.0, lw=1.0)
    axc.set_title('source crystal (φ4,ψ1)', fontsize=8, color=CRY, pad=2)

    a_np = fig.add_subplot(gs[0, 1], projection='polar')
    polar(a_np, [nu_t, nu_s], [CRY, SIM], [('target', '--'), ('achieved', '-')], 'ν(θ) polar',
          rmin=min(0.0, float(nu_t.min()), float(nu_s.min())) * 1.05)

    axn = fig.add_subplot(gs[1, 0])
    lc = C.draw_network(axn, geo, k, cmap='viridis', lw_scale=4.0)
    plt.colorbar(lc, ax=axn, fraction=0.046, label='bond rigidity k')
    axn.set_title(f'designed disordered network\n(k median {np.median(k):.2f})', fontsize=12)

    a_E = fig.add_subplot(gs[1, 1])
    a_E.plot(DEG, E_t, color=CRY, ls='--', lw=2.4, label='crystal E (not a target)')
    a_E.plot(DEG, E_d, color=SOLV, lw=2.0, label='design E (solver)')
    a_E.plot(DEG, E_s, color=SIM, lw=1.8, ls=':', label='design E (sim)')
    a_E.set_xlim(0, 180); a_E.set_xticks(range(0, 181, 45)); a_E.set_ylim(bottom=0)
    a_E.set_xlabel('loading angle θ (deg)'); a_E.set_ylabel('E(θ)')
    a_E.grid(alpha=.3); a_E.legend(fontsize=9)
    a_E.set_title('Young modulus E(θ): NOT targeted (design ~50× softer)', fontsize=11.5)

    fig.suptitle('Recreating a sharp, leaning Poisson ratio on a disordered network — ν(θ) copied from the '
                 'crystal, stiffness E(θ) left free (floppiness limited, reg=5e-3)', fontsize=13.5, y=1.005)
    plt.tight_layout()
    out = os.path.join(HERE, 'fig1_recreate_pointy.png')
    plt.savefig(out, dpi=200, bbox_inches='tight'); plt.close()
    print(f'saved {os.path.basename(out)}')


if __name__ == '__main__':
    main()
