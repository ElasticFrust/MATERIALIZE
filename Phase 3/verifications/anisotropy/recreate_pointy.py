"""
anisotropy / recreate_pointy — CHALLENGE: reproduce the pointy, LEANING crystal ν(θ) on a DISORDERED
network by tuning bond stiffness k alone, with NO uniformity regulariser (reg=0 — floppy bonds
allowed). This isolates the topology question raised in the discussion: the sharp leaning ν(θ) of the
oblique crystal φ=4,ψ=1 (ν in ~[-3.7,+7.3]) comes from its aligned long-bond GEOMETRY; here the
geometry is a fixed ~isotropic disordered point set and only k is free. How close can we get?

Target: ν(θ) of make_crystal(4,1), k=1 (exact, W=0). Design: Objective('nu_theta') on disorder_hi,
reg=0, many restarts. Reports achieved vs target (differentiable readback AND independent PBC sim),
the network's floppiness, and the E(θ) the design produced (free — not targeted). Saves
recreate_pointy.png (target vs achieved ν line + polar, designed network, E(θ) comparison).
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _common as C

CASE, TOPO, N = 'anisotropy', 'disorder_hi', 16
NITER, NREST = 500, 5
REG = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0   # reg=0 -> floppy allowed; pass e.g. 5e-3 for a
TAG = sys.argv[2] if len(sys.argv) > 2 else ''           # real-material attempt. TAG suffixes outputs.


def main():
    th = C.ANG

    # --- target: the pointy leaning crystal nu(theta) (and its E(theta) for reference) ---
    cg = C.make_crystal(4.0, 1.0, half=4.0)
    cg['bond_k'] = np.ones(len(cg['bond_R'])); cg['tri_k'] = cg['bond_k'][cg['tri_bond']]
    C6c = C.sim_region_C6(cg)
    target, E_target = C.nu_E_theta(C6c, th)

    # --- design k on a disordered mesh to fit that nu(theta), no uniformity reg ---
    prob, geo = C.make_case(TOPO, N)
    obj = C.Objective('nu_theta', target=target)
    r = C.optimize(prob, [obj], mode='k', n_iter=NITER, n_restarts=NREST, reg=REG, verbose=False)
    C.apply_k_to_geo(geo, r['k'])

    nu_diff, E_diff = C.nu_E_theta(C.solver_region_C6(prob, r['k']), th)   # differentiable readback
    C6 = C.sim_per_triangle_C6(geo)
    nu_sim, E_sim = C.nu_E_theta(C.region_phys_C6(geo, C6, None), th)      # INDEPENDENT sim
    err = float(np.abs(nu_sim - target).max())
    k = r['k'].detach().numpy()
    print(f"  [recreate_pointy] target=crystal(4,1) nu on {TOPO} N={N} tri={prob.n_tri}  REG={REG}", flush=True)
    print(f"    target  ν(θ): [{target.min():+.3f},{target.max():+.3f}]", flush=True)
    print(f"    achieved ν(θ): diff [{nu_diff.min():+.3f},{nu_diff.max():+.3f}]  "
          f"sim [{nu_sim.min():+.3f},{nu_sim.max():+.3f}]  sim maxerr={err:.3f}", flush=True)
    print(f"    network floppiness: k median={np.median(k):.3f} [min={k.min():.3f}, max={k.max():.3f}]  "
          f"n(k<0.1)={int((k < 0.1).sum())}/{len(k)}", flush=True)

    C.write_csv(os.path.join(C.savedir(CASE), 'recreate_pointy'+TAG+'.csv'),
                ['theta_deg', 'nu_target', 'nu_achieved_sim', 'E_target', 'E_achieved_sim'],
                [(f'{np.degrees(t):.1f}', f'{target[i]:.4f}', f'{nu_sim[i]:.4f}',
                  f'{E_target[i]:.4f}', f'{E_sim[i]:.4f}') for i, t in enumerate(th)])

    deg = np.degrees(th)
    fig = plt.figure(figsize=(14, 11))
    a0 = fig.add_subplot(2, 2, 1)
    a0.plot(deg, target, 'k--', lw=2, label='target (crystal φ=4 ψ=1)')
    a0.plot(deg, nu_diff, color='#1f77b4', lw=1.8, label='achieved (diff-path)')
    a0.plot(deg, nu_sim, color='#d62728', lw=1.6, ls=':', label='achieved (independent sim)')
    a0.axhline(0, color='0.6', lw=.6); a0.set_xlim(0, 180); a0.set_xticks(range(0, 181, 45))
    a0.set_xlabel('θ (deg)'); a0.set_ylabel('ν(θ)'); a0.grid(alpha=.3); a0.legend(fontsize=8)
    a0.set_title(f'ν(θ): target vs achieved (sim maxerr={err:.2f})', fontsize=11)

    a1 = fig.add_subplot(2, 2, 2, projection='polar')
    th2 = np.concatenate([th, th + np.pi])
    for s, col, ls, lab in ((target, 'k', '--', 'target'), (nu_sim, '#d62728', '-', 'achieved')):
        a1.plot(th2, np.concatenate([s, s]), color=col, lw=1.6, ls=ls, label=lab)
    a1.plot(th2, np.zeros_like(th2), color='0.6', lw=.5)
    a1.set_rorigin(min(0.0, target.min(), nu_sim.min()) * 1.05)
    a1.set_title('ν(θ) polar', fontsize=11, pad=14); a1.legend(fontsize=8, loc='upper right')

    a2 = fig.add_subplot(2, 2, 3)
    lc = C.draw_network(a2, geo, k, cmap='viridis', lw_scale=4.0)
    plt.colorbar(lc, ax=a2, fraction=0.046, label='k')
    a2.set_title(f'designed network — k median {np.median(k):.2f}, n(k<0.1)={int((k<0.1).sum())}',
                 fontsize=11)

    a3 = fig.add_subplot(2, 2, 4)
    a3.plot(deg, E_target, 'k--', lw=2, label='target E(θ)')
    a3.plot(deg, E_sim, color='#2ca02c', lw=1.8, label='achieved E(θ)')
    a3.set_xlim(0, 180); a3.set_xticks(range(0, 181, 45)); a3.set_ylim(bottom=0)
    a3.set_xlabel('θ (deg)'); a3.set_ylabel('E(θ)'); a3.grid(alpha=.3); a3.legend(fontsize=8)
    a3.set_title('E(θ): target vs achieved (E was NOT targeted)', fontsize=11)

    fig.suptitle(f'{CASE} / recreate_pointy — reproduce the pointy leaning crystal ν(θ) on a '
                 f'disordered mesh (REG={REG:g})', fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(C.savedir(CASE), 'recreate_pointy'+TAG+'.png')
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f'saved {os.path.basename(path)}')


if __name__ == '__main__':
    main()
