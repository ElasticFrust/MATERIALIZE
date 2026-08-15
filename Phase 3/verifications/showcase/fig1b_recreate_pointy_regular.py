"""
showcase / fig1b — recreate the pointy leaning crystal ν(θ) on a REGULAR lattice (reg=5e-3), WITHOUT
the E(θ) panel. Same as fig1 but TOPO='regular'. Three panels: target-vs-achieved ν(θ) line, ν(θ)
polar, and the designed network. Self-contained.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
VER = os.path.dirname(HERE)
sys.path.insert(0, VER)
import _common as C

TOPO, N, REG, NITER, NREST = 'regular', 16, 5e-3, 400, 3
TH = C.ANG; DEG = np.degrees(TH)


def main():
    cg = C.make_crystal(4.0, 1.0, half=4.0)
    cg['bond_k'] = np.ones(len(cg['bond_R'])); cg['tri_k'] = cg['bond_k'][cg['tri_bond']]
    target = C.nu_E_theta(C.sim_region_C6(cg), TH)[0]

    prob, geo = C.make_case(TOPO, N)
    r = C.optimize(prob, [C.Objective('nu_theta', target)], mode='k',
                   n_iter=NITER, n_restarts=NREST, reg=REG, verbose=False)
    k = np.asarray(r['k'].detach().numpy(), float)
    nu_diff = C.nu_E_theta(C.solver_region_C6(prob, r['k']), TH)[0]
    C.apply_k_to_geo(geo, r['k'])
    nu_sim = C.nu_E_theta(C.sim_bulk_C6(geo), TH)[0]
    disagree = float(np.abs(nu_sim - nu_diff).max())
    print(f"  [regular] target ν∈[{target.min():+.2f},{target.max():+.2f}]  achieved sim "
          f"[{nu_sim.min():+.2f},{nu_sim.max():+.2f}]  |sim-diff|={disagree:.2f}  kmed={np.median(k):.2f} "
          f"n(k<0.1)={int((k<0.1).sum())}/{len(k)}", flush=True)

    plt.rcParams.update({'font.size': 12})
    fig = plt.figure(figsize=(16, 5.4))
    a0 = fig.add_subplot(1, 3, 1)
    a0.plot(DEG, target, 'k--', lw=2.4, label='target (crystal ν)')
    a0.plot(DEG, nu_diff, color='#1f77b4', lw=2.0, label='achieved (solver)')
    a0.plot(DEG, nu_sim, color='#d62728', lw=1.8, ls=':', label='achieved (independent sim)')
    a0.axhline(0, color='0.7', lw=.6); a0.set_xlim(0, 180); a0.set_xticks(range(0, 181, 45))
    a0.set_xlabel('loading angle θ (deg)'); a0.set_ylabel('ν(θ)'); a0.grid(alpha=.3); a0.legend(fontsize=10)
    a0.set_title('sharp leaning ν(θ) on a REGULAR lattice: target vs achieved', fontsize=11.5)

    a1 = fig.add_subplot(1, 3, 2, projection='polar')
    th2 = np.concatenate([TH, TH + np.pi])
    a1.plot(th2, np.concatenate([target, target]), 'k--', lw=2.2, label='target')
    a1.plot(th2, np.concatenate([nu_sim, nu_sim]), color='#d62728', lw=2.0, label='achieved')
    a1.plot(th2, np.zeros_like(th2), color='0.6', lw=.5)
    a1.set_rorigin(min(0.0, float(target.min()), float(nu_sim.min())) * 1.05)
    a1.set_title('ν(θ) polar', fontsize=12, pad=16); a1.legend(fontsize=9, loc='upper right')

    a2 = fig.add_subplot(1, 3, 3)
    lc = C.draw_network(a2, geo, k, cmap='viridis', lw_scale=4.0)
    plt.colorbar(lc, ax=a2, fraction=0.046, label='bond rigidity k')
    a2.set_title(f'designed network (k median {np.median(k):.2f})', fontsize=12)

    fig.suptitle('Fitting a sharp, leaning Poisson ratio on a REGULAR lattice (floppiness limited)',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    out = os.path.join(HERE, 'fig1b_recreate_pointy_regular.png')
    plt.savefig(out, dpi=200, bbox_inches='tight'); plt.close()
    print(f'saved {os.path.basename(out)}')


if __name__ == '__main__':
    main()
