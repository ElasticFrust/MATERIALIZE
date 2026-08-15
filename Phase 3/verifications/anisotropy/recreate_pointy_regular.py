"""
anisotropy / recreate_pointy_regular — recreate the CRAZY pointy leaning crystal ν(θ) on a REGULAR mesh
with LIMITED floppiness.

Target = ν(θ) of the oblique crystal make_crystal(4,1) (ν∈[-3.6,+7.2], sharp leaning spike). Earlier
attempts were on a DISORDERED mesh: reg=0 found a singular mechanism (sim diverged, maxerr 36);
reg=5e-3 gave a floppy ±2.8. Here we design on the `regular` triangular topology and sweep a MODERATE
reg to keep the network a real material (both solvers agree, k not collapsing). Reports achieved ν,
sim-vs-diff agreement, floppiness per reg; plots ν(θ) overlay + polar + the best network.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _common as C

CASE, TOPO, N = 'anisotropy', 'regular', 16
REGS = [1.0e-2, 3.0e-3, 1.0e-3]                 # moderate -> limit floppiness (avoid reg=0 mechanism)
NITER, NREST = 450, 4


def main():
    th = C.ANG; deg = np.degrees(th)
    cg = C.make_crystal(4.0, 1.0, half=4.0)
    cg['bond_k'] = np.ones(len(cg['bond_R'])); cg['tri_k'] = cg['bond_k'][cg['tri_bond']]
    target, _ = C.nu_E_theta(C.sim_region_C6(cg), th)
    print(f"  [recreate_pointy_regular] target crystal(4,1) ν∈[{target.min():+.3f},{target.max():+.3f}] "
          f"on {TOPO} N={N}", flush=True)

    runs = []                                   # (reg, nu_diff, nu_sim, k, err)
    for reg in REGS:
        prob, geo = C.make_case(TOPO, N)
        obj = C.Objective('nu_theta', target=target)
        r = C.optimize(prob, [obj], mode='k', n_iter=NITER, n_restarts=NREST, reg=reg, verbose=False)
        k = r['k'].detach().numpy()
        nu_diff, _ = C.nu_E_theta(C.solver_region_C6(prob, r['k']), th)
        C.apply_k_to_geo(geo, r['k'])
        nu_sim, _ = C.nu_E_theta(C.sim_bulk_C6(geo), th)
        err = float(np.abs(nu_sim - target).max())
        disagree = float(np.abs(nu_sim - nu_diff).max())
        runs.append((reg, nu_diff, nu_sim, k, err, geo))
        print(f"  reg={reg:g}: diff[{nu_diff.min():+.3f},{nu_diff.max():+.3f}] "
              f"sim[{nu_sim.min():+.3f},{nu_sim.max():+.3f}] maxerr={err:.3f} "
              f"|sim-diff|={disagree:.3f} kmed={np.median(k):.3f} kmax={k.max():.2f} "
              f"n(k<0.1)={int((k < 0.1).sum())}/{len(k)}", flush=True)

    C.write_csv(os.path.join(C.savedir(CASE), 'recreate_pointy_regular.csv'),
                ['theta_deg', 'nu_target'] + [f'nu_sim_reg{r[0]:g}' for r in runs],
                [tuple([f'{np.degrees(t):.1f}', f'{target[i]:.4f}'] + [f'{r[2][i]:.4f}' for r in runs])
                 for i, t in enumerate(th)])

    # pick the "best real material": smallest |sim-diff| (most self-consistent) among decent-amplitude
    best = min(runs, key=lambda r: float(np.abs(r[2] - r[1]).max()))
    fig = plt.figure(figsize=(18, 5.5))
    a0 = fig.add_subplot(1, 3, 1)
    lc = C.draw_network(a0, best[5], best[3], cmap='viridis', lw_scale=3.5)
    plt.colorbar(lc, ax=a0, fraction=0.046, label='k')
    a0.set_title(f'best real material — reg={best[0]:g}, k median {np.median(best[3]):.2f}', fontsize=11)

    a1 = fig.add_subplot(1, 3, 2)
    a1.plot(deg, target, 'k--', lw=2.2, label=f'target [{target.min():+.1f},{target.max():+.1f}]')
    cols = plt.cm.plasma(np.linspace(0.2, 0.8, len(runs)))
    for (reg, _, nu_sim, k, err, _), col in zip(runs, cols):
        a1.plot(deg, nu_sim, color=col, lw=1.7,
                label=f'reg={reg:g}: [{nu_sim.min():+.1f},{nu_sim.max():+.1f}] kmed={np.median(k):.2f}')
    a1.axhline(0, color='0.7', lw=.5); a1.set_xlim(0, 180); a1.set_xticks(range(0, 181, 45))
    a1.set_xlabel('θ (deg)'); a1.set_ylabel('ν(θ)'); a1.grid(alpha=.3); a1.legend(fontsize=7)
    a1.set_title('regular mesh, limited floppiness — achieved vs target', fontsize=11)

    a2 = fig.add_subplot(1, 3, 3, projection='polar')
    th2 = np.concatenate([th, th + np.pi])
    a2.plot(th2, np.concatenate([target, target]), 'k--', lw=2.2, label='target')
    a2.plot(th2, np.concatenate([best[2], best[2]]), color='#d62728', lw=1.7, label=f'best (reg={best[0]:g})')
    a2.plot(th2, np.zeros_like(th2), color='0.6', lw=.5)
    a2.set_rorigin(min(0.0, target.min(), best[2].min()) * 1.05)
    a2.set_title('ν(θ) polar', fontsize=11, pad=14); a2.legend(fontsize=8, loc='upper right')

    fig.suptitle(f'{CASE} / recreate_pointy_regular — pointy leaning ν(θ) on a REGULAR mesh, '
                 f'floppiness limited', fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(C.savedir(CASE), 'recreate_pointy_regular.png')
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f'saved {os.path.basename(path)}')


if __name__ == '__main__':
    main()
