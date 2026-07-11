"""
anisotropy / honeycomb_nu — recover the honeycomb Poisson ratio on a REGULAR triangular network.

A regular hexagonal honeycomb is super-soft yet has a well-defined, isotropic ν = 1 (Gibson–Ashby;
the "solved" case). In 2D, ν = 1 means the SHEAR modulus vanishes (ν = (K−μ)/(K+μ) → 1 as μ → 0) —
which is precisely why the honeycomb is super-soft. So we ask the inverse solver to design a REGULAR
triangular network (which starts rigid, ν = 1/3) to reach ν(θ) = 1, and check the twist: recovering the
honeycomb's ν should force the triangular network to become soft too (shear mode → 0, k collapsing).

Sweep reg (uniformity penalty): high reg keeps it stiff and CANNOT reach ν=1; low reg reaches ν≈1 but
goes soft. Report achieved ν(θ), the homogenised stiffness E(θ) (softness), and k floppiness, plus the
k=1 regular reference (ν=1/3). Verified by the independent PBC sim. Saves honeycomb_nu.png.
"""
import os, sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _common as C

CASE, TOPO, N = 'anisotropy', 'regular', 16
TARGET_NU = 1.0                                  # regular honeycomb, isotropic (Gibson–Ashby)
REGS = [1.0e-2, 1.0e-3, 1.0e-4]
NITER, NREST = 450, 3


def main():
    th = C.ANG; deg = np.degrees(th); target = np.full_like(th, TARGET_NU)

    # k=1 regular reference (should be ν≈1/3, isotropic)
    prob0, _ = C.make_case(TOPO, N)
    nu0, E0 = C.nu_E_theta(C.solver_region_C6(prob0, torch.ones(prob0.n_bond)), th)
    print(f"  [honeycomb_nu] target ν=1 (honeycomb) on {TOPO} N={N}; "
          f"k=1 reference: ν≈{nu0.mean():+.3f}, E≈{E0.mean():.3f}", flush=True)

    runs = []                                    # (reg, nu_sim, E_sim, k)
    for reg in REGS:
        prob, geo = C.make_case(TOPO, N)
        obj = C.Objective('nu_theta', target=target)
        r = C.optimize(prob, [obj], mode='k', n_iter=NITER, n_restarts=NREST, reg=reg, verbose=False)
        k = r['k'].detach().numpy()
        nu_diff, E_diff = C.nu_E_theta(C.solver_region_C6(prob, r['k']), th)
        C.apply_k_to_geo(geo, r['k'])
        C6 = C.sim_per_triangle_C6(geo)
        nu_sim, E_sim = C.nu_E_theta(C.region_phys_C6(geo, C6, None), th)
        runs.append((reg, nu_sim, E_sim, k))
        print(f"  reg={reg:g}: ν(θ) sim [{nu_sim.min():+.3f},{nu_sim.max():+.3f}] (mean {nu_sim.mean():+.3f}) "
              f"| E(θ) [{E_sim.min():.3f},{E_sim.max():.3f}] (soft vs E0={E0.mean():.2f}) "
              f"| k median={np.median(k):.3f} n(k<0.1)={int((k < 0.1).sum())}/{len(k)}", flush=True)

    C.write_csv(os.path.join(C.savedir(CASE), 'honeycomb_nu.csv'),
                ['theta_deg', 'nu_target'] + [f'nu_reg{r[0]:g}' for r in runs]
                + [f'E_reg{r[0]:g}' for r in runs],
                [tuple([f'{np.degrees(t):.1f}', f'{target[i]:.4f}']
                        + [f'{r[1][i]:.4f}' for r in runs] + [f'{r[2][i]:.4f}' for r in runs])
                 for i, t in enumerate(th)])

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    a0, a1 = axes
    a0.axhline(TARGET_NU, color='k', ls='--', lw=2, label='honeycomb target ν=1')
    a0.plot(deg, nu0, color='0.5', lw=1.5, label=f'k=1 regular (ν≈{nu0.mean():.2f})')
    cols = plt.cm.plasma(np.linspace(0.2, 0.8, len(runs)))
    for (reg, nu_sim, E_sim, k), col in zip(runs, cols):
        a0.plot(deg, nu_sim, color=col, lw=1.8,
                label=f'reg={reg:g}: ν≈{nu_sim.mean():.2f}, kmed={np.median(k):.2f}')
    a0.set_xlim(0, 180); a0.set_xticks(range(0, 181, 45)); a0.set_ylim(0, 1.15)
    a0.set_xlabel('θ (deg)'); a0.set_ylabel('ν(θ)'); a0.grid(alpha=.3); a0.legend(fontsize=8)
    a0.set_title('ν(θ): approaching the honeycomb ν=1', fontsize=11)

    a1.axhline(E0.mean(), color='0.5', ls='--', lw=1.5, label=f'k=1 regular E≈{E0.mean():.2f}')
    for (reg, nu_sim, E_sim, k), col in zip(runs, cols):
        a1.plot(deg, E_sim, color=col, lw=1.8, label=f'reg={reg:g}: E≈{E_sim.mean():.2f} (ν≈{nu_sim.mean():.2f})')
    a1.set_xlim(0, 180); a1.set_xticks(range(0, 181, 45)); a1.set_ylim(bottom=0)
    a1.set_xlabel('θ (deg)'); a1.set_ylabel('E(θ)'); a1.grid(alpha=.3); a1.legend(fontsize=8)
    a1.set_title('E(θ): the cost of ν→1 is softness (shear μ→0)', fontsize=11)

    fig.suptitle(f'{CASE} / honeycomb_nu — recovering the super-soft honeycomb ν=1 on a regular '
                 f'triangular network', fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(C.savedir(CASE), 'honeycomb_nu.png')
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f'saved {os.path.basename(path)}')


if __name__ == '__main__':
    main()
