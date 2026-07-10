"""
Case anisotropy / triangular_nu — a UNIFORM patch of metamaterial whose Poisson ratio is a strongly
ANISOTROPIC, SAWTOOTH function of direction spanning auxetic to strongly-positive: ν(θ) sweeps from
ν=−0.5 (auxetic) up to ν=+0.8 and back, with an asymmetric (sawtooth) ramp.

  WHY 4-FOLD (period 90°), not a 2-fold sawtooth. Elastic reciprocity (the compliance symmetry
  S₁₂=S₂₁ with E>0) forces ν(θ) and ν(θ+90°) to share a sign for EVERY θ:  ν(θ)·E(θ+90°) =
  ν(θ+90°)·E(θ). A continuous 2-fold profile that sweeps −0.5→+0.8 necessarily produces some
  orthogonal pair straddling zero → FORBIDDEN (an earlier symmetric-triangle attempt saturated its
  auxetic end at +0.09 for exactly this reason). A 4-fold profile has ν(θ)=ν(θ+90°) identically, so
  the same-sign rule holds automatically, and the auxetic minima (0°,90°) and the positive maxima
  (~45°,135°) sit at NON-orthogonal (45°-apart) directions — which is the realizable way to get both
  −0.5 and +0.8 with a sawtooth shape.

Designed by a single directional `nu_theta` objective on a disordered lattice (the most designable
base for directional ν). "Uniform" = one homogeneous material across the whole periodic cell (the
local-ν map shows the spatial uniformity). Verified two independent ways: validate()'s differentiable
readback AND _common.nu_E_theta on the INDEPENDENT NumPy PBC simulation. Outputs: triangular_nu.csv,
triangular_nu.npz, triangular_nu.png.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _common as C

CASE, TOPO, N, NITER, REG = 'anisotropy', 'disorder_hi', 16, 300, 1e-4
NU_LO, NU_HI = -0.5, 0.8                                          # sawtooth sweeps between these
PEAK = 0.7                                                        # peak at 70% of each 90° period (slow rise, fast fall)


def triangle_nu(theta):
    """A 4-FOLD (period 90°) SAWTOOTH ν(θ): within each 90° period, ν rises from NU_LO to NU_HI over
    the first `PEAK` fraction, then falls back to NU_LO over the rest (asymmetric -> sawtooth-like).
    Auxetic minima at 0°,90°,...; positive maxima at 0.7·90°≈63°,153°,... (non-orthogonal to the
    minima). ν(θ)=ν(θ+90°), so the reciprocity same-sign rule is satisfied by construction."""
    ph = np.mod(theta, np.pi / 2) / (np.pi / 2)                  # phase in [0,1) within each 90° period
    ramp = np.where(ph < PEAK, ph / PEAK, 1.0 - (ph - PEAK) / (1.0 - PEAK))   # 0→1→0, asymmetric
    return NU_LO + (NU_HI - NU_LO) * ramp


def main():
    th = C.ANG                                                   # θ grid ∈ [0,π], 37 points
    target = triangle_nu(th)

    prob, geo = C.make_case(TOPO, N)
    obj = C.Objective('nu_theta', target=target)                 # design the whole cell (uniform patch)
    r = C.optimize(prob, [obj], mode='k', n_iter=NITER, n_restarts=3, reg=REG, verbose=False)
    C.apply_k_to_geo(geo, r['k'])

    nu_diff, _ = C.nu_E_theta(C.solver_region_C6(prob, r['k']), th)   # differentiable readback
    C6 = C.sim_per_triangle_C6(geo)
    nu_sim, _ = C.nu_E_theta(C.region_phys_C6(geo, C6, None), th)     # INDEPENDENT sim
    err_diff = float(np.abs(nu_diff - target).max())
    err_sim = float(np.abs(nu_sim - target).max())
    print(f"  [triangular_nu] {TOPO} N={N} tri={prob.n_tri}", flush=True)
    print(f"    target ν(θ): {NU_LO:+.2f} → {NU_HI:+.2f} (triangle)", flush=True)
    print(f"    achieved ν(θ) range: diff-path [{nu_diff.min():+.3f},{nu_diff.max():+.3f}] maxerr={err_diff:.3f}  "
          f"| INDEPENDENT sim [{nu_sim.min():+.3f},{nu_sim.max():+.3f}] maxerr={err_sim:.3f}", flush=True)

    C.write_csv(os.path.join(C.savedir(CASE), 'triangular_nu.csv'),
                ['theta_deg', 'nu_target', 'nu_achieved_diffpath', 'nu_achieved_independent'],
                [(f'{np.degrees(t):.1f}', f'{target[i]:.4f}', f'{nu_diff[i]:.4f}', f'{nu_sim[i]:.4f}')
                 for i, t in enumerate(th)])
    C.save_network(os.path.join(C.savedir(CASE), 'triangular_nu.npz'), geo, r['k'], C6,
                   topo=TOPO, N=N, nu_lo=NU_LO, nu_hi=NU_HI)

    # --- plots: (a) ν(θ) line, (b) ν(θ) polar (mirrored to full circle), (c) local-ν uniformity map ---
    fig = plt.figure(figsize=(17, 5.4))
    a0 = fig.add_subplot(1, 3, 1)
    a0.plot(np.degrees(th), target, 'k--', lw=2, label='target (4-fold sawtooth)')
    a0.plot(np.degrees(th), nu_diff, color='#1f77b4', lw=1.8, label='achieved (diff-path)')
    a0.plot(np.degrees(th), nu_sim, color='#d62728', lw=1.8, ls=':', label='achieved (INDEPENDENT sim)')
    a0.axhline(0, color='0.6', lw=.6)
    a0.set_xlabel('direction θ (deg)'); a0.set_ylabel('Poisson ratio ν(θ)')
    a0.set_title(f'4-fold sawtooth ν(θ): {NU_LO:+.2f} ↔ {NU_HI:+.2f}', fontsize=11)
    a0.legend(fontsize=8); a0.grid(alpha=.3)

    a1 = fig.add_subplot(1, 3, 2, projection='polar')
    th2 = np.concatenate([th, th + np.pi])                       # centrosymmetric: mirror to [0,2π]
    for series, col, lab in ((target, 'k', 'target'), (nu_sim, '#d62728', 'sim')):
        a1.plot(th2, np.concatenate([series, series]), color=col, lw=1.8,
                ls='--' if lab == 'target' else '-', label=lab)
    a1.set_title('ν(θ) polar (mirrored)', fontsize=11, pad=16); a1.legend(fontsize=8, loc='upper right')

    a2 = fig.add_subplot(1, 3, 3)
    nu_local = C.local_field_smooth(geo, C6, quantity='nu')
    pc = C.fill_local_map(a2, geo, nu_local, cmap='RdBu_r', sym=True, vlim=max(abs(NU_LO), NU_HI))
    C.draw_box(a2, geo); plt.colorbar(pc, ax=a2, fraction=0.046)
    a2.set_title('local ν map — spatially UNIFORM patch', fontsize=11)

    fig.suptitle(f'{CASE} / triangular_nu — uniform patch, anisotropic 4-fold SAWTOOTH ν(θ) '
                 f'({NU_LO:+.2f}…{NU_HI:+.2f}; auxetic & positive at non-orthogonal dirs, {TOPO})', fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    path = os.path.join(C.savedir(CASE), 'triangular_nu.png')
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f'saved {os.path.basename(path)}')
    print('done')


if __name__ == '__main__':
    main()
