"""
Case anisotropy / triangular_nu — a UNIFORM patch of metamaterial whose Poisson ratio is an
anisotropic, 2-fold SAWTOOTH function of direction: ν(θ) rises from ν=0 up to ν=+0.5 and falls back,
once per half-turn (period π), with an asymmetric (sawtooth) ramp.

  WHY 0→+0.5 IS REALIZABLE AS A 2-FOLD sawtooth (unlike an earlier −0.5→+0.8 attempt). Elastic
  reciprocity (compliance symmetry S₁₂=S₂₁, E>0) forces ν(θ) and ν(θ+90°) to share a sign for EVERY
  θ. A sign-CROSSING 2-fold profile (e.g. −0.5→+0.8) is therefore forbidden — any continuous sweep
  across zero yields an orthogonal pair of opposite sign. But 0→+0.5 stays NON-NEGATIVE, so the
  same-sign rule is trivially satisfied and the natural single-period sawtooth is allowed.

  DESIGN CHOICE — a GENUINE uniform material, not a floppy skeleton. An unregularised design of a
  strong directional ν(θ) tends to fake it with a sparse stiff scaffold in a near-zero (mechanism)
  matrix. Here a STRONG uniformity regulariser `REG` (the mean((k−1)²) penalty in optimize) keeps the
  bond stiffnesses near-uniform, so the result is a robust homogeneous patch. This trades some
  directional amplitude for uniformity/robustness (shape prioritised over amplitude). The reported
  k-stats (median, contrast) and the local-ν map document how uniform the material actually is.

Designed by a single directional `nu_theta` objective on a disordered lattice. Verified two
independent ways: validate()'s differentiable readback AND _common.nu_E_theta on the INDEPENDENT
NumPy PBC simulation. Outputs: triangular_nu.csv, triangular_nu.npz, triangular_nu.png (network +
ν(θ) line + polar + local-ν map).
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _common as C

CASE, TOPO, N, NITER, REG = 'anisotropy', 'disorder_hi', 16, 320, 3e-3   # REG balances uniformity vs
# reaching the full 0→0.5 amplitude: 2e-2 kept k very uniform (median 0.88) but the troughs stalled at
# +0.13; 3e-3 relaxes just enough to reach ν≈0 in the trough directions while staying a real material.
NU_LO, NU_HI = 0.0, 0.5                                          # sawtooth sweeps between these (same sign)
PEAK = 0.72                                                      # peak at 72% of the period (slow rise, fast fall)


def triangle_nu(theta):
    """A 2-FOLD (period π) SAWTOOTH ν(θ): over each half-turn, ν rises from NU_LO to NU_HI over the
    first `PEAK` fraction of the period, then falls back to NU_LO over the rest (asymmetric ->
    sawtooth). Minima (ν=NU_LO) at 0°,180°; maximum (ν=NU_HI) at 0.72·180°≈130°. Continuous and
    π-periodic (ν(0)=ν(180°)). Same-sign throughout (≥0), so reciprocity-realizable as a 2-fold."""
    ph = np.mod(theta, np.pi) / np.pi                           # phase in [0,1) over the π period
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
    k = r['k'].detach().numpy()
    print(f"  [triangular_nu] {TOPO} N={N} tri={prob.n_tri}  REG={REG}", flush=True)
    print(f"    target ν(θ): {NU_LO:+.2f} → {NU_HI:+.2f} (2-fold sawtooth)", flush=True)
    print(f"    achieved ν(θ): diff-path [{nu_diff.min():+.3f},{nu_diff.max():+.3f}] maxerr={err_diff:.3f}  "
          f"| INDEPENDENT sim [{nu_sim.min():+.3f},{nu_sim.max():+.3f}] maxerr={err_sim:.3f}", flush=True)
    print(f"    network uniformity: k median={np.median(k):.3f} mean={k.mean():.3f} "
          f"[min={k.min():.3f}, max={k.max():.3f}]  n(k<0.1)={int((k < 0.1).sum())}/{len(k)}", flush=True)

    C.write_csv(os.path.join(C.savedir(CASE), 'triangular_nu.csv'),
                ['theta_deg', 'nu_target', 'nu_achieved_diffpath', 'nu_achieved_independent'],
                [(f'{np.degrees(t):.1f}', f'{target[i]:.4f}', f'{nu_diff[i]:.4f}', f'{nu_sim[i]:.4f}')
                 for i, t in enumerate(th)])
    C.save_network(os.path.join(C.savedir(CASE), 'triangular_nu.npz'), geo, r['k'], C6,
                   topo=TOPO, N=N, nu_lo=NU_LO, nu_hi=NU_HI)

    # --- 2x2: (a) designed NETWORK, (b) ν(θ) line, (c) ν(θ) polar, (d) local-ν uniformity map ---
    fig = plt.figure(figsize=(13, 11))
    a0 = fig.add_subplot(2, 2, 1)
    lc = C.draw_network(a0, geo, k, cmap='viridis', lw_scale=4.0)
    plt.colorbar(lc, ax=a0, fraction=0.046, label='bond rigidity k')
    a0.set_title(r'designed NETWORK (colour & width $\propto$ k)  —  k median '
                 f'{np.median(k):.2f}', fontsize=11)

    a1 = fig.add_subplot(2, 2, 2)
    a1.plot(np.degrees(th), target, 'k--', lw=2, label='target (2-fold sawtooth)')
    a1.plot(np.degrees(th), nu_diff, color='#1f77b4', lw=1.8, label='achieved (diff-path)')
    a1.plot(np.degrees(th), nu_sim, color='#d62728', lw=1.8, ls=':', label='achieved (INDEPENDENT sim)')
    a1.axhline(0, color='0.6', lw=.6)
    a1.set_xlabel('direction θ (deg)'); a1.set_ylabel('Poisson ratio ν(θ)')
    a1.set_title(f'2-fold sawtooth ν(θ): {NU_LO:.2f} → {NU_HI:.2f}', fontsize=11)
    a1.legend(fontsize=8); a1.grid(alpha=.3)

    a2 = fig.add_subplot(2, 2, 3, projection='polar')
    th2 = np.concatenate([th, th + np.pi])                       # centrosymmetric: mirror to [0,2π]
    for series, col, lab in ((target, 'k', 'target'), (nu_sim, '#d62728', 'sim')):
        a2.plot(th2, np.concatenate([series, series]), color=col, lw=1.8,
                ls='--' if lab == 'target' else '-', label=lab)
    a2.set_title('ν(θ) polar (mirrored)', fontsize=11, pad=16); a2.legend(fontsize=8, loc='upper right')

    a3 = fig.add_subplot(2, 2, 4)
    nu_local = C.local_field_smooth(geo, C6, quantity='nu')
    pc = C.fill_local_map(a3, geo, nu_local, cmap='RdBu_r', sym=True, vlim=max(0.3, NU_HI))
    C.draw_box(a3, geo); plt.colorbar(pc, ax=a3, fraction=0.046)
    a3.set_title('local ν map — spatial uniformity', fontsize=11)

    fig.suptitle(f'{CASE} / triangular_nu — uniform patch, anisotropic 2-fold SAWTOOTH ν(θ) '
                 f'({NU_LO:.2f}…{NU_HI:.2f}, {TOPO}, REG={REG})', fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(C.savedir(CASE), 'triangular_nu.png')
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f'saved {os.path.basename(path)}')
    print('done')


if __name__ == '__main__':
    main()
