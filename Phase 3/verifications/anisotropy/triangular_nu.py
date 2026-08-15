"""
Case anisotropy / triangular_nu — a UNIFORM patch of metamaterial whose Poisson ratio is an
anisotropic SAWTOOTH function of direction. Two configurations (select on the command line):

  '2fold' (default) — ν(θ) rises from ν=0 up to ν=+0.5 and falls back ONCE per half-turn (period π).
      Stays non-negative because a negative trough is reciprocity-forbidden here (see below).
      Outputs triangular_nu.{png,csv,npz}.
  '4fold' — a symmetric POINTY TRIANGLE wave, FOUR lobes per turn (period π/2), dipping to a genuine
      auxetic trough (ν=−0.1), on a LARGER patch. Symmetric (not sawtooth) because 4-fold ν(θ) is a
      function of the single variable cos4(θ−φ) — its lobes are mirror-symmetric by construction; the
      in-phase 8θ/12θ overtones can only sharpen it toward a triangle, never lean it (see below).
      Outputs triangular_nu_4fold.{png,csv,npz}.

  RECIPROCITY — why the fold count matters for the auxetic trough. Elastic reciprocity (compliance
  symmetry S₁₂=S₂₁, E>0) forces ν(θ) and ν(θ+90°) to share a sign for EVERY θ.
    · In the 2-FOLD profile the trough at θ=0/180° is orthogonal to θ=90°, which sits on the POSITIVE
      ramp — so reciprocity CAPS how auxetic the trough can get (it is driven toward/just past zero
      but may not reach the full −0.1). A fully sign-crossing 2-fold (e.g. −0.5→+0.8) is forbidden.
    · In the 4-FOLD profile ν(θ)=ν(θ+90°) by construction (period π/2), so orthogonal directions are
      IDENTICAL and the same-sign rule is satisfied automatically — the auxetic troughs are free to go
      genuinely negative. This is the regime where a LARGER patch pays off: the extra bond DOF let the
      deep-auxetic 4-fold be a real uniform material instead of a floppy skeleton.

  DESIGN CHOICE — a GENUINE uniform material, not a floppy skeleton. An unregularised design of a
  strong directional ν(θ) tends to fake it with a sparse stiff scaffold in a near-zero (mechanism)
  matrix. A uniformity regulariser `reg` (the mean((k−1)²) penalty in optimize) keeps the bond
  stiffnesses near-uniform. Lower reg → sharper profile but floppier; higher reg → more uniform but
  amplitude/shape softened. The reported k-stats and the local-ν map document the actual uniformity.

Designed by a single directional `nu_theta` objective on a disordered lattice. Verified two
independent ways: the differentiable readback (solver_region_C6) AND _common.nu_E_theta on the
INDEPENDENT NumPy PBC simulation.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _common as C

CASE, TOPO = 'anisotropy', 'disorder_hi'
NU_HI = 0.5                                   # peak value of the profile

# Per-configuration knobs. `peak` sets the position of the maximum within the period: peak=0.72 →
# asymmetric SAWTOOTH (slow rise, fast fall); peak=0.5 → symmetric TRIANGLE wave. `nu_lo` (trough)
# differs by fold because of reciprocity:
#   '2fold' — asymmetric sawtooth, nu_lo=0. A NEGATIVE trough is forbidden here (its orthogonal sits on
#             the positive ramp; a −0.1 attempt capped at ~+0.05 and just made the network floppy). The
#             sawtooth LEAN is reachable because 2-fold has two independent modes (weights 2 and 4).
#   '4fold' — symmetric pointy triangle (peak=0.5), nu_lo=−0.1. With ν(θ)=ν(θ+90°) the auxetic trough
#             is reciprocity-LEGAL. A LEAN is impossible (single weight-4 mode, phase-locked overtones →
#             mirror-symmetric lobes), so we ask for the reachable shape: a SYMMETRIC pointy triangle.
#             The LARGER patch (N=24) supplies the DOF to keep the deep-auxetic result a real material.
# tag='' keeps the 2-fold's canonical triangular_nu.* filenames (read by the maps/param scripts).
CONFIGS = {
    '2fold': dict(fold=2, N=16, n_iter=420, reg=8.0e-4, n_restarts=4, nu_lo=0.0,  peak=0.72, tag=''),
    '4fold': dict(fold=4, N=24, n_iter=360, reg=1.0e-3, n_restarts=3, nu_lo=-0.1, peak=0.50, tag='_4fold'),
}


def profile_shape(peak):
    return 'triangle' if abs(peak - 0.5) < 1e-6 else 'sawtooth'


def triangle_nu(theta, fold, nu_lo, peak):
    """A `fold`-fold piecewise-linear ν(θ) with period P=π/(fold//2) (2-fold→π, 4-fold→π/2): over each
    period ν rises from nu_lo to NU_HI over the first `peak` fraction, then falls back to nu_lo. peak=0.5
    → symmetric TRIANGLE wave; peak≠0.5 → asymmetric SAWTOOTH. Continuous and π-periodic. For 4-fold,
    ν(θ)=ν(θ+90°) so a negative nu_lo (auxetic trough) is reciprocity-legal; for 2-fold nu_lo must ≥0."""
    P = np.pi / (fold // 2)                                     # angular period of one lobe
    ph = np.mod(theta, P) / P                                  # phase in [0,1) over the period
    ramp = np.where(ph < peak, ph / peak, 1.0 - (ph - peak) / (1.0 - peak))   # 0→1→0
    return nu_lo + (NU_HI - nu_lo) * ramp


def main(name, cfg):
    fold, N, tag, nu_lo, peak = cfg['fold'], cfg['N'], cfg['tag'], cfg['nu_lo'], cfg['peak']
    shape = profile_shape(peak)
    base = 'triangular_nu' + tag
    th = C.ANG                                                   # θ grid ∈ [0,π], 37 points
    target = triangle_nu(th, fold, nu_lo, peak)

    prob, geo = C.make_case(TOPO, N)
    obj = C.Objective('nu_theta', target=target)                 # design the whole cell (uniform patch)
    r = C.optimize(prob, [obj], mode='k', n_iter=cfg['n_iter'], n_restarts=cfg['n_restarts'],
                   reg=cfg['reg'], verbose=False)
    C.apply_k_to_geo(geo, r['k'])

    nu_diff, _ = C.nu_E_theta(C.solver_region_C6(prob, r['k']), th)   # differentiable readback
    C6 = C.sim_per_triangle_C6(geo)
    nu_sim, _ = C.nu_E_theta(C.sim_bulk_C6(geo), th)     # INDEPENDENT sim
    err_diff = float(np.abs(nu_diff - target).max())
    err_sim = float(np.abs(nu_sim - target).max())
    k = r['k'].detach().numpy()
    print(f"  [triangular_nu:{name}] {fold}-fold {TOPO} N={N} tri={prob.n_tri}  REG={cfg['reg']}", flush=True)
    print(f"    target ν(θ): {nu_lo:+.2f} → {NU_HI:+.2f} ({fold}-fold {shape})", flush=True)
    print(f"    achieved ν(θ): diff-path [{nu_diff.min():+.3f},{nu_diff.max():+.3f}] maxerr={err_diff:.3f}  "
          f"| INDEPENDENT sim [{nu_sim.min():+.3f},{nu_sim.max():+.3f}] maxerr={err_sim:.3f}", flush=True)
    print(f"    network uniformity: k median={np.median(k):.3f} mean={k.mean():.3f} "
          f"[min={k.min():.3f}, max={k.max():.3f}]  n(k<0.1)={int((k < 0.1).sum())}/{len(k)}", flush=True)

    C.write_csv(os.path.join(C.savedir(CASE), base + '.csv'),
                ['theta_deg', 'nu_target', 'nu_achieved_diffpath', 'nu_achieved_independent'],
                [(f'{np.degrees(t):.1f}', f'{target[i]:.4f}', f'{nu_diff[i]:.4f}', f'{nu_sim[i]:.4f}')
                 for i, t in enumerate(th)])
    C.save_network(os.path.join(C.savedir(CASE), base + '.npz'), geo, r['k'], C6,
                   topo=TOPO, N=N, fold=fold, nu_lo=nu_lo, nu_hi=NU_HI)

    # --- 2x2: (a) designed NETWORK, (b) ν(θ) line, (c) ν(θ) polar, (d) local-ν uniformity map ---
    fig = plt.figure(figsize=(13, 11))
    a0 = fig.add_subplot(2, 2, 1)
    lc = C.draw_network(a0, geo, k, cmap='viridis', lw_scale=4.0)
    plt.colorbar(lc, ax=a0, fraction=0.046, label='bond rigidity k')
    a0.set_title(r'designed NETWORK (colour & width $\propto$ k)  —  k median '
                 f'{np.median(k):.2f}', fontsize=11)

    a1 = fig.add_subplot(2, 2, 2)
    a1.plot(np.degrees(th), target, 'k--', lw=2, label=f'target ({fold}-fold {shape})')
    a1.plot(np.degrees(th), nu_diff, color='#1f77b4', lw=1.8, label='achieved (diff-path)')
    a1.plot(np.degrees(th), nu_sim, color='#d62728', lw=1.8, ls=':', label='achieved (INDEPENDENT sim)')
    a1.axhline(0, color='0.6', lw=.6)
    a1.set_xlabel('direction θ (deg)'); a1.set_ylabel('Poisson ratio ν(θ)')
    a1.set_title(f'{fold}-fold {shape} ν(θ): {nu_lo:.2f} → {NU_HI:.2f}', fontsize=11)
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

    fig.suptitle(f'{CASE} / {base} — uniform patch, anisotropic {fold}-fold {shape.upper()} ν(θ) '
                 f'({nu_lo:.2f}…{NU_HI:.2f}, {TOPO} N={N}, REG={cfg["reg"]})', fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(C.savedir(CASE), base + '.png')
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f'saved {os.path.basename(path)}')
    print('done')


if __name__ == '__main__':
    for nm in (sys.argv[1:] or list(CONFIGS)):
        main(nm, CONFIGS[nm])
