"""
anisotropy / triangular_nu_maps — SUPPLEMENTARY material maps for the designed triangular-sawtooth
ν(θ) patch. Reads the saved triangular_nu.npz (no re-optimisation) and plots the LOAD-INDEPENDENT
material fields the main figure's single directional ν map does not show:

  (a) angle-averaged ⟨ν(θ)⟩ spatial map   — per-triangle mean Poisson ratio over all directions
  (b) angle-averaged ⟨E(θ)⟩ spatial map   — per-triangle mean Young's modulus over all directions
  (c) directional E(θ) line                — the angular map of stiffness (companion to ν(θ))
  (d) directional E(θ) polar               — same, mirrored to the full circle

The angle-averaged maps are a MATERIAL fingerprint (no applied load): each triangle's local physical
C6 is neighbourhood-smoothed, inverted to compliance, and ν(θ)/E(θ) averaged over θ (C.local_nuE_
angleavg). E(θ) comes from the region-homogenised C6 via C.nu_E_theta — the same tensor whose ν(θ)
the design fits. Outputs: triangular_nu_maps.png, triangular_nu_maps.csv (E(θ)).
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _common as C

CASE = 'anisotropy'


def main():
    npz = os.path.join(C.savedir(CASE), 'triangular_nu.npz')
    geo, k_bond, C6_per, meta = C.load_network(npz)
    th = C.ANG                                                   # θ grid ∈ [0,π]

    # --- angle-averaged local material maps (load-independent) ---
    nu_avg, E_avg = C.local_nuE_angleavg(geo, C6_per)           # each (nt,)

    # --- directional E(θ) from the region-homogenised physical tensor (same tensor design fits) ---
    C6_reg = C.sim_bulk_C6(geo)               # whole-cell physical 6-vector
    nu_theta, E_theta = C.nu_E_theta(C6_reg, th)

    print(f"  [triangular_nu_maps] tri={len(nu_avg)}", flush=True)
    print(f"    angle-avg ⟨ν⟩: median={np.median(nu_avg):+.3f}  [p5={np.percentile(nu_avg,5):+.3f}, "
          f"p95={np.percentile(nu_avg,95):+.3f}]", flush=True)
    print(f"    angle-avg ⟨E⟩: median={np.median(E_avg):.3f}  [p5={np.percentile(E_avg,5):.3f}, "
          f"p95={np.percentile(E_avg,95):.3f}]", flush=True)
    print(f"    directional E(θ): [{E_theta.min():.3f}, {E_theta.max():.3f}]  "
          f"anisotropy E_max/E_min={E_theta.max()/E_theta.min():.2f}", flush=True)

    C.write_csv(os.path.join(C.savedir(CASE), 'triangular_nu_maps.csv'),
                ['theta_deg', 'nu_theta', 'E_theta'],
                [(f'{np.degrees(t):.1f}', f'{nu_theta[i]:.4f}', f'{E_theta[i]:.4f}')
                 for i, t in enumerate(th)])

    fig = plt.figure(figsize=(13, 11))

    # (a) angle-averaged ν map
    a0 = fig.add_subplot(2, 2, 1)
    vlim = max(0.3, float(np.nanpercentile(np.abs(nu_avg), 97)))
    pc = C.fill_local_map(a0, geo, nu_avg, cmap='RdBu_r', sym=True, vlim=vlim)
    C.draw_box(a0, geo); plt.colorbar(pc, ax=a0, fraction=0.046)
    a0.set_title(r'angle-averaged $\langle\nu(\theta)\rangle$ map  —  median '
                 f'{np.median(nu_avg):+.2f}', fontsize=11)

    # (b) angle-averaged E map
    a1 = fig.add_subplot(2, 2, 2)
    pc = C.fill_local_map(a1, geo, E_avg, cmap='viridis')
    pc.set_clim(float(np.percentile(E_avg, 3)), float(np.percentile(E_avg, 97)))
    C.draw_box(a1, geo); plt.colorbar(pc, ax=a1, fraction=0.046)
    a1.set_title(r'angle-averaged $\langle E(\theta)\rangle$ map  —  median '
                 f'{np.median(E_avg):.2f}', fontsize=11)

    # (c) directional E(θ) line
    a2 = fig.add_subplot(2, 2, 3)
    a2.plot(np.degrees(th), E_theta, color='#2ca02c', lw=1.9)
    a2.set_xlabel('direction θ (deg)'); a2.set_ylabel('Young modulus E(θ)')
    a2.set_title(f'directional E(θ): [{E_theta.min():.2f}, {E_theta.max():.2f}]  '
                 f'(E$_{{max}}$/E$_{{min}}$={E_theta.max()/E_theta.min():.2f})', fontsize=11)
    a2.grid(alpha=.3); a2.set_ylim(bottom=0)

    # (d) directional E(θ) polar (mirrored to full circle)
    a3 = fig.add_subplot(2, 2, 4, projection='polar')
    th2 = np.concatenate([th, th + np.pi])
    a3.plot(th2, np.concatenate([E_theta, E_theta]), color='#2ca02c', lw=1.9)
    a3.set_title('E(θ) polar (mirrored)', fontsize=11, pad=16)

    fig.suptitle(f'{CASE} / triangular_nu — angle-averaged ⟨ν⟩ & ⟨E⟩ material maps + directional E(θ)',
                 fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(C.savedir(CASE), 'triangular_nu_maps.png')
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f'saved {os.path.basename(path)}')
    print('done')


if __name__ == '__main__':
    main()
