"""
anisotropy / effective_superlattice — carve the oblique crystal OUT of a disordered mesh.

The idea (the structure a well-behaved inverse solver SHOULD find, instead of a random mechanism):
keep STIFF only the disordered bonds whose direction approximates one of the target crystal's bond
directions, and let every other bond go FLOPPY (k -> small). The stiff sub-skeleton is an effective
oblique superlattice embedded in the disorder, so its homogenised response should approximate the sharp
leaning ν(θ) of make_crystal(4,1) -- WITHOUT hand-placing the crystal (the mesh is a fixed disordered
graph; only k is assigned, by directional alignment).

k_b = k_floor + (1-k_floor)*exp(-Δθ_b^2 / 2σ^2),  Δθ_b = angle from bond b to the nearest crystal bond
direction (mod π). This is a DIRECT construction (no optimisation) to test whether the concept holds;
it reports how close the effective-superlattice ν(θ) gets to the target and how floppy the rest is.
Saves effective_superlattice.png.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _common as C

CASE = 'anisotropy'
PHI, PSI = 4.0, 1.0
SIGMA_DEG = 7.0                          # alignment width (deg)
K_FLOORS = [0.05, 0.02, 0.008, 0.003, 0.001]   # sweep: "floppy" residual stiffness -> 0


def unique_dirs(bond_R, tol=1e-3):
    """Distinct bond DIRECTIONS (angle mod π) of a lattice."""
    a = np.mod(np.arctan2(bond_R[:, 1], bond_R[:, 0]), np.pi)
    u = []
    for x in a:
        if not any(abs((x - y + np.pi / 2) % np.pi - np.pi / 2) < tol for y in u):
            u.append(x)
    return np.array(sorted(u))


def main():
    th = C.ANG
    # target: the pointy leaning crystal, and its bond DIRECTIONS
    cg = C.make_crystal(PHI, PSI, half=4.0)
    cg['bond_k'] = np.ones(len(cg['bond_R'])); cg['tri_k'] = cg['bond_k'][cg['tri_bond']]
    target, E_target = C.nu_E_theta(C.sim_region_C6(cg), th)
    dirs = unique_dirs(cg['bond_R'])
    print(f"  [effective_superlattice] crystal bond directions (deg): "
          f"{np.round(np.degrees(dirs), 1)}", flush=True)

    # disordered mesh: assign k by directional alignment to those crystal directions; sweep k_floor
    geo = C.make_lattice(1.0, 1.0, half=8.0, eta=0.35, seed=1)
    a = np.mod(np.arctan2(geo['bond_R'][:, 1], geo['bond_R'][:, 0]), np.pi)
    dtheta = np.min(np.abs((a[:, None] - dirs[None, :] + np.pi / 2) % np.pi - np.pi / 2), axis=1)
    align = np.exp(-dtheta ** 2 / (2 * np.radians(SIGMA_DEG) ** 2))       # 1 = aligned, 0 = off-axis
    stiff_frac = (align > 0.5).mean()
    print(f"    aligned (stiff) bonds: {100*stiff_frac:.0f}% of {len(align)}; target ν(θ): "
          f"[{target.min():+.3f},{target.max():+.3f}]", flush=True)

    deg = np.degrees(th)
    results = []                                              # (kfloor, nu, E, err, k)
    for kf in K_FLOORS:
        k = kf + (1 - kf) * align
        geo['bond_k'] = k; geo['tri_k'] = k[geo['tri_bond']]
        nu, E = C.nu_E_theta(C.sim_region_C6(geo), th)
        err = float(np.abs(nu - target).max())
        results.append((kf, nu, E, err, k.copy()))
        print(f"    k_floor={kf:<6g}: effective ν(θ) [{nu.min():+.3f},{nu.max():+.3f}]  "
              f"E_anis={E.max()/E.min():6.1f}  maxerr={err:.3f}", flush=True)

    C.write_csv(os.path.join(C.savedir(CASE), 'effective_superlattice.csv'),
                ['theta_deg', 'nu_target'] + [f'nu_kf{kf:g}' for kf in K_FLOORS],
                [tuple([f'{np.degrees(t):.1f}', f'{target[i]:.4f}'] +
                       [f'{r[1][i]:.4f}' for r in results]) for i, t in enumerate(th)])

    fig = plt.figure(figsize=(18, 5.5))
    a0 = fig.add_subplot(1, 3, 1)                             # network at the softest stable k_floor
    kf_show, k_show = results[-1][0], results[-1][4]
    lc = C.draw_network(a0, geo, k_show, cmap='viridis', lw_scale=4.5)
    plt.colorbar(lc, ax=a0, fraction=0.046, label='k')
    a0.set_title(f'effective superlattice (k_floor={kf_show:g}) — aligned stiff, rest floppy', fontsize=11)

    a1 = fig.add_subplot(1, 3, 2)
    a1.plot(deg, target, 'k--', lw=2.4, label='target (crystal)')
    cmap = plt.cm.plasma(np.linspace(0.15, 0.85, len(results)))
    for (kf, nu, E, err, _), col in zip(results, cmap):
        a1.plot(deg, nu, color=col, lw=1.5, label=f'k_floor={kf:g} (err {err:.1f})')
    a1.axhline(0, color='0.6', lw=.6); a1.set_xlim(0, 180); a1.set_xticks(range(0, 181, 45))
    a1.set_xlabel('θ (deg)'); a1.set_ylabel('ν(θ)'); a1.grid(alpha=.3); a1.legend(fontsize=7)
    a1.set_title('softer floppy bonds → larger amplitude (toward target)', fontsize=11)

    a2 = fig.add_subplot(1, 3, 3)
    kfs = [r[0] for r in results]
    a2.semilogx(kfs, [r[1].max() for r in results], 'o-', color='#d62728', label='effective ν max')
    a2.axhline(target.max(), color='k', ls='--', lw=1.5, label=f'target ν max={target.max():.1f}')
    a2.set_xlabel('k_floor (floppy stiffness)'); a2.set_ylabel('ν max')
    a2.grid(alpha=.3, which='both'); a2.legend(fontsize=8); a2.invert_xaxis()
    a2.set_title('amplitude vs floppiness (softer → closer, until it destabilises)', fontsize=11)

    fig.suptitle(f'{CASE} / effective_superlattice — keep only crystal-aligned bonds stiff on a '
                 f'DISORDERED mesh (floppy elsewhere), sweep floppiness', fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(C.savedir(CASE), 'effective_superlattice.png')
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f'saved {os.path.basename(path)}')


if __name__ == '__main__':
    main()
