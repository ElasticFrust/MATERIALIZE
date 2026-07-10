"""
anisotropy / crystal_grid — directional response of 32 CRYSTALLINE lattices over a (phi, psi) grid.

For each of 8 phi in (0,4] x 4 psi in (0,4] (=32 sheared single-site crystals, make_crystal, k=1) the
homogenised response is exact on a tiny patch (W=0, affine — see crystal_response.py). Each crystal is
the regular lattice sheared/stretched with topology preserved, so phi is a REAL shear (not Delaunay-
reduced away). We read nu(theta), E(theta)
via the code's PBC relaxation and show:
  crystal_grid.png         — 4x8 grid, one panel per (phi,psi): nu(theta) (red, left axis) and
                             E(theta) (green, right axis), each panel autoscaled.
  crystal_grid_summary.png — heatmaps over the grid: anisotropy E_max/E_min, min nu(theta),
                             max nu(theta) — the trends across the 32 cases at a glance.
Outputs crystal_grid.csv (per-case stats).
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _common as C

CASE = 'anisotropy'
PHIS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]     # shear offset per row (cols)
PSIS = [1.0, 2.0, 3.0, 4.0]                          # row-height scale (rows)
HALF = 4.0


def response(phi, psi):
    geo = C.make_crystal(phi, psi, half=HALF)           # topology-preserving shear (see crystal_response)
    geo['bond_k'] = np.ones(len(geo['bond_R'])); geo['tri_k'] = geo['bond_k'][geo['tri_bond']]
    C6 = C.sim_region_C6(geo)                        # exact (W=0); homogenised physical 6-vector
    nu, E = C.nu_E_theta(C6, C.ANG)
    return nu, E


def main():
    deg = np.degrees(C.ANG)
    nR, nC = len(PSIS), len(PHIS)
    NU = np.empty((nR, nC), object); EE = np.empty((nR, nC), object)
    anis = np.zeros((nR, nC)); numin = np.zeros((nR, nC)); numax = np.zeros((nR, nC))
    rows = []
    for i, psi in enumerate(PSIS):
        for j, phi in enumerate(PHIS):
            nu, E = response(phi, psi)
            NU[i, j], EE[i, j] = nu, E
            anis[i, j] = E.max() / E.min(); numin[i, j] = nu.min(); numax[i, j] = nu.max()
            rows.append((f'{phi:g}', f'{psi:g}', f'{nu.min():.4f}', f'{nu.max():.4f}',
                         f'{E.min():.4f}', f'{E.max():.4f}', f'{anis[i, j]:.3f}'))
            print(f"  phi={phi:<4g} psi={psi:<4g}  nu[{nu.min():+.3f},{nu.max():+.3f}]  "
                  f"E[{E.min():.3f},{E.max():.3f}]  Emax/Emin={anis[i, j]:.2f}", flush=True)

    C.write_csv(os.path.join(C.savedir(CASE), 'crystal_grid.csv'),
                ['phi', 'psi', 'nu_min', 'nu_max', 'E_min', 'E_max', 'E_anisotropy'], rows)

    # --- grid of per-case nu(theta) & E(theta) ---
    fig, axes = plt.subplots(nR, nC, figsize=(24, 12), squeeze=False)
    for i, psi in enumerate(PSIS):
        for j, phi in enumerate(PHIS):
            ax = axes[i, j]; ax2 = ax.twinx()
            ax.plot(deg, NU[i, j], color='#d62728', lw=1.6)
            ax.axhline(0, color='0.7', lw=.5)
            ax2.plot(deg, EE[i, j], color='#2ca02c', lw=1.6)
            ax.set_xlim(0, 180); ax.set_xticks([0, 90, 180])
            ax.tick_params(labelsize=7); ax2.tick_params(labelsize=7, colors='#2ca02c')
            ax.tick_params(axis='y', colors='#d62728')
            ax.set_title(f'φ={phi:g}  ψ={psi:g}', fontsize=9)
            if i == nR - 1:
                ax.set_xlabel('θ (deg)', fontsize=8)
            if j == 0:
                ax.set_ylabel('ν(θ)', color='#d62728', fontsize=8)
            if j == nC - 1:
                ax2.set_ylabel('E(θ)', color='#2ca02c', fontsize=8)
    fig.suptitle('anisotropy / crystal_grid — ν(θ) (red) & E(θ) (green) for 32 crystals, k=1, W=0 exact',
                 fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    p1 = os.path.join(C.savedir(CASE), 'crystal_grid.png')
    plt.savefig(p1, dpi=130, bbox_inches='tight'); plt.close()
    print(f'saved {os.path.basename(p1)}')

    # --- summary heatmaps over the (phi,psi) grid ---
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    ext = [PHIS[0] - 0.25, PHIS[-1] + 0.25, PSIS[0] - 0.5, PSIS[-1] + 0.5]
    panels = [(anis, 'E_max / E_min  (anisotropy)', 'viridis', None),
              (numin, 'min ν(θ)', 'RdBu_r', 'sym'),
              (numax, 'max ν(θ)', 'RdBu_r', 'sym')]
    for ax, (M, ttl, cmap, mode) in zip(axs, panels):
        vlim = np.abs(M).max() if mode == 'sym' else None
        im = ax.imshow(M, origin='lower', aspect='auto', extent=ext, cmap=cmap,
                       vmin=-vlim if vlim else None, vmax=vlim if vlim else None)
        ax.set_xticks(PHIS); ax.set_yticks(PSIS)
        ax.set_xlabel('φ (shear)'); ax.set_ylabel('ψ (row height)'); ax.set_title(ttl, fontsize=11)
        for i, psi in enumerate(PSIS):
            for j, phi in enumerate(PHIS):
                ax.text(phi, psi, f'{M[i, j]:.2f}', ha='center', va='center', fontsize=6.5,
                        color='w' if (mode != 'sym' and M[i, j] > M.mean()) else 'k')
        plt.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle('anisotropy / crystal_grid — trends over the (φ, ψ) grid (32 crystals, k=1)', fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    p2 = os.path.join(C.savedir(CASE), 'crystal_grid_summary.png')
    plt.savefig(p2, dpi=140, bbox_inches='tight'); plt.close()
    print(f'saved {os.path.basename(p2)}')


if __name__ == '__main__':
    main()
