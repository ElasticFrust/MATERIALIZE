"""
Local ν and local E maps for EVERY saved anisotropy design — built by LOADING the saved networks
(anisotropy/networks/*.npz), no re-optimising. Per-test grids:
  A (program ν(θ)) : 3 profiles x 5 topologies
  B (isotropise)   : 3 bases x 7 target ν0
  C (independent)  : 2 cases x 2 topologies
Outputs: anisotropy_maps_{A,B,C}_{nu,E}.png
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import _common as C

ND = os.path.join(HERE, 'networks')
PROFILES = ['two_fold', 'four_fold', 'dir_aux']
TOPOS = ['regular', 'aniso_str', 'aniso_shr', 'disorder_lo', 'disorder_hi']
BASES = ['aniso_str', 'aniso_shr', 'disorder_hi']
NU0S = [0.80, 0.50, 0.33, 0.00, -0.20, -0.50, -0.80]
C_CASES = ['E-anisotropic_nu-isotropic', 'nu-anisotropic_E-isotropic']
C_TOPOS = ['regular', 'aniso_str']


def L(name):
    return C.load_network(os.path.join(ND, name + '.npz'))          # geo, k, C6, meta


def grid(cells, nrows, ncols, quant, path, title):
    cmap = 'RdBu_r' if quant == 'nu' else 'viridis'
    vlim = 0.6 if quant == 'nu' else None
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.7 * ncols, 3.0 * nrows), squeeze=False)
    pc = None
    for ax, cell in zip(axes.ravel(), cells):
        if cell is None:
            ax.axis('off'); continue
        name, lab = cell
        geo, k, C6, meta = L(name)
        val = C.local_field_smooth(geo, C6, quant)
        pc = C.fill_local_map(ax, geo, val, cmap=cmap, sym=(quant == 'nu'), vlim=vlim)
        if quant == 'E':
            pc.set_clim(0, np.nanpercentile(val, 97))
        C.draw_box(ax, geo); ax.set_title(lab, fontsize=8)
    if pc is not None and quant == 'nu':
        fig.colorbar(pc, ax=axes, fraction=0.02, label='local ν', pad=0.01)
    fig.suptitle(title, fontsize=12)
    plt.savefig(path, dpi=140, bbox_inches='tight'); plt.close()
    print('saved', os.path.basename(path))


def main():
    # ---- Test A: 3 profiles x 5 topologies ----
    cellsA = [(f'A_{p}_{t}', f'{p}\n{t}') for p in PROFILES for t in TOPOS]
    grid(cellsA, len(PROFILES), len(TOPOS), 'nu', os.path.join(HERE, 'anisotropy_maps_A_nu.png'),
         'anisotropy TEST A (program ν(θ)) — local ν: profiles (rows) x topologies (cols)')
    grid(cellsA, len(PROFILES), len(TOPOS), 'E', os.path.join(HERE, 'anisotropy_maps_A_E.png'),
         'anisotropy TEST A — local E: profiles (rows) x topologies (cols)')

    # ---- Test B: 3 bases x 7 nu0 ----
    cellsB = [(f'B_{b}_nu{n:+.2f}', f'{b}\nν0={n:+.2f}') for b in BASES for n in NU0S]
    grid(cellsB, len(BASES), len(NU0S), 'nu', os.path.join(HERE, 'anisotropy_maps_B_nu.png'),
         'anisotropy TEST B (isotropise) — local ν: bases (rows) x target ν0 (cols)')
    grid(cellsB, len(BASES), len(NU0S), 'E', os.path.join(HERE, 'anisotropy_maps_B_E.png'),
         'anisotropy TEST B — local E: bases (rows) x target ν0 (cols)')

    # ---- Test C: 2 cases x 2 topologies ----
    cellsC = [(f'C_{c}_{t}', f'{c}\n{t}') for c in C_CASES for t in C_TOPOS]
    grid(cellsC, len(C_CASES), len(C_TOPOS), 'nu', os.path.join(HERE, 'anisotropy_maps_C_nu.png'),
         'anisotropy TEST C (independent E/ν) — local ν')
    grid(cellsC, len(C_CASES), len(C_TOPOS), 'E', os.path.join(HERE, 'anisotropy_maps_C_E.png'),
         'anisotropy TEST C — local E')
    print('done')


if __name__ == '__main__':
    main()
