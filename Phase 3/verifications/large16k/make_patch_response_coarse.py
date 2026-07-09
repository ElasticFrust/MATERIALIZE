"""
COARSE-GRAINED strain and stress response of the decoupled-PATCH design under the three forcings.
Same as make_patch_response.py but each per-triangle strain/stress TENSOR is area-weighted averaged
over blocks of a coarse grid (opposing local strains cancel), then the magnitude is drawn — this
removes the near-zero-k floppy-bond outliers that dominate the fine field, revealing the effective
mesoscale response (the stiff-E region then carries LOW strain / HIGH stress, as expected).
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(HERE, '..', '..', '..', 'verification_tools'))
sys.path.insert(0, os.path.join(HERE, '..', '..', '..', 'Phase 2'))
import _common as C

TOPOS = ['regular', 'disorder_hi']
NCELL = 22
FORCINGS = [('isotropic dilation', (1.0, 1.0, 0.0)),
            ('pure shear (deviatoric)', (1.0, -1.0, 0.0)),
            ('simple/longitudinal shear', (0.0, 0.0, 2.0))]


def coarse(geo, tens):
    """area-weighted block-average of a per-triangle 2x2 tensor field on an NCELL x NCELL grid."""
    frac = C.to_square(geo, np.asarray(geo['centroids'])); areas = np.asarray(geo['areas'])
    ix = np.clip((frac[:, 0] * NCELL).astype(int), 0, NCELL - 1)
    iy = np.clip((frac[:, 1] * NCELL).astype(int), 0, NCELL - 1)
    b = ix * NCELL + iy; out = np.zeros_like(tens)
    for bb in np.unique(b):
        m = b == bb; w = areas[m]
        out[m] = (tens[m] * w[:, None, None]).sum(0) / w.sum()
    return out


def main():
    data = {}
    for topo in TOPOS:
        geo, k, C6, meta = C.load_network(os.path.join(HERE, 'networks', f'patch__{topo}.npz'))
        eps, sig = C.unit_mode_response(geo)
        data[topo] = (geo, meta.get('region'), eps, sig)

    fig, axes = plt.subplots(len(FORCINGS), 4, figsize=(20, 5.6 * len(FORCINGS)), squeeze=False)
    for r, (fname, (c0, c1, c2)) in enumerate(FORCINGS):
        for j, topo in enumerate(TOPOS):
            geo, region, eps, sig = data[topo]
            e = coarse(geo, c0 * eps[0] + c1 * eps[1] + c2 * eps[2])
            s = coarse(geo, c0 * sig[0] + c1 * sig[1] + c2 * sig[2])
            me, ms = C.tensor_mag(e), C.tensor_mag(s)
            pe = C.fill_local_map(axes[r, 2 * j], geo, me, cmap='magma')
            pe.set_clim(0, np.nanpercentile(me, 99)); C.draw_box(axes[r, 2 * j], geo)
            C.mark_region(axes[r, 2 * j], region); plt.colorbar(pe, ax=axes[r, 2 * j], fraction=0.046)
            ps = C.fill_local_map(axes[r, 2 * j + 1], geo, ms, cmap='magma')
            ps.set_clim(0, np.nanpercentile(ms, 99)); C.draw_box(axes[r, 2 * j + 1], geo)
            C.mark_region(axes[r, 2 * j + 1], region); plt.colorbar(ps, ax=axes[r, 2 * j + 1], fraction=0.046)
            if r == 0:
                axes[r, 2 * j].set_title(f'{topo}: coarse ‖strain‖', fontsize=10)
                axes[r, 2 * j + 1].set_title(f'{topo}: coarse ‖stress‖', fontsize=10)
        axes[r, 0].set_ylabel(fname, fontsize=11)
    fig.suptitle(f'Decoupled-PATCH (16k) — COARSE-GRAINED strain & stress ({NCELL}×{NCELL} blocks) under '
                 'dilation / pure shear / simple shear\n'
                 'cyan = stiff-E (E=1.8), lime = auxetic-ν (ν=−0.3) — mesoscale: stiff carries low strain, high stress',
                 fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(os.path.join(HERE, 'large16k_patch_response_coarse.png'), dpi=140, bbox_inches='tight')
    plt.close(); print('saved large16k_patch_response_coarse.png')


if __name__ == '__main__':
    main()
