"""
FAST edge-forcing check: CUT the periodic patch design into an OPEN sheet and STRETCH it along the
boundary only. Clamp the left edge (u_x=0), pull the right edge (u_x=Δ), leave top/bottom and the
whole interior free, and relax as a linear spring truss (axial stiffness κ_e=k_e). This applies the
load as an EXTERNAL EDGE effect (not a bulk/periodic affine strain). We then draw the per-triangle
strain & stress and check the auxetic region's lateral response (ε_yy same sign as ε_xx ⇒ auxetic).
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
import test_cluster_Ceff as CE
import test_cluster_rigidity as TR

TOPOS = ['regular', 'disorder_hi']


def stress(bare, eps):
    A0, A1, A2, A3, A4 = (bare[:, i] for i in range(5))
    exx, eyy, exy = eps[:, 0, 0], eps[:, 1, 1], eps[:, 0, 1]; s = np.zeros_like(eps)
    s[:, 0, 0] = A0 * exx + 2 * A1 * exy + A2 * eyy
    s[:, 0, 1] = s[:, 1, 0] = A1 * exx + 2 * A2 * exy + A3 * eyy
    s[:, 1, 1] = A2 * exx + 2 * A3 * exy + A4 * eyy
    return s


def mag(t):
    return np.sqrt(t[:, 0, 0] ** 2 + 2 * t[:, 0, 1] ** 2 + t[:, 1, 1] ** 2)


def main():
    fig, axes = plt.subplots(len(TOPOS), 2, figsize=(13, 6.2 * len(TOPOS)), squeeze=False)
    for r, topo in enumerate(TOPOS):
        geo, k, C6, meta = C.load_network(os.path.join(HERE, 'networks', f'patch__{topo}.npz'))
        u, nwt = C.open_stretch(geo, axis=0)
        eps = CE.tri_metric_change(geo['edge_vecs'], geo['simplices'], np.eye(2), u)  # (N,2,2)
        sig = stress(TR.bare_tensor(geo), eps)
        cen = np.asarray(geo['centroids']); RE, RN = meta['region']
        inRE = ((cen - RE['center']) ** 2).sum(1) < RE['radius'] ** 2
        inRN = ((cen - RN['center']) ** 2).sum(1) < RN['radius'] ** 2
        for nm, sel in [('whole', nwt), ('R_E stiff', nwt & inRE), ('R_nu aux', nwt & inRN)]:
            exx = eps[sel, 0, 0].mean(); eyy = eps[sel, 1, 1].mean()
            print(f"  {topo:11s} {nm:10s}: <exx>={exx:+.4f} <eyy>={eyy:+.4f}  local nu=-eyy/exx={-eyy/exx:+.3f}"
                  f"  {'AUXETIC' if eyy > 0 else ''}", flush=True)
        gp = {'tri_verts': np.asarray(geo['tri_verts'])[nwt], 'BL1': geo['BL1'], 'BL2': geo['BL2']}
        for col, (fld, lab) in enumerate([(mag(eps)[nwt], '‖strain‖'), (mag(sig)[nwt], '‖stress‖')]):
            pc = C.fill_local_map(axes[r, col], gp, fld, cmap='magma')
            pc.set_clim(0, np.nanpercentile(fld, 98)); C.draw_box(axes[r, col], geo)
            C.mark_region(axes[r, col], meta.get('region'))
            plt.colorbar(pc, ax=axes[r, col], fraction=0.046)
            axes[r, col].set_title(f'{topo}: {lab} (open, edge-stretched →)', fontsize=10)
    fig.suptitle('CUT & STRETCH along the boundary (open sheet, left clamped / right pulled) — actual '
                 'simulation response\ncyan = stiff-E patch, lime = auxetic-ν patch', fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(HERE, 'large16k_cut_stretch.png'), dpi=140, bbox_inches='tight')
    plt.close(); print('saved large16k_cut_stretch.png')


if __name__ == '__main__':
    main()
