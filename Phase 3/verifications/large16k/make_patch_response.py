"""
ACTUAL simulation response of the large decoupled-PATCH design: per-triangle STRAIN and STRESS fields
under three macroscopic forcings — isotropic dilation, pure (deviatoric) shear, and simple
(longitudinal) shear. Computed from the real PBC relaxation (physical_homog.relax), not the
homogenised tensor. Loaded from the saved networks.

For each forcing we apply the macroscopic strain, relax the network, and draw the resulting local
strain magnitude ‖ε(s)‖ and stress magnitude ‖σ(s)‖ (σ = A(s):ε(s), A the per-triangle spring
Hessian). Regions marked: cyan = stiff-E patch, lime = auxetic-ν patch.
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
import physical_homog as PH
import test_cluster_rigidity as TR
import test_cluster_Ceff as CE

TOPOS = ['regular', 'disorder_hi']
# forcing name -> combination of the xx,yy,xy unit-mode strains (eps0,eps1,eps2)
FORCINGS = [('isotropic dilation', (1.0, 1.0, 0.0)),
            ('pure shear (deviatoric)', (1.0, -1.0, 0.0)),
            ('simple/longitudinal shear', (0.0, 0.0, 2.0))]


def augment(geo):
    """Rebuild edge_vecs / actual_len2 (not saved) from the image-correct tri_verts."""
    tv = np.asarray(geo['tri_verts']); p0, p1, p2 = tv[:, 0], tv[:, 1], tv[:, 2]
    geo['edge_vecs'] = np.stack([p1 - p0, p2 - p0, p2 - p1], 1)
    geo['actual_len2'] = (geo['edge_vecs'] ** 2).sum(2)
    geo['simplices'] = np.asarray(geo['simplices'])
    return geo


def unit_mode_strains(geo):
    """Per-triangle total strain (N,2,2) for each unit macro mode xx, yy, xy (from the relaxation)."""
    ev, sx = geo['edge_vecs'], geo['simplices']
    free = np.arange(2, 2 * len(geo['pts']))
    u = PH.relax(geo, free, TR.assemble_K_faff)
    return [CE.tri_metric_change(ev, sx, PH.Fk[k], u[k]) / PH.DELTA for k in range(3)]


def stress(bare, eps):
    """σ(s) = A(s):ε(s) from the bare 5-vector A=[xxxx,xxxy,xxyy,xyyy,yyyy] (full-symmetric 4-tensor)."""
    A0, A1, A2, A3, A4 = (bare[:, i] for i in range(5))
    exx, eyy, exy = eps[:, 0, 0], eps[:, 1, 1], eps[:, 0, 1]
    sxx = A0 * exx + 2 * A1 * exy + A2 * eyy
    sxy = A1 * exx + 2 * A2 * exy + A3 * eyy
    syy = A2 * exx + 2 * A3 * exy + A4 * eyy
    s = np.zeros_like(eps); s[:, 0, 0] = sxx; s[:, 1, 1] = syy; s[:, 0, 1] = s[:, 1, 0] = sxy
    return s


def mag(t):
    return np.sqrt(t[:, 0, 0] ** 2 + 2 * t[:, 0, 1] ** 2 + t[:, 1, 1] ** 2)


def main():
    data = {}
    for topo in TOPOS:
        geo, k, C6, meta = C.load_network(os.path.join(HERE, 'networks', f'patch__{topo}.npz'))
        augment(geo)
        eps = unit_mode_strains(geo); bare = TR.bare_tensor(geo)
        data[topo] = (geo, meta.get('region'), eps, bare)

    fig, axes = plt.subplots(len(FORCINGS), 4, figsize=(20, 5.6 * len(FORCINGS)), squeeze=False)
    for r, (fname, (c0, c1, c2)) in enumerate(FORCINGS):
        for j, topo in enumerate(TOPOS):
            geo, region, eps, bare = data[topo]
            e = c0 * eps[0] + c1 * eps[1] + c2 * eps[2]
            s = stress(bare, e)
            me, ms = mag(e), mag(s)
            pe = C.fill_local_map(axes[r, 2 * j], geo, me, cmap='magma')
            pe.set_clim(0, np.nanpercentile(me, 98)); C.draw_box(axes[r, 2 * j], geo)
            C.mark_region(axes[r, 2 * j], region)
            plt.colorbar(pe, ax=axes[r, 2 * j], fraction=0.046)
            ps = C.fill_local_map(axes[r, 2 * j + 1], geo, ms, cmap='magma')
            ps.set_clim(0, np.nanpercentile(ms, 98)); C.draw_box(axes[r, 2 * j + 1], geo)
            C.mark_region(axes[r, 2 * j + 1], region)
            plt.colorbar(ps, ax=axes[r, 2 * j + 1], fraction=0.046)
            if r == 0:
                axes[r, 2 * j].set_title(f'{topo}: ‖strain‖', fontsize=10)
                axes[r, 2 * j + 1].set_title(f'{topo}: ‖stress‖', fontsize=10)
        axes[r, 0].set_ylabel(fname, fontsize=11)
    fig.suptitle('Decoupled-PATCH design (16k) — actual simulation response: local strain & stress '
                 'under isotropic dilation / pure shear / simple shear\n'
                 '(cyan = stiff-E region, lime = auxetic-ν region)', fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(os.path.join(HERE, 'large16k_patch_response.png'), dpi=140, bbox_inches='tight')
    plt.close(); print('saved large16k_patch_response.png')


if __name__ == '__main__':
    main()
