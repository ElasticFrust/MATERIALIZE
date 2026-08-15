"""
Re-plot the saved large16k_rings "bullseye" networks with two extra views, loaded from the saved
.npz (no re-optimising): (1) the STRAIN response -- local dilation e=(ε_xx+ε_yy)/2 under the same
isotropic stretch (complementary to the stress bands: stiff/high-stress bands strain little, soft
bands strain more); and (2) the material COMPOSITION -- local angle-averaged E and ν maps (how the
alternating rigidity is laid down). One relaxation + one angle-average per topology, no minimiser.
Outputs strain_stress_large16k_rings_<topo>_strain.png and _composition.png.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _common as C

DIR = C.savedir('strain_stress')
TOPOS = ['regular', 'disorder_hi']


def _finite_pct(a, q):
    f = np.abs(a)[np.isfinite(a)]
    return float(np.percentile(f, q)) if f.size else 1.0


def replot(topo):
    geo, k, C6, meta = C.load_network(os.path.join(DIR, f'large16k_rings_{topo}.npz'))
    regions = meta.get('regions', [])

    # (1) STRAIN response: dilation under the ISOTROPIC (ε_xx+ε_yy) stretch, from the separate NumPy sim
    eps, _ = C.unit_mode_response(geo)
    eps_iso = C.MO.vec3(eps[0] + eps[1])
    dil = 0.5 * (eps_iso[:, 0] + eps_iso[:, 2])
    fig, ax = plt.subplots(figsize=(7.5, 7))
    v = _finite_pct(dil, 97)
    pc = C.fill_local_map(ax, geo, np.nan_to_num(dil, nan=0.0, posinf=v, neginf=-v), cmap='RdBu_r', sym=True, vlim=v)
    C.draw_box(ax, geo); C.mark_region(ax, regions)
    plt.colorbar(pc, ax=ax, fraction=0.046)
    ax.set_title(f'strain_stress — BULLSEYE ({topo}): local dilation e=(ε_xx+ε_yy)/2 under isotropic '
                 f'stretch\n(complementary to the stress bands: stiff bands strain LESS)', fontsize=10)
    plt.tight_layout()
    out = os.path.join(DIR, f'strain_stress_large16k_rings_{topo}_strain.png')
    plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    print(f'saved {os.path.basename(out)}', flush=True)

    # (2) COMPOSITION: local angle-averaged nu and E material maps
    nu_avg, E_avg = C.local_nuE_angleavg(geo, C6)
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(13, 6.2))
    vnu = max(_finite_pct(nu_avg, 96), 0.1)
    pc0 = C.fill_local_map(a0, geo, np.nan_to_num(nu_avg, nan=0.0, posinf=vnu, neginf=-vnu),
                           cmap='RdBu_r', sym=True, vlim=vnu)
    C.draw_box(a0, geo); C.mark_region(a0, regions)
    plt.colorbar(pc0, ax=a0, fraction=0.046); a0.set_title('local angle-averaged ν  (blue = auxetic)', fontsize=11)
    vE = max(_finite_pct(E_avg, 96), 1e-6)
    pc1 = C.fill_local_map(a1, geo, np.nan_to_num(E_avg, nan=0.0, posinf=vE, neginf=0.0), cmap='magma')
    pc1.set_clim(0, vE)
    C.draw_box(a1, geo); C.mark_region(a1, regions)
    plt.colorbar(pc1, ax=a1, fraction=0.046); a1.set_title('local angle-averaged E', fontsize=11)
    fig.suptitle(f'strain_stress — BULLSEYE ({topo}): material composition (local angle-averaged ν and E)',
                 fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(DIR, f'strain_stress_large16k_rings_{topo}_composition.png')
    plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    print(f'saved {os.path.basename(out)}', flush=True)


def main():
    for topo in TOPOS:
        replot(topo)
    print('done')


if __name__ == '__main__':
    main()
