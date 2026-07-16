"""
showcase / fig5 — BULLSEYE strain response. A ~16k-triangle network designed with concentric rings of
alternating rigidity; under a uniform isotropic stretch the LOCAL strain (dilation e=(ε_xx+ε_yy)/2)
organises into matching rings — soft rings strain more, stiff rings strain less. Left: the designed
material (local angle-averaged Young's modulus E, showing the rings). Right: the computed strain
response. Self-contained: loads the saved network, computes the strain from the independent PBC sim.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
VER = os.path.dirname(HERE)
sys.path.insert(0, VER)
import _common as C

SS = C.savedir('strain_stress')
TOPO = 'regular'                                          # cleanest bullseye


def _pct(a, q):
    f = np.abs(a)[np.isfinite(a)]
    return float(np.percentile(f, q)) if f.size else 1.0


def main():
    geo, k, C6, meta = C.load_network(os.path.join(SS, f'large16k_rings_{TOPO}.npz'))
    regions = meta.get('regions', [])

    # material: local angle-averaged E (the rings)
    _, E_avg = C.local_nuE_angleavg(geo, C6)
    # strain response: dilation under the isotropic (ε_xx+ε_yy) stretch, from the NumPy PBC sim
    eps, _ = C.unit_mode_response(geo)
    eps_iso = C.CE.vec3(eps[0] + eps[1])
    dil = 0.5 * (eps_iso[:, 0] + eps_iso[:, 2])

    plt.rcParams.update({'font.size': 12})
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(14, 7))

    vE = max(_pct(E_avg, 97), 1e-6)
    pc0 = C.fill_local_map(a0, geo, np.nan_to_num(E_avg, nan=0.0, posinf=vE, neginf=0.0), cmap='magma')
    pc0.set_clim(0, vE); C.draw_box(a0, geo); C.mark_region(a0, regions)
    plt.colorbar(pc0, ax=a0, fraction=0.046, label='local Young modulus E')
    a0.set_title(f'designed material — concentric rigidity rings ({int(len(geo["simplices"])/1000)}k triangles)',
                 fontsize=12)

    v = _pct(dil, 97)
    pc1 = C.fill_local_map(a1, geo, np.nan_to_num(dil, nan=0.0, posinf=v, neginf=-v),
                           cmap='RdBu_r', sym=True, vlim=v)
    C.draw_box(a1, geo); C.mark_region(a1, regions)
    plt.colorbar(pc1, ax=a1, fraction=0.046, label='local dilation  e = (ε_xx+ε_yy)/2')
    a1.set_title('strain response under uniform isotropic stretch\n(soft rings strain more, stiff rings less)',
                 fontsize=12)

    print(f"  bullseye {TOPO}: tri={len(geo['simplices'])}, dilation range "
          f"[{np.nanmin(dil):+.3f},{np.nanmax(dil):+.3f}]", flush=True)
    fig.suptitle('Programmed strain response: a designed "bullseye" and the strain field it produces '
                 'under load', fontsize=14, y=1.0)
    plt.tight_layout()
    out = os.path.join(HERE, 'fig5_bullseye_strain.png')
    plt.savefig(out, dpi=200, bbox_inches='tight'); plt.close()
    print(f'saved {os.path.basename(out)}')


if __name__ == '__main__':
    main()
