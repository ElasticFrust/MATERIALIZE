"""
Diagnostic: is the EMBEDDED reading's sign flip (center reads strongly POSITIVE nu(theta=0) when
measured as a region-average within the glued, periodically-strained composite, vs strongly
NEGATIVE when measured SOLO/isolated) an interfacial boundary-layer artifact that decays away from
the bond line, or does it persist through the ribbon's whole width?

Mechanism under test: under global x-strain, the center ribbon "wants" eyy>0 (auxetic) at the same
x-location the bonded plain ribbons "want" eyy<0 (normal) -- displacement continuity at the bonded
interface forbids both simultaneously. In 3D this relieves via out-of-plane bending near the seam;
this is a pure in-plane (2D) lattice with no such degree of freedom, so the mismatch can only
resolve via in-plane shear/strain redistribution concentrated near the interface -- i.e. a boundary
layer. If that's the whole story, the DEEP-BULK core of the (relatively thin, ~22-wide) center
ribbon should read closer to the SOLO value than the NEAR-INTERFACE band does.

Result: it does NOT decay this way -- the deep-bulk core is FURTHER from solo than the
near-interface band, on both topologies. Likely explanation: the center material's own design
target pushes it into elastic extremity (E spans >10x across loading angles, |nu|>2 -- close to a
soft-mode/instability threshold), where elastic fields decay slowly with distance, plausibly slower
than this ribbon's own width. So the interfacial-incompatibility MECHANISM is still the most likely
cause, but the redistribution zone appears to span the whole ribbon rather than a thin edge layer
for a design this extreme -- confirming the DECAY LENGTH would need a proper multi-width study
(not done here). What doesn't depend on this open question: the PRACTICAL (direct open x-stretch)
reading shows the intended auxetic sign, clearly separated from the plain zones, on both
topologies -- see dir_aux_ribbon.csv / dir_aux_ribbon_nutheta_{tag}.png.

Uses the already-saved dir_aux_ribbon_{tag}.npz and dir_aux_ribbon_center_solo_profile_{tag}.npz
from three_ribbon.py -- no re-optimization, just region-mask reslicing and re-homogenizing the
already-computed per-triangle tensor (cheap).
"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import _common as C
import three_ribbon as TR

BAND = 5.0   # near-interface half-width (x-distance from mid1/mid2), out of a ~22-wide center ribbon


def main():
    rows = []
    for eta, tag in TR.RUNS:
        geo, k, C6, meta = C.load_network(os.path.join(HERE, f'dir_aux_ribbon_{tag}.npz'))
        mid1, mid2 = meta['mid1'], meta['mid2']
        cen = np.asarray(geo['centroids'])
        d_to_iface = np.minimum(np.abs(cen[:, 0] - mid1), np.abs(cen[:, 0] - mid2))
        center_mask = (cen[:, 0] >= mid1) & (cen[:, 0] < mid2)
        near_iface = center_mask & (d_to_iface < BAND)
        deep_bulk = center_mask & (d_to_iface >= BAND)

        solo = np.load(os.path.join(HERE, f'dir_aux_ribbon_center_solo_profile_{tag}.npz'))
        nu_solo0 = float(solo['nu'][0])

        print(f"=== {tag} ===  center tri={center_mask.sum()} "
              f"(near-interface={near_iface.sum()}, deep-bulk={deep_bulk.sum()})", flush=True)
        for name, mask in [('full_center', center_mask), ('near_interface', near_iface),
                           ('deep_bulk', deep_bulk)]:
            idx = np.where(mask)[0]
            nu_th, _ = C.nu_E_theta(C.region_phys_C6(geo, C6, idx), np.array([0.0]))
            print(f"  embedded nu(theta=0) [{name:14s}]: {nu_th[0]:+.4f}   (solo={nu_solo0:+.4f})",
                  flush=True)
            rows.append((tag, name, int(mask.sum()), f'{nu_th[0]:.4f}', f'{nu_solo0:.4f}'))

    C.write_csv(os.path.join(HERE, 'boundary_layer_check.csv'),
                ['topology', 'sub_region', 'n_triangles', 'embedded_nu_theta0', 'solo_nu_theta0'], rows)
    print('saved boundary_layer_check.csv')


if __name__ == '__main__':
    main()
