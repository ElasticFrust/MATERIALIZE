"""
strain_stress / stress_shield_concentrate — a small patch with two shapes that reshape the STRESS field
under a single uniaxial load: a DISC that is SHIELDED (near-zero stress inside — the load flows around
it) and a TRIANGLE that CONCENTRATES stress (a hot spot). Under one pull (+x) the disc reads dark (low
‖σ‖) and the triangle reads bright (high ‖σ‖) against the background. Self-contained; verified with the
independent PBC sim. Outputs stress_shield_concentrate_<topo>.png and the saved network.
"""
import os, sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _common as C
from mode_selective_colocated import mag3, _finite_pct                # reuse (DRY)

CASE, HALF, REG, NITER = 'strain_stress', 16, 3e-4, 260
TOPO = sys.argv[1] if len(sys.argv) > 1 else 'regular'
S_HI = 1.0                                                             # triangle: concentrate to this ‖σ‖
I2 = np.eye(2)
LOAD_X = C.MO.vec3(C.PH.Fk[0].T @ C.PH.Fk[0] - I2) / C.PH.DELTA        # uniaxial pull +x (ε_xx)


def main():
    prob, geo = C.make_case(TOPO, HALF)
    Lx, Ly = C.box(geo)
    disc_spec = {'kind': 'circle', 'center': (0.3 * Lx, 0.5 * Ly), 'radius': 0.17 * Lx, 'color': 'cyan'}
    tri_spec = {'kind': 'polygon', 'verts': C.triangle_verts(0.72 * Lx, 0.5 * Ly, 0.22 * Lx), 'color': 'lime'}
    disc, _ = C.region_shape(prob, disc_spec)
    tri, _ = C.region_shape(prob, tri_spec)
    print(f"  [stress_shield_concentrate {TOPO}] tri={prob.n_tri} disc={len(disc)} triangle={len(tri)}", flush=True)

    # one load, two stress objectives: SHIELD the disc (sigma -> 0), CONCENTRATE in the triangle (sigma high)
    objs = [C.Objective('stress', target=torch.tensor([0.0, 0.0, 0.0]), region=disc, load=LOAD_X, weight=3.0),
            C.Objective('stress', target=torch.tensor([S_HI, 0.0, 0.0]), region=tri, load=LOAD_X, weight=2.0)]
    r = C.optimize(prob, objs, mode='k', n_iter=NITER, n_restarts=2, reg=REG, verbose=False)
    C.apply_k_to_geo(geo, r['k'])

    _, sig = C.unit_mode_response(geo)
    m = mag3(C.MO.vec3(sig[0]))                                        # ‖σ‖ under +x
    bg = np.setdiff1d(np.arange(prob.n_tri), np.concatenate([disc, tri]))

    def rm(idx):
        v = m[idx]; v = v[np.isfinite(v)]
        return float(v.mean()) if v.size else float('nan')
    dm, tm, bm = rm(disc), rm(tri), rm(bg)
    print(f"    ‖σ‖ under +x:  DISC(shield)={dm:.3f}  BACKGROUND={bm:.3f}  TRIANGLE(concentrate)={tm:.3f}", flush=True)
    print(f"    shield ratio disc/bg = {dm/max(bm,1e-9):.2f}×   concentration triangle/bg = {tm/max(bm,1e-9):.2f}×",
          flush=True)

    C6 = C.sim_per_triangle_C6(geo)
    C.save_network(os.path.join(C.savedir(CASE), f'stress_shield_concentrate_{TOPO}.npz'), geo, r['k'], C6,
                   disc=disc_spec, triangle=tri_spec)

    plt.rcParams.update({'font.size': 12})
    vmax = max(_finite_pct(m, 99), S_HI)
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(14, 6.6))
    for ax, mark, ttl in ((a0, True, 'with guides'), (a1, False, 'no guides')):
        pc = C.fill_local_map(ax, geo, np.nan_to_num(m, nan=0.0, posinf=vmax, neginf=0.0), cmap='magma')
        pc.set_clim(0, vmax); C.draw_box(ax, geo)
        if mark:
            C.mark_region(ax, disc_spec); C.mark_region(ax, tri_spec)
        plt.colorbar(pc, ax=ax, fraction=0.046, label='local ‖σ‖')
        ax.set_title(f'stress magnitude under uniaxial pull ({ttl})', fontsize=11)
    fig.suptitle(f'Stress shielding & concentration ({prob.n_tri} tri, {TOPO}): the disc sheds stress '
                 f'(dark), the triangle concentrates it (hot)', fontsize=13, y=1.0)
    plt.tight_layout()
    out = os.path.join(C.savedir(CASE), f'stress_shield_concentrate_{TOPO}.png')
    plt.savefig(out, dpi=190, bbox_inches='tight'); plt.close()
    print(f'saved {os.path.basename(out)}')


if __name__ == '__main__':
    main()
