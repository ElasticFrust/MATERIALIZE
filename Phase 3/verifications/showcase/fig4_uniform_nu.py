"""
showcase / fig4 — a UNIFORM Poisson ratio realised across different topologies. Design k so ν(θ) is a
prescribed uniform value ν* on each of a few topologies (regular, disordered, anisotropic), then show
the per-triangle angle-averaged local-ν MAP — spatially uniform ⇒ a genuine homogeneous material, not a
floppy skeleton. Self-contained: designs freshly (proper budget) + independent-sim material maps.
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

TOPOS = [('regular', 'regular'), ('disorder_hi', 'disordered'), ('aniso_str', 'anisotropic')]
NU_STAR = -0.5                                    # uniform AUXETIC across topologies (the striking case)
N, REG, NITER, NREST = 14, 1.5e-3, 300, 3
TH = C.ANG


def main():
    plt.rcParams.update({'font.size': 12})
    fig, axes = plt.subplots(1, len(TOPOS), figsize=(6.0 * len(TOPOS), 6.2), squeeze=False)
    pcs = []
    for ax, (topo, label) in zip(axes[0], TOPOS):
        prob, geo = C.make_case(topo, N)
        r = C.optimize(prob, [C.Objective('nu_theta', np.full_like(TH, NU_STAR))],
                       mode='k', n_iter=NITER, n_restarts=NREST, reg=REG, verbose=False)
        C.apply_k_to_geo(geo, r['k'])
        C6 = C.sim_per_triangle_C6(geo)
        nu_th = C.nu_E_theta(C.sim_bulk_C6(geo), TH)[0]
        nu_local = C.local_nuE_angleavg(geo, C6)[0]                # per-triangle angle-averaged ν
        pc = C.fill_local_map(ax, geo, nu_local, cmap='RdBu_r', sym=True, vlim=1.0)
        C.draw_box(ax, geo); pcs.append(pc)
        ax.set_title(f'{label}\nν* = {NU_STAR:.2f} → got ⟨ν⟩ = {nu_th.mean():+.2f} '
                     f'(angular spread ±{nu_th.std():.02f})', fontsize=11)
        print(f"  {topo}: ⟨ν⟩={nu_th.mean():+.3f} spread±{nu_th.std():.3f} "
              f"local-ν median={np.median(nu_local):+.3f} IQR={np.subtract(*np.percentile(nu_local,[75,25])):.3f}",
              flush=True)
    cb = fig.colorbar(pcs[-1], ax=axes[0], fraction=0.03, pad=0.02)
    cb.set_label('local angle-averaged ν  (uniform colour = homogeneous material)')
    fig.suptitle(f'One target, many topologies: a UNIFORM Poisson ratio ν = {NU_STAR:.2f} '
                 f'designed across regular, disordered & anisotropic networks', fontsize=14, y=1.0)
    out = os.path.join(HERE, 'fig4_uniform_nu.png')
    plt.savefig(out, dpi=200, bbox_inches='tight'); plt.close()
    print(f'saved {os.path.basename(out)}')


if __name__ == '__main__':
    main()
