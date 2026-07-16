"""
strain_stress / mode_selective_small — a SMALL patch with a circle and a triangle that respond to
DIFFERENT stimuli, tuned so each shape is AS INDISTINGUISHABLE AS POSSIBLE from the background when its
stimulus is OFF. Pull +x → the DISC dilates (triangle invisible); pull +y → the TRIANGLE dilates (disc
invisible). The "off" shape targets the background mean AND matched texture with a strong weight, so it
vanishes into the surround; only the active shape lights up.

Self-contained. Outputs mode_selective_small_<topo>.png (guided + clean dilation maps + ON/OFF contrast
bars) and the saved network.
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

CASE, HALF, REG, NITER = 'strain_stress', 16, 2e-3, 300   # HALF=16 -> ~2.3k tri;
TOPO = sys.argv[1] if len(sys.argv) > 1 else 'disorder_hi'   # pass 'regular' to run on the ordered lattice;
# disorder background (moderately noisy). The off-shape is MERGED into the background objective (same
# value + same, WEAK uniformity), so both carry the same heterogeneity and the off-shape blends in.
MEAN_DIL, DIL = 1.0, 1.25           # active shape dilates above the pull-fixed mean
I2 = np.eye(2)
LOAD_X = C.CE.vec3(C.PH.Fk[0].T @ C.PH.Fk[0] - I2) / C.PH.DELTA
LOAD_Y = C.CE.vec3(C.PH.Fk[1].T @ C.PH.Fk[1] - I2) / C.PH.DELTA


def dilation(v):
    return 0.5 * (v[:, 0] + v[:, 2])


def _pct(a, q):
    f = np.abs(a)[np.isfinite(a)]
    return float(np.percentile(f, q)) if f.size else 1.0


def main():
    prob, geo = C.make_case(TOPO, HALF)
    Lx, Ly = C.box(geo)
    disc_spec = {'kind': 'circle', 'center': (0.3 * Lx, 0.5 * Ly), 'radius': 0.17 * Lx, 'color': 'lime'}
    tri_spec = {'kind': 'polygon', 'verts': C.triangle_verts(0.72 * Lx, 0.5 * Ly, 0.22 * Lx), 'color': 'lime'}
    disc, _ = C.region_shape(prob, disc_spec)
    tri, _ = C.region_shape(prob, tri_spec)
    bg = np.setdiff1d(np.arange(prob.n_tri), np.concatenate([disc, tri]))
    print(f"  [mode_selective_small {TOPO}] tri={prob.n_tri} disc={len(disc)} triangle={len(tri)}", flush=True)

    dil = torch.tensor([DIL, 0.0, DIL])
    mean = torch.tensor([MEAN_DIL, 0.0, MEAN_DIL])
    allt = np.arange(prob.n_tri)
    not_disc = np.setdiff1d(allt, disc)                          # tri + bg = the OFF/background field under +x
    not_tri = np.setdiff1d(allt, tri)                            # disc + bg = the OFF/background field under +y
    HON, HBG = 0.4, 0.12                                         # HBG WEAK -> some uniformity, still heterogeneous
    # Under each pull the OFF shape is put in the SAME objective as the background (same target value AND
    # same homogeneity), so it is tuned IDENTICALLY to the surround -> statistically indistinguishable.
    objs = [C.Objective('strain', target=dil, region=disc, load=LOAD_X, weight=2.0, homogeneity=HON),
            C.Objective('strain', target=mean, region=not_disc, load=LOAD_X, weight=0.7, homogeneity=HBG),
            C.Objective('strain', target=dil, region=tri, load=LOAD_Y, weight=2.0, homogeneity=HON),
            C.Objective('strain', target=mean, region=not_tri, load=LOAD_Y, weight=0.7, homogeneity=HBG)]
    r = C.optimize(prob, objs, mode='k', n_iter=NITER, n_restarts=2, reg=REG, verbose=False)
    C.apply_k_to_geo(geo, r['k'])

    eps, _ = C.unit_mode_response(geo)
    dx, dy = dilation(C.CE.vec3(eps[0])), dilation(C.CE.vec3(eps[1]))

    def stats(d, idx):
        v = d[idx]; v = v[np.isfinite(v)]
        return (float(v.mean()), float(v.std())) if v.size else (np.nan, np.nan)
    bgx_m, bgx_s = stats(dx, bg); bgy_m, bgy_s = stats(dy, bg)
    dcx_m, _ = stats(dx, disc); dcy_m, dcy_s = stats(dy, disc)
    trx_m, trx_s = stats(dx, tri); try_m, _ = stats(dy, tri)
    print(f"    background: +x {bgx_m:.3f}±{bgx_s:.3f}  +y {bgy_m:.3f}±{bgy_s:.3f}", flush=True)
    print(f"    DISC  ON(+x) excess {dcx_m-bgx_m:+.3f} | OFF(+y) excess {dcy_m-bgy_m:+.3f} "
          f"(±{dcy_s:.3f} vs bg ±{bgy_s:.3f})  <- OFF ~0 = invisible", flush=True)
    print(f"    TRI   ON(+y) excess {try_m-bgy_m:+.3f} | OFF(+x) excess {trx_m-bgx_m:+.3f} "
          f"(±{trx_s:.3f} vs bg ±{bgx_s:.3f})  <- OFF ~0 = invisible", flush=True)

    C6 = C.sim_per_triangle_C6(geo)
    C.save_network(os.path.join(C.savedir(CASE), f'mode_selective_small_{TOPO}.npz'), geo, r['k'], C6,
                   disc=disc_spec, triangle=tri_spec, DIL=float(DIL))

    # ---- figure: EXCESS-over-background dilation maps (ON=blob, OFF=blank) + contrast bars ----
    plt.rcParams.update({'font.size': 12})
    vmax = 1.15 * (DIL - MEAN_DIL)                               # scale to the shape EXCESS, not the mean
    ex_x, ex_y = dx - bgx_m, dy - bgy_m                          # subtract the (uniform) background level
    fig = plt.figure(figsize=(20, 5.6))
    panels = [('PULL +x', ex_x, True), ('PULL +x  (no guides)', ex_x, False),
              ('PULL +y', ex_y, True), ('PULL +y  (no guides)', ex_y, False)]
    for i, (ttl, d, mark) in enumerate(panels):
        ax = fig.add_subplot(1, 5, i + 1)
        pc = C.fill_local_map(ax, geo, np.nan_to_num(d, nan=0.0, posinf=vmax, neginf=-vmax),
                              cmap='RdBu_r', sym=True, vlim=vmax)
        C.draw_box(ax, geo)
        if mark:
            C.mark_region(ax, disc_spec); C.mark_region(ax, tri_spec)
        ax.set_title(ttl + '  (dilation − background)', fontsize=10.5)
    a4 = fig.add_subplot(1, 5, 5)
    x = np.arange(2); w = 0.36
    a4.bar(x - w / 2, [dcx_m - bgx_m, trx_m - bgx_m], w, label='+x pull', color='#1f77b4')
    a4.bar(x + w / 2, [dcy_m - bgy_m, try_m - bgy_m], w, label='+y pull', color='#d62728')
    a4.set_xticks(x); a4.set_xticklabels(['DISC', 'TRIANGLE']); a4.axhline(0, color='k', lw=.5)
    a4.set_ylabel('dilation excess over background'); a4.legend(fontsize=9); a4.grid(alpha=.3, axis='y')
    a4.set_title('lights up under its OWN pull;\n~0 (invisible) under the other', fontsize=10)

    fig.suptitle(f'Mode-selective circle & triangle on a small patch ({prob.n_tri} tri, {TOPO}) — '
                 f'each shape invisible when its stimulus is off', fontsize=14, y=1.02)
    plt.tight_layout()
    out = os.path.join(C.savedir(CASE), f'mode_selective_small_{TOPO}.png')
    plt.savefig(out, dpi=190, bbox_inches='tight'); plt.close()
    print(f'saved {os.path.basename(out)}')


if __name__ == '__main__':
    main()
