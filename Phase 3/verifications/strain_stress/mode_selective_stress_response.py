"""
strain_stress / mode_selective_stress_response — the STRESS field of the STRAIN-designed mode-selective
patch. Reloads mode_selective_small_<topo>.npz (designed for DILATION selectivity) and, with NO
re-design, reads its stress response under the same two pulls. This is the mechanical counterpoint to the
dilation maps: a region that over-dilates under displacement-controlled load is COMPLIANT, so it SHEDS
stress (reads dark) — the strain selectivity has a dual stress signature.

Top row PULL +x, bottom PULL +y. Columns: ‖σ‖ map (guided) | ‖σ‖ map (clean) | mean pressure p=½(σxx+σyy)
excess over background | region-mean ‖σ‖ bars. Self-contained; loads, no re-run.
"""
import os, sys, types
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _common as C
from mode_selective_colocated import mag3, _finite_pct                # reuse (DRY)

CASE = 'strain_stress'
TOPO = sys.argv[1] if len(sys.argv) > 1 else 'disorder_hi'


def main():
    path = os.path.join(C.savedir(CASE), f'mode_selective_small_{TOPO}.npz')
    geo, kb, _, meta = C.load_network(path)
    disc_spec, tri_spec = meta['disc'], meta['triangle']
    ns = types.SimpleNamespace(centroids=geo['centroids'])
    disc, _ = C.region_shape(ns, disc_spec)
    tri, _ = C.region_shape(ns, tri_spec)
    nt = len(geo['simplices'])
    bg = np.setdiff1d(np.arange(nt), np.concatenate([disc, tri]))
    print(f"  [mode_selective_stress_response {TOPO}] tri={nt} disc={len(disc)} triangle={len(tri)}", flush=True)

    eps, sig = C.unit_mode_response(geo)
    S = {mk: C.CE.vec3(sig[mk]) for mk in (0, 1)}                      # (nt,3) stress under +x, +y
    mag = {mk: mag3(S[mk]) for mk in (0, 1)}                           # ‖σ‖
    press = {mk: 0.5 * (S[mk][:, 0] + S[mk][:, 2]) for mk in (0, 1)}   # mean stress (pressure)

    def rm(a, idx):
        v = a[idx]; v = v[np.isfinite(v)]
        return float(v.mean()) if v.size else float('nan')
    for mk, lab in ((0, '+x'), (1, '+y')):
        print(f"    pull {lab}:  ‖σ‖  disc={rm(mag[mk],disc):.3f}  tri={rm(mag[mk],tri):.3f}  "
              f"bg={rm(mag[mk],bg):.3f}", flush=True)

    plt.rcParams.update({'font.size': 12})
    fig, axes = plt.subplots(2, 4, figsize=(21, 10.2))
    vmax_m = max(_finite_pct(mag[0], 98), _finite_pct(mag[1], 98))
    pv = max(_finite_pct(press[0] - rm(press[0], bg), 98), _finite_pct(press[1] - rm(press[1], bg), 98))

    for row, (mk, plabel, active, act_idx) in enumerate(
            [(0, '+x', 'DISC', disc), (1, '+y', 'TRIANGLE', tri)]):
        # ‖σ‖ guided + clean
        for col, mark in ((0, True), (1, False)):
            ax = axes[row, col]
            pc = C.fill_local_map(ax, geo, np.nan_to_num(mag[mk], nan=0.0, posinf=vmax_m), cmap='magma')
            pc.set_clim(0, vmax_m); C.draw_box(ax, geo)
            if mark:
                C.mark_region(ax, disc_spec); C.mark_region(ax, tri_spec)
            plt.colorbar(pc, ax=ax, fraction=0.046, label='local ‖σ‖')
            ax.set_title(f'PULL {plabel}: stress magnitude ‖σ‖' + ('' if mark else '  (clean)'), fontsize=11)
        # pressure excess
        axp = axes[row, 2]
        pex = press[mk] - rm(press[mk], bg)
        pc = C.fill_local_map(axp, geo, np.nan_to_num(pex, nan=0.0), cmap='RdBu_r', sym=True, vlim=pv)
        C.draw_box(axp, geo); C.mark_region(axp, disc_spec); C.mark_region(axp, tri_spec)
        plt.colorbar(pc, ax=axp, fraction=0.046, label='pressure − background')
        axp.set_title(f'PULL {plabel}: mean stress excess', fontsize=11)
        # bars
        axb = axes[row, 3]
        vals = [rm(mag[mk], disc), rm(mag[mk], tri), rm(mag[mk], bg)]
        cols = ['#1f77b4', '#ff7f0e', '0.6']
        axb.bar(['disc', 'triangle', 'background'], vals, color=cols)
        axb.axhline(rm(mag[mk], bg), color='0.4', lw=.8, ls='--')
        axb.set_ylabel('mean ‖σ‖'); axb.grid(alpha=.3, axis='y')
        axb.set_title(f'PULL {plabel}: region-mean ‖σ‖\n(active region = {active})', fontsize=10.5)

    fig.suptitle(f'STRESS response of the STRAIN-designed mode-selective patch ({nt} tri, {TOPO}) — '
                 'the over-dilating (compliant) region sheds stress: strain selectivity ⇒ a dual stress signature',
                 fontsize=13, y=1.005)
    plt.tight_layout()
    out = os.path.join(C.savedir(CASE), f'mode_selective_stress_response_{TOPO}.png')
    plt.savefig(out, dpi=180, bbox_inches='tight'); plt.close()
    print(f'saved {os.path.basename(out)}')


if __name__ == '__main__':
    main()
