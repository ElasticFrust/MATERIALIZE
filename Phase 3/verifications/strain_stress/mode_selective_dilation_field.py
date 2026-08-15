"""
strain_stress / mode_selective_dilation_field — the ACTUAL mode-selective response as a LOCAL DILATION
field, under εxx AND εyy, from the saved network (no re-design). This is the correct way to SEE the
selectivity: it is a few-percent local contrast that a plain deformed-mesh plot hides under the
macroscopic stretch. Background (uniform) dilation removed; same symmetric colour scale for both pulls so
the two regions are directly comparable. Regions outlined; per-region excess annotated + bar chart.
"""
import os, sys, types
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _common as C

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

    eps, _ = C.unit_mode_response(geo)
    dil = {mk: 0.5 * (C.MO.vec3(eps[mk])[:, 0] + C.MO.vec3(eps[mk])[:, 2]) for mk in (0, 1)}  # per unit strain

    def rm(a, idx):
        v = a[idx]; v = v[np.isfinite(v)]
        return float(v.mean()) if v.size else float('nan')
    ex = {mk: dil[mk] - rm(dil[mk], bg) for mk in (0, 1)}                 # excess over background
    stat = {(mk, nm): rm(dil[mk], idx) - rm(dil[mk], bg)
            for mk in (0, 1) for nm, idx in (('disc', disc), ('tri', tri))}
    for mk, lab in ((0, 'εxx'), (1, 'εyy')):
        print(f"  {lab}: disc excess {stat[(mk,'disc')]:+.3f}   triangle excess {stat[(mk,'tri')]:+.3f}  "
              f"(per unit strain)", flush=True)

    vlim = np.nanpercentile(np.abs(np.concatenate([ex[0][np.isfinite(ex[0])], ex[1][np.isfinite(ex[1])]])), 98)
    plt.rcParams.update({'font.size': 12})
    fig = plt.figure(figsize=(17, 6.4))
    for i, (mk, lab, active) in enumerate([(0, 'εxx (pull +x)', 'disc'), (1, 'εyy (pull +y)', 'triangle')]):
        ax = fig.add_subplot(1, 3, i + 1)
        pc = C.fill_local_map(ax, geo, np.nan_to_num(ex[mk], nan=0.0), cmap='RdBu_r', sym=True, vlim=vlim)
        C.draw_box(ax, geo)                                          # no shape overlay
        plt.colorbar(pc, ax=ax, fraction=0.046, label='local dilation − background')
        ax.set_title(f'{lab}   [designed-active: {active}]\ndisc {stat[(mk,"disc")]:+.2f}   '
                     f'triangle {stat[(mk,"tri")]:+.2f}', fontsize=11)
    axb = fig.add_subplot(1, 3, 3)
    x = np.arange(2); w = 0.36
    axb.bar(x - w/2, [stat[(0, 'disc')], stat[(0, 'tri')]], w, label='εxx (+x)', color='#1f77b4')
    axb.bar(x + w/2, [stat[(1, 'disc')], stat[(1, 'tri')]], w, label='εyy (+y)', color='#d62728')
    axb.set_xticks(x); axb.set_xticklabels(['DISC', 'TRIANGLE']); axb.axhline(0, color='k', lw=.5)
    axb.set_ylabel('dilation excess over background (per unit strain)')
    axb.legend(fontsize=10); axb.grid(alpha=.3, axis='y')
    axb.set_title('disc: x≫y (selective)\ntriangle: active under BOTH (x≳y — leaky)', fontsize=11)

    fig.suptitle(f'Mode-selective LOCAL DILATION response, εxx vs εyy ({nt} tri, {TOPO}) — background removed '
                 '(the true selectivity, hidden in a plain deformed-mesh plot)', fontsize=13, y=1.01)
    plt.tight_layout()
    out = os.path.join(C.savedir(CASE), f'mode_selective_dilation_field_clean_{TOPO}.png')
    plt.savefig(out, dpi=185, bbox_inches='tight'); plt.close()
    print(f'saved {os.path.basename(out)}')


if __name__ == '__main__':
    main()
