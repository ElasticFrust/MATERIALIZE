"""
Re-plot the saved mode-selective networks WITHOUT the guiding region outlines -- clean stress-pattern
views, so the load-dependent shapes can be seen on their own. Loads each designed network from its
saved .npz (no re-optimising -- one PBC relaxation each), reusing the response/plot helpers from the
design modules. Complements (does not replace) the outlined versions; outputs *_clean.png.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _common as C
from mode_selective_colocated import mag3, independent_stress, _finite_pct, S

DIR = C.savedir('strain_stress')


def clean_plot(npz_name, title, out_name):
    geo, k, C6, meta = C.load_network(os.path.join(DIR, npz_name))    # geo has k installed already
    sig_x, sig_y = independent_stress(geo)
    mx, my = mag3(sig_x), mag3(sig_y)
    vmax = max(_finite_pct(mx, 99), _finite_pct(my, 99), S)
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(13, 6.2))
    for ax, m, t in ((a0, mx, 'STRETCH +x'), (a1, my, 'COMPRESS −y')):
        pc = C.fill_local_map(ax, geo, np.nan_to_num(m, nan=0.0, posinf=vmax, neginf=0.0), cmap='magma')
        pc.set_clim(0, vmax)
        C.draw_box(ax, geo)
        plt.colorbar(pc, ax=ax, fraction=0.046)
        ax.set_title(f'local ‖σ‖   ({t})', fontsize=11)
    fig.suptitle(title, fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(DIR, out_name), dpi=150, bbox_inches='tight'); plt.close()
    print(f'saved {out_name}', flush=True)


JOBS = [
    ('mode_selective_regular.npz',
     'MODE-SELECTIVE, no guides (disc↔+x, triangle↔−y) — regular',
     'strain_stress_mode_selective_regular_clean.png'),
    ('mode_selective_disorder_hi.npz',
     'MODE-SELECTIVE, no guides (disc↔+x, triangle↔−y) — disorder_hi',
     'strain_stress_mode_selective_disorder_hi_clean.png'),
    ('mode_selective_colocated_cross_bars_regular.npz',
     'CO-LOCATED cross-bars, no guides (horizontal↔+x, vertical↔−y) — regular',
     'strain_stress_mode_selective_colocated_cross_bars_regular_clean.png'),
    ('mode_selective_colocated_cross_bars_disorder_hi.npz',
     'CO-LOCATED cross-bars, no guides (horizontal↔+x, vertical↔−y) — disorder_hi',
     'strain_stress_mode_selective_colocated_cross_bars_disorder_hi_clean.png'),
    ('mode_selective_colocated_disc_triangle_regular.npz',
     'CO-LOCATED disc/triangle, no guides (disc↔+x, frame↔−y) — regular',
     'strain_stress_mode_selective_colocated_disc_triangle_regular_clean.png'),
    ('mode_selective_colocated_disc_triangle_disorder_hi.npz',
     'CO-LOCATED disc/triangle, no guides (disc↔+x, frame↔−y) — disorder_hi',
     'strain_stress_mode_selective_colocated_disc_triangle_disorder_hi_clean.png'),
]


def main():
    for npz_name, title, out_name in JOBS:
        clean_plot(npz_name, title, out_name)
    print('done')


if __name__ == '__main__':
    main()
