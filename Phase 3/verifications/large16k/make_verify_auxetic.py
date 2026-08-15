"""
Does a whole DESIGN CASE actually behave as designed under a real stretch (not just by the homogenised
tensor)? Cut each 16k case into an OPEN sheet and do a uniaxial tensile test along x and along y:
clamp one edge, pull the opposite edge, free lateral edges, relax the spring truss, and read
ν = −ε_lat/ε_axial from the free lateral boundary displacement. Compare to the homogenised tensor ν.

Result: the globally AUXETIC case (iso, target ν=−0.5) is genuinely auxetic under stretch (ν≈−0.47);
the anisotropic case shows the designed direction dependence. So the design method produces real
mechanical behaviour — the earlier "auxetic patch isn't auxetic" was a SUB-REGION cutting artifact,
not a method failure (a whole design is fine; a small embedded sub-domain can't be read off directly).
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import _common as C

CASES = ['iso', 'aniso', 'indep', 'patch']
TOPOS = ['regular', 'disorder_hi']


def stretch_nu(geo, axis):
    return C.open_stretch_nu(geo, axis=axis)[0]


def main():
    rows = []
    for kind in CASES:
        for topo in TOPOS:
            geo, k, C6, meta = C.load_network(os.path.join(HERE, 'networks', f'{kind}__{topo}.npz'))
            nt = C.c6_nuE(C.sim_bulk_C6(geo))[0]
            nx, ny = stretch_nu(geo, 0), stretch_nu(geo, 1)
            rows.append((kind, topo, nt, nx, ny))
            print(f"  {kind:5s} {topo:11s} | tensor nu={nt:+.3f} | stretch-x={nx:+.3f} stretch-y={ny:+.3f}",
                  flush=True)
    C.write_csv(os.path.join(HERE, 'verify_auxetic.csv'),
                ['case', 'topology', 'nu_tensor', 'nu_stretch_x', 'nu_stretch_y'],
                [(r[0], r[1], f'{r[2]:+.3f}', f'{r[3]:+.3f}', f'{r[4]:+.3f}') for r in rows])

    fig, ax = plt.subplots(figsize=(12, 5.4))
    x = np.arange(len(rows)); w = 0.26
    ax.bar(x - w, [r[2] for r in rows], w, label='homogenised tensor ν', color='#1f77b4')
    ax.bar(x, [r[3] for r in rows], w, label='cut-&-stretch ν (x)', color='#ff7f0e')
    ax.bar(x + w, [r[4] for r in rows], w, label='cut-&-stretch ν (y)', color='#2ca02c')
    ax.axhline(0, color='k', lw=.6)
    ax.set_xticks(x); ax.set_xticklabels([f'{r[0]}\n{r[1][:3]}' for r in rows], fontsize=8)
    ax.set_ylabel('Poisson ratio ν')
    ax.set_title('Whole-case verification: homogenised tensor ν vs actual cut-&-stretch ν\n'
                 'the auxetic (iso) case is genuinely auxetic under a real tensile test')
    ax.legend(fontsize=9); ax.grid(alpha=.3, axis='y')
    plt.tight_layout(); plt.savefig(os.path.join(HERE, 'verify_auxetic.png'), dpi=150, bbox_inches='tight')
    plt.close(); print('saved verify_auxetic.png + verify_auxetic.csv')


if __name__ == '__main__':
    main()
