r"""Figure: PHANTOM-CENTRE FAN vs NON-CROSSING CHORDS on the SAME tiling.

Both are valid triangulations of the same native tiling (`seeds.seed_tiling(..., method=)`); they
differ in how the non-triangular faces are filled:

  fan    — add a PHANTOM CENTRE vertex per face and fan to its corners. Correct by construction and
           the representation `test_hex_closed_form` validates to 4.4e-06, but it CHANGES THE VERTEX
           SET and adds DOF.
  chord  — fill each face with non-crossing diagonals chosen by ear clipping in that face's own
           unwrapped frame. Keeps the tiling's own vertices: an n-gon becomes n-2 triangles.

**They are physically different networks, not two drawings of one.** The added edges carry `k0=eps`,
and a soft SPOKE lets a face hinge whereas a soft CHORD must still carry that face's shear — which
is why the default is left at `fan` and this figure exists to make the structural difference visible
before anyone switches.

Reading the panels: `plotting.draw_network` draws bonds very close to k=0 DASHED and the rest SOLID,
so **solid = the tiling's native bonds (k0=1.0), dashed = the added triangulating edges (k0=1e-3)**.
The fan's dashed edges all meet at a phantom centre; the chord's dashed edges span corner-to-corner.

Run:  C:\Users\doron\anaconda3\python.exe "Phase 5/verifications/chord_vs_fan_figure.py"
Out:  Phase 5/results/chord_vs_fan/chord_vs_fan.png  (+ elements/, + counts.csv)
"""
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'Phase 3', 'verifications'))   # wires the solver stack
import _common as C                                                  # noqa: F401,E402
sys.path.insert(0, os.path.join(REPO, 'Phase 5'))
sys.path.insert(0, REPO)                                             # root plotting.py
import plotting as P                                                 # noqa: E402
import seeds as S                                                    # noqa: E402

OUT = os.path.join(REPO, 'Phase 5', 'results', 'chord_vs_fan')

# One row per method, one column per tiling -- `montage` fills row-major, so order fan-first.
CASES = [('honeycomb', 3), ('kagome', 2), ('square_octagon', 2)]
METHODS = ('fan', 'chord')


def main():
    os.makedirs(OUT, exist_ok=True)
    items, rows = [], []
    for meth in METHODS:
        for name, reps in CASES:
            rec = S.seed_tiling(name, reps, method=meth)
            geo, k = rec['geo'], rec['k0']
            box = abs(float(np.cross(geo['BL1'], geo['BL2'])))
            n_add = int(np.asarray(rec['is_fictional']).sum())
            rows.append(dict(tiling=f'{name}_r{reps}', method=meth,
                             nodes=len(geo['pts']), bonds=len(geo['bond_u']),
                             triangles=len(geo['tri_bond']), added_edges=n_add,
                             native_edges=len(geo['bond_u']) - n_add,
                             areas_over_box=round(float(geo['areas'].sum()) / box, 9)))
            # Explicit solid/dashed mask, NOT the median-relative default: the fan has TWICE as
            # many soft spokes as native bonds, so the median is itself ~eps and the default dashes
            # nothing (measured: honeycomb-fan and square_octagon-fan drew every spoke solid,
            # contradicting this figure's own caption).
            items.append((geo, k, f'{name}_r{reps} — {meth}\n'
                                  f'{len(geo["pts"])} nodes, {len(geo["bond_u"])} bonds',
                          ~np.asarray(rec['is_fictional'], bool)))

    P.montage(items, os.path.join(OUT, 'chord_vs_fan.png'), ncols=len(CASES),
              suptitle='Phantom-centre FAN (top) vs non-crossing CHORDS (bottom) — same tiling\n'
                       'solid = native tiling bonds (k0=1), dashed = added triangulating edges '
                       '(k0=1e-3)',
              cbar_label='k')

    with open(os.path.join(OUT, 'counts.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    hdr = f'{"tiling":20} {"method":6} {"nodes":>6} {"bonds":>6} {"tris":>6} ' \
          f'{"native":>7} {"added":>6} {"areas/box":>11}'
    print(hdr)
    for r in rows:
        print(f'{r["tiling"]:20} {r["method"]:6} {r["nodes"]:6d} {r["bonds"]:6d} '
              f'{r["triangles"]:6d} {r["native_edges"]:7d} {r["added_edges"]:6d} '
              f'{r["areas_over_box"]:11.9f}')
    print(f'\nwrote {OUT}')


if __name__ == '__main__':
    main()
