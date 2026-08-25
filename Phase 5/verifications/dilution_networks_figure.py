r"""S0b companion — SHOW the diluted networks the sweep scored.

The sweep (`dilution_validity.py`) reports numbers; this renders the actual networks either side of
the measured validity boundary, so the regime is visible rather than tabulated. It imports
`build_diluted` from the sweep so the two cannot drift apart.

Panels are ordered by dilution fraction; **solid = intact bonds (k=1), dashed = diluted bonds**
(explicit mask, not the median-relative default — see `plotting.draw_network`). Titles carry the
solver/sim verdict from the sweep's own CSV where available.

Run:  C:\Users\doron\anaconda3\python.exe "Phase 5/verifications/dilution_networks_figure.py"
Out:  Phase 5/results/dilution_validity/dilution_networks.png (+ elements/)
"""
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, HERE)
sys.path.insert(0, REPO)
import plotting as P                                             # noqa: E402
from dilution_validity import build_diluted, BASES, OUT          # noqa: E402

# Two softness levels bracketing the measured boundary (safe 1e-8 vs dead 1e-40), across f.
SHOW = [(1e-8, f) for f in (0.10, 0.20, 0.40)] + [(1e-40, f) for f in (0.10, 0.20, 0.40)]
BASE = BASES[0]                                                  # the regular lattice
SEED = 0


def verdicts():
    """(k_soft, f) -> 'gap=…' from the sweep's CSV, so the picture carries its own number."""
    path = os.path.join(OUT, 'dilution.csv')
    out = {}
    if not os.path.exists(path):
        return out
    for r in csv.DictReader(open(path)):
        if r['base'] != BASE[0] or r['ok'] != '1' or int(r['seed']) != SEED:
            continue
        out[(float(r['k_soft']), float(r['f']))] = float(r['gap'])
    return out


def main():
    v = verdicts()
    items = []
    for k_soft, f in SHOW:
        geo, k, dil = build_diluted(BASE, f, k_soft, SEED)
        gap = v.get((k_soft, f))
        verdict = '' if gap is None else (f'\ngap={gap:.1e} ' + ('OK' if gap <= 0.05 else 'BROKEN'))
        items.append((geo, k,
                      f'k_soft={k_soft:.0e}, f={f:.2f}{verdict}',
                      ~dil))                                     # solid = intact bonds
    P.montage(items, os.path.join(OUT, 'dilution_networks.png'), ncols=3,
              suptitle='S0b — diluted networks either side of the validity boundary\n'
                       'solid = intact (k=1), dashed = diluted;  top k_soft=1e-8 (SAFE), '
                       'bottom k_soft=1e-40 (DEAD)',
              cbar_label='k')
    print(f'wrote {os.path.join(OUT, "dilution_networks.png")}')


if __name__ == '__main__':
    main()
