"""G1.2-FREED — `g1_2` with the fictional bracing edges FREED, run as a VARIANT.

`run_g1_2.py` takes every topology "FULLY triangulated with ALL edges real (k=1)". Three of its ten
are not triangulations: `square_octagon` (60/114 bonds), `rotating_squares` (16/101) and
`reentrant_honeycomb` (98/194) carry FICTIONAL edges added purely to triangulate them. Setting k=1 on
those WELDS THE HINGES SHUT, and the auxetic motifs stop being auxetic (sim):

    rotating squares      braced +0.2890   ->  freed  -1.0000
    reentrant honeycomb   braced +0.3032   ->  freed  -1.0833

For `reentrant_honeycomb` over HALF the network is scaffolding. The solver evaluates the freed
structures accurately (-0.9805 vs -1.0000; -1.0443 vs -1.0833), so this is not a solver limitation —
the braced sweep simply engineers the defining property out of those topologies before it starts.

This variant sets `USE_SEED_K0=True`, so each of the three uses its seed's OWN `k0` (1.0 on real
edges, 0.001 on fictional) — the configuration the `is_fictional` mask exists for. The other seven
are genuine triangulations with no fictional edges and are unchanged.

**Run as a VARIANT, not a fix.** The braced sweep is a legitimate question in its own right ("what can
geometry do on a fully-braced network?") and it is what the historical `g1_2` data measured, so
`run_g1_2.py` keeps its behaviour and comparability. This answers the question we actually care about.

Run:  python run_g1_2_freed.py
Out:  Phase 5/results/g1_2_freed/ , Phase 5/networks/g1_2_freed/
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'Phase 3', 'verifications'))
sys.path.insert(0, os.path.join(REPO, 'Phase 5'))
sys.path.insert(0, HERE)

import run_g1_2 as G                                                   # noqa: E402


def main():
    G.USE_SEED_K0 = True                      # free the fictional bracing edges (k0: 1.0 / 0.001)
    G.RESDIR = os.path.join(REPO, 'Phase 5', 'results', 'g1_2_freed')
    G.NETDIR = os.path.join(REPO, 'Phase 5', 'networks', 'g1_2_freed')
    print('[g1.2-freed] USE_SEED_K0=True — fictional bracing edges freed (k=0.001)')
    print(f'[g1.2-freed] out -> {G.RESDIR}\n', flush=True)
    return G.main()


if __name__ == '__main__':
    sys.exit(main() or 0)
