"""GOAL 1b — the POSITIVE-nu frontier probe.

`run_goal1.py` sweeps nu over [-0.9 … +0.45] and its results doc reports the reachable window at
`f = 0` as **[-0.82, +0.45]**. Those two numbers are NOT the same kind of claim:

  * **-0.82 is a measurement.** The grid runs to -0.9, the designer tried it and fell short, so the
    negative frontier was genuinely located.
  * **+0.45 is the TOP GRID POINT.** Nothing above it was ever attempted, so it is the edge of the
    search, not a property of the networks. The true ceiling could be 0.5 or 0.95 and that sweep
    could not tell the difference. (`large` at +0.44 is one step below the edge — plausibly censored
    too.)

Both `run_goal1.py`'s docstring ("spanning the realizable 2D range") and `GOAL1.md` ("spans the 2D
physical range (-1, 1)") overstate this: the grid spans (-0.9, +0.45) — 90% of the negative half and
45% of the positive half, an asymmetry that was never documented or justified.

**Why the positive end is worth the compute.** In 2D isotropic elasticity nu = (K - G)/(K + G), so
nu -> -1 needs the area modulus K -> 0 and nu -> +1 needs the shear modulus G -> 0. Both limits are a
vanishing modulus; the difficulty is roughly symmetric, and there is no principled reason to probe to
-0.9 but stop at +0.45. The high-nu end is the SHEAR-soft channel — precisely the channel the A-0
defect corrupted (C_xyxy over-stiff by 29-90% wherever W != 0), so it is the end most likely to have
been mis-measured before the fix and is worth re-measuring now.

This probe extends ONLY the positive side, leaving `run_goal1.py` untouched so Stage 2 stays
comparable with the historical sweep. Same driver, same budget, same verification — only the grid
differs, by monkey-patch (see `campaign_shakedown.py` for the same rationale).

  nu grid : 0.5, 0.6, 0.7, 0.8, 0.9, 0.95   (above the old ceiling, up to near the 2D bound)
  bands   : f = 0.0 (soft) and 0.1 (large)  — the two that reached the old ceiling
  reps    : 2                                -> 6 x 2 x 2 = 24 runs

Interpreting the result:
  * if designs reach ~0.9 trustworthily, "+0.45" was purely a grid artifact;
  * if they saturate at some nu* < 0.9 with SMALL solver-sim gaps, that nu* is a real frontier;
  * if they saturate with LARGE gaps, the limit is the solver's A(s) validity condition
    (CLAUDE.md §3), not the physics — the same failure characterised on the regular lattice.

Run:  python run_goal1_frontier.py
Out:  Phase 5/results/goal1_frontier/ , Phase 5/networks/goal1_frontier/
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, 'Phase 3', 'verifications'))

import run_goal1 as G                                                  # noqa: E402


def main():
    G.NU_GRID = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
    G.BANDS = [('soft', 0.0), ('large', 0.1)]                          # the two that reached +0.45
    G.REPS_PER_COMBO = 2
    G.RESDIR = os.path.join(REPO, 'Phase 5', 'results', 'goal1_frontier')
    G.NETDIR = os.path.join(REPO, 'Phase 5', 'networks', 'goal1_frontier')
    print('[frontier] probing ABOVE the old +0.45 grid ceiling')
    print(f'[frontier]   nu    := {G.NU_GRID}')
    print(f'[frontier]   bands := {[b[0] for b in G.BANDS]}')
    print(f'[frontier]   out   -> {G.RESDIR}\n', flush=True)
    return G.main()


if __name__ == '__main__':
    sys.exit(main() or 0)
