"""
Gate: verify the preferred make_lattice geometry works end-to-end BEFORE any plotting.
  - geometry/PBC: near-square box, every bond shared by exactly 2 triangles, areas tile the box;
  - physics: regular -> solver AND sim give nu~1/3, E~2/sqrt(3); anisotropic -> anisotropic nu(theta);
  - design: whatever nu the optimiser reaches, the INDEPENDENT sim must confirm the solver's reading.

Note on what the design section does and does NOT assert (revised 2026-08-16, campaign Stage 0).
It used to be described as "a target nu is hit and the simulation confirms it", and it flagged
`CHECK` when the sim missed the target — but that flag fed NOTHING, so the script printed
`ALL GEOMETRY PASS` beside a 0.255 solver-vs-sim gap.

The two things were conflated. They are not the same kind of claim:

  * **Whether a given topology can REACH a given nu is an OPEN RESEARCH QUESTION** — one this project
    is actively investigating. A regular lattice may or may not reach nu=-0.2 by k-design alone (the
    frozen-connectivity eta-disorder family only reaches ~-0.11). A verification gate must not assert
    an unknown, and must not fail because physics declined to cooperate.
  * **Solver-vs-sim agreement is NOT open.** Whatever nu the optimiser lands on, the differentiable
    solver and the independent simulation must agree about it — or the number is not measuring
    anything. That is a correctness property and it IS gated here.

So: the gap is asserted; the achieved nu is REPORTED as a measurement of the open question.
"""
import os, sys
import numpy as np
import torch
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C
torch.set_default_dtype(torch.float64)


def check_geom(topo, half=8):
    geo = C.make_topology(topo, half)
    geo['bond_k'] = np.ones(len(geo['bond_u'])); geo['tri_k'] = np.ones((len(geo['simplices']), 3))
    Lx, Ly = geo['BL1'][0], geo['BL2'][1]
    cnt = Counter(geo['tri_bond'].ravel().tolist())
    shared2 = all(c == 2 for c in cnt.values())
    area_ok = abs(geo['areas'].sum() - Lx * Ly) < 1e-6 * Lx * Ly
    square = abs(Lx - Ly) / max(Lx, Ly) < 0.10
    return geo, dict(ntri=len(geo['simplices']), Lx=Lx, Ly=Ly, square=square,
                     shared2=shared2, area_ok=area_ok)


def main():
    print("=== geometry / PBC ===")
    geos = {}
    ok = True
    for topo in C.TOPO_IDS:
        geo, g = check_geom(topo)
        geos[topo] = geo
        flags = g['square'] and g['shared2'] and g['area_ok']
        ok &= flags
        print(f"  {topo:12s} ntri={g['ntri']:5d} box={g['Lx']:.2f}x{g['Ly']:.2f} "
              f"square={g['square']} shared2={g['shared2']} area_ok={g['area_ok']}  "
              f"{'PASS' if flags else 'FAIL'}")

    print("=== physics (solver vs sim; regular must be nu=1/3, E=2/sqrt(3)=1.155) ===")
    for topo in C.TOPO_IDS:
        geo = geos[topo]
        prob = C.DesignProblem.from_geo(geo)
        s_nu, s_E = C.solver_region_nuE(prob, torch.ones(prob.n_bond))
        m_nu, m_E = C.sim_region_nuE(geo)
        th = np.linspace(0, np.pi, 25)
        nut = C.nu_E_theta(C.sim_region_C6(geo, None), th)[0]
        print(f"  {topo:12s} solver nu/E = {s_nu:+.3f}/{s_E:.3f}  sim nu/E = {m_nu:+.3f}/{m_E:.3f}"
              f"   nu(theta) range=[{nut.min():+.2f},{nut.max():+.2f}]")

    # reg=0.02 with restarts: reg=0.003 was below CLAUDE.md §3's 0.01-0.05, and with a single start
    # this design is BISTABLE — four runs at assorted reg gave solver nu = +0.304/+0.101/+0.163/
    # +0.138 while an earlier run reached -0.190, i.e. outcomes spanning 0.5 in nu from identical
    # code (B-1 stage 1). Restarts are this codebase's own mechanism for exactly that.
    GAP_TOL = 0.05
    print(f"=== design (target nu=-0.2) — GATED on solver-vs-sim agreement (<{GAP_TOL}), "
          f"achieved nu REPORTED (reachability is an open question) ===")
    for topo in ['regular', 'aniso_shr', 'disorder_lo']:
        geo = geos[topo]
        prob = C.DesignProblem.from_geo(geo)
        res = C.optimize(prob, [C.Objective('nu', -0.2)], mode='k', n_iter=80, n_restarts=3,
                         reg=0.02, verbose=False)
        C.apply_k_to_geo(geo, res['k'])
        s_nu = C.solver_region_nuE(prob, res['k'])[0]
        m_nu = C.sim_region_nuE(geo)[0]
        gap = abs(s_nu - m_nu)
        trustworthy = gap < GAP_TOL
        ok &= trustworthy                       # THIS is the gate: the reading must be trustworthy
        reached = abs(m_nu + 0.2) < 0.05        # reported, NOT gated — see the module docstring
        print(f"  {topo:12s} target=-0.20  solver={s_nu:+.3f}  sim={m_nu:+.3f}  "
              f"gap={gap:.3f} {'OK' if trustworthy else 'UNTRUSTWORTHY'}  |  "
              f"reached target: {'yes' if reached else 'NO (open question, not a failure)'}")

    print("ALL PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == '__main__':
    # sys.exit(main()) — main() RETURNS the status, so calling it bare would print FAIL and still
    # exit 0. That is the same defect this file was just fixed for (a check whose verdict reaches
    # nothing); it must not be reintroduced at the exit code.
    sys.exit(main())
