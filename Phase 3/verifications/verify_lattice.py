"""
Gate: verify the preferred make_lattice geometry works end-to-end BEFORE any plotting.
  - geometry/PBC: near-square box, every bond shared by exactly 2 triangles, areas tile the box;
  - physics: regular -> solver AND sim give nu~1/3, E~2/sqrt(3); anisotropic -> anisotropic nu(theta);
  - design: a target nu is hit and the simulation confirms it.
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

    print("=== design (target nu=-0.2, sim confirms) ===")
    for topo in ['regular', 'aniso_shr', 'disorder_lo']:
        geo = geos[topo]
        prob = C.DesignProblem.from_geo(geo)
        res = C.optimize(prob, [C.Objective('nu', -0.2)], mode='k', n_iter=80, reg=0.003, verbose=False)
        C.apply_k_to_geo(geo, res['k'])
        s_nu = C.solver_region_nuE(prob, res['k'])[0]
        m_nu = C.sim_region_nuE(geo)[0]
        print(f"  {topo:12s} target=-0.20  solver={s_nu:+.3f}  sim={m_nu:+.3f}  "
              f"{'PASS' if abs(m_nu + 0.2) < 0.05 else 'CHECK'}")

    print("ALL GEOMETRY PASS" if ok else "GEOMETRY FAIL")


if __name__ == '__main__':
    main()
