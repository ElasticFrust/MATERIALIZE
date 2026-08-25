r"""S1 premise check — is `C_eff` really INTENSIVE? (M2_V2_PLAN.md §3.1c, §3.1e)

WHY THIS COMES FIRST
--------------------
The M2 v2 head assembles `C_eff = (1/N) Σ_s C(s)` — a MEAN, hence size-independent by construction —
and §3.1e proposes generating training data as small UNIT CELLS on the claim that, for a crystal,
`C_eff` from the minimal cell EQUALS `C_eff` from any supercell of it (the periodic correction has
the lattice's own periodicity).

**That claim is a premise, not a result.** It needs no model to test, so it is checked before a
single line of the GNN is written: if the SOLVER does not reproduce `C` across supercells, then
- the "train small, deploy large" plan is unsound, and
- the free label-less supercell gate for the model is meaningless.

It also directly probes the defect §3.1c flags in v1: global `mean + sum` pooling is EXTENSIVE, so a
model with a `sum` branch cannot satisfy this even if the physics does.

WHAT IT DOES
------------
For each crystal family, builds the same crystal at several cell sizes (a supercell chain) and
compares the solver's `C_eff`, ν and E. A perfect crystal at η=0 should give IDENTICAL numbers.

Run:  C:\Users\doron\anaconda3\python.exe "Phase 5/verifications/supercell_invariance.py"
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'Phase 3', 'verifications'))
import _common as C                                              # wires the solver stack
sys.path.insert(0, os.path.join(REPO, 'Phase 5'))
import torch                                                     # noqa: E402
import seeds as S                                                # noqa: E402
from inverse_design import DesignProblem                         # noqa: E402


def solver_C(geo, k):
    prob = DesignProblem.from_geo(geo)
    with torch.no_grad():
        c6 = C.solver_region_C6(prob, torch.as_tensor(np.asarray(k, float)))
    nu, E = C.c6_nuE(c6)
    return np.asarray(c6, float), float(nu), float(E), prob.n_tri


def lattice_chain():
    """The same triangular crystal at growing cell size — a genuine supercell chain at eta=0."""
    out = []
    for half in (2.0, 3.0, 4.0, 6.0):
        geo = C.make_lattice(1.0, 1.0, half=half, eta=0.0, seed=0)
        k = np.ones(len(geo['bond_u']))
        geo['bond_k'] = k; geo['tri_k'] = k[geo['tri_bond']]
        out.append((f'half={half:g}', geo, k))
    return out


def aniso_chain():
    """Same, on an ANISOTROPIC crystal — C is then not a multiple of the identity, so this is a
    much sharper test than the isotropic lattice (where ν=1/3 could come out right by accident)."""
    out = []
    for half in (2.0, 3.0, 4.0, 6.0):
        geo = C.make_lattice(1.0, 0.6, half=half, eta=0.0, seed=0)
        k = np.ones(len(geo['bond_u']))
        geo['bond_k'] = k; geo['tri_k'] = k[geo['tri_bond']]
        out.append((f'half={half:g}', geo, k))
    return out


def tiling_chain(name):
    """A tiling at growing reps — also a supercell chain, and it carries SOFT fictional bonds, so it
    tests the invariance in the presence of large stiffness contrast."""
    out = []
    for reps in (2, 3, 4):
        try:
            rec = S.seed_tiling(name, reps)
        except Exception as e:                                   # noqa: BLE001
            print(f'  ({name} r{reps} skipped: {type(e).__name__})')
            continue
        out.append((f'reps={reps}', rec['geo'], rec['k0']))
    return out


def report(title, chain):
    if len(chain) < 2:
        print(f'\n{title}: too few cells to compare')
        return
    print(f'\n{title}')
    print(f"  {'cell':10} {'n_tri':>6} {'nu':>12} {'E':>12}   max|C6 - C6(first)|/|C6|")
    c0 = None
    for label, geo, k in chain:
        c6, nu, E, ntri = solver_C(geo, k)
        if c0 is None:
            c0 = c6
            rel = 0.0
        else:
            rel = float(np.abs(c6 - c0).max() / max(np.abs(c0).max(), 1e-300))
        verdict = '' if rel < 1e-9 else ('   <-- NOT INVARIANT' if rel > 1e-6 else '   (1e-9..1e-6)')
        print(f'  {label:10} {ntri:6d} {nu:12.8f} {E:12.8f}   {rel:.3e}{verdict}')


def main():
    print('S1 premise: is C_eff INTENSIVE? (same crystal, growing supercell -> identical C)')
    report('triangular crystal, uniform k (isotropic)', lattice_chain())
    report('anisotropic crystal psi=0.6, uniform k  (SHARPER: C is not isotropic)', aniso_chain())
    for t in ('honeycomb', 'kagome'):
        report(f'{t} tiling (soft fictional bonds -> tests under large k contrast)',
               tiling_chain(t))
    print('\nA MEAN-pooled model can satisfy this; a model with a SUM branch cannot '
          '(v1 pools mean+sum).')


if __name__ == '__main__':
    main()
