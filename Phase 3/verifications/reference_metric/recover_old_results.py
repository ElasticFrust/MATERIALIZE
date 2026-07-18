"""
Regression: does forward_solver_dgbar recover the isotropic & anisotropic numbers the old
forward_dgbar produced?  The old code reported the FLAT ν/E, so we compare against
poisson_flat / young_flat (and also print the new COVARIANT readout for context).
Reference values are the old full-matrix run (regular lattice, half=6.0, 336 tri, k=1).
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..')))
import _common as C
from forward_solver_dgbar import forward_dgbar

reg = C.make_lattice(1.0, 1.0, half=6.0); n = len(reg['simplices'])
tol = 5e-4

# old (flat) reference values --------------------------------------------------------------------
ISO = {0.1: (0.3333, 1.04973), 0.3: (0.3333, 0.88823), 0.6: (0.3333, 0.72169), 1.0: (0.3333, 0.57735)}
ANI = {(1.0, 0.7): (0.5051, 1.74955), (1.0, 1.4): (0.2033, 0.70409), (1.0, 0.5): (0.6667, 2.30940),
       (1.0, 2.0): (0.1111, 0.38490), (1.5, 1.0): (0.3158, 1.09393), (2.0, 1.0): (0.2727, 0.94475),
       (1.5, 1.4): (0.1966, 0.68102), (2.0, 0.7): (0.3778, 1.30885)}

ok = True
print("ISOTROPIC  ḡ=(1+c)I           new flat (vs old)                 new covariant")
for c, (nu_o, E_o) in ISO.items():
    o = forward_dgbar(reg, 1.0, (1 + c) * np.eye(2))
    d = abs(o['young_flat'] - E_o) + abs(o['poisson_flat'] - nu_o)
    ok &= d < tol
    print(f"  c={c:<4} nu={o['poisson_flat']:+.4f}({nu_o:+.4f}) E={o['young_flat']:.5f}({E_o:.5f})   "
          f"| cov nu={o['poisson']:+.4f} E={o['young']:.5f}   {'ok' if d < tol else 'MISMATCH'}")

print("\nANISOTROPIC  ḡ=FᵀF, F=[[1,(φ-1)/√3],[0,ψ]]")
for (phi, psi), (nu_o, E_o) in ANI.items():
    F = np.array([[1.0, (phi - 1.0) / np.sqrt(3.0)], [0.0, psi]])
    o = forward_dgbar(reg, 1.0, F.T @ F)
    d = abs(o['young_flat'] - E_o) + abs(o['poisson_flat'] - nu_o)
    ok &= d < tol
    print(f"  φ={phi} ψ={psi}: nu={o['poisson_flat']:+.4f}({nu_o:+.4f}) E={o['young_flat']:.5f}({E_o:.5f}) "
          f"| cov nu={o['poisson']:+.4f} E={o['young']:.5f}   {'ok' if d < tol else 'MISMATCH'}")

print(f"\n{'ALL RECOVERED' if ok else 'MISMATCH(ES) FOUND'}  (tol={tol})")
sys.exit(0 if ok else 1)
