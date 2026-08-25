r"""S0b follow-up — IS the dilution boundary just the solver's REGULARISER? (measured)

HYPOTHESIS
----------
`_woodbury_solve_aw` inverts the per-triangle bare tensor with a RELATIVE regulariser:

    eps   = 1e-12 * A3.abs().max()          # global max over ALL triangles
    A_inv = inv(A3 + eps * I3)

With `A(s) = Σ_e (k_e/4ℓ_e²) q_e q_eᵀ` and `|q_e| ~ ℓ²`, the eigenvalues of `A(s)` scale as
`k_e ℓ²/4`. A triangle carrying a diluted bond therefore has an eigenvalue `≈ k_soft ℓ²/4`, while
`eps` is set by the STIFF triangles at `≈ 1e-12 ℓ²/4`. The regulariser overwhelms the physical
eigenvalue when

    k_soft · ℓ²/4  ≲  1e-12 · ℓ²/4     ⇒     k_soft ≲ 1e-12

**The ℓ² cancels**, so the prediction is scale-free and geometry-independent: the boundary sits at
`k_soft ≈ 1e-12` for ANY base. S0b measured exactly that (1e-8 safe, 1e-12 broken) and found the
SAME boundary on both the regular and the η=0.25 base — which is what a geometry-independent
threshold predicts.

WHAT THIS SCRIPT DOES
---------------------
Rebuilds `A(s)` independently of the solver (via `metric_ops.bare_tensor`, the NumPy mirror), takes
its eigenvalues, and measures the fraction of triangles whose smallest eigenvalue falls BELOW `eps`.
If the hypothesis holds, that fraction should jump from ~0 to large exactly across the measured
boundary, and should track the S0b gap.

Run:  C:\Users\doron\anaconda3\python.exe "Phase 5/verifications/dilution_regulariser_check.py"
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'Phase 3', 'verifications'))
import _common as C                                              # noqa: F401  wires sys.path
sys.path.insert(0, HERE)
import metric_ops as MO                                          # noqa: E402
from dilution_validity import build_diluted, BASES               # noqa: E402

REG = 1e-12                      # the constant hard-coded in _woodbury_solve_aw
K_SOFT = [1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14, 1e-20, 1e-40]


def unpack_A(packed):
    """5-component packing [xxxx, xxxy, xxyy, xyyy, yyyy] -> the symmetric 3x3 in vec3 basis."""
    xxxx, xxxy, xxyy, xyyy, yyyy = (packed[:, i] for i in range(5))
    A = np.empty((len(packed), 3, 3))
    A[:, 0, 0] = xxxx
    A[:, 0, 1] = A[:, 1, 0] = xxxy
    A[:, 0, 2] = A[:, 2, 0] = xxyy
    A[:, 1, 1] = 4.0 * xxyy                    # vec3 shear component carries the factor
    A[:, 1, 2] = A[:, 2, 1] = xyyy
    A[:, 2, 2] = yyyy
    return A


def main():
    print(f'regulariser constant in the core: eps = {REG:.0e} * max|A3|')
    print(f'PREDICTION: boundary where lambda_min(A(s)) ~ eps, i.e. k_soft ~ {REG:.0e}\n')
    print(f"{'base':9} {'k_soft':>8} {'eps':>11} {'min lam':>11} {'frac lam<eps':>13} {'cond(A) max':>12}")
    for base in BASES:
        for ks in K_SOFT:
            geo, k, _ = build_diluted(base, 0.20, ks, 0)
            A = unpack_A(np.asarray(MO.bare_tensor(geo), float))
            eps = REG * np.abs(A).max()
            lam = np.linalg.eigvalsh(A)                       # (N,3) ascending
            lmin = lam[:, 0]
            frac = float((lmin < eps).mean())
            with np.errstate(divide='ignore', invalid='ignore'):
                cond = np.where(lmin > 0, lam[:, 2] / lmin, np.inf)
            print(f'  {base[0]:9} {ks:8.0e} {eps:11.3e} {lmin.min():11.3e} {frac:13.3f} '
                  f'{np.nanmax(cond[np.isfinite(cond)]) if np.isfinite(cond).any() else np.inf:12.3e}')
    print('\nRead: `frac lam<eps` is the fraction of triangles whose smallest genuine eigenvalue is')
    print('below the regulariser, i.e. whose inverse is set by eps rather than by physics.')


if __name__ == '__main__':
    main()
