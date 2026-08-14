"""
Shared PHYSICAL homogenisation for sim-vs-solver verification.

The forward solver's homogenised ν/E are the true physical (energy = virial) effective
moduli (after the 2026 unweighted-average fix; use forward(..., physical_units=True) for E in
physical units). The verification "ground truth" must therefore be the physical macroscopic
response of the relaxed network — NOT the legacy metric average `CE.Ceff_nuE`, which biases ν
on disordered / anisotropic meshes. This module provides that reference:

  - virial stress   σ = (1/A_tot) Σ_b (T_b/ℓ_b) R_b⊗R_b,  T_b = k_b·δℓ_b   → C_eff → ν, E
  - energy Hessian  C_eff = (1/A_tot) ∂²U/∂ε²                              (independent cross-check)

Both give identical ν/E (they must — the virial is dU/dε). Accepts any assembler with the
signature assemble(mesh, F) -> (K, f_affine), e.g. test_cluster_rigidity.assemble_K_faff or
pbc_dg_analysis._assemble_K_and_faff.
"""
import numpy as np
import scipy.sparse.linalg as spla

DELTA = 1e-3
MODES = [np.array([[1., 0.], [0., 0.]]), np.array([[0., 0.], [0., 1.]]),
         np.array([[0., .5], [.5, 0.]])]                       # xx, yy, xy
Fk = [np.eye(2) + DELTA * M for M in MODES]


class UnhealthyGeometryError(ValueError):
    """A geometry too near-singular to simulate: the scipy/LAPACK solve would HARD-CRASH (native
    segfault, uncatchable in Python). The sim raises this CATCHABLE error instead, so callers can
    `try/except` it. Screening is automatic at every sim entry — callers need not pre-screen."""


def require_healthy_mesh(mesh, min_area_frac=1e-3):
    """Screen `mesh`; raise `UnhealthyGeometryError` if it is too near-singular to simulate safely:
    any triangle area ≤ 0 (inverted) or min area < `min_area_frac`·mean area (sliver) — the geometric
    conditions that make the scipy/LAPACK solve segfault. Called at the top of every sim entry
    (`relax`, `energy_nuE`) so the SIM SELF-PROTECTS (the crash is sim-only — the torch solver
    degrades gracefully and needs no guard). A finer "is the RESPONSE physical" check belongs in the
    caller/designer, not here — it needs the solver, which the sim must not depend on."""
    a = mesh.get('areas')
    if a is None:
        return                                                # nothing to screen — proceed
    a = np.asarray(a, float)
    if not np.all(a > 0) or a.min() < min_area_frac * a.mean():
        raise UnhealthyGeometryError(
            f"near-singular geometry: min triangle area {a.min():.3e}, mean {a.mean():.3e} "
            f"(inverted or sliver) — refusing to simulate (scipy/LAPACK would hard-crash).")


def relax(mesh, free, assemble):
    """Relax the network under the 3 macro strain modes; return the fluctuation fields (nn,2)."""
    require_healthy_mesh(mesh)                             # self-protect: refuse near-singular geometry
    nn = len(mesh['pts'])
    K, _ = assemble(mesh, np.eye(2))
    Kff = K[free][:, free].tocsc()
    u_modes = []
    for F in Fk:
        fa = assemble(mesh, F)[1]
        u = np.zeros(2 * nn); u[free] = spla.spsolve(Kff, -fa[free])
        u_modes.append(u.reshape(nn, 2))
    return u_modes


def _voigt_nuE(Cv):
    Sc = np.linalg.inv(Cv); Ex, Ey = 1 / Sc[0, 0], 1 / Sc[1, 1]
    return 0.5 * (-Sc[1, 0] * Ex - Sc[0, 1] * Ey), 0.5 * (Ex + Ey)


def _bond_k(mesh):
    k = mesh.get('bond_k')
    return np.ones(len(mesh['bond_u'])) if k is None else k


def virial_nuE(mesh, u_modes):
    """Physical (ν, E) from the macroscopic virial stress of the relaxed field."""
    R = mesh['bond_R']; L = np.sqrt((R ** 2).sum(1)); kb = _bond_k(mesh)
    Cv = np.zeros((3, 3))
    for k, F in enumerate(Fk):
        u = u_modes[k]
        dub = R @ (F - np.eye(2)).T + (u[mesh['bond_v']] - u[mesh['bond_u']])
        T = kb * (R * dub).sum(1) / L
        S = np.einsum('b,bi,bj->ij', T / L, R, R) / mesh['areas'].sum()
        Cv[:, k] = [S[0, 0], S[1, 1], S[0, 1]]
    return _voigt_nuE(Cv / DELTA)


def energy_nuE(mesh, free, assemble, h=1e-3):
    """Physical (ν, E) from the Hessian of the relaxed elastic energy (independent of virial)."""
    require_healthy_mesh(mesh)                             # self-protect: refuse near-singular geometry
    A = mesh['areas'].sum(); nn = len(mesh['pts'])
    K, _ = assemble(mesh, np.eye(2)); Kff = K[free][:, free].tocsc()
    R = mesh['bond_R']; L = np.sqrt((R ** 2).sum(1)); kb = _bond_k(mesh)

    def e(g):
        F = np.eye(2) + np.array([[g[0], g[2] / 2], [g[2] / 2, g[1]]])
        fa = assemble(mesh, F)[1]
        u = np.zeros(2 * nn); u[free] = spla.spsolve(Kff, -fa[free]); U = u.reshape(nn, 2)
        dub = R @ (F - np.eye(2)).T + (U[mesh['bond_v']] - U[mesh['bond_u']])
        return 0.5 * np.sum(kb * ((R * dub).sum(1) / L) ** 2) / A

    C = np.zeros((3, 3))
    for a in range(3):
        ga = np.zeros(3); ga[a] = h; C[a, a] = 2 * e(ga) / h ** 2
    for a in range(3):
        for b in range(a + 1, 3):
            g = np.zeros(3); g[a] = h; g[b] = h
            C[a, b] = C[b, a] = (2 * e(g) / h ** 2 - C[a, a] - C[b, b]) / 2
    return _voigt_nuE(C)


def sim_nuE(mesh, free, assemble):
    """Convenience: physical (ν, E) reference for a periodic mesh (pin node 0 -> free=2..2n)."""
    return virial_nuE(mesh, relax(mesh, free, assemble))
