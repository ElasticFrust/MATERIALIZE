"""
Shared PHYSICAL homogenisation for sim-vs-solver verification.

The forward solver's homogenised ν/E are the true physical (energy = virial) effective
moduli (after the 2026 unweighted-average fix; use forward(..., physical_units=True) for E in
physical units). The verification "ground truth" must therefore be the physical macroscopic
response of the relaxed network — NOT the legacy metric average `CE.Ceff_nuE`, which biases ν
on disordered / anisotropic meshes. This module provides that reference:

  - virial stress   σ = (1/A_tot) Σ_b (T_b/ℓ_b) R_b⊗R_b,  T_b = k_b·δℓ_b   → C_eff → ν, E
  - energy Hessian  C_eff = (1/A_tot) ∂²U/∂ε²                              (independent cross-check)
                    `energy_C` returns that full TENSOR; `energy_nuE` reduces it to ν,E

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


def virial_C(mesh, u_modes):
    """Physical stiffness TENSOR from the macroscopic VIRIAL STRESS of the relaxed field — Voigt
    [xx,yy,xy], the same basis and convention as `energy_C`.

    Costs only the 3 relaxations already in `u_modes` (vs `energy_C`'s 6), so this is the cheap
    independent bulk tensor. It is also a genuinely DIFFERENT reduction of the same relaxed field —
    stress-based rather than energy-based — so `virial_C` and `energy_C` agreeing is a real check,
    not a tautology (the virial is dU/dε; `test_inverse_design` [15] pins them at ~1e-13).
    """
    R = mesh['bond_R']; L = np.sqrt((R ** 2).sum(1)); kb = _bond_k(mesh)
    Cv = np.zeros((3, 3))
    for k, F in enumerate(Fk):
        u = u_modes[k]
        dub = R @ (F - np.eye(2)).T + (u[mesh['bond_v']] - u[mesh['bond_u']])
        T = kb * (R * dub).sum(1) / L
        S = np.einsum('b,bi,bj->ij', T / L, R, R) / mesh['areas'].sum()
        Cv[:, k] = [S[0, 0], S[1, 1], S[0, 1]]
    return Cv / DELTA


def virial_nuE(mesh, u_modes):
    """Physical (ν, E) from the macroscopic virial stress of the relaxed field."""
    return _voigt_nuE(virial_C(mesh, u_modes))


def _relaxed_bond_energy(mesh, free, assemble, Kff, nn, R, L, kb, g):
    """PER-BOND elastic energy of the relaxed network under macro strain `g` (Voigt [xx,yy,xy],
    applied as F = I + [[g0, g2/2],[g2/2, g1]]). The one physical kernel both `energy_C` and
    `energy_C_per_triangle` are built from — they differ only in how these are summed."""
    F = np.eye(2) + np.array([[g[0], g[2] / 2], [g[2] / 2, g[1]]])
    fa = assemble(mesh, F)[1]
    u = np.zeros(2 * nn); u[free] = spla.spsolve(Kff, -fa[free]); U = u.reshape(nn, 2)
    dub = R @ (F - np.eye(2)).T + (U[mesh['bond_v']] - U[mesh['bond_u']])
    return 0.5 * kb * ((R * dub).sum(1) / L) ** 2


def _energy_hessian(efn, h, tail=()):
    """Hessian of a quadratic energy functional w.r.t. macro strain, Voigt [xx,yy,xy].

    `efn(g)` returns a scalar (bulk) or an array with shape `tail` (e.g. per triangle); the result is
    `tail + (3,3)`. The relaxed energy is EXACTLY quadratic in `g` — the relaxation is a linear solve
    and the bond energy is quadratic in the displacement — so these second differences are exact to
    round-off, not a finite-difference approximation. 6 evaluations (3 diagonal + 3 cross)."""
    C = np.zeros(tail + (3, 3))
    diag = []
    for a in range(3):
        ga = np.zeros(3); ga[a] = h
        d = 2 * np.asarray(efn(ga)) / h ** 2
        C[..., a, a] = d; diag.append(d)
    for a in range(3):
        for b in range(a + 1, 3):
            g = np.zeros(3); g[a] = h; g[b] = h
            off = (2 * np.asarray(efn(g)) / h ** 2 - diag[a] - diag[b]) / 2
            C[..., a, b] = off; C[..., b, a] = off
    return C


def _energy_setup(mesh, free, assemble):
    """Shared prep for the energy-Hessian oracles: factorise K once, cache the bond arrays."""
    require_healthy_mesh(mesh)                             # self-protect: refuse near-singular geometry
    nn = len(mesh['pts'])
    K, _ = assemble(mesh, np.eye(2)); Kff = K[free][:, free].tocsc()
    R = mesh['bond_R']; L = np.sqrt((R ** 2).sum(1))
    return Kff, nn, R, L, _bond_k(mesh)


def energy_C(mesh, free, assemble, h=1e-3):
    """Physical stiffness TENSOR from the Hessian of the relaxed elastic energy — Voigt [xx,yy,xy],
    same convention as the solver's c6 → `[[c0,c2,c1],[c2,c5,c4],[c1,c4,c3]]` assembly.

    The full tensor is the sharper oracle: ν,E are only two contractions of it and are weakly
    sensitive to the shear-shear entry, so a component-wise comparison against this is what catches
    a defect confined to one component (see `Phase 2/test_forward_solver.py` [7])."""
    A = mesh['areas'].sum()
    Kff, nn, R, L, kb = _energy_setup(mesh, free, assemble)
    return _energy_hessian(
        lambda g: _relaxed_bond_energy(mesh, free, assemble, Kff, nn, R, L, kb, g).sum() / A, h)


def energy_C_per_triangle(mesh, free, assemble, h=1e-3):
    """PER-TRIANGLE stiffness tensor C(s) — (N_tri, 3, 3), Voigt [xx,yy,xy] — from the Hessian of
    each triangle's OWN relaxed spring energy. **No solver code in the path.**

    This is the local counterpart of `energy_C`, and it exists because everything local was
    previously obtained by pushing the sim's relaxation back through the solver's own
    `_compute_actual_elastic_tensor` — i.e. self-verification, blind to exactly the defect class a
    tensor check exists to catch (audit A-9; the shear defect A-0 is the worked example). Costs the
    SAME 6 relaxations as `energy_C`: only the accumulation differs.

    **Energy attribution is a convention, chosen to match the solver's `A(s)`.**
    `A(s) = Σ_{e∈s} (k_e/4ℓ_e²) q_e q_eᵀ` gives each of a triangle's three edges its FULL `k_e`, so
    each bond is counted fully in BOTH triangles that share it and `Σ_s U_s = 2·U_total` on a torus.
    The same convention is used here, so the result is directly comparable to the solver's per-
    triangle tensor. (The check is therefore of the CONTRACTION, given the partition — which is
    precisely what A-0 got wrong.)

    Note the basis differs from the solver's per-triangle c6, which acts on Δg = FᵀF − I in vec3
    [xx,xy,yy]; converting is a congruence, not a scalar — see `Phase 2/test_forward_solver.py` [8],
    which calibrates it on the η=0 crystal where W≡0 and C(s)=A(s) is known in closed form.

    Requires `mesh['tri_bond']` (triangle → bond index map).
    """
    if 'tri_bond' not in mesh:
        raise KeyError(
            "energy_C_per_triangle needs mesh['tri_bond'] (triangle→bond map) to attribute bond "
            "energy to triangles. Meshes from mesh_build.build_geometry / build_open_mesh and "
            "_common.make_lattice carry it; pbc_dg_analysis.build_periodic_tf_mesh does not "
            "(it carries kkt_arrays instead).")
    tb = np.asarray(mesh['tri_bond'])
    Kff, nn, R, L, kb = _energy_setup(mesh, free, assemble)
    return _energy_hessian(
        lambda g: _relaxed_bond_energy(mesh, free, assemble, Kff, nn, R, L, kb, g)[tb].sum(1),
        h, tail=(len(tb),))


def energy_C_region(mesh, free, assemble, region=None, h=1e-3, C_s=None):
    """Independent stiffness tensor of a REGION of triangles — (3,3), Voigt [xx,yy,xy].

    Same normalisation as `energy_C`: energy per unit area of the region. The ½ undoes the
    double-counting of shared bonds, so `region=None` reproduces `energy_C` EXACTLY — which is the
    internal consistency check that makes the per-triangle construction trustworthy.

    Pass `C_s` (from `energy_C_per_triangle`) to query many regions from one set of relaxations."""
    if C_s is None:
        C_s = energy_C_per_triangle(mesh, free, assemble, h)
    idx = np.arange(len(C_s)) if region is None else np.asarray(region)
    return 0.5 * C_s[idx].sum(0) / mesh['areas'][idx].sum()


def energy_nuE(mesh, free, assemble, h=1e-3):
    """Physical (ν, E) from the Hessian of the relaxed elastic energy (independent of virial)."""
    return _voigt_nuE(energy_C(mesh, free, assemble, h))


def sim_nuE(mesh, free, assemble):
    """Convenience: physical (ν, E) reference for a periodic mesh (pin node 0 -> free=2..2n)."""
    return virial_nuE(mesh, relax(mesh, free, assemble))
