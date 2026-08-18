"""
Metric / tensor operations in the solver's conventions — the NumPy companion to the core.

Purpose: the handful of primitives that express the geometric ("D2C") homogenisation's conventions
in plain NumPy, for the code paths that do not run through `forward_solver_torch.ElasticSolver` —
mesh construction, the sim-vs-solver comparisons, and the design layer's geometry handling.
They ARE the core's conventions, which is why they live in the core layer rather than being
re-derived per consumer.

Implements: Grossman & Boudaoud, PRR 2026 (arXiv:2309.07844) §II — the bare edge/triangle tensor
A(s) and the metric change Δg = FᵀF − ḡ. See `documentation/MATERIALIZE.md` §3–4 and
`Phase 2/SOLVER_GUIDE.md` §1.

Conventions (identical to the solver's):
  - Voigt **vec3 = [xx, xy, yy]**, NO factor 2 on shear.
  - Strain is the METRIC CHANGE Δg = FᵀF − ḡ, geometrically EXACT; the only linearisation in the
    framework is the constitutive law (energy quadratic in Δg). Never call this a geometric
    linearisation.
  - The current flat/compatible setting fixes the gauge ḡ = I, so `tri_metric_change` returns
    FᵀF − I. A non-flat or incompatible ḡ would enter here.
  - float64 throughout.

Layering: this module is **core-adjacent** — the same layer as `forward_solver_torch.py`, imported
by it never, and depending on nothing but NumPy. It is NOT the protected file: only
`forward_solver_torch.py` carries that status, but changes here are gated by the same suite
(`Phase 2/test_forward_solver.py` + the physical `verify_*` suite) because the design layer and the
core gate both build on it.

History: lifted verbatim from `verification_tools/test_cluster_Ceff.py` (`vec3`,
`tri_metric_change`) and `verification_tools/test_cluster_rigidity.py` (`bare_tensor`) by the A-7b
re-layering — the design layer must not depend on the retireable oracle layer
(`documentation/AUDIT_2026-08.md` A-7b). The two pre-existing `bare_tensor` variants were unified
here; see that function's docstring.
"""
import numpy as np


def vec3(g):
    """Symmetric 2×2 tensor(s) → Voigt **[xx, xy, yy]** (NO factor 2 on shear).

    Accepts any leading batch shape: (..., 2, 2) → (..., 3)."""
    return np.stack([g[..., 0, 0], g[..., 0, 1], g[..., 1, 1]], -1)


def tri_metric_change(ev, sx, F, u):
    """Per-triangle metric change Δg = FᵀF − I under macro deformation `F` plus nodal fluctuation `u`.

    Each triangle's actual deformation gradient F_s is read off from how its two reference edge
    vectors (e01, e02) map forward: F_s = E_def · E_ref⁻¹, with the affine part `e @ Fᵀ` and the
    non-affine part the fluctuation difference across the edge. Δg = F_sᵀF_s − I is EXACT in the
    geometry (no small-displacement expansion).

    Args:
        ev: (N_tri, 3, 2) reference edge vectors, rows (e01, e02, e12) — periodic-unwrapped.
        sx: (N_tri, 3) triangle→node indices.
        F:  (2, 2) applied macroscopic deformation gradient.
        u:  (N_node, 2) nodal fluctuation field.
    Returns:
        (N_tri, 2, 2) Δg per triangle. Use `vec3` for the Voigt form.

    NB: ḡ = I is the flat/compatible GAUGE CHOICE of the current setting, not the definition of
    Δg — an incompatible/curved reference metric replaces the `- np.eye(2)` here.
    """
    e01, e02 = ev[:, 0], ev[:, 1]
    du01 = u[sx[:, 1]] - u[sx[:, 0]]; du02 = u[sx[:, 2]] - u[sx[:, 0]]
    Edef = np.stack([e01 @ F.T + du01, e02 @ F.T + du02], -1)
    Eref = np.stack([e01, e02], -1)
    Fs = Edef @ np.linalg.inv(Eref)
    return np.einsum('nki,nkj->nij', Fs, Fs) - np.eye(2)          # (N,2,2)


def bare_tensor(mesh):
    """Per-triangle **bare tensor A(s)**, in the solver's 5-component packing.

    A(s) = Σ_{e∈s} (k_e / 4ℓ_e²) q_e q_eᵀ with the rank-1 edge carrier q_e = Δx_e Δx_eᵀ; packed as
    the 5 independent components [xxxx, xxxy, xxyy, xyyy, yyyy] of the fully-symmetric 4-tensor.
    This is the exact NumPy mirror of the solver's own `bare` (`forward_solver_torch.py`, the
    `factor = rigidities / length2 / 16` block in `forward`), which is why it belongs here and not
    in each consumer.

    Args:
        mesh: geometry dict needing `edge_vecs` (N,3,2) and `actual_len2` (N,3); the per-triangle-
              edge stiffness `tri_k` (N,3) is used **if present** and defaults to 1.
    Returns:
        (N_tri, 5) bare tensor.

    Unifies the two pre-A-7b variants, exactly and not approximately: the k-aware
    `test_cluster_rigidity.bare_tensor` (its meshes always carry `tri_k`) and the k-less
    `test_cluster_Ceff.bare_tensor` (its meshes — from `pbc_dg_analysis.build_periodic_tf_mesh` —
    carry no `tri_k`/`bond_k` at all, so the default 1.0 reproduces `fac = 1/ℓ²/16` bit-for-bit).
    Verified numerically on both mesh kinds by `Phase 3/verifications/relayer_a7b.py`.

    Uses the reference lengths `actual_len2` = ℓ², i.e. the ḡ = I gauge; the solver takes explicit
    `rest_lengths` instead, which is how ℓ₀ ↦ ḡ(ℓ₀) enters there.
    """
    ev, l2 = mesh['edge_vecs'], mesh['actual_len2']
    k = mesh.get('tri_k', 1.0)
    vx, vy = ev[:, :, 0], ev[:, :, 1]; fac = k / np.maximum(l2, 1e-30) / 16.0
    return np.stack([(fac*vx**4).sum(1), (fac*vx**3*vy).sum(1), (fac*vx**2*vy**2).sum(1),
                     (fac*vx*vy**3).sum(1), (fac*vy**4).sum(1)], 1)


# ---- directional response from C6 (MOVED here from `Phase 3/verifications/_common.py`,
# audit A-7c, 2026-08-18). Pure tensor algebra in the solver's Voigt convention, so it belongs
# beside the other metric/tensor ops. `_common` re-exports it. ------------------------------

def _compliance_tensor(C6):
    Cv = np.array([[C6[0], C6[2], C6[1]], [C6[2], C6[5], C6[4]], [C6[1], C6[4], C6[3]]])
    S = np.linalg.inv(Cv)
    Sc = np.zeros((2, 2, 2, 2))
    Sc[0, 0, 0, 0] = S[0, 0]; Sc[1, 1, 1, 1] = S[1, 1]
    Sc[0, 0, 1, 1] = Sc[1, 1, 0, 0] = S[0, 1]
    for i in [(0, 0, 0, 1), (0, 0, 1, 0), (0, 1, 0, 0), (1, 0, 0, 0)]:
        Sc[i] = S[0, 2] / 2
    for i in [(1, 1, 0, 1), (1, 1, 1, 0), (0, 1, 1, 1), (1, 0, 1, 1)]:
        Sc[i] = S[1, 2] / 2
    for i in [(0, 1, 0, 1), (0, 1, 1, 0), (1, 0, 0, 1), (1, 0, 1, 0)]:
        Sc[i] = S[2, 2] / 4
    return Sc


def nu_E_theta(C6, thetas):
    """Directional Poisson ratio nu(theta) and Young's modulus E(theta) from a 6-vector."""
    Sc = _compliance_tensor(C6)
    nu, E = [], []
    for th in thetas:
        m = np.array([np.cos(th), np.sin(th)]); n = np.array([-np.sin(th), np.cos(th)])
        Emm = np.einsum('ijkl,i,j,k,l', Sc, m, m, m, m)
        Emn = np.einsum('ijkl,i,j,k,l', Sc, m, m, n, n)
        nu.append(-Emn / Emm); E.append(1.0 / Emm)
    return np.array(nu), np.array(E)
