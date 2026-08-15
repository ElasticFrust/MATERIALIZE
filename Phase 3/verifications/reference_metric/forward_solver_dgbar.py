"""
forward_solver_dgbar.py  —  reference-metric (ḡ) forward-like solver.

    ⚠  TODO / WORK IN PROGRESS  ⚠
    This is a *linear* (quadratic-energy) reference-metric solver. It gives the effective moduli and
    the RESIDUAL STRESS of an incompatible reference correctly, but it does NOT capture the
    geometric stiffening of the response (the prestress feeding back into C_eff). See "KNOWN
    LIMITATION" below. It is meant as the clean, documented scaffold for the residual-stress build,
    not yet the production core.

--------------------------------------------------------------------------------------------------
WHAT IT DOES
--------------------------------------------------------------------------------------------------
Same D2C (Disc-to-Continuum) homogenisation as Phase 2/forward_solver_torch.py, but each triangle s
carries its own REFERENCE METRIC ḡ_s (rest state), not just the flat ḡ=I. Physically ḡ_s sets the
rest lengths of that triangle's three edges.  Decompose

    actual metric   g_s = g + δg_s          (macroscopic g + per-triangle fluctuation)
    reference       ḡ_s = ḡ + δḡ_s          (average reference + per-triangle fluctuation)
    macro strain    Δg  = g − ḡ
    mismatch        m_s = Δg + δg_s − δḡ_s

with the intrinsic ansatz  δg_s = W_s : Δg.  The elastic energy is

    E = ½ Σ_s  m_s : A_s : m_s ,     A_s = Σ_{e∈s} (k_e / 4 ℓ0,e²) q_e q_eᵀ ,   ℓ0,e² = ḡ_s : q_e .

The reference metric therefore enters TWO ways:

  (1) OPERATOR — the rest length ℓ0,e² = ḡ_s : q_e sets A_s = A(ḡ). Longer rest → softer bond
      (A ∝ k/ℓ0²). This is what changes the modulus, and it is done exactly by feeding
      rest_lengths = √(ḡ_s : q_e) to the Phase 2 core (which builds A(ḡ), solves W, homogenises).

  (2) SOURCE — δḡ_s enters the mismatch as a Δg-INDEPENDENT term. Because it is Δg-independent it
      cannot change W (hence C_eff); it only produces a RESIDUAL STRESS. We solve the relaxed
      prestress fluctuation δg⁰ (minimise the energy at Δg=0 over the compatible field), giving
      σ⁰_s = A_s (δg⁰_s − δḡ_s).  For a COMPATIBLE (flat) ḡ this relaxes away; for an INCOMPATIBLE
      (curved) ḡ a non-relaxable residual stress survives — that is the physics of interest.

--------------------------------------------------------------------------------------------------
COVARIANT READOUT  (important when ḡ ≠ I)
--------------------------------------------------------------------------------------------------
ν and E are NOT covariant scalars — they are frame-dependent. When the reference is not the
identity, reading them with the ordinary (Euclidean) directions gives the wrong, frame-dependent
answer. The covariant moduli contract the compliance with ḡ-ORTHONORMAL directions and carry a
√det ḡ volume factor:

    ν(m) = − S(n,n,m,m) / S(m,m,m,m),         ḡ(m,m)=ḡ(n,n)=1, ḡ(m,n)=0
    E(m) = 1 / [ √det ḡ · S(m,m,m,m) ],       S = C_eff⁻¹

With the covariant readout, the two descriptions of the same material (deformed network with ḡ=I,
vs regular network with the full ḡ) collapse onto one curve (verified to ~1e-15).

--------------------------------------------------------------------------------------------------
KNOWN LIMITATION  (the TODO)
--------------------------------------------------------------------------------------------------
The energy is quadratic in the metric strain with a FIXED A → the equation is linear → by
superposition δg = W·Δg + δg⁰, and the response W (hence C_eff) is provably independent of δḡ
(checked numerically to 1e-14). So this solver reproduces the residual STRESS of an incompatible
reference (validated against a full nonlinear relaxation: pressure-field correlation ≈ 0.99), but
NOT the geometric stiffening — the way that stored stress stiffens the subsequent response (a real
spring's tangent stiffness is k·n̂n̂ + (t/L)(𝟙−n̂n̂); the (t/L) prestress term is absent from a
constant A). Capturing it requires a NON-linear term where σ⁰ multiplies the strain
(the σ⁰:∇∇ / second-order-compatibility term, à la incompatible elasticity / Föppl–von Kármán).
  TODO: add the geometric-stiffness operator G(σ⁰) so C_tangent = A + G(σ⁰).

--------------------------------------------------------------------------------------------------
DIFFERENCES FROM THE PREVIOUS forward_dgbar.py  (that file has been removed; verified equivalent
on the operator+source path by test_forward_solver_dgbar.py — ḡ=I reduction, covariant collapse,
sphere regression)
--------------------------------------------------------------------------------------------------
- `poisson` / `young` are now the COVARIANT moduli (ḡ-orthonormal directions + √det ḡ). The old
  code returned the FLAT (ḡ=I-convention) values as `nu` / `E`, which are frame-dependent and WRONG
  when ḡ≠I. The flat values are still returned, but explicitly as `poisson_flat` / `young_flat`.
- The `ref_in_operator` flag is GONE. This solver ALWAYS puts ḡ in the operator (rest lengths) AND
  as the source — the correct formulation. The old flag's `ref_in_operator=False` ("source-only",
  A built from the actual geometry) was an exploration that computes a compressed-state response,
  not the physical modulus; it is not offered here.
- The exploratory eigenstrain prestress (`stress_eigen = −A(𝟙+W)δḡ`, `sigma0_eigen`) is REMOVED. It
  was found to be a worse (local, un-relaxed) stand-in for the relaxed source and added noise.
- Renamed keys: old `Ceff` → `elastic_tensor`; old `C6` (which was the per-triangle tensor) →
  `per_triangle`. `sigma0`, `stress_field`, `dg0`, `W` are unchanged.
- Structured as a `DGBarSolver` class with documented steps instead of one inline function. The
  underlying operators/physics are identical (both reuse the Phase 2 core), so the numbers on the
  operator+source path match the old code to machine precision.

--------------------------------------------------------------------------------------------------
NOTES
--------------------------------------------------------------------------------------------------
- Does NOT modify the protected Phase 2 core. It REUSES the core's operators (edge/curvature/mean
  constraints and the A3 build) so the physics is bit-identical to the validated solver; ḡ=I
  reproduces the ordinary solve exactly.
- k and ḡ may be scalars/2x2 (broadcast to all triangles) or per-triangle arrays.
"""
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..', '..'))
sys.path.insert(0, os.path.join(_ROOT, 'verification_tools'))
sys.path.insert(0, os.path.join(_ROOT, 'Phase 2'))
sys.path.insert(0, os.path.join(_ROOT, 'Phase 3', 'verifications'))

from solver_build import make_solver          # mounts the protected Phase 2 ElasticSolver
from mesh_build import kkt_from_tri_bond       # periodic interior-edge topology
import _common as _C                                  # nu_E_theta, _compliance_tensor
torch.set_default_dtype(torch.float64)

_I_VEC = np.array([1.0, 0.0, 1.0])                    # identity metric in Voigt [xx, xy, yy]


def _q_voigt(edge_vecs):
    """Per-edge Voigt carrier q_e = [vx², 2 vx vy, vy²]  (shape (n_tri, 3, 3))."""
    vx, vy = edge_vecs[:, :, 0], edge_vecs[:, :, 1]
    return np.stack([vx ** 2, 2.0 * vx * vy, vy ** 2], axis=-1)


def _as_tri_field(x, n_tri, shape):
    """Broadcast a scalar / single tensor to a per-triangle field of the given trailing shape."""
    x = np.asarray(x, dtype=float)
    if x.shape == shape:                              # single value → tile
        return np.broadcast_to(x, (n_tri,) + shape).copy()
    return x


class DGBarSolver:
    """
    Forward-like solver carrying a per-triangle reference metric ḡ_s.

    Usage
    -----
        solver = DGBarSolver(geo)                     # geo from _common.make_lattice(...)
        out = solver.solve(k=1.0, gbar=gbar)          # gbar: (n_tri,2,2) or a single 2x2
        out['young'], out['poisson']                  # covariant moduli (physical when ḡ≠I)
        out['sigma0'], out['stress_field']            # residual stress (macroscopic, per-triangle)

    The heavy lifting (A3, constraints, W solve, homogenisation) is delegated to the protected
    Phase 2 core; this class only (a) turns ḡ into the rest lengths the core expects, (b) solves the
    relaxed residual stress from the δḡ source, and (c) reads ν, E covariantly.
    """

    def __init__(self, geo):
        self.geo = geo
        self.n_tri = len(geo['simplices'])
        self.edge_vecs = geo['edge_vecs']
        self.q = _q_voigt(self.edge_vecs)                                  # (n,3,3)
        self.areas = geo['areas']
        self.area_w = self.areas / self.areas.sum()
        # mount the real Phase 2 solver → gives us its constraint operators and A3 carrier
        self._core = make_solver(geo, kkt_from_tri_bond(geo['tri_bond'], self.edge_vecs))

    # ------------------------------------------------------------------ helpers
    def _rest_lengths(self, gbar_v):
        """ℓ0,e = √(ḡ_s : q_e), the reference edge lengths that put ḡ into the operator A(ḡ)."""
        l0sq = np.einsum('nei,ni->ne', self.q, gbar_v)                     # ḡ:q per edge
        return np.sqrt(np.maximum(l0sq, 1e-30))

    def _A3(self, k, rest):
        """A3_s = Σ_e (k_e / 4 ℓ0,e²) q_e q_eᵀ, built with the CORE's own q_geom (bit-identical)."""
        qg = np.asarray(self._core._q_geom)                               # core's Voigt carrier
        es = k / (4.0 * np.maximum(rest ** 2, 1e-30))
        return np.einsum('ne,nei,nej->nij', es, qg, qg)

    def _residual_stress(self, A3, dgbar_v):
        """
        Relaxed residual stress from the δḡ source, at Δg=0:
          solve  A3 δg⁰ + Cᵀλ = A3 δḡ  (δg⁰ compatible)   →   σ⁰_s = A3_s (δg⁰_s − δḡ_s).
        Uses the CORE's edge/curvature/mean constraints, so the compatible space is the real one.
        """
        import scipy.sparse as sp
        import scipy.sparse.linalg as spla
        n = self.n_tri
        Cmat = sp.vstack([self._core._J_edge_sp, self._core._C_curv_sp, self._core._M_S_sp]).tocsc()
        nC = Cmat.shape[0]
        eps = 1e-10 * float(np.abs(A3).max() or 1.0)                       # tiny reg for the saddle
        KKT = sp.bmat([[sp.block_diag([A3[i] for i in range(n)]), Cmat.T],
                       [Cmat, -eps * sp.eye(nC)]], format='csc')
        rhs = np.einsum('nij,nj->ni', A3, dgbar_v).reshape(3 * n, 1)
        dg0 = spla.splu(KKT).solve(np.vstack([rhs, np.zeros((nC, 1))]))[:3 * n, 0].reshape(n, 3)
        stress_field = np.einsum('nij,nj->ni', A3, dg0 - dgbar_v)          # per-triangle σ⁰
        sigma0 = (stress_field * self.area_w[:, None]).sum(0)              # macroscopic ⟨σ⁰⟩
        return dg0, stress_field, sigma0

    @staticmethod
    def _covariant_nuE(C6, gbar_avg, thetas):
        """
        Covariant ν(θ), E(θ): contract the compliance with ḡ-orthonormal directions and rescale E
        by √det ḡ. Reduces to the ordinary nu_E_theta when ḡ_avg = I.
        """
        Sc = _C._compliance_tensor(np.asarray(C6))
        g = np.asarray(gbar_avg, float)
        e1 = np.array([1.0, 0.0]); e1 = e1 / np.sqrt(e1 @ g @ e1)
        e2 = np.array([0.0, 1.0]); e2 = e2 - (e2 @ g @ e1) * e1; e2 = e2 / np.sqrt(e2 @ g @ e2)
        detg = float(np.linalg.det(g))
        nu, E = [], []
        for th in thetas:
            m = np.cos(th) * e1 + np.sin(th) * e2
            nn = -np.sin(th) * e1 + np.cos(th) * e2
            Smm = np.einsum('ijkl,i,j,k,l', Sc, m, m, m, m)
            Snn = np.einsum('ijkl,i,j,k,l', Sc, nn, nn, m, m)
            nu.append(-Snn / Smm); E.append(1.0 / (np.sqrt(detg) * Smm))
        return np.array(nu), np.array(E)

    # ------------------------------------------------------------------ main solve
    def solve(self, k, gbar, thetas=None):
        """
        Solve the reference-metric forward problem.

        Parameters
        ----------
        k     : scalar or (n_tri, 3) spring constants.
        gbar  : (n_tri, 2, 2) or a single 2x2 reference metric (broadcast to all triangles).
        thetas: optional angle grid for the directional covariant ν(θ), E(θ).

        Returns dict with:
          elastic_tensor : C_eff 6-vector (from the CORE, physical units)
          poisson, young : COVARIANT scalar moduli (using the average ḡ) — physical when ḡ≠I
          poisson_flat, young_flat : the ordinary (ḡ=I convention) readout, for reference
          W, per_triangle          : strain concentration and per-triangle tensor (from the CORE)
          sigma0, stress_field     : macroscopic and per-triangle RESIDUAL STRESS (relaxed)
          dg0                      : relaxed prestress fluctuation
          nu_theta, E_theta        : covariant directional moduli (if thetas given)
        """
        n = self.n_tri
        k = np.full((n, 3), float(k)) if np.isscalar(k) else np.asarray(k, float)
        gbar = _as_tri_field(gbar, n, (2, 2))
        gbar_v = np.stack([gbar[:, 0, 0], gbar[:, 0, 1], gbar[:, 1, 1]], axis=1)   # Voigt [xx,xy,yy]

        # (1) OPERATOR: ḡ → rest lengths → the CORE builds A(ḡ), solves W, homogenises exactly.
        rest = self._rest_lengths(gbar_v)
        core_out = self._core.forward(torch.as_tensor(k), rest_lengths=torch.as_tensor(rest),
                                      method='intrinsic', physical_units=True)
        C6 = np.asarray(core_out['elastic_tensor'])

        # (2) SOURCE: δḡ = ḡ − I → relaxed residual stress.
        A3 = self._A3(k, rest)
        dgbar_v = gbar_v - _I_VEC[None, :]
        dg0, stress_field, sigma0 = self._residual_stress(A3, dgbar_v)

        # covariant moduli (use the area-weighted average reference metric)
        gbar_avg = np.array([[np.average(gbar[:, 0, 0], weights=self.areas),
                              np.average(gbar[:, 0, 1], weights=self.areas)],
                             [np.average(gbar[:, 0, 1], weights=self.areas),
                              np.average(gbar[:, 1, 1], weights=self.areas)]])
        nu_cov, E_cov = self._covariant_nuE(C6, gbar_avg, [0.0])

        out = dict(
            elastic_tensor=C6,                                             # old key was 'Ceff'
            # poisson/young are COVARIANT (old forward_dgbar returned the flat 'nu'/'E' here, which
            # are frame-dependent and wrong when ḡ≠I). Flat values kept as *_flat for reference.
            poisson=float(nu_cov[0]), young=float(E_cov[0]),
            poisson_flat=float(core_out['poisson']), young_flat=float(core_out['young']),
            W=core_out['W'].numpy(), per_triangle=core_out['per_triangle'].numpy(),
            sigma0=sigma0, stress_field=stress_field, dg0=dg0,
            gbar_avg=gbar_avg,
        )
        if thetas is not None:
            nu_th, E_th = self._covariant_nuE(C6, gbar_avg, thetas)
            out['nu_theta'] = nu_th; out['E_theta'] = E_th
        return out


# thin functional wrapper (backward-compatible with the earlier forward_dgbar callers) ------------
def forward_dgbar(geo, k, gbar, thetas=None):
    """Convenience one-shot: DGBarSolver(geo).solve(k, gbar). See DGBarSolver.solve for the dict."""
    return DGBarSolver(geo).solve(k, gbar, thetas=thetas)


if __name__ == '__main__':
    # smoke test: ḡ=I reproduces the ordinary solve (ν=1/3, E=1.1547, W≈0, σ⁰=0)
    reg = _C.make_lattice(1.0, 1.0, half=6.0); nt = len(reg['simplices'])
    o = forward_dgbar(reg, 1.0, np.eye(2))
    print(f"ḡ=I:  ν={o['poisson']:+.5f}  E={o['young']:.5f}  "
          f"‖W‖={np.abs(o['W']).max():.1e}  ‖σ⁰‖={np.abs(o['sigma0']).max():.1e}")
