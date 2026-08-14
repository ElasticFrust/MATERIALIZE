r"""Phase 5 Goal 2 — PHYSICAL directional-target generators (nu(theta), E(theta)).

WHY this module exists
----------------------
A target directional response nu(theta), E(theta) is REALIZABLE only if it is the response of
some valid 2D elastic tensor, i.e. a symmetric POSITIVE-DEFINITE stiffness.  Elastic reciprocity
(symmetry of the compliance tensor) then FORCES

        nu(0) / E(0)  ==  nu(90) / E(90)

for EVERY physical material.  A consequence: a plain  nu(theta) = A*cos(2 theta)  is IMPOSSIBLE
(it makes nu(0) = +A and nu(90) = -A with a flat E, violating reciprocity).  The clean, provable
way to only ever hand the designer PHYSICAL targets is to DERIVE each one from a valid stiffness
tensor.  This module does exactly that, from three sources:

  1. CRYSTAL / LATTICE tensors  (`crystal_targets`)   — forward an anisotropic Bravais lattice
     (stretched / rotated) through the solver with uniform k; use ITS bulk tensor's nu(theta),
     E(theta).  Guaranteed physical (it came from a real network) and genuinely non-simple.
  2. RANDOM VALID tensors       (`random_spd_targets`) — sample C = A A^T + eps I (SPD by
     construction), read off nu(theta), E(theta); REJECT any that fail the realizability filter.
  3. HAND-DESIGNED + FILTER     (`hand_designed_targets`) — build interpretable orthotropic
     compliances (Ex, Ey, nu_xy, G), optionally ROTATED to tilt the principal axes into rich
     multi-lobe profiles; each is VALIDATED as SPD + reciprocal + within the realizability filter,
     and non-physical parameter choices are REJECTED and logged.

Every returned target carries its provenance label, topology class, the generating 6-vector C6,
and the length-37 nu(theta), E(theta) arrays on the canonical ANG grid.

C6 convention (matches _common / inverse_design exactly)
--------------------------------------------------------
The bulk tensor is a 6-vector C6; the symmetric Voigt stiffness matrix (rows/cols = xx, yy, xy) is

        Cv = [[C6[0], C6[2], C6[1]],
              [C6[2], C6[5], C6[4]],
              [C6[1], C6[4], C6[3]]]

so C6 = [Cxxxx, Cxxxy, Cxxyy, Cxyxy, Cyyxy, Cyyyy].  `nuE_of_c6` is a thin wrapper over
`_common.nu_E_theta`, so targets are computed with the SAME formula the designer/sim read out.

Public API
----------
    ANG                                        canonical theta grid (from inverse_design)
    voigt_to_c6(Cv) / c6_to_voigt(c6)          <-> 3x3 Voigt matrix
    nuE_of_c6(c6, thetas=ANG)                  -> (nu(37), E(37))  physical directional response
    rotate_c6(c6, beta)                        rotate the material by angle beta (radians)
    reciprocity_residual(c6)                   -> float  (|nu(0)/E(0) - nu(90)/E(90)|; ~0 = physical)
    is_physical(c6, nu_max=0.95, thetas=ANG)   -> (ok: bool, reason: str)
    make_target(label, provenance, topo_class, c6, beta=0.0)   -> Target dict
    crystal_targets(specs=..., half=4)         -> list[Target]
    random_spd_targets(n=8, seed=0, ...)       -> (kept: list[Target], rejected: list[dict])
    hand_designed_targets(...)                 -> (kept: list[Target], rejected: list[dict])
    all_targets(seed=0)                        -> (targets: list[Target], reject_log: list[dict])

A Target is  dict(label, provenance, topo_class, c6 (6,), beta, nu (37,), E (37,)).
"""
import os, sys
import numpy as np
import torch
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))   # repo root from Phase 5/
sys.path.insert(0, os.path.join(REPO, 'Phase 3', 'verifications'))
import _common as C                 # sets up all other sys.path and imports the solver stack
from inverse_design import DesignProblem, ANG, c6_to_nuE_theta
torch.set_default_dtype(torch.float64)   # REQUIRED — the whole stack is float64


# ---- Voigt <-> 6-vector -----------------------------------------------------------------------
def c6_to_voigt(c6):
    """3x3 symmetric Voigt stiffness (rows/cols = xx, yy, xy) from the 6-vector."""
    c = np.asarray(c6, float)
    return np.array([[c[0], c[2], c[1]], [c[2], c[5], c[4]], [c[1], c[4], c[3]]])


def voigt_to_c6(Cv):
    """6-vector from a 3x3 Voigt stiffness (inverse of c6_to_voigt)."""
    Cv = np.asarray(Cv, float)
    return np.array([Cv[0, 0], Cv[0, 2], Cv[0, 1], Cv[2, 2], Cv[1, 2], Cv[1, 1]])


# ---- material rotation (rotate the 4th-order stiffness tensor) --------------------------------
_IDX = {0: (0, 0), 1: (1, 1), 2: (0, 1)}          # Voigt row/col -> tensor index pair


def _stiff4(Cv):
    """Full 4th-order stiffness tensor Q_ijkl (all minor+major symmetries) from Voigt stiffness."""
    Q = np.zeros((2, 2, 2, 2))
    for a in range(3):
        for b in range(3):
            i, j = _IDX[a]; k, l = _IDX[b]; val = Cv[a, b]
            for ii, jj in ((i, j), (j, i)):
                for kk, ll in ((k, l), (l, k)):
                    Q[ii, jj, kk, ll] = val
    return Q


def _voigt_from_stiff4(Q):
    Cv = np.zeros((3, 3))
    for a in range(3):
        for b in range(3):
            i, j = _IDX[a]; k, l = _IDX[b]; Cv[a, b] = Q[i, j, k, l]
    return Cv


def rotate_c6(c6, beta):
    """Rotate the MATERIAL by angle `beta` (radians): C6 of the rotated tensor.  SPD-preserving, so
    the result is guaranteed physical whenever the input is.  (Equivalent to sampling the original
    tensor's response at ANG - beta — verified to machine precision in the self-test.)"""
    if beta == 0.0:
        return np.asarray(c6, float).copy()
    Q = _stiff4(c6_to_voigt(c6))
    c, s = np.cos(beta), np.sin(beta)
    R = np.array([[c, -s], [s, c]])
    Qr = np.einsum('ip,jq,kr,ls,pqrs->ijkl', R, R, R, R, Q)
    return voigt_to_c6(_voigt_from_stiff4(Qr))


# ---- directional response + physicality checks ------------------------------------------------
def nuE_of_c6(c6, thetas=ANG):
    """Physical directional response (nu(theta), E(theta)) of a 6-vector — thin wrapper over
    `_common.nu_E_theta` (the exact formula the designer and the independent sim read out)."""
    nu, E = C.nu_E_theta(np.asarray(c6, float), np.asarray(thetas, float))
    return np.asarray(nu, float), np.asarray(E, float)


def reciprocity_residual(c6):
    """|nu(0)/E(0) - nu(90)/E(90)| — zero for every physical (reciprocal) tensor.  A NON-zero value
    would flag a target that cannot come from a valid elastic tensor (e.g. a plain cos(2 theta))."""
    nu, E = nuE_of_c6(c6)
    i0, i90 = 0, len(ANG) // 2
    return float(abs(nu[i0] / E[i0] - nu[i90] / E[i90]))


def is_physical(c6, nu_max=0.95, e_min=0.05, e_max=25.0, thetas=ANG):
    """Realizability filter for a candidate tensor.  Physical iff the Voigt stiffness is SPD
    (fundamental) AND its directional response is well-behaved: E(theta) finite and in
    [e_min, e_max], |nu(theta)| < nu_max everywhere, and reciprocity holds.  Returns (ok, reason);
    `reason` names the FIRST violated criterion (for the rejection log)."""
    Cv = c6_to_voigt(c6)
    if not np.all(np.isfinite(Cv)):
        return False, 'non-finite tensor'
    w = np.linalg.eigvalsh(0.5 * (Cv + Cv.T))
    if w.min() <= 1e-9:
        return False, f'not SPD (min eig {w.min():.2e})'
    nu, E = nuE_of_c6(c6, thetas)
    if not (np.all(np.isfinite(nu)) and np.all(np.isfinite(E))):
        return False, 'non-finite response'
    if E.min() <= e_min or E.max() >= e_max:
        return False, f'E out of [{e_min},{e_max}] (range [{E.min():.3f},{E.max():.3f}])'
    if np.abs(nu).max() >= nu_max:
        return False, f'|nu| >= {nu_max} (max |nu| = {np.abs(nu).max():.3f})'
    if reciprocity_residual(c6) > 1e-6:
        return False, 'reciprocity violated'
    return True, 'ok'


# ---- Target record ----------------------------------------------------------------------------
def make_target(label, provenance, topo_class, c6, beta=0.0):
    """Bundle a validated 6-vector into a Target dict (label, provenance, topo_class, c6, beta,
    nu(37), E(37))."""
    c6 = np.asarray(c6, float)
    nu, E = nuE_of_c6(c6)
    return dict(label=label, provenance=provenance, topo_class=topo_class,
                c6=c6, beta=float(beta), nu=nu, E=E)


# ---- 1. crystal / lattice tensors -------------------------------------------------------------
_CRYSTAL_SPECS = (
    # (phi, psi, rotation_deg) — anisotropic Bravais lattices, some rotated to tilt the axes
    (1.30, 1.00,  0.0),
    (1.50, 0.75,  0.0),
    (0.70, 1.30,  0.0),
    (1.35, 1.00, 30.0),
    (1.50, 0.80, 45.0),
)


def crystal_targets(specs=_CRYSTAL_SPECS, half=4):
    """Forward anisotropic (stretched, optionally rotated) Bravais lattices through the solver with
    UNIFORM k=1 to obtain a real bulk tensor C6, and use its nu(theta), E(theta) as a target.
    Guaranteed physical (it is the response of an actual network) and genuinely non-simple."""
    out = []
    for phi, psi, rot_deg in specs:
        geo = C.make_lattice(phi, psi, half=half)
        prob = DesignProblem.from_geo(geo)
        fwd = prob.forward(torch.ones(prob.n_bond))
        c6_base = prob.region_tensor(fwd['per_triangle'], None).detach().numpy()
        beta = np.radians(rot_deg)
        c6 = rotate_c6(c6_base, beta)
        lbl = f'crystal_phi{phi}_psi{psi}' + (f'_rot{int(rot_deg)}' if rot_deg else '')
        out.append(make_target(lbl, 'crystal', 'crystal_bravais', c6, beta))
    return out


# ---- 2. random valid (SPD) tensors ------------------------------------------------------------
def random_spd_targets(n=8, seed=0, scale=1.0, eps=0.3, nu_max=0.95):
    """Sample random SPD tensors  Cv = s * (A A^T) + eps I  (A ~ N(0,1) 3x3), read off nu(theta),
    E(theta), and KEEP only those passing `is_physical` (SPD is automatic; the filter rejects
    responses with E<=0, out-of-range E, or |nu| >= nu_max — the plan's realizability criteria).
    Returns (kept, rejected) where each rejected entry logs the label + reason."""
    rng = np.random.default_rng(seed)
    kept, rejected = [], []
    for i in range(n):
        A = rng.normal(size=(3, 3))
        Cv = scale * (A @ A.T) + eps * np.eye(3)
        # normalise overall stiffness scale so E(theta) lands in a designable band
        Cv = Cv / np.trace(Cv) * 3.0
        c6 = voigt_to_c6(Cv)
        ok, reason = is_physical(c6, nu_max=nu_max)
        if ok:
            kept.append(make_target(f'random_spd_s{seed}_{i}', 'random', 'random_tensor', c6))
        else:
            rejected.append(dict(label=f'random_spd_s{seed}_{i}', provenance='random', reason=reason))
    return kept, rejected


# ---- 3. hand-designed (orthotropic, optionally rotated) + filter ------------------------------
def _orthotropic_c6(Ex, Ey, nu_xy, G):
    """6-vector of an orthotropic material from engineering constants (axes aligned with x,y).
    Built as the inverse of the engineering-Voigt COMPLIANCE, matching _common's convention."""
    S = np.array([[1.0 / Ex, -nu_xy / Ex, 0.0],
                  [-nu_xy / Ex, 1.0 / Ey, 0.0],
                  [0.0, 0.0, 1.0 / G]])
    Cv = np.linalg.inv(S)
    return voigt_to_c6(Cv)


# (Ex, Ey, nu_xy, G, rotation_deg) — richer profiles; rotation tilts the lobes / adds C16
_HAND_SPECS = (
    (1.60, 0.80,  0.30, 0.45,  0.0),   # orthotropic, axis-aligned (cos4-like, asymmetric E)
    (1.60, 0.80,  0.30, 0.45, 30.0),   # same, rotated 30 deg -> tilted multi-lobe (nonzero C16)
    (1.20, 1.20, -0.30, 0.30,  0.0),   # near-isotropic-E but auxetic-ish nu with a strong shear
    (2.00, 0.70,  0.10, 0.80, 22.5),   # stiff-axis + high shear, rotated -> rich E(theta)
    (1.00, 1.00,  0.55, 0.20,  0.0),   # high-nu (non-auxetic) with soft shear -> kept (physical)
    (1.00, 1.00,  1.10, 0.30,  0.0),   # nu_xy>1 -> compliance NOT positive-definite -> REJECTED
)


def hand_designed_targets(specs=_HAND_SPECS, nu_max=0.95):
    """Construct interpretable orthotropic tensors (optionally ROTATED for tilted multi-lobe
    profiles), VALIDATE each with `is_physical` (SPD + reciprocity + realizability filter), and
    keep only the physical ones.  Returns (kept, rejected) with reasons logged for the rejects."""
    kept, rejected = [], []
    for (Ex, Ey, nu_xy, G, rot_deg) in specs:
        lbl = f'hand_Ex{Ex}_Ey{Ey}_nu{nu_xy}_G{G}' + (f'_rot{int(rot_deg)}' if rot_deg else '')
        try:
            c6_base = _orthotropic_c6(Ex, Ey, nu_xy, G)
        except np.linalg.LinAlgError:
            rejected.append(dict(label=lbl, provenance='hand', reason='singular compliance'))
            continue
        beta = np.radians(rot_deg)
        c6 = rotate_c6(c6_base, beta)
        ok, reason = is_physical(c6, nu_max=nu_max)
        if ok:
            kept.append(make_target(lbl, 'hand', 'hand_orthotropic', c6, beta))
        else:
            rejected.append(dict(label=lbl, provenance='hand', reason=reason))
    return kept, rejected


# ---- combined ---------------------------------------------------------------------------------
def all_targets(seed=0, half=4):
    """Assemble the full physical target set across the three sources, plus the rejection log.
    Returns (targets, reject_log)."""
    targets = list(crystal_targets(half=half))
    r_kept, r_rej = random_spd_targets(n=8, seed=seed)
    h_kept, h_rej = hand_designed_targets()
    targets += r_kept + h_kept
    return targets, (r_rej + h_rej)


# ---- self-test --------------------------------------------------------------------------------
def _self_test():
    # rotation == angle-shift, to machine precision
    rng = np.random.default_rng(3)
    A = rng.normal(size=(3, 3)); c6 = voigt_to_c6(A @ A.T + 0.4 * np.eye(3))
    nu_s, E_s = nuE_of_c6(c6, ANG - 0.6)
    nu_r, E_r = nuE_of_c6(rotate_c6(c6, 0.6))
    assert max(np.abs(nu_s - nu_r).max(), np.abs(E_s - E_r).max()) < 1e-10, "rotate != shift"

    # a plain cos(2theta) nu is correctly flagged NON-physical by reciprocity (sanity of the check):
    # there is no c6 producing it, so we just assert the reciprocity residual concept on a real one
    for t in crystal_targets():
        assert reciprocity_residual(t['c6']) < 1e-6, f"{t['label']} not reciprocal?!"

    targets, rejects = all_targets()
    print(f"{'label':44s} {'prov':8s} {'topo':16s} "
          f"{'nu range':>18} {'E range':>18} {'recip':>9}")
    print('-' * 118)
    for t in targets:
        ok, reason = is_physical(t['c6'])
        assert ok, f"{t['label']} slipped through non-physical: {reason}"
        print(f"{t['label'][:44]:44s} {t['provenance']:8s} {t['topo_class']:16s} "
              f"[{t['nu'].min():+.3f},{t['nu'].max():+.3f}]  "
              f"[{t['E'].min():.3f},{t['E'].max():.3f}]  {reciprocity_residual(t['c6']):.1e}")
    print('-' * 118)
    print(f"KEPT {len(targets)} physical targets "
          f"(crystal={sum(t['provenance']=='crystal' for t in targets)}, "
          f"random={sum(t['provenance']=='random' for t in targets)}, "
          f"hand={sum(t['provenance']=='hand' for t in targets)})")
    print(f"REJECTED {len(rejects)} non-physical candidates:")
    for r in rejects:
        print(f"    [{r['provenance']}] {r['label']}: {r['reason']}")

    # non-triviality: none of the kept targets is a plain single cosine / flat pair
    for t in targets:
        nu_var = t['nu'].std(); E_var = t['E'].std()
        assert nu_var + E_var > 1e-3, f"{t['label']} is trivial (flat)"
    print("\nPHYS_TARGETS SELF-TEST PASSED")


if __name__ == '__main__':
    _self_test()
