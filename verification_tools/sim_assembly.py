"""
Linear spring-network assembly for the INDEPENDENT simulation oracle.

Purpose: build the stiffness matrix K and the affine residual force f_aff of a central-force
spring network under an applied macroscopic F, so the oracle can relax the network and read its
response back physically. This is the assembler `physical_homog.relax` / `energy_C` / `virial_nuE`
are handed (`assemble(mesh, F) -> (K, f_aff)`).

Layering — this is ORACLE code and stays in `verification_tools/` on purpose. Together with
`physical_homog.py` it is the whole of the temporary, retireable validation oracle
(project `CLAUDE.md` §1–2): a NumPy/SciPy nodal relaxation that shares **no code** with the
differentiable design path, which is exactly what makes it a genuine independent check rather
than self-verification. It must therefore never be imported by `Phase 2/` or `Phase 3/` design
code, and it depends on nothing but NumPy/SciPy.

History: split out of `verification_tools/test_cluster_rigidity.py` by the A-7b re-layering
(`documentation/AUDIT_2026-08.md`). The function did not change layer — but reaching into an
experiment script (whose `main()` runs a 30×3-seed sweep) to obtain the oracle's assembler was the
same smell one level down. `verification_tools/` now reads as {`physical_homog`, `sim_assembly`} =
the oracle, everything else = experiments.
"""
import numpy as np
import scipy.sparse as sp


def assemble_K_faff(mesh, F):
    """Stiffness K (sparse) and affine residual force f_aff for an applied macroscopic F.

    Central-force springs: each bond contributes k·R̂⊗R̂ (with R̂ = R/|R|) between its two nodes.
    `f_aff` is the force the affine placement u = (F−I)X leaves unbalanced, so relaxing
    K·u_fluct = −f_aff gives the non-affine fluctuation field.

    Args:
        mesh: dict with `bond_u`, `bond_v`, `bond_R`, `bond_k`, `pts`.
        F: (2,2) applied macroscopic deformation gradient.
    Returns:
        (K, f_aff) — K as (2N, 2N) sparse CSC, f_aff flattened to (2N,).
    """
    bu, bv, R, k = mesh['bond_u'], mesh['bond_v'], mesh['bond_R'], mesh['bond_k']
    Nn = len(mesh['pts']); l2 = (R**2).sum(1)
    S = k[:, None, None]*np.einsum('bp,bq->bpq', R, R)/l2[:, None, None]   # k * Rhat⊗Rhat
    nb = len(bu); pp = np.array([0, 0, 1, 1]); qq = np.array([0, 1, 0, 1])
    def blk(i, j, sg):
        return ((2*i[:, None]+pp).ravel(), (2*j[:, None]+qq).ravel(),
                (sg*S.reshape(nb, 4)).ravel())
    R_, C_, V_ = [], [], []
    for ia, ja, sg in [(bu, bu, 1.), (bv, bv, 1.), (bu, bv, -1.), (bv, bu, -1.)]:
        r, c, v = blk(ia, ja, sg); R_.append(r); C_.append(c); V_.append(v)
    K = sp.coo_matrix((np.concatenate(V_), (np.concatenate(R_), np.concatenate(C_))),
                      shape=(2*Nn, 2*Nn)).tocsc()
    HR = R @ (F-np.eye(2)).T
    fb = np.einsum('bpq,bq->bp', S, HR)
    faff = np.zeros((Nn, 2)); np.add.at(faff, bv, fb); np.add.at(faff, bu, -fb)
    return K, faff.ravel()
