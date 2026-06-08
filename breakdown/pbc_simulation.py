"""
[DEPRECATED / SUPERSEDED — not used by current work, kept for reference only]
Legacy, unvalidated. Superseded by the self-contained PBC solve in
breakdown/pbc_dg_analysis.py. Canonical terminology lives there:
  ḡ = reference metric;  g = total metric = ḡ + Δg;  Δg = global strain = g − ḡ;
  g_s = local per-triangle metric;  δg_s = g_s − g = W_s · Δg  (non-affine).

PBC spring-network simulation — direct sparse linear solve.

Since rest lengths = actual lengths, the equilibrium is exactly linear:
K · δv = -f_aff, where K is the stiffness matrix and f_aff is the affine force.

This replaces the L-BFGS-B approach (~50x faster for N=20).
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def _build_stiffness_and_forces(mesh, eps_modes):
    """Build stiffness matrix K and affine force vectors for each strain mode.

    K_{2i+p, 2j+q} = Σ_{bond b=(i,j)} k/|R|² * R_p * R_q (standard spring stiffness)

    f_aff_{α, 2i+p} = -Σ_{bond b touching i} sign_b * k/|R|² * (R^T ε_α R) * R_p

    Returns (K_csc, f_vecs) where:
      K_csc : (2*N_nodes, 2*N_nodes) sparse CSC matrix
      f_vecs: (n_modes, 2*N_nodes) array of affine force vectors
    """
    simplices   = mesh['simplices']
    edge_vecs   = mesh['edge_vecs']     # (N_tri, 3, 2) unwrapped
    actual_len2 = mesh['actual_len2']   # (N_tri, 3)
    N_nodes = len(mesh['pts'])
    N_tri   = len(simplices)

    edge_pairs = [(0, 1, 0), (0, 2, 1), (1, 2, 2)]  # (ka, kb, ei)

    # Deduplicate bonds
    seen = {}
    for ti in range(N_tri):
        for ka, kb, ei in edge_pairs:
            u  = int(simplices[ti, ka])
            v  = int(simplices[ti, kb])
            R  = edge_vecs[ti, ei]    # oriented from u to v
            l2 = float(actual_len2[ti, ei])
            key = (min(u, v), max(u, v))
            if key not in seen:
                # Store with canonical u→v orientation
                seen[key] = (u, v, R.copy(), l2)

    bonds_u = []; bonds_v = []; bonds_R = []; bonds_l2 = []
    for (u, v, R, l2) in seen.values():
        bonds_u.append(u); bonds_v.append(v)
        bonds_R.append(R); bonds_l2.append(l2)

    bonds_u  = np.array(bonds_u,  dtype=np.int64)
    bonds_v  = np.array(bonds_v,  dtype=np.int64)
    bonds_R  = np.array(bonds_R,  dtype=np.float64)  # (n_bonds, 2)
    bonds_l2 = np.array(bonds_l2, dtype=np.float64)  # (n_bonds,)
    n_bonds  = len(bonds_u)

    # Bond stiffness n⊗n where n = R/|R|, k=1
    # stiff[b, p, q] = k/|R|² * R[p] * R[q]  (k=1 everywhere, rest_length=|R|)
    stiff = np.einsum('bp,bq->bpq', bonds_R, bonds_R) / bonds_l2[:, None, None]

    # Build sparse K
    rows, cols, vals = [], [], []
    def add(i, j, M):
        for p in range(2):
            for q in range(2):
                rows.append(2*i + p)
                cols.append(2*j + q)
                vals.append(M[p, q])

    for b in range(n_bonds):
        u, v = int(bonds_u[b]), int(bonds_v[b])
        S = stiff[b]
        add(u, u,  S);  add(v, v,  S)
        add(u, v, -S);  add(v, u, -S)

    K = sp.csc_matrix((vals, (rows, cols)), shape=(2*N_nodes, 2*N_nodes))

    # Affine forces for each loading mode
    # stretch_b(alpha) = R^T eps_alpha R / |R|  (to first order in delta)
    # but we work per unit delta, so divide out delta:
    # stretch_b = R^T eps_alpha R / |R|²   (using |R|² = l2)
    n_modes  = len(eps_modes)
    f_vecs   = np.zeros((n_modes, 2*N_nodes))

    for alpha, eps in enumerate(eps_modes):
        # stretch_b = R^T eps R / |R|²  [scalar per bond]
        eps_R    = bonds_R @ eps.T              # (n_bonds, 2): eps * R
        stretch_b = (bonds_R * eps_R).sum(1) / bonds_l2  # (n_bonds,)

        # Force on node bonds_v: -stretch_b * R/|R|² * |R| = -stretch_b * R / l2 * l2^0.5
        # Actually: f_b = stretch * (R / |R|) = stretch * R / sqrt(l2)
        # But more precisely: K_reduced * dv = -f_aff, where f_aff = K * v_aff.
        # At the affine solution v_aff, the "residual force" on each free DOF is:
        # f[i] = -Σ_{b: bonds_v[b]=i} stiff_b · (R_aff_b - R_b * l_b_aff/l0_b)
        # For small delta: R_aff_b ≈ R_b + delta*eps*R_b, so:
        # f[i] ≈ -Σ_{b: bonds_v[b]=i} (R_b⊗R_b/|R|²) · delta*eps*R_b
        #       = -Σ_{b: bonds_v[b]=i} (R_b · delta*eps*R_b)/|R|² * R_b
        # We work per unit delta:
        force_b = stretch_b[:, None] * bonds_R  # (n_bonds, 2): force from each bond

        np.add.at(f_vecs[alpha].reshape(N_nodes, 2), bonds_v,  force_b)
        np.add.at(f_vecs[alpha].reshape(N_nodes, 2), bonds_u, -force_b)

    return K, f_vecs, bonds_u, bonds_v, bonds_R, bonds_l2


def _solve_pbc_linear(K, f_vec, N_nodes):
    """Solve K·v = f with v[0]=v[1]=0 (fix first node to remove translation).

    Returns v_fluct (N_nodes, 2).
    """
    # Remove rows/cols 0 and 1 (node 0, both DOFs)
    free = np.arange(2, 2 * N_nodes)
    K_red = K[free][:, free].tocsc()
    f_red = f_vec[free]

    try:
        v_free = spla.spsolve(K_red, f_red)
    except Exception:
        # Fallback: least-norm solution via LSQR
        res = spla.lsqr(K_red, f_red, atol=1e-12, btol=1e-12)
        v_free = res[0]

    v = np.zeros(2 * N_nodes)
    v[free] = v_free
    return v.reshape(N_nodes, 2)


def extract_W_sim(mesh, delta=1e-3):
    """Extract per-triangle non-affine response W_sim (N_tri, 3, 3).

    W_sim[s, :, alpha] = (g_def(s,alpha) - g_aff(s,alpha)) / delta
    where alpha ∈ {xx, yy, xy}.

    Uses direct sparse linear solve (replaces L-BFGS-B).
    """
    eps_modes = [
        np.array([[1.0, 0.0], [0.0, 0.0]]),
        np.array([[0.0, 0.0], [0.0, 1.0]]),
        np.array([[0.0, 0.5], [0.5, 0.0]]),
    ]
    N_nodes = len(mesh['pts'])
    N_tri   = len(mesh['simplices'])
    simplices = mesh['simplices']
    edge_vecs = mesh['edge_vecs']

    K, f_vecs, *_ = _build_stiffness_and_forces(mesh, eps_modes)

    W_sim = np.zeros((N_tri, 3, 3))

    for alpha, eps in enumerate(eps_modes):
        F = np.eye(2) + delta * eps
        v_fluct = _solve_pbc_linear(K, f_vecs[alpha] * delta, N_nodes)

        for ti in range(N_tri):
            v0 = int(simplices[ti, 0])
            v1 = int(simplices[ti, 1])
            v2 = int(simplices[ti, 2])

            e01_ref = edge_vecs[ti, 0]
            e02_ref = edge_vecs[ti, 1]

            dv01 = v_fluct[v1] - v_fluct[v0]
            dv02 = v_fluct[v2] - v_fluct[v0]

            e01_def = F @ e01_ref + dv01
            e02_def = F @ e02_ref + dv02
            e01_aff = F @ e01_ref
            e02_aff = F @ e02_ref

            def metric(e1, e2):
                return np.array([e1 @ e1, 2.0 * (e1 @ e2), e2 @ e2])

            W_sim[ti, :, alpha] = (metric(e01_def, e02_def) - metric(e01_aff, e02_aff)) / delta

    return W_sim


def extract_pbc_elastic_tensor(mesh, delta=1e-3):
    """Extract elastic tensor using direct sparse-solve formula.

    C[alpha, beta] = C_affine[alpha, beta] - (1/Vol) * f_alpha · u_beta

    where K · u_alpha = f_alpha  (non-affine displacement per unit strain amplitude).

    Returns (C_3x3, E_modulus, nu_poisson).
    """
    eps_modes = [
        np.array([[1.0, 0.0], [0.0, 0.0]]),
        np.array([[0.0, 0.0], [0.0, 1.0]]),
        np.array([[0.0, 0.5], [0.5, 0.0]]),
    ]
    n_modes  = 3
    N_nodes  = len(mesh['pts'])
    BL1, BL2 = mesh['BL1'], mesh['BL2']
    Vol = abs(BL1[0] * BL2[1] - BL1[1] * BL2[0])

    K, f_vecs, bonds_u, bonds_v, bonds_R, bonds_l2 = _build_stiffness_and_forces(mesh, eps_modes)

    # Solve for non-affine displacement per loading mode (per unit delta)
    # f_vecs[alpha] is per unit delta; we solve K·u = f_vecs[alpha]
    U = np.zeros((n_modes, 2 * N_nodes))
    for alpha in range(n_modes):
        v = _solve_pbc_linear(K, f_vecs[alpha], N_nodes)
        U[alpha] = v.ravel()

    # Affine elastic tensor
    # C_affine[alpha, beta] = (1/Vol) * Σ_b stretch_a * stretch_b
    # where stretch = (R·ε·R)/|R|  (physical bond extension per unit strain)
    bonds_l = np.sqrt(bonds_l2)
    C_aff = np.zeros((n_modes, n_modes))
    for alpha, eps_a in enumerate(eps_modes):
        for beta, eps_b in enumerate(eps_modes):
            eps_a_R = bonds_R @ eps_a.T
            eps_b_R = bonds_R @ eps_b.T
            stretch_a = (bonds_R * eps_a_R).sum(1) / bonds_l
            stretch_b = (bonds_R * eps_b_R).sum(1) / bonds_l
            C_aff[alpha, beta] = stretch_a @ stretch_b / Vol

    # Non-affine correction: C[alpha,beta] -= (1/Vol) * f_alpha · u_beta
    # f_vecs[alpha] is the affine force per unit delta (same as the stiffness times affine displacement)
    C_naff = np.zeros((n_modes, n_modes))
    for alpha in range(n_modes):
        for beta in range(n_modes):
            C_naff[alpha, beta] = f_vecs[alpha] @ U[beta] / Vol

    C = C_aff - C_naff

    # Engineering constants
    try:
        S = np.linalg.inv(C)
        Ex = 1.0 / S[0, 0]; Ey = 1.0 / S[1, 1]
        E_mod = 0.5 * (Ex + Ey)
        nu    = 0.5 * (-S[1, 0] * Ex - S[0, 1] * Ey)
    except np.linalg.LinAlgError:
        E_mod, nu = np.nan, np.nan

    return C, E_mod, nu
