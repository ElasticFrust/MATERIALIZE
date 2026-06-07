"""
all_models_comp — four solver variants, full directional elastic properties.

For each of 4 solver variants:
  1. Regular MF          — arithmetic constraint Σ_s δg(s)=0, no KKT
  2. Area-weighted MF    — volume-weighted constraint Σ_s V_s δg(s)=0, no KKT
  3. Regular MF + KKT    — arithmetic constraint + KKT edge-compatibility
  4. Area-wtd MF + KKT   — volume-weighted constraint + KKT

Physical motivation (paper appendix):
  The arithmetic constraint is better for GEOMETRIC response (Poisson ratio):
  soft/small triangles deform easily and matter equally for shape.
  The volume-weighted constraint is better for ENERGETIC response (Young's modulus):
  larger triangles store proportionally more energy.

  Formally, the area-weighted Woodbury replaces 1/N → w_n = a_n/A_total
  everywhere, so the 1/N correction factor in W0 vanishes (Σ w_n = 1 already).

  Standard:       S = (1/N) Σ δA A^{-1},  W0 = -(y + A^{-1} z / N)
  Area-weighted:  S = Σ w_n δA_aw A^{-1},  W0 = -(y + A^{-1} z)  [no /N]

Extracts directional engineering constants via Voigt compliance S = C_V^{-1}:
  x-loading (σ_22=σ_12=0):  E_x = 1/S[0,0],  ν_xy = -S[1,0]/S[0,0]
  y-loading (σ_11=σ_12=0):  E_y = 1/S[1,1],  ν_yx = -S[0,1]/S[1,1]

Isotropy ↔ E_x ≈ E_y and ν_xy ≈ ν_yx.

Output: all_models_comp_nu.png, all_models_comp_E.png
"""
import sys, os, time
from collections import defaultdict
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla

ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'Phase 2'))

import Disc_2_Cont_optimized as D2C
import forward_solver_torch as fst
from forward_solver_torch import (
    _batch_to_9x9, _batch_to_9vec,
    _woodbury_solve, _woodbury_kkt_sparse,
    _compute_actual_elastic_tensor,
)

ETA_VALUES = np.linspace(0.0, 0.5, 11)
N_TRIALS   = 10
SIZE       = (20, 20)
CACHE      = os.path.join(os.path.dirname(__file__), 'all_models_comp_data.npz')

print(f"size={SIZE}  n_trials={N_TRIALS}  n_eta={len(ETA_VALUES)}")
print(f"eta: {np.round(ETA_VALUES, 3).tolist()}\n")

CASES = [
    (False, False, 'Regular MF',        'C0', 'o',  '-'),
    (True,  False, 'Area-weighted MF',  'C1', 's',  '--'),
    (False, True,  'Regular MF + KKT',  'C2', '^',  '-'),
    (True,  True,  'Area-wtd MF + KKT', 'C3', 'D',  '--'),
]


# ── Area-weighted Woodbury (paper appendix) ───────────────────────────────────

def _woodbury_kkt_sparse_aw(A_blocks, B_blocks, dA_vecs, kkt_arrays, w):
    """
    Area-weighted sparse KKT Woodbury solve.

    Implements Σ_n w_n δg(n) = 0 self-consistency (volume-weighted constraint).
    Compared to the standard _woodbury_kkt_sparse, every 1/N is replaced by w_n,
    and the /N factor in the W0 correction vanishes.

    G = J P_aw^{-1} J^T = G_local + H (I - S_aw)^{-1} K_aw^T
    where H = J A_diag^{-1} U  (unchanged — no δA dependence)
          K_aw = w_n δA_aw A_diag^{-1} J^T  (area-weighted)
          M_mat = (I - S_aw) + K_aw^T G_local^{-1} H  (no N factor)

    Returns W as numpy array (N, 9).
    """
    s1_arr, s2_arr, q_arr = kkt_arrays
    E_int = len(s1_arr)
    M_c   = 3 * E_int
    N     = A_blocks.shape[0]

    A_np  = A_blocks.detach().double().numpy()
    B_np  = B_blocks.detach().double().numpy()   # δA_aw (unweighted deviations)
    dA_np = dA_vecs.detach().double().numpy()
    # w: (N,) numpy, sums to 1

    eps   = 1e-14 * np.abs(A_np).max()
    I9    = np.eye(9)
    A_inv = np.linalg.inv(A_np + eps * I9[None])

    # ── W0 via area-weighted Woodbury ─────────────────────────────────────────
    y    = np.einsum('nij,nj->ni', A_inv, dA_np)
    Vy   = np.einsum('n,nij,nj->i',  w, B_np, y)           # Σ w_n δA_aw y_n
    S    = np.einsum('n,nij,njk->ik', w, B_np, A_inv)      # Σ w_n δA_aw A_inv_n  (no /N)
    IminusS = I9 - S
    z    = np.linalg.solve(IminusS, Vy)
    W0   = -(y + np.einsum('nij,j->ni', A_inv, z))         # no /N factor!

    # ── H (standard, no δA dependence) and K_aw (area-weighted) ─────────────
    # BA_inv_aw[n] = w_n * δA_aw(n) @ A_inv(n)
    BA_inv_aw = np.einsum('n,nij,njk->nik', w, B_np, A_inv)

    H = np.zeros((M_c, 9))
    K = np.zeros((M_c, 9))
    for k in range(3):
        rk = np.arange(E_int) + k * E_int
        for loc in range(3):
            col = 3 * loc + k
            H[rk] += q_arr[:, loc:loc+1] * (A_inv[s1_arr, col, :] - A_inv[s2_arr, col, :])
            K[rk] += q_arr[:, loc:loc+1] * (BA_inv_aw[s1_arr, :, col] - BA_inv_aw[s2_arr, :, col])

    # ── Sparse G_local = J A_diag^{-1} J^T (identical to standard) ───────────
    tri_edges = defaultdict(list)
    for e in range(E_int):
        tri_edges[s1_arr[e]].append((e, +1))
        tri_edges[s2_arr[e]].append((e, -1))

    rows_g, cols_g, data_g = [], [], []
    LOC_ROWS = [np.array([k, 3+k, 6+k]) for k in range(3)]
    for n, elist in tri_edges.items():
        An = A_inv[n]
        for e1, sg1 in elist:
            for e2, sg2 in elist:
                val_sg = sg1 * sg2
                for k1 in range(3):
                    row = k1 * E_int + e1
                    for k2 in range(3):
                        col = k2 * E_int + e2
                        A_sub = An[np.ix_(LOC_ROWS[k1], LOC_ROWS[k2])]
                        val = val_sg * q_arr[e1] @ A_sub @ q_arr[e2]
                        rows_g.append(row); cols_g.append(col); data_g.append(val)

    G_local = sp.coo_matrix((data_g, (rows_g, cols_g)), shape=(M_c, M_c)).tocsc()
    G_local = G_local + 1e-12 * abs(max(data_g, default=1.0)) * sp.eye(M_c, format='csc')

    # ── r = J W0 ──────────────────────────────────────────────────────────────
    W0_flat = W0.reshape(-1)
    r = np.zeros(M_c)
    for k in range(3):
        rk = np.arange(E_int) + k * E_int
        for loc in range(3):
            col = 3 * loc + k
            r[rk] += q_arr[:, loc] * (W0_flat[s1_arr*9+col] - W0_flat[s2_arr*9+col])

    # ── Woodbury on G_aw = G_local + H (I-S_aw)^{-1} K_aw^T ─────────────────
    # M_mat_aw = (I - S_aw) + K_aw^T G_local^{-1} H   [no N factor!]
    G_lu  = spla.factorized(G_local)
    Lam0  = G_lu(r)
    Y     = np.column_stack([G_lu(H[:, j]) for j in range(9)])
    M_mat = IminusS + K.T @ Y                          # (I - S_aw) + K^T Y
    c     = np.linalg.solve(M_mat, K.T @ Lam0)
    Lambda = Lam0 - Y @ c

    # ── W = W0 - P_aw^{-1} J^T Λ ────────────────────────────────────────────
    JtLam = np.zeros(9 * N)
    for k in range(3):
        rk = np.arange(E_int) + k * E_int
        Lk  = Lambda[rk]
        for loc in range(3):
            col = 3 * loc + k
            np.add.at(JtLam, s1_arr*9+col,  q_arr[:, loc] * Lk)
            np.add.at(JtLam, s2_arr*9+col, -q_arr[:, loc] * Lk)

    JtLam_b   = JtLam.reshape(N, 9)
    AinvJtLam = np.einsum('nij,nj->ni', A_inv, JtLam_b)
    gs        = np.einsum('n,nij,nj->i', w, B_np, AinvJtLam)   # Σ w_n δA_aw AinvJtLam_n
    z_c       = np.linalg.solve(IminusS, gs)
    PinvJtLam = AinvJtLam + np.einsum('nij,j->ni', A_inv, z_c)  # no /N!

    return W0 - PinvJtLam


# ── Mesh helpers ──────────────────────────────────────────────────────────────

def _triangle_areas(solver):
    pos = solver.positions.detach().numpy()
    sim = solver.simplices.detach().numpy()
    v0, v1, v2 = pos[sim[:, 0]], pos[sim[:, 1]], pos[sim[:, 2]]
    return 0.5 * np.abs(
        (v1[:, 0]-v0[:, 0])*(v2[:, 1]-v0[:, 1]) -
        (v1[:, 1]-v0[:, 1])*(v2[:, 0]-v0[:, 0])
    )


def _directional_constants(C6):
    """C6 = [C_1111, C_1112, C_1122, C_2112, C_2122, C_2222].
    Returns (E_x, E_y, nu_xy, nu_yx) via 3×3 Voigt compliance inversion."""
    C0, C1, C2, C3, C4, C5 = C6
    C_V = np.array([[C0, C2, C1],
                    [C2, C5, C4],
                    [C1, C4, C3]])
    try:
        S = np.linalg.inv(C_V)
    except np.linalg.LinAlgError:
        return np.nan, np.nan, np.nan, np.nan
    def _safe(num, den):
        return num / den if abs(den) > 1e-15 else np.nan
    E_x   = _safe(1.0, S[0, 0])
    E_y   = _safe(1.0, S[1, 1])
    nu_xy = _safe(-S[1, 0], S[0, 0])
    nu_yx = _safe(-S[0, 1], S[1, 1])
    return E_x, E_y, nu_xy, nu_yx


# ── Forward pass for all 4 cases ─────────────────────────────────────────────

def run_all_cases(solver, rigidities, rest_lengths):
    """Run all 4 variants on one network; return list of (E_x, E_y, nu_xy, nu_yx)."""
    vx = solver.edge_vecs[:, :, 0]
    vy = solver.edge_vecs[:, :, 1]
    factor = rigidities / rest_lengths**2 / 16.0
    bare = torch.stack([
        (factor * vx**4).sum(1),
        (factor * vx**3 * vy).sum(1),
        (factor * vx**2 * vy**2).sum(1),
        (factor * vx * vy**3).sum(1),
        (factor * vy**4).sum(1),
    ], dim=1)  # (N, 5)

    areas_np = _triangle_areas(solver)           # (N,) numpy
    w_np     = areas_np / areas_np.sum()         # area weights, sum=1
    w_t      = torch.tensor(w_np, dtype=torch.float64).unsqueeze(1)  # (N,1)

    results = []
    for area_weighted, use_kkt, *_ in CASES:
        # Reference medium: arithmetic or area-weighted mean
        mean_t = (bare * w_t).sum(0) if area_weighted else bare.mean(0)
        delta  = bare - mean_t
        A_b    = _batch_to_9x9(bare)
        B_b    = _batch_to_9x9(delta)
        dA     = _batch_to_9vec(delta)

        with torch.no_grad():
            if area_weighted:
                if use_kkt and solver.kkt_arrays is not None:
                    # Large mesh: area-weighted sparse KKT (paper appendix)
                    W_np = _woodbury_kkt_sparse_aw(A_b, B_b, dA, solver.kkt_arrays, w_np)
                    W = torch.as_tensor(W_np, dtype=bare.dtype)
                else:
                    # Small mesh or no-KKT: area-weighted dense Woodbury
                    # Absorb w into B so standard formula applies (no /N → weights do it)
                    B_b_aw = B_b * w_t.unsqueeze(2)  # (N,9,9), each block scaled by w_n
                    S_aw   = torch.einsum('nij,njk->ik', B_b_aw, torch.linalg.inv(
                                 A_b + 1e-14*A_b.abs().max()*torch.eye(9,dtype=A_b.dtype)))
                    A_inv  = torch.linalg.inv(A_b + 1e-14*A_b.abs().max()*torch.eye(9,dtype=A_b.dtype))
                    y_aw   = torch.einsum('nij,nj->ni', A_inv, dA)
                    Vy_aw  = torch.einsum('nij,nj->i', B_b_aw, y_aw)
                    z_aw   = torch.linalg.solve(torch.eye(9,dtype=A_b.dtype) - S_aw, Vy_aw)
                    W      = -(y_aw + torch.einsum('nij,j->ni', A_inv, z_aw))  # no /N
            else:
                # Standard arithmetic Woodbury ± KKT
                if use_kkt and solver.kkt_arrays is not None:
                    W_np = _woodbury_kkt_sparse(A_b, B_b, dA, solver.kkt_arrays)
                    W = torch.as_tensor(W_np, dtype=bare.dtype)
                elif use_kkt and solver.J is not None:
                    W = _woodbury_solve(A_b, B_b, dA, J=solver.J.to(dtype=bare.dtype))
                else:
                    W = _woodbury_solve(A_b, B_b, dA, J=None)

        actual = _compute_actual_elastic_tensor(bare, W)
        # Homogenization matches the constraint used in the solve
        C6 = ((actual * w_t).sum(0) if area_weighted else actual.mean(0)).numpy()
        results.append(_directional_constants(C6))
    return results


# ── Compute or load ───────────────────────────────────────────────────────────
n_cases = len(CASES)
n_eta   = len(ETA_VALUES)

if os.path.exists(CACHE):
    print(f"Loading cached data from {CACHE}")
    d = np.load(CACHE)
    Ex_arr   = d['Ex_arr'];   Ey_arr   = d['Ey_arr']
    nuxy_arr = d['nuxy_arr']; nuyx_arr = d['nuyx_arr']
    etas     = d['etas']
else:
    Ex_arr   = np.full((n_cases, n_eta, N_TRIALS), np.nan)
    Ey_arr   = np.full((n_cases, n_eta, N_TRIALS), np.nan)
    nuxy_arr = np.full((n_cases, n_eta, N_TRIALS), np.nan)
    nuyx_arr = np.full((n_cases, n_eta, N_TRIALS), np.nan)
    t0 = time.time()

    for i_eta, eta in enumerate(ETA_VALUES):
        for trial in range(N_TRIALS):
            seed = 100 * i_eta + trial
            np.random.seed(seed)

            DT     = D2C.generate_foam_points(SIZE, eta)
            solver, rigs, rl = fst.from_triangulation(DT)
            res    = run_all_cases(solver, rigs, rl)

            for c, (Ex, Ey, nuxy, nuyx) in enumerate(res):
                Ex_arr[c, i_eta, trial]   = Ex
                Ey_arr[c, i_eta, trial]   = Ey
                nuxy_arr[c, i_eta, trial] = nuxy
                nuyx_arr[c, i_eta, trial] = nuyx

            elapsed = time.time() - t0
            row_nu = "  ".join(f"{res[c][2]:+.4f}" for c in range(n_cases))
            print(f"  η={eta:.2f} t={trial:2d}  ν_xy=[{row_nu}]  {elapsed:.0f}s", flush=True)
        print()

    etas = np.asarray(ETA_VALUES)
    np.savez(CACHE, Ex_arr=Ex_arr, Ey_arr=Ey_arr,
             nuxy_arr=nuxy_arr, nuyx_arr=nuyx_arr, etas=etas)
    print(f"Data saved to {CACHE}")


# ── Plotting ──────────────────────────────────────────────────────────────────
ETA_STABLE = 0.35
jitter_off = np.linspace(-0.009, 0.009, n_cases)


def _plot_panel(ax, arr, ylabel, etas, dir_label):
    for c, (_, _, label, color, marker, ls) in enumerate(CASES):
        data = arr[c]
        med = np.nanmedian(data, axis=1)
        q1  = np.nanpercentile(data, 25, axis=1)
        q3  = np.nanpercentile(data, 75, axis=1)
        ax.errorbar(etas + jitter_off[c], med,
                    yerr=[np.clip(med-q1, 0, None), np.clip(q3-med, 0, None)],
                    fmt=f'{marker}{ls}', color=color, capsize=3,
                    markersize=5, label=label, alpha=0.9)
        ax.fill_between(etas, q1, q3, alpha=0.08, color=color)
    ax.axvline(ETA_STABLE, color='gray', lw=1.2, ls=':', alpha=0.6)
    ax.set_xlabel(r'$\eta$', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f'Loading direction: {dir_label}', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)


# ── Figure 1: Poisson ratio ───────────────────────────────────────────────────
NU_LO, NU_HI = -1.0, 0.45
nuxy_clip = np.clip(nuxy_arr, NU_LO, NU_HI)
nuyx_clip = np.clip(nuyx_arr, NU_LO, NU_HI)

fig1, axes1 = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
_plot_panel(axes1[0], nuxy_clip, r"Poisson's ratio $\nu$", etas, r'$x$  (σ₂₂=0, σ₁₂=0)')
_plot_panel(axes1[1], nuyx_clip, r"Poisson's ratio $\nu$", etas, r'$y$  (σ₁₁=0, σ₁₂=0)')
for ax in axes1:
    ax.set_ylim(NU_LO, NU_HI)
fig1.suptitle(
    rf"Poisson's ratio $\nu(\eta)$ — {SIZE[0]}×{SIZE[1]} network, {N_TRIALS} trials  "
    r"[isotropy: left ≈ right]",
    fontsize=13, y=1.02)
fig1.tight_layout()
path1 = os.path.join(os.path.dirname(__file__), 'all_models_comp_nu.png')
fig1.savefig(path1, dpi=150, bbox_inches='tight')
print(f"Saved: {path1}")

# ── Figure 2: Young's modulus ─────────────────────────────────────────────────
valid_E = np.concatenate([Ex_arr[np.isfinite(Ex_arr)], Ey_arr[np.isfinite(Ey_arr)]])
E_lo = max(0.0, float(np.percentile(valid_E, 2)))
E_hi = float(np.percentile(valid_E, 98)) * 1.1
Ex_clip = np.clip(Ex_arr, E_lo, E_hi)
Ey_clip = np.clip(Ey_arr, E_lo, E_hi)

fig2, axes2 = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
_plot_panel(axes2[0], Ex_clip, r"Young's modulus $E$", etas, r'$x$  (σ₂₂=0, σ₁₂=0)')
_plot_panel(axes2[1], Ey_clip, r"Young's modulus $E$", etas, r'$y$  (σ₁₁=0, σ₁₂=0)')
for ax in axes2:
    ax.set_ylim(E_lo, E_hi)
fig2.suptitle(
    rf"Young's modulus $E(\eta)$ — {SIZE[0]}×{SIZE[1]} network, {N_TRIALS} trials  "
    r"[isotropy: left ≈ right]",
    fontsize=13, y=1.02)
fig2.tight_layout()
path2 = os.path.join(os.path.dirname(__file__), 'all_models_comp_E.png')
fig2.savefig(path2, dpi=150, bbox_inches='tight')
print(f"Saved: {path2}")

# ── Isotropy and variant comparison summary ───────────────────────────────────
print(f"\nIsotropy check — mean |ν_xy−ν_yx| and |E_x−E_y|/E_x:")
print(f"{'η':>5}", end='')
for _, _, label, *_ in CASES:
    print(f"  {label:>26}", end='')
print()
print('-' * (6 + 28 * n_cases))
for i, eta in enumerate(etas):
    print(f"  {eta:.2f}", end='')
    for c in range(n_cases):
        dnu = np.nanmean(np.abs(nuxy_arr[c,i] - nuyx_arr[c,i]))
        dE  = np.nanmean(np.abs(Ex_arr[c,i] - Ey_arr[c,i]) /
                         (np.abs(Ex_arr[c,i]) + 1e-12))
        print(f"  |Δν|={dnu:.4f} |ΔE/E|={dE:.4f}", end='')
    print()
