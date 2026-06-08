"""
Before vs after the /N accuracy fix — showing exactly what changed.

Only Std+edge was affected: the bug was in the non-AW (weights=None) branch
of _woodbury_kkt_sparse_combined. AW+edge, Std, AW were all unaffected.

Strategy: monkey-patch fst._woodbury_kkt_sparse_combined with the buggy
version, call solver.forward(use_kkt=True, area_weighted=False) to collect
pre-fix results, then restore.  Identical seeds/meshes as the fixed run.
"""
import sys, os, time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'Phase 2'))

import Disc_2_Cont_optimized as D2C
import forward_solver_torch as fst

ETA_VALUES = np.linspace(0.0, 0.5, 11)
SIZE       = (20, 20)
TRIM_FRAC  = 0.85
N_TRIALS   = 10

CACHE_BUGGY = os.path.join(os.path.dirname(__file__), 'mf20_buggy_stdedge.npz')
OUT         = os.path.join(os.path.dirname(__file__), 'new', 'before_after_fix.png')

# ── Buggy replacement for _woodbury_kkt_sparse_combined ──────────────────────
# Identical to the production function except for two /N lines restored
# in the non-AW (weights=None) branch.

def _woodbury_kkt_combined_BUGGY(A_blocks, B_blocks, dA_vecs,
                                  kkt_arrays, angle_arrays=None, weights=None):
    """Pre-fix version: two spurious /N divisions in non-AW Woodbury."""
    if weights is not None:
        # AW branch was correct — delegate to the real fixed function
        return fst._woodbury_kkt_sparse_combined_FIXED(
            A_blocks, B_blocks, dA_vecs, kkt_arrays, angle_arrays, weights)

    s1_arr, s2_arr, q_arr = kkt_arrays
    E_int  = len(s1_arr)
    M_c    = E_int
    N      = A_blocks.shape[0]

    A_np  = A_blocks.detach().double().numpy()
    B_np  = B_blocks.detach().double().numpy()
    dA_np = dA_vecs.detach().double().numpy()

    IDX3  = np.array([0, 3, 6])
    M_np  = A_np[:, IDX3[:, None], IDX3[None, :]]
    dM_np = B_np[:, IDX3[:, None], IDX3[None, :]]
    eps3  = 1e-14 * np.abs(M_np).max()
    I3    = np.eye(3)
    M_inv = np.linalg.inv(M_np + eps3 * I3[None])

    dA_3 = np.stack([dA_np[:, [k, 3+k, 6+k]] for k in range(3)], axis=2)
    y3   = np.einsum('nij,njk->nik', M_inv, dA_3)

    Vy3      = np.einsum('nij,njk->ik', dM_np, y3) / N   # ← BUG: /N
    S3       = np.einsum('nij,njk->ik', dM_np, M_inv) / N
    IminusS3 = I3 - S3
    Z3       = np.linalg.solve(IminusS3, Vy3)
    W0_3     = -(y3 + np.einsum('nij,jk->nik', M_inv, Z3) / N)

    W0 = np.zeros((N, 9))
    for k in range(3):
        W0[:, [k, 3+k, 6+k]] = W0_3[:, :, k]

    BMA_inv = np.einsum('nij,njk->nik', dM_np, M_inv)

    tri_edges = defaultdict(list)
    for e in range(E_int):
        tri_edges[s1_arr[e]].append((e, +1))
        tri_edges[s2_arr[e]].append((e, -1))
    rows_g, cols_g, data_g = [], [], []
    for n, elist in tri_edges.items():
        Mn = M_inv[n]
        for e1, sg1 in elist:
            for e2, sg2 in elist:
                val = sg1 * sg2 * float(q_arr[e1] @ Mn @ q_arr[e2])
                rows_g.append(e1); cols_g.append(e2); data_g.append(val)
    G_0  = sp.coo_matrix((data_g, (rows_g, cols_g)), shape=(M_c, M_c)).tocsc()
    reg  = 1e-12 * max(abs(v) for v in data_g) if data_g else 1e-12
    G_0  = G_0 + reg * sp.eye(M_c, format='csc')
    G_lu = spla.factorized(G_0)

    H0 = np.zeros((M_c, 3))
    K0 = np.zeros((M_c, 3))
    qM1  = np.einsum('el,elm->em', q_arr, M_inv[s1_arr])
    qM2  = np.einsum('el,elm->em', q_arr, M_inv[s2_arr])
    qBM1 = np.einsum('el,elm->em', q_arr, BMA_inv[s1_arr])
    qBM2 = np.einsum('el,elm->em', q_arr, BMA_inv[s2_arr])
    H0[:E_int] = qM1 - qM2
    K0[:E_int] = qBM1 - qBM2

    Y0     = np.column_stack([G_lu(H0[:, j]) for j in range(3)])
    M_mat3 = N * IminusS3 + K0.T @ Y0

    Lambda = np.zeros((M_c, 3))
    for k in range(3):
        r_k  = np.zeros(M_c)
        diff = W0_3[s1_arr, :, k] - W0_3[s2_arr, :, k]
        r_k[:E_int] = np.einsum('el,el->e', q_arr, diff)
        lam0  = G_lu(r_k)
        c_vec = np.linalg.solve(M_mat3, K0.T @ lam0)
        Lambda[:, k] = lam0 - Y0 @ c_vec

    CtLam_3 = np.zeros((N, 3, 3))
    for k in range(3):
        Lk_e = Lambda[:E_int, k]
        np.add.at(CtLam_3[:, :, k], s1_arr,  q_arr * Lk_e[:, None])
        np.add.at(CtLam_3[:, :, k], s2_arr, -q_arr * Lk_e[:, None])

    AinvCt = np.einsum('nij,njk->nik', M_inv, CtLam_3)
    gs3    = np.einsum('nij,njk->ik', dM_np, AinvCt) / N   # ← BUG: /N
    PinvCt = AinvCt + np.einsum('nij,jk->nik', M_inv,
                                  np.linalg.solve(IminusS3, gs3)) / N

    dW = np.zeros((N, 9))
    for k in range(3):
        dW[:, [k, 3+k, 6+k]] = PinvCt[:, :, k]
    return W0 - dW


# ── Mesh builders ─────────────────────────────────────────────────────────────

def make_df(eta):
    return D2C.generate_foam_distort_first(SIZE, eta, trim_frac=TRIM_FRAC)


def make_tf(eta):
    DM = D2C.generate_foam_points(SIZE, eta)
    centroids = np.mean(DM.points[DM.simplices], axis=1)
    cx, cy = centroids[:, 0], centroids[:, 1]
    xc = (cx.max() + cx.min()) / 2; yc = (cy.max() + cy.min()) / 2
    hw = (cx.max() - cx.min()) / 2; hh = (cy.max() - cy.min()) / 2
    mask = (np.abs(cx - xc) <= TRIM_FRAC * hw) & (np.abs(cy - yc) <= TRIM_FRAC * hh)
    DM.simplices = DM.simplices[mask]
    return DM


def ec(C6):
    C0, C1, C2, C3, C4, C5 = C6
    C_V = np.array([[C0, C2, C1], [C2, C5, C4], [C1, C4, C3]])
    try:
        S = np.linalg.inv(C_V)
    except Exception:
        return np.nan, np.nan
    Ex = 1 / S[0, 0]; Ey = 1 / S[1, 1]
    return 0.5 * (Ex + Ey), 0.5 * (-S[1, 0] * Ex + -S[0, 1] * Ey)


# ── Collect buggy results ─────────────────────────────────────────────────────

def run_buggy(cache):
    if os.path.exists(cache):
        print("  Loading buggy cache")
        d = np.load(cache)
        return {k: d[k] for k in d if k != 'etas'}

    # Save a reference to the fixed function so AW branch can still call it
    fst._woodbury_kkt_sparse_combined_FIXED = fst._woodbury_kkt_sparse_combined
    fst._woodbury_kkt_sparse_combined       = _woodbury_kkt_combined_BUGGY

    n_eta = len(ETA_VALUES)
    res   = {
        'E_df':  np.full((n_eta, N_TRIALS), np.nan),
        'nu_df': np.full((n_eta, N_TRIALS), np.nan),
        'E_tf':  np.full((n_eta, N_TRIALS), np.nan),
        'nu_tf': np.full((n_eta, N_TRIALS), np.nan),
    }
    t0 = time.time()
    try:
        for i_eta, eta in enumerate(ETA_VALUES):
            for trial in range(N_TRIALS):
                np.random.seed(100 * i_eta + trial)
                for key, builder in [('df', make_df), ('tf', make_tf)]:
                    try:
                        DM = builder(eta)
                        solver, rigs, rl = fst.from_triangulation(DM)
                        with torch.no_grad():
                            r = solver.forward(rigs, rl,
                                               use_kkt=True, area_weighted=False,
                                               use_angle_kkt=False)
                        C6 = r['elastic_tensor'].numpy()
                        E_v, nu_v = ec(C6)
                        res[f'E_{key}'][i_eta, trial]  = E_v
                        res[f'nu_{key}'][i_eta, trial] = nu_v
                    except Exception as exc:
                        print(f"    buggy {key} eta={eta:.2f} t={trial}: {exc}")
            elapsed = time.time() - t0
            print(f"  buggy eta={eta:.2f}  "
                  f"df E={np.nanmedian(res['E_df'][i_eta]):.4f} "
                  f"nu={np.nanmedian(res['nu_df'][i_eta]):+.4f}  "
                  f"tf E={np.nanmedian(res['E_tf'][i_eta]):.4f} "
                  f"nu={np.nanmedian(res['nu_tf'][i_eta]):+.4f}  [{elapsed:.0f}s]")
    finally:
        fst._woodbury_kkt_sparse_combined = fst._woodbury_kkt_sparse_combined_FIXED
        print("  Restored fixed function.")
    np.savez(cache, etas=ETA_VALUES, **res)
    return res


# ── Main ──────────────────────────────────────────────────────────────────────

t_start = time.time()
print("=== Running buggy Std+edge on DF + TF meshes ===")
buggy = run_buggy(CACHE_BUGGY)

print("Loading fixed results...")
d_df = np.load(os.path.join(os.path.dirname(__file__), 'mf20_df_data.npz'))
d_tf = np.load(os.path.join(os.path.dirname(__file__), 'mf20_tf_data.npz'))
d_sim_df = np.load(os.path.join(os.path.dirname(__file__), 'sim20_df_data.npz'))
d_sim_tf = np.load(os.path.join(os.path.dirname(__file__), 'sim20_tf_data.npz'))
print(f"All done in {time.time()-t_start:.0f}s")

etas = ETA_VALUES

def med(arr): return np.nanmedian(arr, axis=1)
def q25(arr): return np.nanpercentile(arr, 25, axis=1)
def q75(arr): return np.nanpercentile(arr, 75, axis=1)


def norm(arr2d):
    E0 = np.nanmedian(arr2d[0])
    return arr2d / max(abs(E0), 1e-10)


# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
fig.suptitle('Before vs After /N accuracy fix — Std+edge, 20×20 mesh', fontsize=13)

panel_cfg = [
    (axes[0, 0], 'DF: E/E₀ vs η',
     norm(d_sim_df['E_sim']), norm(d_df['E_Std+edge']), norm(buggy['E_df']),
     'E/E₀', True),
    (axes[0, 1], 'TF: E/E₀ vs η',
     norm(d_sim_tf['E_sim']), norm(d_tf['E_Std+edge']), norm(buggy['E_tf']),
     'E/E₀', True),
    (axes[1, 0], 'DF: ν vs η',
     d_sim_df['nu_sim'], d_df['nu_Std+edge'], buggy['nu_df'],
     'ν',    False),
    (axes[1, 1], 'TF: ν vs η',
     d_sim_tf['nu_sim'], d_tf['nu_Std+edge'], buggy['nu_tf'],
     'ν',    False),
]

for ax, title, sim_arr, fixed_arr, buggy_arr, ylabel, clip_e in panel_cfg:
    ax.set_title(title, fontsize=11)
    ax.set_xlabel('η')
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)

    # Simulation
    ax.fill_between(etas, q25(sim_arr), q75(sim_arr), color='gray', alpha=0.2)
    ax.plot(etas, med(sim_arr), 'k-o', lw=2, ms=5, label='Simulation')

    # Fixed (after fix)
    ax.plot(etas, med(fixed_arr), color='#2ca02c', ls='-',  marker='s', ms=5,
            lw=1.8, label='Std+edge (fixed)')

    # Buggy (before fix)
    ax.plot(etas, med(buggy_arr), color='#d62728', ls='--', marker='^', ms=5,
            lw=1.8, label='Std+edge (pre-fix)')

    if not clip_e:
        ax.axhline(0, color='gray', lw=0.5, ls=':')

    ax.legend(fontsize=9)

# Clip extreme E/E0 for readability
for ax in [axes[0, 0], axes[0, 1]]:
    ax.set_ylim(-0.05, 1.6)

plt.tight_layout()
plt.savefig(OUT, dpi=150, bbox_inches='tight')
print(f"Saved: {OUT}")
