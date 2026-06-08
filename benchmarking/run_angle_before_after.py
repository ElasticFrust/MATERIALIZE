"""
6-method before/after K0-transpose fix comparison.

K0 controls how the mean-field Woodbury correction feeds into the KKT multiplier
equation. The bug: BMA_inv (= δM·M⁻¹) was contracted with q/a in the wrong order
(effectively transposed), affecting all KKT methods (edge and angle).

Unaffected: Std, AW (no KKT).
Affected:   Std+edge, AW+edge, Std+full, AW+full.

Runs DF 20×20, 10 trials, same seeds. Monkey-patches both bugs back for 'before'.
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

ETA_VALUES  = np.linspace(0.0, 0.5, 11)
SIZE        = (20, 20)
TRIM_FRAC   = 0.85
N_TRIALS    = 10

CACHE_BUGGY = os.path.join(os.path.dirname(__file__), 'angle_buggy_data.npz')
CACHE_FIXED = os.path.join(os.path.dirname(__file__), 'angle_fixed_data.npz')
OUT         = os.path.join(os.path.dirname(__file__), 'new', 'angle_kkt_before_after.png')

CASES = [
    (False, False, False, 'Std'),
    (False, True,  False, 'Std+edge'),
    (False, True,  True,  'Std+full'),
    (True,  False, False, 'AW'),
    (True,  True,  False, 'AW+edge'),
    (True,  True,  True,  'AW+full'),
]


def make_df(eta):
    return D2C.generate_foam_distort_first(SIZE, eta, trim_frac=TRIM_FRAC)


def ec(C6):
    C0, C1, C2, C3, C4, C5 = C6
    C_V = np.array([[C0, C2, C1], [C2, C5, C4], [C1, C4, C3]])
    try:
        S = np.linalg.inv(C_V)
    except Exception:
        return np.nan, np.nan
    Ex = 1 / S[0, 0]; Ey = 1 / S[1, 1]
    return 0.5 * (Ex + Ey), 0.5 * (-S[1, 0] * Ex + -S[0, 1] * Ey)


# ── Buggy replacement (both qBM and aBM transposed) ───────────────────────────

def _woodbury_kkt_BUGGY(A_blocks, B_blocks, dA_vecs,
                         kkt_arrays, angle_arrays=None, weights=None):
    """Pre-fix: qBM1/qBM2 use 'elm' and aBM uses 'plm' (both transposed)."""
    s1_arr, s2_arr, q_arr = kkt_arrays
    E_int  = len(s1_arr)
    if angle_arrays is not None:
        v_arr, sv_arr, a_arr, n_int = angle_arrays
    else:
        v_arr = np.empty(0, np.int64); sv_arr = np.empty(0, np.int64)
        a_arr = np.empty((0, 3)); n_int = 0
    n_pairs = len(v_arr)
    M_c = E_int + n_int
    N   = A_blocks.shape[0]

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

    if weights is not None:
        w   = weights
        Vy3 = np.einsum('n,nij,njk->ik', w, dM_np, y3)
        S3  = np.einsum('n,nij,njk->ik', w, dM_np, M_inv)
    else:
        Vy3 = np.einsum('nij,njk->ik', dM_np, y3)
        S3  = np.einsum('nij,njk->ik', dM_np, M_inv) / N
    IminusS3 = I3 - S3
    Z3  = np.linalg.solve(IminusS3, Vy3)
    if weights is not None:
        W0_3 = -(y3 + np.einsum('nij,jk->nik', M_inv, Z3))
    else:
        W0_3 = -(y3 + np.einsum('nij,jk->nik', M_inv, Z3) / N)

    W0 = np.zeros((N, 9))
    for k in range(3):
        W0[:, [k, 3+k, 6+k]] = W0_3[:, :, k]

    if weights is not None:
        BMA_inv = np.einsum('n,nij,njk->nik', w, dM_np, M_inv)
    else:
        BMA_inv = np.einsum('nij,njk->nik', dM_np, M_inv)

    # G_0 sparse
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
    if n_pairs > 0:
        ord_ = np.argsort(sv_arr, kind='stable')
        sv_s = sv_arr[ord_]; v_s = v_arr[ord_]; a_s = a_arr[ord_]
        bnd  = np.where(np.diff(sv_s, prepend=-1, append=-1))[0]
        for i in range(len(bnd) - 1):
            sl = slice(bnd[i], bnd[i+1])
            n  = sv_s[bnd[i]]
            a_g = a_s[sl]; v_g = v_s[sl]
            G_vals = a_g @ M_inv[n] @ a_g.T
            ri = E_int + v_g; ci = E_int + v_g
            ri2, ci2 = np.meshgrid(ri, ci, indexing='ij')
            rows_g.extend(ri2.ravel()); cols_g.extend(ci2.ravel())
            data_g.extend(G_vals.ravel())
        tri_angle = defaultdict(list)
        for j in range(n_pairs):
            tri_angle[sv_arr[j]].append(j)
        for n, jlist in tri_angle.items():
            if n not in tri_edges:
                continue
            Mn = M_inv[n]
            for (e, sg) in tri_edges[n]:
                for j in jlist:
                    val = sg * float(q_arr[e] @ Mn @ a_arr[j])
                    rows_g.append(e); cols_g.append(E_int + v_arr[j]); data_g.append(val)
                    rows_g.append(E_int + v_arr[j]); cols_g.append(e); data_g.append(val)
    G_0 = sp.coo_matrix((data_g, (rows_g, cols_g)), shape=(M_c, M_c)).tocsc()
    reg = 1e-12 * max(abs(v) for v in data_g) if data_g else 1e-12
    G_0 = G_0 + reg * sp.eye(M_c, format='csc')
    G_lu = spla.factorized(G_0)

    H0 = np.zeros((M_c, 3)); K0 = np.zeros((M_c, 3))
    qM1  = np.einsum('el,elm->em', q_arr, M_inv[s1_arr])
    qM2  = np.einsum('el,elm->em', q_arr, M_inv[s2_arr])
    # ← BUGGY: elm instead of eml for BMA
    qBM1 = np.einsum('el,elm->em', q_arr, BMA_inv[s1_arr])
    qBM2 = np.einsum('el,elm->em', q_arr, BMA_inv[s2_arr])
    H0[:E_int] = qM1 - qM2
    K0[:E_int] = qBM1 - qBM2
    if n_pairs > 0:
        aM  = np.einsum('pl,plm->pm', a_arr, M_inv[sv_arr])
        # ← BUGGY: plm instead of pml for BMA
        aBM = np.einsum('pl,plm->pm', a_arr, BMA_inv[sv_arr])
        np.add.at(H0, E_int + v_arr, aM)
        np.add.at(K0, E_int + v_arr, aBM)

    Y0 = np.column_stack([G_lu(H0[:, j]) for j in range(3)])
    M_mat3 = (IminusS3 if weights is not None else N * IminusS3) + K0.T @ Y0

    Lambda = np.zeros((M_c, 3))
    for k in range(3):
        r_k = np.zeros(M_c)
        diff = W0_3[s1_arr, :, k] - W0_3[s2_arr, :, k]
        r_k[:E_int] = np.einsum('el,el->e', q_arr, diff)
        if n_pairs > 0:
            np.add.at(r_k, E_int + v_arr,
                      np.einsum('pl,pl->p', a_arr, W0_3[sv_arr, :, k]))
        lam0  = G_lu(r_k)
        c_vec = np.linalg.solve(M_mat3, K0.T @ lam0)
        Lambda[:, k] = lam0 - Y0 @ c_vec

    CtLam_3 = np.zeros((N, 3, 3))
    for k in range(3):
        Lk_e = Lambda[:E_int, k]
        np.add.at(CtLam_3[:, :, k], s1_arr,  q_arr * Lk_e[:, None])
        np.add.at(CtLam_3[:, :, k], s2_arr, -q_arr * Lk_e[:, None])
        if n_pairs > 0:
            Lk_a = Lambda[E_int + v_arr, k]
            np.add.at(CtLam_3[:, :, k], sv_arr, a_arr * Lk_a[:, None])

    AinvCt = np.einsum('nij,njk->nik', M_inv, CtLam_3)
    if weights is not None:
        gs3    = np.einsum('n,nij,njk->ik', w, dM_np, AinvCt)
        PinvCt = AinvCt + np.einsum('nij,jk->nik', M_inv,
                                     np.linalg.solve(IminusS3, gs3))
    else:
        gs3    = np.einsum('nij,njk->ik', dM_np, AinvCt)
        PinvCt = AinvCt + np.einsum('nij,jk->nik', M_inv,
                                     np.linalg.solve(IminusS3, gs3)) / N

    dW = np.zeros((N, 9))
    for k in range(3):
        dW[:, [k, 3+k, 6+k]] = PinvCt[:, :, k]
    return W0 - dW


# ── Run helper ─────────────────────────────────────────────────────────────────

def run_all_cases(cache, buggy=False):
    if os.path.exists(cache):
        print(f"  Loading {'buggy' if buggy else 'fixed'} cache")
        d = np.load(cache)
        return {k: d[k] for k in d if k != 'etas'}

    if buggy:
        fst._woodbury_kkt_sparse_combined_FIXED = fst._woodbury_kkt_sparse_combined
        fst._woodbury_kkt_sparse_combined       = _woodbury_kkt_BUGGY

    n_eta = len(ETA_VALUES)
    res   = {}
    for c in CASES:
        k = c[3]
        res[f'E_{k}']  = np.full((n_eta, N_TRIALS), np.nan)
        res[f'nu_{k}'] = np.full((n_eta, N_TRIALS), np.nan)
    t0 = time.time()
    try:
        for i_eta, eta in enumerate(ETA_VALUES):
            for trial in range(N_TRIALS):
                np.random.seed(100 * i_eta + trial)
                try:
                    DM = make_df(eta)
                    solver, rigs, rl = fst.from_triangulation(DM)
                    with torch.no_grad():
                        for aw, kkt, ang, key in CASES:
                            r = solver.forward(rigs, rl, area_weighted=aw,
                                               use_kkt=kkt, use_angle_kkt=ang)
                            E_v, nu_v = ec(r['elastic_tensor'].numpy())
                            res[f'E_{key}'][i_eta, trial]  = E_v
                            res[f'nu_{key}'][i_eta, trial] = nu_v
                except Exception as exc:
                    print(f"    eta={eta:.2f} t={trial}: {exc}")
            elapsed = time.time() - t0
            tag = 'buggy' if buggy else 'fixed'
            print(f"  [{tag}] eta={eta:.2f}  "
                  f"Std+edge nu={np.nanmedian(res['nu_Std+edge'][i_eta]):+.4f}  "
                  f"Std+full nu={np.nanmedian(res['nu_Std+full'][i_eta]):+.4f}  "
                  f"[{elapsed:.0f}s]", flush=True)
    finally:
        if buggy:
            fst._woodbury_kkt_sparse_combined = fst._woodbury_kkt_sparse_combined_FIXED
    np.savez(cache, etas=ETA_VALUES, **res)
    return res


# ── Main ──────────────────────────────────────────────────────────────────────
t_start = time.time()
print("=== Buggy (pre-fix K0) ===")
buggy = run_all_cases(CACHE_BUGGY, buggy=True)
print("=== Fixed ===")
fixed = run_all_cases(CACHE_FIXED, buggy=False)
print(f"Done in {time.time()-t_start:.0f}s")

# Load simulation
d_sim = np.load(os.path.join(os.path.dirname(__file__), 'sim20_df_data.npz'))

etas = ETA_VALUES
def med(arr): return np.nanmedian(arr, axis=1)
def norm(arr2d):
    E0 = np.nanmedian(arr2d[0])
    return arr2d / max(abs(E0), 1e-10)

# ── Plot: 2 rows (E/E0, ν) × 3 cols (no-KKT, edge-only, full) ────────────────
COLORS_AW   = {'Std': '#1f77b4', 'AW': '#ff7f0e',
               'Std+edge': '#2ca02c', 'AW+edge': '#d62728',
               'Std+full': '#9467bd', 'AW+full': '#8c564b'}

fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True)
fig.suptitle('6-method angle/edge KKT — before vs after K0 transpose fix\n'
             'DF 20×20 mesh, 10 trials', fontsize=12)

col_methods = [
    ('No KKT',    ['Std', 'AW']),
    ('Edge only', ['Std+edge', 'AW+edge']),
    ('Edge+angle',['Std+full', 'AW+full']),
]

sim_E_n  = norm(d_sim['E_sim'])
sim_nu   = d_sim['nu_sim']

for col, (col_title, keys) in enumerate(col_methods):
    ax_E  = axes[0, col]
    ax_nu = axes[1, col]
    ax_E.set_title(col_title, fontsize=11)
    for ax in (ax_E, ax_nu):
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('η')

    # Simulation
    ax_E.fill_between(etas, np.nanpercentile(sim_E_n, 25, 1),
                      np.nanpercentile(sim_E_n, 75, 1), color='gray', alpha=0.2)
    ax_E.plot(etas, med(sim_E_n),  'k-o', lw=2, ms=4, label='Sim')
    ax_nu.fill_between(etas, np.nanpercentile(sim_nu, 25, 1),
                       np.nanpercentile(sim_nu, 75, 1), color='gray', alpha=0.2)
    ax_nu.plot(etas, med(sim_nu), 'k-o', lw=2, ms=4, label='Sim')

    for key in keys:
        c = COLORS_AW[key]
        E_fix  = norm(fixed[f'E_{key}'])
        nu_fix = fixed[f'nu_{key}']
        E_bug  = norm(buggy[f'E_{key}'])
        nu_bug = buggy[f'nu_{key}']
        ax_E.plot(etas, med(E_fix),  color=c, ls='-',  marker='s', ms=4, lw=1.8, label=f'{key} (fixed)')
        ax_E.plot(etas, med(E_bug),  color=c, ls='--', marker='^', ms=3, lw=1.2, alpha=0.6, label=f'{key} (pre-fix)')
        ax_nu.plot(etas, med(nu_fix), color=c, ls='-',  marker='s', ms=4, lw=1.8, label=f'{key} (fixed)')
        ax_nu.plot(etas, med(nu_bug), color=c, ls='--', marker='^', ms=3, lw=1.2, alpha=0.6, label=f'{key} (pre-fix)')

    ax_E.set_ylabel('E/E₀')
    ax_nu.set_ylabel('ν')
    ax_nu.axhline(0, color='gray', lw=0.5, ls=':')
    ax_E.legend(fontsize=7)
    ax_nu.legend(fontsize=7)

# Clip E for readability (angle can blow up)
for ax in axes[0]:
    ax.set_ylim(-0.1, 2.0)

plt.tight_layout()
plt.savefig(OUT, dpi=150, bbox_inches='tight')
print(f"Saved: {OUT}")
