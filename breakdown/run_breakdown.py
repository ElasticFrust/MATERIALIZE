"""
PBC simulation vs MF methods — per-triangle W comparison.

W_sim[s, :, α] = actual non-affine metric change of triangle s per unit applied
strain δ for mode α (= (g_def - g_aff_local)/δ from the sparse linear solve).

D2C defines the non-affine response tensor W via:
  δg(s) = W(s) · Δg   (paper sign convention, W > 0 for soft/long-bond triangles)

The code computes W_code via (A-B)W = -δA where δA = A_s - Ā (triangle minus mean).
The paper uses δA_paper = Ā - A_s (opposite sign), so W_code = -W_paper.
Therefore the correct D2C prediction is:

  W_mf_pred[s, :, α] = -W_code[s] @ G̅[:,α] = W_paper[s] @ G̅[:,α] ≈ W_sim

where G̅[:,α] = mean_s(Δg_aff(s,α)) is the mean affine metric change per unit δ.
"""
import sys, os, time
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'Phase 2'))
sys.path.insert(0, HERE)  # for periodic_mesh and pbc_simulation

import forward_solver_torch as fst
from periodic_mesh import make_periodic_tf_mesh, build_bare_tensors
from pbc_simulation import extract_W_sim, extract_pbc_elastic_tensor

ETA_VALUES = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
N          = 20
N_TRIALS   = 5
PLOT_ETA   = 0.4

PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

CASES = [
    (False, False, 'Std'),
    (True,  False, 'AW'),
    (False, True,  'Std+edge'),
    (True,  True,  'AW+edge'),
]
COLORS  = {'Std': '#1f77b4', 'AW': '#ff7f0e',
           'Std+edge': '#2ca02c', 'AW+edge': '#d62728'}
MARKERS = {'Std': 'o', 'AW': 'o', 'Std+edge': 's', 'AW+edge': 's'}
LS      = {'Std': '-', 'AW': '--', 'Std+edge': '-', 'AW+edge': '--'}


def elastic_constants_from_C_V(C_V):
    """3×3 Voigt stiffness → (E, nu)."""
    try:
        S = np.linalg.inv(C_V)
        Ex = 1.0 / S[0, 0]; Ey = 1.0 / S[1, 1]
        return 0.5 * (Ex + Ey), 0.5 * (-S[1, 0] * Ex - S[0, 1] * Ey)
    except Exception:
        return np.nan, np.nan


def compute_Gbar_mat(mesh):
    """Compute 3×3 mean affine loading matrix G̅.

    G̅[:,α] = mean_s(Δg_aff(s, ε_α)) / δ   for α ∈ {xx, yy, xy}.
    Metric convention: [g11, 2*g12, g22] engineering Voigt.
    Δg_aff(s, ε_xx) = [2*e01x², 4*e01x*e02x, 2*e02x²]
    Δg_aff(s, ε_yy) = [2*e01y², 4*e01y*e02y, 2*e02y²]
    Δg_aff(s, ε_xy) = [2*e01x*e01y, 2*(e01x*e02y+e01y*e02x), 2*e02x*e02y]
    """
    ev = mesh['edge_vecs']   # (N_tri, 3, 2)
    e1x = ev[:, 0, 0];  e1y = ev[:, 0, 1]
    e2x = ev[:, 1, 0];  e2y = ev[:, 1, 1]

    G = np.zeros((3, 3))
    # Mode 0: ε_xx
    G[:, 0] = np.mean(np.stack([2*e1x**2, 4*e1x*e2x, 2*e2x**2], axis=1), axis=0)
    # Mode 1: ε_yy
    G[:, 1] = np.mean(np.stack([2*e1y**2, 4*e1y*e2y, 2*e2y**2], axis=1), axis=0)
    # Mode 2: ε_xy (symmetric half-shear)
    G[:, 2] = np.mean(np.stack([2*e1x*e1y,
                                  2*(e1x*e2y + e1y*e2x),
                                  2*e2x*e2y], axis=1), axis=0)
    return G   # (3, 3)


def run_mf(mesh, area_weighted, use_kkt):
    """Run one MF variant; return (W_mf_pred (N_tri,3,3), E, nu).

    W_mf_pred[s,:,α] = W_mf[s] @ G̅[:,α]  — in the same basis as W_sim.
    """
    bare_np = build_bare_tensors(mesh)
    N_tri = bare_np.shape[0]
    bare = torch.as_tensor(bare_np, dtype=torch.float64)
    areas_np = mesh['areas']
    w_np = areas_np / areas_np.sum() if area_weighted else None

    if area_weighted:
        mean_np = (bare_np * w_np[:, None]).sum(axis=0)
    else:
        mean_np = bare_np.mean(axis=0)

    mean_t = torch.as_tensor(mean_np, dtype=torch.float64)
    delta_b = bare - mean_t
    A_bl = fst._batch_to_9x9(bare)
    B_bl = fst._batch_to_9x9(delta_b)
    dA   = fst._batch_to_9vec(delta_b)

    if use_kkt:
        W_np = fst._woodbury_kkt_sparse_combined(
            A_bl, B_bl, dA, mesh['kkt_arrays'], None, weights=w_np)
    else:
        w_t = torch.as_tensor(w_np, dtype=torch.float64) if w_np is not None else None
        W_np = fst._woodbury_solve(A_bl, B_bl, dA, J=None, weights=w_t).detach().numpy()

    # W_np (N_tri,9) → W_mf (N_tri,3,3): W_mf[s,i,k]=response of metric i to MF mode k
    W_mf = W_np.reshape(N_tri, 3, 3)

    # Code W_code = -(A-B)^{-1}(A_s - Ā) = -W_paper (opposite sign to paper's δA = Ā - A_s).
    # Paper: δg = W_paper · Δg, so the correct prediction is -W_code @ Gbar = W_paper @ Gbar.
    Gbar = compute_Gbar_mat(mesh)               # (3, 3)
    W_mf_pred = -np.einsum('sik,ka->sia', W_mf, Gbar)  # (N_tri, 3, 3)

    # Elastic constants
    W_t = torch.as_tensor(W_np, dtype=torch.float64)
    actual = fst._compute_actual_elastic_tensor(bare, W_t)  # (N_tri, 6)
    if area_weighted:
        C6 = (actual.detach().numpy() * w_np[:, None]).sum(axis=0)
    else:
        C6 = actual.detach().numpy().mean(axis=0)

    # C6 = [C1111, C1112, C1212, C1221, C2122, C2222]
    # Voigt 3×3 using Cauchy symmetry C1122=C1212 (valid for spring networks)
    C_V = np.array([[C6[0], C6[2], C6[1]],
                    [C6[2], C6[5], C6[4]],
                    [C6[1], C6[4], C6[3]]])
    E, nu = elastic_constants_from_C_V(C_V)
    return W_mf_pred, E, nu


def compute_W_discrepancy(W_sim, W_mf_pred, mesh):
    """Per-triangle discrepancy measures.

    W_sim, W_mf_pred: (N_tri, 3, 3) — both in strain basis (per unit δ).
    """
    dW = W_sim - W_mf_pred           # (N_tri, 3, 3)

    dW_frob   = np.sqrt((dW**2).sum(axis=(1, 2)))
    Wsim_frob = np.sqrt((W_sim**2).sum(axis=(1, 2)))
    rel_err   = dW_frob / np.maximum(Wsim_frob, 1e-12)

    # Flat reference metric ḡ = [1, 0, 1] (equilateral unit cell, g12=0)
    g_flat = np.array([1.0, 0.0, 1.0])

    # Bulk (isotropic) part: contraction ḡ·ΔW·ḡ, averaged over loading modes
    # shape: (N_tri, 3_modes) → scalar per triangle
    bulk = np.einsum('i,sia,i->sa', g_flat, dW, g_flat)   # (N_tri, 3)
    dW_bulk_norm  = np.sqrt((bulk**2).sum(axis=1))

    # Shear (deviatoric) part: |dW| - |bulk component| in Frobenius sense
    # Bulk projector in (3,3) space per mode: dW - bulk*(ḡ⊗ḡ/||ḡ||²)
    g2 = g_flat @ g_flat
    bulk_proj = np.einsum('sa,i->sia', bulk / g2, g_flat)  # (N_tri, 3, 3)
    dW_dev = dW - bulk_proj
    dW_shear_norm = np.sqrt((dW_dev**2).sum(axis=(1, 2)))

    # Min triangle angle for scatter coloring
    ev = mesh['edge_vecs']
    e01 = ev[:, 0]; e02 = ev[:, 1]; e12 = ev[:, 2]
    def cos_a(a, b):
        return np.clip(
            (a * b).sum(1) / np.maximum(
                np.sqrt((a**2).sum(1)) * np.sqrt((b**2).sum(1)), 1e-30),
            -1, 1)
    ang0 = np.arccos(cos_a(e01, e02))
    ang1 = np.arccos(cos_a(-e01, e12))
    ang2 = np.abs(np.pi - ang0 - ang1)
    min_angle = np.degrees(np.minimum(ang0, np.minimum(ang1, ang2)))

    return {
        'dW_frob':       dW_frob,
        'rel_err':       rel_err,
        'dW_bulk_norm':  dW_bulk_norm,
        'dW_shear_norm': dW_shear_norm,
        'Wsim_frob':     Wsim_frob,
        'min_angle':     min_angle,
        'dW':            dW,
    }


# ── Main loop ─────────────────────────────────────────────────────────────────

n_eta = len(ETA_VALUES)
res = {
    'E_pbc':  np.full((n_eta, N_TRIALS), np.nan),
    'nu_pbc': np.full((n_eta, N_TRIALS), np.nan),
}
for _, _, lb in CASES:
    res[f'E_{lb}']          = np.full((n_eta, N_TRIALS), np.nan)
    res[f'nu_{lb}']         = np.full((n_eta, N_TRIALS), np.nan)
    res[f'rel_err_{lb}']    = [[] for _ in range(n_eta)]
    res[f'dW_bulk_{lb}']    = [[] for _ in range(n_eta)]
    res[f'dW_shear_{lb}']   = [[] for _ in range(n_eta)]

scatter_mf  = {lb: None for _, _, lb in CASES}
scatter_sim_frob   = None
scatter_min_angle  = None

t0 = time.time()
for i_eta, eta in enumerate(ETA_VALUES):
    print(f'\n── eta={eta:.1f} ──')
    for trial in range(N_TRIALS):
        seed = 1000 * i_eta + trial
        try:
            mesh = make_periodic_tf_mesh(N, eta, seed=seed)

            _, E_p, nu_p = extract_pbc_elastic_tensor(mesh, delta=1e-3)
            W_sim = extract_W_sim(mesh, delta=1e-3)

            res['E_pbc'][i_eta, trial]  = E_p
            res['nu_pbc'][i_eta, trial] = nu_p

            for aw, kkt, lb in CASES:
                W_pred, E_m, nu_m = run_mf(mesh, aw, kkt)
                res[f'E_{lb}'][i_eta, trial]  = E_m
                res[f'nu_{lb}'][i_eta, trial] = nu_m

                disc = compute_W_discrepancy(W_sim, W_pred, mesh)
                res[f'rel_err_{lb}'][i_eta].extend(disc['rel_err'].tolist())
                res[f'dW_bulk_{lb}'][i_eta].extend(disc['dW_bulk_norm'].tolist())
                res[f'dW_shear_{lb}'][i_eta].extend(disc['dW_shear_norm'].tolist())

                if abs(eta - PLOT_ETA) < 1e-9 and trial == 0:
                    scatter_mf[lb]       = disc
                    if scatter_sim_frob is None:
                        scatter_sim_frob  = disc['Wsim_frob']
                        scatter_min_angle = disc['min_angle']

        except Exception as exc:
            import traceback; traceback.print_exc()
            print(f'  eta={eta:.1f} trial={trial}: {exc}')

    for _, _, lb in CASES:
        rl = [x for x in res[f'rel_err_{lb}'][i_eta] if np.isfinite(x)]
        print(f'  {lb:10s} E={np.nanmedian(res[f"E_{lb}"][i_eta]):.4f} '
              f'nu={np.nanmedian(res[f"nu_{lb}"][i_eta]):+.4f} '
              f'relW_med={np.median(rl):.3f}' if rl else f'  {lb}: no data')
    print(f'  PBC       E={np.nanmedian(res["E_pbc"][i_eta]):.4f} '
          f'nu={np.nanmedian(res["nu_pbc"][i_eta]):+.4f} '
          f'[{time.time()-t0:.0f}s]')

print(f'\nTotal: {time.time()-t0:.0f}s')

# ── Helpers ───────────────────────────────────────────────────────────────────

def med(a):       return np.nanmedian(a, axis=1)
def q25(a):       return np.nanpercentile(a, 25, axis=1)
def q75(a):       return np.nanpercentile(a, 75, axis=1)
def med_l(ll):    return np.array([np.median(x)    if x else np.nan for x in ll])
def q25_l(ll):    return np.array([np.percentile(x, 25) if x else np.nan for x in ll])
def q75_l(ll):    return np.array([np.percentile(x, 75) if x else np.nan for x in ll])

etas = ETA_VALUES

# ── Normalize E by each method's η=0 value ───────────────────────────────────
def norm_eta0(arr2d):
    E0 = np.nanmedian(arr2d[0])
    return arr2d / max(abs(E0), 1e-12)

E_pbc_n = norm_eta0(res['E_pbc'])
E_mf_n  = {lb: norm_eta0(res[f'E_{lb}']) for _, _, lb in CASES}

# ── Plot 1: E/E₀ and ν vs η ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(f'PBC simulation vs MF — {N}×{N} periodic TF mesh, {N_TRIALS} trials',
             fontsize=12)

ax = axes[0]
ax.fill_between(etas, q25(E_pbc_n), q75(E_pbc_n), color='gray', alpha=0.25)
ax.plot(etas, med(E_pbc_n), 'k-o', lw=2.5, ms=6, label='PBC sim', zorder=10)
for _, _, lb in CASES:
    ax.plot(etas, med(E_mf_n[lb]), color=COLORS[lb], ls=LS[lb],
            marker=MARKERS[lb], ms=4, lw=1.6, label=lb)
ax.set_xlabel('η'); ax.set_ylabel('E/E₀'); ax.set_title("Young's modulus E/E₀ vs η")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

ax = axes[1]
ax.fill_between(etas, q25(res['nu_pbc']), q75(res['nu_pbc']), color='gray', alpha=0.25)
ax.plot(etas, med(res['nu_pbc']), 'k-o', lw=2.5, ms=6, label='PBC sim', zorder=10)
for _, _, lb in CASES:
    ax.plot(etas, med(res[f'nu_{lb}']), color=COLORS[lb], ls=LS[lb],
            marker=MARKERS[lb], ms=4, lw=1.6, label=lb)
ax.axhline(0, color='gray', lw=0.5, ls=':')
ax.set_xlabel('η'); ax.set_ylabel('ν'); ax.set_title('Poisson ratio ν vs η')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'E_nu_vs_eta.png'), dpi=150, bbox_inches='tight')
plt.close()

# ── Plot 2: W relative error and bulk/shear decomposition ─────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Per-triangle W discrepancy vs η', fontsize=12)

ax = axes[0]
ax.set_title('Relative W error: ||ΔW||_F / ||W_sim||_F')
for _, _, lb in CASES:
    ll = res[f'rel_err_{lb}']
    m = med_l(ll); lo = q25_l(ll); hi = q75_l(ll)
    ax.fill_between(etas, lo, hi, color=COLORS[lb], alpha=0.15)
    ax.plot(etas, m, color=COLORS[lb], ls=LS[lb],
            marker=MARKERS[lb], ms=4, lw=1.6, label=lb)
ax.set_xlabel('η'); ax.set_ylabel('||ΔW||_F / ||W_sim||_F')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

ax = axes[1]
ax.set_title('Bulk vs shear ΔW norm')
for _, _, lb in CASES:
    mb = med_l(res[f'dW_bulk_{lb}'])
    ms_d = med_l(res[f'dW_shear_{lb}'])
    ax.plot(etas, mb, color=COLORS[lb], ls='-',
            marker=MARKERS[lb], ms=4, lw=1.6, label=f'{lb} bulk')
    ax.plot(etas, ms_d, color=COLORS[lb], ls='--',
            ms=3, lw=1.0, label=f'{lb} shear', alpha=0.7)
ax.set_xlabel('η'); ax.set_ylabel('||ΔW component||')
ax.legend(fontsize=7.5, ncol=2); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'W_discrepancy_vs_eta.png'), dpi=150, bbox_inches='tight')
plt.close()

# ── Plot 3: per-triangle scatter at PLOT_ETA ──────────────────────────────────
if scatter_sim_frob is not None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle(f'Per-triangle ΔW at η={PLOT_ETA}, trial 0\n'
                 '(color = log₁₀||W_sim||_F, x = min triangle angle)',
                 fontsize=11)
    for idx, (_, _, lb) in enumerate(CASES):
        ax = axes[idx // 2, idx % 2]
        disc = scatter_mf[lb]
        sc = ax.scatter(scatter_min_angle, disc['rel_err'],
                        c=np.log10(np.maximum(scatter_sim_frob, 1e-12)),
                        cmap='viridis', s=8, alpha=0.7)
        plt.colorbar(sc, ax=ax, label='log₁₀||W_sim||_F')
        ax.set_xlabel('Min triangle angle (°)')
        ax.set_ylabel('||ΔW||_F / ||W_sim||_F')
        ax.set_title(lb)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'W_scatter_per_triangle.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

print('\nSaved plots to breakdown/plots/')
