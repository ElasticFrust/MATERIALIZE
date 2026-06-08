"""
True before/after: original _woodbury_kkt_sparse (pre-refactor) vs
new _woodbury_kkt_sparse_combined (post-fix).

The buggy intermediate state (from the broken unified routing) is also shown
for completeness but is clearly labeled as a temporary regression.
"""
import sys, os, time
import numpy as np
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

CACHE_OLD = os.path.join(os.path.dirname(__file__), 'mf20_old_kkt_data.npz')
OUT       = os.path.join(os.path.dirname(__file__), 'new', 'before_after_fix.png')


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


def run_old_kkt(cache):
    """Run using the original _woodbury_kkt_sparse (pre-unification) directly."""
    if os.path.exists(cache):
        print("  Loading old KKT cache")
        d = np.load(cache)
        return {k: d[k] for k in d if k != 'etas'}

    n_eta = len(ETA_VALUES)
    res   = {f'{q}_{m}': np.full((n_eta, N_TRIALS), np.nan)
             for q in ('E', 'nu') for m in ('df', 'tf')}
    t0 = time.time()

    for i_eta, eta in enumerate(ETA_VALUES):
        for trial in range(N_TRIALS):
            np.random.seed(100 * i_eta + trial)
            for key, builder in [('df', make_df), ('tf', make_tf)]:
                try:
                    DM = builder(eta)
                    solver, rigs, rl = fst.from_triangulation(DM)

                    # Call the OLD function directly, bypassing forward() routing
                    vx = solver.edge_vecs[:, :, 0]
                    vy = solver.edge_vecs[:, :, 1]
                    length2 = solver.actual_length2
                    factor  = rigs / length2 / 16.0
                    bare = torch.stack([
                        (factor * vx**4).sum(1),
                        (factor * vx**3 * vy).sum(1),
                        (factor * vx**2 * vy**2).sum(1),
                        (factor * vx * vy**3).sum(1),
                        (factor * vy**4).sum(1),
                    ], dim=1)
                    mean_tensor = bare.mean(0)
                    delta    = bare - mean_tensor
                    A_blocks = fst._batch_to_9x9(bare)
                    B_blocks = fst._batch_to_9x9(delta)
                    dA_vecs  = fst._batch_to_9vec(delta)

                    # Use the original function directly
                    W_np = fst._woodbury_kkt_sparse(
                        A_blocks, B_blocks, dA_vecs, solver.kkt_arrays)
                    W    = torch.as_tensor(W_np, dtype=bare.dtype)

                    actual = fst._compute_actual_elastic_tensor(bare, W)
                    C      = actual.mean(0).numpy()
                    E_v, nu_v = ec(C)
                    res[f'E_{key}'][i_eta, trial]  = E_v
                    res[f'nu_{key}'][i_eta, trial] = nu_v

                except Exception as exc:
                    print(f"    old KKT {key} eta={eta:.2f} t={trial}: {exc}")

        elapsed = time.time() - t0
        print(f"  old KKT eta={eta:.2f}  "
              f"df E={np.nanmedian(res['E_df'][i_eta]):.4f} "
              f"nu={np.nanmedian(res['nu_df'][i_eta]):+.4f}  "
              f"tf E={np.nanmedian(res['E_tf'][i_eta]):.4f} "
              f"nu={np.nanmedian(res['nu_tf'][i_eta]):+.4f}  [{elapsed:.0f}s]")
    np.savez(cache, etas=ETA_VALUES, **res)
    return res


# ── Main ──────────────────────────────────────────────────────────────────────

t_start = time.time()
print("=== Original _woodbury_kkt_sparse (pre-refactor) ===")
old = run_old_kkt(CACHE_OLD)

print("Loading fixed + buggy + sim data...")
d_df     = np.load(os.path.join(os.path.dirname(__file__), 'mf20_df_data.npz'))
d_tf     = np.load(os.path.join(os.path.dirname(__file__), 'mf20_tf_data.npz'))
d_sim_df = np.load(os.path.join(os.path.dirname(__file__), 'sim20_df_data.npz'))
d_sim_tf = np.load(os.path.join(os.path.dirname(__file__), 'sim20_tf_data.npz'))
d_buggy  = np.load(os.path.join(os.path.dirname(__file__), 'mf20_buggy_stdedge.npz'))
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
fig.suptitle('Before vs After precision fix — Std+edge, 20×20 mesh, 10 trials',
             fontsize=13)

panel_cfg = [
    (axes[0, 0], 'DF: E/E₀ vs η',
     norm(d_sim_df['E_sim']),
     norm(d_df['E_Std+edge']),
     norm(old['E_df']),
     norm(d_buggy['E_df']),
     'E/E₀'),
    (axes[0, 1], 'TF: E/E₀ vs η',
     norm(d_sim_tf['E_sim']),
     norm(d_tf['E_Std+edge']),
     norm(old['E_tf']),
     norm(d_buggy['E_tf']),
     'E/E₀'),
    (axes[1, 0], 'DF: ν vs η',
     d_sim_df['nu_sim'],
     d_df['nu_Std+edge'],
     old['nu_df'],
     d_buggy['nu_df'],
     'ν'),
    (axes[1, 1], 'TF: ν vs η',
     d_sim_tf['nu_sim'],
     d_tf['nu_Std+edge'],
     old['nu_tf'],
     d_buggy['nu_tf'],
     'ν'),
]

for ax, title, sim_arr, new_arr, orig_arr, buggy_arr, ylabel in panel_cfg:
    ax.set_title(title, fontsize=11)
    ax.set_xlabel('η'); ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)

    ax.fill_between(etas, q25(sim_arr), q75(sim_arr), color='gray', alpha=0.2)
    ax.plot(etas, med(sim_arr), 'k-o', lw=2, ms=5, label='Simulation')

    ax.plot(etas, med(orig_arr), color='#1f77b4', ls='--', marker='o', ms=4,
            lw=1.8, label='Before refactor (original)')
    ax.plot(etas, med(new_arr),  color='#2ca02c', ls='-',  marker='s', ms=4,
            lw=1.8, label='After fix (new unified)')
    ax.plot(etas, med(buggy_arr), color='#d62728', ls=':', marker='^', ms=3,
            lw=1.2, alpha=0.6, label='Buggy interim (regression)')

    if ylabel == 'ν':
        ax.axhline(0, color='gray', lw=0.5, ls=':')

    ax.legend(fontsize=8)

for ax in [axes[0, 0], axes[0, 1]]:
    ax.set_ylim(-0.05, 1.6)

plt.tight_layout()
plt.savefig(OUT, dpi=150, bbox_inches='tight')
print(f"Saved: {OUT}")
