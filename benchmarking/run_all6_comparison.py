"""
All-correct methods comparison: Std, AW, Std+edge, AW+edge, Std+full, AW+full
vs simulation — DF and TF meshes, 20×20 grid, 10 trials.

Loads existing sim + 4-method MF caches; runs only the two "full" (edge+angle)
variants if their cache is missing.
Output: benchmarking/new/all6_comparison.png
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

SIM_CACHE_DF   = os.path.join(os.path.dirname(__file__), 'sim20_df_data.npz')
SIM_CACHE_TF   = os.path.join(os.path.dirname(__file__), 'sim20_tf_data.npz')
MF4_CACHE_DF   = os.path.join(os.path.dirname(__file__), 'mf20_df_data.npz')
MF4_CACHE_TF   = os.path.join(os.path.dirname(__file__), 'mf20_tf_data.npz')
FULL_CACHE_DF  = os.path.join(os.path.dirname(__file__), 'mf20_full_df_data.npz')
FULL_CACHE_TF  = os.path.join(os.path.dirname(__file__), 'mf20_full_tf_data.npz')
OUT            = os.path.join(os.path.dirname(__file__), 'new', 'all6_comparison.png')

FULL_CASES = [
    (False, 'Std+full'),
    (True,  'AW+full'),
]


def make_df(eta):
    return D2C.generate_foam_distort_first(SIZE, eta, trim_frac=TRIM_FRAC)


def make_tf(eta):
    DM = D2C.generate_foam_points(SIZE, eta)
    centroids = np.mean(DM.points[DM.simplices], axis=1)
    cx, cy = centroids[:, 0], centroids[:, 1]
    xc = (cx.max() + cx.min()) / 2
    yc = (cy.max() + cy.min()) / 2
    hw = (cx.max() - cx.min()) / 2
    hh = (cy.max() - cy.min()) / 2
    mask = (np.abs(cx - xc) <= TRIM_FRAC * hw) & (np.abs(cy - yc) <= TRIM_FRAC * hh)
    DM.simplices = DM.simplices[mask]
    return DM


def directional_constants(C6):
    C0, C1, C2, C3, C4, C5 = C6
    C_V = np.array([[C0, C2, C1], [C2, C5, C4], [C1, C4, C3]])
    try:
        S = np.linalg.inv(C_V)
    except Exception:
        return np.nan, np.nan
    Ex  = 1 / S[0, 0]
    Ey  = 1 / S[1, 1]
    nuxy = -S[1, 0] * Ex
    nuyx = -S[0, 1] * Ey
    return 0.5 * (Ex + Ey), 0.5 * (nuxy + nuyx)


def run_or_load_full(cache, builder, label):
    if os.path.exists(cache):
        print(f"  Loading {label} full-KKT cache")
        d = np.load(cache)
        return {k: d[k] for k in d if k != 'etas'}

    n_eta   = len(ETA_VALUES)
    results = {f'{q}_{k}': np.full((n_eta, N_TRIALS), np.nan)
               for q in ('E', 'nu') for _, k in FULL_CASES}
    t0 = time.time()

    for i_eta, eta in enumerate(ETA_VALUES):
        for trial in range(N_TRIALS):
            np.random.seed(100 * i_eta + trial)
            try:
                DM = builder(eta)
                solver, rigs, rl = fst.from_triangulation(DM)
                with torch.no_grad():
                    for aw, key in FULL_CASES:
                        res = solver.forward(rigs, rl, area_weighted=aw,
                                             use_kkt=True, use_angle_kkt=True)
                        C6 = res['elastic_tensor'].numpy()
                        E_v, nu_v = directional_constants(C6)
                        results[f'E_{key}'][i_eta, trial]  = E_v
                        results[f'nu_{key}'][i_eta, trial] = nu_v
            except Exception as exc:
                print(f"    full {label} eta={eta:.2f} trial={trial}: {exc}")

        elapsed = time.time() - t0
        print(f"  {label} full  eta={eta:.2f}  "
              f"Std+full E={np.nanmedian(results['E_Std+full'][i_eta]):.4f}  "
              f"nu={np.nanmedian(results['nu_Std+full'][i_eta]):+.4f}  "
              f"AW+full E={np.nanmedian(results['E_AW+full'][i_eta]):.4f}  "
              f"nu={np.nanmedian(results['nu_AW+full'][i_eta]):+.4f}  [{elapsed:.0f}s]")

    np.savez(cache, etas=ETA_VALUES, **results)
    return results


# ── Load all data ─────────────────────────────────────────────────────────────

t_start = time.time()

print("Loading sim data...")
d_sim_df = np.load(SIM_CACHE_DF)
d_sim_tf = np.load(SIM_CACHE_TF)
E_sim_df, nu_sim_df = d_sim_df['E_sim'], d_sim_df['nu_sim']
E_sim_tf, nu_sim_tf = d_sim_tf['E_sim'], d_sim_tf['nu_sim']

print("Loading 4-method MF data...")
d_df4 = np.load(MF4_CACHE_DF)
d_tf4 = np.load(MF4_CACHE_TF)

print("=== Full-KKT: DF ===")
full_df = run_or_load_full(FULL_CACHE_DF, make_df, "DF")

print("=== Full-KKT: TF ===")
full_tf = run_or_load_full(FULL_CACHE_TF, make_tf, "TF")

print(f"\nAll data ready in {time.time()-t_start:.0f}s")

# ── Merge into unified dicts ──────────────────────────────────────────────────

ALL_KEYS = ['Std', 'AW', 'Std+edge', 'AW+edge', 'Std+full', 'AW+full']

mf_df = {}
mf_tf = {}
for key in ALL_KEYS:
    if key in ('Std+full', 'AW+full'):
        mf_df[f'E_{key}']  = full_df[f'E_{key}']
        mf_df[f'nu_{key}'] = full_df[f'nu_{key}']
        mf_tf[f'E_{key}']  = full_tf[f'E_{key}']
        mf_tf[f'nu_{key}'] = full_tf[f'nu_{key}']
    else:
        mf_df[f'E_{key}']  = d_df4[f'E_{key}']
        mf_df[f'nu_{key}'] = d_df4[f'nu_{key}']
        mf_tf[f'E_{key}']  = d_tf4[f'E_{key}']
        mf_tf[f'nu_{key}'] = d_tf4[f'nu_{key}']

# ── Normalize E by each method's own η=0 value ───────────────────────────────
etas = ETA_VALUES

def med(a):  return np.nanmedian(a, axis=1)
def q25(a):  return np.nanpercentile(a, 25, axis=1)
def q75(a):  return np.nanpercentile(a, 75, axis=1)

def norm_by_eta0(arr2d):
    E0 = np.nanmedian(arr2d[0])
    return arr2d / max(abs(E0), 1e-10)

E_sim_df_n = norm_by_eta0(E_sim_df)
E_sim_tf_n = norm_by_eta0(E_sim_tf)
for key in ALL_KEYS:
    mf_df[f'E_{key}_n'] = norm_by_eta0(mf_df[f'E_{key}'])
    mf_tf[f'E_{key}_n'] = norm_by_eta0(mf_tf[f'E_{key}'])

# ── Plot ──────────────────────────────────────────────────────────────────────
COLORS = {
    'Std':      '#1f77b4',
    'AW':       '#ff7f0e',
    'Std+edge': '#2ca02c',
    'AW+edge':  '#d62728',
    'Std+full': '#9467bd',
    'AW+full':  '#8c564b',
}
MARKERS = {
    'Std': 'o', 'AW': 'o',
    'Std+edge': 's', 'AW+edge': 's',
    'Std+full': '^', 'AW+full': '^',
}
LS = {
    'Std': '-',  'AW': '--',
    'Std+edge': '-', 'AW+edge': '--',
    'Std+full': '-', 'AW+full': '--',
}

fig, axes = plt.subplots(2, 2, figsize=(16, 11), sharex=True)
fig.suptitle('All methods vs simulation — 20×20 mesh, 10 trials\n'
             '(Std/AW: mean-field only; +edge: edge-compat KKT; +full: edge+angle KKT)',
             fontsize=12)

def plot_panel(ax, ylabel, sim_arr, mf_dict, metric, normalized=False):
    ax.fill_between(etas, q25(sim_arr), q75(sim_arr), color='gray', alpha=0.2)
    ax.plot(etas, med(sim_arr), 'k-o', lw=2.5, ms=6, label='Simulation', zorder=10)
    for key in ALL_KEYS:
        dict_key = f'{metric}_{key}_n' if normalized else f'{metric}_{key}'
        arr = mf_dict[dict_key]
        ax.plot(etas, med(arr), color=COLORS[key], ls=LS[key],
                marker=MARKERS[key], ms=4, lw=1.6, label=key)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    if metric == 'nu':
        ax.axhline(0, color='gray', lw=0.5, ls=':')
    ax.legend(fontsize=7.5, ncol=2)

axes[0, 0].set_title('DF: E/E₀ vs η', fontsize=11)
plot_panel(axes[0, 0], 'E/E₀', E_sim_df_n, mf_df, 'E', normalized=True)

axes[0, 1].set_title('TF: E/E₀ vs η', fontsize=11)
plot_panel(axes[0, 1], 'E/E₀', E_sim_tf_n, mf_tf, 'E', normalized=True)

axes[1, 0].set_title('DF: ν vs η', fontsize=11)
plot_panel(axes[1, 0], 'ν', nu_sim_df, mf_df, 'nu')

axes[1, 1].set_title('TF: ν vs η', fontsize=11)
plot_panel(axes[1, 1], 'ν', nu_sim_tf, mf_tf, 'nu')

for ax in axes[1]:
    ax.set_xlabel('η')

plt.tight_layout()
plt.savefig(OUT, dpi=150, bbox_inches='tight')
print(f"Saved: {OUT}")
print(f"Total time: {time.time()-t_start:.0f}s")
