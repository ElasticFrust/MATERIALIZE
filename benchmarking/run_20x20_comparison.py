"""
20×20 MF vs simulation comparison — DF and TF meshes, 10 trials.

Methods: Std, AW, Std+edge, AW+edge (no angle constraints)
Mesh types: DF (distort-first Delaunay) and TF (triangulate-first)
Grid size: 20×20
Trials: 10 per eta value

Produces side-by-side plots for E/E0 and nu vs eta.
"""
import sys, os, time
import numpy as np
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'Phase 2'))

import Disc_2_Cont_optimized as D2C
import forward_solver_torch as fst

# ── Parameters ───────────────────────────────────────────────────────────────
ETA_VALUES  = np.linspace(0.0, 0.5, 11)
SIZE        = (20, 20)
TRIM_FRAC   = 0.85
N_TRIALS    = 10
EPS_APPLIED = 0.005

SIM_CACHE_DF  = os.path.join(os.path.dirname(__file__), 'sim20_df_data.npz')
SIM_CACHE_TF  = os.path.join(os.path.dirname(__file__), 'sim20_tf_data.npz')
MF_CACHE_DF   = os.path.join(os.path.dirname(__file__), 'mf20_df_data.npz')
MF_CACHE_TF   = os.path.join(os.path.dirname(__file__), 'mf20_tf_data.npz')
OUT           = os.path.join(os.path.dirname(__file__), 'comparison_20x20.png')

CASES = [
    (False, False, 'Std'),
    (True,  False, 'AW'),
    (False, True,  'Std+edge'),
    (True,  True,  'AW+edge'),
]

# ── Mesh builders ─────────────────────────────────────────────────────────────

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


# ── Simulation helpers ────────────────────────────────────────────────────────

def build_unique_edges(simplices):
    edge_set = {}
    for tri in simplices:
        for i, j in [(tri[0], tri[1]), (tri[0], tri[2]), (tri[1], tri[2])]:
            key = (min(i, j), max(i, j))
            edge_set[key] = True
    return np.array(sorted(edge_set.keys()), dtype=np.int64)


def ribbon_sim(DM, eps=EPS_APPLIED):
    pts = DM.points
    simplices = DM.simplices
    active = np.unique(simplices.ravel())
    remap  = {old: new for new, old in enumerate(active)}
    p      = pts[active].copy()
    simps  = np.array([[remap[n] for n in tri] for tri in simplices])
    edges  = build_unique_edges(simps)

    n  = len(p)
    l0 = np.sqrt(((p[edges[:, 0]] - p[edges[:, 1]]) ** 2).sum(1))
    k  = np.ones(len(edges))

    y    = p[:, 1]
    ymax, ymin = y.max(), y.min()
    L    = ymax - ymin
    W    = p[:, 0].max() - p[:, 0].min()

    top_idx = np.where(y > ymax - 1.5)[0]
    bot_idx = np.where(y < ymin + 1.5)[0]

    n_dof      = 2 * n
    fixed_mask = np.zeros(n_dof, dtype=bool)
    fixed_val  = p.ravel().copy()

    delta = eps * L / 2
    for i in top_idx:
        fixed_mask[2 * i + 1] = True
        fixed_val [2 * i + 1] = p[i, 1] + delta
    for i in bot_idx:
        fixed_mask[2 * i + 1] = True
        fixed_val [2 * i + 1] = p[i, 1] - delta

    free_idx  = np.where(~fixed_mask)[0]
    fixed_idx = np.where( fixed_mask)[0]

    x0_all             = p.ravel().copy()
    x0_all[1::2]       = p[:, 1] * (1 + eps)
    x0_all[fixed_idx]  = fixed_val[fixed_idx]
    x0_free = x0_all[free_idx]

    def energy_grad(free_flat):
        pos_flat = fixed_val.copy()
        pos_flat[free_idx] = free_flat
        pos = pos_flat.reshape(n, 2)
        dr  = pos[edges[:, 0]] - pos[edges[:, 1]]
        l   = np.sqrt((dr ** 2).sum(1))
        ext = l - l0
        E   = 0.5 * (k * ext ** 2).sum()
        fac = k * ext / np.maximum(l, 1e-15)
        f   = np.zeros((n, 2))
        np.add.at(f, edges[:, 0],  fac[:, None] * dr)
        np.add.at(f, edges[:, 1], -fac[:, None] * dr)
        return E, f.ravel()[free_idx]

    res = minimize(energy_grad, x0_free, jac=True, method='L-BFGS-B',
                   options={'maxiter': 10000, 'ftol': 1e-15, 'gtol': 1e-10})
    pos_flat = fixed_val.copy()
    pos_flat[free_idx] = res.x
    pos = pos_flat.reshape(n, 2)

    ycen  = (ymax + ymin) / 2
    band  = L / 6
    mid   = (np.abs(p[:, 1] - ycen) < band) & (np.abs(p[:, 0]) > 0.1)
    if mid.sum() < 3:
        return np.nan, np.nan

    x_ref  = p  [mid, 0]
    x_def  = pos[mid, 0]
    eps_xx = np.dot(x_def - x_ref, x_ref) / np.dot(x_ref, x_ref)
    nu     = -eps_xx / eps

    F_y = 0.0
    for i in top_idx:
        neighbors = np.where((edges[:, 0] == i) | (edges[:, 1] == i))[0]
        for e in neighbors:
            a, b  = edges[e]
            dr_e  = pos[a] - pos[b]
            l_e   = np.sqrt((dr_e ** 2).sum())
            ext_e = l_e - l0[e]
            sign  = 1 if a == i else -1
            F_y  += sign * k[e] * ext_e * dr_e[1] / np.maximum(l_e, 1e-15)

    sigma_yy = F_y / W
    E_mod    = sigma_yy / eps
    return E_mod, nu


# ── MF helper ────────────────────────────────────────────────────────────────

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


def run_or_load_sim(cache, builder, label):
    if os.path.exists(cache):
        print(f"  Loading {label} sim cache")
        d = np.load(cache)
        return d['E_sim'], d['nu_sim']
    n_eta  = len(ETA_VALUES)
    E_sim  = np.full((n_eta, N_TRIALS), np.nan)
    nu_sim = np.full((n_eta, N_TRIALS), np.nan)
    t0 = time.time()
    for i_eta, eta in enumerate(ETA_VALUES):
        for trial in range(N_TRIALS):
            np.random.seed(100 * i_eta + trial)
            try:
                DM = builder(eta)
                E_sim[i_eta, trial], nu_sim[i_eta, trial] = ribbon_sim(DM)
            except Exception as exc:
                print(f"    sim {label} eta={eta:.2f} trial={trial}: {exc}")
        elapsed = time.time() - t0
        done = i_eta + 1
        print(f"  {label} sim eta={eta:.2f}  E={np.nanmedian(E_sim[i_eta]):.4f}  "
              f"nu={np.nanmedian(nu_sim[i_eta]):+.4f}  [{elapsed:.0f}s]")
    np.savez(cache, E_sim=E_sim, nu_sim=nu_sim, etas=ETA_VALUES)
    return E_sim, nu_sim


def run_or_load_mf(cache, builder, label):
    if os.path.exists(cache):
        print(f"  Loading {label} MF cache")
        d = np.load(cache)
        return {k: d[k] for k in d if k not in ('etas',)}, d.get('etas', ETA_VALUES)
    n_eta  = len(ETA_VALUES)
    results = {k: np.full((n_eta, N_TRIALS), np.nan) for k in ['E_' + c[2] for c in CASES] +
               ['nu_' + c[2] for c in CASES]}
    t0 = time.time()
    for i_eta, eta in enumerate(ETA_VALUES):
        for trial in range(N_TRIALS):
            np.random.seed(100 * i_eta + trial)
            try:
                DM = builder(eta)
                solver, rigs, rl = fst.from_triangulation(DM)
                with torch.no_grad():
                    for aw, kkt, key in CASES:
                        res = solver.forward(rigs, rl, area_weighted=aw,
                                             use_kkt=kkt, use_angle_kkt=False)
                        C6 = res['elastic_tensor'].numpy()
                        E_v, nu_v = directional_constants(C6)
                        results[f'E_{key}'][i_eta, trial]  = E_v
                        results[f'nu_{key}'][i_eta, trial] = nu_v
            except Exception as exc:
                print(f"    MF {label} eta={eta:.2f} trial={trial}: {exc}")
        elapsed = time.time() - t0
        print(f"  {label} MF  eta={eta:.2f}  "
              f"Std+edge E={np.nanmedian(results['E_Std+edge'][i_eta]):.4f}  "
              f"nu={np.nanmedian(results['nu_Std+edge'][i_eta]):+.4f}  [{elapsed:.0f}s]")
    np.savez(cache, etas=ETA_VALUES, **results)
    return results, ETA_VALUES


# ── Main ──────────────────────────────────────────────────────────────────────

t_start = time.time()

print("=== Simulation: DF ===")
E_sim_df, nu_sim_df = run_or_load_sim(SIM_CACHE_DF, make_df, "DF")

print("=== Simulation: TF ===")
E_sim_tf, nu_sim_tf = run_or_load_sim(SIM_CACHE_TF, make_tf, "TF")

print("=== MF: DF ===")
mf_df, _ = run_or_load_mf(MF_CACHE_DF, make_df, "DF")

print("=== MF: TF ===")
mf_tf, _ = run_or_load_mf(MF_CACHE_TF, make_tf, "TF")

t_total = time.time() - t_start
print(f"\nAll data ready in {t_total:.0f}s")

# ── Normalize by eta=0 value ──────────────────────────────────────────────────
etas = ETA_VALUES

def norm(arr2d):
    """Normalize rows by median of first row (eta=0)."""
    E0 = np.nanmedian(arr2d[0])
    if abs(E0) < 1e-10:
        return arr2d.copy()
    return arr2d / E0

E_sim_df_n = norm(E_sim_df)
E_sim_tf_n = norm(E_sim_tf)

for key in [c[2] for c in CASES]:
    E0 = np.nanmedian(mf_df[f'E_{key}'][0])
    mf_df[f'E_{key}_n'] = mf_df[f'E_{key}'] / max(E0, 1e-10)
    E0 = np.nanmedian(mf_tf[f'E_{key}'][0])
    mf_tf[f'E_{key}_n'] = mf_tf[f'E_{key}'] / max(E0, 1e-10)

# ── Plot ──────────────────────────────────────────────────────────────────────
COLORS = {
    'Std':      '#1f77b4',
    'AW':       '#ff7f0e',
    'Std+edge': '#2ca02c',
    'AW+edge':  '#d62728',
}
LS_MF  = {'Std': '-', 'AW': '--', 'Std+edge': '-', 'AW+edge': '--'}

fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
fig.suptitle('MF vs Simulation — 20×20 mesh, 10 trials', fontsize=14)

titles = ['DF: E/E₀ vs η', 'TF: E/E₀ vs η', 'DF: ν vs η', 'TF: ν vs η']
for ax, title in zip(axes.ravel(), titles):
    ax.set_title(title)
    ax.set_xlabel('η')
    ax.grid(True, alpha=0.3)

def med(arr2d):
    return np.nanmedian(arr2d, axis=1)

def q25(arr2d):
    return np.nanpercentile(arr2d, 25, axis=1)

def q75(arr2d):
    return np.nanpercentile(arr2d, 75, axis=1)


# ── DF E/E0 ──────────────────────────────────────────────────────────────────
ax = axes[0, 0]
ax.fill_between(etas, q25(E_sim_df_n), q75(E_sim_df_n), color='gray', alpha=0.25)
ax.plot(etas, med(E_sim_df_n), 'k-o', lw=2, ms=5, label='Sim (DF)')
for aw, kkt, key in CASES:
    ax.plot(etas, med(mf_df[f'E_{key}_n']), color=COLORS[key],
            ls=LS_MF[key], marker='s', ms=4, label=key)
ax.set_ylabel('E/E₀')
ax.legend(fontsize=8)

# ── TF E/E0 ──────────────────────────────────────────────────────────────────
ax = axes[0, 1]
ax.fill_between(etas, q25(E_sim_tf_n), q75(E_sim_tf_n), color='gray', alpha=0.25)
ax.plot(etas, med(E_sim_tf_n), 'k-o', lw=2, ms=5, label='Sim (TF)')
for aw, kkt, key in CASES:
    ax.plot(etas, med(mf_tf[f'E_{key}_n']), color=COLORS[key],
            ls=LS_MF[key], marker='s', ms=4, label=key)
ax.set_ylabel('E/E₀')
ax.legend(fontsize=8)

# ── DF ν ─────────────────────────────────────────────────────────────────────
ax = axes[1, 0]
ax.fill_between(etas, q25(nu_sim_df), q75(nu_sim_df), color='gray', alpha=0.25)
ax.plot(etas, med(nu_sim_df), 'k-o', lw=2, ms=5, label='Sim (DF)')
for aw, kkt, key in CASES:
    ax.plot(etas, med(mf_df[f'nu_{key}']), color=COLORS[key],
            ls=LS_MF[key], marker='s', ms=4, label=key)
ax.set_ylabel('ν')
ax.axhline(0, color='gray', lw=0.5, ls=':')
ax.legend(fontsize=8)

# ── TF ν ─────────────────────────────────────────────────────────────────────
ax = axes[1, 1]
ax.fill_between(etas, q25(nu_sim_tf), q75(nu_sim_tf), color='gray', alpha=0.25)
ax.plot(etas, med(nu_sim_tf), 'k-o', lw=2, ms=5, label='Sim (TF)')
for aw, kkt, key in CASES:
    ax.plot(etas, med(mf_tf[f'nu_{key}']), color=COLORS[key],
            ls=LS_MF[key], marker='s', ms=4, label=key)
ax.set_ylabel('ν')
ax.axhline(0, color='gray', lw=0.5, ls=':')
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(OUT, dpi=150, bbox_inches='tight')
print(f"Saved: {OUT}")
print(f"Total time: {time.time()-t_start:.0f}s")
