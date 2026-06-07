"""
all_models_comp — four solver variants, full directional elastic properties.

For each of 4 solver variants:
  1. Regular MF          — uniform mean, no KKT
  2. Area-weighted MF    — area-weighted mean, no KKT
  3. Regular MF + KKT    — uniform mean, with KKT correction
  4. Area-wtd MF + KKT   — area-weighted mean, with KKT correction

Extracts directional engineering constants from the compliance tensor S = C_V^{-1}:

  C_V (Voigt 3×3, e = [ε_11, ε_22, 2ε_12]):
      [[C_1111, C_1122, C_1112],
       [C_1122, C_2222, C_2122],
       [C_1112, C_2122, C_1212]]

  x-loading (σ_22 = σ_12 = 0):  E_x = 1/S[0,0],  ν_xy = -S[1,0]/S[0,0]
  y-loading (σ_11 = σ_12 = 0):  E_y = 1/S[1,1],  ν_yx = -S[0,1]/S[1,1]

Isotropy ↔ E_x ≈ E_y  and  ν_xy ≈ ν_yx.

Produces: all_models_comp_nu.png  (Poisson ratio, x and y loading)
          all_models_comp_E.png   (Young's modulus, x and y loading)

20×20 network (~3726 triangles), 10 trials per η value.
"""
import sys, os, time
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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


def _triangle_areas(solver):
    pos = solver.positions.detach().numpy()
    sim = solver.simplices.detach().numpy()
    v0, v1, v2 = pos[sim[:, 0]], pos[sim[:, 1]], pos[sim[:, 2]]
    return 0.5 * np.abs(
        (v1[:, 0]-v0[:, 0])*(v2[:, 1]-v0[:, 1]) -
        (v1[:, 1]-v0[:, 1])*(v2[:, 0]-v0[:, 0])
    )


def _directional_constants(C6):
    """
    C6: length-6 array [C_1111, C_1112, C_1122, C_2112, C_2122, C_2222]
    Returns E_x, E_y, nu_xy, nu_yx from the 3×3 Voigt compliance.
    """
    C0, C1, C2, C3, C4, C5 = C6
    # Voigt matrix:  σ = C_V @ [ε11, ε22, 2ε12]
    C_V = np.array([[C0, C2, C1],
                    [C2, C5, C4],
                    [C1, C4, C3]])
    try:
        S = np.linalg.inv(C_V)
    except np.linalg.LinAlgError:
        return np.nan, np.nan, np.nan, np.nan
    E_x   =  1.0 / S[0, 0] if abs(S[0, 0]) > 1e-15 else np.nan
    E_y   =  1.0 / S[1, 1] if abs(S[1, 1]) > 1e-15 else np.nan
    nu_xy = -S[1, 0] / S[0, 0] if abs(S[0, 0]) > 1e-15 else np.nan
    nu_yx = -S[0, 1] / S[1, 1] if abs(S[1, 1]) > 1e-15 else np.nan
    return E_x, E_y, nu_xy, nu_yx


def run_all_cases(solver, rigidities, rest_lengths):
    """Run all 4 variants; return list of (E_x, E_y, nu_xy, nu_yx)."""
    vx = solver.edge_vecs[:, :, 0]
    vy = solver.edge_vecs[:, :, 1]
    factor = rigidities / rest_lengths**2 / 16.0
    bare = torch.stack([
        (factor * vx**4).sum(1),
        (factor * vx**3 * vy).sum(1),
        (factor * vx**2 * vy**2).sum(1),
        (factor * vx * vy**3).sum(1),
        (factor * vy**4).sum(1),
    ], dim=1)

    areas = torch.tensor(_triangle_areas(solver), dtype=torch.float64)
    w = (areas / areas.sum()).unsqueeze(1)

    results = []
    for area_weighted, use_kkt, *_ in CASES:
        mean_t = (bare * w).sum(0) if area_weighted else bare.mean(0)
        delta  = bare - mean_t
        A_b = _batch_to_9x9(bare)
        B_b = _batch_to_9x9(delta)
        dA  = _batch_to_9vec(delta)

        with torch.no_grad():
            if use_kkt and solver.kkt_arrays is not None:
                W_np = _woodbury_kkt_sparse(A_b, B_b, dA, solver.kkt_arrays)
                W = torch.as_tensor(W_np, dtype=bare.dtype)
            elif use_kkt and solver.J is not None:
                W = _woodbury_solve(A_b, B_b, dA, J=solver.J.to(dtype=bare.dtype))
            else:
                W = _woodbury_solve(A_b, B_b, dA, J=None)

        C6 = _compute_actual_elastic_tensor(bare, W).mean(0).numpy()
        results.append(_directional_constants(C6))
    return results   # list of (E_x, E_y, nu_xy, nu_yx) per case


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


# ── Plotting helpers ──────────────────────────────────────────────────────────
ETA_STABLE = 0.35
jitter_off = np.linspace(-0.009, 0.009, n_cases)


def _plot_panel(ax, arr_clip, ylabel, case_metas, etas, direction_label):
    for c, (_, _, label, color, marker, ls) in enumerate(case_metas):
        data = arr_clip[c]                       # (n_eta, N_TRIALS)
        med = np.nanmedian(data, axis=1)
        q1  = np.nanpercentile(data, 25, axis=1)
        q3  = np.nanpercentile(data, 75, axis=1)
        ax.errorbar(
            etas + jitter_off[c], med,
            yerr=[np.clip(med - q1, 0, None), np.clip(q3 - med, 0, None)],
            fmt=f'{marker}{ls}', color=color, capsize=3,
            markersize=5, label=label, alpha=0.9,
        )
        ax.fill_between(etas, q1, q3, alpha=0.08, color=color)
    ax.axvline(ETA_STABLE, color='gray', lw=1.2, ls=':', alpha=0.6)
    ax.text(ETA_STABLE + 0.004, ax.get_ylim()[0] if ax.get_ylim()[0] != 0 else 0.01,
            'unstable', color='gray', fontsize=8, rotation=90, va='bottom')
    ax.set_xlabel(r'$\eta$', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f'Loading direction: {direction_label}', fontsize=11)
    ax.legend(fontsize=9, loc='best')
    ax.grid(alpha=0.3)


# ── Figure 1: Poisson ratio ───────────────────────────────────────────────────
NU_LO, NU_HI = -1.0, 0.45
nuxy_clip = np.clip(nuxy_arr, NU_LO, NU_HI)
nuyx_clip = np.clip(nuyx_arr, NU_LO, NU_HI)

fig1, axes1 = plt.subplots(1, 2, figsize=(15, 5), sharey=True)

_plot_panel(axes1[0], nuxy_clip, r"Poisson's ratio $\nu$", CASES, etas, r'$x$ (σ₂₂=0, σ₁₂=0)')
_plot_panel(axes1[1], nuyx_clip, r"Poisson's ratio $\nu$", CASES, etas, r'$y$ (σ₁₁=0, σ₁₂=0)')

for ax in axes1:
    ax.set_ylim(NU_LO, NU_HI)

fig1.suptitle(
    rf"Poisson's ratio $\nu(\eta)$ — {SIZE[0]}×{SIZE[1]} network, {N_TRIALS} trials  "
    r"[isotropy: left ≈ right]",
    fontsize=13, y=1.02,
)
fig1.tight_layout()
path1 = os.path.join(os.path.dirname(__file__), 'all_models_comp_nu.png')
fig1.savefig(path1, dpi=150, bbox_inches='tight')
print(f"Saved: {path1}")

# ── Figure 2: Young's modulus ─────────────────────────────────────────────────
# Clip to 2–98th percentile of valid values (robust to singular C matrices)
valid_E = np.concatenate([Ex_arr[np.isfinite(Ex_arr)], Ey_arr[np.isfinite(Ey_arr)]])
E_lo = max(0.0, float(np.percentile(valid_E, 2)))
E_hi = float(np.percentile(valid_E, 98)) * 1.1
Ex_clip = np.clip(Ex_arr, E_lo, E_hi)
Ey_clip = np.clip(Ey_arr, E_lo, E_hi)

fig2, axes2 = plt.subplots(1, 2, figsize=(15, 5), sharey=True)

_plot_panel(axes2[0], Ex_clip, r"Young's modulus $E$", CASES, etas, r'$x$ (σ₂₂=0, σ₁₂=0)')
_plot_panel(axes2[1], Ey_clip, r"Young's modulus $E$", CASES, etas, r'$y$ (σ₁₁=0, σ₁₂=0)')

for ax in axes2:
    ax.set_ylim(E_lo, E_hi)

fig2.suptitle(
    rf"Young's modulus $E(\eta)$ — {SIZE[0]}×{SIZE[1]} network, {N_TRIALS} trials  "
    r"[isotropy: left ≈ right]",
    fontsize=13, y=1.02,
)
fig2.tight_layout()
path2 = os.path.join(os.path.dirname(__file__), 'all_models_comp_E.png')
fig2.savefig(path2, dpi=150, bbox_inches='tight')
print(f"Saved: {path2}")

# ── Isotropy summary ──────────────────────────────────────────────────────────
print(f"\n{'Isotropy check — mean |ν_xy − ν_yx| and |E_x − E_y| / E_x':}")
print(f"{'η':>5}", end='')
for _, _, label, *_ in CASES:
    print(f"  {label:>25}", end='')
print()
print('-' * (6 + 27 * n_cases))
for i, eta in enumerate(etas):
    print(f"  {eta:.2f}", end='')
    for c in range(n_cases):
        dnu = np.nanmean(np.abs(nuxy_arr[c,i] - nuyx_arr[c,i]))
        dE  = np.nanmean(np.abs(Ex_arr[c,i] - Ey_arr[c,i]) /
                         (np.abs(Ex_arr[c,i]) + 1e-12))
        print(f"  |Δν|={dnu:.4f} |ΔE/E|={dE:.4f}", end='')
    print()
