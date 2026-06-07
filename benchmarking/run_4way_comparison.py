"""
Four-way comparison of forward solver variants: ν and E vs η.

Cases:
  1. Regular MF          — uniform mean, no KKT
  2. Area-weighted MF    — area-weighted mean, no KKT
  3. Regular MF + KKT    — uniform mean, with KKT edge-compatibility correction
  4. Area-wtd MF + KKT   — area-weighted mean, with KKT correction

Area-weighted MF uses triangle areas as importance weights for the reference medium
(Ā = Σ_n (a_n/A_tot) A(n)), replacing the arithmetic mean. Larger triangles
contribute proportionally more to the effective background stiffness.

20×20 network (~3726 triangles, ~2270 vertices), 10 trials per η.
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

CACHE = os.path.join(os.path.dirname(__file__), 'poisson_young_4way_data.npz')
print(f"size={SIZE}  n_trials={N_TRIALS}  n_eta={len(ETA_VALUES)}")
print(f"eta: {np.round(ETA_VALUES, 3).tolist()}\n")

CASES = [
    # (area_weighted, use_kkt, label, color, marker, linestyle)
    (False, False, 'Regular MF',        'C0', 'o',  '-'),
    (True,  False, 'Area-weighted MF',  'C1', 's',  '--'),
    (False, True,  'Regular MF + KKT',  'C2', '^',  '-'),
    (True,  True,  'Area-wtd MF + KKT', 'C3', 'D',  '--'),
]


def _compute_triangle_areas(solver):
    pos = solver.positions.detach().numpy()
    sim = solver.simplices.detach().numpy()
    v0, v1, v2 = pos[sim[:, 0]], pos[sim[:, 1]], pos[sim[:, 2]]
    areas = 0.5 * np.abs(
        (v1[:, 0] - v0[:, 0]) * (v2[:, 1] - v0[:, 1]) -
        (v1[:, 1] - v0[:, 1]) * (v2[:, 0] - v0[:, 0])
    )
    return torch.tensor(areas, dtype=torch.float64)


def _elastic_scalars(C):
    poisson = ((C[2]*C[3] - C[1]*C[4]) / (C[0]*C[3] - C[1]**2)).item()
    young   = (
        (C[2]**2*C[3] - 2*C[1]*C[2]*C[4] + C[1]**2*C[5]
         + C[0]*(C[4]**2 - C[3]*C[5]))
        / (C[1]**2 - C[0]*C[3])
    ).item()
    return poisson, young


def run_all_cases(solver, rigidities, rest_lengths):
    """Run all 4 variants on one network; return list of (poisson, young)."""
    vx = solver.edge_vecs[:, :, 0]
    vy = solver.edge_vecs[:, :, 1]
    length2 = rest_lengths ** 2
    factor  = rigidities / length2 / 16.0

    bare = torch.stack([
        (factor * vx**4).sum(1),
        (factor * vx**3 * vy).sum(1),
        (factor * vx**2 * vy**2).sum(1),
        (factor * vx * vy**3).sum(1),
        (factor * vy**4).sum(1),
    ], dim=1)  # (N, 5)

    areas = _compute_triangle_areas(solver)
    w = (areas / areas.sum()).unsqueeze(1)  # (N, 1)

    outputs = []
    for area_weighted, use_kkt, *_ in CASES:
        mean_t = (bare * w).sum(0) if area_weighted else bare.mean(0)
        delta   = bare - mean_t
        A_b = _batch_to_9x9(bare)
        B_b = _batch_to_9x9(delta)
        dA  = _batch_to_9vec(delta)

        with torch.no_grad():
            if use_kkt and solver.kkt_arrays is not None:
                W_np = _woodbury_kkt_sparse(A_b, B_b, dA, solver.kkt_arrays)
                W = torch.as_tensor(W_np, dtype=bare.dtype)
            elif use_kkt and solver.J is not None:
                W = _woodbury_solve(A_b, B_b, dA,
                                    J=solver.J.to(dtype=bare.dtype))
            else:
                W = _woodbury_solve(A_b, B_b, dA, J=None)

        C = _compute_actual_elastic_tensor(bare, W).mean(0)
        outputs.append(_elastic_scalars(C))
    return outputs  # list of (poisson, young) for each case


# ── Compute or load ────────────────────────────────────────────────────────────
if os.path.exists(CACHE):
    print(f"Loading cached data from {CACHE}")
    d = np.load(CACHE)
    nu_arr = d['nu_arr']
    E_arr  = d['E_arr']
    etas   = d['etas']
else:
    n_cases = len(CASES)
    nu_arr  = np.full((n_cases, len(ETA_VALUES), N_TRIALS), np.nan)
    E_arr   = np.full((n_cases, len(ETA_VALUES), N_TRIALS), np.nan)
    t0      = time.time()

    for i_eta, eta in enumerate(ETA_VALUES):
        for trial in range(N_TRIALS):
            seed = 100 * i_eta + trial
            np.random.seed(seed)

            DT = D2C.generate_foam_points(SIZE, eta)
            solver, rigs, rl = fst.from_triangulation(DT)

            results = run_all_cases(solver, rigs, rl)

            for c_idx, (nu, E) in enumerate(results):
                nu_arr[c_idx, i_eta, trial] = nu
                E_arr[c_idx,  i_eta, trial] = E

            elapsed = time.time() - t0
            row = "  ".join(f"{results[c][0]:+.4f}" for c in range(n_cases))
            print(f"  η={eta:.2f}  trial={trial:2d}  ν=[{row}]  t={elapsed:.0f}s",
                  flush=True)
        print()

    etas = np.asarray(ETA_VALUES)
    np.savez(CACHE, nu_arr=nu_arr, E_arr=E_arr, etas=etas)
    print(f"Data saved to {CACHE}")

# ── Plot ───────────────────────────────────────────────────────────────────────
YMIN_NU, YMAX_NU = -1.0, 0.45
ETA_STABLE = 0.35

nu_clip = np.clip(nu_arr, YMIN_NU, YMAX_NU)

# Clip E to the 5–95th percentile range (robust to outliers)
E_lo = max(0, float(np.nanpercentile(E_arr, 2)))
E_hi = float(np.nanpercentile(E_arr, 98)) * 1.1
E_clip = np.clip(E_arr, E_lo, E_hi)

fig, axes = plt.subplots(2, 2, figsize=(16, 10))

jitter_offsets = np.linspace(-0.009, 0.009, len(CASES))

for row, (arr, clip_arr, ylabel, ylo, yhi) in enumerate([
    (nu_arr, nu_clip, r"Poisson's ratio $\nu$", YMIN_NU, YMAX_NU),
    (E_arr,  E_clip,  r"Young's modulus $E$",   E_lo,    E_hi),
]):
    for col in range(2):
        ax = axes[row, col]

        for c_idx, (_, _, label, color, marker, ls) in enumerate(CASES):
            jit = jitter_offsets[c_idx]
            data = clip_arr[c_idx]          # (11, 10)

            if col == 0:                    # raw scatter
                for i_eta, eta in enumerate(etas):
                    ax.scatter(
                        np.full(N_TRIALS, eta) + jit, data[i_eta],
                        marker=marker, facecolors='none', edgecolors=color,
                        s=22, linewidths=1.1, alpha=0.75,
                        label=label if i_eta == 0 else None,
                    )
            else:                           # median ± IQR
                med = np.median(data, axis=1)
                q1  = np.percentile(data, 25, axis=1)
                q3  = np.percentile(data, 75, axis=1)
                ax.errorbar(
                    etas + jit, med, yerr=[med - q1, q3 - med],
                    fmt=f'{marker}{ls}', color=color, capsize=3,
                    markersize=5, label=label, alpha=0.9,
                )
                ax.fill_between(etas, q1, q3, alpha=0.08, color=color)

        ax.axvline(ETA_STABLE, color='gray', lw=1.2, ls=':', alpha=0.6)
        ax.set_xlabel(r'$\eta$', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_ylim(ylo, yhi)
        ax.legend(fontsize=9, loc='best')
        ax.grid(alpha=0.3)

        if col == 0:
            clip_str = f' (clipped [{ylo:.2g},{yhi:.2g}])' if row == 0 else f' (clipped [{ylo:.2g},{yhi:.2g}])'
            ax.set_title(f'Raw scatter — {N_TRIALS} trials per η{clip_str}', fontsize=11)
        else:
            ax.set_title('Median ± IQR (robust)', fontsize=11)

fig.suptitle(
    rf"$\nu$ and $E$ vs disorder $\eta$ — "
    rf"4 solver variants, {SIZE[0]}×{SIZE[1]} network, {N_TRIALS} trials",
    fontsize=14, y=1.01,
)
fig.tight_layout()

out_path = os.path.join(os.path.dirname(__file__), 'poisson_young_4way_new.png')
fig.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\nSaved: {out_path}")

# ── Summary table ──────────────────────────────────────────────────────────────
print(f"\n{'η':>5}", end='')
for _, _, label, *_ in CASES:
    print(f"  {label:>22}", end='')
print()
print('-' * (6 + 24 * len(CASES)))
for i_eta, eta in enumerate(etas):
    print(f"  {eta:.2f}", end='')
    for c_idx in range(len(CASES)):
        nu_m = np.nanmean(nu_arr[c_idx, i_eta])
        E_m  = np.nanmean(E_arr[c_idx, i_eta])
        print(f"  ν={nu_m:+.4f} E={E_m:.4f}", end='')
    print()
