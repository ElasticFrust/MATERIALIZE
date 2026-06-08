#!/usr/bin/env python
"""Anisotropy analysis: compute nu_xy, nu_yx, E_x, E_y for all sweep cases."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'Phase 2'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'Phase 3'))
os.environ['MPLBACKEND'] = 'Agg'

import numpy as np
import torch
import matplotlib.pyplot as plt
import Disc_2_Cont_optimized as D2C
from forward_solver_torch import from_triangulation
from inverse_optimize import run_property_optimization

OUT = os.path.dirname(os.path.abspath(__file__))


def full_anisotropic_properties(C):
    """Extract full anisotropic properties from 6-component elastic tensor.

    C = [C_xxxx, C_xxxy, C_xxyy, C_xyxy, C_xyyy, C_yyyy]

    Voigt matrix:
        | C_xxxx  C_xxyy  C_xxxy |     | C[0]  C[2]  C[1] |
        | C_xxyy  C_yyyy  C_xyyy |  =  | C[2]  C[5]  C[4] |
        | C_xxxy  C_xyyy  C_xyxy |     | C[1]  C[4]  C[3] |
    """
    C_voigt = np.array([
        [C[0], C[2], C[1]],
        [C[2], C[5], C[4]],
        [C[1], C[4], C[3]],
    ])
    S = np.linalg.inv(C_voigt)

    E_x = 1.0 / S[0, 0]
    E_y = 1.0 / S[1, 1]
    nu_xy = -S[1, 0] / S[0, 0]   # load x, contraction y
    nu_yx = -S[0, 1] / S[1, 1]   # load y, contraction x
    G_xy = 1.0 / S[2, 2]

    return dict(E_x=E_x, E_y=E_y, nu_xy=nu_xy, nu_yx=nu_yx, G_xy=G_xy,
                C_voigt=C_voigt, S=S)


# ── Configuration (same as sweep) ─────────────────────────────────────────
POISSON_TARGETS = [-0.9, -0.7, -0.5, -0.3, -0.1, 0.0, 0.1, 0.3, 0.5, 0.7, 0.9]
TOPO_SEEDS = [42, 137, 256, 314, 999]
DESIGN_VARS = ['rigidities', 'rest_lengths', 'both']
OPT_SEEDS = [7, 144, 281]

# ── Generate topologies ───────────────────────────────────────────────────
topologies = []
for seed in TOPO_SEEDS:
    np.random.seed(seed)
    tri = D2C.generate_foam_points(size=(10, 10), eta=0.2)
    solver, default_rigs, _ = from_triangulation(tri)
    topologies.append({'tri': tri, 'solver': solver,
                       'n_tri': len(tri.simplices), 'seed': seed})

# ── Run optimizations and collect anisotropic properties ──────────────────
data = {dv: {'target': [], 'topo': [], 'nu_xy': [], 'nu_yx': [],
             'E_x': [], 'E_y': [], 'G': [], 'C1': [], 'C4': []}
        for dv in DESIGN_VARS}

header = f"{'target':>7s} {'dv':>13s} {'topo':>4s} {'nu_xy':>9s} {'nu_yx':>9s} " \
         f"{'E_x':>11s} {'E_y':>11s} {'G_xy':>11s} {'C_xxxy':>11s} {'C_xyyy':>11s} {'iso?':>5s}"
print(header)
print("-" * len(header))

total = len(POISSON_TARGETS) * len(TOPO_SEEDS) * len(DESIGN_VARS)
done = 0

for target_nu in POISSON_TARGETS:
    for topo_idx, topo in enumerate(topologies):
        for dv in DESIGN_VARS:
            best = None
            for opt_seed in OPT_SEEDS:
                try:
                    r = run_property_optimization(
                        solver=topo['solver'], n_triangles=topo['n_tri'],
                        target_poisson=target_nu, weight_poisson=1.0,
                        design_variable=dv, max_iter=500, lr=0.05, tol=1e-14,
                        optimizer_type='lbfgs', seed=opt_seed, verbose=False)
                    if best is None or r['final_loss'] < best['final_loss']:
                        best = r
                except Exception:
                    pass

            done += 1
            if best is None:
                continue

            C = best['pred_tensor']
            props = full_anisotropic_properties(C)

            is_iso = (abs(props['nu_xy'] - props['nu_yx']) < 0.01 and
                      abs(props['E_x'] - props['E_y']) / max(abs(props['E_x']), 1e-20) < 0.1)

            d = data[dv]
            d['target'].append(target_nu)
            d['topo'].append(topo_idx)
            d['nu_xy'].append(props['nu_xy'])
            d['nu_yx'].append(props['nu_yx'])
            d['E_x'].append(props['E_x'])
            d['E_y'].append(props['E_y'])
            d['G'].append(props['G_xy'])
            d['C1'].append(C[1])
            d['C4'].append(C[4])

            print(f"{target_nu:+7.1f} {dv:>13s} {topo_idx:4d} "
                  f"{props['nu_xy']:+9.4f} {props['nu_yx']:+9.4f} "
                  f"{props['E_x']:11.4e} {props['E_y']:11.4e} {props['G_xy']:11.4e} "
                  f"{C[1]:11.4e} {C[4]:11.4e} "
                  f"{'YES' if is_iso else 'NO':>5s}")

# Convert to arrays
for dv in DESIGN_VARS:
    for k in data[dv]:
        data[dv][k] = np.array(data[dv][k])

# ══════════════════════════════════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════════════════════════════════
dv_colors = {'rigidities': 'C0', 'rest_lengths': 'C1', 'both': 'C2'}
dv_labels = {'rigidities': 'Rigidities', 'rest_lengths': 'Rest lengths', 'both': 'Both'}

fig, axes = plt.subplots(2, 3, figsize=(18, 11))

# ── (0,0): nu_xy vs nu_yx scatter ────────────────────────────────────────
ax = axes[0, 0]
for dv in DESIGN_VARS:
    ax.scatter(data[dv]['nu_xy'], data[dv]['nu_yx'],
               c=dv_colors[dv], label=dv_labels[dv], s=15, alpha=0.6)
ax.plot([-1, 1], [-1, 1], 'k--', lw=0.8, label='Isotropic')
ax.set_xlabel(r'$\nu_{xy}$ (load in x)')
ax.set_ylabel(r'$\nu_{yx}$ (load in y)')
ax.set_title(r'$\nu_{xy}$ vs $\nu_{yx}$ — isotropy check')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

# ── (0,1): nu_xy and nu_yx vs target (mean ± std) ───────────────────────
ax = axes[0, 1]
for dv in DESIGN_VARS:
    targets = data[dv]['target']
    unique_t = np.array(sorted(set(targets)))
    for arr, style, label_suf in [(data[dv]['nu_xy'], 'o-', r'$\nu_{xy}$'),
                                   (data[dv]['nu_yx'], 's--', r'$\nu_{yx}$')]:
        means = np.array([arr[targets == t].mean() for t in unique_t])
        stds = np.array([arr[targets == t].std() for t in unique_t])
        ax.plot(unique_t, means, style, color=dv_colors[dv], markersize=4,
                label=f'{dv_labels[dv]} {label_suf}',
                alpha=1.0 if 'o' in style else 0.6)
        ax.fill_between(unique_t, means - stds, means + stds,
                        alpha=0.08, color=dv_colors[dv])
ax.plot([-1, 1], [-1, 1], 'k:', lw=0.8, label='Perfect')
ax.set_xlabel('Target ν')
ax.set_ylabel('Achieved ν')
ax.set_title(r'$\nu_{xy}$ (solid) vs $\nu_{yx}$ (dashed) vs target')
ax.legend(fontsize=5.5, ncol=2)
ax.grid(True, alpha=0.3)

# ── (0,2): |nu_xy - nu_yx| vs target ────────────────────────────────────
ax = axes[0, 2]
for dv in DESIGN_VARS:
    targets = data[dv]['target']
    gap = np.abs(data[dv]['nu_xy'] - data[dv]['nu_yx'])
    unique_t = np.array(sorted(set(targets)))
    means = np.array([gap[targets == t].mean() for t in unique_t])
    stds = np.array([gap[targets == t].std() for t in unique_t])
    ax.plot(unique_t, means, 'o-', color=dv_colors[dv],
            label=dv_labels[dv], markersize=4)
    ax.fill_between(unique_t, np.maximum(means - stds, 1e-16),
                    means + stds, alpha=0.15, color=dv_colors[dv])
ax.set_xlabel('Target ν')
ax.set_ylabel(r'$|\nu_{xy} - \nu_{yx}|$')
ax.set_title('Poisson ratio anisotropy gap')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_yscale('log')

# ── (1,0): E_x vs E_y scatter ───────────────────────────────────────────
ax = axes[1, 0]
for dv in DESIGN_VARS:
    ex = np.abs(data[dv]['E_x'])
    ey = np.abs(data[dv]['E_y'])
    ax.scatter(ex, ey, c=dv_colors[dv], label=dv_labels[dv], s=15, alpha=0.6)
ax.plot([1e-12, 1], [1e-12, 1], 'k--', lw=0.8)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'$E_x$')
ax.set_ylabel(r'$E_y$')
ax.set_title(r'$E_x$ vs $E_y$ — stiffness isotropy')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# ── (1,1): E_x/E_y ratio vs target ──────────────────────────────────────
ax = axes[1, 1]
for dv in DESIGN_VARS:
    targets = data[dv]['target']
    ratio = np.abs(data[dv]['E_x']) / np.maximum(np.abs(data[dv]['E_y']), 1e-30)
    unique_t = np.array(sorted(set(targets)))
    means = np.array([ratio[targets == t].mean() for t in unique_t])
    stds = np.array([ratio[targets == t].std() for t in unique_t])
    ax.plot(unique_t, means, 'o-', color=dv_colors[dv],
            label=dv_labels[dv], markersize=4)
    ax.fill_between(unique_t, np.maximum(means - stds, 1e-3),
                    means + stds, alpha=0.15, color=dv_colors[dv])
ax.axhline(1.0, color='k', ls='--', lw=0.8)
ax.set_xlabel('Target ν')
ax.set_ylabel(r'$E_x / E_y$')
ax.set_title('Stiffness anisotropy ratio (= 1 for isotropic)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_yscale('log')

# ── (1,2): Off-diagonal C components ────────────────────────────────────
ax = axes[1, 2]
for dv in DESIGN_VARS:
    targets = data[dv]['target']
    c1 = np.abs(data[dv]['C1'])
    c4 = np.abs(data[dv]['C4'])
    unique_t = np.array(sorted(set(targets)))
    m1 = np.array([c1[targets == t].mean() for t in unique_t])
    m4 = np.array([c4[targets == t].mean() for t in unique_t])
    ax.plot(unique_t, m1, 'o-', color=dv_colors[dv], markersize=4,
            label=f'{dv_labels[dv]} |C_xxxy|')
    ax.plot(unique_t, m4, 's--', color=dv_colors[dv], markersize=4,
            label=f'{dv_labels[dv]} |C_xyyy|', alpha=0.6)
ax.set_xlabel('Target ν')
ax.set_ylabel('|Off-diagonal C|')
ax.set_title(r'Off-diagonal components $C_{xxxy}, C_{xyyy}$ (= 0 for isotropic)')
ax.legend(fontsize=6, ncol=2)
ax.grid(True, alpha=0.3)
ax.set_yscale('log')

fig.suptitle('Anisotropy Analysis: is the optimized material isotropic?',
             fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'summary_anisotropy.png'), dpi=150, bbox_inches='tight')
print("\nSaved summary_anisotropy.png")
plt.close()
