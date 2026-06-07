"""
4-panel comparison: Standard MF / Area-weighted MF / AW+KKT
for two mesh-generation orders:
  Left pair  — distort first, then triangulate, then trim
  Right pair — triangulate first, then distort, then trim

Each pair: Young's modulus (top) and Poisson's ratio (bottom).

Area-weighted homogenisation uses Σ w_s C_s(W) for C_eff,
arithmetic homogenisation uses (1/N) Σ C_s(W).
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
from forward_solver_torch import _compute_actual_elastic_tensor

ETA_VALUES = np.linspace(0.0, 0.5, 11)
N_TRIALS   = 10
SIZE       = (20, 20)
TRIM_FRAC  = 0.85
CACHE      = os.path.join(os.path.dirname(__file__), 'mf_4panel_data.npz')
OUT        = os.path.join(os.path.dirname(__file__), 'mf_4panel.png')

# 3 solver variants per mesh
CASES = [
    # (area_weighted, use_kkt, label, color, marker, ls)
    (False, False, 'Standard MF',      'C0', 'o', '-'),
    (True,  False, 'Area-weighted MF', 'C1', 's', '--'),
    (True,  True,  'AW MF + KKT',     'C2', '^', ':'),
]
n_cases = len(CASES)
n_mesh  = 2   # 0=distort-first, 1=tri-first-then-trim
n_eta   = len(ETA_VALUES)

# ── Mesh builders ─────────────────────────────────────────────────────────────

def make_distort_first(size, eta, trim_frac=TRIM_FRAC):
    return D2C.generate_foam_distort_first(size, eta, trim_frac=trim_frac)


def make_tri_first_trimmed(size, eta, trim_frac=TRIM_FRAC):
    """Triangulate on regular lattice, distort positions, then trim edge triangles."""
    DM = D2C.generate_foam_points(size, eta)
    centroids = np.mean(DM.points[DM.simplices], axis=1)
    cx, cy = centroids[:, 0], centroids[:, 1]
    xc = (cx.max() + cx.min()) / 2
    yc = (cy.max() + cy.min()) / 2
    hw = (cx.max() - cx.min()) / 2
    hh = (cy.max() - cy.min()) / 2
    mask = (np.abs(cx - xc) <= trim_frac * hw) & (np.abs(cy - yc) <= trim_frac * hh)
    DM.simplices = DM.simplices[mask]
    return DM

MESH_BUILDERS = [make_distort_first, make_tri_first_trimmed]
MESH_LABELS   = ['Distort → triangulate → trim', 'Triangulate → distort → trim']

# ── Voigt compliance → directional constants ──────────────────────────────────

def directional_constants(C6):
    """C6 = [C1111, C1112, C1122, C2112, C2122, C2222] → (Ex, Ey, nuxy, nuyx)."""
    C0, C1, C2, C3, C4, C5 = C6
    C_V = np.array([[C0, C2, C1],
                    [C2, C5, C4],
                    [C1, C4, C3]])
    try:
        S = np.linalg.inv(C_V)
    except np.linalg.LinAlgError:
        return np.nan, np.nan, np.nan, np.nan
    def _safe(n, d):
        return n / d if abs(d) > 1e-15 else np.nan
    return (_safe(1.0, S[0,0]), _safe(1.0, S[1,1]),
            _safe(-S[1,0], S[0,0]), _safe(-S[0,1], S[1,1]))

# ── Run or load ───────────────────────────────────────────────────────────────
# Arrays: (n_mesh, n_cases, n_eta, N_TRIALS)

if os.path.exists(CACHE):
    print(f"Loading cached data from {CACHE}")
    d      = np.load(CACHE)
    Ex_all = d['Ex_all'];   Ey_all = d['Ey_all']
    nx_all = d['nx_all'];   ny_all = d['ny_all']
    etas   = d['etas']
else:
    shape  = (n_mesh, n_cases, n_eta, N_TRIALS)
    Ex_all = np.full(shape, np.nan)
    Ey_all = np.full(shape, np.nan)
    nx_all = np.full(shape, np.nan)
    ny_all = np.full(shape, np.nan)
    t0     = time.time()

    for m, build in enumerate(MESH_BUILDERS):
        print(f"\n=== Mesh {m}: {MESH_LABELS[m]} ===")
        for i_eta, eta in enumerate(ETA_VALUES):
            for trial in range(N_TRIALS):
                seed = 100 * i_eta + trial
                np.random.seed(seed)

                DT             = build(SIZE, eta)
                solver, rigs, rl = fst.from_triangulation(DT)
                w              = solver.area_weights          # (N,) sums to 1

                # Compute bare tensors once, reuse for all 3 cases
                vx = solver.edge_vecs[:, :, 0]
                vy = solver.edge_vecs[:, :, 1]
                factor = rigs / rl**2 / 16.0
                bare = torch.stack([
                    (factor * vx**4).sum(1),
                    (factor * vx**3 * vy).sum(1),
                    (factor * vx**2 * vy**2).sum(1),
                    (factor * vx * vy**3).sum(1),
                    (factor * vy**4).sum(1),
                ], dim=1)

                for c, (aw, kkt, *_) in enumerate(CASES):
                    with torch.no_grad():
                        res = solver.forward(rigs, rl,
                                             area_weighted=aw, use_kkt=kkt)
                    C6 = res['elastic_tensor'].numpy()
                    Ex, Ey, nuxy, nuyx = directional_constants(C6)
                    Ex_all[m, c, i_eta, trial] = Ex
                    Ey_all[m, c, i_eta, trial] = Ey
                    nx_all[m, c, i_eta, trial] = nuxy
                    ny_all[m, c, i_eta, trial] = nuyx

            elapsed = time.time() - t0
            row = "  ".join(
                f"{CASES[c][2][:6]}:E={np.nanmedian(Ex_all[m,c,i_eta]):.4f}"
                for c in range(n_cases))
            print(f"  η={eta:.2f}  {row}  {elapsed:.0f}s", flush=True)

    etas = np.asarray(ETA_VALUES)
    np.savez(CACHE, Ex_all=Ex_all, Ey_all=Ey_all,
             nx_all=nx_all, ny_all=ny_all, etas=etas)
    print(f"\nData saved to {CACHE}")

# ── Derived scalars ───────────────────────────────────────────────────────────
E_all  = 0.5 * (Ex_all + Ey_all)
nu_all = 0.5 * (nx_all + ny_all)

# Normalise E by η=0 standard-MF median (mesh 0, case 0)
E0 = np.nanmedian(E_all[0, 0, 0])
print(f"\nE0 = {E0:.5f}")

E_n = E_all / E0

# Print summary table
for m in range(n_mesh):
    print(f"\n{MESH_LABELS[m]}")
    header = f"{'eta':>5}  " + "  ".join(f"{CASES[c][2]:>18}" for c in range(n_cases))
    print(header + "   (E/E0 | ν)")
    for i, eta in enumerate(etas):
        vals = []
        for c in range(n_cases):
            e = np.nanmedian(E_n[m, c, i])
            n = np.nanmedian(nu_all[m, c, i])
            vals.append(f"{e:6.3f}|{n:+6.3f}")
        print(f"{eta:5.2f}  " + "  ".join(f"{v:>18}" for v in vals))

# ── Plot ──────────────────────────────────────────────────────────────────────
jitter = np.linspace(-0.008, 0.008, n_cases)

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
PANEL_DATA = [
    (E_n,   r'$E / E_0$',          True,  (0.0, 1.15)),
    (nu_all, r"Poisson's ratio $\nu$", False, (-0.8, 0.45)),
]

for col, (m, mesh_label) in enumerate(zip(range(n_mesh), MESH_LABELS)):
    for row, (arr, ylabel, normalised, ylim) in enumerate(PANEL_DATA):
        ax = axes[row, col]
        for c, (aw, kkt, label, color, marker, ls) in enumerate(CASES):
            data = arr[m, c]          # (n_eta, N_TRIALS)
            med  = np.nanmedian(data, axis=1)
            q1   = np.nanpercentile(data, 25, axis=1)
            q3   = np.nanpercentile(data, 75, axis=1)
            ax.errorbar(
                etas + jitter[c], med,
                yerr=[np.clip(med - q1, 0, None), np.clip(q3 - med, 0, None)],
                fmt=f'{marker}{ls}', color=color, capsize=3,
                markersize=5, label=label, alpha=0.9, lw=1.8,
            )
            ax.fill_between(etas, q1, q3, alpha=0.10, color=color)

        if normalised:
            ax.axhline(1.0, color='gray', lw=0.8, ls=':')
        else:
            ax.axhline(0.0, color='gray', lw=0.8, ls=':')
        ax.set_xlabel(r'Disorder $\eta$', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_ylim(*ylim)
        ax.set_xlim(-0.02, 0.52)
        ax.legend(fontsize=9, loc='best')
        ax.grid(alpha=0.3)
        ax.set_title(mesh_label, fontsize=11)

fig.suptitle(
    rf"Elastic constants vs $\eta$ — pure MF vs AW MF vs AW+KKT  "
    rf"({SIZE[0]}×{SIZE[1]} network, {N_TRIALS} trials, trim={TRIM_FRAC})",
    fontsize=13, y=1.01,
)
fig.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches='tight')
print(f"\nSaved: {OUT}")
