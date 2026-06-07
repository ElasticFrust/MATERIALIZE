"""
Ribbon uniaxial simulation.

Geometry : 1:4 aspect ratio patch (width × height, height along y).
BC       : y-DOFs of topmost and bottommost node rows are clamped (stretched
           by ±delta). x-DOFs of those rows — and ALL DOFs of interior nodes —
           are free. No lateral constraint anywhere.
Measure  : transverse strain ε_xx at the middle third of the ribbon via
           linear regression of x_displaced vs x_reference.
Output   : ν = -ε_xx / ε_yy,  E = σ_yy / ε_yy  (σ_yy from reaction force).

Compared alongside the 4 MF variants from the 50×50 distort-first run.
"""
import sys, os, time
import numpy as np
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'Phase 2'))

import Disc_2_Cont_optimized as D2C

# ── Parameters ─────────────────────────────────────────────────────────────
ETA_VALUES   = np.linspace(0.0, 0.5, 11)
SIZE         = (10, 40)        # 1:4 ribbon
TRIM_FRAC    = 0.85
N_TRIALS     = 20
EPS_APPLIED  = 0.005           # applied y-strain (small → linear regime)
MF_CACHE     = os.path.join(os.path.dirname(__file__), 'mf_50x50_distortfirst_data.npz')
SIM_CACHE    = os.path.join(os.path.dirname(__file__), 'ribbon_sim_data.npz')
OUT          = os.path.join(os.path.dirname(__file__), 'new', 'mf_50x50_with_ribbon_sim.png')

# ── Mesh / spring helpers ───────────────────────────────────────────────────

def build_unique_edges(simplices):
    edge_set = {}
    for tri in simplices:
        for i, j in [(tri[0],tri[1]),(tri[0],tri[2]),(tri[1],tri[2])]:
            key = (min(i,j), max(i,j))
            edge_set[key] = True
    edges = np.array(sorted(edge_set.keys()), dtype=np.int64)
    return edges


def ribbon_sim(pts, simplices, eps=EPS_APPLIED):
    """Uniaxial ribbon test; returns (E, nu)."""
    # Active nodes only
    active = np.unique(simplices.ravel())
    remap  = {old: new for new, old in enumerate(active)}
    p      = pts[active].copy()
    simps  = np.array([[remap[n] for n in tri] for tri in simplices])
    edges  = build_unique_edges(simps)

    n  = len(p)
    l0 = np.sqrt(((p[edges[:,0]] - p[edges[:,1]])**2).sum(1))  # rest = actual
    k  = np.ones(len(edges))

    y    = p[:,1]
    ymax, ymin = y.max(), y.min()
    L    = ymax - ymin
    W    = p[:,0].max() - p[:,0].min()

    # Topmost / bottommost rows (within 1.5 of extremes)
    top_idx = np.where(y > ymax - 1.5)[0]
    bot_idx = np.where(y < ymin + 1.5)[0]

    # Fixed DOFs: y of top row (stretched up) and y of bottom row (stretched down)
    # All x DOFs and y DOFs of interior nodes are free.
    n_dof      = 2 * n
    fixed_mask = np.zeros(n_dof, dtype=bool)
    fixed_val  = p.ravel().copy()

    delta = eps * L / 2
    for i in top_idx:
        fixed_mask[2*i+1] = True
        fixed_val [2*i+1] = p[i,1] + delta
    for i in bot_idx:
        fixed_mask[2*i+1] = True
        fixed_val [2*i+1] = p[i,1] - delta

    free_idx  = np.where(~fixed_mask)[0]
    fixed_idx = np.where( fixed_mask)[0]

    # Initial guess: affine stretch in y, no lateral motion
    x0_all         = p.ravel().copy()
    x0_all[1::2]   = p[:,1] * (1 + eps)     # y stretched
    x0_all[fixed_idx] = fixed_val[fixed_idx]
    x0_free = x0_all[free_idx]

    def energy_grad(free_flat):
        pos_flat = fixed_val.copy()
        pos_flat[free_idx] = free_flat
        pos = pos_flat.reshape(n, 2)
        dr  = pos[edges[:,0]] - pos[edges[:,1]]
        l   = np.sqrt((dr**2).sum(1))
        ext = l - l0
        E   = 0.5 * (k * ext**2).sum()
        fac = k * ext / np.maximum(l, 1e-15)
        f   = np.zeros((n, 2))
        np.add.at(f, edges[:,0],  fac[:,None] * dr)
        np.add.at(f, edges[:,1], -fac[:,None] * dr)
        return E, f.ravel()[free_idx]

    res  = minimize(energy_grad, x0_free, jac=True, method='L-BFGS-B',
                    options={'maxiter':10000,'ftol':1e-15,'gtol':1e-10})
    pos_flat = fixed_val.copy()
    pos_flat[free_idx] = res.x
    pos = pos_flat.reshape(n, 2)

    # ── Transverse strain at middle third ──────────────────────────────────
    ycen  = (ymax + ymin) / 2
    band  = L / 6                              # middle third
    mid   = (np.abs(p[:,1] - ycen) < band) & \
            (np.abs(p[:,0])  > 0.1)            # exclude nodes near x=0
    if mid.sum() < 3:
        return np.nan, np.nan

    x_ref  = p  [mid, 0]
    x_def  = pos[mid, 0]
    # linear regression: x_def = (1+eps_xx)*x_ref
    eps_xx = np.dot(x_def - x_ref, x_ref) / np.dot(x_ref, x_ref)
    eps_yy = eps
    nu     = -eps_xx / eps_yy

    # ── Young's modulus from reaction force on top row ─────────────────────
    # Force in y on each top node = sum of spring forces
    F_y = 0.0
    for i in top_idx:
        neighbors = np.where((edges[:,0]==i)|(edges[:,1]==i))[0]
        for e in neighbors:
            a, b = edges[e]
            dr_e = pos[a] - pos[b]
            l_e  = np.sqrt((dr_e**2).sum())
            ext_e = l_e - l0[e]
            sign  = 1 if a == i else -1
            F_y  += sign * k[e] * ext_e * dr_e[1] / np.maximum(l_e, 1e-15)

    sigma_yy = F_y / W
    E_mod    = sigma_yy / eps_yy

    return E_mod, nu


# ── Run or load ─────────────────────────────────────────────────────────────
if os.path.exists(SIM_CACHE):
    print(f"Loading simulation cache: {SIM_CACHE}")
    sd     = np.load(SIM_CACHE)
    E_sim  = sd['E_sim']
    nu_sim = sd['nu_sim']
    etas   = sd['etas']
else:
    n_eta  = len(ETA_VALUES)
    E_sim  = np.full((n_eta, N_TRIALS), np.nan)
    nu_sim = np.full((n_eta, N_TRIALS), np.nan)
    t0     = time.time()

    for i_eta, eta in enumerate(ETA_VALUES):
        for trial in range(N_TRIALS):
            seed = 100 * i_eta + trial
            np.random.seed(seed)
            DT   = D2C.generate_foam_distort_first(SIZE, eta, trim_frac=TRIM_FRAC)
            E_mod, nu = ribbon_sim(DT.points, DT.simplices)
            E_sim [i_eta, trial] = E_mod
            nu_sim[i_eta, trial] = nu

        elapsed = time.time() - t0
        print(f"  η={eta:.2f}  E={np.nanmedian(E_sim[i_eta]):.4f}  "
              f"ν={np.nanmedian(nu_sim[i_eta]):.4f}  {elapsed:.0f}s", flush=True)

    etas = np.asarray(ETA_VALUES)
    np.savez(SIM_CACHE, E_sim=E_sim, nu_sim=nu_sim, etas=etas)
    print(f"Saved: {SIM_CACHE}")

# ── Load MF data ─────────────────────────────────────────────────────────────
mf     = np.load(MF_CACHE)
Ex_all = mf['Ex_all'];  Ey_all = mf['Ey_all']
nx_all = mf['nx_all'];  ny_all = mf['ny_all']

E_mf   = 0.5 * (Ex_all[0] + Ey_all[0])   # (n_cases, n_eta, N_MF_TRIALS)
nu_mf  = 0.5 * (nx_all[0] + ny_all[0])

E0_mf  = np.nanmedian(E_mf[0, 0])         # standard MF at η=0
E0_sim = np.nanmedian(E_sim[0])            # simulation at η=0
E_mf_n = E_mf  / E0_mf
E_sim_n = E_sim / E0_sim

print(f"\nE0_mf={E0_mf:.5f}  E0_sim={E0_sim:.5f}")
print(f"\n{'eta':>5}  {'std_MF':>8}  {'AW_MF':>8}  {'std+KKT':>8}  {'AW+KKT':>8}  {'sim_E':>8}  {'sim_nu':>8}")
for i, eta in enumerate(etas):
    row = [np.nanmedian(E_mf_n[c,i]) for c in range(4)]
    print(f"{eta:5.2f}  {'  '.join(f'{v:8.3f}' for v in row)}"
          f"  {np.nanmedian(E_sim_n[i]):8.3f}  {np.nanmedian(nu_sim[i]):8.3f}")

# ── Plot ─────────────────────────────────────────────────────────────────────
MF_CASES = [
    ('Standard MF',        'C0', 'o', '-'),
    ('Area-weighted MF',   'C1', 's', '--'),
    ('Standard MF + KKT',  'C2', '^', '-'),
    ('AW MF + KKT',        'C3', 'D', '--'),
]
jitter = np.linspace(-0.009, 0.009, 4)

fig, (ax_E, ax_nu) = plt.subplots(1, 2, figsize=(14, 5))

for ax, mf_arr, sim_arr, ylabel, ylim in [
    (ax_E,  E_mf_n,  E_sim_n,  r'$E / E_0$',            (0.0, 1.15)),
    (ax_nu, nu_mf,   nu_sim,   r"Poisson's ratio $\nu$", (-0.5, 0.45)),
]:
    for c, (label, color, marker, ls) in enumerate(MF_CASES):
        data = mf_arr[c]
        med  = np.nanmedian(data, axis=1)
        q1   = np.nanpercentile(data, 25, axis=1)
        q3   = np.nanpercentile(data, 75, axis=1)
        ax.errorbar(etas + jitter[c], med,
                    yerr=[np.clip(med-q1,0,None), np.clip(q3-med,0,None)],
                    fmt=f'{marker}{ls}', color=color, capsize=3,
                    markersize=5, label=label, alpha=0.85, lw=1.6)
        ax.fill_between(etas, q1, q3, alpha=0.08, color=color)

    # Ribbon simulation
    med_s = np.nanmedian(sim_arr, axis=1)
    q1_s  = np.nanpercentile(sim_arr, 25, axis=1)
    q3_s  = np.nanpercentile(sim_arr, 75, axis=1)
    ax.errorbar(etas, med_s,
                yerr=[np.clip(med_s-q1_s,0,None), np.clip(q3_s-med_s,0,None)],
                fmt='kP-', capsize=4, markersize=7, lw=2.2, zorder=5,
                label=f'Ribbon sim ({N_TRIALS} trials, {SIZE[0]}×{SIZE[1]})')
    ax.fill_between(etas, q1_s, q3_s, alpha=0.15, color='k')

    ref = 1.0 if ax is ax_E else 0.0
    ax.axhline(ref, color='gray', lw=0.8, ls=':')
    ax.set_xlabel(r'Disorder $\eta$', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_ylim(*ylim)
    ax.set_xlim(-0.02, 0.52)
    ax.legend(fontsize=9, loc='best')
    ax.grid(alpha=0.3)

fig.suptitle(
    rf"MF variants + ribbon simulation — distort-first mesh, trim={TRIM_FRAC}  "
    rf"(MF: 50×50, {mf['Ex_all'].shape[3]} trials | Sim: {SIZE[0]}×{SIZE[1]}, {N_TRIALS} trials)",
    fontsize=11, y=1.02)
fig.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches='tight')
print(f"\nSaved: {OUT}")
