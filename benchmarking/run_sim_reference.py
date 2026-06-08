"""
Physical simulation reference for the 50x50 distort-first MF comparison.

Uses L-BFGS-B energy minimization with KUBC boundary conditions (same as
Phase 3/mechanical_simulation.py) on the identical mesh as the MF run.
Spring constants k=1, rest lengths = actual edge lengths (no pre-stress).

Outputs:
  sim_reference_50x50_data.npz
  new/mf_50x50_with_sim_reference.png   — MF curves + simulation reference
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
sys.path.insert(0, os.path.join(ROOT, 'Phase 3'))

import Disc_2_Cont_optimized as D2C

ETA_VALUES    = np.linspace(0.0, 0.5, 11)
SIZE          = (50, 50)
TRIM_FRAC     = 0.85
N_TRIALS_SIM  = 10
MF_CACHE      = os.path.join(os.path.dirname(__file__), 'mf_50x50_distortfirst_data.npz')
SIM_CACHE     = os.path.join(os.path.dirname(__file__), 'sim_reference_50x50_data.npz')
OUT           = os.path.join(os.path.dirname(__file__), 'new', 'mf_50x50_with_sim_reference.png')

# ── Simulation helpers (from Phase 3/mechanical_simulation.py) ────────────────

def build_unique_edges(simplices, rigs_per_tri):
    edge_dict = {}
    for ti, sv in enumerate(simplices):
        for ei, (i, j) in enumerate([(sv[0], sv[1]),
                                      (sv[0], sv[2]),
                                      (sv[1], sv[2])]):
            key = (min(i, j), max(i, j))
            if key not in edge_dict:
                edge_dict[key] = []
            edge_dict[key].append(rigs_per_tri[ti, ei])
    keys   = sorted(edge_dict.keys())
    edges  = np.array(keys, dtype=int)
    k_vals = np.array([np.mean(edge_dict[k]) for k in keys])
    return edges, k_vals


def classify_boundary(points, margin_frac=0.10):
    lo = points.min(axis=0);  hi = points.max(axis=0)
    m  = (hi - lo) * margin_frac
    return ((points[:, 0] < lo[0] + m[0]) | (points[:, 0] > hi[0] - m[0]) |
            (points[:, 1] < lo[1] + m[1]) | (points[:, 1] > hi[1] - m[1]))


def energy_and_grad(free_flat, fixed_pos, free_idx, fixed_idx,
                    edges, k, rest_lengths, n_nodes):
    pos = np.empty((n_nodes, 2))
    pos[fixed_idx] = fixed_pos
    pos[free_idx]  = free_flat.reshape(-1, 2)
    dr      = pos[edges[:, 0]] - pos[edges[:, 1]]
    lengths = np.sqrt(np.sum(dr ** 2, axis=1))
    ext     = lengths - rest_lengths
    energy  = 0.5 * np.dot(k, ext ** 2)
    fac     = k * ext / np.maximum(lengths, 1e-15)
    forces  = np.zeros((n_nodes, 2))
    fdr     = fac[:, None] * dr
    np.add.at(forces, edges[:, 0],  fdr)
    np.add.at(forces, edges[:, 1], -fdr)
    return energy, forces[free_idx].ravel().copy()


def simulate_strain(points, edges, k, rest_lengths, boundary_mask,
                    strain_tensor, tol=1e-12):
    n_nodes   = len(points)
    deformed  = points + points @ strain_tensor.T
    free_idx  = np.where(~boundary_mask)[0]
    fixed_idx = np.where(boundary_mask)[0]
    fixed_pos = deformed[fixed_idx]
    x0        = deformed[free_idx].ravel()
    res = minimize(energy_and_grad, x0,
                   args=(fixed_pos, free_idx, fixed_idx, edges, k, rest_lengths, n_nodes),
                   method='L-BFGS-B', jac=True,
                   options={'maxiter': 10000, 'ftol': 1e-15, 'gtol': tol})
    final = np.empty((n_nodes, 2))
    final[fixed_idx] = fixed_pos
    final[free_idx]  = res.x.reshape(-1, 2)
    dr  = final[edges[:, 0]] - final[edges[:, 1]]
    ext = np.sqrt(np.sum(dr**2, axis=1)) - rest_lengths
    return 0.5 * np.dot(k, ext**2)


def extract_elastic_tensor(points, simplices, edges, k, rest_lengths,
                           boundary_mask, delta=0.005):
    v0, v1, v2 = points[simplices[:,0]], points[simplices[:,1]], points[simplices[:,2]]
    A = 0.5 * np.abs((v1[:,0]-v0[:,0])*(v2[:,1]-v0[:,1]) -
                     (v1[:,1]-v0[:,1])*(v2[:,0]-v0[:,0])).sum()

    strains = {
        'xx':    np.array([[1,0],[0,0]]),
        'yy':    np.array([[0,0],[0,1]]),
        'xy':    np.array([[0,.5],[.5,0]]),
        'xx+yy': np.array([[1,0],[0,1]]),
        'xx+xy': np.array([[1,.5],[.5,0]]),
        'yy+xy': np.array([[0,.5],[.5,1]]),
    }
    E = {name: simulate_strain(points, edges, k, rest_lengths,
                               boundary_mask, delta * eps)
         for name, eps in strains.items()}

    d2 = delta**2
    C  = np.zeros((3, 3))
    C[0,0] = 2*E['xx']    / (A*d2)
    C[1,1] = 2*E['yy']    / (A*d2)
    C[2,2] = 2*E['xy']    / (A*d2)
    C[0,1] = C[1,0] = (2*E['xx+yy']/(A*d2) - C[0,0] - C[1,1]) / 2
    C[0,2] = C[2,0] = (2*E['xx+xy']/(A*d2) - C[0,0] - C[2,2]) / 2
    C[1,2] = C[2,1] = (2*E['yy+xy']/(A*d2) - C[1,1] - C[2,2]) / 2
    return C


def simulate_foam(DM):
    """Run KUBC simulation on a trimmed Delaunay mesh; return (Ex,Ey,nuxy,nuyx)."""
    # Restrict to nodes that appear in trimmed simplices
    active = np.unique(DM.simplices.ravel())
    remap  = {old: new for new, old in enumerate(active)}
    pts    = DM.points[active]
    simps  = np.array([[remap[n] for n in tri] for tri in DM.simplices])

    rigs_per_tri = np.ones((len(simps), 3))
    edges, k_vals = build_unique_edges(simps, rigs_per_tri)
    rest_lengths  = np.sqrt(np.sum((pts[edges[:,0]] - pts[edges[:,1]])**2, axis=1))
    boundary_mask = classify_boundary(pts, margin_frac=0.10)

    C = extract_elastic_tensor(pts, simps, edges, k_vals, rest_lengths, boundary_mask)
    try:
        S    = np.linalg.inv(C)
        Ex   = 1.0 / S[0, 0]
        Ey   = 1.0 / S[1, 1]
        nuxy = -S[0, 1] / S[0, 0]
        nuyx = -S[0, 1] / S[1, 1]
        return Ex, Ey, nuxy, nuyx
    except np.linalg.LinAlgError:
        return np.nan, np.nan, np.nan, np.nan


# ── Run or load simulation ────────────────────────────────────────────────────
if os.path.exists(SIM_CACHE):
    print(f"Loading simulation cache: {SIM_CACHE}")
    sd     = np.load(SIM_CACHE)
    Ex_sim = sd['Ex_sim'];  Ey_sim = sd['Ey_sim']
    nx_sim = sd['nx_sim'];  ny_sim = sd['ny_sim']
    etas   = sd['etas']
else:
    n_eta  = len(ETA_VALUES)
    Ex_sim = np.full((n_eta, N_TRIALS_SIM), np.nan)
    Ey_sim = np.full((n_eta, N_TRIALS_SIM), np.nan)
    nx_sim = np.full((n_eta, N_TRIALS_SIM), np.nan)
    ny_sim = np.full((n_eta, N_TRIALS_SIM), np.nan)
    t0     = time.time()

    for i_eta, eta in enumerate(ETA_VALUES):
        for trial in range(N_TRIALS_SIM):
            seed = 100 * i_eta + trial
            np.random.seed(seed)
            DT = D2C.generate_foam_distort_first(SIZE, eta, trim_frac=TRIM_FRAC)
            Ex, Ey, nuxy, nuyx = simulate_foam(DT)
            Ex_sim[i_eta, trial] = Ex
            Ey_sim[i_eta, trial] = Ey
            nx_sim[i_eta, trial] = nuxy
            ny_sim[i_eta, trial] = nuyx
        elapsed = time.time() - t0
        print(f"  η={eta:.2f}  E={np.nanmedian(Ex_sim[i_eta]):.4f}  "
              f"ν={np.nanmedian(nx_sim[i_eta]):.4f}  {elapsed:.0f}s", flush=True)

    etas = np.asarray(ETA_VALUES)
    np.savez(SIM_CACHE, Ex_sim=Ex_sim, Ey_sim=Ey_sim,
             nx_sim=nx_sim, ny_sim=ny_sim, etas=etas)
    print(f"Simulation data saved to {SIM_CACHE}")

# ── Load MF data ──────────────────────────────────────────────────────────────
mf = np.load(MF_CACHE)
Ex_all = mf['Ex_all'];  Ey_all = mf['Ey_all']
nx_all = mf['nx_all'];  ny_all = mf['ny_all']

E_mf  = 0.5 * (Ex_all[0] + Ey_all[0])   # (n_cases, n_eta, N_TRIALS_MF)
nu_mf = 0.5 * (nx_all[0] + ny_all[0])

E_sim  = 0.5 * (Ex_sim + Ey_sim)
nu_sim = 0.5 * (nx_sim + ny_sim)

E0 = np.nanmedian(E_mf[0, 0])            # standard MF, η=0
print(f"\nE0 = {E0:.5f}")
E_mf_n  = E_mf  / E0
E_sim_n = E_sim / E0

# ── Plot ──────────────────────────────────────────────────────────────────────
MF_CASES = [
    (False, False, 'Standard MF',        'C0', 'o', '-'),
    (True,  False, 'Area-weighted MF',   'C1', 's', '--'),
    (False, True,  'Standard MF + KKT',  'C2', '^', '-'),
    (True,  True,  'AW MF + KKT',        'C3', 'D', '--'),
]
N_MF = mf['Ex_all'].shape[3]
jitter = np.linspace(-0.009, 0.009, len(MF_CASES))

fig, (ax_E, ax_nu) = plt.subplots(1, 2, figsize=(14, 5))

for ax, E_arr, nu_arr, ylabel, ylim in [
    (ax_E,  E_mf_n,  nu_mf,  r'$E / E_0$',             (0.0, 1.15)),
    (ax_nu, E_mf_n,  nu_mf,  r"Poisson's ratio $\nu$",  (-0.8, 0.45)),
]:
    arr = E_arr if ax is ax_E else nu_arr
    for c, (_, _, label, color, marker, ls) in enumerate(MF_CASES):
        data = arr[c]
        med  = np.nanmedian(data, axis=1)
        q1   = np.nanpercentile(data, 25, axis=1)
        q3   = np.nanpercentile(data, 75, axis=1)
        ax.errorbar(etas + jitter[c], med,
                    yerr=[np.clip(med-q1, 0, None), np.clip(q3-med, 0, None)],
                    fmt=f'{marker}{ls}', color=color, capsize=3,
                    markersize=5, label=label, alpha=0.85, lw=1.6)
        ax.fill_between(etas, q1, q3, alpha=0.08, color=color)

    # Simulation reference
    sim_arr = E_sim_n if ax is ax_E else nu_sim
    med_s   = np.nanmedian(sim_arr, axis=1)
    q1_s    = np.nanpercentile(sim_arr, 25, axis=1)
    q3_s    = np.nanpercentile(sim_arr, 75, axis=1)
    ax.errorbar(etas, med_s,
                yerr=[np.clip(med_s-q1_s, 0, None), np.clip(q3_s-med_s, 0, None)],
                fmt='kP-', capsize=4, markersize=7, lw=2.2,
                label=f'Simulation (KUBC, {N_TRIALS_SIM} trials)', zorder=5)
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
    rf"Elastic constants vs $\eta$ — MF variants + KUBC simulation reference  "
    rf"({SIZE[0]}×{SIZE[1]} distort-first mesh, trim={TRIM_FRAC})",
    fontsize=12, y=1.02)
fig.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches='tight')
print(f"\nSaved: {OUT}")
