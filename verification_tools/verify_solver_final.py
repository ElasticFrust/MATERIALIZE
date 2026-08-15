"""
Final test: 10 different (network structure, rigidity-distribution) combinations -- NOT the
eta-disorder sweep and NOT the tanh virtual-distortion (VD) contrast used in
verify_solver_sweep.py. Each case is a perturbed periodic triangular lattice (own N, eta,
seed -- i.e. its own structure) with a per-bond rigidity k drawn from a different
distribution / spatial pattern (random, lognormal, bimodal, clustered, graded, power-law,
orientation-dependent, sparse inclusions, smooth random field).

For each case we compare the homogenised Poisson ratio nu and Young's modulus E from the
PBC simulation (truth) vs the PRODUCTION forward solver (ElasticSolver.forward,
method='intrinsic'), driven exactly (rigidities = per-edge bond k, rest_lengths = reference
edge length).
"""
import os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..', 'Phase 2'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from mesh_build import kkt_from_tri_bond
import verify_solver_sweep as svs       # sim_nuE, solver_nuE (make_solver now: solver_build)
import metric_ops as MO
import mesh_build as MB
import sim_assembly as SA
import solver_build as SB
torch.set_default_dtype(torch.float64)


# ---- 10 rigidity-distribution generators (operate on per-bond arrays) -------------------
def k_uniform_random(geo, rng):
    nb = len(geo['bond_R'])
    return rng.uniform(0.3, 1.7, nb)


def k_lognormal(geo, rng):
    nb = len(geo['bond_R'])
    return np.clip(rng.lognormal(0.0, 0.5, nb), 0.05, None)


def k_bimodal(geo, rng):
    nb = len(geo['bond_R'])
    return np.where(rng.random(nb) < 0.5, 0.2, 1.8)


def k_bimodal_clusters(geo, rng):
    mid = geo['pts'][geo['bond_u']] + geo['bond_R'] / 2
    N = geo['N']
    Lx, Ly = N, N * np.sqrt(3) / 2
    bx = np.floor((mid[:, 0] % Lx) / (Lx / 2)).astype(int)
    by = np.floor((mid[:, 1] % Ly) / (Ly / 2)).astype(int)
    return np.where((bx + by) % 2 == 0, 0.2, 1.8)


def k_gradient_x(geo, rng):
    mid = geo['pts'][geo['bond_u']] + geo['bond_R'] / 2
    N = geo['N']
    xfrac = (mid[:, 0] % N) / N
    return 0.2 + 1.6 * xfrac


def k_power_law(geo, rng):
    nb = len(geo['bond_R'])
    return np.clip(rng.pareto(2.0, nb) * 0.3 + 0.1, 0.05, 5.0)


def k_orientation(geo, rng):
    R = geo['bond_R']
    ang = np.arctan2(R[:, 1], R[:, 0])
    return 1.0 + 0.8 * np.cos(2 * ang)


def k_sparse_soft(geo, rng):
    nb = len(geo['bond_R'])
    k = np.ones(nb)
    k[rng.random(nb) < 0.1] = 0.05
    return k


def k_sparse_stiff(geo, rng):
    nb = len(geo['bond_R'])
    k = np.ones(nb)
    k[rng.random(nb) < 0.1] = 5.0
    return k


def k_smooth_field(geo, rng):
    N = geo['N']
    field = rng.normal(1.0, 0.4, (N, N))
    smooth = (field + np.roll(field, 1, 0) + np.roll(field, -1, 0)
              + np.roll(field, 1, 1) + np.roll(field, -1, 1)) / 5
    sf = smooth.ravel()
    return np.clip((sf[geo['bond_u']] + sf[geo['bond_v']]) / 2, 0.1, None)


# ---- the 10 cases: each its own (network structure) + (rigidity distribution) -----------
CASES = [
    ('uniform-random k~U(0.3,1.7)',       16, 0.10, 101, k_uniform_random),
    ('lognormal k',                       18, 0.15, 102, k_lognormal),
    ('bimodal k in {0.2,1.8} (random)',   14, 0.00, 103, k_bimodal),
    ('bimodal clusters (checkerboard)',   20, 0.20, 104, k_bimodal_clusters),
    ('graded k(x): 0.2->1.8',             16, 0.10, 105, k_gradient_x),
    ('power-law (heavy-tail) k',          14, 0.00, 106, k_power_law),
    ('orientation-dependent k(theta)',    16, 0.05, 107, k_orientation),
    ('sparse soft inclusions (10%)',      18, 0.10, 108, k_sparse_soft),
    ('sparse stiff inclusions (10%)',     18, 0.10, 109, k_sparse_stiff),
    ('smooth random field k',             20, 0.15, 110, k_smooth_field),
]


def run_case(name, N, eta, seed, kfun):
    rng = np.random.default_rng(seed)
    geo = MB.build_geometry(N, eta, seed=seed)
    geo['bond_k'] = kfun(geo, rng)
    geo['tri_k'] = geo['bond_k'][geo['tri_bond']]
    kkt = kkt_from_tri_bond(geo['tri_bond'], geo['edge_vecs'])
    sv = SB.make_solver(geo, kkt)
    rl = torch.as_tensor(np.sqrt(geo['actual_len2']), dtype=torch.float64)
    ns, Es = svs.sim_nuE(geo, SA.assemble_K_faff, MO.bare_tensor(geo))
    ni, Ei = svs.solver_nuE(sv, geo['tri_k'], rl)
    return ns, Es, ni, Ei, geo['bond_k']


def main():
    os.makedirs(os.path.join(HERE, 'plots'), exist_ok=True)
    rows = []
    print(f"{'case':<34} {'N':>3} {'eta':>5} | {'nu_sim':>8} {'nu_int':>8} {'dnu':>8}"
          f" | {'E_sim':>9} {'E_int':>9} {'dE/E':>7}")
    for name, N, eta, seed, kfun in CASES:
        ns, Es, ni, Ei, bk = run_case(name, N, eta, seed, kfun)
        rows.append((name, N, eta, ns, Es, ni, Ei))
        print(f"{name:<34} {N:>3} {eta:>5.2f} | {ns:>8.4f} {ni:>8.4f} {ni-ns:>8.4f}"
              f" | {Es:>9.5f} {Ei:>9.5f} {(Ei-Es)/Es:>7.2%}   "
              f"k in [{bk.min():.3f},{bk.max():.3f}]", flush=True)

    names = [r[0] for r in rows]
    nu_s = np.array([r[3] for r in rows]); nu_i = np.array([r[5] for r in rows])
    E_s  = np.array([r[4] for r in rows]); E_i  = np.array([r[6] for r in rows])

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, ys, yi, ttl in [(axes[0], nu_s, nu_i, 'Poisson ratio ν'),
                            (axes[1], E_s, E_i, "Young's modulus E")]:
        lo, hi = min(ys.min(), yi.min()), max(ys.max(), yi.max())
        pad = 0.05 * (hi - lo if hi > lo else 1.0)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], 'k--', lw=1, alpha=0.5)
        ax.scatter(ys, yi, c=range(len(ys)), cmap='tab10', s=60, zorder=3)
        for i, n in enumerate(names):
            ax.annotate(str(i + 1), (ys[i], yi[i]), fontsize=8,
                         xytext=(4, 4), textcoords='offset points')
        ax.set_xlabel('simulation'); ax.set_ylabel('forward solver (intrinsic)')
        ax.set_title(ttl); ax.grid(alpha=0.3); ax.set_aspect('equal', adjustable='box')
    fig.suptitle('Forward solver vs simulation — 10 networks, varied structure & rigidity '
                  'distributions\n(legend: 1=' + ', '.join(f'{i+1}={n}' for i, n in
                  enumerate(names)) + ')', fontsize=8)
    plt.tight_layout(rect=[0, 0, 1, 0.88])
    p = os.path.join(HERE, 'plots', 'dg_solver_final_10networks.png')
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print('saved', p)

    np.savez(os.path.join(HERE, 'plots', 'dg_solver_final_10networks.npz'),
             names=np.array(names, object), N=np.array([r[1] for r in rows]),
             eta=np.array([r[2] for r in rows]),
             nu_sim=nu_s, E_sim=E_s, nu_int=nu_i, E_int=E_i)


if __name__ == '__main__':
    main()
