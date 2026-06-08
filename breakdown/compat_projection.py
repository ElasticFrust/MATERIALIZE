"""
Compatibility projection test — is the MF's residual error just INCOMPATIBILITY?

A per-triangle non-affine metric-change field {delta_g(s)} is COMPATIBLE iff it is the
(linearized) strain of a single periodic node-displacement field u:
    delta_g(s) = 2 sym( dU_s @ E_ref_s^-1 ),   dU_s = [u1-u0, u2-u0] (columns)
The simulation's field is compatible by construction; the mean-field computes each triangle
independently and is generically NOT.

We assemble the linear map B: u -> {delta_g(s)} (sparse, 3*N_tri x 2*N_node) and, for each
method, solve the least squares  min_u ||B u - delta_g_method||  to get the CLOSEST
compatible field  delta_g_proj = B u*.  Then:
  - incompatibility fraction  = ||delta_g - delta_g_proj|| / ||delta_g||   (0 = compatible)
  - corr(delta_g, sim)        before projection
  - corr(delta_g_proj, sim)   after projection
If projecting onto the compatible subspace makes the MF jump toward the simulation, then
the missing physics is COMPATIBILITY (edge + vertex), of which edge-KKT supplies only part.

Run:  python compat_projection.py
"""
import os
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'dg_analysis_data')
PLOTS = os.path.join(HERE, 'plots')
METHODS = ['Std', 'AW', 'Std+edge', 'AW+edge']
COLORS = {'sim': 'k', 'Std': '#1f77b4', 'AW': '#ff7f0e', 'Std+edge': '#2ca02c', 'AW+edge': '#d62728'}


def build_B(edge_vecs, simplices, n_node):
    """Sparse map u (2*n_node) -> per-triangle delta_g [dg11,dg12,dg22] (3*n_tri).

    Rows 3*t+{0,1,2} = dg11, dg12, dg22.  delta_g = 2 sym(dU @ M), M = E_ref^-1.
    """
    n = len(simplices)
    M = np.linalg.inv(np.stack([edge_vecs[:, 0], edge_vecs[:, 1]], axis=-1))  # (n,2,2)
    M00, M01 = M[:, 0, 0], M[:, 0, 1]
    M10, M11 = M[:, 1, 0], M[:, 1, 1]
    n0, n1, n2 = simplices[:, 0], simplices[:, 1], simplices[:, 2]
    t = np.arange(n)
    rows, cols, data = [], [], []

    def add(r, c, v):
        rows.append(r); cols.append(c); data.append(v)

    # dg11 = 2[(u1x-u0x)M00 + (u2x-u0x)M10]
    add(3 * t + 0, 2 * n1 + 0, 2 * M00)
    add(3 * t + 0, 2 * n2 + 0, 2 * M10)
    add(3 * t + 0, 2 * n0 + 0, -2 * (M00 + M10))
    # dg22 = 2[(u1y-u0y)M01 + (u2y-u0y)M11]
    add(3 * t + 2, 2 * n1 + 1, 2 * M01)
    add(3 * t + 2, 2 * n2 + 1, 2 * M11)
    add(3 * t + 2, 2 * n0 + 1, -2 * (M01 + M11))
    # dg12 = (u1x-u0x)M01 + (u2x-u0x)M11 + (u1y-u0y)M00 + (u2y-u0y)M10
    add(3 * t + 1, 2 * n1 + 0, M01)
    add(3 * t + 1, 2 * n2 + 0, M11)
    add(3 * t + 1, 2 * n0 + 0, -(M01 + M11))
    add(3 * t + 1, 2 * n1 + 1, M00)
    add(3 * t + 1, 2 * n2 + 1, M10)
    add(3 * t + 1, 2 * n0 + 1, -(M00 + M10))

    rows = np.concatenate([np.atleast_1d(r) for r in rows])
    cols = np.concatenate([np.atleast_1d(c) for c in cols])
    data = np.concatenate([np.atleast_1d(d) for d in data])
    return sp.coo_matrix((data, (rows, cols)), shape=(3 * n, 2 * n_node)).tocsc()


def project(B, splu, free, d):
    """Least-squares u* minimizing ||B u - d|| (gauge fixed by `free`), return B u*."""
    rhs = (B.T @ d)[free]
    u = np.zeros(B.shape[1])
    u[free] = splu.solve(rhs)
    return B @ u


def dg_to_vec(dg):                       # (n,2,2) -> (3n,) [dg11,dg12,dg22] per triangle
    return np.stack([dg[:, 0, 0], dg[:, 0, 1], dg[:, 1, 1]], axis=1).ravel()


def corr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return np.corrcoef(a[m], b[m])[0, 1] if a[m].std() > 0 else np.nan


def main():
    d = np.load(os.path.join(DATA, 'summary.npz'), allow_pickle=True)
    etas = d['eta_values']
    names = ['sim'] + METHODS
    incompat = {k: [] for k in names}
    corr_before = {k: [] for k in METHODS}
    corr_after = {k: [] for k in METHODS}

    for eta in etas:
        inc = {k: [] for k in names}
        cb = {k: [] for k in METHODS}
        ca = {k: [] for k in METHODS}
        for t in range(int(d['n_trials'])):
            f = os.path.join(DATA, f'sample_eta{eta:.2f}_trial{t}.npz')
            if not os.path.exists(f):
                continue
            s = np.load(f, allow_pickle=True)
            n_node = len(s['pts'])
            B = build_B(s['edge_vecs'], s['simplices'], n_node)
            # gauge: pin node0 (x,y) and node1 x -> remove translation(2)+rotation(1)
            free = np.array([i for i in range(2 * n_node) if i not in (0, 1, 2)])
            N = (B.T @ B).tocsc()[free][:, free]
            splu = spla.splu(N)

            sim_v = dg_to_vec(s['dg_sim'])
            for nm in names:
                dv = sim_v if nm == 'sim' else dg_to_vec(s[f'dg_{nm}'])
                proj = project(B, splu, free, dv)
                inc[nm].append(np.linalg.norm(dv - proj) / max(np.linalg.norm(dv), 1e-30))
                if nm != 'sim':
                    cb[nm].append(corr(dv, sim_v))
                    ca[nm].append(corr(proj, sim_v))
        for nm in names:
            incompat[nm].append(np.nanmean(inc[nm]) if inc[nm] else np.nan)
        for nm in METHODS:
            corr_before[nm].append(np.nanmean(cb[nm]) if cb[nm] else np.nan)
            corr_after[nm].append(np.nanmean(ca[nm]) if ca[nm] else np.nan)

    # ── report ──
    print("Incompatibility fraction  ||dg - P dg|| / ||dg||  (0 = fully compatible):")
    print(f"{'eta':>5} " + " ".join(f"{nm:>10}" for nm in names))
    for i, eta in enumerate(etas):
        print(f"{eta:>5.1f} " + " ".join(f"{incompat[nm][i]:>10.3f}" for nm in names))
    print("\ncorr(dg, sim) before -> after compatibility projection:")
    print(f"{'eta':>5} " + " ".join(f"{nm:>16}" for nm in METHODS))
    for i, eta in enumerate(etas):
        print(f"{eta:>5.1f} " + " ".join(
            f"{corr_before[nm][i]:.2f}->{corr_after[nm][i]:.2f}" for nm in METHODS).replace('nan', ' - '))

    # ── plot ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    for nm in names:
        ax.plot(etas, incompat[nm], '-o', ms=4, color=COLORS[nm],
                label=nm, lw=2.2 if nm == 'sim' else 1.4)
    ax.set_xlabel(r'$\eta$'); ax.set_ylabel(r'$\|\delta g - P\,\delta g\|/\|\delta g\|$')
    ax.set_title('Incompatibility fraction of the $\\delta g$ field\n(0 = realizable by a node-displacement field)')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    ax = axes[1]
    for nm in METHODS:
        ax.plot(etas, corr_before[nm], '--o', ms=4, color=COLORS[nm], alpha=0.6, label=f'{nm} raw')
        ax.plot(etas, corr_after[nm], '-s', ms=4, color=COLORS[nm], label=f'{nm} projected')
    ax.set_xlabel(r'$\eta$'); ax.set_ylabel('corr with simulation'); ax.set_ylim(-0.05, 1.05)
    ax.set_title('Match to simulation: raw (dashed) vs\ncompatibility-projected (solid)')
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)

    fig.suptitle('Compatibility projection: does forcing the MF field to be a real '
                 'displacement field recover the simulation?', fontsize=12)
    plt.tight_layout()
    out = os.path.join(PLOTS, 'dg_compat_projection.png')
    plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    print('\nsaved', out)


if __name__ == '__main__':
    main()
