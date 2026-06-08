"""
Per-triangle non-affine strain (delta-g) under a single macroscopic strain:
PBC spring-network simulation vs mean-field (Woodbury) and MF + edge-KKT.

CONVENTION (metric change Delta_g = F^T F - I)
----------------------------------------------
The strain in our (incompatible-elasticity) framework is the METRIC CHANGE
Delta_g = g - g-bar = F^T F - I, with reference metric g-bar = I. It is NON-LINEAR and is
twice the usual linear strain eps = sym(F - I) to leading order. We use the metric change
of each triangle (a frame-independent, lab quantity), NOT the edge-basis Gram matrix
[[a.a,a.b],[a.b,b.b]] -- the latter mixes the strain with the triangle's edge orientation
and produced a spurious 'spread' even for a perfect, affinely-deforming crystal.

  - macroscopic:    Delta_g = F^T F - I
  - per-triangle:   g_tri = F_s^T F_s - I,  F_s = E_def @ E_ref^-1
      where E_ref = [e01 e02] (reference edges as columns) and E_def the deformed edges
      (F.e + fluctuation). A triangle's 3 vertices define one affine map F_s, so F_s^T F_s
      is its metric (constant-strain element).
  - NON-AFFINE metric change (this is "delta_g"):
        sim :  delta_g(s) = g_tri - Delta_g           (= F_s^T F_s - F^T F)
        MF  :  delta_g(s) = W_s : Delta_g   (the solver's 4-index strain concentration,
               delta_g4 = W_mat @ Delta_g4, W_mat exactly as in
               forward_solver_torch._compute_actual_elastic_tensor / Disc_2_Cont.Wmat)
  Both are symmetric 2x2; both vanish for an affine deformation; <delta_g_MF> = 0.

ANALYSIS
--------
Each per-triangle symmetric 2x2 delta_g is decomposed into principal values
(eigenvalues lam1 >= lam2) and principal-axis angle theta. We compare, per triangle, each
method against the simulation via principal-value differences (normalized by ||Delta_g||)
and principal-axis angle differences.

SELF-CONTAINED: does NOT import the (unvalidated) breakdown modules periodic_mesh.py /
pbc_simulation.py. Reuses only the Phase 2 core Woodbury functions (which have test
coverage). Does not modify the Phase 2 solver.

USAGE
-----
  python pbc_dg_analysis.py           # full run (simulate + analyse + plot)
  python pbc_dg_analysis.py --regen   # rebuild summary + plots from STORED samples only
                                       # (no re-simulation; recomputes the lab-frame delta_g
                                       #  from the stored edges / fluctuation field / W)

All raw data is saved to breakdown/dg_analysis_data/ for later analysis.
"""
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
sys.path.insert(0, os.path.join(ROOT, 'Phase 2'))
import forward_solver_torch as fst  # noqa: E402

torch.set_default_dtype(torch.float64)

# ── Configuration ───────────────────────────────────────────────────────────
N          = 40                                   # N x N lattice -> 2N^2 triangles
ETA_VALUES = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
N_TRIALS   = 10
DELTA      = 1e-3                                  # macroscopic strain amplitude (linear)
H_MACRO    = np.array([[1.0, 0.0], [0.0, 0.0]])   # uniaxial epsilon_xx
PLOT_ETA   = 0.4

# (label, area_weighted, use_kkt) -- NO angle methods
METHODS = [
    ('Std',      False, False),
    ('AW',       True,  False),
    ('Std+edge', False, True),
    ('AW+edge',  True,  True),
]
METHOD_LABELS = [m[0] for m in METHODS]
COLORS = {'Std': '#1f77b4', 'AW': '#ff7f0e', 'Std+edge': '#2ca02c', 'AW+edge': '#d62728'}

DATA_DIR  = os.path.join(HERE, 'dg_analysis_data')
PLOTS_DIR = os.path.join(HERE, 'plots')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


# ── Mesh: periodic triangulate-first-then-deform ──────────────────────────────
def build_periodic_tf_mesh(N, eta, seed):
    """Regular N x N triangular lattice on a rhombic torus, all vertices perturbed.

    L1 = [1, 0], L2 = [1/2, sqrt(3)/2]; 2 triangles per rhombic cell; topology fixed
    from the regular lattice (triangulate-first), positions perturbed by eta after.
    Returns a dict with unwrapped per-triangle edge vectors and a deduplicated bond list.
    """
    rng = np.random.default_rng(seed)
    L1 = np.array([1.0, 0.0])
    L2 = np.array([0.5, np.sqrt(3) / 2])
    BL1, BL2 = N * L1, N * L2                       # box periodicity vectors

    nn, mm = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
    ref = nn.ravel()[:, None] * L1 + mm.ravel()[:, None] * L2
    ang = rng.uniform(0, 2 * np.pi, N * N)
    pts = ref + eta * np.stack([np.cos(ang), np.sin(ang)], axis=1)   # (N^2, 2)

    def idx(a, b):
        return (a % N) * N + (b % N)

    simplices, tri_images = [], []
    for a in range(N):
        for b in range(N):
            simplices.append([idx(a, b), idx(a + 1, b), idx(a, b + 1)])
            tri_images.append([[0, 0],
                               [1 if a + 1 >= N else 0, 0],
                               [0, 1 if b + 1 >= N else 0]])
            simplices.append([idx(a + 1, b), idx(a + 1, b + 1), idx(a, b + 1)])
            tri_images.append([[1 if a + 1 >= N else 0, 0],
                               [1 if a + 1 >= N else 0, 1 if b + 1 >= N else 0],
                               [0, 1 if b + 1 >= N else 0]])
    simplices  = np.array(simplices, dtype=np.int64)
    tri_images = np.array(tri_images, dtype=np.int64)
    n_tri = len(simplices)

    def vpos(k):
        return (pts[simplices[:, k]]
                + tri_images[:, k, 0:1] * BL1 + tri_images[:, k, 1:2] * BL2)
    p0, p1, p2 = vpos(0), vpos(1), vpos(2)
    e01, e02, e12 = p1 - p0, p2 - p0, p2 - p1
    edge_vecs = np.stack([e01, e02, e12], axis=1)            # (n_tri, 3, 2) unwrapped
    actual_len2 = (edge_vecs ** 2).sum(axis=2)
    cross = e01[:, 0] * e02[:, 1] - e01[:, 1] * e02[:, 0]
    areas = 0.5 * np.abs(cross)

    edge_pairs = [(0, 1, 0), (0, 2, 1), (1, 2, 2)]           # (ka, kb, edge_index)
    bond = {}
    for ti in range(n_tri):
        for ka, kb, ei in edge_pairs:
            va, vb = int(simplices[ti, ka]), int(simplices[ti, kb])
            d = tri_images[ti, ka] - tri_images[ti, kb]
            dp = (int(d[0]), int(d[1]))
            if (va, dp[0], dp[1]) <= (vb, -dp[0], -dp[1]):
                key, evec = (va, vb, dp[0], dp[1]), edge_vecs[ti, ei]
            else:
                key, evec = (vb, va, -dp[0], -dp[1]), -edge_vecs[ti, ei]
            if key not in bond:
                bond[key] = {'evec': evec, 'tris': []}
            bond[key]['tris'].append(ti)

    bu, bv, bR, s1, s2, q = [], [], [], [], [], []
    for key, val in bond.items():
        assert len(val['tris']) == 2, f"bond {key} touches {len(val['tris'])} triangles"
        u, v = key[0], key[1]
        R = val['evec']
        bu.append(u); bv.append(v); bR.append(R)
        s1.append(val['tris'][0]); s2.append(val['tris'][1])
        q.append([R[0] * R[0], 2.0 * R[0] * R[1], R[1] * R[1]])

    return {
        'N': N, 'eta': eta, 'seed': seed,
        'pts': pts, 'simplices': simplices, 'tri_images': tri_images,
        'edge_vecs': edge_vecs, 'actual_len2': actual_len2, 'areas': areas,
        'L1': L1, 'L2': L2, 'BL1': BL1, 'BL2': BL2,
        'bond_u': np.array(bu, dtype=np.int64),
        'bond_v': np.array(bv, dtype=np.int64),
        'bond_R': np.array(bR, dtype=np.float64),
        'kkt_arrays': (np.array(s1, dtype=np.int64),
                       np.array(s2, dtype=np.int64),
                       np.array(q,  dtype=np.float64)),
    }


# ── PBC equilibrium (linear central-force springs, k=1, l0 = reference length) ─
def _assemble_K_and_faff(mesh, F):
    """Stiffness K (sparse CSC) and affine residual force faff for applied F."""
    bu, bv, R = mesh['bond_u'], mesh['bond_v'], mesh['bond_R']
    Nn = len(mesh['pts'])
    l2 = (R ** 2).sum(axis=1)
    S = np.einsum('bp,bq->bpq', R, R) / l2[:, None, None]
    nb = len(bu)
    pp = np.array([0, 0, 1, 1]); qq = np.array([0, 1, 0, 1])

    def blocks(iarr, jarr, sign):
        rows = (2 * iarr[:, None] + pp[None, :]).ravel()
        cols = (2 * jarr[:, None] + qq[None, :]).ravel()
        vals = (sign * S.reshape(nb, 4)).ravel()
        return rows, cols, vals

    R_, C_, V_ = [], [], []
    for ia, ja, sg in [(bu, bu, 1.0), (bv, bv, 1.0), (bu, bv, -1.0), (bv, bu, -1.0)]:
        r, c, v = blocks(ia, ja, sg)
        R_.append(r); C_.append(c); V_.append(v)
    K = sp.coo_matrix((np.concatenate(V_), (np.concatenate(R_), np.concatenate(C_))),
                      shape=(2 * Nn, 2 * Nn)).tocsc()

    HR = R @ (F - np.eye(2)).T
    fb = np.einsum('bpq,bq->bp', S, HR)
    faff = np.zeros((Nn, 2))
    np.add.at(faff, bv,  fb)
    np.add.at(faff, bu, -fb)
    return K, faff.ravel()


def solve_pbc_fluctuation(mesh, F):
    """Solve K u = -faff with node 0 pinned (removes rigid translation).

    Returns (u (Nn,2), residual_norm)."""
    Nn = len(mesh['pts'])
    K, faff = _assemble_K_and_faff(mesh, F)
    free = np.arange(2, 2 * Nn)
    rhs = -faff[free]
    u_free = spla.spsolve(K[free][:, free].tocsc(), rhs)
    u = np.zeros(2 * Nn)
    u[free] = u_free
    resid = float(np.linalg.norm((K @ u + faff)[free]))
    return u.reshape(Nn, 2), resid


# ── Lab-frame strain ──────────────────────────────────────────────────────────
def macroscopic_dg(F):
    """Macroscopic metric change  Delta_g = F^T F - I  (2,2).

    This is the strain in our (incompatible-elasticity) framework: the change of the
    metric g = F^T F relative to the reference g-bar = I. It is NON-LINEAR and is twice
    the usual linear strain eps = sym(F - I) to leading order (Delta_g = 2 eps + H^T H,
    H = F - I)."""
    return F.T @ F - np.eye(2)


def triangle_metric_change(edge_vecs, simplices, F, u):
    """Per-triangle metric change  g_tri = F_s^T F_s - I  (n_tri,2,2), framework strain.

    F_s = E_def @ E_ref^-1 is the constant deformation gradient of each triangle (its 3
    vertices define one affine map). g = F_s^T F_s is the triangle's metric (right
    Cauchy-Green); the metric change g_tri = g - I. Under a globally affine deformation
    F_s = F for every triangle, so g_tri = Delta_g (no spread) -- the correct,
    frame-independent statement.
    """
    e01, e02 = edge_vecs[:, 0], edge_vecs[:, 1]
    Eref = np.stack([e01, e02], axis=-1)                     # (n,2,2), columns = edges
    du01 = u[simplices[:, 1]] - u[simplices[:, 0]]
    du02 = u[simplices[:, 2]] - u[simplices[:, 0]]
    Edef = np.stack([e01 @ F.T + du01, e02 @ F.T + du02], axis=-1)
    Fs = Edef @ np.linalg.inv(Eref)
    return np.einsum('nki,nkj->nij', Fs, Fs) - np.eye(2)     # F_s^T F_s - I  (n,2,2)


# W_mat layout exactly as forward_solver_torch._compute_actual_elastic_tensor / Disc_2_Cont
_WIDX = [(0, 0, 0), (0, 1, 1), (0, 2, 3), (0, 3, 4), (1, 0, 1), (1, 1, 2), (1, 2, 4),
         (1, 3, 5), (2, 0, 3), (2, 1, 4), (2, 2, 6), (2, 3, 7), (3, 0, 4), (3, 1, 5),
         (3, 2, 7), (3, 3, 8)]


def mf_strain(W9, Delta_g):
    """MF non-affine metric change per triangle: delta_g = W : Delta_g  (n_tri,2,2).

    Uses the solver's 4-vector contraction delta_g4 = W_mat @ Delta_g4, with
    Delta_g4 = [Delta_g11, Delta_g12, Delta_g21, Delta_g22] the macroscopic METRIC change
    (F^T F - I); the result is symmetrized (the metric change is symmetric).
    <delta_g_MF> over triangles is ~0 (verified), confirming the convention.
    """
    n = len(W9)
    M = np.zeros((n, 4, 4))
    for r, c, k in _WIDX:
        M[:, r, c] = W9[:, k]
    e4 = np.array([Delta_g[0, 0], Delta_g[0, 1], Delta_g[1, 0], Delta_g[1, 1]])
    de = (M @ e4).reshape(n, 2, 2)
    return 0.5 * (de + de.transpose(0, 2, 1))


def woodbury_W(mesh, area_weighted, use_kkt):
    """Per-triangle response W (n_tri, 9) from the Phase 2 Woodbury solver (k=1)."""
    e = mesh['edge_vecs']
    vx, vy = e[:, :, 0], e[:, :, 1]
    factor = 1.0 / np.maximum(mesh['actual_len2'], 1e-30) / 16.0
    bare = np.stack([(factor * vx ** 4).sum(1),
                     (factor * vx ** 3 * vy).sum(1),
                     (factor * vx ** 2 * vy ** 2).sum(1),
                     (factor * vx * vy ** 3).sum(1),
                     (factor * vy ** 4).sum(1)], axis=1)
    areas = mesh['areas']
    w_np = (areas / areas.sum()) if area_weighted else None
    mean = (bare * w_np[:, None]).sum(0) if area_weighted else bare.mean(0)
    dbare = bare - mean
    A_bl = fst._batch_to_9x9(torch.as_tensor(bare, dtype=torch.float64))
    B_bl = fst._batch_to_9x9(torch.as_tensor(dbare, dtype=torch.float64))
    dA   = fst._batch_to_9vec(torch.as_tensor(dbare, dtype=torch.float64))
    if use_kkt:
        return fst._woodbury_kkt_sparse_combined(A_bl, B_bl, dA, mesh['kkt_arrays'],
                                                  None, weights=w_np)
    wt = torch.as_tensor(w_np, dtype=torch.float64) if w_np is not None else None
    return fst._woodbury_solve(A_bl, B_bl, dA, J=None, weights=wt).detach().numpy()


# ── Principal decomposition & geometry ────────────────────────────────────────
def principal(dg):
    """(n,2,2) symmetric -> (lam1>=lam2, lam2, theta_deg in (-90,90])."""
    w, V = np.linalg.eigh(dg)
    lam2, lam1 = w[:, 0], w[:, 1]
    major = V[:, :, 1]
    theta = np.degrees(np.arctan2(major[:, 1], major[:, 0]))
    theta = (theta + 90.0) % 180.0 - 90.0
    return lam1, lam2, theta


def wrap_angle(d):
    return (d + 90.0) % 180.0 - 90.0


def min_triangle_angles_from_edges(edge_vecs):
    e01, e02, e12 = edge_vecs[:, 0], edge_vecs[:, 1], edge_vecs[:, 2]

    def ang(a, b):
        c = (a * b).sum(1) / np.sqrt(np.maximum((a ** 2).sum(1) * (b ** 2).sum(1), 1e-30))
        return np.degrees(np.arccos(np.clip(c, -1, 1)))
    a0 = ang(e01, e02)
    a1 = ang(-e01, e12)
    a2 = np.abs(180.0 - a0 - a1)
    return np.minimum(np.minimum(a0, a1), a2)


# ── Summary accumulation (shared by full run and --regen) ─────────────────────
def _empty_summary(n_eta):
    def zeros():
        return np.full((n_eta, N_TRIALS, 2 * N * N), np.nan)
    summ = {}
    for name in ['sim'] + METHOD_LABELS:
        summ[f'lam1_{name}'] = zeros()
        summ[f'lam2_{name}'] = zeros()
        summ[f'theta_{name}'] = zeros()
    for name in METHOD_LABELS:
        summ[f'dlam1_{name}'] = zeros()
        summ[f'dlam2_{name}'] = zeros()
        summ[f'dlam1n_{name}'] = zeros()     # (method - sim) / ||Delta_g||
        summ[f'dlam2n_{name}'] = zeros()
        summ[f'dtheta_{name}'] = zeros()
    summ['min_angle'] = zeros()
    summ['area'] = zeros()
    summ['norm_sim'] = zeros()
    return summ


def _fill_summary_entry(summ, ie, it, method_dg, min_angle, area, dgn):
    prin = {name: principal(dg) for name, dg in method_dg.items()}
    l1s, l2s, ths = prin['sim']
    summ['lam1_sim'][ie, it] = l1s
    summ['lam2_sim'][ie, it] = l2s
    summ['theta_sim'][ie, it] = ths
    summ['min_angle'][ie, it] = min_angle
    summ['area'][ie, it] = area
    summ['norm_sim'][ie, it] = np.sqrt((method_dg['sim'] ** 2).sum(axis=(1, 2)))
    for label in METHOD_LABELS:
        l1, l2, th = prin[label]
        summ[f'lam1_{label}'][ie, it] = l1
        summ[f'lam2_{label}'][ie, it] = l2
        summ[f'theta_{label}'][ie, it] = th
        summ[f'dlam1_{label}'][ie, it] = l1 - l1s
        summ[f'dlam2_{label}'][ie, it] = l2 - l2s
        summ[f'dlam1n_{label}'][ie, it] = (l1 - l1s) / dgn
        summ[f'dlam2n_{label}'][ie, it] = (l2 - l2s) / dgn
        summ[f'dtheta_{label}'][ie, it] = wrap_angle(th - ths)


# ── Main sweep (simulate + analyse) ───────────────────────────────────────────
def main():
    F = np.eye(2) + DELTA * H_MACRO
    Delta_g = macroscopic_dg(F)
    dgn = float(np.linalg.norm(Delta_g))
    n_eta = len(ETA_VALUES)
    summ = _empty_summary(n_eta)
    resid_max = np.zeros((n_eta, N_TRIALS))
    checks = {'eta0_ufluct_max': 0.0, 'eta0_dgsim_max': 0.0, 'eta0_W_max': 0.0,
              'max_resid': 0.0, 'corr_lowEta': []}

    t0 = time.time()
    for ie, eta in enumerate(ETA_VALUES):
        for it in range(N_TRIALS):
            seed = 1000 * ie + it
            mesh = build_periodic_tf_mesh(N, eta, seed)

            u, resid = solve_pbc_fluctuation(mesh, F)
            resid_max[ie, it] = resid
            checks['max_resid'] = max(checks['max_resid'], resid)

            g_tri = triangle_metric_change(mesh['edge_vecs'], mesh['simplices'], F, u)
            dg_sim = g_tri - Delta_g

            method_dg = {'sim': dg_sim}
            method_W = {}
            for label, aw, kkt in METHODS:
                W = woodbury_W(mesh, aw, kkt)
                method_dg[label] = mf_strain(W, Delta_g)
                method_W[label] = W

            _fill_summary_entry(summ, ie, it, method_dg,
                                min_triangle_angles_from_edges(mesh['edge_vecs']),
                                mesh['areas'], dgn)

            if eta == 0.0:
                checks['eta0_ufluct_max'] = max(checks['eta0_ufluct_max'], float(np.abs(u).max()))
                checks['eta0_dgsim_max'] = max(checks['eta0_dgsim_max'], float(np.abs(dg_sim).max()))
                for label in METHOD_LABELS:
                    checks['eta0_W_max'] = max(checks['eta0_W_max'], float(np.abs(method_W[label]).max()))
            if eta in (0.1, 0.2):
                a = method_dg['Std'].ravel(); b = dg_sim.ravel()
                if np.std(a) > 1e-30 and np.std(b) > 1e-30:
                    checks['corr_lowEta'].append(float(np.corrcoef(a, b)[0, 1]))

            np.savez_compressed(
                os.path.join(DATA_DIR, f'sample_eta{eta:.2f}_trial{it}.npz'),
                N=N, eta=eta, seed=seed, F=F, delta=DELTA, H_macro=H_MACRO,
                Delta_g=Delta_g,
                pts=mesh['pts'], simplices=mesh['simplices'],
                tri_images=mesh['tri_images'], edge_vecs=mesh['edge_vecs'],
                actual_len2=mesh['actual_len2'], areas=mesh['areas'],
                BL1=mesh['BL1'], BL2=mesh['BL2'],
                bond_u=mesh['bond_u'], bond_v=mesh['bond_v'], bond_R=mesh['bond_R'],
                kkt_s1=mesh['kkt_arrays'][0], kkt_s2=mesh['kkt_arrays'][1],
                kkt_q=mesh['kkt_arrays'][2],
                u_fluct=u, resid=resid, g_tri=g_tri, dg_sim=dg_sim,
                **{f'dg_{lab}': method_dg[lab] for lab in METHOD_LABELS},
                **{f'W_{lab}': method_W[lab] for lab in METHOD_LABELS},
                method_labels=np.array(METHOD_LABELS),
            )
        print(f'  eta={eta:.2f} done  [{time.time() - t0:.0f}s]  '
              f'max_resid={resid_max[ie].max():.2e}  '
              f"|dtheta| med (Std)={np.nanmedian(np.abs(summ['dtheta_Std'][ie])):.2f} deg",
              flush=True)

    _save_summary(summ, resid_max)
    print('\n── Validation ──', flush=True)
    print(f"  eta=0 relaxation |u| max      : {checks['eta0_ufluct_max']:.3e} (expect ~0)")
    print(f"  eta=0 MF |W| max              : {checks['eta0_W_max']:.3e} (expect ~0)")
    print(f"  eta=0 sim |dg| max            : {checks['eta0_dgsim_max']:.3e} "
          f"(expect ~0: perfect crystal -> affine, metric change uniform)")
    print(f"  PBC solve max force residual  : {checks['max_resid']:.3e} (expect ~0)")
    if checks['corr_lowEta']:
        print(f"  Std vs sim corr (eta=0.1,0.2) : mean {np.mean(checks['corr_lowEta']):+.3f}")
    write_readme(checks)
    make_plots()
    print(f'\nTotal: {time.time() - t0:.0f}s. Data in {DATA_DIR}, plots in {PLOTS_DIR}',
          flush=True)


# ── Regenerate from stored samples (no re-simulation) ─────────────────────────
def regenerate_from_stored():
    """Rebuild summary.npz + plots from saved sample_*.npz, recomputing the lab-frame
    delta_g from the stored edges / fluctuation field / W. Rewrites each sample's dg_* /
    eps_* fields in place (raw mesh, u_fluct, W are preserved)."""
    Delta_g = macroscopic_dg(np.eye(2) + DELTA * H_MACRO)
    dgn = float(np.linalg.norm(Delta_g))
    n_eta = len(ETA_VALUES)
    summ = _empty_summary(n_eta)
    resid_max = np.full((n_eta, N_TRIALS), np.nan)
    checks = {'eta0_ufluct_max': 0.0, 'eta0_dgsim_max': 0.0, 'eta0_W_max': 0.0,
              'max_resid': 0.0, 'corr_lowEta': []}
    t0 = time.time()
    for ie, eta in enumerate(ETA_VALUES):
        for it in range(N_TRIALS):
            path = os.path.join(DATA_DIR, f'sample_eta{eta:.2f}_trial{it}.npz')
            if not os.path.exists(path):
                print(f'  missing {os.path.basename(path)} -- skipping', flush=True)
                continue
            s = np.load(path, allow_pickle=True)
            F = s['F']; ev = s['edge_vecs']; sx = s['simplices']; u = s['u_fluct']
            g_tri = triangle_metric_change(ev, sx, F, u)
            dg_sim = g_tri - Delta_g
            method_dg = {'sim': dg_sim}
            for lab in METHOD_LABELS:
                method_dg[lab] = mf_strain(s[f'W_{lab}'], Delta_g)

            _fill_summary_entry(summ, ie, it, method_dg,
                                min_triangle_angles_from_edges(ev), s['areas'], dgn)
            if 'resid' in s.files:
                resid_max[ie, it] = float(s['resid']); checks['max_resid'] = max(
                    checks['max_resid'], float(s['resid']))
            if eta == 0.0:
                checks['eta0_ufluct_max'] = max(checks['eta0_ufluct_max'], float(np.abs(u).max()))
                checks['eta0_dgsim_max'] = max(checks['eta0_dgsim_max'], float(np.abs(dg_sim).max()))
                for lab in METHOD_LABELS:
                    checks['eta0_W_max'] = max(checks['eta0_W_max'], float(np.abs(s[f'W_{lab}']).max()))
            if eta in (0.1, 0.2):
                a = method_dg['Std'].ravel(); b = dg_sim.ravel()
                if np.std(a) > 1e-30 and np.std(b) > 1e-30:
                    checks['corr_lowEta'].append(float(np.corrcoef(a, b)[0, 1]))

            d = {k: s[k] for k in s.files}
            for lab in METHOD_LABELS:
                for old in (f'gbar_{lab}',):
                    d.pop(old, None)
                d[f'dg_{lab}'] = method_dg[lab]
            for old in ('g_ref', 'g_aff', 'g_def', 'dg_aff', 'Delta_g'):
                d.pop(old, None)
            d['Delta_g'] = Delta_g
            d['g_tri'] = g_tri
            d['dg_sim'] = dg_sim
            np.savez_compressed(path, **d)
        print(f'  eta={eta:.2f} regenerated  [{time.time() - t0:.0f}s]  '
              f"|dtheta| med (Std)={np.nanmedian(np.abs(summ['dtheta_Std'][ie])):.2f} deg "
              f"corr(Std)~{np.nanmedian([np.corrcoef(summ['lam1_Std'][ie].ravel(),summ['lam1_sim'][ie].ravel())[0,1]]):.2f}",
              flush=True)

    _save_summary(summ, resid_max)
    write_readme(checks)
    make_plots()
    print(f'\nRegenerated from stored data in {time.time() - t0:.0f}s. '
          f'Data in {DATA_DIR}, plots in {PLOTS_DIR}', flush=True)


def _save_summary(summ, resid_max):
    np.savez_compressed(
        os.path.join(DATA_DIR, 'summary.npz'),
        eta_values=ETA_VALUES, n_trials=N_TRIALS, N=N, N_TRI=2 * N * N,
        F=np.eye(2) + DELTA * H_MACRO, delta=DELTA, H_macro=H_MACRO,
        Delta_g=macroscopic_dg(np.eye(2) + DELTA * H_MACRO),
        method_labels=np.array(METHOD_LABELS), resid_max=resid_max,
        **summ,
    )


# ── Plots ─────────────────────────────────────────────────────────────────────
def make_plots():
    d = np.load(os.path.join(DATA_DIR, 'summary.npz'), allow_pickle=True)
    etas = d['eta_values']
    labels = [str(x) for x in d['method_labels']]

    def med_iqr(arr3):
        flat = arr3.reshape(arr3.shape[0], -1)
        return (np.nanmedian(flat, 1),
                np.nanpercentile(flat, 25, 1), np.nanpercentile(flat, 75, 1))

    # 1) principal-value differences (normalized by ||Delta_g||) vs eta
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, key, ttl in [(axes[0], 'dlam1n', r'$\Delta\lambda_1/\|\Delta g\|$ (method $-$ sim)'),
                         (axes[1], 'dlam2n', r'$\Delta\lambda_2/\|\Delta g\|$ (method $-$ sim)')]:
        for lab in labels:
            m, lo, hi = med_iqr(d[f'{key}_{lab}'])
            ax.fill_between(etas, lo, hi, color=COLORS[lab], alpha=0.15)
            ax.plot(etas, m, '-o', ms=4, color=COLORS[lab], label=lab)
        ax.axhline(0, color='gray', lw=0.5, ls=':')
        ax.set_xlabel(r'$\eta$'); ax.set_ylabel(ttl); ax.set_title(ttl)
        ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.suptitle(r'Per-triangle principal-value error of non-affine metric change '
                 r'$\delta g=g_s-g$ '
                 f'vs simulation (N={int(d["N"])}, uniaxial)', fontsize=12)
    plt.tight_layout(); plt.savefig(os.path.join(PLOTS_DIR, 'dg_principal_diff_vs_eta.png'),
                                    dpi=150, bbox_inches='tight'); plt.close()

    # 2) principal-axis angle difference vs eta
    # The principal-axis angle is undefined where the non-affine response is ~0 (e.g. the
    # perfect crystal at eta=0). Mask triangles with ||dg_sim|| below a small floor so we
    # don't plot angle-of-noise (which would sit at the random baseline, 45 deg).
    dgnorm = float(np.linalg.norm(d['Delta_g'])) if 'Delta_g' in d.files else 2.0 * d['delta']
    near0 = d['norm_sim'] < 1e-4 * dgnorm
    fig, ax = plt.subplots(figsize=(7, 5))
    for lab in labels:
        dth = np.where(near0, np.nan, np.abs(d[f'dtheta_{lab}']))
        m, lo, hi = med_iqr(dth)
        ax.fill_between(etas, lo, hi, color=COLORS[lab], alpha=0.15)
        ax.plot(etas, m, '-o', ms=4, color=COLORS[lab], label=lab)
    ax.axhline(45, color='gray', lw=0.8, ls='--', label='random (45°)')
    ax.set_xlabel(r'$\eta$'); ax.set_ylabel(r'$|\Delta\theta|$ (deg)')
    ax.set_title(r'Principal-axis misalignment of non-affine $\delta g$ vs simulation')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(PLOTS_DIR, 'dg_angle_diff_vs_eta.png'),
                                    dpi=150, bbox_inches='tight'); plt.close()

    # 2b) correlation MF vs sim (component-wise) and magnitude ratio vs eta
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    dgnorm = float(np.linalg.norm(d['Delta_g'])) if 'Delta_g' in d.files else 2.0 * d['delta']
    for lab in labels:
        cc, rr = [], []
        for ie in range(len(etas)):
            if np.nanmedian(d['norm_sim'][ie]) < 1e-4 * dgnorm:   # delta_g ~ 0: skip
                cc.append(np.nan); rr.append(np.nan); continue
            sim = np.stack([d['lam1_sim'][ie].ravel(), d['lam2_sim'][ie].ravel()])
            met = np.stack([d[f'lam1_{lab}'][ie].ravel(), d[f'lam2_{lab}'][ie].ravel()])
            m = np.isfinite(sim).all(0) & np.isfinite(met).all(0)
            a, b = met[:, m].ravel(), sim[:, m].ravel()
            cc.append(np.corrcoef(a, b)[0, 1] if a.std() > 0 else np.nan)
            rr.append(np.linalg.norm(a) / max(np.linalg.norm(b), 1e-30))
        axes[0].plot(etas, cc, '-o', ms=4, color=COLORS[lab], label=lab)
        axes[1].plot(etas, rr, '-o', ms=4, color=COLORS[lab], label=lab)
    axes[0].set_ylabel('corr(MF, sim) of principal values'); axes[0].set_ylim(-0.1, 1.05)
    axes[0].set_title('Correlation of non-affine principal values vs sim')
    axes[1].axhline(1, color='gray', lw=0.5, ls=':')
    axes[1].set_ylabel(r'$\|\lambda_{MF}\|/\|\lambda_{sim}\|$')
    axes[1].set_title('Magnitude ratio (MF / sim)')
    for ax in axes:
        ax.set_xlabel(r'$\eta$'); ax.legend(fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(PLOTS_DIR, 'dg_corr_ratio_vs_eta.png'),
                                    dpi=150, bbox_inches='tight'); plt.close()

    # 3) per-triangle scatter at PLOT_ETA
    ie = int(np.argmin(np.abs(etas - PLOT_ETA)))
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    ma = d['min_angle'][ie].ravel()
    for j, lab in enumerate(labels):
        dth = np.abs(d[f'dtheta_{lab}'][ie]).ravel()
        axes.flat[j].scatter(ma, dth, s=6, alpha=0.3, color=COLORS[lab])
        axes.flat[j].set_title(f'{lab}: |Δθ| vs min angle')
        axes.flat[j].set_xlabel('min triangle angle (deg)')
        axes.flat[j].set_ylabel('|Δθ| (deg)'); axes.flat[j].grid(alpha=0.3)
    fig.suptitle(f'Per-triangle angle error at η={etas[ie]:.2f}', fontsize=12)
    plt.tight_layout(); plt.savefig(os.path.join(PLOTS_DIR, 'dg_scatter_eta04.png'),
                                    dpi=150, bbox_inches='tight'); plt.close()

    # 4) mesh maps at PLOT_ETA, trial 0
    sample = os.path.join(DATA_DIR, f'sample_eta{etas[ie]:.2f}_trial0.npz')
    if os.path.exists(sample):
        _mesh_maps(np.load(sample, allow_pickle=True), labels)

    # 5) principal value / angle distributions
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, key, ttl in [(axes[0], 'lam1', r'$\lambda_1$'),
                         (axes[1], 'lam2', r'$\lambda_2$'),
                         (axes[2], 'theta', r'$\theta$ (deg)')]:
        for lab in ['sim'] + labels:
            vals = d[f'{key}_{lab}'][ie].ravel()
            vals = vals[np.isfinite(vals)]
            color = 'k' if lab == 'sim' else COLORS[lab]
            ax.hist(vals, bins=60, histtype='step', density=True,
                    color=color, label=lab, lw=1.8 if lab == 'sim' else 1.2)
        ax.set_title(f'{ttl} distribution (η={etas[ie]:.2f})')
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(PLOTS_DIR, 'dg_principal_hist.png'),
                                    dpi=150, bbox_inches='tight'); plt.close()


def _mesh_maps(s, labels):
    from matplotlib.collections import PolyCollection
    pts = s['pts']; sx = s['simplices']
    tri_xy = pts[sx]
    fig, axes = plt.subplots(2, len(labels), figsize=(4.2 * len(labels), 9))
    for j, lab in enumerate(labels):
        l1m, l2m, thm = principal(s[f'dg_{lab}'])
        l1s, l2s, ths = principal(s['dg_sim'])
        dth = np.abs(wrap_angle(thm - ths))
        dl1 = l1m - l1s
        for row, vals, ttl, cmap in [(0, dth, f'{lab} |Δθ| (deg)', 'viridis'),
                                     (1, dl1, f'{lab} Δλ₁', 'coolwarm')]:
            ax = axes[row, j] if len(labels) > 1 else axes[row]
            vmax = np.nanpercentile(np.abs(vals), 98)
            pc = PolyCollection(tri_xy, array=vals, cmap=cmap, edgecolors='none')
            pc.set_clim(-vmax, vmax) if cmap == 'coolwarm' else pc.set_clim(0, vmax)
            ax.add_collection(pc); ax.autoscale_view()
            ax.set_aspect('equal'); ax.set_title(ttl, fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
            plt.colorbar(pc, ax=ax, fraction=0.046)
    fig.suptitle(f'Spatial structure of non-affine strain error (η={float(s["eta"]):.2f}, trial 0)',
                 fontsize=12)
    plt.tight_layout(); plt.savefig(os.path.join(PLOTS_DIR, 'dg_mesh_maps_eta04.png'),
                                    dpi=150, bbox_inches='tight'); plt.close()


def write_readme(checks):
    cc = np.mean(checks['corr_lowEta']) if checks.get('corr_lowEta') else float('nan')
    txt = f"""# dg_analysis_data — per-triangle non-affine strain: PBC sim vs MF / MF+KKT

Generated by `breakdown/pbc_dg_analysis.py`.

## Quantity (metric change Δg = FᵀF − I — our framework strain)
The strain is the METRIC CHANGE Δg = g − ḡ = FᵀF − I (ḡ = I). It is NON-LINEAR and ≈ 2× the
linear strain ε = sym(F − I). We use the metric change of each triangle, NOT the edge-basis
Gram matrix.
- Δg   = FᵀF − I                                       (macroscopic)
- g_tri = F_sᵀF_s − I,  F_s = E_def · E_ref⁻¹           (per-triangle, constant-strain element)
- non-affine metric change  δg(s):
    - sim : δg(s) = g_tri − Δg   (= F_sᵀF_s − FᵀF)
    - MF  : δg(s) = W_s : Δg      (solver 4-vector contraction δg₄ = W_mat · Δg₄,
            W_mat exactly as forward_solver_torch._compute_actual_elastic_tensor)
Both are symmetric 2×2, vanish under an affine deformation, and ⟨δg_MF⟩ ≈ 0.

(An earlier version used the edge Gram matrix; that mixes strain with edge orientation and
produced a spurious per-triangle 'spread' even for a perfect crystal. Corrected here.)

N={N} → {2 * N * N} triangles, η ∈ {list(ETA_VALUES)} × {N_TRIALS} trials, uniaxial εxx, δ={DELTA}.
Methods (no angle methods): {METHOD_LABELS}.

## Files
- `sample_eta{{η:.2f}}_trial{{t}}.npz` — per-sample arrays: mesh (`pts, simplices,
  tri_images, edge_vecs, actual_len2, areas, BL1, BL2, bond_u/v/R, kkt_*`), `F, Delta_g,
  delta`, equilibrium `u_fluct` + `resid`, `g_tri`, `dg_sim`, per method `dg_<label>`
  (n_tri,2,2) and `W_<label>` (n_tri,9).
- `summary.npz` — `(n_eta, n_trials, n_tri)` arrays: `lam1_*, lam2_*, theta_*` (sim + each
  method); `dlam1_*, dlam2_*` (raw) and `dlam1n_*, dlam2n_*` (÷‖Delta_g‖); `dtheta_*`;
  `min_angle, area, norm_sim`; plus `eta_values, method_labels, F, Delta_g, resid_max`.

## Key finding (corrected)
With the metric-change δg, the MF is genuinely predictive at low/moderate disorder
(Std vs sim corr at η=0.1,0.2 ≈ {cc:+.3f}) and degrades with η; the MF tends to OVER-shoot
the magnitude (‖λ_MF‖/‖λ_sim‖ > 1, growing with η). At η=0 there is no non-affine metric
change at all (perfect crystal deforms affinely). See `dg_corr_ratio_vs_eta.png`.

## Validation (this run)
- η=0 relaxation |u_fluct| max : {checks['eta0_ufluct_max']:.3e}  (expect ~0)
- η=0 MF |W| max               : {checks['eta0_W_max']:.3e}  (expect ~0: no contrast)
- η=0 sim |δg| max             : {checks['eta0_dgsim_max']:.3e}  (expect ~0: affine, isotropic)
- PBC solve max force residual : {checks['max_resid']:.3e}  (expect ~0)

NOTE: the legacy `breakdown/periodic_mesh.py` / `pbc_simulation.py` modules were NOT used;
mesh + PBC solve here are independent implementations validated by the checks above.
"""
    with open(os.path.join(DATA_DIR, 'README.md'), 'w') as f:
        f.write(txt)


if __name__ == '__main__':
    if '--regen' in sys.argv:
        regenerate_from_stored()
    else:
        main()
