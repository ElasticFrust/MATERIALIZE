"""
Per-triangle non-affine metric change (delta-g) under a single macroscopic strain:
PBC spring-network simulation vs mean-field (Woodbury) and MF + edge-KKT.

TERMINOLOGY (agreed)
--------------------
  g-bar (gbar) : the REFERENCE metric -- the ideal/rest distances. Per triangle this is
                 its own rest metric `g_ref(s)` (the identity tensor in normalized units).
  g            : the actual TOTAL (macroscopic) metric  =  g-bar + Delta_g.
  Delta_g      : the GLOBAL strain  =  g - g-bar  (up to a factor 2). One tensor for the
                 whole sample; here the mean affine metric change <g_aff - g_ref>.
  g_s          : the LOCAL per-triangle metric (not a metric per se).
  delta_g_s    : the NON-AFFINE local deformation of triangle s
                 = g_s - g  =  g_s - g-bar - Delta_g.
  Linear response (small Delta_g):  delta_g_s = W_s . Delta_g.

So, per triangle:
  - MF / MF+KKT : delta_g(s) = W(s) @ Delta_g          (W from the Phase 2 Woodbury solve)
  - simulation  : delta_g(s) = (g_def(s) - g_ref(s)) - Delta_g
Both are referenced to the SAME single global affine field Delta_g (there is no
per-triangle "local affine").

The metric of a triangle is g = [[a.a, a.b],[a.b, b.b]] for its two basis edge vectors
a = e01, b = e02. g (and hence delta_g) is invariant to rigid translation/rotation, so it
is read straight off the deformed edge vectors -- no body frame needs to be fixed.

ANALYSIS
--------
Each per-triangle symmetric 2x2 `delta_g` is decomposed into principal values
(eigenvalues lam1 >= lam2) and principal-axis angle theta. We compare, per triangle, each
method against the simulation via principal-value differences and angle differences.

This script is SELF-CONTAINED: it does NOT import the (unvalidated) breakdown modules
`periodic_mesh.py` / `pbc_simulation.py`. It reuses only the Phase 2 core Woodbury
functions, which have existing test coverage. It does not modify the Phase 2 solver.

USAGE
-----
  python pbc_dg_analysis.py           # full run (simulate + analyse + plot)
  python pbc_dg_analysis.py --regen   # rebuild summary + plots from STORED samples only
                                       # (no re-simulation; applies the current delta_g
                                       #  definitions to the already-saved raw arrays)

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
            # T1: (a,b), (a+1,b), (a,b+1)
            simplices.append([idx(a, b), idx(a + 1, b), idx(a, b + 1)])
            tri_images.append([[0, 0],
                               [1 if a + 1 >= N else 0, 0],
                               [0, 1 if b + 1 >= N else 0]])
            # T2: (a+1,b), (a+1,b+1), (a,b+1)
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

    # Deduplicate bonds on the torus (each bond shared by exactly 2 triangles).
    edge_pairs = [(0, 1, 0), (0, 2, 1), (1, 2, 2)]           # (ka, kb, edge_index)
    bond = {}
    for ti in range(n_tri):
        for ka, kb, ei in edge_pairs:
            va, vb = int(simplices[ti, ka]), int(simplices[ti, kb])
            d = tri_images[ti, ka] - tri_images[ti, kb]
            dp = (int(d[0]), int(d[1]))
            if (va, dp[0], dp[1]) <= (vb, -dp[0], -dp[1]):
                key, evec = (va, vb, dp[0], dp[1]), edge_vecs[ti, ei]      # u->v
            else:
                key, evec = (vb, va, -dp[0], -dp[1]), -edge_vecs[ti, ei]   # u->v
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
    """Stiffness K (sparse CSC) and affine residual force faff for applied F.

    Linearized spring energy:  E = 1/2 sum_b ((u_v - u_u + (F-I)R_b) . Rhat_b)^2
    Equilibrium:  K u = -faff,  faff_i = sum_{v=i} fb - sum_{u=i} fb,
    fb = (R (x) R / |R|^2) @ ((F-I) R).
    """
    bu, bv, R = mesh['bond_u'], mesh['bond_v'], mesh['bond_R']
    Nn = len(mesh['pts'])
    l2 = (R ** 2).sum(axis=1)
    S = np.einsum('bp,bq->bpq', R, R) / l2[:, None, None]      # (nb, 2, 2) = Rhat (x) Rhat
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

    HR = R @ (F - np.eye(2)).T                                # (F-I) R, per bond (nb, 2)
    fb = np.einsum('bpq,bq->bp', S, HR)                       # (nb, 2)
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


# ── Metric helpers ────────────────────────────────────────────────────────────
def _gram(a, b):
    """(n,2),(n,2) -> (n,2,2) symmetric Gram matrices [[a.a,a.b],[a.b,b.b]]."""
    g = np.empty((len(a), 2, 2))
    g[:, 0, 0] = (a * a).sum(1)
    g[:, 1, 1] = (b * b).sum(1)
    g[:, 0, 1] = g[:, 1, 0] = (a * b).sum(1)
    return g


def _to_eng(g):
    """(n,2,2) -> (n,3) engineering metric vector [g11, 2 g12, g22]."""
    return np.stack([g[:, 0, 0], 2.0 * g[:, 0, 1], g[:, 1, 1]], axis=1)


def _from_eng(v):
    """(n,3) [g11, 2 g12, g22] -> (n,2,2) symmetric tensor."""
    g = np.empty((len(v), 2, 2))
    g[:, 0, 0] = v[:, 0]
    g[:, 1, 1] = v[:, 2]
    g[:, 0, 1] = g[:, 1, 0] = 0.5 * v[:, 1]
    return g


def affine_dg(mesh, F):
    """Per-triangle affine metric change g_aff - g_ref (n_tri,2,2), and g_ref, g_aff."""
    e01, e02 = mesh['edge_vecs'][:, 0], mesh['edge_vecs'][:, 1]
    g_ref = _gram(e01, e02)
    a_aff, b_aff = e01 @ F.T, e02 @ F.T
    g_aff = _gram(a_aff, b_aff)
    return g_aff - g_ref, g_ref, g_aff


def global_strain(dg_aff):
    """Delta_g (3,) = global strain = mean (over triangles) affine metric change.

    Single tensor for the whole sample (engineering basis). Areas are near-uniform here,
    so unweighted vs area-weighted is negligible; we use the unweighted mean.
    """
    return _to_eng(dg_aff).mean(0)


def deformed_metric(mesh, F, u):
    """Per-triangle deformed local metric g_def (= g_s) from the equilibrium field u."""
    sx = mesh['simplices']
    e01, e02 = mesh['edge_vecs'][:, 0], mesh['edge_vecs'][:, 1]
    a_aff, b_aff = e01 @ F.T, e02 @ F.T
    du01 = u[sx[:, 1]] - u[sx[:, 0]]
    du02 = u[sx[:, 2]] - u[sx[:, 0]]
    return _gram(a_aff + du01, b_aff + du02)


def sim_dg(g_def, g_ref, Delta_g):
    """Simulation non-affine delta_g = g_s - g = (g_def - g_ref) - Delta_g (n_tri,2,2)."""
    Dg_t = _from_eng(Delta_g[None, :])[0]                    # (2,2)
    return (g_def - g_ref) - Dg_t[None, :, :]


# ── Mean-field non-affine delta-g ─────────────────────────────────────────────
def mf_dg(mesh, Delta_g, area_weighted, use_kkt):
    """MF non-affine delta_g per triangle (n_tri,2,2) and the response W (n_tri,9).

    delta_g_MF(s) = W(s) @ Delta_g, with Delta_g the global strain (engineering 3-vec).
    """
    e = mesh['edge_vecs']
    vx, vy = e[:, :, 0], e[:, :, 1]
    factor = 1.0 / np.maximum(mesh['actual_len2'], 1e-30) / 16.0      # k = 1
    bare = np.stack([(factor * vx ** 4).sum(1),
                     (factor * vx ** 3 * vy).sum(1),
                     (factor * vx ** 2 * vy ** 2).sum(1),
                     (factor * vx * vy ** 3).sum(1),
                     (factor * vy ** 4).sum(1)], axis=1)               # (n_tri, 5)

    areas = mesh['areas']
    w_np = (areas / areas.sum()) if area_weighted else None
    mean = (bare * w_np[:, None]).sum(0) if area_weighted else bare.mean(0)
    dbare = bare - mean

    bt  = torch.as_tensor(bare,  dtype=torch.float64)
    dbt = torch.as_tensor(dbare, dtype=torch.float64)
    A_bl = fst._batch_to_9x9(bt)
    B_bl = fst._batch_to_9x9(dbt)
    dA   = fst._batch_to_9vec(dbt)

    if use_kkt:
        W = fst._woodbury_kkt_sparse_combined(A_bl, B_bl, dA, mesh['kkt_arrays'],
                                               None, weights=w_np)
    else:
        wt = torch.as_tensor(w_np, dtype=torch.float64) if w_np is not None else None
        W = fst._woodbury_solve(A_bl, B_bl, dA, J=None, weights=wt).detach().numpy()

    Wm = W.reshape(-1, 3, 3)   # Wm[s, i, k]: metric component i, loading mode k
    # delta_g_MF = W @ Delta_g. (Sign is the Phase 2 solver convention; note that with the
    # global-affine definition the per-triangle MF delta_g is ~uncorrelated with the
    # simulation -- see README -- so the sign cannot be fixed from this comparison.)
    dg_naff_eng = np.einsum('sik,k->si', Wm, Delta_g)                  # (n_tri, 3)
    return _from_eng(dg_naff_eng), W


# ── Principal decomposition & geometry ────────────────────────────────────────
def principal(dg):
    """(n,2,2) symmetric -> (lam1>=lam2, lam2, theta_deg in (-90,90])."""
    w, V = np.linalg.eigh(dg)                       # ascending eigenvalues
    lam2, lam1 = w[:, 0], w[:, 1]
    major = V[:, :, 1]                              # eigenvector of the larger eigenvalue
    theta = np.degrees(np.arctan2(major[:, 1], major[:, 0]))
    theta = (theta + 90.0) % 180.0 - 90.0
    return lam1, lam2, theta


def wrap_angle(d):
    """Wrap an angle difference (deg) into (-90, 90] (axes are defined mod 180)."""
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
    N_TRI = 2 * N * N

    def zeros():
        return np.full((n_eta, N_TRIALS, N_TRI), np.nan)
    summ = {}
    for name in ['sim'] + METHOD_LABELS:
        summ[f'lam1_{name}'] = zeros()
        summ[f'lam2_{name}'] = zeros()
        summ[f'theta_{name}'] = zeros()
    for name in METHOD_LABELS:
        summ[f'dlam1_{name}'] = zeros()     # method - sim
        summ[f'dlam2_{name}'] = zeros()
        summ[f'dtheta_{name}'] = zeros()    # wrapped
    summ['min_angle'] = zeros()
    summ['area'] = zeros()
    summ['norm_sim'] = zeros()              # ||delta_g_sim||_F per triangle
    return summ


def _fill_summary_entry(summ, ie, it, method_dg, min_angle, area):
    """Decompose each method's delta_g and store principal values / diffs vs sim."""
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
        summ[f'dtheta_{label}'][ie, it] = wrap_angle(th - ths)


# ── Main sweep (simulate + analyse) ───────────────────────────────────────────
def main():
    F = np.eye(2) + DELTA * H_MACRO
    n_eta = len(ETA_VALUES)
    summ = _empty_summary(n_eta)
    resid_max = np.zeros((n_eta, N_TRIALS))
    checks = {'eta0_ufluct_max': 0.0, 'eta0_dgsim_max': 0.0, 'eta0_W_max': 0.0,
              'max_resid': 0.0, 'sign_corr_lowEta': []}

    t0 = time.time()
    for ie, eta in enumerate(ETA_VALUES):
        for it in range(N_TRIALS):
            seed = 1000 * ie + it
            mesh = build_periodic_tf_mesh(N, eta, seed)

            dg_aff, g_ref, g_aff = affine_dg(mesh, F)
            Delta_g = global_strain(dg_aff)                  # (3,) single global strain

            u, resid = solve_pbc_fluctuation(mesh, F)
            resid_max[ie, it] = resid
            checks['max_resid'] = max(checks['max_resid'], resid)

            g_def = deformed_metric(mesh, F, u)
            dg_sim = sim_dg(g_def, g_ref, Delta_g)

            method_dg = {'sim': dg_sim}
            method_W = {}
            for label, aw, kkt in METHODS:
                dg_m, W = mf_dg(mesh, Delta_g, aw, kkt)
                method_dg[label] = dg_m
                method_W[label] = W

            _fill_summary_entry(summ, ie, it, method_dg,
                                min_triangle_angles_from_edges(mesh['edge_vecs']),
                                mesh['areas'])

            if eta == 0.0:
                checks['eta0_ufluct_max'] = max(
                    checks['eta0_ufluct_max'], float(np.abs(u).max()))
                checks['eta0_dgsim_max'] = max(
                    checks['eta0_dgsim_max'], float(np.abs(dg_sim).max()))
                for label in METHOD_LABELS:
                    checks['eta0_W_max'] = max(
                        checks['eta0_W_max'], float(np.abs(method_W[label]).max()))
            if eta in (0.1, 0.2):
                a = method_dg['Std'].ravel(); b = dg_sim.ravel()
                if np.std(a) > 1e-30 and np.std(b) > 1e-30:
                    checks['sign_corr_lowEta'].append(float(np.corrcoef(a, b)[0, 1]))

            np.savez_compressed(
                os.path.join(DATA_DIR, f'sample_eta{eta:.2f}_trial{it}.npz'),
                N=N, eta=eta, seed=seed, F=F, delta=DELTA, H_macro=H_MACRO,
                pts=mesh['pts'], simplices=mesh['simplices'],
                tri_images=mesh['tri_images'], edge_vecs=mesh['edge_vecs'],
                actual_len2=mesh['actual_len2'], areas=mesh['areas'],
                BL1=mesh['BL1'], BL2=mesh['BL2'],
                bond_u=mesh['bond_u'], bond_v=mesh['bond_v'], bond_R=mesh['bond_R'],
                kkt_s1=mesh['kkt_arrays'][0], kkt_s2=mesh['kkt_arrays'][1],
                kkt_q=mesh['kkt_arrays'][2],
                u_fluct=u, resid=resid,
                g_ref=g_ref, g_aff=g_aff, g_def=g_def, dg_aff=dg_aff,
                Delta_g=Delta_g, dg_sim=dg_sim,
                **{f'dg_{lab}': method_dg[lab] for lab in METHOD_LABELS},
                **{f'W_{lab}': method_W[lab] for lab in METHOD_LABELS},
                method_labels=np.array(METHOD_LABELS),
            )
        print(f'  eta={eta:.2f} done  [{time.time() - t0:.0f}s]  '
              f'max_resid={resid_max[ie].max():.2e}  '
              f"|dtheta| med (Std)="
              f"{np.nanmedian(np.abs(summ['dtheta_Std'][ie])):.2f} deg", flush=True)

    _save_summary(summ, resid_max)
    print('\n── Validation ──', flush=True)
    print(f"  eta=0 relaxation |u| max      : {checks['eta0_ufluct_max']:.3e} "
          f"(expect ~0: perfect crystal deforms affinely)")
    print(f"  eta=0 MF |W| max              : {checks['eta0_W_max']:.3e} "
          f"(expect ~0: no stiffness/geometry contrast)")
    print(f"  eta=0 sim |dg| max            : {checks['eta0_dgsim_max']:.3e} "
          f"(NONZERO and expected: 2 sublattice orientations strain differently; MF gives 0)")
    print(f"  PBC solve max force residual  : {checks['max_resid']:.3e} (expect ~0)")
    if checks['sign_corr_lowEta']:
        print(f"  Std vs sim corr (eta=0.1,0.2) : "
              f"mean {np.mean(checks['sign_corr_lowEta']):+.3f} "
              f"(diagnostic; ~0 under the global-affine definition)")
    write_readme(checks)
    make_plots()
    print(f'\nTotal: {time.time() - t0:.0f}s. Data in {DATA_DIR}, plots in {PLOTS_DIR}',
          flush=True)


# ── Regenerate from stored samples (no re-simulation) ─────────────────────────
def regenerate_from_stored():
    """Rebuild summary.npz + plots from already-saved sample_*.npz, applying the current
    delta_g definitions (delta_g = g_s - g = (g_def - g_ref) - Delta_g) to the stored raw
    arrays. Also rewrites each sample's `dg_*` / `Delta_g` fields in place (raw g_*/W/u are
    preserved, so nothing is lost)."""
    n_eta = len(ETA_VALUES)
    summ = _empty_summary(n_eta)
    resid_max = np.full((n_eta, N_TRIALS), np.nan)
    corr_low = []
    eta0 = {'u': 0.0, 'dg': 0.0, 'W': 0.0}
    t0 = time.time()
    for ie, eta in enumerate(ETA_VALUES):
        for it in range(N_TRIALS):
            path = os.path.join(DATA_DIR, f'sample_eta{eta:.2f}_trial{it}.npz')
            if not os.path.exists(path):
                print(f'  missing {os.path.basename(path)} -- skipping', flush=True)
                continue
            s = np.load(path, allow_pickle=True)
            g_ref, g_aff, g_def = s['g_ref'], s['g_aff'], s['g_def']
            Delta_g = global_strain(g_aff - g_ref)
            dg_sim = sim_dg(g_def, g_ref, Delta_g)

            method_dg = {'sim': dg_sim}
            for lab in METHOD_LABELS:
                Wm = s[f'W_{lab}'].reshape(-1, 3, 3)
                method_dg[lab] = _from_eng(np.einsum('sik,k->si', Wm, Delta_g))

            _fill_summary_entry(summ, ie, it, method_dg,
                                min_triangle_angles_from_edges(s['edge_vecs']),
                                s['areas'])
            if 'resid' in s.files:
                resid_max[ie, it] = float(s['resid'])
            if eta in (0.1, 0.2):
                a = method_dg['Std'].ravel(); b = dg_sim.ravel()
                if np.std(a) > 1e-30 and np.std(b) > 1e-30:
                    corr_low.append(float(np.corrcoef(a, b)[0, 1]))
            if eta == 0.0:
                if 'u_fluct' in s.files:
                    eta0['u'] = max(eta0['u'], float(np.abs(s['u_fluct']).max()))
                eta0['dg'] = max(eta0['dg'], float(np.abs(dg_sim).max()))
                for lab in METHOD_LABELS:
                    eta0['W'] = max(eta0['W'], float(np.abs(s[f'W_{lab}']).max()))

            # rewrite the sample with corrected delta_g (raw arrays preserved)
            d = {k: s[k] for k in s.files}
            for lab in METHOD_LABELS:
                d.pop(f'gbar_{lab}', None)          # drop legacy per-method gbar
                d[f'dg_{lab}'] = method_dg[lab]
            d['Delta_g'] = Delta_g
            d['dg_sim'] = dg_sim
            np.savez_compressed(path, **d)
        print(f'  eta={eta:.2f} regenerated  [{time.time() - t0:.0f}s]  '
              f"|dtheta| med (Std)={np.nanmedian(np.abs(summ['dtheta_Std'][ie])):.2f} deg",
              flush=True)

    _save_summary(summ, resid_max)
    checks = {'eta0_ufluct_max': eta0['u'], 'eta0_dgsim_max': eta0['dg'],
              'eta0_W_max': eta0['W'], 'max_resid': float(np.nanmax(resid_max)),
              'sign_corr_lowEta': corr_low}
    write_readme(checks)
    make_plots()
    print(f'\nRegenerated from stored data in {time.time() - t0:.0f}s. '
          f'Data in {DATA_DIR}, plots in {PLOTS_DIR}', flush=True)


def _save_summary(summ, resid_max):
    np.savez_compressed(
        os.path.join(DATA_DIR, 'summary.npz'),
        eta_values=ETA_VALUES, n_trials=N_TRIALS, N=N, N_TRI=2 * N * N,
        F=np.eye(2) + DELTA * H_MACRO, delta=DELTA, H_macro=H_MACRO,
        method_labels=np.array(METHOD_LABELS), resid_max=resid_max,
        **summ,
    )


# ── Plots ─────────────────────────────────────────────────────────────────────
def make_plots():
    d = np.load(os.path.join(DATA_DIR, 'summary.npz'), allow_pickle=True)
    etas = d['eta_values']
    labels = [str(x) for x in d['method_labels']]

    def med_iqr(arr3):                      # arr3: (n_eta, n_trials, N_TRI)
        flat = arr3.reshape(arr3.shape[0], -1)
        return (np.nanmedian(flat, 1),
                np.nanpercentile(flat, 25, 1), np.nanpercentile(flat, 75, 1))

    # 1) principal-value differences vs eta
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, key, ttl in [(axes[0], 'dlam1', r'$\Delta\lambda_1$ (method $-$ sim)'),
                         (axes[1], 'dlam2', r'$\Delta\lambda_2$ (method $-$ sim)')]:
        for lab in labels:
            m, lo, hi = med_iqr(d[f'{key}_{lab}'])
            ax.fill_between(etas, lo, hi, color=COLORS[lab], alpha=0.15)
            ax.plot(etas, m, '-o', ms=4, color=COLORS[lab], label=lab)
        ax.axhline(0, color='gray', lw=0.5, ls=':')
        ax.set_xlabel(r'$\eta$'); ax.set_ylabel(ttl); ax.set_title(ttl)
        ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.suptitle(r'Per-triangle principal-value error of non-affine $\delta g=g_s-g$ '
                 f'vs simulation (N={int(d["N"])}, uniaxial)', fontsize=12)
    plt.tight_layout(); plt.savefig(os.path.join(PLOTS_DIR, 'dg_principal_diff_vs_eta.png'),
                                    dpi=150, bbox_inches='tight'); plt.close()

    # 2) principal-axis angle difference vs eta
    fig, ax = plt.subplots(figsize=(7, 5))
    for lab in labels:
        m, lo, hi = med_iqr(np.abs(d[f'dtheta_{lab}']))
        ax.fill_between(etas, lo, hi, color=COLORS[lab], alpha=0.15)
        ax.plot(etas, m, '-o', ms=4, color=COLORS[lab], label=lab)
    ax.axhline(45, color='gray', lw=0.8, ls='--', label='random (45°)')
    ax.set_xlabel(r'$\eta$'); ax.set_ylabel(r'$|\Delta\theta|$ (deg)')
    ax.set_title(r'Principal-axis misalignment of non-affine $\delta g=g_s-g$ vs simulation')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(PLOTS_DIR, 'dg_angle_diff_vs_eta.png'),
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

    # 4) mesh maps at PLOT_ETA, trial 0 (|Δθ| and Δλ1)
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
    tri_xy = pts[sx]                                    # (n_tri, 3, 2)
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
    fig.suptitle(f'Spatial structure of δg error (η={float(s["eta"]):.2f}, trial 0)',
                 fontsize=12)
    plt.tight_layout(); plt.savefig(os.path.join(PLOTS_DIR, 'dg_mesh_maps_eta04.png'),
                                    dpi=150, bbox_inches='tight'); plt.close()


def write_readme(checks):
    sc = (np.mean(checks['sign_corr_lowEta'])
          if checks.get('sign_corr_lowEta') else float('nan'))
    txt = f"""# dg_analysis_data — per-triangle non-affine δg: PBC sim vs MF / MF+KKT

Generated by `breakdown/pbc_dg_analysis.py`.

## Terminology
- ḡ (g-bar) : REFERENCE metric (ideal/rest distances; per-triangle `g_ref`, identity in
  normalized units).
- g         : actual TOTAL macroscopic metric = ḡ + Δg.
- Δg        : GLOBAL strain = g − ḡ (one tensor for the sample; here the mean affine
  metric change ⟨g_aff − g_ref⟩, engineering basis). Stored as `Delta_g`.
- g_s       : LOCAL per-triangle metric.
- δg_s      : NON-AFFINE local deformation = g_s − g = (g_def − g_ref) − Δg.
  Linear response: δg_s = W_s · Δg.

## What this is
Per-triangle NON-AFFINE δg under a single macroscopic strain (uniaxial εxx, amplitude
δ={DELTA}), on periodic triangulate-first-then-deform meshes (N={N} → {2 * N * N}
triangles), for η ∈ {list(ETA_VALUES)} × {N_TRIALS} trials.
  - MF / MF+KKT : δg(s) = W(s) · Δg
  - simulation  : δg(s) = (g_def(s) − g_ref(s)) − Δg
Both referenced to the SAME single global affine field Δg (no per-triangle local affine).
Metric g = [[a·a,a·b],[a·b,b·b]] from edges a=e01, b=e02 (rotation/translation invariant).

Methods (no angle methods): {METHOD_LABELS}.

## Files
- `sample_eta{{η:.2f}}_trial{{t}}.npz` — full per-sample arrays:
  mesh (`pts, simplices, tri_images, edge_vecs, actual_len2, areas, BL1, BL2,
  bond_u, bond_v, bond_R, kkt_s1, kkt_s2, kkt_q`), applied `F`/`H_macro`/`delta`,
  the equilibrium fluctuation field `u_fluct` and its `resid`,
  per-triangle `g_ref, g_aff, g_def, dg_aff`, the global strain `Delta_g` (3,),
  `dg_sim` and per method `dg_<label>` (n_tri,2,2), `W_<label>` (n_tri,9).
  → W_sim for this strain is reconstructable from `u_fluct`+`F`+mesh; other strain
     modes only need re-solving on the stored mesh.
- `summary.npz` — stacked `(n_eta, n_trials, n_tri)` arrays:
  `lam1_*, lam2_*, theta_*` for sim and each method;
  `dlam1_*, dlam2_*, dtheta_*` (method − sim); `min_angle, area, norm_sim`;
  plus `eta_values, method_labels, F, resid_max`.

## Principal decomposition
δg (symmetric 2×2) → eigenvalues λ1≥λ2 and principal-axis angle θ∈(−90,90].
Comparison vs sim: Δλ1, Δλ2, Δθ (wrapped to (−90,90]).

## Key finding
With this (correct) global-affine definition, the MF per-triangle δg = W·Δg is
~uncorrelated with the simulation (Std-vs-sim corr at η=0.1,0.2 ≈ {sc:+.3f}). The MF
computes an exact per-triangle response but is too local — it does not encode the
inter-triangle compatibility/confluency constraints that couple the real network. That
coupling is the missing link.

## Validation (this run)
- η=0 relaxation |u_fluct| max : {checks.get('eta0_ufluct_max', float('nan')):.3e}  (expect ~0: a perfect crystal deforms affinely — no non-affine RELAXATION)
- η=0 MF |W| max               : {checks['eta0_W_max']:.3e}  (expect ~0: no stiffness/geometry contrast)
- η=0 sim |δg| max             : {checks.get('eta0_dgsim_max', float('nan')):.3e}  (NONZERO and expected: the 2 sublattice triangle orientations strain differently under uniaxial load, so g_s − g ≠ 0; the MF predicts δg=0 here)
- PBC solve max force residual : {checks['max_resid']:.3e}  (expect ~0)

NOTE: the `breakdown/periodic_mesh.py` / `pbc_simulation.py` modules were NOT used or
trusted; mesh + PBC solve here are independent implementations validated by the checks
above.
"""
    with open(os.path.join(DATA_DIR, 'README.md'), 'w') as f:
        f.write(txt)


if __name__ == '__main__':
    if '--regen' in sys.argv:
        regenerate_from_stored()
    else:
        main()
