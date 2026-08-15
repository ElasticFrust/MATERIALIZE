"""
Mesh construction for the forward solver — periodic and open triangulated spring networks.

Purpose: turn a lattice specification or a point cloud into the geometry dict the solver stack
consumes (`pts`, `simplices`, `edge_vecs`, `actual_len2`, `areas`, deduplicated `bond_*`, and the
triangle→bond map `tri_bond`), plus the interior-edge topology the intrinsic constrained solve
needs. Everything here is pure geometry/topology bookkeeping: no physics, no torch, no solver.

Implements the discrete setting of Grossman & Boudaoud, PRR 2026 (arXiv:2309.07844) §II: a planar
triangulated network of central-force springs, periodic (torus) or open (free boundary). See
`documentation/MATERIALIZE.md` §3 and `Phase 2/SOLVER_GUIDE.md` §1–2.

Conventions:
  - Per-triangle edges are ordered **(e01, e02, e12)** everywhere — the solver's order.
  - Periodic edge vectors are **unwrapped** (image-corrected), so a bond crossing the box has its
    true vector, not the wrapped-short one.
  - A bond is canonicalised by (min node, max node, image offset), so the two triangles sharing it
    agree on one index — this is what makes per-BOND design variables well defined.
  - float64 throughout.

Layering: **core-adjacent** — same layer as `forward_solver_torch.py`, depending on nothing but
NumPy (and SciPy only where a caller passes a `scipy.spatial` triangulation). Only
`forward_solver_torch.py` is the protected file, but changes here are gated by the same suite,
since `Phase 3/inverse_design.py`'s `DesignProblem` constructors are built directly on it.

History: lifted verbatim by the A-7b re-layering (`documentation/AUDIT_2026-08.md`) out of the
retireable oracle layer, which the design layer must not depend on —
`build_geometry`/`set_VD` from `verification_tools/test_cluster_VD.py`,
`clean_tri`/`build_open_mesh` from `verification_tools/verify_solver_open.py`,
`kkt_from_tri_bond` from `verification_tools/test_intrinsic_VD.py`.
"""
from types import SimpleNamespace

import numpy as np


# --------------------------------------------------------------------------- periodic (PBC)
def build_geometry(N, eta, seed):
    """Perturbed periodic triangular lattice; returns geometry + tri->bond map (no rigidity).

    An N×N rhombic torus (L1=[1,0], L2=[½,√3/2]), 2 triangles per cell. The TOPOLOGY is fixed from
    the perfect lattice and the node positions are then displaced by a random unit vector scaled by
    `eta` — i.e. frozen-connectivity, MAGNITUDE-η disorder, with no re-triangulation (the canonical
    disorder intent; see the project CLAUDE.md §3 "Disorder — two intents"). η < 0.5.

    Args:
        N: half-size — the lattice is N×N cells, so 2N² triangles.
        eta: disorder magnitude (0 = perfect crystal).
        seed: RNG seed; randomness is isolated to the `default_rng(seed)` here and nowhere else.
    Returns:
        geometry dict — `pts`, `simplices`, `edge_vecs` (unwrapped), `actual_len2`, `bond_u`,
        `bond_v`, `bond_R`, `tri_bond`, `areas`. Carries NO stiffness: call `set_VD` for that.
    """
    rng = np.random.default_rng(seed)
    L1 = np.array([1.0, 0.0]); L2 = np.array([0.5, np.sqrt(3)/2]); BL1, BL2 = N*L1, N*L2
    nn, mm = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
    ref = nn.ravel()[:, None]*L1 + mm.ravel()[:, None]*L2
    ang = rng.uniform(0, 2*np.pi, N*N)
    pts = ref + eta*np.stack([np.cos(ang), np.sin(ang)], 1)
    idx = lambda a, b: (a % N)*N + (b % N)
    simp, img = [], []
    for a in range(N):
        for b in range(N):
            simp.append([idx(a, b), idx(a+1, b), idx(a, b+1)])
            img.append([[0, 0], [1 if a+1 >= N else 0, 0], [0, 1 if b+1 >= N else 0]])
            simp.append([idx(a+1, b), idx(a+1, b+1), idx(a, b+1)])
            img.append([[1 if a+1 >= N else 0, 0],
                        [1 if a+1 >= N else 0, 1 if b+1 >= N else 0], [0, 1 if b+1 >= N else 0]])
    simp = np.array(simp, np.int64); img = np.array(img, np.int64); n_tri = len(simp)
    vpos = lambda k: pts[simp[:, k]] + img[:, k, 0:1]*BL1 + img[:, k, 1:2]*BL2
    p0, p1, p2 = vpos(0), vpos(1), vpos(2)
    edge_vecs = np.stack([p1-p0, p2-p0, p2-p1], 1); l2 = (edge_vecs**2).sum(2)
    pairs = [(0, 1, 0), (0, 2, 1), (1, 2, 2)]; keymap = {}; tri_bond = np.zeros((n_tri, 3), np.int64); bonds = []
    for ti in range(n_tri):
        for ka, kb, ei in pairs:
            va, vb = int(simp[ti, ka]), int(simp[ti, kb])
            d = img[ti, ka]-img[ti, kb]; dp = (int(d[0]), int(d[1]))
            if (va, dp[0], dp[1]) <= (vb, -dp[0], -dp[1]):
                key, R = (va, vb, dp[0], dp[1]), edge_vecs[ti, ei]
            else:
                key, R = (vb, va, -dp[0], -dp[1]), -edge_vecs[ti, ei]
            if key not in keymap:
                keymap[key] = len(bonds); bonds.append((key[0], key[1], R))
            tri_bond[ti, ei] = keymap[key]
    e01, e02 = edge_vecs[:, 0], edge_vecs[:, 1]
    return dict(N=N, pts=pts, simplices=simp, edge_vecs=edge_vecs, actual_len2=l2,
                bond_u=np.array([b[0] for b in bonds], np.int64),
                bond_v=np.array([b[1] for b in bonds], np.int64),
                bond_R=np.array([b[2] for b in bonds], float),
                tri_bond=tri_bond,
                areas=0.5*np.abs(e01[:, 0]*e02[:, 1] - e01[:, 1]*e02[:, 0]))


def set_VD(mesh, a):
    """Set virtual-distortion (VD) bond stiffness in place: k = 1 + tanh(a·(|R| − 1)).

    `a` = 0 gives uniform k = 1 (the usual "no rigidity contrast" initialisation, which is how the
    design layer uses it); a > 0 stiffens stretched bonds, a < 0 stiffens compressed ones, and
    |a| ~ 10 approaches binary k ∈ {0, 2}. Writes `bond_k` (per bond) and `tri_k` (per
    triangle-edge, scattered through `tri_bond`) — the two keys the rest of the stack reads.
    """
    dl = np.sqrt((mesh['bond_R']**2).sum(1)) - 1.0          # |R| - ideal spacing
    mesh['bond_k'] = 1.0 + np.tanh(a*dl)
    mesh['tri_k'] = mesh['bond_k'][mesh['tri_bond']]


def kkt_from_tri_bond(tri_bond, edge_vecs):
    """Interior-edge (s1,s2,q) arrays from the triangle->bond map (torus: every bond shared).

    These are the edge-compatibility constraint (C1) data of the intrinsic solve: for every edge
    shared by two triangles (s1, s2), the Voigt carrier q = [Δx², 2Δx Δy, Δy²] against which the
    two triangles' metric changes must agree. Bonds touched by only one triangle (a free boundary)
    contribute no constraint and are skipped, so this is correct for open meshes too.
    See `INTRINSIC_METRIC_SOLVE.md` and `Phase 2/SOLVER_GUIDE.md` §4.

    NB the factor 2 on the shear component here is the CONSTRAINT-side carrier convention, not the
    `[xx, xy, yy]` state convention of `metric_ops.vec3`.
    """
    occ = {}
    for ti in range(len(tri_bond)):
        for ei in range(3):
            occ.setdefault(int(tri_bond[ti, ei]), []).append((ti, ei))
    s1, s2, q = [], [], []
    for b, lst in occ.items():
        if len(lst) == 2:
            (t1, e1), (t2, e2) = lst
            v = edge_vecs[t1, e1]
            s1.append(t1); s2.append(t2); q.append([v[0]**2, 2*v[0]*v[1], v[1]**2])
    return np.array(s1, np.int64), np.array(s2, np.int64), np.array(q, float)


# --------------------------------------------------------------------------- open (free boundary)
def clean_tri(tri):
    """Drop points not referenced by simplices (generators keep the full cloud) and reindex,
    so no isolated nodes make the stiffness singular. Returns .points/.simplices namespace."""
    simp = np.asarray(tri.simplices, np.int64)
    used = np.unique(simp)
    remap = -np.ones(np.asarray(tri.points).shape[0], np.int64)
    remap[used] = np.arange(len(used))
    return SimpleNamespace(points=np.asarray(tri.points, float)[used], simplices=remap[simp])


def build_open_mesh(tri, vd_a=None):
    """clean triangulation -> non-periodic mesh dict (bonds = unique edges, no wrap).
    vd_a: if not None, per-bond k = 1 + tanh(vd_a*(|R|-1)); else uniform k=1."""
    pts  = np.asarray(tri.points, float)
    simp = np.asarray(tri.simplices, np.int64)
    nt = len(simp)
    p0, p1, p2 = pts[simp[:, 0]], pts[simp[:, 1]], pts[simp[:, 2]]
    edge_vecs = np.stack([p1 - p0, p2 - p0, p2 - p1], 1)               # edges (0,1),(0,2),(1,2)
    l2 = (edge_vecs ** 2).sum(2)
    areas = 0.5 * np.abs((p1 - p0)[:, 0] * (p2 - p0)[:, 1]
                         - (p1 - p0)[:, 1] * (p2 - p0)[:, 0])
    pairs = [(0, 1, 0), (0, 2, 1), (1, 2, 2)]
    keymap = {}; bu, bv, bR = [], [], []; tri_bond = np.zeros((nt, 3), np.int64)
    for ti in range(nt):
        for ka, kb, ei in pairs:
            va, vb = int(simp[ti, ka]), int(simp[ti, kb])
            key = (va, vb) if va < vb else (vb, va)
            if key not in keymap:
                keymap[key] = len(bu)
                bu.append(key[0]); bv.append(key[1]); bR.append(pts[key[1]] - pts[key[0]])
            tri_bond[ti, ei] = keymap[key]
    bu = np.array(bu, np.int64); bv = np.array(bv, np.int64); bR = np.array(bR, float)
    if vd_a is None:
        bond_k = np.ones(len(bu))
    else:
        bond_k = 1.0 + np.tanh(vd_a * (np.sqrt((bR ** 2).sum(1)) - 1.0))
    return dict(pts=pts, simplices=simp, edge_vecs=edge_vecs, actual_len2=l2, areas=areas,
                bond_u=bu, bond_v=bv, bond_R=bR, bond_k=bond_k, tri_bond=tri_bond,
                tri_k=bond_k[tri_bond])
