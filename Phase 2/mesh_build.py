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


def signed_areas(mesh):
    """Per-triangle SIGNED area. Negative => the triangle is INVERTED (folded).

    Uses the UNWRAPPED triangle vectors (`tri_verts` if present, else `edge_vecs`) — never the raw
    `pts`. Under PBC the wrapped coordinates fabricate inversions, and computing this from `pts` is
    exactly the mistake that made an earlier check dismiss real folding as a "Delaunay orientation
    convention" (audit A-17)."""
    tv = mesh.get('tri_verts')
    if tv is not None:
        tv = np.asarray(tv, float)
        a, b = tv[:, 1] - tv[:, 0], tv[:, 2] - tv[:, 0]
    else:
        ev = np.asarray(mesh['edge_vecs'], float)      # [p1-p0, p2-p0, p2-p1]
        a, b = ev[:, 0], ev[:, 1]
    return 0.5 * (a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])


def check_mesh_preconditions(mesh, periodic=True):
    """The TWO conditions the SOLVER requires of a mesh (audit A-17). Returns (ok, failures).

    Measured 2026-08-17: with both satisfied the solver agrees with the independent sim to
    0.00000-1e-11 across the crystal, nine disorder families, EIGHT orders of stiffness contrast, and
    deep re-entrant geometry (nu = -9.99). Violate either and it is wrong — on `rotating_squares`
    (known answer nu = -1) the sim gives -1.00000 while the solver reaches -0.685.

      (1) COMBINATORIALLY CLOSED  — every bond in exactly 2 triangles, and V - E + F = 0.
          **PERIODIC MESHES ONLY.** An OPEN mesh legitimately has boundary bonds in ONE triangle;
          the single open hexagon (6 triangles) is exact. Pass `periodic=False` for open meshes.
      (2) GEOMETRICALLY CONSISTENT — no INVERTED (negative signed-area) triangle. Applies to both.

    They are INDEPENDENT: the chord-triangulated re-entrant hexagon satisfies (1) and violates (2)
    (8/32 folded, gap up to 28); the non-triangulation tilings satisfy (2) and violate (1).

    *Why the solver cares and the sim does not:* the metric formulation works on `q_e = dx dx^T` and
    triangle AREAS, both orientation-blind, so a folded triangle is computed as though correctly
    oriented; and open combinatorics corrupts the edge-compatibility / curvature operators. The nodal
    sim only has springs between nodes."""
    failures = []
    tb = np.asarray(mesh['tri_bond'])
    nE = int(np.max(tb)) + 1 if tb.size else 0
    counts = np.bincount(tb.ravel(), minlength=nE)
    if periodic:
        bad = np.flatnonzero(counts != 2)
        if bad.size:
            hist = {int(c): int(n) for c, n in zip(*np.unique(counts, return_counts=True))}
            failures.append(f'not closed: {bad.size} bond(s) not in exactly 2 triangles '
                            f'(bond->#tri histogram {hist})')
        V, F = len(mesh['pts']), len(mesh['simplices'])
        chi = V - nE + F
        if chi != 0:
            failures.append(f'not a torus: V-E+F = {chi} (expected 0)')
    sa = signed_areas(mesh)
    n_inv = int((sa < 0).sum())
    if n_inv:
        failures.append(f'{n_inv} of {len(sa)} triangle(s) INVERTED (negative signed area; '
                        f'most negative {sa.min():.3e})')
    return (not failures), failures


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


# ---- lattice construction (MOVED here from `Phase 3/verifications/_common.py`, audit A-7c,
# 2026-08-18). Phase 5 PRODUCTION code (designer, seeds, triangulation, dataset) needs these, and
# reaching into a module that lives inside a *verifications* directory inverted the layering.
# `_common` re-exports them, so the ~100 verification scripts are untouched. ----------------
from scipy.spatial import Delaunay          # noqa: E402  (used by _periodic_delaunay)

def box(geo):
    """(Lx, Ly) -- the box edge lengths (BL1 along x, BL2 along y) as plain floats."""
    return float(geo['BL1'][0]), float(geo['BL2'][1])


def _ny_commensurate(phi, Lx, row_h):
    """Row count = multiple of the period p (smallest with p·φ/2 integer, so a rectangular box is a
    true periodic supercell) whose height Ny·row_h is CLOSEST to Lx (→ box as square as possible)."""
    period = 2
    for base in (2, 3, 4, 5, 6, 8, 10, 12):
        if abs(base * phi / 2 - round(base * phi / 2)) < 1e-9:
            period = base
            break
    ny_real = Lx / row_h
    lo = max(period, (int(ny_real) // period) * period)
    hi = lo + period
    return lo if abs(lo * row_h - Lx) <= abs(hi * row_h - Lx) else hi


def _periodic_delaunay(pts, Lx, Ly):
    """Periodic Delaunay of a point cloud in [0,Lx)x[0,Ly) via the 3x3-tile trick -> geo dict with
    an axis-aligned SQUARE-ish box BL1=(Lx,0), BL2=(0,Ly)."""
    n = len(pts); box = np.array([Lx, Ly])
    shifts = np.array([(i, j) for i in (-1, 0, 1) for j in (-1, 0, 1)])
    tiled = np.concatenate([pts + s * box for s in shifts], axis=0)
    shift_of = np.repeat(shifts, n, axis=0)
    simp_t = Delaunay(tiled).simplices
    cen = tiled[simp_t].mean(1)
    keep = (cen[:, 0] >= 0) & (cen[:, 0] < Lx) & (cen[:, 1] >= 0) & (cen[:, 1] < Ly)
    simp_t = simp_t[keep]; nt = len(simp_t)
    canon = simp_t % n; sft = shift_of[simp_t]
    p0, p1, p2 = tiled[simp_t[:, 0]], tiled[simp_t[:, 1]], tiled[simp_t[:, 2]]
    centroids = (p0 + p1 + p2) / 3.0                     # TRUE centroids (image-correct, in the box)
    tri_verts = np.stack([p0, p1, p2], 1)               # (nt,3,2) image-correct vertices (for fills)
    edge_vecs = np.stack([p1 - p0, p2 - p0, p2 - p1], 1)
    l2 = (edge_vecs ** 2).sum(2)
    areas = 0.5 * np.abs(edge_vecs[:, 0, 0] * edge_vecs[:, 1, 1]
                         - edge_vecs[:, 0, 1] * edge_vecs[:, 1, 0])
    pairs = [(0, 1, 0), (0, 2, 1), (1, 2, 2)]
    keymap = {}; bonds = []; tri_bond = np.zeros((nt, 3), np.int64)
    for ti in range(nt):
        for ka, kb, ei in pairs:
            ca, cb = int(canon[ti, ka]), int(canon[ti, kb])
            d = sft[ti, kb] - sft[ti, ka]; dp = (int(d[0]), int(d[1])); R = edge_vecs[ti, ei]
            if (ca, dp[0], dp[1]) <= (cb, -dp[0], -dp[1]):
                key, Rk = (ca, cb, dp[0], dp[1]), R
            else:
                key, Rk = (cb, ca, -dp[0], -dp[1]), -R
            if key not in keymap:
                keymap[key] = len(bonds); bonds.append((key[0], key[1], Rk))
            tri_bond[ti, ei] = keymap[key]
    return dict(pts=pts, simplices=canon, edge_vecs=edge_vecs, actual_len2=l2,
                bond_u=np.array([b[0] for b in bonds], np.int64),
                bond_v=np.array([b[1] for b in bonds], np.int64),
                bond_R=np.array([b[2] for b in bonds], float),
                tri_bond=tri_bond, areas=areas, centroids=centroids, tri_verts=tri_verts,
                BL1=np.array([Lx, 0.0]), BL2=np.array([0.0, Ly]))


def make_lattice(phi, psi, half=10.0, seed=0, eta=0.0, half_y=None):
    """Preferred constructor. Base vectors v1=(1,0), v2=(φ/2, ψ·√3/2) (φ=ψ=1 → regular triangular);
    lattice = all m·v1+n·v2; keep a SQUARE real-space region (|x|,|y| ≤ half) as an axis-aligned
    PERIODIC box. Optional eta perturbs positions (disordered); optional half_y makes a RECTANGULAR
    ribbon (y half-height half_y instead of half). Returns a geo dict (k not set)."""
    Nx = max(4, int(round(2 * half)))
    row_h = psi * np.sqrt(3) / 2
    Lx = float(Nx)
    Ny = _ny_commensurate(phi, 2 * half_y if half_y is not None else Lx, row_h)
    Ly = Ny * row_h
    m, n = np.meshgrid(np.arange(Nx), np.arange(Ny), indexing='ij')
    x = (m.ravel() + n.ravel() * phi / 2.0) % Nx
    y = n.ravel() * row_h
    pts = np.stack([x, y], 1).astype(float)
    if eta > 0:
        rng = np.random.default_rng(seed)
        ang = rng.uniform(0, 2 * np.pi, len(pts))
        pts = pts + eta * np.stack([np.cos(ang), np.sin(ang)], 1)
        pts[:, 0] %= Lx; pts[:, 1] %= Ly
    return _periodic_delaunay(pts, Lx, Ly)
