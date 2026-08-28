"""Phase 5 — seed zoo generator (the RICH, physically-structured starting-network set).

Produces a broad but bounded pool of periodic triangulated networks for the M1 designer /
M2 GNN training set.  Five physically-grounded ingredients:

    Bravais family      -> seed_bravais          (make_lattice scan, disorder overlaid)
    random patches      -> random_patch          (uniform / Poisson-disk / blue-noise / graded)
    complex-basis nets   -> seed_with_basis / honeycomb / kagome
    Archimedean tilings  -> seed_tiling           (square / honeycomb / kagome / square_octagon,
                                                    as triangulations with SOFT 'fictional' edges)
    auxetic motifs       -> auxetic_motifs         (rotating squares, re-entrant honeycomb)
    unit cells w/ basis  -> seed_cells             (N=1..12 basis nodes => 2..24 triangles;
                                                    the crystalline covering + the S1 anchors)

Every generator yields a UNIFORM record (see `make_record`) so `designer.py` consumes them all
identically:  dict(name, geo, k0 (n_bond,), is_fictional (n_bond,) bool).

Non-triangular tilings are formally triangulated (a valid solver topology) but the diagonals
added to split non-triangular faces are tagged `is_fictional=True` and given near-zero rigidity
(EPS) in `k0`, so the reference mechanics match the tiling while every edge stays a real design
DOF.  Native tiling bonds get k0=1.0.

Public API
----------
    make_record(name, geo, k0=None, is_fictional=None)          -> record dict
    seed_bravais(...)                                            -> yields (name, geo)
    random_patch(n_nodes, seed, L=None, process='poisson_disk') -> record
    sample_points(n_nodes, L, seed, process)                    -> pts (n,2)
    seed_with_basis(v1, v2, basis, reps, Lx, Ly, eta=, seed=)   -> record
    honeycomb(reps, eta=, seed=) / kagome(reps, eta=, seed=)    -> record
    seed_tiling(name, reps, eps=1e-3)                           -> record
    auxetic_motifs()                                            -> yields record
    seed_from_phase4()                                          -> yields record (best-effort)
    seed_cell(n_basis, seed, aspect=...)                        -> record or None
    seed_cells(n_basis_range=range(1,13), n_cfg=6, report=None) -> yields record
    seed_cells_anchors()                                        -> yields record (known answer)
    seed_pool(n_random=12, n_nodes=120, include=(...))          -> yields record
"""
# ---- §0 preamble (verbatim; seeds.py lives directly in Phase 5/) ------------------------------
import os, sys
import numpy as np
import torch
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))   # repo root from Phase 5/
sys.path.insert(0, os.path.join(REPO, 'Phase 3', 'verifications'))
import _common as C                 # sets up all other sys.path and imports the solver stack
from inverse_design import (DesignProblem, Objective, optimize, validate, ANG,
                            c6_to_nuE, c6_to_nuE_theta)
torch.set_default_dtype(torch.float64)   # REQUIRED — the whole stack is float64

from triangulation import geo_from_simplices, delaunay_tris
import mesh_build as MB              # check_mesh_preconditions — the solver's mesh gate (A-17)


# ---- uniform seed record ---------------------------------------------------------------------
def make_record(name, geo, k0=None, is_fictional=None):
    """Wrap a geo into the uniform record designer.py consumes.  k0 = per-bond initial stiffness
    (uniform 1.0 by default; native=1.0 / fictional=EPS for tilings).  is_fictional = per-bond
    bool mask (True for the soft added triangulating edges)."""
    nbond = len(geo['bond_R'])
    if k0 is None:
        k0 = np.ones(nbond)
    if is_fictional is None:
        is_fictional = np.zeros(nbond, bool)
    return dict(name=name, geo=geo, k0=np.asarray(k0, float),
                is_fictional=np.asarray(is_fictional, bool))


# ---- 1. Bravais family -----------------------------------------------------------------------
def seed_bravais(phi_vals=(0.6, 0.8, 1.0, 1.2, 1.5), psi_vals=(0.6, 0.8, 1.0, 1.2),
                 etas=(0.0, 0.15, 0.30), seeds=(0, 1), half=8):
    """Scan the 2D BRAVAIS family: make_lattice(phi,psi) sweeps oblique/rectangular/square/
    hexagonal/sheared single-site lattices.  eta>0 overlays disorder on each (several seeds).
    Yields (name, geo) — wrap with make_record downstream (see seed_pool)."""
    for phi in phi_vals:
        for psi in psi_vals:
            for eta in etas:
                for s in (seeds if eta > 0 else (0,)):
                    yield (f'bravais_phi{phi}_psi{psi}_eta{eta}_s{s}',
                           C.make_lattice(phi, psi, half=half, eta=eta, seed=s))


# ---- 2. random patches (point-process dispatcher) --------------------------------------------
def sample_points(n_nodes, L, seed, process='poisson_disk'):
    """Dispatch on the POINT PROCESS.  Returns (n,2) points in [0,L)^2 (n ~= n_nodes)."""
    if process == 'uniform':
        return _uniform_points(n_nodes, L, seed)
    if process == 'poisson_disk':
        return _poisson_disk(n_nodes, L, seed)
    if process == 'blue_noise':
        return _blue_noise(n_nodes, L, seed)
    if process == 'graded':
        return _graded(n_nodes, L, seed)
    raise ValueError(f'unknown process {process!r}')


def _uniform_points(n, L, seed):
    return np.random.default_rng(seed).uniform(0, L, (n, 2))


def _poisson_disk(n, L, seed, k=30):
    """Periodic Bridson Poisson-disk sampling: min spacing r ~= 0.8*(L/sqrt(n)); toroidal
    distances.  Yields ~n well-separated points (exact count not guaranteed)."""
    rng = np.random.default_rng(seed)
    r = 0.8 * (L / np.sqrt(n))
    cell = r / np.sqrt(2.0)
    gw = max(1, int(np.ceil(L / cell)))
    cell = L / gw
    rr = int(np.ceil(r / cell)) + 1
    grid = -np.ones((gw, gw), int)
    pts = []
    active = []

    def gidx(p):
        return int(p[0] / cell) % gw, int(p[1] / cell) % gw

    p0 = rng.uniform(0, L, 2)
    pts.append(p0); active.append(0); grid[gidx(p0)] = 0
    while active:
        ai = int(rng.integers(len(active)))
        base = pts[active[ai]]
        placed = False
        for _ in range(k):
            ang = rng.uniform(0, 2 * np.pi)
            rad = rng.uniform(r, 2 * r)
            cand = (base + rad * np.array([np.cos(ang), np.sin(ang)])) % L
            gx, gy = gidx(cand)
            ok = True
            for dx in range(-rr, rr + 1):
                for dy in range(-rr, rr + 1):
                    q = grid[(gx + dx) % gw, (gy + dy) % gw]
                    if q >= 0:
                        d = cand - pts[q]
                        d -= np.round(d / L) * L
                        if d @ d < r * r:
                            ok = False
                            break
                if not ok:
                    break
            if ok:
                grid[gx, gy] = len(pts)
                active.append(len(pts))
                pts.append(cand)
                placed = True
                break
        if not placed:
            active.pop(ai)
    return np.array(pts, float)


def _blue_noise(n, L, seed, iters=5):
    """Uniform seed then a few iterations of PERIODIC Lloyd relaxation (each point -> mean of its
    periodic-Delaunay neighbours, minimal-image).  Approximates Voronoi-centroid relaxation."""
    rng = np.random.default_rng(seed)
    pts = rng.uniform(0, L, (n, 2))
    box = np.array([L, L])
    for _ in range(iters):
        geo = C._periodic_delaunay(pts, L, L)
        acc = np.zeros((n, 2)); cnt = np.zeros(n)
        for u, v in zip(geo['bond_u'], geo['bond_v']):
            d = pts[v] - pts[u]
            d -= np.round(d / box) * box
            acc[u] += pts[u] + d; cnt[u] += 1
            acc[v] += pts[v] - d; cnt[v] += 1
        good = cnt > 0
        pts[good] = (acc[good] / cnt[good, None]) % box
    return pts


def _graded(n, L, seed):
    """Spatially varying density (denser in a central vertical band) via rejection sampling."""
    rng = np.random.default_rng(seed)
    out = []
    rho_max = 4.0
    while len(out) < n:
        c = rng.uniform(0, L, 2)
        rho = 1.0 + 3.0 * np.exp(-((c[0] - L / 2) / (0.15 * L)) ** 2)
        if rng.uniform(0, rho_max) < rho:
            out.append(c)
    return np.array(out, float)


def random_patch(n_nodes, seed, L=None, process='poisson_disk'):
    """Rich random periodic point cloud -> record.  Spacing ~= 1 so L ~= sqrt(n_nodes)."""
    L = float(np.sqrt(n_nodes)) if L is None else float(L)
    pts = sample_points(n_nodes, L, seed, process)
    geo = C._periodic_delaunay(pts, L, L)
    return make_record(f'random_{process}_n{n_nodes}_s{seed}', geo)


# ---- 3. complex-basis lattices ---------------------------------------------------------------
def seed_with_basis(bravais_v1, bravais_v2, basis, reps, Lx, Ly, eta=0.0, seed=0, name=None):
    """Place a multi-point `basis` at each Bravais site (m*v1 + n*v2), wrap into [0,Lx)x[0,Ly),
    optional disorder, then _periodic_delaunay the point cloud (honeycomb=2/cell, kagome=3/cell).
    Returns a uniform record (all-native, fully triangulated)."""
    v1 = np.asarray(bravais_v1, float); v2 = np.asarray(bravais_v2, float)
    m, n = np.meshgrid(range(reps), range(reps), indexing='ij')
    sites = m.ravel()[:, None] * v1 + n.ravel()[:, None] * v2
    pts = (sites[:, None, :] + np.asarray(basis, float)[None, :, :]).reshape(-1, 2)
    pts[:, 0] %= Lx; pts[:, 1] %= Ly
    if eta > 0:
        rng = np.random.default_rng(seed)
        a = rng.uniform(0, 2 * np.pi, len(pts))
        pts = pts + eta * np.stack([np.cos(a), np.sin(a)], 1)
        pts[:, 0] %= Lx; pts[:, 1] %= Ly
    geo = C._periodic_delaunay(pts, Lx, Ly)
    _basis_mesh_ok = True
    # Same degeneracy repair as `_delaunay_nondegenerate` (audit A-17): a symmetric basis (honeycomb
    # is the case in point) is full of cocircular quadruples, so the periodic Delaunay tie-break can
    # produce a NON-MANIFOLD mesh — measured honeycomb V-E+F = -5, and solver-vs-sim 0.0285 with no
    # optimisation involved. All edges here are native, so mesh validity is the only criterion.
    if not MB.check_mesh_preconditions(geo, periodic=True)[0]:
        box = np.array([Lx, Ly])
        for s_ in range(24):
            amp = 1e-9 * (10.0 ** (s_ // 8))
            jit = np.mod(pts + amp * np.random.default_rng(s_).standard_normal(pts.shape), box)
            try:
                g2 = C._periodic_delaunay(jit, Lx, Ly)
            except Exception:                            # noqa: BLE001 — try the next tie-break
                continue
            if MB.check_mesh_preconditions(g2, periodic=True)[0]:
                pts, geo = jit, g2
                break
        else:
            _basis_mesh_ok = False      # tagged, not raised — see _delaunay_nondegenerate
    if name is None:
        name = f'basis_r{reps}_eta{eta}_s{seed}'
    rec = make_record(name, geo)
    rec['mesh_ok'] = bool(_basis_mesh_ok)   # False => solver-invalid (A-17); use sim-only
    return rec


def honeycomb(reps=3, eta=0.0, seed=0):
    """Honeycomb via a 4-atom RECTANGULAR cell (commensurate box) -> fully-triangulated record."""
    W = np.sqrt(3.0)
    basis = [(0.0, 0.0), (0.0, 1.0), (W / 2, 1.5), (W / 2, 2.5)]
    return seed_with_basis(np.array([W, 0.0]), np.array([0.0, 3.0]), basis,
                           reps, W * reps, 3.0 * reps, eta=eta, seed=seed,
                           name=f'honeycomb_basis_r{reps}_eta{eta}_s{seed}')


def kagome(reps=3, eta=0.0, seed=0):
    """Kagome via a 6-atom RECTANGULAR cell (commensurate box) -> fully-triangulated record."""
    r3 = np.sqrt(3.0)
    basis = [(0.0, 0.0), (1.0, 0.0), (0.5, r3 / 2), (1.0, r3), (0.0, r3), (1.5, 3 * r3 / 2)]
    return seed_with_basis(np.array([2.0, 0.0]), np.array([0.0, 2 * r3]), basis,
                           reps, 2.0 * reps, 2 * r3 * reps, eta=eta, seed=seed,
                           name=f'kagome_basis_r{reps}_eta{eta}_s{seed}')


# ---- 4. tilings as triangulations with soft 'fictional' edges --------------------------------
def _edge_sig(A, B, box, dp=5):
    """Translation/direction-invariant signature for the undirected edge A--B (minimal-image),
    computed IDENTICALLY for a native-edge definition and for a geo bond (A=pts[u], B=A+bond_R),
    so the two can be matched by set membership."""
    A = np.asarray(A, float); B = np.asarray(B, float)
    d = B - A
    d -= np.round(d / box) * box
    aw = np.round(np.mod(A, box), dp)
    bw = np.round(np.mod(A + d, box), dp)
    dr = np.round(d, dp)
    ka = (aw[0], aw[1]); kb = (bw[0], bw[1]); kd = (dr[0], dr[1])
    if kd > (-dr[0], -dr[1]):
        return (ka, kb, kd)
    return (kb, ka, (-dr[0], -dr[1]))


def _dedupe_pts(raw, Lx, Ly, dp=6):
    """Wrap raw coords into [0,Lx)x[0,Ly) and drop duplicates (shared tiling vertices)."""
    box = np.array([Lx, Ly])
    W = np.mod(np.asarray(raw, float), box)
    seen = {}; out = []
    for p in W:
        key = (round(p[0], dp), round(p[1], dp))
        if key not in seen:
            seen[key] = 1
            out.append([p[0], p[1]])
    return np.array(out, float)


def _native_by_distance(pts, box, bond_len, tol=1e-3):
    """Native edge signatures = all minimal-image vertex pairs at distance ~= bond_len."""
    native = set()
    n = len(pts)
    for i in range(n):
        diff = pts - pts[i]
        diff -= np.round(diff / box) * box
        L = np.hypot(diff[:, 0], diff[:, 1])
        for j in np.where(np.abs(L - bond_len) < tol)[0]:
            if j == i:
                continue
            native.add(_edge_sig(pts[i], pts[i] + diff[j], box))
    return native


def _native_by_displacements(pts, box, disps, tol=1e-3):
    """Native edge signatures = pairs at one of the given displacement vectors (both signs)."""
    native = set()
    n = len(pts)
    for i in range(n):
        diff = pts - pts[i]
        diff -= np.round(diff / box) * box
        for d in disps:
            d = np.asarray(d, float)
            for j in np.where(np.hypot(diff[:, 0] - d[0], diff[:, 1] - d[1]) < tol)[0]:
                native.add(_edge_sig(pts[i], pts[i] + d, box))
    return native


def _natives_survive(geo, required, box):
    """True iff every REQUIRED native edge is present in `geo` (same signature test the caller uses)."""
    if not required:
        return True
    A = geo['pts'][geo['bond_u']]
    B = A + geo['bond_R']
    sigs = {_edge_sig(A[i], B[i], box) for i in range(len(geo['bond_u']))}
    return not (required - sigs)


def _delaunay_nondegenerate(name, pts, Lx, Ly, seeds_=24, required=None):
    """Periodic Delaunay that is guaranteed to be a VALID CLOSED TRIANGULATION.

    **Why this exists (audit A-17, fixed 2026-08-17).** These tilings are highly symmetric, so their
    point sets are full of COCIRCULAR quadruples where the Delaunay tie-break is arbitrary. The
    periodic stitching then produces a NON-MANIFOLD mesh — bonds in 1, 3 or even 4 triangles instead
    of exactly 2. Measured as-built: honeycomb V−E+F = −5, square_octagon −6, rotating_squares −3,
    reentrant_honeycomb −1. That silently breaks the SOLVER (its edge-compatibility and curvature
    operators assume a closed surface) while the nodal sim is unaffected — square_octagon read
    solver +0.0966 vs sim +0.3177, a gap of 0.22 with NO optimisation involved.

    Fix: if the mesh is degenerate, retry with a TINY symmetry-breaking jitter (1e-9 — nine orders
    below the unit edge length, and four below `_edge_sig`'s 5-decimal rounding, so native/fictional
    tagging is untouched; the caller's `required` assert verifies that). The jitter is KEPT: snapping
    back to the exact symmetric coordinates re-creates the degeneracy (measured — honeycomb and
    reentrant fail again, gaps 0.52 and 0.21).

    After the repair all four tilings give solver-vs-sim gap **0.0000** (from 0.0285, 0.2211, 0.0231,
    0.0403), so they are usable by the solver rather than sim-only."""
    box = np.array([Lx, Ly])
    required = set(required or ())
    tris = delaunay_tris(pts, Lx, Ly)
    geo = geo_from_simplices(pts, tris, Lx, Ly)
    if MB.check_mesh_preconditions(geo, periodic=True)[0]:
        return pts, tris, geo, True
    # A retry must satisfy BOTH criteria. Requiring only mesh validity is not enough: the jitter
    # changes WHICH edges Delaunay produces, and a tie-break that fixes the manifold can drop a
    # native rib (measured on reentrant_honeycomb, which then failed the caller's `required` assert).
    for s in range(seeds_):
        amp = 1e-9 * (10.0 ** (s // 8))                  # 1e-9, then 1e-8, then 1e-7
        jit = np.mod(pts + amp * np.random.default_rng(s).standard_normal(pts.shape), box)
        try:
            t2 = delaunay_tris(jit, Lx, Ly)
            g2 = geo_from_simplices(jit, t2, Lx, Ly)
        except Exception:                                # noqa: BLE001 — try the next tie-break
            continue
        if MB.check_mesh_preconditions(g2, periodic=True)[0] and _natives_survive(g2, required, box):
            return jit, t2, g2, True
    # UNREPAIRABLE. Do NOT raise: some tilings genuinely cannot be Delaunay-triangulated both as a
    # manifold AND keeping every native rib (a honeycomb's ribs are not all Delaunay edges of its
    # vertex set — the old code only "kept" them by being non-manifold). Raising here would break
    # every driver that uses these seeds. Instead return the best effort TAGGED, so the A-17 gate in
    # `designer.PHYSICALITY_CHECKS` rejects it at the point of USE and the caller can go sim-only or
    # switch representation (a centre-vertex fan triangulates any polygon validly — cf. dhex).
    return pts, tris, geo, False


def _native_adjacency(pts, box, bond_len=None, tol=1e-3, disps=None):
    """i -> [(j, d)] for native edges, `d` the minimal-image displacement i->j.

    Selects native neighbours EITHER by a single bond LENGTH (`bond_len` — the Archimedean tilings,
    whose edges are all one length) OR by an explicit DISPLACEMENT set (`disps`, matching +/- each
    vector). The re-entrant honeycomb needs the second: its ribs come in two different lengths (the
    vertical rib `v` and the two diagonals), so a single-length match silently drops one rib family
    and the face traversal then fails Euler."""
    if (bond_len is None) == (disps is None):
        raise ValueError('give exactly one of bond_len / disps')
    adj = {i: [] for i in range(len(pts))}
    targets = None if disps is None else np.array(
        [u for d in disps for u in (np.asarray(d, float), -np.asarray(d, float))], float)
    for i in range(len(pts)):
        d = pts - pts[i]
        d -= np.round(d / box) * box
        if targets is None:
            hit = np.where(np.abs(np.hypot(d[:, 0], d[:, 1]) - bond_len) < tol)[0]
        else:
            hit = np.where((np.abs(d[:, None, :] - targets[None, :, :]).max(2) < tol).any(1))[0]
        for j in hit:
            if j != i:
                adj[i].append((int(j), d[j].copy()))
    return adj


def _native_faces(adj):
    """Faces of the periodic native graph, by next-edge-clockwise traversal of half-edges.

    At `j`, arriving along `d`, the next half-edge of the (CCW) face is the neighbour one step
    CLOCKWISE from the reversed incoming direction. Returns a list of faces, each a list of
    (vertex, incoming displacement). Verify with Euler: on a torus V - E + F = 0."""
    order, idx = {}, {}
    for i, nb in adj.items():
        order[i] = sorted(nb, key=lambda t: np.arctan2(t[1][1], t[1][0]))      # CCW by angle
        for p, (j, d) in enumerate(order[i]):
            idx[(i, j, round(d[0], 6), round(d[1], 6))] = p
    seen, out = set(), []
    for i in order:
        for (j, d) in order[i]:
            he = (i, j, round(d[0], 6), round(d[1], 6))
            if he in seen:
                continue
            face, cur = [], he
            while cur not in seen:
                seen.add(cur)
                ci, cj, dx, dy = cur
                face.append((ci, np.array([dx, dy])))
                p = idx[(cj, ci, round(-dx, 6), round(-dy, 6))]
                nxt = order[cj][(p - 1) % len(order[cj])]
                cur = (cj, nxt[0], round(nxt[1][0], 6), round(nxt[1][1], 6))
            out.append(face)
    return out


def _fan_and_tag(name, pts, native, Lx, Ly, bond_len, eps, disps=None):
    """Triangulate a tiling by adding a PHANTOM CENTRE VERTEX inside every non-triangular face and
    fanning to its corners. Natives keep k0=1.0; the added SPOKES are `is_fictional` with k0=eps.

    **Why this replaces Delaunay-with-chords (2026-08-22).** `_triangulate_and_tag` Delaunay-
    triangulates the tiling's vertices, which splits each polygon with CHORDS. On the honeycomb and
    kagome the periodic cocircular ties produce chords that **cross each other** — measured 4
    crossings on `tiling_honeycomb_r3`, 5 on `_r4`, 2 on `tiling_kagome_r2` — i.e. overlapping
    triangles, not a mesh. Those three were correctly flagged `mesh_ok=False` (audit A-17) and yet
    still entered `goal1`'s design pool, where `tiling_honeycomb_r3` was the worst solver-vs-sim
    disagreement in every run (0.311 -> 0.323 -> 0.708 as the position budget grew).

    A fan cannot produce a crossing: every added edge joins the face's own centre to its own corner,
    so added edges meet only at that centre and never leave the face. It is also the representation
    the project has already VALIDATED — `test_hex_closed_form` builds a hexagon as centre + 6 spokes
    and recovers the analytic free-hinge nu(r) to 4.4e-06, with the residual first order in k_spoke,
    because a soft SPOKE lets the polygon hinge whereas a soft CHORD must still carry its shear.

    A proper chord-based tiling (choosing a non-crossing diagonal set) is a separate TODO; this is
    the correct-by-construction option and it is what ships now."""
    box = np.array([Lx, Ly])
    adj = _native_adjacency(pts, box, bond_len, disps=disps)
    faces = _native_faces(adj)
    nV, nE = len(pts), sum(len(v) for v in adj.values()) // 2
    assert nV - nE + len(faces) == 0,         f'{name}: face traversal gives V-E+F = {nV - nE + len(faces)}, expected 0 on a torus'

    new_pts, tris = [p.copy() for p in pts], []

    def shift(base, world):
        return np.round((world - new_pts[base]) / box).astype(int)

    for face in faces:
        corners, acc = [], np.zeros(2)
        for (vi, d) in face:
            corners.append((vi, acc.copy()))
            acc = acc + d
        if not np.allclose(acc, 0.0, atol=1e-6):        # closes only around the torus: not a face
            continue
        world = [pts[face[0][0]] + off for (_, off) in corners]
        if len(corners) == 3:
            tris.append([[corners[t][0], *shift(corners[t][0], world[t])] for t in range(3)])
            continue
        centre = np.mean(world, axis=0)
        ci = len(new_pts)
        new_pts.append(np.mod(centre, box))
        for t in range(len(corners)):
            a, b = t, (t + 1) % len(corners)
            tris.append([[ci, *shift(ci, centre)],
                         [corners[a][0], *shift(corners[a][0], world[a])],
                         [corners[b][0], *shift(corners[b][0], world[b])]])

    geo = geo_from_simplices(np.array(new_pts), np.array(tris, np.int64), Lx, Ly)
    A = geo['pts'][geo['bond_u']]
    sigs = [_edge_sig(A[bi], A[bi] + geo['bond_R'][bi], box) for bi in range(len(geo['bond_u']))]
    is_fic = np.array([sg not in native for sg in sigs], bool)
    missing = native - set(sigs)
    assert not missing, f'{name}: fan dropped {len(missing)} of {len(native)} native edges'
    assert is_fic.sum() > 0, f'{name}: no spokes produced (nothing to triangulate?)'
    assert (geo['areas'] > 0).all(), f'{name}: non-positive triangle area'
    rec = make_record(name, geo, k0=np.where(is_fic, eps, 1.0), is_fictional=is_fic)
    rec['mesh_ok'] = True            # correct by construction; asserted by the caller's checks
    return rec


def _ear_clip(poly, tol=1e-12):
    """Triangulate a SIMPLE polygon using ONLY ITS OWN VERTICES (ear clipping).

    Returns index triples into `poly` (an (n,2) array of the polygon's corners in order). Every
    diagonal it emits stays strictly inside the polygon: an ear is cut only when its apex is CONVEX
    *and* no other corner lies inside it, which is what makes the result **non-crossing by
    construction** — the property the Delaunay chord split failed to provide (see `_chord_and_tag`).

    Handles non-convex faces too, so it does not assume the tiling's polygons are convex.
    """
    n = len(poly)
    idx = list(range(n))
    area2 = sum(poly[i][0] * poly[(i + 1) % n][1] - poly[(i + 1) % n][0] * poly[i][1]
                for i in range(n))
    if area2 < 0:                                  # ear clipping below assumes CCW
        idx.reverse()

    def cross(o, p, q):
        return (p[0] - o[0]) * (q[1] - o[1]) - (p[1] - o[1]) * (q[0] - o[0])

    def inside(p, a, b, c):
        d1, d2, d3 = cross(a, b, p), cross(b, c, p), cross(c, a, p)
        return (d1 > tol and d2 > tol and d3 > tol)          # strictly interior only

    out = []
    while len(idx) > 3:
        for k in range(len(idx)):
            i0, i1, i2 = idx[k - 1], idx[k], idx[(k + 1) % len(idx)]
            a, b, c = poly[i0], poly[i1], poly[i2]
            if cross(a, b, c) <= tol:                        # reflex or collinear: not an ear
                continue
            if any(inside(poly[m], a, b, c) for m in idx if m not in (i0, i1, i2)):
                continue
            out.append((i0, i1, i2))
            idx.pop(k)
            break
        else:                                                # fail fast (project default)
            raise RuntimeError(f'ear clipping stalled on a {n}-gon: not a simple polygon?')
    out.append(tuple(idx))
    return out


def _chord_and_tag(name, pts, native, Lx, Ly, bond_len, eps):
    """Triangulate a tiling with NON-CROSSING CHORDS — the tiling's own vertices only, no phantoms.

    **Why this exists alongside `_fan_and_tag`.** The fan is correct by construction and is what
    ships, but it inserts a PHANTOM CENTRE per face, which changes the vertex set and adds DOF
    (`tiling_honeycomb_r3`: 36 -> 54 nodes, 110 -> 162 bonds). Anything reasoning about the tiling's
    own coordination — or feeding node counts to M2 — sees a different graph. This keeps the vertex
    set of the actual tiling: an n-gon becomes n-2 triangles via n-3 chords.

    **It is NOT a drop-in replacement for the fan, and the difference is physical, not cosmetic.**
    A soft SPOKE lets a polygon hinge; a soft CHORD must still carry the face's shear (`_fan_and_tag`,
    and `NEXT_SESSION.md`: honouring `k0` on chords "would only make everything near-mechanism").
    So a chord tiling with `eps` chords is a genuinely different network from a fan with `eps`
    spokes. Which one a study should use is a modelling choice, not a default to flip silently.

    The failure this fixes is the Delaunay chord split (`_triangulate_and_tag`), which chose chords
    globally and produced edges that CROSS on periodic cocircular ties — 4 crossings on
    `tiling_honeycomb_r3`, 5 on `_r4`, 2 on `tiling_kagome_r2`, i.e. overlapping triangles. Here the
    chords are chosen PER FACE by ear clipping in the face's own unwrapped frame, so a chord can
    neither leave its face nor cross another chord of the same face.
    """
    box = np.array([Lx, Ly])
    adj = _native_adjacency(pts, box, bond_len)
    faces = _native_faces(adj)
    nV, nE = len(pts), sum(len(v) for v in adj.values()) // 2
    assert nV - nE + len(faces) == 0, \
        f'{name}: face traversal gives V-E+F = {nV - nE + len(faces)}, expected 0 on a torus'

    tris = []

    def shift(base, world):
        return np.round((world - pts[base]) / box).astype(int)

    for face in faces:
        corners, acc = [], np.zeros(2)
        for (vi, d) in face:
            corners.append((vi, acc.copy()))
            acc = acc + d
        if not np.allclose(acc, 0.0, atol=1e-6):        # closes only around the torus: not a face
            continue
        world = [pts[face[0][0]] + off for (_, off) in corners]
        local = [(corners[t][0], shift(corners[t][0], world[t])) for t in range(len(corners))]
        if len(corners) == 3:
            tris.append([[local[t][0], *local[t][1]] for t in range(3)])
            continue
        for (a, b, c) in _ear_clip(np.array(world)):    # chords chosen INSIDE this face only
            tris.append([[local[a][0], *local[a][1]],
                         [local[b][0], *local[b][1]],
                         [local[c][0], *local[c][1]]])

    geo = geo_from_simplices(np.array(pts), np.array(tris, np.int64), Lx, Ly)
    A = geo['pts'][geo['bond_u']]
    sigs = [_edge_sig(A[bi], A[bi] + geo['bond_R'][bi], box) for bi in range(len(geo['bond_u']))]
    is_fic = np.array([sg not in native for sg in sigs], bool)
    missing = native - set(sigs)
    assert not missing, f'{name}: chording dropped {len(missing)} of {len(native)} native edges'
    assert (geo['areas'] > 0).all(), f'{name}: non-positive triangle area'
    # Coverage cross-check: on a torus an exact triangulation tiles the box exactly once, so the
    # areas must sum to Lx*Ly. NECESSARY BUT NOT SUFFICIENT, and measured as such against the
    # known-bad Delaunay split: it catches `honeycomb_r3` (0.9537) and `_r4` (0.9948) but NOT
    # `kagome_r2`, which lands at 1.000000 despite its 2 crossings -- an overlap and a gap can
    # cancel in the total. Note the bad cases come out UNDER 1 (net gaps), not over.
    # The real guarantee here is STRUCTURAL: ear clipping confines every chord to the interior of
    # its own face, so no chord can cross another. This assert only catches gross construction bugs.
    tot = float(geo['areas'].sum())
    assert abs(tot - Lx * Ly) < 1e-6 * Lx * Ly, \
        f'{name}: triangle areas sum to {tot:.6f}, expected {Lx * Ly:.6f} -> mesh does not tile'
    rec = make_record(name, geo, k0=np.where(is_fic, eps, 1.0), is_fictional=is_fic)
    rec['mesh_ok'] = True            # non-crossing by construction; asserted above
    return rec


def _triangulate_and_tag(name, pts, native, Lx, Ly, eps, required=None):
    """Delaunay-triangulate the tiling vertices, tag each geo bond native/fictional by matching its
    signature against `native`, and assert every REQUIRED edge survived (else the mesh is
    untrustworthy).  `required` defaults to `native` (all natives must survive); pass a subset when
    `native` intentionally over-lists (e.g. both square diagonals, only one of which Delaunay can
    realise) — then only the subset is guaranteed and the extras are tagged if present."""
    box = np.array([Lx, Ly])
    if required is None:
        required = native
    pts, tris, geo, mesh_ok = _delaunay_nondegenerate(name, pts, Lx, Ly, required=required)
    A = geo['pts'][geo['bond_u']]
    B = A + geo['bond_R']
    sigs = [_edge_sig(A[bi], B[bi], box) for bi in range(len(geo['bond_u']))]
    is_fic = np.array([s not in native for s in sigs], bool)
    missing = required - set(sigs)
    assert not missing, \
        f"{name}: triangulation dropped {len(missing)} of {len(required)} required native edges " \
        f"(not safe to use)"
    assert is_fic.sum() > 0, f"{name}: no fictional edges produced (nothing to triangulate?)"
    assert (geo['areas'] > 0).all(), f"{name}: non-positive triangle area"
    k0 = np.where(is_fic, eps, 1.0)
    rec = make_record(name, geo, k0=k0, is_fictional=is_fic)
    rec['mesh_ok'] = bool(mesh_ok)      # False => solver-invalid (A-17); sim-only
    return rec


def _raw_square(reps, s=1.0):
    pts = [(i * s, j * s) for i in range(reps) for j in range(reps)]
    return np.array(pts, float), reps * s, reps * s, s


def _raw_honeycomb(reps):
    W = np.sqrt(3.0); H = 3.0
    cell = [(0.0, 0.0), (0.0, 1.0), (W / 2, 1.5), (W / 2, 2.5)]
    pts = [(i * W + cx, j * H + cy) for i in range(reps) for j in range(reps) for cx, cy in cell]
    return np.array(pts, float), reps * W, reps * H, 1.0


def _raw_kagome(reps):
    W = 2.0; r3 = np.sqrt(3.0); H = 2 * r3
    cell = [(0.0, 0.0), (1.0, 0.0), (0.5, r3 / 2), (1.0, r3), (0.0, r3), (1.5, 3 * r3 / 2)]
    pts = [(i * W + cx, j * H + cy) for i in range(reps) for j in range(reps) for cx, cy in cell]
    return np.array(pts, float), reps * W, reps * H, 1.0


def _raw_square_octagon(reps):
    """Truncated-square tiling 4.8.8: regular octagons (edge 1) on a square lattice period P=1+sqrt2;
    the interstitial squares reuse octagon vertices, so the octagon vertex set is the full set."""
    P = 1.0 + np.sqrt(2.0)
    u = 0.5 + np.sqrt(2.0) / 2; t = 0.5
    off = [(u, t), (t, u), (-t, u), (-u, t), (-u, -t), (-t, -u), (t, -u), (u, -t)]
    pts = [(i * P + ox, j * P + oy) for i in range(reps) for j in range(reps) for ox, oy in off]
    return np.array(pts, float), reps * P, reps * P, 1.0


_TILINGS = {'square': _raw_square, 'honeycomb': _raw_honeycomb,
            'kagome': _raw_kagome, 'square_octagon': _raw_square_octagon}


def seed_tiling(name, reps, eps=1e-3, method='fan'):
    """A non-triangular tiling represented as a triangulation with SOFT 'fictional' diagonal edges.
    name in {'square','honeycomb','kagome','square_octagon'}.  Native bonds -> k0=1.0; added
    triangulating diagonals -> k0=eps and is_fictional=True.

    `method` selects how the non-triangular faces are triangulated:
      'fan'   (DEFAULT, what ships) — phantom centre vertex per face, fanned to its corners.
              Correct by construction and the representation `test_hex_closed_form` validates.
      'chord' — non-crossing chords, the tiling's OWN vertices only (no phantoms, no extra DOF).
              Use when node/bond counts must match the actual tiling (e.g. feeding M2).
    **They are physically different networks, not two spellings of one:** a soft SPOKE lets a face
    hinge, a soft CHORD must still carry its shear. Choosing between them is a modelling decision —
    see `_chord_and_tag`."""
    if name not in _TILINGS:
        raise ValueError(f"unknown tiling {name!r}; have {sorted(_TILINGS)}")
    if method not in ('fan', 'chord'):
        raise ValueError(f"unknown method {method!r}; have 'fan', 'chord'")
    raw, Lx, Ly, bond_len = _TILINGS[name](reps)
    pts = _dedupe_pts(raw, Lx, Ly)
    native = _native_by_distance(pts, np.array([Lx, Ly]), bond_len)
    # NOT Delaunay chords (2026-08-22): a global Delaunay CROSSES on honeycomb/kagome — measured
    # 4/5/2 crossings, i.e. overlapping triangles. Both options below are non-crossing.
    tag = _fan_and_tag if method == 'fan' else _chord_and_tag
    return tag(f'tiling_{name}_r{reps}', pts, native, Lx, Ly, bond_len, eps)


# ---- 5. Phase 4 point layouts (OPTIONAL, best-effort) ----------------------------------------
def seed_from_phase4():
    """Best-effort reuse of Phase 4 point clouds.  Phase 4's generators are NOT periodic (buffered
    + interior-filtered domains), so we wrap them into a periodic box heuristically and only yield
    the ones that produce a valid geo.  Any failure is logged and skipped."""
    try:
        import importlib.util
        p4 = os.path.join(REPO, 'Phase 4', 'data', 'topology_generators.py')
        if not os.path.exists(p4):
            print('[seed_from_phase4] Phase 4/data/topology_generators.py not found — skipping')
            return
        spec = importlib.util.spec_from_file_location('p4_topogen', p4)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    except Exception as e:                                   # noqa: BLE001
        print(f'[seed_from_phase4] import failed ({e}) — skipping')
        return
    candidates = [('iso_crystal', 'generate_iso_crystal', (6,)),
                  ('poisson_delaunay', 'generate_poisson_delaunay', (6,)),
                  ('blue_noise', 'generate_blue_noise', (6,))]
    for tag, fn, args in candidates:
        try:
            res = getattr(mod, fn)(*args)
            pts = np.asarray(res.points, float)
            lo = pts.min(0); span = pts.max(0) - lo
            L = float(max(span[0], span[1]))
            if L <= 0 or len(pts) < 4:
                continue
            pw = np.mod(pts - lo, np.array([L, L]))
            geo = C._periodic_delaunay(pw, L, L)
            if (geo['areas'] > 0).all():
                yield make_record(f'phase4_{tag}', geo)
        except Exception as e:                              # noqa: BLE001
            print(f'[seed_from_phase4] {tag} failed ({e}) — skipping')
            continue


# ---- 6. auxetic motifs (soft-edge construction) ----------------------------------------------
def _rotating_squares(reps=4, theta_deg=25.0, eps=1e-3):
    """Rigid squares (side 1) hinged corner-to-corner, rotated +/-theta on a checkerboard.  Square
    EDGES (required) and one square DIAGONAL are native/stiff (each square a rigid body); the
    interstitial rhombic voids are triangulated with soft diagonals -> the rotating-squares auxetic
    mechanism.  Needs EVEN reps for a commensurate periodic box (odd reps break the seam)."""
    reps += reps % 2                                         # force even (commensurate box)
    th = np.radians(theta_deg)
    p = np.cos(th) + np.sin(th)                              # center spacing for corner coincidence
    Lx = Ly = reps * p
    box = np.array([Lx, Ly])
    corners_local = np.array([(0.5, 0.5), (-0.5, 0.5), (-0.5, -0.5), (0.5, -0.5)])

    def rot(a):
        return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])

    raw = []
    squares = []
    for i in range(reps):
        for j in range(reps):
            a = th if (i + j) % 2 == 0 else -th
            c = np.array([i * p, j * p])
            cor = c + corners_local @ rot(a).T
            squares.append(cor)
            raw.extend(cor.tolist())
    pts = _dedupe_pts(np.array(raw, float), Lx, Ly)
    edges = set()                                            # square edges — must all survive
    native = set()                                           # + both diagonals (Delaunay picks one)
    for cor in squares:
        for a, b in [(0, 1), (1, 2), (2, 3), (3, 0)]:
            s = _edge_sig(cor[a], cor[b], box); edges.add(s); native.add(s)
        for a, b in [(0, 2), (1, 3)]:
            native.add(_edge_sig(cor[a], cor[b], box))
    return _triangulate_and_tag(f'auxetic_rotating_squares_t{theta_deg}', pts, native,
                                Lx, Ly, eps, required=edges)


def _reentrant_honeycomb(reps=4, v=1.5, eps=1e-3):
    """Re-entrant (auxetic) honeycomb: honeycomb connectivity with the vertical rib (length v)
    LONGER than the diagonal ribs' vertical extent, so the diagonal ribs slope back inward.  Native
    ribs (two lengths) -> stiff; the diagonals added to triangulate the re-entrant hexagons -> soft.
    Needs EVEN reps for a commensurate staggered box (odd reps drop 3 seam ribs)."""
    lo, hi = REENTRANT_V_RANGE
    if not (lo <= v <= hi):
        raise ValueError(f'v={v} outside the validated range [{lo}, {hi}] — outside it the basis '
                         f'wraps past the hardcoded cell height 2.0 and triangles INVERT, which '
                         f'the orientation-blind solver does not reject (it returned nu=+1.15 for '
                         f'an inverted re-entrant honeycomb). See REENTRANT_V_RANGE.')
    reps += reps % 2                                         # force even (commensurate stagger)
    W = np.sqrt(3.0)
    basis = [(0.0, 0.0), (0.0, v), (W / 2, 1.0), (W / 2, 1.0 + v)]
    raw = [(i * W + cx, j * 2.0 + cy)
           for i in range(reps) for j in range(reps) for cx, cy in basis]
    Lx = W * reps; Ly = 2.0 * reps
    box = np.array([Lx, Ly])
    pts = _dedupe_pts(np.array(raw, float), Lx, Ly)
    disps = [(0.0, v), (W / 2, v - 1.0), (-W / 2, v - 1.0)]  # u1 (vertical), u2, u3 (diagonals)
    native = _native_by_displacements(pts, box, disps)
    # PHANTOM-CENTRE FAN, not the Delaunay chord split (A-17's tail, fixed 2026-08-25). The chord
    # version left this the ONE pool member failing `check_mesh_preconditions` — 9 bonds not in
    # exactly 2 triangles, V-E+F = -1 — which mattered out of all proportion to one topology,
    # because `auxetic` had only two members and this was half of them: the rare mechanism-driven
    # corner a design tool exists to reach was effectively a single valid network.
    return _fan_and_tag(f'auxetic_reentrant_honeycomb_v{v}', pts, native, Lx, Ly,
                        bond_len=None, eps=eps, disps=disps)


#: the auxetic family's OWN parameter axes (M2_V2_PLAN §3.1b). A blanket position-disorder eta is
#: the WRONG axis here: perturbing a re-entrant honeycomb makes it generic, so it DEPOPULATES the
#: rare auxetic region it was included to fill. Traversing the motif parameter moves through the
#: family while STAYING auxetic.
ROT_SQUARE_THETAS = (10.0, 17.5, 25.0, 32.5, 40.0)      # 0 = degenerate square lattice, 45 = closed

#: `v` is the vertical rib length. MEASURED 2026-08-25 (reps=4, uniform k), and the sweep is chosen
#: to straddle the transition rather than to sit on one side of it:
#:
#:      v = 0.6   nu = +1.20        v < 1 : ribs slope OUTWARD, ordinary honeycomb-like
#:      v = 0.8   nu = +1.97
#:      v = 1.0   nu = +0.0005      <-- the RE-ENTRANT TRANSITION, at v = 1 exactly
#:      v = 1.1   nu = -3.55        <-- deepest in the project (previous best -0.60, auxetic_sweep)
#:      v = 1.5   nu = -1.08        deep v asymptotes to nu ~ -1
#:      v = 1.95  nu = -1.02
#:
#: VALID RANGE v in [0.6, 1.95]. The cell height is hardcoded 2.0, so at v = 2.0 the face traversal
#: finds no spokes (AssertionError) and at v >= 2.1 the basis wraps past the box and **128 of 192
#: triangles INVERT** — which the solver does NOT reject on its own: it happily returned nu = +1.15
#: for a structure that cannot be positive, because the metric formulation is orientation-blind
#: (`check_mesh_preconditions` docstring). Widening past 1.95 needs the box height to scale with v,
#: which is a change to the MOTIF GEOMETRY and is deliberately not attempted here.
REENTRANT_V_RANGE = (0.6, 1.95)
REENTRANT_VS = (0.7, 0.85, 1.0, 1.15, 1.3, 1.5, 1.7, 1.9)


def auxetic_motifs(reps=4, eps=1e-3, thetas=ROT_SQUARE_THETAS, vs=REENTRANT_VS):
    """Yield classic auxetic/mechanism seeds, SWEPT over their own motif parameters.  `reps` is
    forced even inside each builder (both need a commensurate periodic box).

    Pass `thetas=(25.0,)`, `vs=(1.5,)` for the single-point behaviour this had before 2026-08-25,
    when the whole family was two topologies at one parameter value each."""
    for th in thetas:
        yield _rotating_squares(reps=reps, theta_deg=th, eps=eps)
    for v in vs:
        yield _reentrant_honeycomb(reps=reps, v=v, eps=eps)


# ---- 6a. BRAVAIS LATTICES, BUILT BY CONSTRUCTION (no Delaunay) --------------------------------
#
# A 2D Bravais lattice needs NO triangulation step.  Pick the lattice vectors a1, a2; bond each site
# along `a1`, along `a2`, and along ONE DIAGONAL of the primitive parallelogram.  That is already a
# triangulation: V=1, E=3, F=2 per primitive cell, and V - E + F = 0 on the torus exactly.
#
# WHY THIS REPLACES `make_lattice` + `_periodic_delaunay` FOR THIS FAMILY (user, 2026-08-25):
#
#   * The DIAGONAL IS A CHOICE, NOT A DEGENERACY.  Delaunay picks whichever diagonal is shorter, so
#     the connectivity FLIPS discontinuously as phi varies.  Measured on the phi sweep at psi=1:
#     nu = 0.2406 at phi=0 but 0.0883 at phi=0.125 -- the triangulation switched diagonals, not the
#     physics.  Building it explicitly makes `a2-a1` vs `a1+a2` a sampled AXIS carrying two
#     legitimate networks, instead of a tie-break decided by floating point.
#   * IT REMOVES THE COCIRCULAR AMBIGUITY.  At phi=0 (rectangular / square) all four corners of
#     every cell are cocircular, so the Delaunay is genuinely ambiguous -- the A-17 degeneracy class
#     that forced the jitter repair in `seed_with_basis`.  Two representations of the SAME lattice
#     (phi=0 and phi=2) disagreed by 5.35e-03 in nu for exactly this reason, while every
#     non-degenerate pair agreed to 1e-16.
#   * ETA-DISORDER STAYS CANONICAL.  Positions are perturbed on the FIXED connectivity, which is the
#     project's mandated frozen-connectivity magnitude-eta method (`CLAUDE.md` §3) -- no
#     re-triangulation, which a point-cloud generator cannot promise.
#
# phi's FUNDAMENTAL DOMAIN IS [0, 1] (measured over 0..3 in steps of 0.125): a2 -> a2 + n*a1 sends
# phi -> phi + 2 and generates the identical lattice, and phi -> -phi is the mirror, so nu(phi) has
# period 2 and mirrors about phi=1 -- verified to 1e-16 at phi = 0.25, 0.5, 0.75, 1.0 against phi+2.
# Sampling beyond [0,1] duplicates.  psi (the row spacing) is the knob that is genuinely open.

BRAVAIS_DIAGONALS = ('a2-a1', 'a1+a2')


def _commensurate_ny(phi, Lx, row_h, max_den=64):
    """Rows per rectangular supercell: the smallest period `p` with `p*phi/2` integral, scaled up to
    a height near `Lx` (so the box comes out roughly square).

    Computed from `phi` rather than taken from `mesh_build._ny_commensurate`, whose period is looked
    up in a hardcoded list (2,3,4,5,6,8,10,12).  That list cannot express phi = 1/8, which needs a
    period of 16, so a uniform sweep over the fundamental domain [0,1] silently loses half its
    points -- measured: phi = 0.125, 0.375, 0.625, 0.875 all failed.  A Fraction gives the exact
    period for any rational phi, and `max_den` bounds how large a cell a nearby rational may ask
    for."""
    from fractions import Fraction
    p = Fraction(phi / 2.0).limit_denominator(max_den).denominator
    n_target = max(1, int(round(Lx / row_h)))
    return int(p * max(1, round(n_target / p)))


def _bravais_site(m, n, Nx, Ny, phi, row_h, Lx, Ly, xshift):
    """Lattice indices (m, n) -> (flat base index, sx, sy) with periodic image shifts.

    Crossing the y-boundary also shifts x, because the box is a rectangular supercell of a SHEARED
    lattice: `_ny_commensurate` chooses Ny so that `Ny*phi/2` is an integer, and that integer
    (`xshift`) is how far the lattice slides sideways in one trip around y.

    The shifts are computed from POSITIONS, not from index arithmetic.  Index arithmetic alone is
    wrong and silently so: the stored coordinate is `(m + n*phi/2) % Nx`, so the modulo has already
    folded some sites back into the box by an amount that depends on BOTH m and n.  Deriving the
    shift from the wrap count of the indices misses that fold, which reconstructs the triangle at
    the wrong image -- measured as inverted triangles and nu = 0.116 where the regular triangular
    lattice must give 1/3.  (It passed at phi = 0, where there is no shear and the two agree, which
    is exactly what made it look plausible.)"""
    ny_wrap, n0 = divmod(n, Ny)
    _, m0 = divmod(m + ny_wrap * xshift, Nx)
    base = m0 * Ny + n0
    true_x, true_y = m + n * phi / 2.0, n * row_h
    stored_x, stored_y = (m0 + n0 * phi / 2.0) % Nx, n0 * row_h
    sx, sy = (true_x - stored_x) / Lx, (true_y - stored_y) / Ly
    isx, isy = int(round(sx)), int(round(sy))
    assert abs(sx - isx) < 1e-9 and abs(sy - isy) < 1e-9, \
        f'non-integer image shift ({sx}, {sy}) at (m,n)=({m},{n}) — box is not a periodic supercell'
    return base, isx, isy


def bravais_lattice(phi, psi, reps=6, diagonal='a2-a1', eta=0.0, seed=0, name=None,
                    disp_fn=None, disp_tag=''):
    """One Bravais lattice as an explicit periodic triangulation.  Returns a uniform record.

    a1 = (1, 0), a2 = (phi/2, psi*sqrt(3)/2) -- the parametrisation `make_lattice` uses, so
    phi = psi = 1 is the regular triangular lattice.  The five planar Bravais types sit at
    phi=1,psi=1 (triangular), phi=0,psi=2/sqrt(3) (square), phi=0 (rectangular), phi=1 (centred
    rectangular) and general phi,psi (oblique).

    `diagonal` selects which diagonal of the primitive cell is bonded, and the two are DIFFERENT
    MATERIALS rather than two views of one: at phi=psi=1 the diagonal `a2-a1` has length 1 (giving
    the regular triangular lattice, nu=1/3) while `a1+a2` has length sqrt(3).

    `eta` perturbs positions on the FIXED connectivity.  eta > 0 is a DIFFERENT CATEGORY from the
    ordered crystal and the caller labels it as one (3.1b: hold families out at LOW disorder and
    treat high disorder as its own regime)."""
    if diagonal not in BRAVAIS_DIAGONALS:
        raise ValueError('diagonal must be one of %s, got %r' % (BRAVAIS_DIAGONALS, diagonal))
    Nx = int(reps)
    row_h = psi * np.sqrt(3) / 2.0
    Ny = _commensurate_ny(phi, float(Nx), row_h)
    xshift = int(round(Ny * phi / 2.0))
    if abs(Ny * phi / 2.0 - xshift) > 1e-9:
        raise ValueError('phi=%g gives no rectangular supercell at Ny=%d' % (phi, Ny))
    Lx, Ly = float(Nx), Ny * row_h

    m, n = np.meshgrid(np.arange(Nx), np.arange(Ny), indexing='ij')
    pts = np.stack([(m.ravel() + n.ravel() * phi / 2.0) % Nx, n.ravel() * row_h], 1).astype(float)

    tris = []
    for mm in range(Nx):
        for nn in range(Ny):
            args = (Nx, Ny, phi, row_h, Lx, Ly, xshift)
            p00 = _bravais_site(mm, nn, *args)
            p10 = _bravais_site(mm + 1, nn, *args)
            p01 = _bravais_site(mm, nn + 1, *args)
            p11 = _bravais_site(mm + 1, nn + 1, *args)
            if diagonal == 'a2-a1':           # bonds a1, a2, a2-a1
                tris.append([p00, p10, p01])
                tris.append([p10, p11, p01])
            else:                              # bonds a1, a2, a1+a2
                tris.append([p00, p10, p11])
                tris.append([p00, p11, p01])

    if eta > 0:                                # frozen connectivity: perturb AFTER building tris
        # NOT wrapped back into the box, deliberately. The triangles carry fixed integer image
        # shifts computed from the IDEAL lattice, so wrapping a perturbed point changes its stored
        # coordinate by +/-L while its shifts stay put, and the triangle is then reconstructed a
        # whole box away. Measured: a mod here inverted 13 of 72 triangles with signed area -15.2
        # against a true triangle area of ~0.43 -- a reconstruction error masquerading as a
        # geometric one, and it fired even at eta = 0.05 because only the points near x = 0 wrap.
        # `geo_from_simplices` reads positions as `pts[base] + shift*box`, so points a little
        # outside [0,L) are perfectly well defined; the periodicity lives in the shifts, not in the
        # stored coordinates being in-box.
        rng = np.random.default_rng(seed)
        if disp_fn is None:
            a = rng.uniform(0, 2 * np.pi, len(pts))
            d = np.stack([np.cos(a), np.sin(a)], 1)          # white noise: zero correlation length
        else:
            # LONG-RANGE disorder: the caller supplies the displacement field, so a correlated
            # (long-wavelength) perturbation can be applied without importing `m2.fields` here --
            # `m2` depends on this module, so the reverse import would be circular.
            #
            # It must be applied HERE and not via `m2.fields.displace`, which ends with
            # `np.mod(pts + d, box)`. Wrapping is right for the point-cloud families (they are
            # re-triangulated afterwards) and WRONG for a frozen-connectivity lattice, for exactly
            # the reason documented just above.
            d = np.asarray(disp_fn(pts, rng, Lx, Ly), float)
            if d.shape != pts.shape:
                raise ValueError('disp_fn returned %s, expected %s' % (d.shape, pts.shape))
        pts = pts + eta * d

    geo = geo_from_simplices(pts, np.asarray(tris, np.int64), Lx, Ly)
    if name is None:
        name = 'bravais_phi%g_psi%g_%s_r%d_eta%g%s_s%d' % (phi, psi, diagonal, reps, eta,
                                                          disp_tag, seed)
    rec = make_record(name, geo)
    rec.update(mesh_ok=bool(MB.check_mesh_preconditions(geo, periodic=True)[0]),
               phi=float(phi), psi=float(psi), diagonal=diagonal, eta=float(eta))
    return rec


# ---- 6b. UNIT CELLS WITH AN N-POINT BASIS — the crystalline covering ---------------------------
#
# `M2_V2_PLAN.md` §3.1e: the UNIT CELL is the fundamental object and everything else is repetition,
# so a systematic sweep over small cells with N-node bases is not a cheap corner of the space but a
# near-complete covering of the CRYSTALLINE part of it (N=1 Bravais, N=2 honeycomb-like,
# N=3 kagome-like, N>=4 the rest).  Verified premise: `C_eff` is INTENSIVE, so the minimal cell and
# any supercell of it give the same answer to 5e-16..7e-13
# (`Phase 5/results/supercell_invariance/SUPERCELL_INVARIANCE.md`).
#
# MEASURED 2026-08-25, and it is why this generator exists at all:
#
#   * BASIS SIZE IS THE MINIMAL-CELL SIZE KNOB.  A triangulated torus has V - E + F = 0 and
#     2E = 3F, so a cell with N basis nodes has **E = 3N bonds and F = 2N triangles**, exactly
#     (checked N = 1..16).  N=1 -> 2 triangles, N=12 -> 24.  That is precisely §3.1c's "minimal
#     (2-20 tri)" band, so one generator serves both the covering and the minimal-cell anchor.
#   * `make_lattice` CANNOT REACH THIS.  Its smallest output is 16 nodes / 32 triangles, so the
#     minimal cells are new reach, not a re-parametrisation of the existing pool.
#   * THE ANALYTIC ANCHORS ARE W=0 CELLS.  `W == 0` is a property of SYMMETRIC cells, not of small
#     ones: the braced rectangle (N=1) and the regular triangular lattice (N=2 in its rectangular
#     representation, nu=1/3 and E=2/sqrt(3) to 1e-16 on FOUR triangles) are both exactly affine,
#     while GENERIC random-basis cells at the same N have max|W| = O(1..10).  So minimal cells do
#     carry the non-affine content the head must learn, and the affine ones are the gates.
#   * SCALAR nu HIDES THEM.  The N=1 braced rectangle has nu(theta) spanning [0, +1] while its
#     DIRECTION-AVERAGED nu is 0 for every k tested (three orders) and every aspect ratio.  Label
#     these cells by the TENSOR (decision D2), never by scalar nu, or the family looks degenerate
#     when it is merely anisotropic.
#
# Oblique Bravais lattices need no separate knob: `_periodic_delaunay` takes a RECTANGLE, so an
# oblique lattice is just a basis inside a rectangular box and the position sampling covers it.

CELL_AREA_PER_NODE = np.sqrt(3.0) / 2.0     # area per node of the unit-spacing triangular lattice


def cell_box(n_basis, aspect):
    """Rectangular box holding `n_basis` nodes at ~unit spacing, with Ly/Lx = `aspect`.

    Spacing ~1 keeps edge lengths ~1 and so keeps `E` comparable across the pool (nu is
    scale-invariant, E is not)."""
    area = float(n_basis) * CELL_AREA_PER_NODE
    Lx = np.sqrt(area / float(aspect))
    return Lx, float(aspect) * Lx


def _min_periodic_gap(pts, Lx, Ly):
    """Smallest minimum-image distance between any two basis points (inf for a single point).

    Rejecting tight pairs is what keeps the periodic Delaunay from producing slivers, which is the
    geometric route to `A(s)` rank loss (`CLAUDE.md` §3)."""
    n = len(pts)
    if n < 2:
        return np.inf
    d = pts[:, None, :] - pts[None, :, :]
    d[..., 0] -= Lx * np.round(d[..., 0] / Lx)          # minimum image
    d[..., 1] -= Ly * np.round(d[..., 1] / Ly)
    r = np.hypot(d[..., 0], d[..., 1])
    iu = np.triu_indices(n, 1)
    return float(r[iu].min())


def _degree_signature(geo):
    """Sorted degree sequence of the periodic graph — a cheap, label-independent topology tag.

    Used only as a COVERAGE DIAGNOSTIC (how many distinct connectivities the sampler reached), not
    to de-duplicate: basis positions are continuous, so two cells sharing a degree sequence are
    still different networks and both are legitimate samples."""
    u = np.asarray(geo['bond_u']); v = np.asarray(geo['bond_v'])
    deg = np.zeros(len(geo['pts']), int)
    np.add.at(deg, u, 1)
    np.add.at(deg, v, 1)
    return tuple(sorted(deg.tolist()))


def _place_basis(n_basis, Lx, Ly, rng, min_sep, tries=200):
    """Dart-throwing (periodic Poisson-disk) placement of `n_basis` points at >= `min_sep` apart.

    Uniform i.i.d. sampling with a fixed rejection radius FAILS at large N — measured acceptance
    100 % at N<=6 but 33 % at N=9, **0 % at N=10** and 8 % at N=12, because a uniform draw almost
    never happens to be well separated once the box is crowded. Left in, that would have biased the
    pool silently toward small cells, which is the §3.1f trap (a reject-and-resample sampler whose
    acceptance rate is not reported yields a biased subset). Placing points one at a time against
    the accepted set instead succeeds at every N in range."""
    box = np.array([Lx, Ly])
    pts = np.zeros((0, 2))
    for _ in range(n_basis):
        for _t in range(tries):
            cand = rng.random(2) * box
            if len(pts):
                d = cand - pts
                d -= box * np.round(d / box)                       # minimum image
                if np.hypot(d[:, 0], d[:, 1]).min() < min_sep:
                    continue
            pts = np.vstack([pts, cand])
            break
        else:
            return None                                            # could not place them all
    return pts


#: smallest basis the sweep uses. N=1 and N=2 are EXCLUDED by decision (user, 2026-08-25) and the
#: measurements back it: self-loops (a node bonded to its own periodic image, `u == v`) occur ONLY
#: at N=1 (3 of 3 bonds) and N=2 (2 of 6) and are ZERO for N>=3, and those two sizes are the ONLY
#: ones that make the solver's angle/curvature code emit `invalid value` warnings. A 1-vertex cell
#: is a genuine Bravais crystal physically, but as a GRAPH it is all self-loops and no neighbours —
#: nothing for message passing to do. Flooring at 3 removes all three problems at once.
N_BASIS_MIN = 3


def seed_cell(n_basis, seed=0, aspect=1.0, min_sep_frac=0.55, max_tries=60, quality_floor=0.05):
    """ONE unit cell with an `n_basis`-point basis, or None if no healthy draw was found.

    Samples fractional basis positions, rejects tight pairs (`min_sep_frac` x unit spacing), builds
    the periodic Delaunay, and accepts only if the mesh passes the A-17 preconditions
    (`check_mesh_preconditions`) AND its worst triangle clears `quality_floor`.  Returns a uniform
    record with `mesh_ok=True`, plus `n_basis`, `aspect`, `degree_signature` and `tries` in it.

    `quality_floor` is a SHAPE floor (`positions.tri_shape_quality`), not a stiffness floor — it
    guards the sliver route to rank loss, and §3.1f's rule is to gate on shape rather than on a
    scalar disorder amplitude, because a given displacement is not equally safe on every mesh."""
    import positions as POS          # LAZY: seeds -> positions -> designer -> seeds is a cycle
    if n_basis < N_BASIS_MIN:
        raise ValueError(f'n_basis={n_basis} < N_BASIS_MIN={N_BASIS_MIN}; see the note there')
    Lx, Ly = cell_box(n_basis, aspect)
    min_sep = min_sep_frac * np.sqrt(CELL_AREA_PER_NODE)
    rng = np.random.default_rng(seed)
    for t in range(max_tries):
        pts = _place_basis(n_basis, Lx, Ly, rng, min_sep)
        if pts is None:
            continue
        try:
            geo = C._periodic_delaunay(pts, Lx, Ly)
        except Exception:                                # noqa: BLE001 — degenerate draw, resample
            continue
        if not MB.check_mesh_preconditions(geo, periodic=True)[0]:
            continue
        if float(np.min(POS.tri_shape_quality(geo))) < quality_floor:
            continue
        rec = make_record(f'cell_N{n_basis}_a{aspect:.2f}_s{seed}', geo)
        rec.update(mesh_ok=True, n_basis=int(n_basis), aspect=float(aspect),
                   degree_signature=_degree_signature(geo), tries=t + 1)
        return rec
    return None


def seed_cells(n_basis_range=range(N_BASIS_MIN, 13), n_cfg=6, aspects=(0.75, 1.0, 1.35),
               seed=0, report=None, **kw):
    """The crystalline covering: `n_cfg` accepted cells per (basis size, aspect).

    Yields uniform records.  `n_basis_range` defaults to **3..12**, i.e. **6..24 triangles** — the
    floor is `N_BASIS_MIN` (see the note there) and the ceiling is where distinct connectivities
    stop being cheap and `random_patch` takes over.

    If `report` is a dict it receives the ACCEPTANCE RATE per basis size and the number of distinct
    degree signatures reached.  Reporting it is not optional bookkeeping: §3.1f records that a
    reject-and-resample sampler whose acceptance rate goes unreported yields a silently BIASED
    subset, the same trap as an eta-sweep that does not say how many seeds survived."""
    stats = {}
    for N in n_basis_range:
        asked = made = 0
        sigs = set()
        for aspect in aspects:
            for j in range(n_cfg):
                asked += 1
                rec = seed_cell(N, seed=seed + 1000 * N + 37 * j + int(100 * aspect),
                                aspect=aspect, **kw)
                if rec is None:
                    continue
                made += 1
                sigs.add(rec['degree_signature'])
                yield rec
        stats[N] = dict(asked=asked, made=made,
                        accept=(made / asked if asked else 0.0), n_signatures=len(sigs))
    if report is not None:
        report.update(stats)


class _TriObj:
    """Minimal stand-in for a scipy triangulation: `build_open_mesh` / `DesignProblem.open` only
    ever read `.points` and `.simplices`."""

    def __init__(self, points, simplices):
        self.points = np.asarray(points, float)
        self.simplices = np.asarray(simplices, np.int64)


def anchor_single_triangle(shape='equilateral', k=(1.0, 1.0, 1.0)):
    """THE analytic anchor for the head: ONE triangle, 3 nodes, OPEN (non-periodic).

    This is the smallest object the per-triangle head must get right, and it has a CLOSED FORM with
    no solve in it at all:

        C(s) = A(s) = sum_e (k_e / 4 l_e^2) q_e q_e^T ,      q_e = dx_e dx_e^T   (vec3)

    because a single triangle has no neighbour to be incompatible with, so `W == 0` identically and
    `C(s) = (1+W)^T A(s) (1+W)` collapses to `A(s)`.  In the §2.1 parametrisation
    `C(s) = Q(s) MM^T Q(s)^T` with `Q = [q1 q2 q3]`, that is the sharp prediction that **`MM^T` is
    DIAGONAL with entries `k_e/4 l_e^2`** — the S1 gate.  Everything the model has to LEARN lives in
    the off-diagonals, which are exactly the non-affine content, and which are zero here.

    Three shapes, because the gate must not be satisfiable by an equilateral special case alone:
    `equilateral` (the symmetric reference), `right` and `scalene` (general `q_e`, so `Q` is a
    genuinely generic basis of vec3).  `k` is per-edge in `build_open_mesh`'s edge order
    (0,1), (0,2), (1,2), so a non-uniform `k` makes the predicted diagonal non-uniform too."""
    pts = {'equilateral': [(0.0, 0.0), (1.0, 0.0), (0.5, np.sqrt(3) / 2)],
           'right':       [(0.0, 0.0), (1.0, 0.0), (0.0, 0.8)],
           'scalene':     [(0.0, 0.0), (1.3, 0.0), (0.35, 0.9)]}[shape]
    tri = _TriObj(pts, [[0, 1, 2]])
    rec = dict(name=f'anchor_single_triangle_{shape}', tri=tri, geo=None,
               k0=np.asarray(k, float), is_fictional=np.zeros(3, bool),
               periodic=False, mesh_ok=True, n_basis=3,
               anchor='C_equals_A_closed_form')
    return rec


def seed_cells_anchors():
    """The cells with a KNOWN answer — the S1 gates, as records.  NOT training data.

    * `anchor_single_triangle_{equilateral,right,scalene}` — 3 nodes, ONE triangle, OPEN.
      `C(s) = A(s)` in closed form (see `anchor_single_triangle`).  The direct test of the head's
      per-triangle parametrisation, with no periodic images and no self-loops involved.
    * `anchor_triangular_N2` — the regular triangular lattice in its RECTANGULAR representation
      (Lx=1, Ly=sqrt(3)); FOUR triangles, `W == 0`, **nu = 1/3 and E = 2/sqrt(3) to 1e-16**.  The
      project's oldest sanity value, at the smallest cell that can express it — `make_lattice`
      cannot go below 16 nodes / 32 triangles.

    Both are affine (`W == 0`), which is what makes them gates rather than training samples.

    *(N counts vertices IN THE RECTANGULAR BOX, so it is a property of the representation, not of
    the material: the triangular lattice is a ONE-site Bravais crystal in oblique coordinates and
    N=2 here.  The `N_BASIS_MIN = 3` floor is therefore a floor on the representation, which is why
    this N=2 gate is exempt — it has a known answer and is never trained on.)*"""
    for shape in ('equilateral', 'right', 'scalene'):
        yield anchor_single_triangle(shape)

    r3 = np.sqrt(3.0)
    g = C._periodic_delaunay(np.array([[0.0, 0.0], [0.5, r3 / 2]]), 1.0, r3)
    r = make_record('anchor_triangular_N2', g)
    r.update(mesh_ok=True, n_basis=2, aspect=r3, anchor='nu_1_3', periodic=True,
             degree_signature=_degree_signature(g))
    yield r


# ---- 7. the bounded seed pool ----------------------------------------------------------------
def seed_pool(n_random=12, n_nodes=120,
              include=('bravais', 'random', 'tiling', 'basis', 'auxetic')):
    """A BROAD but BOUNDED generator of records across the families (Bravais subset + random
    patches over the 4 point processes + the 4 tilings + honeycomb/kagome bases + auxetic motifs).
    Every yielded item is a uniform record."""
    if 'bravais' in include:
        for name, geo in seed_bravais(phi_vals=(0.8, 1.0, 1.2), psi_vals=(0.8, 1.0),
                                      etas=(0.0, 0.15), seeds=(0,), half=6):
            yield make_record(name, geo)
    if 'random' in include:
        procs = ('uniform', 'poisson_disk', 'blue_noise', 'graded')
        for i in range(n_random):
            yield random_patch(n_nodes, seed=i, process=procs[i % len(procs)])
    if 'tiling' in include:
        reps_of = {'square': 4, 'honeycomb': 3, 'kagome': 3, 'square_octagon': 3}
        for tname in ('square', 'honeycomb', 'kagome', 'square_octagon'):
            yield seed_tiling(tname, reps_of[tname])
    if 'basis' in include:
        yield honeycomb(reps=3)
        yield kagome(reps=3)
    if 'auxetic' in include:
        yield from auxetic_motifs(reps=3)
    if 'phase4' in include:
        yield from seed_from_phase4()


# ---- self-test -------------------------------------------------------------------------------
def _nuE_of(geo, k):
    prob = DesignProblem.from_geo(geo)
    out = prob.forward(torch.as_tensor(np.asarray(k, float)))
    nu, E = c6_to_nuE(prob.region_tensor(out['per_triangle'], None))
    return float(nu), float(E)


def _self_test():
    print(f"{'name':44s} {'n_tri':>6} {'n_bond':>7} {'n_fic':>6} {'nu':>9} {'E':>9}")
    print('-' * 90)
    records = []
    for rec in seed_pool(n_random=4):
        records.append(rec)
        if len(records) >= 20:
            break

    for rec in records:
        geo = rec['geo']
        assert (geo['areas'] > 0).all(), f"{rec['name']}: non-positive areas"
        prob = DesignProblem.from_geo(geo)
        out = prob.forward(torch.as_tensor(rec['k0']))
        nu, E = (float(x) for x in c6_to_nuE(prob.region_tensor(out['per_triangle'], None)))
        n_fic = int(rec['is_fictional'].sum())
        print(f"{rec['name']:44s} {len(geo['simplices']):6d} {len(geo['bond_u']):7d} "
              f"{n_fic:6d} {nu:9.4f} {E:9.4f}")

    # tilings: confirm nonzero fictional bonds AND that soft edges matter (soft != fully-stiff)
    print('-' * 90)
    for tname in ('square', 'honeycomb', 'kagome', 'square_octagon'):
        rec = seed_tiling(tname, 3)
        n_fic = int(rec['is_fictional'].sum())
        assert n_fic > 0, f"{tname}: expected fictional bonds, got 0"
        nu_soft, E_soft = _nuE_of(rec['geo'], rec['k0'])
        nu_full, E_full = _nuE_of(rec['geo'], np.ones(len(rec['geo']['bond_u'])))
        diff = abs(nu_soft - nu_full) + abs(E_soft - E_full)
        assert diff > 1e-3, f"{tname}: soft-edge response == fully-triangulated (soft edges inert)"
        print(f"tiling {tname:16s} n_fic={n_fic:4d}  soft(nu={nu_soft:.4f},E={E_soft:.4f})  "
              f"full(nu={nu_full:.4f},E={E_full:.4f})  |diff|={diff:.4f}")

    # auxetic motifs: confirm they build (soft edges) and register a distinct response
    print('-' * 90)
    for rec in auxetic_motifs():
        geo = rec['geo']
        assert (geo['areas'] > 0).all(), f"{rec['name']}: non-positive areas"
        n_fic = int(rec['is_fictional'].sum())
        assert n_fic > 0, f"{rec['name']}: expected fictional bonds, got 0"
        nu_soft, E_soft = _nuE_of(geo, rec['k0'])
        nu_full, E_full = _nuE_of(geo, np.ones(len(geo['bond_u'])))
        diff = abs(nu_soft - nu_full) + abs(E_soft - E_full)
        assert diff > 1e-3, f"{rec['name']}: soft-edge response == fully-triangulated"
        print(f"{rec['name']:40s} n_fic={n_fic:4d}  soft(nu={nu_soft:.4f},E={E_soft:.4f})  "
              f"full(nu={nu_full:.4f},E={E_full:.4f})")

    # save one example of each tiling for gallery.py
    ndir = os.path.join(os.path.dirname(__file__), 'networks')
    os.makedirs(ndir, exist_ok=True)
    for tname in ('square', 'honeycomb', 'kagome', 'square_octagon'):
        rec = seed_tiling(tname, 3)
        C.apply_k_to_geo(rec['geo'], rec['k0'])
        path = os.path.join(ndir, f'seed_{tname}.npz')
        C.save_network(path, rec['geo'], rec['k0'], is_fictional=rec['is_fictional'].tolist())
        print(f"saved {path}")

    print("SEEDS SELF-TEST PASSED")


if __name__ == '__main__':
    _self_test()
