"""Phase 5 — seed zoo generator (the RICH, physically-structured starting-network set).

Produces a broad but bounded pool of periodic triangulated networks for the M1 designer /
M2 GNN training set.  Five physically-grounded ingredients:

    Bravais family      -> seed_bravais          (make_lattice scan, disorder overlaid)
    random patches      -> random_patch          (uniform / Poisson-disk / blue-noise / graded)
    complex-basis nets   -> seed_with_basis / honeycomb / kagome
    Archimedean tilings  -> seed_tiling           (square / honeycomb / kagome / square_octagon,
                                                    as triangulations with SOFT 'fictional' edges)
    auxetic motifs       -> auxetic_motifs         (rotating squares, re-entrant honeycomb)

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


def _native_adjacency(pts, box, bond_len, tol=1e-3):
    """i -> [(j, d)] for native edges, `d` the minimal-image displacement i->j."""
    adj = {i: [] for i in range(len(pts))}
    for i in range(len(pts)):
        d = pts - pts[i]
        d -= np.round(d / box) * box
        L = np.hypot(d[:, 0], d[:, 1])
        for j in np.where(np.abs(L - bond_len) < tol)[0]:
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


def _fan_and_tag(name, pts, native, Lx, Ly, bond_len, eps):
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
    adj = _native_adjacency(pts, box, bond_len)
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
    return _triangulate_and_tag(f'auxetic_reentrant_honeycomb_v{v}', pts, native, Lx, Ly, eps)


def auxetic_motifs(reps=4, eps=1e-3):
    """Yield classic auxetic/mechanism seeds (soft-edge construction).  reps is forced even inside
    each builder (both motifs need a commensurate periodic box)."""
    yield _rotating_squares(reps=reps, eps=eps)
    yield _reentrant_honeycomb(reps=reps, eps=eps)


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
