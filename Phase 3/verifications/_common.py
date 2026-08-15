"""
Shared harness for Phase 3 inverse-design verifications.

For every case we run a matrix of TOPOLOGIES x SIZES, design k with the inverse designer, then
INDEPENDENTLY simulate the designed network (PBC relaxation → physical per-triangle tensor) and
check it does as prescribed — globally and per-region. Results are plotted and saved per case in
Phase 3/verifications/<case>/.

Topologies (periodic): regular triangular, two non-symmetric (affine-stretched / sheared)
lattices, two disordered (perturbed) lattices. Sizes span the dense (<=600 tri) and adjoint
(>600 tri) solver paths.

Verification strength (be explicit in plots):
  - GLOBAL nu/E: the sim ground truth (energy = virial) is independent of the solver's
    homogenisation → a genuinely independent check.
  - LOCAL/regional nu: defined via the region-averaged per-triangle physical tensor; the sim
    uses the same definition, so it validates the design→realise→simulate loop and the spatial
    pattern (not a fully independent measurement of a sub-region's modulus).
"""
import os, sys
import numpy as np
import torch
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.spatial import Delaunay, cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
P3 = os.path.dirname(HERE)
ROOT = os.path.dirname(P3)
sys.path.insert(0, P3)
sys.path.insert(0, os.path.join(ROOT, 'Phase 2'))
sys.path.insert(0, os.path.join(ROOT, 'verification_tools'))
sys.path.insert(0, ROOT)
import forward_solver_torch as fst
import physical_homog as PH
from inverse_design import (DesignProblem, Objective, optimize, validate, c6_to_nuE,
                            c6_to_nu_theta, c6_to_E_theta, ANG, constrain, isotropic_c6,
                            per_triangle_strain_stress, region_mean_vec3)
import metric_ops as MO
import mesh_build as MB
import sim_assembly as SA
torch.set_default_dtype(torch.float64)

# ---- topology x size matrix ------------------------------------------------------------------
TOPOS = [
    ('regular',     'regular (phi=psi=1)',  None),
    ('aniso_str',   'non-sym (psi=0.6)',    None),
    ('aniso_shr',   'non-sym (phi=1.5)',    None),
    ('disorder_lo', 'disordered eta=0.20',  None),
    ('disorder_hi', 'disordered eta=0.35',  None),
]
TOPO_IDS = [t[0] for t in TOPOS]
SIZES = [8, 12]                                          # square half-size -> ~600 / ~1300 triangles
_TOPO_PARAMS = {                                         # (phi, psi, eta) for the preferred make_lattice
    'regular':     (1.0, 1.0, 0.00),
    'aniso_str':   (1.0, 0.6, 0.00),                     # rows compressed in y
    'aniso_shr':   (1.5, 1.0, 0.00),                     # sheared rows
    'disorder_lo': (1.0, 1.0, 0.20),
    'disorder_hi': (1.0, 1.0, 0.35),
}
# legacy affine constants (used only by make_topology_affine, kept for reference)
_L1 = np.array([1.0, 0.0]); _L2 = np.array([0.5, np.sqrt(3) / 2])
_TOPO_M = {'regular': np.eye(2), 'aniso_str': np.diag([1.5, 0.8]),
           'aniso_shr': np.array([[1.0, 0.4], [0.0, 1.0]]),
           'disorder_lo': np.eye(2), 'disorder_hi': np.eye(2)}


def _affine(geo, M):
    geo['pts'] = geo['pts'] @ M.T
    geo['edge_vecs'] = geo['edge_vecs'] @ M.T
    geo['bond_R'] = geo['bond_R'] @ M.T
    geo['actual_len2'] = (geo['edge_vecs'] ** 2).sum(2)
    geo['areas'] = geo['areas'] * abs(np.linalg.det(M))
    return geo


def make_topology_affine(topo_id, N, seed=0):
    """DEPRECATED (kept for reference). Old affine-transform topologies whose parallelogram box is
    hidden by the square/fractional view. Superseded by make_lattice / make_topology below."""
    eta = {'regular': 0.0, 'aniso_str': 0.0, 'aniso_shr': 0.0,
           'disorder_lo': 0.20, 'disorder_hi': 0.35}[topo_id]
    M = _TOPO_M[topo_id]
    geo = MB.build_geometry(N, eta, seed=seed)
    if not np.allclose(M, np.eye(2)):
        geo = _affine(geo, M)
    MB.set_VD(geo, 0)
    geo['BL1'] = M @ (N * _L1); geo['BL2'] = M @ (N * _L2)
    return geo


# ---- PREFERRED lattice constructor (square real-space region, PBC via periodic Delaunay) -----
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


def _bond_kv(pts, bond_u, bond_v, bond_R, k):
    """dict {sorted(rounded endpoint pair): k} for one piece's own designed bonds, in the frame it
    will occupy in the combined geometry (pts already positioned/shifted there)."""
    pA = np.asarray(pts)[bond_u]; pB = pA + np.asarray(bond_R)
    return {tuple(sorted([tuple(a), tuple(b)])): float(kk)
            for a, b, kk in zip(np.round(pA, 5), np.round(pB, 5), np.asarray(k))}


def glue(pieces, Lx, Ly):
    """Physically GLUE several INDEPENDENTLY-designed periodic patches into one combined geometry,
    instead of jointly optimising one connected lattice (which lets the optimiser exploit shared
    interface bonds as a zero-stiffness mechanism). Each piece is a dict with:
      'pts_keep' - this piece's points, already positioned in the combined frame, to include in the
                   combined point cloud (may be a subset, e.g. a matrix with a hole punched out)
      'pts_full' - ALL of this piece's own points in the combined frame (bond_u/bond_v index into
                   this; used only to resolve each of its bonds' real endpoint positions)
      'bond_u','bond_v','bond_R','k' - the piece's own designed bonds/stiffnesses
    The union of every piece's pts_keep is retriangulated (periodic Delaunay over Lx,Ly). Each new
    bond gets the k of a matching ORIGINAL bond (same real endpoint positions, from any piece);
    unmatched bonds (the new interface seam) default to k=1 -- a plain undesigned "glue" spring,
    since no piece's own optimisation ever touched them. Returns (geo, glued_mask)."""
    pts_union = np.concatenate([p['pts_keep'] for p in pieces], axis=0)
    geo = _periodic_delaunay(pts_union, Lx, Ly)
    kv = {}
    for p in pieces:
        kv.update(_bond_kv(p['pts_full'], p['bond_u'], p['bond_v'], p['bond_R'], p['k']))
    ptsU = geo['pts']; pA = ptsU[geo['bond_u']]; pB = pA + geo['bond_R']
    keys = [tuple(sorted([tuple(a), tuple(b)])) for a, b in zip(np.round(pA, 5), np.round(pB, 5))]
    k_new = np.array([kv.get(key, 1.0) for key in keys])
    geo['bond_k'] = k_new; geo['tri_k'] = k_new[geo['tri_bond']]
    return geo, np.array([key not in kv for key in keys])


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


def make_crystal(phi, psi, half=4.0):
    """Single-site triangular crystal SHEARED/stretched by (phi, psi) with its BOND TOPOLOGY PRESERVED.
    Unlike make_lattice (whose periodic-Delaunay retriangulation reduces the basis to nearest
    neighbours — so a large shift v2=(phi/2,·) collapses back to v2-v1 and the shear is lost), this
    starts from the regular triangular lattice and applies the affine map M=[[1,(phi-1)/√3],[0,psi]]
    that carries v1=(1,0),v2=(1/2,√3/2) to v1=(1,0),v2=(phi/2, psi·√3/2). The bonds v1, v2, v2-v1 are
    kept at their true (possibly long, sheared) lengths, so a genuinely oblique crystal results
    (nonzero shear-normal coupling C_xxxy). Still a Bravais lattice → W=0 (affine strain IS the
    equilibrium), so the homogenised response is exact on a tiny patch."""
    geo = make_lattice(1.0, 1.0, half=half)             # regular triangular; correct NN bond topology
    M = np.array([[1.0, (phi - 1.0) / np.sqrt(3)], [0.0, psi]])
    for key in ('pts', 'edge_vecs', 'bond_R', 'centroids', 'tri_verts'):
        geo[key] = geo[key] @ M.T                        # affine-map every geometric field (topology kept)
    geo['actual_len2'] = (geo['edge_vecs'] ** 2).sum(-1)
    geo['areas'] = geo['areas'] * abs(np.linalg.det(M))
    geo['BL1'] = M @ geo['BL1']; geo['BL2'] = M @ geo['BL2']
    return geo


def make_topology(topo_id, half, seed=0):
    """Named topology at square half-size `half`, built with the preferred make_lattice
    (square real-space PBC region; anisotropy visible in the real geometry)."""
    phi, psi, eta = _TOPO_PARAMS[topo_id]
    return make_lattice(phi, psi, half=half, seed=seed, eta=eta)


# ---- square / fractional plotting (periodic wrap) --------------------------------------------
def _box(geo):
    return np.column_stack([geo['BL1'], geo['BL2']])    # 2x2, columns = box vectors


def to_square(geo, pts):
    """Map real positions to fractional lattice coords in [0,1)^2 (periodic wrap → square region)."""
    frac = np.linalg.solve(_box(geo), np.atleast_2d(np.asarray(pts)).T).T
    return frac % 1.0


def square_frame(ax, geo, pad=0.04):
    """Real geometry, but force a SQUARE axes frame that bounds the periodic box (undistorted:
    equal aspect, so a stretched/sheared lattice fills a parallelogram within the square)."""
    corners = np.array([[0, 0], geo['BL1'], geo['BL2'], geo['BL1'] + geo['BL2']], float)
    cx, cy = corners.mean(0)
    half = 0.5 * max(np.ptp(corners[:, 0]), np.ptp(corners[:, 1])) * (1 + pad)
    ax.set_xlim(cx - half, cx + half); ax.set_ylim(cy - half, cy + half)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])


def draw_box(ax, geo, **kw):
    B1, B2 = geo['BL1'], geo['BL2']
    poly = np.array([[0, 0], B1, B1 + B2, B2, [0, 0]], float)
    ax.plot(poly[:, 0], poly[:, 1], color=kw.pop('color', '0.55'), lw=kw.pop('lw', 0.9),
            ls=kw.pop('ls', '--'), zorder=1)


def draw_network(ax, geo, k_bond, cmap='viridis', lw_scale=3.0, box=True):
    """Draw the network in REAL geometry (per-bond line color AND width ∝ rigidity k), inside a
    square axes frame. Bonds crossing the periodic boundary appear as short stubs at the edges."""
    from matplotlib.collections import LineCollection
    u = geo['pts'][geo['bond_u']]
    segs = np.stack([u, u + geo['bond_R']], axis=1)      # real-coordinate segments
    k = np.asarray(k_bond, float)
    lw = 0.25 + lw_scale * k / (k.max() + 1e-12)
    lc = LineCollection(segs, array=k, cmap=cmap, linewidths=lw, zorder=2)
    ax.add_collection(lc)
    if box:
        draw_box(ax, geo)
    square_frame(ax, geo)
    return lc


def draw_lattice_zoom(ax, geo, color='0.2', cells=3.0, lw=1.4):
    """Draw a (crystal) geo inside a CENTRED SQUARE window ~`cells` lattice-constants wide, drawing ALL
    interior bonds and letting the window EDGES CUT them (clipped) — so the square is filled edge-to-edge
    with no missing elements/voids. Centred in the interior, away from the periodic-boundary stubs."""
    from matplotlib.collections import LineCollection
    p0 = geo['pts'][geo['bond_u']]; p1 = p0 + geo['bond_R']
    c = geo['pts'].mean(0)
    L = float(np.sqrt(geo['actual_len2']).min())         # shortest bond ≈ lattice constant
    ext = 0.5 * cells * L
    near = (np.abs(p0[:, 0] - c[0]) <= ext + L) & (np.abs(p0[:, 1] - c[1]) <= ext + L)  # keep local (fast)
    lc = LineCollection(np.stack([p0[near], p1[near]], 1), colors=color, linewidths=lw)
    ax.add_collection(lc)                                # clipped to axes bbox -> edges cut the bonds
    ax.set_xlim(c[0] - ext, c[0] + ext); ax.set_ylim(c[1] - ext, c[1] + ext)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])


def local_field_smooth(geo, C6_per, quantity='nu', k=18):
    """Per-triangle local ν or E, each triangle = physical homogenisation of its k nearest
    neighbours (smooth field for FILLED-triangle maps). Returns (nt,) values."""
    cen = geo['centroids']
    kk = min(k, len(cen))
    _, idx = cKDTree(cen).query(cen, k=kk)
    if kk == 1:
        idx = idx[:, None]
    C6m = C6_per[idx].mean(1)                                    # (nt,6) neighbourhood mean
    fac = 8.0 * kk / geo['areas'][idx].sum(1)                    # per-triangle physical factor
    C = C6m * fac[:, None]
    nu = (C[:, 2]*C[:, 3] - C[:, 1]*C[:, 4]) / (C[:, 0]*C[:, 3] - C[:, 1]**2)
    E = (C[:, 2]**2*C[:, 3] - 2*C[:, 1]*C[:, 2]*C[:, 4] + C[:, 1]**2*C[:, 5]
         + C[:, 0]*(C[:, 4]**2 - C[:, 3]*C[:, 5])) / (C[:, 1]**2 - C[:, 0]*C[:, 3])
    return nu if quantity == 'nu' else E


def local_nuE_angleavg(geo, C6_per, k=18):
    """Per-triangle ANGLE-AVERAGED Poisson ratio ⟨ν(θ)⟩ and Young's modulus ⟨E(θ)⟩ -- a
    load-independent MATERIAL map. Neighbourhood-smooth the local physical C6 (as local_field_smooth),
    invert to compliance, then average ν(θ)/E(θ) over ANG. Vectorised (nu_E_theta's 4-tensor
    contraction, batched over triangles). Returns (nu_avg, E_avg), each (nt,)."""
    cen = np.asarray(geo['centroids']); kk = min(k, len(cen))
    _, idx = cKDTree(cen).query(cen, k=kk)
    if kk == 1:
        idx = idx[:, None]
    C6m = np.asarray(C6_per)[idx].mean(1)
    Cphys = C6m * (8.0 * kk / np.asarray(geo['areas'])[idx].sum(1))[:, None]
    nt = len(cen)
    Cv = np.zeros((nt, 3, 3))                                    # Voigt stiffness (0=xx,1=yy,2=xy)
    Cv[:, 0, 0] = Cphys[:, 0]; Cv[:, 0, 1] = Cv[:, 1, 0] = Cphys[:, 2]; Cv[:, 0, 2] = Cv[:, 2, 0] = Cphys[:, 1]
    Cv[:, 1, 1] = Cphys[:, 5]; Cv[:, 1, 2] = Cv[:, 2, 1] = Cphys[:, 4]; Cv[:, 2, 2] = Cphys[:, 3]
    with np.errstate(all='ignore'):
        S = np.linalg.pinv(Cv)                                   # compliance (pinv: robust to a near-
    Sc = np.zeros((nt, 2, 2, 2, 2))                              # singular floppy triangle)
    Sc[:, 0, 0, 0, 0] = S[:, 0, 0]; Sc[:, 1, 1, 1, 1] = S[:, 1, 1]
    Sc[:, 0, 0, 1, 1] = Sc[:, 1, 1, 0, 0] = S[:, 0, 1]
    for a, b, c, d in [(0, 0, 0, 1), (0, 0, 1, 0), (0, 1, 0, 0), (1, 0, 0, 0)]:
        Sc[:, a, b, c, d] = S[:, 0, 2] / 2
    for a, b, c, d in [(1, 1, 0, 1), (1, 1, 1, 0), (0, 1, 1, 1), (1, 0, 1, 1)]:
        Sc[:, a, b, c, d] = S[:, 1, 2] / 2
    for a, b, c, d in [(0, 1, 0, 1), (0, 1, 1, 0), (1, 0, 0, 1), (1, 0, 1, 0)]:
        Sc[:, a, b, c, d] = S[:, 2, 2] / 4
    nu_sum = np.zeros(nt); E_sum = np.zeros(nt)
    with np.errstate(all='ignore'):
        for th in ANG:
            m = np.array([np.cos(th), np.sin(th)]); n = np.array([-np.sin(th), np.cos(th)])
            Emm = np.einsum('nijkl,i,j,k,l->n', Sc, m, m, m, m)
            Emn = np.einsum('nijkl,i,j,k,l->n', Sc, m, m, n, n)
            nu_sum += -Emn / Emm; E_sum += 1.0 / Emm
    return nu_sum / len(ANG), E_sum / len(ANG)


def fill_local_map(ax, geo, val, cmap='RdBu_r', sym=False, vlim=None):
    """FILLED per-triangle map (colored triangle interiors, image-correct) in a square frame."""
    from matplotlib.collections import PolyCollection
    pc = PolyCollection(list(geo['tri_verts']), array=np.asarray(val), cmap=cmap, edgecolors='none')
    if sym:
        v = vlim if vlim is not None else np.nanpercentile(np.abs(val), 97)
        pc.set_clim(-v, v)
    ax.add_collection(pc)
    square_frame(ax, geo)
    return pc


def mark_region(ax, region):
    """Outline a target region (or a LIST of them). Supported specs:
    {'kind':'circle','center','radius'} · {'kind':'rect','center','w','h'} ·
    {'kind':'ring','center','r_in','r_out'} · {'kind':'polygon','verts':[(x,y),...]} ·
    {'kind':'vlines','xs':[...]}. Optional 'color' (default lime)."""
    if not region:
        return
    if isinstance(region, (list, tuple)):
        for r in region:
            mark_region(ax, r)
        return
    import matplotlib.pyplot as plt
    col = region.get('color', 'lime'); k = region['kind']
    if k == 'circle':
        ax.add_patch(plt.Circle(region['center'], region['radius'], facecolor='none',
                                edgecolor=col, lw=2.0, zorder=6))
    elif k == 'rect':
        cx, cy = region['center']; w, h = region['w'], region['h']
        ax.add_patch(plt.Rectangle((cx - w / 2, cy - h / 2), w, h, facecolor='none',
                                   edgecolor=col, lw=2.0, zorder=6))
    elif k == 'ring':
        cx, cy = region['center']
        for rr in (region['r_in'], region['r_out']):
            ax.add_patch(plt.Circle((cx, cy), rr, facecolor='none', edgecolor=col, lw=2.0, zorder=6))
    elif k == 'polygon':
        ax.add_patch(plt.Polygon(region['verts'], closed=True, facecolor='none',
                                 edgecolor=col, lw=2.0, zorder=6))
    elif k == 'vlines':
        for x in region['xs']:
            ax.axvline(x, color=col, lw=1.0, ls='--', zorder=6)


def region_shape(prob, spec):
    """Triangle indices whose CENTROID falls in a shape spec (same schema as mark_region's shapes).
    Returns (indices, spec) so the caller can pass `spec` straight to mark_region."""
    c = prob.centroids; k = spec['kind']
    if k == 'circle':
        cen = np.asarray(spec['center']); idx = np.where(((c - cen) ** 2).sum(1) < spec['radius'] ** 2)[0]
    elif k == 'rect':
        cx, cy = spec['center']
        idx = np.where((np.abs(c[:, 0] - cx) <= spec['w'] / 2) & (np.abs(c[:, 1] - cy) <= spec['h'] / 2))[0]
    elif k == 'ring':
        cen = np.asarray(spec['center']); r2 = ((c - cen) ** 2).sum(1)
        idx = np.where((r2 >= spec['r_in'] ** 2) & (r2 < spec['r_out'] ** 2))[0]
    elif k == 'polygon':
        from matplotlib.path import Path
        idx = np.where(Path(np.asarray(spec['verts'])).contains_points(c))[0]
    else:
        raise ValueError(k)
    return idx, spec


def box(geo):
    """(Lx, Ly) -- the box edge lengths (BL1 along x, BL2 along y) as plain floats."""
    return float(geo['BL1'][0]), float(geo['BL2'][1])


def triangle_verts(cx, cy, s):
    """3 vertices of an upward-pointing equilateral-ish triangle of 'radius' s centered at (cx,cy),
    for a 'polygon' region_shape/mark_region spec."""
    return [(cx, cy + s), (cx - 0.87 * s, cy - 0.5 * s), (cx + 0.87 * s, cy - 0.5 * s)]


def write_csv(path, header, rows):
    import csv
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)


def local_scalar_field(geo, C6_per, quantity='nu', ncell=None):
    """Per-triangle local ν or E, binned on a fractional [0,1]^2 grid but returned at REAL
    triangle-centroid positions (so it plots on the real geometry). Returns (real_centroids, val).
    Grid resolution defaults to ~7 triangles/cell so the map fills."""
    cen = geo.get('centroids', geo['pts'][geo['simplices']].mean(1))
    if ncell is None:
        ncell = max(6, int(round(np.sqrt(len(cen) / 7.0))))
    fc = to_square(geo, cen)                             # fractional coords only for binning
    val = np.full(len(cen), np.nan)
    for i in range(ncell):
        for j in range(ncell):
            sel = np.where((fc[:, 0] >= i / ncell) & (fc[:, 0] < (i + 1) / ncell) &
                           (fc[:, 1] >= j / ncell) & (fc[:, 1] < (j + 1) / ncell))[0]
            if len(sel) >= 3:
                nu, E = c6_nuE(region_phys_C6(geo, C6_per, sel))
                val[sel] = nu if quantity == 'nu' else E
    return cen, val


def make_case(topo_id, N, seed=0):
    """(DesignProblem, geo). geo shares arrays with the solver; set geo['bond_k']/'tri_k' to the
    designed k before simulating."""
    geo = make_topology(topo_id, N, seed)
    prob = DesignProblem.from_geo(geo)
    return prob, geo


def apply_k_to_geo(geo, k_bond):
    """Install a designed per-bond k onto geo (for the independent simulation)."""
    k = k_bond.detach().numpy() if torch.is_tensor(k_bond) else np.asarray(k_bond)
    geo['bond_k'] = k
    geo['tri_k'] = k[geo['tri_bond']]


# ---- independent simulation of the designed network ------------------------------------------
_Fk = PH.Fk
_Dgt = [F.T @ F - np.eye(2) for F in _Fk]
_Dinv = np.linalg.inv(np.stack([MO.vec3(g) for g in _Dgt], 1))


def sim_per_triangle_C6(geo):
    """Per-triangle physical-response tensor (nt,6) in INTERNAL units, from one full PBC
    relaxation of geo (uses geo['tri_k']). Compute once, then query any region with
    region_phys_C6 (patch / outside / grid cells all share this relaxation)."""
    ev, sx = geo['edge_vecs'], geo['simplices']; nn = len(geo['pts']); nt = len(sx)
    u_modes = PH.relax(geo, np.arange(2, 2 * nn), SA.assemble_K_faff)
    D = np.zeros((nt, 3, 3))
    for k, (F, u) in enumerate(zip(_Fk, u_modes)):
        D[:, :, k] = MO.vec3(MO.tri_metric_change(ev, sx, F, u) - _Dgt[k])
    W3 = D @ _Dinv
    bare = MO.bare_tensor(geo)
    return fst._compute_actual_elastic_tensor(torch.as_tensor(bare),
                                              torch.as_tensor(W3.reshape(-1, 9))).numpy()


def region_phys_C6(geo, C6_per, region=None):
    """Physical homogenised 6-vector over a region from precomputed per-triangle tensors."""
    idx = np.arange(len(C6_per)) if region is None else np.asarray(region)
    return C6_per[idx].mean(0) * (8.0 * len(idx) / geo['areas'][idx].sum())


def sim_region_C6(geo, region=None):
    """Convenience: physical 6-vector over a region (one relaxation)."""
    return region_phys_C6(geo, sim_per_triangle_C6(geo), region)


def c6_nuE(C6):
    """nu, E from a 6-vector, same formula the solver/designer use."""
    nu, E = c6_to_nuE(torch.as_tensor(C6))
    return float(nu), float(E)


def sim_region_nuE(geo, region=None):
    return c6_nuE(sim_region_C6(geo, region))


def decoupled_ENu_design(prob, geo, n_iter, reg):
    """Design k so E differs only in a region R_E (cyan, target 1.8 vs background 1.0) and nu differs
    only in a DIFFERENT region R_nu (lime, target -0.30 vs background 0.20) -- independent spatial
    control of the two moduli. This is the 'auxetic_patch decoupled' recipe shared by
    design_and_verify.py's group3() and design_all.py's run_decoupled(). Installs k on geo (via
    apply_k_to_geo) and returns (k, C6_per, RE_spec, RN_spec, R_E_idx, R_N_idx, out_E_idx, out_N_idx)."""
    Lx, Ly = box(geo)
    RE = {'kind': 'circle', 'center': (0.30 * Lx, 0.50 * Ly), 'radius': 0.16 * Lx, 'color': 'cyan'}
    RN = {'kind': 'circle', 'center': (0.70 * Lx, 0.50 * Ly), 'radius': 0.16 * Lx, 'color': 'lime'}
    R_E, _ = region_shape(prob, RE); R_N, _ = region_shape(prob, RN)
    out_E = np.setdiff1d(np.arange(prob.n_tri), R_E); out_N = np.setdiff1d(np.arange(prob.n_tri), R_N)
    objs = [Objective('nu', 0.20, region=out_N, weight=3.0), Objective('nu', -0.30, region=R_N, weight=4.0),
            Objective('E', 1.0, region=out_E, weight=1.0), Objective('E', 1.8, region=R_E, weight=1.5)]
    r = optimize(prob, objs, mode='k', n_iter=n_iter, reg=reg, verbose=False)
    apply_k_to_geo(geo, r['k']); C6 = sim_per_triangle_C6(geo)
    return r['k'], C6, RE, RN, R_E, R_N, out_E, out_N


# ---- OPEN-boundary ground truth (periodic-vs-open counterpart to sim_per_triangle_C6 above) ---
def nonwrap_mask(geo):
    """Bonds/triangles that don't cross the periodic boundary (their real endpoint separation
    matches the stored bond vector) -- what's left after literally cutting geo open."""
    pts = np.asarray(geo['pts'])
    nwb = np.abs(np.asarray(geo['bond_R']) - (pts[geo['bond_v']] - pts[geo['bond_u']])).max(1) < 1e-6
    nwt = nwb[geo['tri_bond']].all(1)
    return nwb, nwt


def spring_K(npts, a, b, R, kap):
    """Assemble the (2*npts, 2*npts) axial-spring stiffness matrix for bonds (a,b) with rest
    vectors R and stiffness kap (each bond only resists stretch along its own direction)."""
    L = np.sqrt((R ** 2).sum(1)); nx, ny = R[:, 0] / L, R[:, 1] / L
    bxx, bxy, byy = kap * nx * nx, kap * nx * ny, kap * ny * ny
    a0, a1, b0, b1 = 2 * a, 2 * a + 1, 2 * b, 2 * b + 1
    r = np.concatenate([a0, a0, a1, a1, b0, b0, b1, b1, a0, a0, a1, a1, b0, b0, b1, b1])
    c = np.concatenate([a0, a1, a0, a1, b0, b1, b0, b1, b0, b1, b0, b1, a0, a1, a0, a1])
    v = np.concatenate([bxx, bxy, bxy, byy, bxx, bxy, bxy, byy,
                        -bxx, -bxy, -bxy, -byy, -bxx, -bxy, -bxy, -byy])
    return sp.coo_matrix((v, (r, c)), shape=(2 * npts, 2 * npts)).tocsr()


def open_stretch(geo, axis=0, regularize=False, margin=1.3):
    """Cut geo's periodic network into an OPEN sheet and solve a displacement-controlled uniaxial
    stretch: clamp the min-`axis` boundary to 0 and the max-`axis` boundary to 1 (linear -> any
    other stretch is this scaled), free everywhere else, relax as a linear spring truss. Returns
    (u, nwt) -- nodal displacement field (n,2) and the non-wrapping triangle mask (the open domain
    to actually plot/measure over; wrapping triangles were cut and are not physically meaningful).
    `regularize=True` adds a small (1e-3) identity term to pin floppy/dangling nodes -- off by
    default to match the majority of call sites; turn on for networks with under-constrained bonds."""
    nwb, nwt = nonwrap_mask(geo)
    pts = np.asarray(geo['pts']); n = len(pts)
    K = spring_K(n, geo['bond_u'][nwb], geo['bond_v'][nwb],
                  np.asarray(geo['bond_R'])[nwb], np.asarray(geo['bond_k'])[nwb])
    if regularize:
        K = K + 1e-3 * sp.identity(2 * n)
    lo = np.where(pts[:, axis] < pts[:, axis].min() + margin)[0]
    hi = np.where(pts[:, axis] > pts[:, axis].max() - margin)[0]
    fix = np.concatenate([2 * lo + axis, 2 * hi + axis])
    uf = np.concatenate([np.zeros(len(lo)), np.ones(len(hi))])
    u = np.zeros(2 * n); u[fix] = uf
    free = np.setdiff1d(np.arange(2 * n), fix)
    u[free] = spla.spsolve(K[free][:, free].tocsc(), -(K[free][:, fix] @ uf))
    return u.reshape(n, 2), nwt


def open_stretch_nu(geo, axis=0, regularize=False, margin=1.3):
    """nu from an open cut-and-stretch: axial strain from the clamped-boundary gauge (lo/hi mean
    coordinates), lateral strain from the mean displacement of the free top/bottom (or left/right,
    whichever is perpendicular to `axis`) edges. Returns (nu, u, nwt)."""
    u, nwt = open_stretch(geo, axis=axis, regularize=regularize, margin=margin)
    pts = np.asarray(geo['pts'])
    lo = np.where(pts[:, axis] < pts[:, axis].min() + margin)[0]
    hi = np.where(pts[:, axis] > pts[:, axis].max() - margin)[0]
    e_ax = 1.0 / (pts[hi, axis].mean() - pts[lo, axis].mean())
    lat = 1 - axis
    t = pts[:, lat] > pts[:, lat].max() - margin; b = pts[:, lat] < pts[:, lat].min() + margin
    e_lat = (u[t, lat].mean() - u[b, lat].mean()) / (pts[t, lat].mean() - pts[b, lat].mean())
    return -e_lat / e_ax, u, nwt


def glue_square_hole(geoM, kM, geoI, kI, Lx, Ly, cx, cy):
    """Punch a hole matching geoI's own box out of geoM's centre (at (cx,cy)) and glue() the two
    independently-built patches into one geometry, then open-cut-and-stretch it along x (regularized,
    matching two_region/demo.py's cut_stretch). The 'matrix with a square inclusion' recipe shared by
    two_region/inclusion_square.py and inclusion_rigidity_only.py. Returns (geo, glued_mask, spec,
    C6_per, disc_idx, out_idx, nu_disc, E_disc, nu_matrix, E_matrix, u, nwt); spec is the 'rect'
    region_shape/mark_region spec for the inclusion's footprint."""
    Lxi, Lyi = float(geoI['BL1'][0]), float(geoI['BL2'][1])
    ptsI = np.asarray(geoI['pts']) + [cx - Lxi / 2, cy - Lyi / 2]
    ptsM = np.asarray(geoM['pts'])
    hole = (np.abs(ptsM[:, 0] - cx) < Lxi / 2) & (np.abs(ptsM[:, 1] - cy) < Lyi / 2)
    ptsM_keep = ptsM[~hole]
    pieces = [dict(pts_keep=ptsM_keep, pts_full=ptsM, bond_u=geoM['bond_u'], bond_v=geoM['bond_v'],
                   bond_R=geoM['bond_R'], k=kM),
              dict(pts_keep=ptsI, pts_full=ptsI, bond_u=geoI['bond_u'], bond_v=geoI['bond_v'],
                   bond_R=geoI['bond_R'], k=kI)]
    geo, glued = glue(pieces, Lx, Ly)
    spec = {'kind': 'rect', 'center': (float(cx), float(cy)), 'w': Lxi, 'h': Lyi, 'color': 'lime'}
    print(f"    GLUED: {glued.sum()} default/interface bonds ({glued.mean()*100:.1f}%)", flush=True)
    C6 = sim_per_triangle_C6(geo); cen = np.asarray(geo['centroids'])
    disc, _ = region_shape(DesignProblem.from_geo(geo), spec)
    out = np.setdiff1d(np.arange(len(cen)), disc)
    nd, Ed = c6_nuE(region_phys_C6(geo, C6, disc)); no, Eo = c6_nuE(region_phys_C6(geo, C6, out))
    print(f"    after gluing: disc nu={nd:+.3f} E={Ed:.2f}   matrix nu={no:+.3f} E={Eo:.2f}", flush=True)
    u, nwt = open_stretch(geo, axis=0, regularize=True)
    return geo, glued, spec, C6, disc, out, nd, Ed, no, Eo, u, nwt


def bare_stress(bare, eps):
    """Voigt stress sigma = A:eps from the per-triangle bare-tensor 5-vector
    A=[xxxx,xxxy,xxyy,xyyy,yyyy] (full-symmetric 4-tensor, e.g. from MO.bare_tensor)."""
    A0, A1, A2, A3, A4 = (bare[:, i] for i in range(5))
    exx, eyy, exy = eps[:, 0, 0], eps[:, 1, 1], eps[:, 0, 1]
    s = np.zeros_like(eps)
    s[:, 0, 0] = A0 * exx + 2 * A1 * exy + A2 * eyy
    s[:, 0, 1] = s[:, 1, 0] = A1 * exx + 2 * A2 * exy + A3 * eyy
    s[:, 1, 1] = A2 * exx + 2 * A3 * exy + A4 * eyy
    return s


def tensor_mag(t):
    """||t|| for a per-triangle symmetric 2x2 tensor field, treating off-diagonal twice (Voigt norm)."""
    return np.sqrt(t[:, 0, 0] ** 2 + 2 * t[:, 0, 1] ** 2 + t[:, 1, 1] ** 2)


def unit_mode_response(geo):
    """Per-triangle strain eps_k and stress sig_k (each a length-3 list of (nt,2,2) arrays) for the
    3 unit macro-strain modes (xx, yy, xy), from a real PBC relaxation (physical_homog.relax) -- the
    ACTUAL simulated response, not the homogenised tensor. Any macro forcing (c0,c1,c2) is
    c0*eps[0]+c1*eps[1]+c2*eps[2] (and the same combination of sig, by linearity of bare_stress)."""
    ev, sx = geo['edge_vecs'], geo['simplices']
    u = PH.relax(geo, np.arange(2, 2 * len(geo['pts'])), SA.assemble_K_faff)
    eps = [MO.tri_metric_change(ev, sx, PH.Fk[k], u[k]) / PH.DELTA for k in range(3)]
    bare = MO.bare_tensor(geo)
    sig = [bare_stress(bare, e) for e in eps]
    return eps, sig


def solver_region_C6(prob, k_bond, region=None):
    """The FORWARD SOLVER's physical 6-vector for the designed k over a region."""
    with torch.no_grad():
        out = prob.forward(k_bond, physical_units=True)
        return prob.region_tensor(out['per_triangle'], region).numpy()


def solver_region_nuE(prob, k_bond, region=None):
    """The FORWARD SOLVER's own prediction (physical) for the designed k over a region — plot
    alongside the simulation to show the solver's prediction is itself confirmed."""
    return c6_nuE(solver_region_C6(prob, k_bond, region))


# ---- directional response nu(theta), E(theta) (for the anisotropy case) ----------------------
def _compliance_tensor(C6):
    Cv = np.array([[C6[0], C6[2], C6[1]], [C6[2], C6[5], C6[4]], [C6[1], C6[4], C6[3]]])
    S = np.linalg.inv(Cv)
    Sc = np.zeros((2, 2, 2, 2))
    Sc[0, 0, 0, 0] = S[0, 0]; Sc[1, 1, 1, 1] = S[1, 1]
    Sc[0, 0, 1, 1] = Sc[1, 1, 0, 0] = S[0, 1]
    for i in [(0, 0, 0, 1), (0, 0, 1, 0), (0, 1, 0, 0), (1, 0, 0, 0)]:
        Sc[i] = S[0, 2] / 2
    for i in [(1, 1, 0, 1), (1, 1, 1, 0), (0, 1, 1, 1), (1, 0, 1, 1)]:
        Sc[i] = S[1, 2] / 2
    for i in [(0, 1, 0, 1), (0, 1, 1, 0), (1, 0, 0, 1), (1, 0, 1, 0)]:
        Sc[i] = S[2, 2] / 4
    return Sc


def nu_E_theta(C6, thetas):
    """Directional Poisson ratio nu(theta) and Young's modulus E(theta) from a 6-vector."""
    Sc = _compliance_tensor(C6)
    nu, E = [], []
    for th in thetas:
        m = np.array([np.cos(th), np.sin(th)]); n = np.array([-np.sin(th), np.cos(th)])
        Emm = np.einsum('ijkl,i,j,k,l', Sc, m, m, m, m)
        Emn = np.einsum('ijkl,i,j,k,l', Sc, m, m, n, n)
        nu.append(-Emn / Emm); E.append(1.0 / Emm)
    return np.array(nu), np.array(E)


# ---- reference tensors (for isotropize / anisotropize cross-targets) -------------------------
def reference_C6(kind, half=8):
    """Physical 6-vector of a uniform-k reference lattice: 'iso' = regular triangular (nu=1/3),
    'aniso' = compressed-row lattice. Size-independent (bulk value); used as design targets."""
    geo = make_topology({'iso': 'regular', 'aniso': 'aniso_str'}[kind], half)
    MB.set_VD(geo, 0)                                    # uniform k=1 (sets bond_k / tri_k)
    return sim_region_C6(geo, None)


# ---- small plotting helpers ------------------------------------------------------------------
TOPO_COLORS = {'regular': '#000000', 'aniso_str': '#1f77b4', 'aniso_shr': '#17becf',
               'disorder_lo': '#2ca02c', 'disorder_hi': '#d62728'}
SIZE_MARKERS = {SIZES[0]: 'o', SIZES[1]: '^'}


def savedir(case):
    d = os.path.join(HERE, case)
    os.makedirs(d, exist_ok=True)
    return d


def networks_dir(case):
    d = os.path.join(HERE, case, 'networks')
    os.makedirs(d, exist_ok=True)
    return d


def save_network(path, geo, bond_k, C6_per=None, **meta):
    """Persist a DESIGNED network so later analysis/plots can reload it (load_network) WITHOUT
    re-running the minimizer. Stores geometry + designed per-bond k + (optional) per-triangle
    physical tensor + metadata (topo, size, target, region, achieved...)."""
    import json
    k = bond_k.detach().numpy() if torch.is_tensor(bond_k) else np.asarray(bond_k)
    np.savez_compressed(path, pts=geo['pts'], tri_verts=geo['tri_verts'], centroids=geo['centroids'],
                        simplices=geo['simplices'], bond_u=geo['bond_u'], bond_v=geo['bond_v'],
                        bond_R=geo['bond_R'], tri_bond=geo['tri_bond'], areas=geo['areas'],
                        BL1=geo['BL1'], BL2=geo['BL2'], bond_k=k,
                        C6_per=(np.asarray(C6_per) if C6_per is not None else np.zeros(0)),
                        meta=json.dumps(meta))


def load_network(path):
    """Reload (geo, bond_k, C6_per, meta) written by save_network. geo has bond_k/tri_k installed,
    plus edge_vecs/actual_len2 rebuilt from tri_verts (needed by tri_metric_change/bare_tensor and
    the open_stretch* family) -- so it plugs straight into draw_network / fill_local_map /
    region_phys_C6 / nu_E_theta / open_stretch_nu with no extra per-caller reconstruction."""
    import json
    d = np.load(path, allow_pickle=True)
    geo = {k: d[k] for k in ['pts', 'tri_verts', 'centroids', 'simplices', 'bond_u', 'bond_v',
                             'bond_R', 'tri_bond', 'areas', 'BL1', 'BL2']}
    geo['bond_k'] = d['bond_k']; geo['tri_k'] = d['bond_k'][geo['tri_bond']]
    tv = geo['tri_verts']; p0, p1, p2 = tv[:, 0], tv[:, 1], tv[:, 2]
    geo['edge_vecs'] = np.stack([p1 - p0, p2 - p0, p2 - p1], 1)
    geo['actual_len2'] = (geo['edge_vecs'] ** 2).sum(2)
    C6 = d['C6_per']; C6 = None if C6.size == 0 else C6
    return geo, d['bond_k'], C6, json.loads(str(d['meta']))


def _detail_row(axr, name, geo, kb, C6, region, show_titles):
    """One row: [ rigidity network | filled local ν | filled local E ], target region marked."""
    import matplotlib.pyplot as plt
    lc = draw_network(axr[0], geo, kb, cmap='viridis')
    plt.colorbar(lc, ax=axr[0], fraction=0.046, label='k')
    axr[0].set_ylabel(name, fontsize=9)
    nu = local_field_smooth(geo, C6, 'nu')
    pnu = fill_local_map(axr[1], geo, nu, cmap='RdBu_r', sym=True)
    draw_box(axr[1], geo); mark_region(axr[1], region); plt.colorbar(pnu, ax=axr[1], fraction=0.046)
    E = local_field_smooth(geo, C6, 'E')
    pE = fill_local_map(axr[2], geo, E, cmap='viridis')
    draw_box(axr[2], geo); mark_region(axr[2], region); plt.colorbar(pE, ax=axr[2], fraction=0.046)
    if show_titles:
        axr[0].set_title('designed rigidity k (width∝k)', fontsize=10)
        axr[1].set_title('local ν (lime = target region)', fontsize=10)
        axr[2].set_title('local E', fontsize=10)


def design_detail_figure(path, entries, title):
    """5-row grid. entries: list of (name, geo, k_bond, C6_per, region)."""
    import matplotlib.pyplot as plt
    n = len(entries)
    fig, axes = plt.subplots(n, 3, figsize=(13.5, 4.1 * n), squeeze=False)
    for r, e in enumerate(entries):
        _detail_row(axes[r], *e, show_titles=(r == 0))
    fig.suptitle(title, fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()


def design_detail_per_topology(directory, prefix, entries, title_fmt):
    """One file per entry (topology): [ rigidity | local ν | local E ]."""
    import matplotlib.pyplot as plt
    for (name, geo, kb, C6, region) in entries:
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))
        _detail_row(axes, name, geo, kb, C6, region, show_titles=True)
        fig.suptitle(title_fmt.format(name=name), fontsize=12)
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        safe = name.split('(')[0].strip().replace(' ', '_').replace('→', 'to')
        p = os.path.join(directory, f'{prefix}_{safe}.png')
        plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()


def nuE_row_grid(path, title, entries, figsize=(3.0, 6.2)):
    """2-row grid of local ν (top, RdBu_r, sym vlim=0.6, region marked) / E (bottom, viridis, clipped
    to the 97th percentile, region marked) maps, one column per entry -- the "one case/kind across
    several topologies" layout. entries: list of (col_title, col_title_row2_or_None, geo, C6_per,
    region). figsize is (per-column width, height)."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, len(entries), figsize=(figsize[0] * len(entries), figsize[1]), squeeze=False)
    for col, (t0, t1, geo, C6, region) in enumerate(entries):
        nu = local_field_smooth(geo, C6, 'nu')
        fill_local_map(axes[0, col], geo, nu, cmap='RdBu_r', sym=True, vlim=0.6)
        draw_box(axes[0, col], geo); mark_region(axes[0, col], region)
        axes[0, col].set_title(t0, fontsize=9)
        E = local_field_smooth(geo, C6, 'E')
        pE = fill_local_map(axes[1, col], geo, E, cmap='viridis')
        pE.set_clim(0, np.nanpercentile(E, 97))
        draw_box(axes[1, col], geo); mark_region(axes[1, col], region)
        if t1:
            axes[1, col].set_title(t1, fontsize=9)
    axes[0, 0].set_ylabel('local ν', fontsize=10); axes[1, 0].set_ylabel('local E', fontsize=10)
    fig.suptitle(title, fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(path, dpi=140, bbox_inches='tight'); plt.close()
