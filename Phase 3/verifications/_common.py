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
from scipy.spatial import Delaunay, cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
P3 = os.path.dirname(HERE)
ROOT = os.path.dirname(P3)
sys.path.insert(0, P3)
sys.path.insert(0, os.path.join(ROOT, 'Phase 2'))
sys.path.insert(0, os.path.join(ROOT, 'verification_tools'))
sys.path.insert(0, ROOT)
import forward_solver_torch as fst
import test_cluster_VD as VD
import test_cluster_rigidity as TR
import test_cluster_Ceff as CE
import physical_homog as PH
from inverse_design import DesignProblem, Objective, optimize, validate, c6_to_nuE
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
    geo = VD.build_geometry(N, eta, seed=seed)
    if not np.allclose(M, np.eye(2)):
        geo = _affine(geo, M)
    VD.set_VD(geo, 0)
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


def make_lattice(phi, psi, half=10.0, seed=0, eta=0.0):
    """Preferred constructor. Base vectors v1=(1,0), v2=(φ/2, ψ·√3/2) (φ=ψ=1 → regular triangular);
    lattice = all m·v1+n·v2; keep a SQUARE real-space region (|x|,|y| ≤ half) as an axis-aligned
    PERIODIC box. Optional eta perturbs positions (disordered). Returns a geo dict (k not set)."""
    Nx = max(4, int(round(2 * half)))
    row_h = psi * np.sqrt(3) / 2
    Lx = float(Nx)
    Ny = _ny_commensurate(phi, Lx, row_h)
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
    """Outline a target region: {'kind':'circle','center':(x,y),'radius':r} or
    {'kind':'vlines','xs':[...]}"""
    if not region:
        return
    import matplotlib.pyplot as plt
    if region['kind'] == 'circle':
        ax.add_patch(plt.Circle(region['center'], region['radius'], facecolor='none',
                                edgecolor='lime', lw=2.0, zorder=6))
    elif region['kind'] == 'vlines':
        for x in region['xs']:
            ax.axvline(x, color='lime', lw=1.0, ls='--', zorder=6)


def write_csv(path, header, rows):
    import csv
    with open(path, 'w', newline='') as f:
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
_Dinv = np.linalg.inv(np.stack([CE.vec3(g) for g in _Dgt], 1))


def sim_per_triangle_C6(geo):
    """Per-triangle physical-response tensor (nt,6) in INTERNAL units, from one full PBC
    relaxation of geo (uses geo['tri_k']). Compute once, then query any region with
    region_phys_C6 (patch / outside / grid cells all share this relaxation)."""
    ev, sx = geo['edge_vecs'], geo['simplices']; nn = len(geo['pts']); nt = len(sx)
    u_modes = PH.relax(geo, np.arange(2, 2 * nn), TR.assemble_K_faff)
    D = np.zeros((nt, 3, 3))
    for k, (F, u) in enumerate(zip(_Fk, u_modes)):
        D[:, :, k] = CE.vec3(CE.tri_metric_change(ev, sx, F, u) - _Dgt[k])
    W3 = D @ _Dinv
    bare = TR.bare_tensor(geo)
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
    VD.set_VD(geo, 0)                                    # uniform k=1 (sets bond_k / tri_k)
    return sim_region_C6(geo, None)


# ---- small plotting helpers ------------------------------------------------------------------
TOPO_COLORS = {'regular': '#000000', 'aniso_str': '#1f77b4', 'aniso_shr': '#17becf',
               'disorder_lo': '#2ca02c', 'disorder_hi': '#d62728'}
SIZE_MARKERS = {SIZES[0]: 'o', SIZES[1]: '^'}


def savedir(case):
    d = os.path.join(HERE, case)
    os.makedirs(d, exist_ok=True)
    return d


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
