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
    ('regular',     'regular triangular',        None),
    ('aniso_str',   'non-sym (stretch 1.5,0.8)', np.diag([1.5, 0.8])),
    ('aniso_shr',   'non-sym (shear 0.4)',       np.array([[1.0, 0.4], [0.0, 1.0]])),
    ('disorder_lo', 'disordered eta=0.20',       None),
    ('disorder_hi', 'disordered eta=0.35',       None),
]
TOPO_IDS = [t[0] for t in TOPOS]
SIZES = [12, 20]                                         # 288 tri (dense path) / 800 tri (adjoint path)


def _affine(geo, M):
    geo['pts'] = geo['pts'] @ M.T
    geo['edge_vecs'] = geo['edge_vecs'] @ M.T
    geo['bond_R'] = geo['bond_R'] @ M.T
    geo['actual_len2'] = (geo['edge_vecs'] ** 2).sum(2)
    geo['areas'] = geo['areas'] * abs(np.linalg.det(M))
    return geo


def make_topology(topo_id, N, seed=0):
    """Return a periodic geometry dict (uniform k=1) for a named topology at size N."""
    if topo_id == 'regular':
        geo = VD.build_geometry(N, 0.0, seed=seed)
    elif topo_id == 'aniso_str':
        geo = _affine(VD.build_geometry(N, 0.0, seed=seed), np.diag([1.5, 0.8]))
    elif topo_id == 'aniso_shr':
        geo = _affine(VD.build_geometry(N, 0.0, seed=seed), np.array([[1.0, 0.4], [0.0, 1.0]]))
    elif topo_id == 'disorder_lo':
        geo = VD.build_geometry(N, 0.20, seed=seed)
    elif topo_id == 'disorder_hi':
        geo = VD.build_geometry(N, 0.35, seed=seed)
    else:
        raise ValueError(topo_id)
    VD.set_VD(geo, 0)                                    # uniform k=1
    return geo


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
def reference_C6(kind, N=16):
    """Physical 6-vector of a uniform-k reference lattice: 'iso' = regular triangular (nu=1/3),
    'aniso' = stretched lattice. Size-independent (bulk value); used as design targets."""
    if kind == 'iso':
        geo = make_topology('regular', N)
    elif kind == 'aniso':
        geo = make_topology('aniso_str', N)
    else:
        raise ValueError(kind)
    return sim_region_C6(geo, None)


# ---- small plotting helpers ------------------------------------------------------------------
TOPO_COLORS = {'regular': '#000000', 'aniso_str': '#1f77b4', 'aniso_shr': '#17becf',
               'disorder_lo': '#2ca02c', 'disorder_hi': '#d62728'}
SIZE_MARKERS = {SIZES[0]: 'o', SIZES[1]: '^'}


def savedir(case):
    d = os.path.join(HERE, case)
    os.makedirs(d, exist_ok=True)
    return d
