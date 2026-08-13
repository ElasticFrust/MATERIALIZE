"""
Phase 3 — inverse design on the (physical) intrinsic forward solver.

Find per-bond rigidities k (and, optionally, rest lengths l0) that realise a target elastic
response, by gradient descent through `forward(method='intrinsic', physical_units=True)`. Targets
can be:
  - GLOBAL: the whole-network homogenised ν / E / full tensor,
  - LOCAL: a prescribed response in a sub-region (subset of triangles),
  - MIXED: several objectives at once (e.g. a global-average ν with a stiffer local patch).

Design variables (softplus-positive): k per bond (primary); l0 per bond is plumbed but is
mathematically degenerate with k in the current solver (it enters only as k/l0²) — see
SOLVER_GUIDE.md §7. Works at any mesh size: gradients flow through the dense path (≤600 tri) or
the adjoint sparse path (>600), automatically. Both periodic and open domains are supported.

Quick start:
    from inverse_design import DesignProblem, Objective, optimize, validate
    prob = DesignProblem.periodic(N=14, eta=0.3, seed=0)          # or .open(tri)
    objs = [Objective(kind='nu', target=-0.2)]                    # global auxetic target
    res = optimize(prob, objs, mode='k', optimizer='lbfgs', n_iter=80)
    print(validate(prob, res['k'], res['l0'], objs))
"""
import os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'Phase 2'))
sys.path.insert(0, os.path.join(ROOT, 'verification_tools'))
sys.path.insert(0, ROOT)
import forward_solver_torch as fst
from verify_solver_open import clean_tri, build_open_mesh
import test_cluster_VD as VD
from test_intrinsic_VD import kkt_from_tri_bond
from verify_solver_sweep import make_solver
torch.set_default_dtype(torch.float64)

BETA = 5.0                       # softplus sharpness (k ≈ raw for moderate raw)


# --------------------------------------------------------------------------- parameterisation
def _softplus(raw):
    """Map an unconstrained optimiser variable `raw` to a POSITIVE stiffness `k = softplus(raw)`.
    Optimising `raw` freely then squashing through softplus keeps `k>0` without a hard constraint."""
    return torch.nn.functional.softplus(raw, beta=BETA)


def _inv_softplus(k):
    """Inverse of `_softplus`: the raw variable that yields a given positive `k` (used to seed the
    optimiser from a starting stiffness, e.g. all-ones)."""
    k = torch.as_tensor(k, dtype=torch.float64)
    return torch.log(torch.expm1(BETA * k)) / BETA


def c6_to_nuE(C):
    """ν, E from a 6-component elastic tensor, using the SAME formula the solver's forward uses
    (so a global-region objective matches out['poisson']/out['young'] exactly). Torch, autograd-safe.
    C may be (6,) (one tensor) or (...,6) (batched, e.g. per-triangle) -- indexed on the last axis."""
    nu = (C[..., 2] * C[..., 3] - C[..., 1] * C[..., 4]) / (C[..., 0] * C[..., 3] - C[..., 1] ** 2)
    E = (C[..., 2] ** 2 * C[..., 3] - 2 * C[..., 1] * C[..., 2] * C[..., 4] + C[..., 1] ** 2 * C[..., 5]
         + C[..., 0] * (C[..., 4] ** 2 - C[..., 3] * C[..., 5])) / (C[..., 1] ** 2 - C[..., 0] * C[..., 3])
    return nu, E


# angle grid for directional ν(θ) targets (θ ∈ [0,π]); shared by objective + validate
ANG = np.linspace(0.0, np.pi, 37)


def c6_to_nuE_theta(C, thetas):
    """Directional Poisson ratio ν(θ) AND Young's modulus E(θ) from a 6-vector (torch,
    autograd-safe). Matches _common.nu_E_theta exactly: build the compliance 4-tensor and contract
    with the axial (m) / transverse (n) directions."""
    Cv = torch.stack([torch.stack([C[0], C[2], C[1]]),
                      torch.stack([C[2], C[5], C[4]]),
                      torch.stack([C[1], C[4], C[3]])])
    S = torch.linalg.inv(Cv)
    Sc = C.new_zeros((2, 2, 2, 2))
    Sc[0, 0, 0, 0] = S[0, 0]; Sc[1, 1, 1, 1] = S[1, 1]
    Sc[0, 0, 1, 1] = S[0, 1]; Sc[1, 1, 0, 0] = S[0, 1]
    for i in [(0, 0, 0, 1), (0, 0, 1, 0), (0, 1, 0, 0), (1, 0, 0, 0)]:
        Sc[i] = S[0, 2] / 2
    for i in [(1, 1, 0, 1), (1, 1, 1, 0), (0, 1, 1, 1), (1, 0, 1, 1)]:
        Sc[i] = S[1, 2] / 2
    for i in [(0, 1, 0, 1), (0, 1, 1, 0), (1, 0, 0, 1), (1, 0, 1, 0)]:
        Sc[i] = S[2, 2] / 4
    th = torch.as_tensor(thetas, dtype=torch.float64)
    c, s = torch.cos(th), torch.sin(th)
    M = torch.stack([c, s], 1); Nn = torch.stack([-s, c], 1)             # (T,2) axial / transverse
    Emm = torch.einsum('ijkl,ti,tj,tk,tl->t', Sc, M, M, M, M)
    Emn = torch.einsum('ijkl,ti,tj,tk,tl->t', Sc, M, M, Nn, Nn)
    return -Emn / Emm, 1.0 / Emm


def c6_to_nu_theta(C, thetas):
    """Directional Poisson ratio ν(θ) only (thin wrapper over c6_to_nuE_theta) — for `nu_theta` objectives."""
    return c6_to_nuE_theta(C, thetas)[0]


def c6_to_E_theta(C, thetas):
    """Directional Young's modulus E(θ) only (thin wrapper over c6_to_nuE_theta) — for `E_theta` objectives."""
    return c6_to_nuE_theta(C, thetas)[1]


def per_triangle_strain_stress(bare, W, load_dg):
    """Per-triangle ACTUAL local response under an applied macroscopic metric-change load, from one
    solver forward() call. Differentiable end to end (autograd-connected through bare/W to k_bond).

    This framework's native 'strain' is the metric change Delta_g = F.T@F - I (see module docstring
    / the pbc_dg_analysis.py convention note), not linearised engineering strain -- so `load_dg` and
    the returned fields are exact metric-change vec3 components [xx, xy, yy], matching
    physical_homog.Fk / _common._Dgt precisely (verified equal to _common.unit_mode_response in
    test_inverse_design.py).

    bare    : (N,5) per-triangle bare tensor A=[xxxx,xxxy,xxyy,xyyy,yyyy] (out['bare']).
    W       : (N,9) per-triangle strain-concentration, reshaped to (N,3,3); W3[loc,k] =
              d(local Delta_g)_loc / d(macro Delta_g)_k in the vec3=[xx,xy,yy] basis (out['W']).
    load_dg : (3,) applied macro Delta_g, vec3 = [dg_xx, dg_xy, dg_yy].

    Returns (strain_local, stress_local), each (N,3) vec3=[xx,xy,yy].
    """
    W3 = W.reshape(-1, 3, 3)
    load = torch.as_tensor(load_dg, dtype=bare.dtype, device=bare.device)
    strain_local = load + torch.einsum('nij,j->ni', W3, load)          # (I + W) @ load
    a0, a1, a2, a3, a4 = (bare[:, i] for i in range(5))
    gxx, gxy, gyy = strain_local[:, 0], strain_local[:, 1], strain_local[:, 2]
    stress_local = torch.stack([
        a0 * gxx + 2 * a1 * gxy + a2 * gyy,
        a1 * gxx + 2 * a2 * gxy + a3 * gyy,
        a2 * gxx + 2 * a3 * gxy + a4 * gyy,
    ], dim=1)
    return strain_local, stress_local


def _region_rows(field, region):
    """field[region] (or field itself if region is None) -- the per-triangle rows within a region,
    for any per-triangle field (strain/stress vec3, or a per-triangle nu/E column)."""
    return field if region is None else field[torch.as_tensor(region, dtype=torch.long)]


def region_mean_vec3(field, region):
    """Region-mean of a per-triangle (N,3) strain/stress vec3 field -- plain mean, no physical
    rescaling (unlike DesignProblem.region_tensor's 8N/A factor, which is specific to converting the
    per-triangle elastic tensor C6 to physical units; strain/stress here are already physical)."""
    return _region_rows(field, region).mean(0)


def _anisotropy(C):
    """Relative squared deviation of a 6-vector from the nearest ISOTROPIC tensor (0 = isotropic).
    Voigt (V11,V12,V16,V22,V26,V66)=(C0,C2,C1,C5,C4,C3); isotropy: C0=C5, C1=C4=0, C3=(C0−C2)/2."""
    dev = (C[0] - C[5]) ** 2 + C[1] ** 2 + C[4] ** 2 + (C[3] - (C[0] - C[2]) / 2) ** 2
    return dev / (C[0] ** 2 + C[5] ** 2 + C[2] ** 2 + C[3] ** 2 + 1e-12)


def isotropic_c6(nu, E):
    """The (physical) 6-vector of an ISOTROPIC 2D material with Poisson ratio nu and modulus E
    (plane-stress: C11=E/(1−ν²), C12=νC11·... ; C16=C26=0, C22=C11, C66=(C11−C12)/2). Use as a full
    'tensor' target to pin an EXACT isotropic response (ν and E fixed, nothing left free)."""
    C11 = E / (1 - nu ** 2); C12 = nu * E / (1 - nu ** 2); C66 = (C11 - C12) / 2
    return torch.tensor([C11, 0.0, C12, C66, 0.0, C11], dtype=torch.float64)


# --------------------------------------------------------------------------- objectives
class Objective:
    """One design target over a region.

    region : 1-D int array of triangle indices, or None = whole network (global).
    kind   :
      'nu' | 'E'          — a SCALAR target meaning ISOTROPIC ν / E, i.e. that value in EVERY
                            direction (ν(θ)/E(θ) held flat). This is the default meaning of "ν=v".
      'nu_dir' | 'E_dir'  — the legacy single-direction scalar (one contraction of the tensor);
                            constrains only one orientation, so the tensor can still be anisotropic.
      'tensor'            — the full physical 6-vector.
      'nu_theta'|'E_theta'— a directional profile over `thetas`.
      'isotropy'          — force direction-independence (level free).
      'strain' | 'stress' — the ACTUAL per-triangle metric-change response (vec3=[xx,xy,yy], this
                            framework's native strain — see `per_triangle_strain_stress`) under an
                            applied macro load `load` (required for these two kinds; a (3,) vec3
                            Delta_g, e.g. from `physical_homog.Fk`/`_common._Dgt`). `target` is
                            either a (3,) vec3 (region-MEAN response) or a (len(region),3) array
                            (per-triangle field target). NOTE: region-mean STRAIN over the WHOLE
                            cell (region=None) is degenerate — it equals `load` exactly, since the
                            fluctuation has zero cell-mean — so a 'strain' objective is only
                            meaningful on a sub-region or as a per-triangle field target. 'stress'
                            has no such degeneracy and is designable globally too.
    target : float (nu/E/nu_dir/E_dir), (6,) array (tensor), (3,)/(len(region),3) array
             (strain/stress), or profile over `thetas`.
    load   : (3,) applied macro Delta_g vec3=[xx,xy,yy] — REQUIRED for 'strain'/'stress', unused
             otherwise.
    thetas : angle grid for the directional kinds (default ANG = linspace(0,π,37)); a scalar target
             broadcasts to a flat (isotropic) profile.
    weight : scalar weight in the total loss.
    homogeneity : >0 adds a penalty on the VARIANCE of the per-triangle local response WITHIN this
             objective's region (not just the region-mean's deviation from target) — discourages the
             optimiser from satisfying the mean via a few floppy (k→0) hinge triangles while the rest
             of the region is untouched (the loss-level analogue of the geometry-level `glue()`
             workaround — see [[glue-not-joint-design]]). Only supported for kind in
             ('nu','E','strain','stress'); 0.0 (off) for any other kind.
    """
    def __init__(self, kind, target=None, region=None, weight=1.0, thetas=None, load=None,
                homogeneity=0.0):
        """Validate `kind` and normalise `target` into the tensor form `_loss` expects for that kind:
        a scalar nu/E broadcasts to a flat ν(θ)/E(θ) profile over `thetas`; a `tensor` target is the
        6-vector; a `strain`/`stress` target is a vec3 (and requires `load`). Stores `region`,
        `weight`, `homogeneity` for the loss to consume."""
        assert kind in ('nu', 'E', 'nu_dir', 'E_dir', 'tensor', 'nu_theta', 'E_theta', 'isotropy',
                        'strain', 'stress')
        if homogeneity:
            assert kind in ('nu', 'E', 'strain', 'stress'), \
                f"homogeneity is not supported for kind={kind!r}"
        self.kind = kind
        self.region = None if region is None else np.asarray(region, dtype=np.int64)
        self.weight = float(weight)
        self.homogeneity = float(homogeneity)
        if kind in ('nu', 'E', 'nu_theta', 'E_theta'):           # scalar nu/E -> flat (isotropic) profile
            self.thetas = ANG if thetas is None else np.asarray(thetas, dtype=float)
            self.target = torch.as_tensor(np.array(np.broadcast_to(target, self.thetas.shape),
                                                   dtype=float), dtype=torch.float64)
        elif kind == 'tensor':
            self.target = torch.as_tensor(target, dtype=torch.float64)
        elif kind in ('strain', 'stress'):
            assert load is not None, f"Objective(kind={kind!r}) requires `load` (macro Delta_g vec3)"
            self.load = torch.as_tensor(load, dtype=torch.float64)
            self.target = torch.as_tensor(target, dtype=torch.float64)
        else:                                                    # nu_dir, E_dir, isotropy
            self.target = target


def constrain(region=None, weight=1.0, *, nu=None, E=None, isotropic=False, tensor=None,
              nu_theta=None, E_theta=None, thetas=None, nu_scalar=None, E_scalar=None):
    """Build a list of Objectives that fix a chosen set of quantities on `region` EXACTLY (over all
    directions) and leave everything else FREE. Compose several regions by concatenating the lists.

    EXACT (direction-complete) knobs — the recommended ones:
      nu        : isotropic Poisson ratio — ν(θ)=nu at EVERY angle (flat ν(θ)); E left free.
      E         : isotropic Young's modulus — E(θ)=E at every angle; ν left free.
      isotropic : True → force the response direction-independent (penalise the anisotropic part);
                  the level(s) stay free. Combine with nu=/E= to also fix the level(s).
      tensor    : a full physical 6-vector (pins the entire elastic tensor). `isotropic_c6(nu,E)`
                  builds the isotropic one → an exact isotropic (ν,E) material, nothing free.
      nu_theta / E_theta : a full or partial directional profile over `thetas` (default ANG).

    LEGACY 'ish' (single-direction) knobs — kept, but they constrain only ONE orientation so the
    tensor can still be anisotropic:
      nu_scalar / E_scalar : the old Objective('nu'|'E', ...).
    """
    objs = []
    if tensor is not None:
        objs.append(Objective('tensor', tensor, region, weight))
    if isotropic:
        objs.append(Objective('isotropy', None, region, weight))
    if nu is not None:
        objs.append(Objective('nu_theta', nu, region, weight, thetas=thetas))
    if E is not None:
        objs.append(Objective('E_theta', E, region, weight, thetas=thetas))
    if nu_theta is not None:
        objs.append(Objective('nu_theta', nu_theta, region, weight, thetas=thetas))
    if E_theta is not None:
        objs.append(Objective('E_theta', E_theta, region, weight, thetas=thetas))
    if nu_scalar is not None:
        objs.append(Objective('nu_dir', nu_scalar, region, weight))
    if E_scalar is not None:
        objs.append(Objective('E_dir', E_scalar, region, weight))
    if not objs:
        raise ValueError("constrain: specify at least one quantity (nu, E, isotropic, tensor, ...)")
    return objs


# --------------------------------------------------------------------------- design problem
class DesignProblem:
    """Wraps geometry + solver + per-bond→per-triangle map for periodic or open networks."""

    def __init__(self, solver, tri_bond, bond_len, areas, centroids, rl_ref):
        """Store the forward `solver` and the geometry needed to (a) map per-BOND design variables to
        the per-triangle-edge arrays the solver wants (`tri_bond`), and (b) reduce per-triangle output
        over regions (`areas`, `centroids`). Prefer the `periodic`/`open`/`from_geo` constructors."""
        self.solver = solver
        self.tri_bond = torch.as_tensor(tri_bond, dtype=torch.long)     # (N,3)
        self.bond_len = torch.as_tensor(bond_len, dtype=torch.float64)  # (n_bond,)
        self.areas = torch.as_tensor(areas, dtype=torch.float64)        # (N,)
        self.centroids = np.asarray(centroids)                          # (N,2)
        self.rl_ref = torch.as_tensor(rl_ref, dtype=torch.float64)      # (N,3)
        self.n_bond = int(self.bond_len.shape[0])
        self.n_tri = int(self.tri_bond.shape[0])

    # ---- constructors ----
    @classmethod
    def from_geo(cls, geo):
        """Build from a ready PERIODIC geometry dict (keys: pts, simplices, edge_vecs, bond_R,
        areas, tri_bond, actual_len2; bond_k/tri_k set to 1 if absent). Use this for custom
        topologies (e.g. affine-transformed anisotropic lattices)."""
        if 'tri_k' not in geo:
            VD.set_VD(geo, 0)                                    # default uniform k=1
        kkt = kkt_from_tri_bond(geo['tri_bond'], geo['edge_vecs'])
        solver = make_solver(geo, kkt)
        bond_len = np.sqrt((geo['bond_R'] ** 2).sum(1))
        cen = geo.get('centroids', geo['pts'][geo['simplices']].mean(1))   # image-correct if present
        return cls(solver, geo['tri_bond'], bond_len, geo['areas'], cen,
                   np.sqrt(geo['actual_len2']))

    @classmethod
    def periodic(cls, N=14, eta=0.3, seed=0):
        """Build a PERIODIC unit-cell problem: a triangular lattice of half-size `N`, positionally
        perturbed by disorder `eta` (0 = perfect crystal), with uniform starting stiffness k=1."""
        geo = VD.build_geometry(N, eta, seed=seed); VD.set_VD(geo, 0)
        return cls.from_geo(geo)

    @classmethod
    def open(cls, tri):
        """Build an OPEN (finite, free-boundary) problem from a scipy-style triangulation `tri`
        (e.g. `Disc_2_Cont_optimized.generate_foam_points`)."""
        tri = clean_tri(tri)
        mesh = build_open_mesh(tri)
        solver, _, rl = fst.from_triangulation(tri)
        bond_len = np.sqrt((mesh['bond_R'] ** 2).sum(1))
        cen = mesh['pts'][mesh['simplices']].mean(1)
        return cls(solver, mesh['tri_bond'], bond_len, mesh['areas'], cen, rl.numpy())

    # ---- region helpers ----
    def region_in_circle(self, center, radius):
        """A region = the indices of triangles whose CENTROID lies within `radius` of `center`
        (the basic building block for LOCAL objectives)."""
        c = np.asarray(center)
        return np.where(((self.centroids - c) ** 2).sum(1) < radius ** 2)[0]

    def region_where(self, predicate):
        """predicate(centroids (N,2)) -> bool mask."""
        return np.where(predicate(self.centroids))[0]

    # ---- forward ----
    def forward(self, k_bond, l0_bond=None, physical_units=True):
        """Run the differentiable forward solve for a per-BOND stiffness `k_bond`: scatter it to the
        per-triangle-edge array via `tri_bond`, then call the intrinsic solver. Returns the solver's
        dict (poisson, young, elastic_tensor, per_triangle, bare, W). Differentiable w.r.t. `k_bond`."""
        tri_k = k_bond[self.tri_bond]                                    # (N,3)
        rl = self.rl_ref if l0_bond is None else l0_bond[self.tri_bond]
        return self.solver.forward(tri_k, rest_lengths=rl,
                                   method='intrinsic', physical_units=physical_units)

    def region_tensor(self, per_triangle, region):
        """Physical homogenised 6-vector over a region (unweighted mean × physical factor)."""
        if region is None:
            C6 = per_triangle.mean(0); A = self.areas.sum(); n = self.n_tri
        else:
            idx = torch.as_tensor(region, dtype=torch.long)
            C6 = per_triangle[idx].mean(0); A = self.areas[idx].sum(); n = idx.shape[0]
        return C6 * (8.0 * n / A)                                        # → physical units


# --------------------------------------------------------------------------- loss & optimise
def _params_to_kl(raw, prob, mode):
    """raw dict {'k':..,'l0':..} → (k_bond, l0_bond or None)."""
    k_bond = _softplus(raw['k']) if 'k' in raw else torch.ones(prob.n_bond)
    l0_bond = prob.bond_len * torch.exp(raw['l0']) if 'l0' in raw else None
    return k_bond, l0_bond


def _loss(prob, objectives, k_bond, l0_bond, reg=0.0):
    """The scalar training loss minimised by `optimize` (fully differentiable w.r.t. `k_bond`).
    LOGIC: one forward solve gives the per-triangle tensor `per`, bare tensor `bare`, and
    strain-concentration `W`; then each Objective contributes `weight·‖achieved − target‖²`, where
    'achieved' is computed for that objective's KIND (nu/E/tensor/directional from the region tensor;
    raw strain/stress from `bare,W` under the objective's load) over its REGION (or globally). An
    optional `homogeneity` term adds the within-region variance of the local field, and `reg` adds a
    small mean((k−mean k)²) = k-variance (pull toward a constant level, free to float) that
    discourages drifting into floppy/unstable configurations."""
    out = prob.forward(k_bond, l0_bond, physical_units=True)
    per, bare, W = out['per_triangle'], out['bare'], out['W']
    total = torch.zeros((), dtype=torch.float64)
    for ob in objectives:
        if ob.kind in ('strain', 'stress'):                              # actual local response
            strain_local, stress_local = per_triangle_strain_stress(bare, W, ob.load)
            field_region = _region_rows(strain_local if ob.kind == 'strain' else stress_local, ob.region)
            response = field_region if ob.target.dim() == 2 else field_region.mean(0)  # per-tri or mean
            total = total + ob.weight * ((response - ob.target) ** 2).mean()
            if ob.homogeneity:
                total = total + ob.homogeneity * field_region.var(0).sum()
            continue
        C6 = prob.region_tensor(per, ob.region)
        if ob.kind in ('nu', 'nu_theta'):                                # isotropic (scalar) or profile
            nth = c6_to_nu_theta(C6, ob.thetas)
            total = total + ob.weight * ((nth - ob.target) ** 2).mean()
            if ob.kind == 'nu' and ob.homogeneity:
                nu_pertri, _ = c6_to_nuE(per)
                total = total + ob.homogeneity * _region_rows(nu_pertri, ob.region).var()
        elif ob.kind in ('E', 'E_theta'):
            eth = c6_to_E_theta(C6, ob.thetas)
            total = total + ob.weight * ((eth - ob.target) ** 2).mean()
            if ob.kind == 'E' and ob.homogeneity:
                _, E_pertri = c6_to_nuE(per)
                total = total + ob.homogeneity * _region_rows(E_pertri, ob.region).var()
        elif ob.kind == 'nu_dir':                                        # legacy single-direction scalar
            nu, _ = c6_to_nuE(C6)
            total = total + ob.weight * (nu - ob.target) ** 2
        elif ob.kind == 'E_dir':
            _, E = c6_to_nuE(C6)
            total = total + ob.weight * (E - ob.target) ** 2
        elif ob.kind == 'isotropy':                                      # penalise the anisotropic part
            total = total + ob.weight * _anisotropy(C6)
        else:                                                            # 'tensor'
            total = total + ob.weight * ((C6 - ob.target) ** 2).mean()
    if reg > 0.0:                                                        # keep k near a CONSTANT level:
        total = total + reg * ((k_bond - k_bond.mean()) ** 2).mean()    # penalise k VARIANCE — the level is
                                                                        # free to float (e.g. for E-scale),
                                                                        # discourages floppy/unstable designs
    return total


def _init_raw(prob, mode, seed):
    """Initialise the raw (unconstrained) optimiser variables for a restart: `k` seeded at the
    uniform lattice (`softplus⁻¹(1)`) plus small random noise so different restarts explore different
    basins; `l0` (if designed) seeded near zero. Returns a dict of leaf tensors requiring grad."""
    rng = torch.Generator().manual_seed(seed)
    raw = {}
    if mode in ('k', 'both'):
        base = _inv_softplus(torch.ones(prob.n_bond))
        raw['k'] = (base + 0.3 * torch.randn(prob.n_bond, generator=rng)).requires_grad_(True)
    if mode in ('l0', 'both'):
        raw['l0'] = (0.05 * torch.randn(prob.n_bond, generator=rng)).requires_grad_(True)
    return raw


def optimize(prob, objectives, mode='k', optimizer='lbfgs', n_iter=80,
             n_restarts=1, seed=0, reg=0.0, verbose=True):
    """Design k (and/or l0) to meet the objectives. `reg` (>0) adds a mean((k−mean k)²) = k-variance
    penalty that keeps k near a constant level (uniform; the level floats freely, e.g. for an E
    target) — discourages the optimiser from exploiting floppy/unstable
    configurations that satisfy a scalar target but collapse in simulation. Returns the best
    result over restarts: dict(k, l0, loss, history, raw)."""
    assert mode in ('k', 'l0', 'both')
    best = None
    for r in range(n_restarts):
        raw = _init_raw(prob, mode, seed + r)
        params = list(raw.values())
        history = []
        if optimizer == 'lbfgs':
            opt = torch.optim.LBFGS(params, lr=1.0, max_iter=n_iter,
                                    line_search_fn='strong_wolfe', tolerance_grad=1e-12)

            def closure():
                opt.zero_grad()
                k_bond, l0_bond = _params_to_kl(raw, prob, mode)
                l = _loss(prob, objectives, k_bond, l0_bond, reg)
                l.backward(); history.append(l.item()); return l
            opt.step(closure)
        else:                                                            # adam
            opt = torch.optim.Adam(params, lr=0.05)
            for _ in range(n_iter):
                opt.zero_grad()
                k_bond, l0_bond = _params_to_kl(raw, prob, mode)
                l = _loss(prob, objectives, k_bond, l0_bond, reg)
                l.backward(); opt.step(); history.append(l.item())
        with torch.no_grad():
            k_bond, l0_bond = _params_to_kl(raw, prob, mode)
            final = float(_loss(prob, objectives, k_bond, l0_bond, reg))
        if verbose:
            print(f"  restart {r}: loss {history[0]:.3e} -> {final:.3e} ({len(history)} evals)")
        if best is None or final < best['loss']:
            best = dict(k=k_bond.detach(), l0=None if l0_bond is None else l0_bond.detach(),
                        loss=final, history=history, raw=raw)
    return best


def validate(prob, k_bond, l0_bond, objectives):
    """Re-evaluate each objective at the designed params; return achieved vs target."""
    with torch.no_grad():
        out = prob.forward(k_bond, l0_bond, physical_units=True)
        per, bare, W = out['per_triangle'], out['bare'], out['W']
        report = []
        for ob in objectives:
            reg = 'global' if ob.region is None else f'{len(ob.region)} tri'
            if ob.kind in ('strain', 'stress'):                          # actual local response
                strain_local, stress_local = per_triangle_strain_stress(bare, W, ob.load)
                field_region = _region_rows(strain_local if ob.kind == 'strain' else stress_local,
                                            ob.region)
                if ob.target.dim() == 2:                                 # per-triangle field target
                    ach = field_region.numpy()
                else:                                                    # region-mean target
                    ach = field_region.mean(0).numpy()
                tgt = ob.target.numpy()
                rec = dict(kind=ob.kind, region=reg, load=ob.load.numpy(), target=tgt,
                          achieved=ach, err=float(np.abs(ach - tgt).max()))
                if ob.homogeneity:                                       # within-region spread (not err)
                    rec['homogeneity_spread'] = float(field_region.var(0).sum())
                report.append(rec)
                continue
            C6 = prob.region_tensor(per, ob.region)
            nu, E = c6_to_nuE(C6)
            if ob.kind in ('nu', 'E', 'nu_theta', 'E_theta'):
                fn = c6_to_nu_theta if ob.kind in ('nu', 'nu_theta') else c6_to_E_theta
                got = fn(C6, ob.thetas).numpy(); tgt = ob.target.numpy()
                if ob.kind in ('nu', 'E'):                       # isotropic scalar: report mean + spread
                    rec = dict(kind=ob.kind, region=reg, target=float(tgt.flat[0]),
                              achieved=float(got.mean()), spread=float(np.ptp(got)),
                              err=float(np.abs(got - tgt).max()))
                    if ob.homogeneity:                            # WITHIN-region per-triangle spread
                        nu_pertri, E_pertri = c6_to_nuE(per)
                        pertri = nu_pertri if ob.kind == 'nu' else E_pertri
                        rec['homogeneity_spread'] = float(_region_rows(pertri, ob.region).var())
                    report.append(rec)
                else:                                            # directional profile
                    report.append(dict(kind=ob.kind, region=reg, thetas=ob.thetas, target=tgt,
                                       achieved=got, err=float(np.abs(got - tgt).max())))
            elif ob.kind in ('nu_dir', 'E_dir'):                 # legacy single-direction scalar
                val = float(nu) if ob.kind == 'nu_dir' else float(E)
                report.append(dict(kind=ob.kind, region=reg, target=ob.target, achieved=val,
                                   err=abs(val - ob.target)))
            elif ob.kind == 'isotropy':
                a = float(_anisotropy(C6))
                report.append(dict(kind='isotropy', region=reg, target=0.0, achieved=a, err=a ** 0.5))
            else:
                ach = C6.numpy()
                report.append(dict(kind='tensor', region=reg, target=ob.target.numpy(),
                                   achieved=ach, err=float(np.linalg.norm(ach - ob.target.numpy()))))
    return report
