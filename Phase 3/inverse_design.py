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
    return torch.nn.functional.softplus(raw, beta=BETA)


def _inv_softplus(k):
    k = torch.as_tensor(k, dtype=torch.float64)
    return torch.log(torch.expm1(BETA * k)) / BETA


def c6_to_nuE(C):
    """ν, E from a 6-component elastic tensor, using the SAME formula the solver's forward uses
    (so a global-region objective matches out['poisson']/out['young'] exactly). Torch, autograd-safe."""
    nu = (C[2] * C[3] - C[1] * C[4]) / (C[0] * C[3] - C[1] ** 2)
    E = (C[2] ** 2 * C[3] - 2 * C[1] * C[2] * C[4] + C[1] ** 2 * C[5]
         + C[0] * (C[4] ** 2 - C[3] * C[5])) / (C[1] ** 2 - C[0] * C[3])
    return nu, E


# --------------------------------------------------------------------------- objectives
class Objective:
    """One design target over a region.

    region : 1-D int array of triangle indices, or None = whole network (global).
    kind   : 'nu' | 'E' | 'tensor'.
    target : float (nu/E) or (6,) array (tensor, physical units).
    weight : scalar weight in the total loss.
    """
    def __init__(self, kind, target, region=None, weight=1.0):
        assert kind in ('nu', 'E', 'tensor')
        self.kind = kind
        self.region = None if region is None else np.asarray(region, dtype=np.int64)
        self.target = target if kind != 'tensor' else torch.as_tensor(target, dtype=torch.float64)
        self.weight = float(weight)


# --------------------------------------------------------------------------- design problem
class DesignProblem:
    """Wraps geometry + solver + per-bond→per-triangle map for periodic or open networks."""

    def __init__(self, solver, tri_bond, bond_len, areas, centroids, rl_ref):
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
        geo = VD.build_geometry(N, eta, seed=seed); VD.set_VD(geo, 0)
        return cls.from_geo(geo)

    @classmethod
    def open(cls, tri):
        tri = clean_tri(tri)
        mesh = build_open_mesh(tri)
        solver, _, rl = fst.from_triangulation(tri)
        bond_len = np.sqrt((mesh['bond_R'] ** 2).sum(1))
        cen = mesh['pts'][mesh['simplices']].mean(1)
        return cls(solver, mesh['tri_bond'], bond_len, mesh['areas'], cen, rl.numpy())

    # ---- region helpers ----
    def region_in_circle(self, center, radius):
        c = np.asarray(center)
        return np.where(((self.centroids - c) ** 2).sum(1) < radius ** 2)[0]

    def region_where(self, predicate):
        """predicate(centroids (N,2)) -> bool mask."""
        return np.where(predicate(self.centroids))[0]

    # ---- forward ----
    def forward(self, k_bond, l0_bond=None, physical_units=True):
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
    out = prob.forward(k_bond, l0_bond, physical_units=True)
    per = out['per_triangle']
    total = torch.zeros((), dtype=torch.float64)
    for ob in objectives:
        C6 = prob.region_tensor(per, ob.region)
        if ob.kind == 'nu':
            nu, _ = c6_to_nuE(C6)
            total = total + ob.weight * (nu - ob.target) ** 2
        elif ob.kind == 'E':
            _, E = c6_to_nuE(C6)
            total = total + ob.weight * (E - ob.target) ** 2
        else:                                                            # 'tensor'
            total = total + ob.weight * ((C6 - ob.target) ** 2).mean()
    if reg > 0.0:                                                        # keep k near-uniform:
        total = total + reg * ((k_bond - 1.0) ** 2).mean()              # discourages floppy/unstable designs
    return total


def _init_raw(prob, mode, seed):
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
    """Design k (and/or l0) to meet the objectives. `reg` (>0) adds a mean((k-1)²) penalty that
    keeps k near-uniform — discourages the optimiser from exploiting floppy/unstable
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
        per = out['per_triangle']
        report = []
        for ob in objectives:
            C6 = prob.region_tensor(per, ob.region)
            nu, E = c6_to_nuE(C6)
            reg = 'global' if ob.region is None else f'{len(ob.region)} tri'
            if ob.kind == 'nu':
                report.append(dict(kind='nu', region=reg, target=ob.target, achieved=float(nu),
                                   err=abs(float(nu) - ob.target)))
            elif ob.kind == 'E':
                report.append(dict(kind='E', region=reg, target=ob.target, achieved=float(E),
                                   err=abs(float(E) - ob.target)))
            else:
                ach = C6.numpy()
                report.append(dict(kind='tensor', region=reg, target=ob.target.numpy(),
                                   achieved=ach, err=float(np.linalg.norm(ach - ob.target.numpy()))))
    return report
