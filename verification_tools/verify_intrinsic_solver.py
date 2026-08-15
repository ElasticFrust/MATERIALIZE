"""
Verify the intrinsic solver now living in forward_solver_torch.ElasticSolver.forward(method=
'intrinsic') reproduces the PBC simulation, and that the toggles work.

We build an ElasticSolver on each periodic pda mesh (injecting the periodic-correct geometry:
edge_vecs, kkt_arrays, area_weights), then:
  (A) homogenised nu/E vs eta:  solver intrinsic vs sim vs solver woodbury(plain MF).
  (B) eta=0 sanity: nu=1/3.
  (C) toggles: confirm use_kkt / use_angle_kkt / area_weighted can be turned off and change
      the result, and that the all-on default equals explicit all-on.
"""
import os, sys
import numpy as np
import scipy.sparse.linalg as spla
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..', 'Phase 2'))
import forward_solver_torch as fst
import pbc_dg_analysis as pda
import test_cluster_Ceff as CE
import metric_ops as MO
torch.set_default_dtype(torch.float64)


def make_solver(mesh):
    """ElasticSolver with the pda periodic geometry injected (edge_vecs/kkt/area correct)."""
    pts, sx, ev = mesh['pts'], mesh['simplices'], mesh['edge_vecs']
    # node-index pairs matching pda's (e01=p1-p0, e02=p2-p0, e12=p2-p1) => a-b convention
    edges = np.stack([sx[:, [1, 0]], sx[:, [2, 0]], sx[:, [2, 1]]], axis=1)   # (N,3,2)
    solver = fst.ElasticSolver(pts, sx, edges)
    # override geometry buffers with the periodic-correct ones
    solver.edge_vecs = torch.as_tensor(ev, dtype=torch.float64)
    solver.actual_length2 = torch.as_tensor((ev ** 2).sum(2), dtype=torch.float64)
    solver.area_weights = torch.as_tensor(mesh['areas'] / mesh['areas'].sum(), dtype=torch.float64)
    solver.kkt_arrays = mesh['kkt_arrays']
    solver._build_intrinsic_constraints(np.asarray(sx))      # rebuild with corrected geometry
    return solver


def sim_nuE(mesh):
    ev, sx = mesh['edge_vecs'], mesh['simplices']; nn = len(mesh['pts']); nt = len(sx)
    bare = MO.bare_tensor(mesh)
    Fk = [np.eye(2) + CE.DELTA*M for M in CE.MODES]
    Dg_k = [MO.vec3(F.T@F - np.eye(2)) for F in Fk]
    Dinv = np.linalg.inv(np.stack(Dg_k, axis=1))
    K, _ = pda._assemble_K_and_faff(mesh, np.eye(2)); free = np.arange(2, 2*nn)
    Ds = np.zeros((nt, 3, 3))
    for k, F in enumerate(Fk):
        fa = pda._assemble_K_and_faff(mesh, F)[1]
        u = np.zeros(2*nn); u[free] = spla.spsolve(K[free][:, free].tocsc(), -fa[free])
        Ds[:, :, k] = MO.vec3(MO.tri_metric_change(ev, sx, F, u.reshape(nn, 2)) - (F.T@F-np.eye(2)))
    return CE.Ceff_nuE(mesh, Ds @ Dinv, bare)


def main():
    N = 16
    print(f"N={N}.  (A) homogenised nu / E vs eta:  sim | intrinsic (solver) | woodbury MF (solver)")
    print(f"{'eta':>5} | {'nu_sim':>7} {'nu_int':>7} {'nu_wood':>8} | {'E_sim':>8} {'E_int':>8} {'E_wood':>8}")
    for eta in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        mesh = pda.build_periodic_tf_mesh(N, eta, seed=0)
        solver = make_solver(mesh)
        rig = torch.ones(len(mesh['simplices']), 3, dtype=torch.float64)
        out_i = solver.forward(rig, rest_lengths=solver.actual_length2.sqrt(), method='intrinsic')
        out_w = solver.forward(rig, rest_lengths=solver.actual_length2.sqrt(),
                               method='woodbury', area_weighted=False, use_kkt=False)
        ns, Es = sim_nuE(mesh)
        print(f"{eta:>5.1f} | {ns:>7.3f} {float(out_i['poisson']):>7.3f} {float(out_w['poisson']):>8.3f}"
              f" | {Es:>8.4f} {float(out_i['young']):>8.4f} {float(out_w['young']):>8.4f}", flush=True)

    # (C) toggles
    print("\n(C) toggle check (eta=0.3): each flag changes the result; default == explicit all-on")
    mesh = pda.build_periodic_tf_mesh(N, 0.3, seed=0); solver = make_solver(mesh)
    rl = solver.actual_length2.sqrt(); rig = torch.ones(len(mesh['simplices']), 3, dtype=torch.float64)
    base = solver.forward(rig, rest_lengths=rl, method='intrinsic')              # all on (default)
    allon = solver.forward(rig, rest_lengths=rl, method='intrinsic',
                           area_weighted=True, use_kkt=True, use_angle_kkt=True)
    no_edge = solver.forward(rig, rest_lengths=rl, method='intrinsic', use_kkt=False)
    no_ang = solver.forward(rig, rest_lengths=rl, method='intrinsic', use_angle_kkt=False)
    no_aw = solver.forward(rig, rest_lengths=rl, method='intrinsic', area_weighted=False)
    def nu(o): return float(o['poisson'])
    print(f"   default nu          = {nu(base):.4f}")
    print(f"   explicit all-on nu  = {nu(allon):.4f}   (default==all-on: {abs(nu(base)-nu(allon))<1e-9})")
    print(f"   edge OFF nu         = {nu(no_edge):.4f}")
    print(f"   angle/curv OFF nu   = {nu(no_ang):.4f}")
    print(f"   area-weight OFF nu  = {nu(no_aw):.4f}   (sim nu={sim_nuE(mesh)[0]:.4f})")


if __name__ == '__main__':
    main()
