"""
OPEN-DOMAIN (non-periodic) verification of forward(method='intrinsic') vs simulation.

Every prior sim-vs-solver check is periodic (PBC). This validates the solver on OPEN foam /
crystal meshes (free boundary), whose interior-edge / interior-vertex constraint handling was
never checked against a simulation.

Two independent simulation controls, both compared against the SAME solver:

  (1) STRAIN control — clamp every boundary node to the affine displacement u_b=(F-I)X_b
      (boundary fluctuation = 0), relax the interior, assemble C_eff from the 3 macro modes.
      Gives per-triangle W(s), the full C_eff tensor, and (nu,E). This is the affine-clamped
      patch method with the patch = the whole interior. nu comes from C_eff (transverse info
      is in the reaction stress C12), not from a relaxed transverse displacement.

  (2) STRESS control — uniaxial-x: grip thin left/right slabs in x (transverse y free, top/
      bottom traction-free), relax, read the transverse contraction directly:
      nu = -eyy/exx, E = sxx/exx from the grip reaction. The physical uniaxial test
      (directional: for the anisotropic crystal it reports E_x, nu_xy).

Networks: crystalline (isotropic AND anisotropic/rotated, where orientation matters),
disordered foam, and virtual-distortion (VD) rigidity contrast, each swept small -> large.
Agreement is judged by (a) per-triangle W(s) correlation (strain), (b) full C_eff relative
error, and (c) nu,E. Runs at l0 = actual length (zero prestress, the validated regime).

Plots are written next to the solver: Phase 2/verification_open_domain/.
"""
import os, sys
import numpy as np
import scipy.sparse.linalg as spla
from collections import Counter
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(HERE, '..', 'Phase 2'))
import forward_solver_torch as fst
import Disc_2_Cont_optimized as D2C
import test_cluster_Ceff as CE          # tri_metric_change, vec3, c6_to_nuE, MODES, DELTA
# clean_tri / build_open_mesh MOVED to the core layer by the A-7b re-layering
# (Phase 2/mesh_build.py): Phase 3/inverse_design.py's DesignProblem.open is built on them, and the
# design layer must not depend on this retireable oracle layer. Imported back here, where this
# module uses them.
from mesh_build import clean_tri, build_open_mesh
import metric_ops as MO
import sim_assembly as SA
torch.set_default_dtype(torch.float64)

DELTA = CE.DELTA
MODES = CE.MODES
Fk   = [np.eye(2) + DELTA * M for M in MODES]
Dgt  = [F.T @ F - np.eye(2) for F in Fk]
Dinv = np.linalg.inv(np.stack([MO.vec3(g) for g in Dgt], 1))
OUTDIR = os.path.join(HERE, '..', 'Phase 2', 'verification_open_domain')


def boundary_nodes(mesh):
    """Vertices on a boundary edge (edge in exactly one triangle)."""
    cnt = Counter()
    tb = mesh['tri_bond']
    for ti in range(len(tb)):
        for ei in range(3):
            cnt[int(tb[ti, ei])] += 1
    bnd = set()
    for b, c in cnt.items():
        if c == 1:
            bnd.add(int(mesh['bond_u'][b])); bnd.add(int(mesh['bond_v'][b]))
    return bnd


def homog_C6(mesh, W3):
    """Area-weighted homogenised elastic tensor (6,) from per-triangle response W3 (N,3,3),
    in the SOLVER's internal (bare-tensor /16, metric) convention — for apples-to-apples relC."""
    W9 = torch.as_tensor(W3.reshape(-1, 9)); A = torch.as_tensor(MO.bare_tensor(mesh))
    C6 = fst._compute_actual_elastic_tensor(A, W9).numpy()
    w = mesh['areas'] / mesh['areas'].sum()
    return (C6 * w[:, None]).sum(0)


def virial_stress(mesh, du_bond):
    """Physical Cauchy stress (2,2) via the bond virial for a given relative bond displacement."""
    R = mesh['bond_R']; L = np.sqrt((R ** 2).sum(1)); kb = mesh['bond_k']
    T = kb * (R * du_bond).sum(1) / L
    return np.einsum('b,bi,bj->ij', T / L, R, R) / mesh['areas'].sum()


def phys_nuE_from_modes(mesh, u_modes):
    """Physical (real-unit) nu, E from the 3 macro-mode relaxed fluctuation fields, via virial."""
    Cv = np.zeros((3, 3))                                  # cols=modes, rows=[sxx,syy,sxy]
    for k, F in enumerate(Fk):
        u = u_modes[k]
        dub = mesh['bond_R'] @ (F - np.eye(2)).T + (u[mesh['bond_v']] - u[mesh['bond_u']])
        S = virial_stress(mesh, dub)
        Cv[:, k] = [S[0, 0], S[1, 1], S[0, 1]]
    Cv /= DELTA                                            # E_mat = DELTA*I for these 3 modes
    Sc = np.linalg.inv(Cv); Ex, Ey = 1 / Sc[0, 0], 1 / Sc[1, 1]
    return 0.5 * (-Sc[1, 0] * Ex - Sc[0, 1] * Ey), 0.5 * (Ex + Ey)


def sim_strain(mesh, bnd):
    """Affine-clamped boundary (fluctuation=0), relax interior. Returns per-triangle W3(s),
    the internal-convention C6 (for relC vs solver), and PHYSICAL nu,E (virial, real units)."""
    ev, sx = mesh['edge_vecs'], mesh['simplices']; nn = len(mesh['pts']); nt = len(sx)
    K, _ = SA.assemble_K_faff(mesh, np.eye(2))
    interior = np.array(sorted(set(range(nn)) - bnd))
    fdof = np.sort(np.concatenate([2 * interior, 2 * interior + 1]))
    Kff = K[fdof][:, fdof].tocsc()
    D = np.zeros((nt, 3, 3)); u_modes = []
    for k, F in enumerate(Fk):
        fa = SA.assemble_K_faff(mesh, F)[1]
        u = np.zeros(2 * nn)
        u[fdof] = spla.spsolve(Kff, -fa[fdof])
        Uk = u.reshape(nn, 2); u_modes.append(Uk)
        D[:, :, k] = MO.vec3(MO.tri_metric_change(ev, sx, F, Uk) - Dgt[k])
    W3 = D @ Dinv
    C6 = homog_C6(mesh, W3)
    nu, E = phys_nuE_from_modes(mesh, u_modes)             # physical (real units)
    return W3, C6, nu, E


def sim_stress(mesh, eps_ax=1e-3, grip_frac=0.10):
    """Uniaxial-x: grip thin left/right slabs in x (transverse free), relax, then MEASURE the
    macroscopic strain and stress from the relaxed field (no grip-reaction / bbox normalisation):
      - strain ε = sym(∂u/∂X) from an affine least-squares fit over the interior (non-grip) nodes;
      - stress σ = (1/A_tot) Σ_b (T_b/ℓ_b) R_b⊗R_b  (bond virial), T_b = k_b·δℓ_b, δℓ_b=R_b·Δu_b/ℓ_b.
    ν = −ε_yy/ε_xx, E = σ_xx/ε_xx; also returns σ_yy/σ_xx as a uniaxiality check."""
    pts = mesh['pts']; nn = len(pts)
    K, _ = SA.assemble_K_faff(mesh, np.eye(2))
    x, y = pts[:, 0], pts[:, 1]
    xmin, xmax = x.min(), x.max(); W = xmax - xmin
    left  = np.where(x <= xmin + grip_frac * W)[0]
    right = np.where(x >= xmax - grip_frac * W)[0]
    grips = np.concatenate([left, right])
    xL = x[left].mean()
    pin = int(left[np.argmin(np.abs(y[left] - y[left].mean()))])
    fixed = np.concatenate([2 * grips, [2 * pin + 1]])
    u = np.zeros(2 * nn)
    u[2 * grips] = eps_ax * (x[grips] - xL)              # load: affine x-stretch on the grips
    free = np.setdiff1d(np.arange(2 * nn), fixed)
    u[free] = spla.spsolve(K[free][:, free].tocsc(), -(K[free][:, fixed] @ u[fixed]))
    U = u.reshape(nn, 2)

    # macroscopic strain: affine fit u ≈ A·X + b over interior (non-grip) nodes
    sel = np.setdiff1d(np.arange(nn), grips)
    Xa = np.column_stack([pts[sel], np.ones(len(sel))])
    coef, *_ = np.linalg.lstsq(Xa, U[sel], rcond=None)     # (3,2): d(u)/d[X,Y,1]
    A = coef[:2].T                                         # ∂u_i/∂X_j
    eps = 0.5 * (A + A.T)
    eps_xx, eps_yy = eps[0, 0], eps[1, 1]

    # macroscopic stress: bond virial over all bonds / total meshed area
    R = mesh['bond_R']; L = np.sqrt((R ** 2).sum(1)); kb = mesh['bond_k']
    du = U[mesh['bond_v']] - U[mesh['bond_u']]
    T = kb * (R * du).sum(1) / L                          # bond tension
    Sig = np.einsum('b,bi,bj->ij', T / L, R, R) / mesh['areas'].sum()

    nu = -eps_yy / eps_xx
    E = Sig[0, 0] / eps_xx
    return nu, E, Sig[1, 1] / Sig[0, 0]


def solver_out(tri, mesh):
    """forward(method='intrinsic') on the open mesh -> W3(s), C6, nu, E (nu/E via same c6_to_nuE)."""
    solver, _, default_rl = fst.from_triangulation(clean_tri(tri))
    out = solver.forward(torch.as_tensor(mesh['tri_k'], dtype=torch.float64),
                         rest_lengths=default_rl, method='intrinsic')
    W3 = out['W'].detach().numpy().reshape(-1, 3, 3)
    C6 = out['elastic_tensor'].detach().numpy()
    nu, E = CE.c6_to_nuE(C6)
    return W3, C6, nu, E


CASES = [
    # name, tri_builder(size)->triangulation, vd_a, show iso nu=1/3 reference
    ('crystal_iso',   lambda s: D2C.generate_cryratl_points((s, s), (1, 1), 0.0),          None,  True),
    ('crystal_aniso', lambda s: D2C.generate_cryratl_points((s, s), (1.5, 0.8), np.pi/6),  None,  False),
    ('disordered',    lambda s: D2C.generate_foam_points((s, s), 0.3),                     None,  False),
    ('VD_a+5',        lambda s: D2C.generate_foam_points((s, s), 0.3),                     5.0,   False),
]
SIZES = [3, 5, 8]


def run_case(name, builder, vd_a, show_ref):
    rec = {k: [] for k in ['ntri', 'nu_sc', 'E_sc', 'nu_ss', 'E_ss', 'nu_sv', 'E_sv',
                           'relC', 'corr_all', 'corr_int']}
    big = None
    print(f"\n=== {name} ===")
    print(f"{'size':>4} {'ntri':>6} | {'nu_strain':>9} {'nu_stress':>9} {'nu_solv':>8} | "
          f"{'E_strain':>8} {'E_stress':>8} {'E_solv':>8} | {'relC':>8} {'corrInt':>7} {'sYY/sXX':>7}")
    for s in SIZES:
        np.random.seed(0)
        tri = clean_tri(builder(s))
        mesh = build_open_mesh(tri, vd_a=vd_a)
        bnd = boundary_nodes(mesh)
        nt = len(mesh['simplices'])
        W3_sc, C6_sc, nu_sc, E_sc = sim_strain(mesh, bnd)
        nu_ss, E_ss, uniax = sim_stress(mesh)
        W3_sv, C6_sv, nu_sv, E_sv = solver_out(tri, mesh)
        relC = np.linalg.norm(C6_sv - C6_sc) / max(np.linalg.norm(C6_sc), 1e-30)
        fs, fv = W3_sc.reshape(-1, 9), W3_sv.reshape(-1, 9)
        tri_is_bnd = np.array([any(int(v) in bnd for v in mesh['simplices'][t]) for t in range(nt)])
        corr_all = np.corrcoef(fs.ravel(), fv.ravel())[0, 1]
        intm = ~tri_is_bnd
        corr_int = np.corrcoef(fs[intm].ravel(), fv[intm].ravel())[0, 1] if intm.sum() > 1 else np.nan
        for k, v in zip(rec, [nt, nu_sc, E_sc, nu_ss, E_ss, nu_sv, E_sv, relC, corr_all, corr_int]):
            rec[k].append(v)
        print(f"{s:>4} {nt:>6} | {nu_sc:>9.3f} {nu_ss:>9.3f} {nu_sv:>8.3f} | "
              f"{E_sc:>8.4f} {E_ss:>8.4f} {E_sv:>8.4f} | {relC:>8.1e} {corr_int:>7.4f} {uniax:>7.3f}",
              flush=True)
        big = dict(mesh=mesh, W3_sc=W3_sc, W3_sv=W3_sv, tri_is_bnd=tri_is_bnd,
                   corr_all=corr_all, corr_int=corr_int, nt=nt)

    nt = np.array(rec['ntri'])
    fig, ax = plt.subplots(2, 2, figsize=(13, 10))

    a = ax[0, 0]
    a.plot(nt, rec['nu_sc'], 'k-o', lw=2, label='strain-ctrl sim')
    a.plot(nt, rec['nu_ss'], '-^', color='#2ca02c', lw=2, label='stress-ctrl sim (uniax-x)')
    a.plot(nt, rec['nu_sv'], '--s', color='#d62728', lw=2, label='intrinsic solver')
    if show_ref:
        a.axhline(1/3, color='gray', lw=0.8, ls=':', label='ν=1/3 (iso ref)')
    a.set_xlabel('# triangles'); a.set_ylabel('ν'); a.set_title('Poisson ratio vs mesh size (comparable)')
    a.legend(fontsize=8); a.grid(alpha=0.3)

    a = ax[0, 1]
    a.plot(nt, rec['E_sc'], 'k-o', lw=2, label='strain-ctrl sim (physical)')
    a.plot(nt, rec['E_ss'], '-^', color='#2ca02c', lw=2, label='stress-ctrl sim (physical)')
    if show_ref:
        a.axhline(2/np.sqrt(3), color='gray', lw=0.8, ls=':', label='E=2/√3 (iso ref)')
    a.set_xlabel('# triangles'); a.set_ylabel('E (physical units)')
    a.set_title("Young's modulus — physical cross-check\n(solver E is internal-convention; see relC)")
    a.legend(fontsize=8); a.grid(alpha=0.3)

    a = ax[1, 0]
    fs, fv = big['W3_sc'].reshape(-1, 9), big['W3_sv'].reshape(-1, 9)
    ib = big['tri_is_bnd']
    a.scatter(fs[~ib].ravel(), fv[~ib].ravel(), s=6, alpha=0.4, color='#1f77b4', label='interior tri')
    a.scatter(fs[ib].ravel(), fv[ib].ravel(), s=6, alpha=0.4, color='#ff7f0e', label='boundary tri')
    lo, hi = fs.min(), fs.max(); a.plot([lo, hi], [lo, hi], 'k--', lw=1)
    a.set_xlabel('sim W3(s) [strain ctrl]'); a.set_ylabel('solver W3(s)')
    a.set_title(f"per-triangle W parity (largest, {big['nt']} tri)\n"
                f"corr all={big['corr_all']:.4f}, interior={big['corr_int']:.4f}")
    a.legend(fontsize=8); a.grid(alpha=0.3)

    a = ax[1, 1]
    a.semilogy(nt, rec['relC'], 'k-o', lw=2)
    a.set_xlabel('# triangles'); a.set_ylabel('‖C_solver − C_sim‖ / ‖C_sim‖')
    a.set_title('Full C_eff tensor relative error (strain ctrl)'); a.grid(alpha=0.3, which='both')

    fig.suptitle(f'Open-domain verification — forward(intrinsic) vs simulation: {name}',
                 fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    p = os.path.join(OUTDIR, f'open_{name}.png')
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print('saved', p)
    return rec


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    for name, builder, vd_a, show_ref in CASES:
        run_case(name, builder, vd_a, show_ref)


if __name__ == '__main__':
    main()
