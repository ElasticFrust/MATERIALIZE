"""
Complete eta sweep of the PRODUCTION forward solver (ElasticSolver.forward, method=
'intrinsic') vs the PBC simulation, for the disordered (geometric, k=1) case and for
virtual-distortion (VD) rigidity contrasts a in {-10,-2,5,10,100}.

  k_bond = 1 + tanh(a*(|R|-1))   (a>0 stiffer where stretched, a<0 where compressed)

Two runs:
  (1) ensemble:  N=20 (20x20, 800 triangles), 10 realisations per eta  -> mean +/- std band
  (2) single:    N=50 (50x50, 5000 triangles), one realisation

Both plot homogenised Poisson ratio nu and Young's modulus E vs eta, forward solver vs sim.
The forward solver is driven exactly: rigidities = per-edge bond k (uniform for disordered,
tri_k for VD), rest_lengths = reference edge length, so its bare tensor matches the sim.
"""
import os, sys, time
import numpy as np
import scipy.sparse.linalg as spla
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..', 'Phase 2'))
import forward_solver_torch as fst
import pbc_dg_analysis as pda
import test_cluster_Ceff as CE
import test_cluster_rigidity as TR
import test_cluster_VD as VD
from test_intrinsic_VD import kkt_from_tri_bond
import physical_homog as PH
torch.set_default_dtype(torch.float64)

DELTA = CE.DELTA
MODES = CE.MODES
CONTRASTS = [-10, -2, 5, 10, 100]
CASES = ['disordered'] + [f'VD a={a:+d}' for a in CONTRASTS]
ETAS = np.round(np.arange(0.0, 0.5001, 0.05), 4)      # complete sweep, 11 points

Fk   = [np.eye(2) + DELTA * M for M in MODES]
Dg_k = [CE.vec3(F.T @ F - np.eye(2)) for F in Fk]
Dinv = np.linalg.inv(np.stack(Dg_k, axis=1))


def _mount(solver, ev, l2, areas, kkt, sx):
    """Override an ElasticSolver's geometry buffers with periodic-correct arrays."""
    solver.edge_vecs      = torch.as_tensor(ev, dtype=torch.float64)
    solver.actual_length2 = torch.as_tensor(l2, dtype=torch.float64)
    solver.area_weights   = torch.as_tensor(areas / areas.sum(), dtype=torch.float64)
    solver.kkt_arrays     = kkt
    solver._build_intrinsic_constraints(np.asarray(sx))
    return solver


def make_solver(geo, kkt):
    pts, sx, ev = geo['pts'], geo['simplices'], geo['edge_vecs']
    edges = np.stack([sx[:, [1, 0]], sx[:, [2, 0]], sx[:, [2, 1]]], axis=1)   # (e01,e02,e12)
    s = fst.ElasticSolver(pts, sx, edges)
    return _mount(s, ev, (ev ** 2).sum(2), geo['areas'], kkt, sx)


def sim_nuE(mesh, assemble, bare=None):
    """PHYSICAL (virial = energy) homogenised nu, E from the relaxed PBC network (truth).
    (Was the legacy metric average CE.Ceff_nuE; that biased ν on disordered/anisotropic meshes.
    `bare` is kept for call-site compatibility but unused.)"""
    free = np.arange(2, 2 * len(mesh['pts']))
    return PH.sim_nuE(mesh, free, assemble)


def solver_nuE(solver, rig_np, rl):
    out = solver.forward(torch.as_tensor(rig_np, dtype=torch.float64),
                         rest_lengths=rl, method='intrinsic', physical_units=True)
    return float(out['poisson']), float(out['young'])


def one_realisation(N, eta, seed):
    """Return {case: (nu_sim, E_sim, nu_solver, E_solver)} for one geometry realisation."""
    out = {}
    # ---- disordered (geometric, k=1) ----
    mesh = pda.build_periodic_tf_mesh(N, float(eta), seed=seed)
    s = make_solver(mesh, mesh['kkt_arrays'])
    rl = s.actual_length2.sqrt()
    rig1 = np.ones((len(mesh['simplices']), 3))
    ns, Es = sim_nuE(mesh, pda._assemble_K_and_faff, CE.bare_tensor(mesh))
    ni, Ei = solver_nuE(s, rig1, rl)
    out['disordered'] = (ns, Es, ni, Ei)
    # ---- VD contrasts (one geometry, constraints reused across contrasts) ----
    geo = VD.build_geometry(N, float(eta), seed=seed)
    kkt = kkt_from_tri_bond(geo['tri_bond'], geo['edge_vecs'])
    sv  = make_solver(geo, kkt)
    rlv = torch.as_tensor(np.sqrt(geo['actual_len2']), dtype=torch.float64)
    for a in CONTRASTS:
        VD.set_VD(geo, a)
        ns, Es = sim_nuE(geo, TR.assemble_K_faff, TR.bare_tensor(geo))
        ni, Ei = solver_nuE(sv, geo['tri_k'], rlv)
        out[f'VD a={a:+d}'] = (ns, Es, ni, Ei)
    return out


def run(N, seeds):
    """results[case][quantity] -> array (n_eta, n_seed)."""
    keys = ['nu_s', 'E_s', 'nu_i', 'E_i']
    res = {c: {k: np.full((len(ETAS), len(seeds)), np.nan) for k in keys} for c in CASES}
    t0 = time.time()
    for ie, eta in enumerate(ETAS):
        for js, seed in enumerate(seeds):
            r = one_realisation(N, eta, seed)
            for c in CASES:
                ns, Es, ni, Ei = r[c]
                res[c]['nu_s'][ie, js] = ns; res[c]['E_s'][ie, js] = Es
                res[c]['nu_i'][ie, js] = ni; res[c]['E_i'][ie, js] = Ei
        print(f"  N={N} eta={eta:.2f} done ({len(seeds)} real)  [{time.time()-t0:.0f}s]", flush=True)
    return res


def plot(res, N, n_seed, fname, title):
    nC = len(CASES)
    fig, axes = plt.subplots(nC, 2, figsize=(11, 2.5 * nC), squeeze=False)
    band = n_seed > 1
    for i, c in enumerate(CASES):
        for j, (qs, qi, ttl) in enumerate([('nu_s', 'nu_i', "Poisson ratio ν"),
                                           ('E_s', 'E_i', "Young's modulus E")]):
            ax = axes[i, j]
            for q, col, lab in [(qs, 'k', 'simulation'), (qi, '#d62728', 'forward solver')]:
                m = np.nanmean(res[c][q], axis=1)
                ax.plot(ETAS, m, '-o', color=col, ms=4, lw=2.0, label=lab)
                if band:
                    sd = np.nanstd(res[c][q], axis=1)
                    ax.fill_between(ETAS, m - sd, m + sd, color=col, alpha=0.18, lw=0)
            if qs == 'nu_s':
                ax.axhline(0, color='gray', lw=0.5, ls=':')
            ax.set_ylabel(f'{ttl}\n({c})', fontsize=9); ax.grid(alpha=0.3)
            if i == 0 and j == 0:
                ax.legend(fontsize=8)
            if i == nC - 1:
                ax.set_xlabel('η')
    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    p = os.path.join(HERE, 'plots', fname)
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print('saved', p, flush=True)
    return p


def main():
    os.makedirs(os.path.join(HERE, 'plots'), exist_ok=True)
    # (1) ensemble 20x20, 10 realisations
    print("=== ensemble N=20 (20x20), 10 realisations ===", flush=True)
    res20 = run(20, list(range(10)))
    plot(res20, 20, 10, 'dg_solver_sweep_20x20_ens10.png',
         'Forward solver vs simulation — 20×20, 10 realisations (mean ± std)')
    np.savez(os.path.join(HERE, 'plots', 'dg_solver_sweep_20x20.npz'),
             etas=ETAS, cases=np.array(CASES, object),
             **{f'{c}__{k}': res20[c][k] for c in CASES for k in res20[c]})
    # (2) single 50x50 realisation
    print("=== single N=50 (50x50), 1 realisation ===", flush=True)
    res50 = run(50, [0])
    plot(res50, 50, 1, 'dg_solver_sweep_50x50_single.png',
         'Forward solver vs simulation — single 50×50 realisation')
    np.savez(os.path.join(HERE, 'plots', 'dg_solver_sweep_50x50.npz'),
             etas=ETAS, cases=np.array(CASES, object),
             **{f'{c}__{k}': res50[c][k] for c in CASES for k in res50[c]})


if __name__ == '__main__':
    main()
