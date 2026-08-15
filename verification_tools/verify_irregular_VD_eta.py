"""
Sim-vs-solver sanity check on IRREGULAR (disordered) networks, in PHYSICAL units.

Perturbed periodic triangular lattice at varying eta (0 -> 0.5; increasingly irregular), with
virtual-distortion (VD) rigidity contrast k = 1 + tanh(a*(|R|-1)) at a = -2, +2, +5. For each
(eta, a) we homogenise nu and E from:
  - the PHYSICAL simulation (virial = energy of the relaxed PBC network; physical_homog.sim_nuE),
  - forward(method='intrinsic', physical_units=True)  [the full-constraint solver, physical units].

Post-2026 fix: the solver's homogenisation is the UNWEIGHTED per-triangle average (+ physical_units
rescale), i.e. the true energy/virial effective tensor — so it is compared against the PHYSICAL
simulation, not the legacy metric average. All curves in one figure (rows = a, cols = nu / E).
"""
import os, sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..', 'Phase 2'))
import physical_homog as PH               # physical (virial/energy) reference
from mesh_build import kkt_from_tri_bond
from solver_build import make_solver
import mesh_build as MB
import sim_assembly as SA
torch.set_default_dtype(torch.float64)

CONTRASTS = [-2, 2, 5]
ETAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
N = 14


def solver_nuE(geo):
    kkt = kkt_from_tri_bond(geo['tri_bond'], geo['edge_vecs'])
    sv = make_solver(geo, kkt)
    rl = torch.as_tensor(np.sqrt(geo['actual_len2']), dtype=torch.float64)
    out = sv.forward(torch.as_tensor(geo['tri_k'], dtype=torch.float64),
                     rest_lengths=rl, method='intrinsic', physical_units=True)
    return float(out['poisson']), float(out['young'])


def main():
    res = {a: {k: [] for k in ['nu_s', 'E_s', 'nu_i', 'E_i']} for a in CONTRASTS}
    print(f"N={N}, irregular lattice, VD k=1+tanh(a*(|R|-1)); PHYSICAL solver vs PHYSICAL sim")
    for eta in ETAS:
        geo = MB.build_geometry(N, eta, seed=0)
        free = np.arange(2, 2 * len(geo['pts']))
        for a in CONTRASTS:
            MB.set_VD(geo, a)
            ns, Es = PH.sim_nuE(geo, free, SA.assemble_K_faff)
            ni, Ei = solver_nuE(geo)
            for kk, vv in zip(['nu_s', 'E_s', 'nu_i', 'E_i'], [ns, Es, ni, Ei]):
                res[a][kk].append(vv)
            print(f"  eta={eta:.1f} a={a:>+3}: nu sim/solver = {ns:+.3f}/{ni:+.3f}"
                  f"   E = {Es:.3f}/{Ei:.3f}", flush=True)

    nC = len(CONTRASTS)
    fig, axes = plt.subplots(nC, 2, figsize=(11, 3.0 * nC), squeeze=False)
    for i, a in enumerate(CONTRASTS):
        for j, (key, ttl) in enumerate([('nu', 'ν'), ('E', 'E')]):
            ax = axes[i, j]
            ax.plot(ETAS, res[a][f'{key}_s'], 'k-o', lw=2.2, ms=4, label='physical sim (truth)')
            ax.plot(ETAS, res[a][f'{key}_i'], '-^', color='#1f77b4', ms=6, label='forward solver')
            if key == 'nu':
                ax.axhline(0, color='gray', lw=0.5, ls=':')
            ax.set_ylabel(f'{ttl}  (a={a:+d})'); ax.grid(alpha=0.3)
            if i == 0:
                ax.set_title(f'{ttl} vs η'); ax.legend(fontsize=8)
            if i == nC - 1:
                ax.set_xlabel('η')
    fig.suptitle('Irregular networks, VD a=-2,+2,+5 — PHYSICAL forward solver vs physical simulation '
                 '(ν and E)', fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    os.makedirs(os.path.join(HERE, 'plots'), exist_ok=True)
    p = os.path.join(HERE, 'plots', 'dg_irregular_VD_eta_a-2_2_5.png')
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print('saved', p)


if __name__ == '__main__':
    main()
