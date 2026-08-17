r"""nu(d) for the hexagon family on LINEAR axes, against the closed-form solution.

Restricted to d <= 2 on purpose: nu diverges as d -> 3 (like 1/y^2), so a linear axis is only
readable below the regular hexagon. Over d in [0.05, 2] the range is nu in [-0.32, +1].

CLOSED FORM (user, 2026-08-17), for a rigid perimeter with FREE hinges, r = d/2:

    nu(r) = -((2 - 4r)(r + 1/2)) / (3 + 4r - 4r^2)  =  (4r^2 - 1) / (3 + 4r - 4r^2)

Derivation. All six perimeter edges have length 1, so with rim vertices
V0=(r,0), V1=(1/2,y), V2=(-1/2,y), V3=(-r,0), V4=(-1/2,-y), V5=(1/2,-y):

    |V0V1| = 1  =>  (r-1/2)^2 + y^2 = 1  =>  y^2 = 1 - (r-1/2)^2 = (3 + 4r - 4r^2)/4

The cell dimensions along the mechanism are

    L_x = 1 + 2r        <-- the LATTICE CONSTANT (a1 = (1+d,0)), NOT the hexagon's width 2r
    L_y = 2y

and differentiating along the single soft mode (dy/dr = -(r-1/2)/y):

    eps_xx = 2 dr/(1+2r),   eps_yy = -(r-1/2) dr / y^2
    nu = -eps_yy/eps_xx = (r-1/2)(1+2r) / (2 y^2) = (4r^2-1)/(3+4r-4r^2)

**The subtle step is L_x.** Using the hexagon's own width 2r instead of the lattice constant 1+2r
gives (4r^2-2r)/(3+4r-4r^2), which predicts nu = 2/3 at the regular hexagon instead of 1.

nu is measured by PULLING ALONG the diameter and reading the perpendicular response,
nu = -eps_yy/eps_xx = -S[1,0]/S[0,0] from the compliance S = C^-1 in Voigt [xx,yy,xy].

Curves drawn: the closed form; the solver in the FREE-HINGE limit (k_spoke=1e-8, which reproduces it
to 4 significant figures); the solver at the default k_spoke=1e-3; and the INDEPENDENT sim on the
periodic lattice. The k=1e-3 offset is not solver error — it is the finite hinge stiffness, and it is
first order in k_spoke (a tenfold reduction divides the error by ten).

Run:  python hex_nu_linear.py [n_points]
Out:  Phase 5/results/hex_validation/hex_nu_linear.png
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'Phase 3', 'verifications'))
import _common as C                                                    # noqa: E402
sys.path.insert(0, os.path.join(REPO, 'Phase 5'))
sys.path.insert(0, os.path.join(REPO, 'Phase 2'))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                        # noqa: E402
import mesh_build as MB                                                # noqa: E402
import solver_build as SB                                              # noqa: E402
import plotting as P                                                   # noqa: E402
import single_hexagon as S                                             # noqa: E402
import hex_solver_validation as HV                                     # noqa: E402
from inverse_design import DesignProblem                               # noqa: E402

torch.set_default_dtype(torch.float64)
OUT = os.path.join(REPO, 'Phase 5', 'results', 'hex_validation')


def nu_closed_form(d):
    r = np.asarray(d, float) / 2.0
    return -((2 - 4 * r) * (r + 0.5)) / (3 + 4 * r - 4 * r * r)


def _nu_from_c6(c6):
    c6 = np.asarray(c6, float)
    Cv = np.array([[c6[0], c6[2], c6[1]], [c6[2], c6[5], c6[4]], [c6[1], c6[4], c6[3]]])
    Sm = np.linalg.inv(Cv)
    return float(-Sm[1, 0] / Sm[0, 0])            # pull along x (the diameter), read y


def nu_solver_single(d, k_spoke):
    """Single OPEN hexagon: 7 nodes, 6 triangles, no periodicity."""
    tri, _ = S.hexagon(d)
    mesh = MB.build_open_mesh(tri)
    u, v = np.asarray(mesh['bond_u']), np.asarray(mesh['bond_v'])
    k = np.where((u == 6) | (v == 6), k_spoke, 1.0)          # index 6 = the centre vertex
    m = dict(mesh); m['bond_k'] = k; m['tri_k'] = k[m['tri_bond']]
    sv = SB.make_solver(m, MB.kkt_from_tri_bond(m['tri_bond'], m['edge_vecs']))
    o = sv.forward(torch.as_tensor(m['tri_k']),
                   rest_lengths=torch.as_tensor(np.sqrt(m['actual_len2'])),
                   method='intrinsic', physical_units=True)
    return _nu_from_c6(o['elastic_tensor'].detach().numpy())


def nu_sim_lattice(d, k_spoke=1e-3):
    geo, k0, is_soft = HV.build_dhex(2, 2, float(d), eps=k_spoke)
    C.apply_k_to_geo(geo, k0)
    return _nu_from_c6(C.sim_bulk_C6(geo))


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    os.makedirs(OUT, exist_ok=True)
    ds = np.linspace(0.05, 2.0, n)

    exact = nu_closed_form(ds)
    free = np.array([nu_solver_single(d, 1e-8) for d in ds])
    dflt = np.array([nu_solver_single(d, 1e-3) for d in ds])
    sim = []
    for d in ds:
        try:
            sim.append(nu_sim_lattice(d))
        except Exception:
            sim.append(np.nan)
    sim = np.array(sim)
    np.savez(os.path.join(OUT, 'hex_nu_linear.npz'), d=ds, exact=exact, free=free,
             default=dflt, sim=sim)

    fig, axes = plt.subplots(1, 2, figsize=(P.STYLE.PANEL[0] * 2.3, P.STYLE.PANEL[1] * 1.45))
    ax = axes[0]
    ax.plot(ds, exact, '-', color='k', lw=3.0, alpha=0.30, label=r'closed form  $(4r^2-1)/(3+4r-4r^2)$')
    ax.plot(ds, free, '-', color='tab:blue', lw=1.8, label=r'solver, free hinges ($k_{spoke}=10^{-8}$)')
    ax.plot(ds, dflt, '--', color='tab:red', lw=1.5, label=r'solver, default ($k_{spoke}=10^{-3}$)')
    ax.plot(ds, sim, ':', color='tab:green', lw=1.8, label=r'independent sim ($k_{spoke}=10^{-3}$)')
    ax.axhline(0, color='0.6', lw=0.8)
    ax.axvline(1.0, color='0.75', lw=0.8, ls=':')
    ax.axvline(2.0, color='0.75', lw=0.8, ls=':')
    ax.plot([2.0], [1.0], 'o', ms=8, mfc='none', mec='tab:orange', mew=2)
    ax.annotate('regular hexagon\n' + r'$\nu=1$', xy=(2.0, 1.0), xytext=(1.45, 0.72), fontsize=8.5,
                color='tab:orange', arrowprops=dict(arrowstyle='->', color='tab:orange', lw=1.2))
    ax.annotate(r'$\nu=0$ at $d=1$', xy=(1.0, 0.0), xytext=(0.62, 0.30), fontsize=8.5,
                color='0.35', arrowprops=dict(arrowstyle='->', color='0.5', lw=1.0))
    ax.text(0.30, -0.26, 're-entrant\n(auxetic)', fontsize=8.5, color='tab:purple', ha='center')
    ax.set_xlabel('diameter  $d$   ($r=d/2$)')
    ax.set_ylabel(r'$\nu$   (pull ALONG $d$, read $\perp$)')
    ax.set_title('LINEAR axes, $d\\leq 2$', fontsize=10)
    ax.grid(alpha=0.25)

    ax2 = axes[1]
    ax2.semilogy(ds, np.abs(dflt - exact) / np.maximum(np.abs(exact), 1e-12), '--',
                 color='tab:red', lw=1.5, label=r'$k_{spoke}=10^{-3}$')
    ax2.semilogy(ds, np.abs(free - exact) / np.maximum(np.abs(exact), 1e-12), '-',
                 color='tab:blue', lw=1.8, label=r'$k_{spoke}=10^{-8}$')
    ax2.set_xlabel('diameter  $d$')
    ax2.set_ylabel('relative deviation from the closed form')
    ax2.set_title('the residual IS the finite hinge stiffness\n(first order in $k_{spoke}$)', fontsize=9.5)
    ax2.grid(alpha=0.25, which='both')
    ax2.legend(fontsize=8.5, frameon=False)

    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=4, fontsize=8.5, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle('Hexagon family: solver vs the CLOSED-FORM solution (rigid perimeter, free hinges)',
                 fontsize=11)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    P.save_fig(fig, os.path.join(OUT, 'hex_nu_linear.png'))
    print('wrote', os.path.join(OUT, 'hex_nu_linear.png'))
    ok = np.isfinite(free) & np.isfinite(exact)
    print('max |solver(free) - closed form| over d<=2 : %.3e'
          % np.max(np.abs(free[ok] - exact[ok])))
    fin = np.isfinite(sim) & np.isfinite(dflt)
    print('max |solver - sim| at k=1e-3              : %.3e'
          % np.max(np.abs(dflt[fin] - sim[fin])))


if __name__ == '__main__':
    main()
