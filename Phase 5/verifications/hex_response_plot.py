r"""The hexagon diameter family: BOTH directional responses, solver vs sim, high resolution.

Construction (user's spec): one hexagon, six perimeter edges HARD (k=1) and length 1, a centre
vertex with six SOFT radial spokes (k=1e-3) giving exactly 6 triangles. The separation `d` of two
opposite rim vertices is the control; every perimeter edge stays length 1, so

    y = sqrt(1 - ((1-d)/2)^2)      d=2 regular, d<1 re-entrant, d->0 bow-tie, d->3 flat sliver

Two directional Poisson ratios, from the SAME tensor (reciprocity ties them, checked to 1e-14):

    PULL ALONG d      nu = -eps_yy / eps_xx = -S[1,0]/S[0,0]     diverges as d->3
    PULL PERP to d    nu = -eps_xx / eps_yy = -S[0,1]/S[1,1]     -> y^2/2 as d->3

THREE curves per panel, because they should coincide:
    * solver, SINGLE hexagon  — 7 nodes, 6 triangles, NO periodicity (open mesh)
    * solver, LATTICE 2x2     — the periodic tiling of the same hexagon
    * SIM,    LATTICE 2x2     — the INDEPENDENT oracle (physical_homog virial), no solver code

Known anchors marked on the plot: nu=1 at the regular hexagon (d=2, textbook honeycomb, and
isotropic there since E_x=E_y), and nu=0 at d=1.

nu spans ~1e-3 to ~1e3, so the y axis is SYMLOG (linear near 0, log outside) — a linear axis would
render everything except the divergence as a flat line at zero.

Run:  python hex_response_plot.py [n_points]
Out:  Phase 5/results/hex_validation/hex_response.png
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
import physical_homog as PH                                            # noqa: E402
import plotting as P                                                   # noqa: E402
import single_hexagon as S                                             # noqa: E402
import hex_solver_validation as H                                      # noqa: E402
from inverse_design import DesignProblem                               # noqa: E402

torch.set_default_dtype(torch.float64)
OUT = os.path.join(REPO, 'Phase 5', 'results', 'hex_validation')


def both_nu(c6):
    """(nu_pull_along_d, nu_pull_perp) from a 6-vector; NaN if C is not invertible."""
    c6 = np.asarray(c6, float)
    Cv = np.array([[c6[0], c6[2], c6[1]], [c6[2], c6[5], c6[4]], [c6[1], c6[4], c6[3]]])
    try:
        Sm = np.linalg.inv(Cv)
    except np.linalg.LinAlgError:
        return np.nan, np.nan
    if abs(Sm[0, 0]) < 1e-300 or abs(Sm[1, 1]) < 1e-300:
        return np.nan, np.nan
    return float(-Sm[1, 0] / Sm[0, 0]), float(-Sm[0, 1] / Sm[1, 1])


def single_hex_c6(d):
    tri, _ = S.hexagon(d)
    mesh = MB.build_open_mesh(tri)
    k = S.k_of(mesh)
    m = dict(mesh); m['bond_k'] = k; m['tri_k'] = k[m['tri_bond']]
    sv = SB.make_solver(m, MB.kkt_from_tri_bond(m['tri_bond'], m['edge_vecs']))
    o = sv.forward(torch.as_tensor(m['tri_k']),
                   rest_lengths=torch.as_tensor(np.sqrt(m['actual_len2'])),
                   method='intrinsic', physical_units=True)
    return o['elastic_tensor'].detach().numpy()


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 140
    os.makedirs(OUT, exist_ok=True)
    ds = np.linspace(0.05, 2.98, n)

    rows = {kk: [] for kk in ('hex_d', 'hex_p', 'lat_d', 'lat_p', 'sim_d', 'sim_p')}
    for i, d in enumerate(ds):
        try:
            a, b = both_nu(single_hex_c6(float(d)))
        except Exception:
            a = b = np.nan
        rows['hex_d'].append(a); rows['hex_p'].append(b)
        try:
            g, k, _ = H.build_dhex(2, 2, float(d)); C.apply_k_to_geo(g, k)
            a2, b2 = both_nu(C.solver_region_C6(DesignProblem.from_geo(g), torch.as_tensor(k)))
        except Exception:
            a2 = b2 = np.nan
        rows['lat_d'].append(a2); rows['lat_p'].append(b2)
        try:
            a3, b3 = both_nu(C.sim_bulk_C6(g))
        except Exception:
            a3 = b3 = np.nan
        rows['sim_d'].append(a3); rows['sim_p'].append(b3)
        if (i + 1) % 20 == 0:
            print(f'  {i+1}/{n}', flush=True)
    R = {kk: np.array(v) for kk, v in rows.items()}
    np.savez(os.path.join(OUT, 'hex_response.npz'), d=ds, **R)

    fig, axes = plt.subplots(1, 2, figsize=(P.STYLE.PANEL[0] * 2.4, P.STYLE.PANEL[1] * 1.5))
    panels = ((0, 'PULL ALONG the diameter  (read perpendicular)', 'd'),
              (1, 'PULL PERPENDICULAR  (read along the diameter)', 'p'))
    for ax, (j, title, sfx) in zip(axes, panels):
        # solver vs sim ON THE SAME PANEL (CLAUDE.md plotting policy)
        ax.plot(ds, R['sim_' + sfx], '-', color='k', lw=3.2, alpha=0.30,
                label='SIM, lattice 2x2 (independent oracle)')
        ax.plot(ds, R['lat_' + sfx], '-', color='tab:blue', lw=1.8, label='solver, lattice 2x2')
        ax.plot(ds, R['hex_' + sfx], '--', color='tab:red', lw=1.6,
                label='solver, SINGLE hexagon (no PBC)')
        ax.set_yscale('symlog', linthresh=1e-2)
        ax.axhline(0, color='0.6', lw=0.8)
        ax.axvline(1.0, color='0.75', lw=0.8, ls=':')
        ax.axvline(2.0, color='0.75', lw=0.8, ls=':')
        ax.plot([2.0], [1.0], 'o', ms=7, mfc='none', mec='tab:green', mew=2,
                label=r'anchor: regular hexagon $\nu=1$' if j == 0 else None)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('diameter  d      (d=1 dotted, d=2 regular hexagon)')
        ax.grid(alpha=0.25, which='both')
    axes[0].set_ylabel(r'$\nu$   (symlog)')
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=4, fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle('Hexagon diameter family: both directional responses, solver vs independent sim\n'
                 'perimeter k=1 (len 1), centre spokes k=1e-3;  d<1 re-entrant, d=2 regular, '
                 r'd$\to$3 flat sliver', fontsize=11)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    P.save_fig(fig, os.path.join(OUT, 'hex_response.png'))
    print('wrote', os.path.join(OUT, 'hex_response.png'))

    fin = np.isfinite(R['sim_d']) & np.isfinite(R['lat_d'])
    print('max |solver-sim| over d, pull-along : %.3e' % np.max(np.abs(R['lat_d'][fin] - R['sim_d'][fin])))
    fin = np.isfinite(R['sim_p']) & np.isfinite(R['lat_p'])
    print('max |solver-sim| over d, pull-perp  : %.3e' % np.max(np.abs(R['lat_p'][fin] - R['sim_p'][fin])))


if __name__ == '__main__':
    main()
