"""
showcase / fig3 — ISOTROPIZE: design k so an ANISOTROPIC base becomes ISOTROPIC at a chosen level ν0
(isotropic is not only 1/3). One panel per anisotropic base: its native (uniform-k) ν(θ) wobble, then
the designed ν(θ) flattened onto each chosen ν0. Each panel carries a small INSET of an isotropized
network. Self-contained: designs freshly (proper budget) + verifies with the independent PBC sim.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

HERE = os.path.dirname(os.path.abspath(__file__))
VER = os.path.dirname(HERE)
sys.path.insert(0, VER)
import _common as C

BASES = [('aniso_str', 'rows compressed (ψ=0.6)'), ('aniso_shr', 'sheared rows (φ=1.5)'),
         ('disorder_hi', 'disordered η=0.35')]
NU0S = [0.6, 1 / 3., 0.0, -0.3, -0.5]                    # clean, achievable spread (incl. auxetic)
N, REG, NITER, NREST = 14, 1e-3, 340, 3   # bump budget: the ψ=0.6 base needs restarts+iters to flatten
INSET_NU0 = 1 / 3.
TH = C.ANG; DEG = np.degrees(TH)


def design(base, nu0):
    prob, geo = C.make_case(base, N)
    r = C.optimize(prob, [C.Objective('nu_theta', np.full_like(TH, nu0))],
                   mode='k', n_iter=NITER, n_restarts=NREST, reg=REG, verbose=False)
    C.apply_k_to_geo(geo, r['k'])
    nu = C.nu_E_theta(C.sim_bulk_C6(geo), TH)[0]
    return nu, geo, np.asarray(r['k'].detach().numpy(), float)


def base_nu(base):
    prob, geo = C.make_case(base, N)
    geo['bond_k'] = np.ones(len(geo['bond_u'])); geo['tri_k'] = geo['bond_k'][geo['tri_bond']]
    return C.nu_E_theta(C.sim_region_C6(geo), TH)[0]


def main():
    plt.rcParams.update({'font.size': 12})
    cmap = plt.cm.coolwarm_r; norm = plt.Normalize(-0.65, 0.65)
    fig, axes = plt.subplots(1, len(BASES), figsize=(6.2 * len(BASES), 6.0), squeeze=False)

    for ax, (base, blabel) in zip(axes[0], BASES):
        ax.plot(DEG, base_nu(base), color='0.35', ls=':', lw=2.6, label='native base (uniform k)')
        inset = None
        for nu0 in NU0S:
            nu, geo, k = design(base, nu0)
            ax.axhline(nu0, color=cmap(norm(nu0)), lw=0.7, ls='--', alpha=0.45)
            ax.plot(DEG, nu, '-', color=cmap(norm(nu0)), lw=2.2,
                    label=f'→ ν₀={nu0:+.2f}  (got {nu.mean():+.2f}±{nu.std():.02f})')
            if abs(nu0 - INSET_NU0) < 1e-6:
                inset = (geo, k)
        ax.set_title(f'{blabel}', fontsize=12); ax.set_xlabel('loading angle θ (deg)')
        ax.set_ylabel('ν(θ)  (independent sim)'); ax.set_xlim(0, 180); ax.set_ylim(-0.8, 0.9)
        ax.grid(alpha=0.3); ax.legend(fontsize=8.5, loc='lower center')
        print(f"  {base}: base wobble ±{base_nu(base).std():.2f}", flush=True)
        if inset is not None:
            from matplotlib.collections import LineCollection
            iax = inset_axes(ax, width='34%', height='34%', loc='upper right', borderpad=0.5)
            g = inset[0]; u = g['pts'][g['bond_u']]              # bare TOPOLOGY (fixed colour, no k)
            iax.add_collection(LineCollection(np.stack([u, u + g['bond_R']], 1), colors='0.3', linewidths=0.5))
            C.square_frame(iax, g); iax.set_title('topology', fontsize=8)

    fig.suptitle('ISOTROPIZE — flatten an anisotropic base to any chosen isotropic ν₀ '
                 '(dotted = native anisotropy; dashed = target; inset = designed network)',
                 fontsize=14, y=1.01)
    plt.tight_layout()
    out = os.path.join(HERE, 'fig3_isotropize.png')
    plt.savefig(out, dpi=200, bbox_inches='tight'); plt.close()
    print(f'saved {os.path.basename(out)}')


if __name__ == '__main__':
    main()
