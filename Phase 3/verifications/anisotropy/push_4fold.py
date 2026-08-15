"""
anisotropy / push_4fold — is the 4-fold symmetric-triangle ν(θ) undershoot a real wall or under-pushed?

The committed 4-fold (N=24, peak=0.5 triangle, reg=1e-3) reached ν∈[-0.078,+0.404] vs target
[-0.10,+0.50] (k median 0.39). Here we push the amplitude lever (lower reg) at N=24 and report whether
the peak/trough approach the ±target and how uniform the network stays. Overlays the ν(θ) curves and
prints range + floppiness per reg. (The reg=1e-3 reference is loaded from triangular_nu_4fold.csv.)
"""
import os, sys, csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _common as C

CASE, TOPO = 'anisotropy', 'disorder_hi'
NU_LO, NU_HI, PEAK, FOLD, N = -0.1, 0.5, 0.5, 4, 24
REGS = [5.0e-4, 2.0e-4]                         # push below the committed 1e-3
NITER, NREST = 420, 3


def triangle_nu(theta):
    P = np.pi / (FOLD // 2)
    ph = np.mod(theta, P) / P
    ramp = np.where(ph < PEAK, ph / PEAK, 1.0 - (ph - PEAK) / (1.0 - PEAK))
    return NU_LO + (NU_HI - NU_LO) * ramp


def load_ref():
    """reg=1e-3 achieved ν(θ) from the committed triangular_nu_4fold.csv (independent sim column)."""
    p = os.path.join(C.savedir(CASE), 'triangular_nu_4fold.csv')
    if not os.path.exists(p):
        return None
    nu = []
    for row in csv.DictReader(open(p)):
        nu.append(float(row['nu_achieved_independent']))
    return np.array(nu)


def main():
    th = C.ANG; deg = np.degrees(th); target = triangle_nu(th)
    runs = []                                   # (reg, nu_sim, kstats)
    for reg in REGS:
        prob, geo = C.make_case(TOPO, N)
        obj = C.Objective('nu_theta', target=target)
        r = C.optimize(prob, [obj], mode='k', n_iter=NITER, n_restarts=NREST, reg=reg, verbose=False)
        k = r['k'].detach().numpy()
        nu_diff, _ = C.nu_E_theta(C.solver_region_C6(prob, r['k']), th)
        C.apply_k_to_geo(geo, r['k'])
        nu_sim, _ = C.nu_E_theta(C.sim_bulk_C6(geo), th)
        err = float(np.abs(nu_sim - target).max())
        runs.append((reg, nu_sim, k))
        print(f"  reg={reg:g}: diff[{nu_diff.min():+.3f},{nu_diff.max():+.3f}] "
              f"sim[{nu_sim.min():+.3f},{nu_sim.max():+.3f}] maxerr={err:.3f} "
              f"kmed={np.median(k):.3f} n(k<0.1)={int((k < 0.1).sum())}/{len(k)}", flush=True)

    ref = load_ref()
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(deg, target, 'k--', lw=2.2, label='target (-0.1 → 0.5 triangle)')
    if ref is not None:
        ax.plot(deg, ref, color='0.5', lw=1.6, label='reg=1e-3 (committed): [%.2f, %.2f]' % (ref.min(), ref.max()))
    cols = plt.cm.viridis(np.linspace(0.2, 0.8, len(runs)))
    for (reg, nu_sim, k), col in zip(runs, cols):
        ax.plot(deg, nu_sim, color=col, lw=1.8,
                label=f'reg={reg:g}: [{nu_sim.min():+.2f},{nu_sim.max():+.2f}] kmed={np.median(k):.2f}')
    ax.axhline(0, color='0.7', lw=.5); ax.set_xlim(0, 180); ax.set_xticks(range(0, 181, 45))
    ax.set_xlabel('θ (deg)'); ax.set_ylabel('ν(θ)'); ax.grid(alpha=.3); ax.legend(fontsize=8)
    ax.set_title(f'push 4-fold triangle (N={N}) — lower reg vs amplitude/uniformity', fontsize=12)
    plt.tight_layout()
    path = os.path.join(C.savedir(CASE), 'push_4fold.png')
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f'saved {os.path.basename(path)}')


if __name__ == '__main__':
    main()
