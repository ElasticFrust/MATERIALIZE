"""
Case dir_aux_ribbon, STEP 1 — find a REALIZABLE ν(θ) target with a very strong (|ν|>1) auxetic
response for loading along x (θ=0): pull along x, the material should expand in y MORE than it
contracts in x. Not every ν(θ) curve corresponds to an actual elastic tensor (see
anisotropy/diagnostics/seed_robustness.py) -- so before designing anything we check achievability
the same way that diagnostic does: build the target, run a real optimize(), and see whether the
achieved value actually reaches it (realizable) or plateaus short (unrealizable).

Two ways to target ν(0):
  FULL profile  ν(θ) = ν0 - A·cos(2θ) over the whole ANG grid -- ties ν(0)=ν0-A to an equally
                extreme (and, it turns out, unrealizable) ν(π/2)=ν0+A on the OTHER axis, so the
                achieved max-error over the whole curve grows with A even though ν(0) itself is
                still hit accurately.
  NARROW window ν(θ)=ν0 only for θ in a small band around 0 -- only constrains the one loading
                direction this experiment actually needs; everything else is left free for the
                optimiser to choose, so it should converge cleanly to a stronger ν(0).

Both are checked, on the ordered anisotropic base (aniso_str, η=0) and the slightly-disordered
version (η=0.08) that three_ribbon.py will use for its two topologies. Saves
dir_aux_profile_scan.csv/.png. The NARROW-window ν(0)=-1.3 result is the target three_ribbon.py
actually designs to.
"""
import os, sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import _common as C
from inverse_design import Objective, optimize, validate, ANG
torch.set_default_dtype(torch.float64)

N, NITER, REG = 10, 150, 1e-4
NU0 = -0.30                                     # background level of the full cos(2*theta) family
TOPO = 'aniso_str'
FULL_AMPS = [0.6, 0.9, 1.2, 1.5, 1.8]
NARROW_TARGETS = [-0.8, -1.0, -1.3, -1.6, -1.9]
NARROW_THETAS = np.linspace(0.0, 0.20, 5)       # a ~11.5 deg window around theta=0 (x-axis)


def make_prob(eta, seed=0):
    phi, psi, _ = C._TOPO_PARAMS[TOPO]
    geo = C.make_lattice(phi, psi, half=N, eta=eta, seed=seed)
    return C.DesignProblem.from_geo(geo), geo


def run_full(eta, A):
    prob, geo = make_prob(eta)
    target = NU0 - A * np.cos(2 * ANG)          # theta=0 -> NU0-A (auxetic); theta=90 -> NU0+A
    r = optimize(prob, [Objective('nu_theta', target)], mode='k', n_iter=NITER, reg=REG, verbose=False)
    rep = validate(prob, r['k'], None, [Objective('nu_theta', target)])[0]
    return float(rep['achieved'][0]), float(target[0]), float(rep['err'])


def run_narrow(eta, nu0_target):
    prob, geo = make_prob(eta)
    r = optimize(prob, [Objective('nu_theta', nu0_target, thetas=NARROW_THETAS)],
                 mode='k', n_iter=NITER, reg=REG, verbose=False)
    rep = validate(prob, r['k'], None, [Objective('nu_theta', nu0_target, thetas=NARROW_THETAS)])[0]
    return float(rep['achieved'][0]), float(rep['err'])


def main():
    rows = []
    results = {}
    for eta, elabel in [(0.0, 'ordered'), (0.08, 'disordered')]:
        print(f"=== {elabel} {TOPO} (eta={eta}) ===", flush=True)
        full = []
        for A in FULL_AMPS:
            ach, tgt, err = run_full(eta, A)
            full.append((A, tgt, ach, err))
            rows.append(('full_cos2theta', elabel, A, tgt, ach, err))
            print(f"  FULL   A={A:.2f}  target nu(0)={tgt:+.3f}  achieved nu(0)={ach:+.3f}  maxerr={err:.4f}",
                  flush=True)
        narrow = []
        for nu0_t in NARROW_TARGETS:
            ach, err = run_narrow(eta, nu0_t)
            narrow.append((nu0_t, ach, err))
            rows.append(('narrow_window', elabel, np.nan, nu0_t, ach, err))
            print(f"  NARROW target nu(0)={nu0_t:+.3f}  achieved nu(0)={ach:+.3f}  err={err:.4f}", flush=True)
        results[elabel] = dict(full=full, narrow=narrow)

    C.write_csv(os.path.join(HERE, 'dir_aux_profile_scan.csv'),
                ['family', 'topology', 'amplitude_A', 'target_nu0', 'achieved_nu0', 'maxerr'], rows)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    for ax, elabel in zip(axes, ['ordered', 'disordered']):
        full = results[elabel]['full']; narrow = results[elabel]['narrow']
        tf = [r[1] for r in full]; af = [r[2] for r in full]
        tn = [r[0] for r in narrow]; an = [r[1] for r in narrow]
        ax.plot([-2.2, 0.5], [-2.2, 0.5], 'k--', lw=1, alpha=0.4, label='ideal (achieved=target)')
        ax.plot(tf, af, 'o-', color='#d62728', label='full cos(2θ) profile (fights θ=90° too)')
        ax.plot(tn, an, 's-', color='#2ca02c', label='narrow window at θ≈0 only')
        ax.axhline(-1.0, color='gray', ls=':', lw=1, label='|ν|=1 threshold')
        ax.set_xlabel('target ν(0)'); ax.set_ylabel('achieved ν(0)')
        ax.set_title(f'{TOPO}, {elabel}'); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle('dir_aux_ribbon STEP 1 — realizability of a strong auxetic ν(θ=0): narrow-window '
                 'targeting reaches |ν|>1 cleanly; the full symmetric profile does not', fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(os.path.join(HERE, 'dir_aux_profile_scan.png'), dpi=150, bbox_inches='tight')
    plt.close(); print('saved dir_aux_profile_scan.png/.csv')


if __name__ == '__main__':
    main()
