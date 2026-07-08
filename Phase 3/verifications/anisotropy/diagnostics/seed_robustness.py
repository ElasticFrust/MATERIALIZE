"""
Diagnostic: does the inverse OPTIMISER fail on anisotropy targets, or is the target unreachable?

On the regular lattice, inverse-design two ν(θ) targets from several random seeds:
  (A) a REALIZABLE anisotropic profile — the ν(θ) of an actual directional k-field
      k = 1 + 0.8·cos(2·φ_bond), so it is guaranteed to be in the achievable set;
  (B) the UNREALIZABLE 'dir_aux' = 0.45·cos(2θ) target (off the ν(θ) manifold).

If the optimiser were the problem, errors would scatter across seeds. Result: (A) recovers to ~0 for
every seed (the optimiser reaches strong anisotropy when it is achievable), while (B) sticks at the
best-PD-tensor floor (~0.47) for every seed — so the 'failure' is the target's realizability, not
the optimiser or the seed. (See realizable_manifold.py for the tensor-level floor.)

Saves: seed_robustness.csv (seed, realizable_err, unrealizable_err) and seed_robustness.png.
"""
import os, sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', '..'))          # Phase 3/
sys.path.insert(0, os.path.join(HERE, '..', '..'))                # Phase 3/verifications/
import _common as C
from inverse_design import Objective, optimize, validate, c6_to_nu_theta, ANG
torch.set_default_dtype(torch.float64)

N, NSEED, NITER = 10, 6, 110


def main():
    prob, geo = C.make_case('regular', N)
    R = geo['bond_R']; ang = np.arctan2(R[:, 1], R[:, 0])
    k_true = torch.as_tensor(1.0 + 0.8 * np.cos(2 * ang))         # an actual directional k-field
    C6A = prob.region_tensor(prob.forward(k_true)['per_triangle'], None).detach()
    targetA = c6_to_nu_theta(C6A, ANG).numpy()                    # REALIZABLE anisotropic profile
    dir_aux = 0.45 * np.cos(2 * ANG)                             # UNREALIZABLE target
    print(f"realizable targetA nu span {np.ptp(targetA):.2f}; dir_aux span {np.ptp(dir_aux):.2f}", flush=True)

    nd = os.path.join(HERE, 'networks'); os.makedirs(nd, exist_ok=True)
    rows = []; eA, eB = [], []
    for seed in range(NSEED):
        rA = optimize(prob, [Objective('nu_theta', targetA)], mode='k', n_iter=NITER, reg=1e-4,
                      seed=seed, verbose=False)
        a = validate(prob, rA['k'], None, [Objective('nu_theta', targetA)])[0]['err']
        rB = optimize(prob, [Objective('nu_theta', dir_aux)], mode='k', n_iter=NITER, reg=1e-4,
                      seed=seed, verbose=False)
        b = validate(prob, rB['k'], None, [Objective('nu_theta', dir_aux)])[0]['err']
        C.save_network(os.path.join(nd, f'seed{seed}_realizable.npz'), geo, rA['k'], None,
                       target='realizable_anisotropic', seed=seed, N=N, maxerr=float(a))
        C.save_network(os.path.join(nd, f'seed{seed}_dir_aux.npz'), geo, rB['k'], None,
                       target='dir_aux_unrealizable', seed=seed, N=N, maxerr=float(b))
        eA.append(a); eB.append(b); rows.append((seed, f'{a:.4f}', f'{b:.4f}'))
        print(f"  seed {seed} | realizable maxerr={a:.4f} | dir_aux maxerr={b:.4f}", flush=True)

    C.write_csv(os.path.join(HERE, 'seed_robustness.csv'),
                ['seed', 'realizable_maxerr', 'unrealizable_dir_aux_maxerr'], rows)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    s = np.arange(NSEED)
    ax.plot(s, eA, 'o-', color='#2ca02c', ms=7, label='realizable anisotropic target')
    ax.plot(s, eB, 's-', color='#d62728', ms=7, label="unrealizable 'dir_aux' (0.45cos2θ)")
    ax.axhline(np.mean(eB), color='#d62728', ls=':', lw=1, alpha=0.6)
    ax.set_xlabel('random seed'); ax.set_ylabel('achieved ν(θ) max|error|')
    ax.set_ylim(-0.02, max(eB) * 1.15)
    ax.set_title(f'Seed robustness on the regular lattice (N={N})\n'
                 'realizable target -> ~0 every seed (optimiser is fine); '
                 'unrealizable -> stuck at the manifold floor every seed')
    ax.grid(alpha=0.3); ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(HERE, 'seed_robustness.png'), dpi=150, bbox_inches='tight')
    plt.close(); print('saved seed_robustness.{csv,png}')


if __name__ == '__main__':
    main()
