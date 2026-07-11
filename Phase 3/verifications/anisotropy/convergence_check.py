"""
anisotropy / convergence_check — does the LBFGS design actually converge, or stop at max_iter?

optimize() runs ONE LBFGS .step() with max_iter=n_iter (strong-Wolfe, tolerance_grad=1e-12,
tolerance_change=1e-9 default). We rerun a representative design (4-fold triangle target, N=16,
reg=1e-3) with a LARGE cap (max_iter=MAXITER) and read `opt.state['n_iter']` = the ACTUAL number of
LBFGS iterations used. If that is < MAXITER, LBFGS hit a tolerance (converged); if == MAXITER it is
iteration-limited. Also logs the loss curve and the loss at the cap the real runs use (n_iter≈360), so
we can see whether those runs are already converged or truncated. Saves convergence_check.png.
"""
import os, sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _common as C
import inverse_design as ID

CASE, TOPO, N = 'anisotropy', 'disorder_hi', 16
MAXITER, REG = 4000, 1.0e-3
NU_LO, NU_HI, PEAK, FOLD = -0.1, 0.5, 0.5, 4     # the 4-fold triangle target


def triangle_nu(theta):
    P = np.pi / (FOLD // 2)
    ph = np.mod(theta, P) / P
    ramp = np.where(ph < PEAK, ph / PEAK, 1.0 - (ph - PEAK) / (1.0 - PEAK))
    return NU_LO + (NU_HI - NU_LO) * ramp


def main():
    th = C.ANG
    target = triangle_nu(th)
    prob, geo = C.make_case(TOPO, N)
    obj = C.Objective('nu_theta', target=target)

    raw = ID._init_raw(prob, 'k', 0)
    params = list(raw.values())
    history = []
    opt = torch.optim.LBFGS(params, lr=1.0, max_iter=MAXITER,
                            line_search_fn='strong_wolfe', tolerance_grad=1e-12)

    def closure():
        opt.zero_grad()
        kb, lb = ID._params_to_kl(raw, prob, 'k')
        l = ID._loss(prob, [obj], kb, lb, REG)
        l.backward(); history.append(l.item()); return l

    opt.step(closure)

    state = opt.state[params[0]]
    n_iter_actual = int(state.get('n_iter', -1))
    # final gradient (recompute)
    opt.zero_grad()
    kb, lb = ID._params_to_kl(raw, prob, 'k')
    l = ID._loss(prob, [obj], kb, lb, REG); l.backward()
    gnorm = float(params[0].grad.abs().max())

    conv = n_iter_actual < MAXITER
    print(f"  [convergence_check] {TOPO} N={N} tri={prob.n_tri} target=4-fold-triangle reg={REG}", flush=True)
    print(f"    max_iter cap = {MAXITER}", flush=True)
    print(f"    LBFGS actual iterations = {n_iter_actual}   ({'CONVERGED (hit tolerance)' if conv else 'STOPPED AT MAX_ITER'})", flush=True)
    print(f"    function evals = {len(history)}", flush=True)
    print(f"    loss {history[0]:.4e} -> {history[-1]:.4e}   final |grad|_inf = {gnorm:.2e}", flush=True)
    # where the real runs (n_iter~360) sit: loss at eval index ~360 (evals ~ iters here for strong Wolfe)
    for cap in (100, 200, 360, 500, 1000):
        if cap < len(history):
            print(f"    loss at eval {cap:5d} = {history[cap]:.4e}   "
                  f"(gap to final = {history[cap] - history[-1]:.2e})", flush=True)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.semilogy(np.arange(len(history)), np.array(history) - min(history) + 1e-16, lw=1.2, color='#1f77b4')
    for cap in (360,):
        if cap < len(history):
            ax.axvline(cap, color='#d62728', ls='--', lw=1, label=f'real-run cap (n_iter={cap})')
    ax.set_xlabel('LBFGS function eval'); ax.set_ylabel('loss − loss_min (+1e-16)')
    ax.grid(alpha=.3, which='both'); ax.legend(fontsize=9)
    ax.set_title(f'LBFGS convergence — {n_iter_actual} iters used of {MAXITER} cap '
                 f"({'converged' if conv else 'iter-limited'})", fontsize=11)
    plt.tight_layout()
    path = os.path.join(C.savedir(CASE), 'convergence_check.png')
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f'saved {os.path.basename(path)}')


if __name__ == '__main__':
    main()
