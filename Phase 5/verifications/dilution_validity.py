r"""S0b — where does BOND DILUTION break the solver? (M2_V2_PLAN.md D8)

WHY
---
Dilution — driving a fraction `f` of bonds to a low stiffness `k_soft` — is the richest axis the M2
dataset is missing: it sweeps coordination `z` from 6 (triangular) down through the 2D isostatic
point `z_c = 4`, i.e. rigidity percolation, where `E → 0` and `ν` swings hard.

It is ALSO the axis most likely to produce labels **the solver itself gets wrong**. `A(s) =
Σ_e (k_e/4ℓ_e²) q_e q_eᵀ` loses rank when bonds go dead, and the solver's regularised inverse then
returns an answer set by the REGULARISER, not by physics (`CLAUDE.md` §3 — measured on a regular
lattice driven to ν = −0.2: 11 % of bonds dead gave solver ν = −0.110 vs sim ν = **+0.136**,
OPPOSITE SIGNS).

And softness alone does NOT predict failure: the hexagon gate runs `k_spoke = 1e-8` and still matches
the closed form to 4.4e-06, because there the soft edge is a **spoke** that lets the face hinge. What
matters is the soft bond's structural role, which we cannot read off `k`.

So this maps the boundary EMPIRICALLY, before any dataset is built on top of it.

WHAT IT MEASURES
----------------
For each (base network, dilution fraction `f`, softness `k_soft`, seed): the SOLVER's ν,E and the
INDEPENDENT SIM's ν,E on the same network, plus diagnostics that might predict the divergence
(fraction of triangles whose `A(s)` is near rank-deficient, min shape quality, live coordination z).

Output is a validity boundary: the largest `f` at which solver and sim still agree within tolerance.

Run:  C:\Users\doron\anaconda3\python.exe "Phase 5/verifications/dilution_validity.py" [--quick]
Out:  Phase 5/results/dilution_validity/{dilution.csv, dilution_nu.png, dilution_gap.png}
"""
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'Phase 3', 'verifications'))
import _common as C                                              # wires the solver stack
sys.path.insert(0, REPO)
import plotting as P                                             # noqa: E402
import torch                                                     # noqa: E402
import physical_homog as PH                                      # noqa: E402
import sim_assembly as SA                                        # noqa: E402
from inverse_design import DesignProblem                         # noqa: E402

OUT = os.path.join(REPO, 'Phase 5', 'results', 'dilution_validity')

# Bases: the regular lattice (where the dead-k failure was measured) and a disordered one, since
# the failure mode is about STRUCTURAL ROLE and disorder changes which bonds carry shear.
BASES = [('regular', 1.0, 1.0, 0.0, 0), ('eta0.25', 1.0, 1.0, 0.25, 3)]
# Reach the ACTUALLY DEAD regime. The documented failure had bonds driven to ~1e-40 by an
# OPTIMISER, not to 1e-3 -- a first pass over 1e-1..1e-5 found perfect solver/sim agreement even
# at f = 0.40, i.e. moderate softness is benign and the boundary is much further down.
K_SOFT = [1e-2, 1e-4, 1e-6, 1e-8, 1e-12, 1e-20, 1e-40]
FRACS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40]
SEEDS = [0, 1, 2]
HALF = 5.0                     # keep the SIM affordable (dense LAPACK)


def rank_deficient_fraction(prob, k, tol=1e-8):
    """Fraction of triangles whose bare tensor A(s) is near rank-deficient — the documented route to
    a wrong read-back. Computed from the solver's own per-triangle A."""
    with torch.no_grad():
        out = prob.solver.forward(torch.as_tensor(k)[prob.tri_bond], rest_lengths=prob.rl_ref,
                                  method='intrinsic', physical_units=True)
    bare = out['bare'].numpy()                                   # (N,5) packed
    # rebuild A(s) as 3x3 from the packed bare tensor is version-specific; use the q-form instead
    return float('nan'), out


def build_diluted(base, f, k_soft, seed):
    """The diluted network for one case. Shared by the sweep and the figure script so the two
    cannot drift apart. Returns (geo, k, is_diluted mask)."""
    name, phi, psi, eta, gseed = base
    geo = C.make_lattice(phi, psi, half=HALF, eta=eta, seed=gseed)
    nb = len(geo['bond_u'])
    rng = np.random.default_rng(1000 * seed + int(1e4 * f) + int(-np.log10(k_soft)))
    k = np.ones(nb)
    dil = np.zeros(nb, bool)
    n_dil = int(round(f * nb))
    if n_dil:
        idx = rng.choice(nb, n_dil, replace=False)
        k[idx] = k_soft
        dil[idx] = True
    geo['bond_k'] = k
    geo['tri_k'] = k[geo['tri_bond']]
    return geo, k, dil


def one_case(base, f, k_soft, seed):
    """Solver vs INDEPENDENT SIM on one diluted network. Returns a row dict or None if unhealthy."""
    name = base[0]
    geo, k, _dil = build_diluted(base, f, k_soft, seed)
    nb = len(geo['bond_u'])
    n_dil = int(round(f * nb))

    row = dict(base=name, f=f, k_soft=k_soft, seed=seed, n_bonds=nb, n_diluted=n_dil)
    # live coordination z = 2 * (# bonds above the soft population) / n_nodes  -> Maxwell axis
    live = nb - n_dil                      # exact: the un-diluted bonds
    row['z_live'] = 2.0 * live / len(geo['pts'])

    prob = DesignProblem.from_geo(geo)
    with torch.no_grad():
        cs = C.solver_region_C6(prob, torch.as_tensor(k))
    nu_s, E_s = C.c6_nuE(cs)
    row['nu_solver'], row['E_solver'] = float(nu_s), float(E_s)

    try:                                        # the sim self-screens near-singular geometry
        free = np.arange(2, 2 * len(geo['pts']))
        nu_p, E_p = PH.virial_nuE(geo, PH.relax(geo, free, SA.assemble_K_faff))
        row['nu_sim'], row['E_sim'] = float(nu_p), float(E_p)
        row['ok'] = 1
    except Exception as e:                                       # noqa: BLE001
        row['nu_sim'], row['E_sim'], row['ok'] = float('nan'), float('nan'), 0
        row['err'] = f'{type(e).__name__}'
        return row

    # the project's per-design gap metric (CLAUDE.md §3), on the scalar pair
    row['d_nu'] = abs(row['nu_solver'] - row['nu_sim'])
    row['gap'] = (row['d_nu'] / (abs(row['nu_sim']) + 0.05)
                  + abs(row['E_solver'] - row['E_sim']) / max(abs(row['E_sim']), 1e-12))
    row['sign_flip'] = int(np.sign(row['nu_solver']) != np.sign(row['nu_sim']))
    return row


def main():
    quick = '--quick' in sys.argv
    fracs = FRACS[::2] if quick else FRACS
    ksoft = K_SOFT[::2] if quick else K_SOFT
    seeds = SEEDS[:1] if quick else SEEDS
    os.makedirs(OUT, exist_ok=True)

    rows = []
    for base in BASES:
        for ks in ksoft:
            for f in fracs:
                for sd in (seeds if f > 0 else seeds[:1]):
                    r = one_case(base, f, ks, sd)
                    rows.append(r)
                    flag = ''
                    if r['ok'] and r.get('sign_flip'):
                        flag = '   <<< SIGN FLIP'
                    elif not r['ok']:
                        flag = f"   (sim rejected: {r.get('err')})"
                    print(f"  {r['base']:8} k_soft={ks:.0e} f={f:.2f} s{sd} "
                          f"z={r['z_live']:.2f} nu_solver={r['nu_solver']:+.4f} "
                          f"nu_sim={r['nu_sim']:+.4f} gap={r.get('gap', float('nan')):.2e}{flag}",
                          flush=True)

    cols = sorted({c for r in rows for c in r})
    with open(os.path.join(OUT, 'dilution.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader(); w.writerows(rows)

    # ---- figures: solver vs sim SHARE a panel (CLAUDE.md §3) --------------------------------
    for base, _, _, _, _ in BASES:
        pass
    panels_nu, panels_gap = {}, {}
    for base in {r['base'] for r in rows}:
        for ks in ksoft:
            sel = [r for r in rows if r['base'] == base and r['k_soft'] == ks and r['ok']]
            if not sel:
                continue
            xs = sorted({r['f'] for r in sel})
            sv = np.array([[np.mean([r['nu_solver'] for r in sel if r['f'] == x]) for x in xs]])
            sm = np.array([[np.mean([r['nu_sim'] for r in sel if r['f'] == x]) for x in xs]])
            gp = np.array([[np.mean([r['gap'] for r in sel if r['f'] == x]) for x in xs]])
            key = f'{base}, k_soft={ks:.0e}'
            panels_nu[key] = {'solver': (np.array(xs), sv), 'independent sim': (np.array(xs), sm)}
            panels_gap[key] = {'gap': (np.array(xs), gp)}

    fig = P.plot_overlay_grid(panels_nu, xlabel='dilution fraction f', ylabel='ν',
                              suptitle='S0b — dilution: SOLVER vs INDEPENDENT SIM\n'
                                       'divergence marks where dilution labels become untrustworthy',
                              ncols=4, hline0=True)
    P.save_fig(fig, os.path.join(OUT, 'dilution_nu.png'))
    fig = P.plot_overlay_grid(panels_gap, xlabel='dilution fraction f', ylabel='gap',
                              suptitle='S0b — per-design gap vs dilution (flag at gap > 0.05)',
                              ncols=4)
    P.save_fig(fig, os.path.join(OUT, 'dilution_gap.png'))

    ok = [r for r in rows if r['ok']]
    print(f'\n{len(ok)}/{len(rows)} cases the sim accepted')
    print(f'sign flips: {sum(r["sign_flip"] for r in ok)}')
    for ks in ksoft:
        good = [r['f'] for r in ok if r['k_soft'] == ks and r['gap'] <= 0.05]
        print(f'  k_soft={ks:.0e}: gap<=0.05 up to f = {max(good) if good else float("nan")}')
    print(f'\nwrote {OUT}')


if __name__ == '__main__':
    main()
