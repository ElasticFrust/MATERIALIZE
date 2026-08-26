r"""S1, the simplest possible case: can a SHALLOW GNN learn the ordered Bravais crystals at k = 1?

WHY THIS IS THE RIGHT FIRST TEST (the user's framing, 2026-08-26)
    At k = 1 on a perfect Bravais lattice, `W == 0` (measured: max|W| = 1.3e-14).  So

        C(s) = A(s) = sum_e q_e q_e^T / (16 l_e^2)

    is a CLOSED FORM in that triangle's own three edges -- no neighbour enters.  A model that cannot
    pass a single message should already be exact, which isolates the HEAD and the FEATURES from the
    graph machinery entirely.  If it fails here, nothing more complicated is worth running.

    And if `k` varies it is no longer a lattice -- the k-field breaks the translational symmetry, so
    the crystal case is k = 1 and only k = 1.

HOW MANY SAMPLES THIS REALLY IS
    ONE per crystal.  Every triangle of a Bravais crystal is equivalent by translation, and the two
    triangles of the primitive cell are inversion-related while `C` is even under inversion -- so a
    96-triangle crystal carries exactly ONE distinct value of `C(s)` (verified: 4920 triangles over
    50 crystals -> 50 distinct).  Resolution therefore has to come from the (phi, psi) GRID, not
    from bigger cells, and the model has to be small to match.

THE PARAMETRISATION, measured rather than assumed
    a1 = (1, 0),  a2 = (phi/2, psi*sqrt(3)/2), bonded along a1, a2 and ONE diagonal.
    * phi is NOT periodic at fixed diagonal: nu runs monotonically 0 -> +1/3 -> 0 -> -0.579 ->
      -1.381 over phi in [0, 4] on `a2-a1`.  (An earlier period-2 claim came from the DELAUNAY
      generator, which always takes the shorter diagonal and so silently re-folds phi into [0,1].)
    * the DIAGONAL FLAG IS REDUNDANT with phi: `(phi, a1+a2)` == `(phi+2, a2-a1)` to 0.00e+00 at
      every phi tested.  One diagonal with phi swept wide covers both.
    * psi in (0, 4] is valid throughout (A-17 clean), with nu from +16.7 at psi=0.1 to +0.25 at
      psi=4 -- almost all the variation below psi ~ 1, so psi is sampled logarithmically.

Run:  C:\Users\doron\anaconda3\python.exe "Phase 5/verifications/m2_s1_crystals.py"
Outputs -> Phase 5/results/m2_s1_crystals/
"""
import os
import sys
import time
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'Phase 3', 'verifications'))
import _common as C                                                       # noqa: E402
sys.path.insert(0, os.path.join(REPO, 'Phase 5'))
sys.path.insert(0, os.path.join(REPO, 'Phase 5', 'm2'))
sys.path.insert(0, REPO)
import matplotlib                                                         # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                           # noqa: E402
import torch                                                              # noqa: E402
import model_v2 as M                                                      # noqa: E402
import plotting as P                                                      # noqa: E402
import seeds as S                                                         # noqa: E402
from inverse_design import DesignProblem, c6_to_nuE                       # noqa: E402
from mesh_build import check_mesh_preconditions                           # noqa: E402

warnings.simplefilter('ignore')
torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)
OUT = os.path.join(REPO, 'Phase 5', 'results', 'm2_s1_crystals')

#: phi in steps of 1/8 keeps the commensurate supercell small (phi/2 = k/16 -> period <= 16).
PHIS = np.arange(0.0, 4.0001, 0.125)
#: psi logarithmic: nearly all the variation in nu is below psi ~ 1.
PSIS = np.geomspace(0.15, 4.0, 24)
REPS = 4
DIAGONAL = 'a2-a1'          # the other diagonal is the same family shifted by phi -> phi + 2


def build():
    """The (phi, psi) grid of k = 1 crystals, each labelled with its single distinct C(s)."""
    recs, t0 = [], time.time()
    for phi in PHIS:
        for psi in PSIS:
            try:
                r = S.bravais_lattice(float(phi), float(psi), reps=REPS, diagonal=DIAGONAL)
            except (ValueError, AssertionError):
                continue
            geo = r['geo']
            if not check_mesh_preconditions(geo, periodic=True)[0]:
                continue
            k = np.ones(len(geo['bond_u']))
            geo['bond_k'] = k
            geo['tri_k'] = k[geo['tri_bond']]
            prob = DesignProblem.from_geo(geo)
            with torch.no_grad():
                out = prob.forward(torch.as_tensor(k), physical_units=True)
                # `per_triangle` is ALWAYS in INTERNAL units -- `physical_units=True` rescales only
                # `elastic_tensor` / `young`, never this field. `build_dataset.py` applies the
                # factor explicitly and so must this, or the target sits in a different unit system
                # from the prediction. Omitted at first, it made the target wrong by a PER-CRYSTAL
                # factor of 18..220 (the spread of `phys` on this grid), which no scale-invariant
                # model can absorb -- the oracle check caught it at rel err 22.9 median, 122 max.
                per = (np.asarray(out['per_triangle'], float)
                       * (8.0 * len(out['per_triangle']) / float(np.asarray(geo['areas']).sum())))
                wmax = float(np.abs(np.asarray(out['W'], float)).max())
            if wmax > 1e-9 or not np.isfinite(per).all():
                continue                                   # not a crystal / unusable
            nu, E = (float(x) for x in c6_to_nuE(torch.as_tensor(per.mean(0))))
            recs.append(dict(phi=float(phi), psi=float(psi), geo=geo, k=k,
                             C6_per=per, nu=nu, E=E, w_max=wmax, n_tri=len(per)))
    print('built %d crystals in %.0f s  (grid %d x %d)' % (len(recs), time.time() - t0,
                                                           len(PHIS), len(PSIS)))
    d = np.stack([r['C6_per'][0] for r in recs])
    print('  distinct C(s) per crystal: %.2f   (1.0 expected -- translational symmetry)'
          % np.mean([len(np.unique(np.round(r['C6_per'], 10), axis=0)) for r in recs]))
    print('  nu range [%+.3f, %+.3f]   max|W| %.1e' % (min(r['nu'] for r in recs),
                                                       max(r['nu'] for r in recs),
                                                       max(r['w_max'] for r in recs)))
    return recs, d


def prep(r):
    """The model's inputs for one crystal.  Mirrors `train_v2.prepare` (angle features kept)."""
    geo = r['geo']
    g = dict(pts=geo['pts'], bond_u=np.asarray(geo['bond_u']), bond_v=np.asarray(geo['bond_v']),
             bond_R=np.asarray(geo['bond_R']), tri_bond=np.asarray(geo['tri_bond']),
             tri_verts=np.asarray(geo['simplices']), areas=np.asarray(geo['areas']),
             k=r['k'], n_nodes=len(geo['pts']))
    ev = g['bond_R'][g['tri_bond'].astype(np.int64)]
    lbar = max(float(np.hypot(g['bond_R'][:, 0], g['bond_R'][:, 1]).mean()), 1e-12)
    Q = M.edge_carriers(ev / lbar)
    t = dict(bond_u=torch.as_tensor(g['bond_u'].astype(np.int64)),
             bond_v=torch.as_tensor(g['bond_v'].astype(np.int64)),
             tri_bond=torch.as_tensor(g['tri_bond'].astype(np.int64)),
             tri_verts=torch.as_tensor(g['tri_verts'].astype(np.int64)),
             edge_feat=M.edge_features(torch.as_tensor(g['k']), g['bond_R']),
             node_feat=M.node_angle_features(g['bond_u'], g['bond_v'], g['bond_R'], g['n_nodes']),
             n_nodes=g['n_nodes'], Q=Q, tri_feat=M.triangle_angle_features(Q),
             target=torch.as_tensor(r['C6_per']))
    t['phys'] = 8.0 * len(g['tri_verts']) / torch.as_tensor(g['areas']).sum() * (lbar ** 2)
    t['kbar'] = torch.as_tensor(g['k']).mean()
    return t


def run(recs, layers, hidden, epochs, seed=0):
    """Train one (layers, hidden) model; return per-crystal validation error."""
    torch.manual_seed(seed)
    prepped = [prep(r) for r in recs]
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(prepped))
    n_val = max(1, len(order) // 5)
    va, tr = order[:n_val], order[n_val:]
    allt = torch.cat([prepped[i]['target'] for i in tr])
    sd = allt.std(0).clamp_min(1e-12)

    net = M.ForwardGNNv2(hidden=hidden, n_layers=layers)
    opt = torch.optim.Adam(net.parameters(), lr=5e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    hist = []
    for ep in range(epochs):
        net.train()
        idx = rng.permutation(tr)
        for b0 in range(0, len(idx), 32):
            ts = [prepped[i] for i in idx[b0:b0 + 32]]
            t = collate(ts)
            G = net(t)
            Cp, _ = M.assemble(t['Q'], G, physical_factor=t['scale'])
            loss = (((M.sym3_to_c6(Cp) - t['target']) / sd) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
        if ep % max(1, epochs // 8) == 0 or ep == epochs - 1:
            hist.append((ep, float(loss), evaluate(net, prepped, va, sd)))
    err = per_crystal_error(net, prepped, sd)
    return net, err, va, hist


def collate(ts):
    nb = bo = 0
    o = dict(bu=[], bv=[], tb=[], tv=[], ef=[], nf=[], tf=[], Q=[], tg=[], sc=[])
    for t in ts:
        o['bu'].append(t['bond_u'] + nb); o['bv'].append(t['bond_v'] + nb)
        o['tb'].append(t['tri_bond'] + bo); o['tv'].append(t['tri_verts'] + nb)
        o['ef'].append(t['edge_feat']); o['nf'].append(t['node_feat'])
        o['tf'].append(t['tri_feat']); o['Q'].append(t['Q']); o['tg'].append(t['target'])
        o['sc'].append(torch.full((len(t['Q']),), float(t['phys'] * t['kbar'])))
        nb += t['n_nodes']; bo += len(t['edge_feat'])
    return dict(bond_u=torch.cat(o['bu']), bond_v=torch.cat(o['bv']), tri_bond=torch.cat(o['tb']),
                tri_verts=torch.cat(o['tv']), edge_feat=torch.cat(o['ef']),
                node_feat=torch.cat(o['nf']), tri_feat=torch.cat(o['tf']), n_nodes=nb,
                Q=torch.cat(o['Q']), target=torch.cat(o['tg']),
                scale=torch.cat(o['sc']).reshape(-1, 1, 1))


def evaluate(net, prepped, idx, sd):
    net.eval()
    with torch.no_grad():
        t = collate([prepped[i] for i in idx])
        G = net(t)
        Cp, _ = M.assemble(t['Q'], G, physical_factor=t['scale'])
        return float(((M.sym3_to_c6(Cp) - t['target']).abs() / sd).mean())


def per_crystal_error(net, prepped, sd):
    net.eval()
    out = []
    with torch.no_grad():
        for t in prepped:
            G = net(t)
            Cp, _ = M.assemble(t['Q'], G, physical_factor=t['phys'] * t['kbar'])
            out.append(float(((M.sym3_to_c6(Cp) - t['target']).abs() / sd).mean()))
    return np.array(out)


def main():
    os.makedirs(OUT, exist_ok=True)
    recs, _ = build()
    phi = np.array([r['phi'] for r in recs]); psi = np.array([r['psi'] for r in recs])

    print('\n%-8s %8s %10s   %-12s %s' % ('layers', 'hidden', 'params', 'val MAE/std', 'note'))
    results = {}
    for layers, hidden in ((0, 16), (0, 32), (1, 16), (1, 32), (2, 32)):
        net, err, va, hist = run(recs, layers, hidden, epochs=400)
        n_par = sum(p.numel() for p in net.parameters())
        results[(layers, hidden)] = (err, va, hist)
        print('%-8d %8d %10d   %-12.5f %s'
              % (layers, hidden, n_par, err[va].mean(),
                 '0 layers = NO message passing' if layers == 0 else ''))

    best = min(results, key=lambda kk: results[kk][0][results[kk][1]].mean())
    err, va, hist = results[best]
    print('\nbest: layers=%d hidden=%d   val MAE/std = %.5f' % (*best, err[va].mean()))

    fig, ax = plt.subplots(1, 3, figsize=(P.STYLE.PANEL[0] * 3, P.STYLE.PANEL[1]))
    for (L, H), (e, v, h) in results.items():
        ax[0].plot([x[0] for x in h], [x[2] for x in h], 'o-', ms=3,
                   label='layers=%d hidden=%d' % (L, H))
    ax[0].set_yscale('log'); ax[0].set_xlabel('epoch'); ax[0].set_ylabel('val MAE/std')
    ax[0].set_title('learning curves'); ax[0].legend(fontsize=7); ax[0].grid(alpha=.3)

    sc = ax[1].scatter(phi, psi, c=np.log10(np.maximum(err, 1e-12)), s=26, cmap='viridis')
    ax[1].set_yscale('log'); ax[1].set_xlabel(r'$\varphi$'); ax[1].set_ylabel(r'$\psi$')
    ax[1].set_title('log$_{10}$ error over the grid\n(best model)')
    fig.colorbar(sc, ax=ax[1])

    nu = np.array([r['nu'] for r in recs])
    ax[2].scatter(nu, err, s=18, alpha=.6)
    ax[2].set_yscale('log'); ax[2].set_xlabel(r'$\nu$ of the crystal')
    ax[2].set_ylabel('MAE/std'); ax[2].set_title(r'error vs $\nu$'); ax[2].grid(alpha=.3)
    fig.suptitle('S1 simplest case: k=1 Bravais crystals, W=0, C(s)=A(s) closed form '
                 '(%d crystals)' % len(recs))
    fig.tight_layout()
    P.save_fig(fig, os.path.join(OUT, 'crystals.png'))
    np.savez_compressed(os.path.join(OUT, 'crystals.npz'), phi=phi, psi=psi, nu=nu,
                        err=err, val_idx=va)
    print('wrote', OUT)


if __name__ == '__main__':
    main()
