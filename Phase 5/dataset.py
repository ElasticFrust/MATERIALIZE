r"""Phase 5 — the labelled dataset generator (training data for the future M2 GNN).

Per PLAN.md §5, we accumulate `(topology + k -> nu(theta), E(theta))` samples two ways:

  1. FORWARD scan (cheap, guarantees realisability): take many topologies from the seed zoo, put a
     few stiffness patterns on each (uniform / random), and record the SOLVER's directional
     response nu(theta),E(theta).  No optimisation, no sim — the solver's forward pass IS the label
     the M2 net will learn (it amortises the differentiable solver).

  2. DESIGN scan: for a grid of (nu, E) targets, run the inverse designer, and record each kept,
     INDEPENDENTLY-VERIFIED design (these carry a sim-confirmed response + target_err_sim).

Each sample's geometry is saved as an .npz (via _common.save_network) under Phase 5/dataset/nets/,
and a consolidated index Phase 5/dataset/dataset.npz stores the stacked response arrays + metadata
(so M2 can load geometry + label together).  We also build the minimal (nu, E) archive PLAN §5
mentions: bin samples on a coarse (mean-nu, mean-E) grid, keep the best per cell.

Run:  "C:\Users\doron\anaconda3\python.exe" "Phase 5\dataset.py"
"""
# ---- §0 preamble (dataset.py lives directly in Phase 5/) --------------------------------------
import os, sys, json
import numpy as np
import torch
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(REPO, 'Phase 3', 'verifications'))
import _common as C
from inverse_design import DesignProblem, ANG, c6_to_nuE_theta, c6_to_nuE
torch.set_default_dtype(torch.float64)

import seeds
import designer

DATA_DIR = os.path.join(os.path.dirname(__file__), 'dataset')
NET_DIR = os.path.join(DATA_DIR, 'nets')


# ---- solver response (the label) -------------------------------------------------------------
def solver_response(geo, k):
    """SOLVER directional response for a (geo, k): returns (nu_theta(37,), E_theta(37,), C6(6,))."""
    prob = DesignProblem.from_geo(geo)
    out = prob.forward(torch.as_tensor(np.asarray(k, float)))
    C6 = prob.region_tensor(out['per_triangle'], None)
    nu_th, E_th = (t.detach().numpy() for t in c6_to_nuE_theta(C6, ANG))
    return nu_th, E_th, C6.detach().numpy()


def _k_patterns(nbond, k0, seed):
    """A few stiffness patterns to sample per topology: the seed's own k0 (native/soft for
    tilings), uniform 1, and a couple of random log-normal fields (structured disorder in k)."""
    rng = np.random.default_rng(seed)
    yield 'k0', np.asarray(k0, float)
    yield 'uniform', np.ones(nbond)
    for i in range(2):
        yield f'lognormal{i}', np.exp(rng.normal(0.0, 0.6, nbond))


# ---- 1. forward scan -------------------------------------------------------------------------
def forward_scan(n_random=8, n_nodes=90, seed=0):
    """Yield forward samples: seed-zoo topologies × a few k patterns, labelled by the solver."""
    pool = list(seeds.seed_pool(n_random=n_random, n_nodes=n_nodes,
                                include=('bravais', 'random', 'tiling', 'basis', 'auxetic')))
    for j, rec in enumerate(pool):
        geo, nbond = rec['geo'], len(rec['geo']['bond_R'])
        for kname, k in _k_patterns(nbond, rec['k0'], seed + j):
            try:
                nu_th, E_th, C6 = solver_response(geo, k)
            except Exception as e:                     # skip degenerate/unstable evaluations
                print(f"    [skip forward] {rec['name']}/{kname}: {e}")
                continue
            if not (np.all(np.isfinite(nu_th)) and np.all(np.isfinite(E_th))):
                continue
            yield dict(source='forward', name=f"{rec['name']}::{kname}", geo=geo, k=k,
                       nu_theta=nu_th, E_theta=E_th, C6=C6,
                       is_fictional=rec['is_fictional'], target_err_sim=np.nan)


# ---- 2. design scan --------------------------------------------------------------------------
def design_scan(nu_grid=(-0.3, 0.0, 0.3), E_grid=(1.0,), keep=3, **design_kw):
    """Yield design samples: for each (nu,E) target, the inverse designer's kept verified designs."""
    for nu in nu_grid:
        for E in E_grid:
            tag = f"nu{nu:+.2f}_E{E:.2f}".replace('+', 'p').replace('-', 'm').replace('.', '')
            reports = designer.design(nu, E, tag=f'ds_{tag}', keep=keep, **design_kw)
            for r in reports:
                geo, k, _, _ = None, None, None, None
                # reload the saved design (report has the path) to recover geo+k
                g, kb, C6_per, meta = C.load_network(r['path'])
                nu_th, E_th, C6 = solver_response(g, kb)
                yield dict(source='design', name=f"{r['seed_name']}::{tag}", geo=g, k=kb,
                           nu_theta=nu_th, E_theta=E_th, C6=C6,
                           is_fictional=np.asarray(meta.get('is_fictional',
                                                            np.zeros(len(g['bond_R']), bool)), bool),
                           target_err_sim=r['target_err_sim'])


# ---- 3. persist ------------------------------------------------------------------------------
def save_dataset(samples):
    """Save each sample's geometry as an .npz and a consolidated index (stacked responses)."""
    os.makedirs(NET_DIR, exist_ok=True)
    nu_all, E_all, C6_all, names, sources, terr, netfiles = [], [], [], [], [], [], []
    for i, s in enumerate(samples):
        net = os.path.join(NET_DIR, f'sample_{i:05d}.npz')
        C.apply_k_to_geo(s['geo'], s['k'])
        C.save_network(net, s['geo'], s['k'], C6_per=None,
                       name=s['name'], source=s['source'],
                       is_fictional=np.asarray(s['is_fictional'], bool).tolist())
        nu_all.append(s['nu_theta']); E_all.append(s['E_theta']); C6_all.append(s['C6'])
        names.append(s['name']); sources.append(s['source']); terr.append(s['target_err_sim'])
        netfiles.append(os.path.basename(net))
    idx = os.path.join(DATA_DIR, 'dataset.npz')
    np.savez_compressed(idx,
                        nu_theta=np.array(nu_all), E_theta=np.array(E_all), C6=np.array(C6_all),
                        thetas=ANG, names=np.array(names), source=np.array(sources),
                        target_err_sim=np.array(terr, float), netfile=np.array(netfiles),
                        meta=json.dumps(dict(n=len(names), n_theta=len(ANG))))
    return idx, len(names)


# ---- 4. minimal (nu, E) archive (PLAN §5) ----------------------------------------------------
def nu_E_archive(samples, nu_bins=np.linspace(-0.6, 0.6, 13), E_bins=np.linspace(0.0, 2.5, 11)):
    """Bin samples on a coarse (mean-nu, mean-E) grid; keep the 'best' per cell (design samples
    ranked by target_err_sim; forward samples by |mean-nu| spread as a mild novelty proxy)."""
    cells = {}
    for s in samples:
        mnu, mE = float(np.mean(s['nu_theta'])), float(np.mean(s['E_theta']))
        bi, bj = int(np.digitize(mnu, nu_bins)), int(np.digitize(mE, E_bins))
        score = s['target_err_sim'] if np.isfinite(s['target_err_sim']) else 1.0
        cur = cells.get((bi, bj))
        if cur is None or score < cur[0]:
            cells[(bi, bj)] = (score, s['name'], mnu, mE)
    return cells


# ---- main ------------------------------------------------------------------------------------
def main(do_design=True):
    os.makedirs(DATA_DIR, exist_ok=True)
    print("=" * 84)
    print("BUILDING M2 DATASET")
    print("=" * 84)

    print("\n[1/3] forward scan (seed zoo x k patterns, solver-labelled)...")
    samples = list(forward_scan(n_random=8, n_nodes=90, seed=0))
    print(f"      -> {len(samples)} forward samples")

    if do_design:
        print("\n[2/3] design scan (a few (nu,E) targets, verified designs)...")
        dsamples = list(design_scan(nu_grid=(-0.3, 0.0, 0.3), E_grid=(1.0,), keep=3,
                                    n_iter=70, n_restarts=1, reg=0.02))
        print(f"      -> {len(dsamples)} design samples")
        samples += dsamples

    print("\n[3/3] saving + (nu,E) archive...")
    idx, n = save_dataset(samples)
    cells = nu_E_archive(samples)
    print(f"      saved {n} samples -> {idx}")
    print(f"      geometry .npz     -> {NET_DIR}\\sample_*.npz")
    print(f"      (nu,E) archive: {len(cells)} occupied cells (coverage of response space)")

    # coverage scatter (mean nu vs mean E of every sample)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        mnu = [float(np.mean(s['nu_theta'])) for s in samples]
        mE = [float(np.mean(s['E_theta'])) for s in samples]
        col = ['tab:blue' if s['source'] == 'forward' else 'tab:red' for s in samples]
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(mnu, mE, c=col, s=18, alpha=0.7)
        ax.set_xlabel('mean nu'); ax.set_ylabel('mean E')
        ax.set_title('dataset coverage (blue=forward, red=design)')
        ax.axvline(0, color='k', lw=0.5); ax.grid(alpha=0.3)
        out = os.path.join(DATA_DIR, 'coverage.png')
        fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)
        print(f"      coverage plot     -> {out}")
    except Exception as e:
        print(f"      (coverage plot skipped: {e})")

    assert n >= 20, "expected a reasonable number of dataset samples"
    print("\nDATASET BUILD DONE")


if __name__ == '__main__':
    main()
