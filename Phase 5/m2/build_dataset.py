"""Build the M2 v2 training set:  graph -> elastic tensor C6.

Implements `Phase 5/m2/M2_V2_PLAN.md` §3 (dataset) under decisions D2/D3/D4/D5/D8/D9.
REPLACES the v1 builder wholesale -- v1's labels are verified stale (37/41 drift, worst |dnu|=1.73,
audit A-19) and its sampling covered roughly one of the four `k` knobs.

WHAT IS LABELLED
----------------
The **TENSOR** (D2), not the derived curves.  nu(theta), E(theta) are ratios/reciprocals of quartics
in `C`, so a model predicting 74 numbers directly can emit profiles **no positive-definite `C` can
produce**.  nu and E are stored too, but as DERIVED diagnostics -- never as the training target.

THREADS ARE PINNED (D4)
-----------------------
BLAS thread count changes a design outcome (nu -0.150 -> -0.128, objective error 450x, same seed and
commit -- `b1_thread_local.py`).  Unpinned, the labels would carry unlabelled noise of that size.
The env vars MUST be set before torch/numpy import, which is why they are at the very top of this
file, above every other import.

THE TWO LEAKAGE TRAPS (§3.5), both handled here
-----------------------------------------------
1. TRAJECTORY leakage -- steps within one optimisation are near-duplicates, so a per-SAMPLE split
   puts near-copies on both sides.  Every sample carries `traj_id`; split on THAT.
2. INGEST leakage -- a designed network inherits the family of the seed it came from.  Unattributed,
   a held-out family walks back in through the ingest.  Every sample carries `family`.

Run:
    python "Phase 5/m2/build_dataset.py" --smoke        # a few hundred samples, minutes
    python "Phase 5/m2/build_dataset.py"                # the full build
"""
import os

# D4: pin BEFORE numpy/torch are imported anywhere. Recorded per sample as provenance.
N_THREADS = int(os.environ.get('MATERIALIZE_THREADS', '1'))
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_v] = str(N_THREADS)
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import argparse                                                          # noqa: E402
import subprocess                                                        # noqa: E402
import sys                                                               # noqa: E402
import time                                                              # noqa: E402
import warnings                                                          # noqa: E402

import numpy as np                                                       # noqa: E402
import torch                                                             # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'Phase 3', 'verifications'))
import _common as C                                                      # noqa: E402
sys.path.insert(0, os.path.join(REPO, 'Phase 5'))
sys.path.insert(0, HERE)
import fields as F                                                       # noqa: E402
import seeds as S                                                        # noqa: E402
from inverse_design import (DesignProblem, ANG, c6_to_nuE,               # noqa: E402
                            c6_to_nuE_theta)
import mesh_build as MB                                                  # noqa: E402

torch.set_default_dtype(torch.float64)
torch.set_num_threads(N_THREADS)

OUT_DIR = os.path.join(HERE, 'data')

#: (structure, marginal) pairs actually sampled. `uniform` is paired only with `iid` because the
#: marginal makes the driver irrelevant -- every other combination with it is the same field.
K_COMBOS = [('iid', 'uniform'),
            ('iid', 'lognormal'), ('iid', 'bimodal'), ('iid', 'heavy_tail'),
            ('correlated', 'lognormal'), ('correlated', 'bimodal'), ('correlated', 'heavy_tail'),
            ('gradient', 'lognormal'), ('gradient', 'bimodal'),
            ('orientation', 'lognormal'), ('orientation', 'bimodal'),
            ('sublattice', 'lognormal'), ('sublattice', 'bimodal'),
            ('length', 'lognormal')]

#: Dilution fractions. f = 0.40 puts live coordination at z ~ 3.6, BELOW the 2D isostatic point
#: z_c = 4 -- the rigidity region the v1 zoo never visited (every mesh there is a triangulation, so
#: live z = 6 exactly). Safe because `k_soft` stays inside the S0b bound; `fields.dilute` enforces it.
DILUTION_FRACS = (0.10, 0.20, 0.30, 0.40)
DILUTION_K_SOFT = 1e-6                      # 100x inside the measured 1e-8 boundary

#: The BRAVAIS sweep, over the measured fundamental domain phi in [0, 1] (a2 -> a2 + n*a1 sends
#: phi -> phi + 2 and phi -> -phi is the mirror, so nu(phi) has period 2 and mirrors about phi = 1;
#: verified to 1e-16). `psi` is the row spacing and is the knob that is genuinely open.
#: The DIAGONAL is a sampled axis, not a tie-break, and it flips the sign of nu: at psi=1 the
#: `a2-a1` branch runs 0 -> +0.333 across phi while `a1+a2` runs 0 -> -0.579, the two meeting at
#: phi = 0 where they are equivalent by symmetry. The auxetic branch is unreachable by Delaunay,
#: which always takes the shorter diagonal.
BRAVAIS_PHI = (0.0, 0.25, 0.5, 0.75, 1.0)
BRAVAIS_PSI = (0.6, 0.8, 1.0, 1.4, 1.8)
BRAVAIS_REPS = 6

#: DISORDER IS ITS OWN CATEGORY (3.1b), not a knob inside `bravais`: a perturbed crystal is a
#: different class of material, and mixing the two blurs the leave-one-family-out holdout -- the
#: plan's instruction is to hold families out at LOW disorder and treat high disorder as its own
#: regime, reporting the score against the disorder parameter. Frozen connectivity throughout (no
#: re-triangulation), which is what `bravais_lattice(eta=...)` gives and a point-cloud generator
#: cannot promise.
DISORDER_ETAS = (0.15, 0.30)
DISORDER_SEEDS = (0, 1, 2)


def _commit():
    try:
        h = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=REPO,
                           capture_output=True, text=True, timeout=10).stdout.strip()
        d = subprocess.run(['git', 'status', '--porcelain'], cwd=REPO,
                           capture_output=True, text=True, timeout=10).stdout.strip()
        return h, bool(d)
    except Exception:                                                    # noqa: BLE001
        return 'unknown', True


def solver_label(geo, k):
    """Label one (graph, k) with the solver's tensor.  Returns a dict, or None if it is unusable.

    Returns the tensor in PHYSICAL units (the convention every stored result in this project uses),
    plus derived nu/E and the diagnostics the validation protocol asks to report: SPD, `max|W|` (how
    much non-affine content the sample actually carries) and the worst triangle shape quality."""
    import positions as POS                       # lazy: positions -> designer -> seeds is a cycle
    geo = dict(geo)
    geo['bond_k'] = k
    geo['tri_k'] = k[geo['tri_bond']]
    prob = DesignProblem.from_geo(geo)
    with torch.no_grad():
        out = prob.forward(torch.as_tensor(k), physical_units=True)
        c6 = np.asarray(prob.region_tensor(out['per_triangle'], None), float)
        nu, E = (float(x) for x in c6_to_nuE(torch.as_tensor(c6)))
        nt, Et = (np.asarray(t) for t in c6_to_nuE_theta(torch.as_tensor(c6), ANG))
        wmax = float(np.abs(np.asarray(out['W'], float)).max())
    if not (np.isfinite(c6).all() and np.isfinite(nu) and np.isfinite(E)):
        return None
    Cm = np.array([[c6[0], c6[1], c6[2]], [c6[1], c6[3], c6[4]], [c6[2], c6[4], c6[5]]])
    eig = np.linalg.eigvalsh(Cm)
    return dict(C6=c6, nu=nu, E=E, nu_theta=nt, E_theta=Et, w_max=wmax,
                spd=bool(eig.min() > 0), min_eig=float(eig.min()),
                anisotropy=float(Et.max() / max(Et.min(), 1e-300)),
                min_quality=float(np.min(POS.tri_shape_quality(geo))),
                label_source='solver')


def graph_of(geo, k, is_fictional=None):
    """The stored graph.  Everything the GNN needs and nothing it does not.

    `is_fictional` is stored as PROVENANCE, and is deliberately **NOT a model input**.

    A fictional bond is an edge added only to triangulate a non-triangular face, held at ~eps so it
    carries almost no load.  But the forward map is `C = f(geometry, k)`: the solver never sees the
    label, only the stiffness, so the label adds nothing physical.  The model's edge features
    already carry `k/k_mean` and `log(k/k_mean)`, where a fictional bond reads as log ~ -7 against
    ~0 for a real rib -- it is REDUNDANT with an input the model already has.

    And feeding it in would be worse than redundant: "has fictional bonds" is almost perfectly
    correlated with FAMILY (`tiling`, `auxetic`), so it is a construction-provenance shortcut, and a
    model that latched onto it would inflate the leave-one-family-out score -- the single number the
    whole validation protocol exists to protect.

    It is stored because the BUILDER needs it (which bonds to leave soft when a k-field is drawn --
    see `respect_fictional`), the PLOTTER needs it (every edge in the compute is drawn, fictional
    ones dashed), and analysis needs it (live coordination z, stratification, re-derivation)."""
    nb = len(geo['bond_R'])
    fict = np.zeros(nb, bool) if is_fictional is None else np.asarray(is_fictional, bool)
    return dict(is_fictional=fict,
                pts=np.asarray(geo['pts'], float),
                bond_u=np.asarray(geo['bond_u'], np.int32),
                bond_v=np.asarray(geo['bond_v'], np.int32),
                bond_R=np.asarray(geo['bond_R'], float),
                tri_bond=np.asarray(geo['tri_bond'], np.int32),
                tri_verts=np.asarray(geo['simplices'], np.int32),
                areas=np.asarray(geo['areas'], float),
                k=np.asarray(k, float),
                Lx=float(geo['BL1'][0]), Ly=float(geo['BL2'][1]))


def respect_fictional(k, k0, fict):
    """Re-impose the fictional bonds' softness after a k-field has been drawn over every bond.

    THE BUG THIS FIXES (found 2026-08-25, from the rendered figure, by the user).  A fictional bond
    represents a bond that IS NOT THERE -- it exists only so the face is a valid triangle, and sits
    at k = eps so it carries no load.  Sampling a k-field over ALL bonds gives those edges real
    stiffness, which BRACES the face and turns the structure into a different material: measured on
    `_reentrant_honeycomb(v=1.15)`, nu went **-2.514 -> +0.195** and the re-entrant honeycomb became
    an ordinary triangulated mesh.  Across the family it made **42 of 45 samples labelled `auxetic`
    not auxetic** (2 % with nu < 0, against 67 % on the motif's own k0) -- not merely lost coverage
    but MISLABELLED data, which under leave-one-family-out would inflate the held-out score.

    The native:fictional RATIO is preserved (rather than a fixed eps) because only the shape of k
    matters -- `fields` normalises to mean 1, so the ratio is the invariant."""
    if fict is None or not fict.any() or fict.all():
        return k
    ratio = float(np.median(k0[fict]) / np.median(k0[~fict]))     # e.g. 1e-3
    out = np.array(k, float)
    out[fict] = ratio * float(np.mean(out[~fict]))
    return out / out.mean()


def topologies(smoke=False, n_random=24, n_nodes=120, seed=0):
    """Yield `(family, topology_id, record)` across every family, with each family swept along ITS
    OWN parameter (§3.1b) rather than a blanket disorder amplitude.

    A blanket eta is not a universal axis: on `random` it adds nothing (already disordered), on
    `tiling` the geometry IS the tiling, and on `auxetic` it DESTROYS the motif -- depopulating the
    rare region the family exists to populate."""
    if smoke:
        yield from (('cells', r['name'], r) for r in
                    S.seed_cells(n_basis_range=(3, 5, 8), n_cfg=2, aspects=(1.0,), seed=seed))
        yield ('anchor', 'anchor_triangular_N2', list(S.seed_cells_anchors())[-1])
        for phi in (0.0, 0.5, 1.0):
            for diag in S.BRAVAIS_DIAGONALS:
                r = S.bravais_lattice(phi, 1.0, reps=BRAVAIS_REPS, diagonal=diag)
                yield ('bravais', r['name'], r)
        for eta in (0.25,):
            r = S.bravais_lattice(1.0, 1.0, reps=BRAVAIS_REPS, eta=eta, seed=0)
            yield ('disordered', r['name'], r)
        for i in range(3):
            r = S.random_patch(60, seed=i, process=('uniform', 'poisson_disk', 'graded')[i])
            yield ('random', r['name'], r)
        yield ('tiling', 'tiling_kagome_r2', S.seed_tiling('kagome', 2))
        for r in S.auxetic_motifs(reps=4, thetas=(25.0,), vs=(0.85, 1.15)):
            yield ('auxetic', r['name'], r)
        return

    for r in S.seed_cells(n_basis_range=range(S.N_BASIS_MIN, 13), n_cfg=6, seed=seed):
        yield ('cells', r['name'], r)
    yield from (('anchor', r['name'], r) for r in S.seed_cells_anchors() if r.get('geo') is not None)
    for phi in BRAVAIS_PHI:                      # ORDERED crystals -- family 'bravais'
        for psi in BRAVAIS_PSI:
            for diag in S.BRAVAIS_DIAGONALS:
                try:
                    r = S.bravais_lattice(phi, psi, reps=BRAVAIS_REPS, diagonal=diag)
                except (ValueError, AssertionError):
                    continue
                yield ('bravais', r['name'], r)
    for phi in (0.0, 0.5, 1.0):                  # DISORDERED -- its own family, frozen connectivity
        for psi in (0.8, 1.0, 1.4):
            for diag in S.BRAVAIS_DIAGONALS:
                for eta in DISORDER_ETAS:
                    for sd in DISORDER_SEEDS:
                        try:
                            r = S.bravais_lattice(phi, psi, reps=BRAVAIS_REPS, diagonal=diag,
                                                  eta=eta, seed=sd)
                        except (ValueError, AssertionError):
                            continue
                        yield ('disordered', r['name'], r)
    procs = ('uniform', 'poisson_disk', 'blue_noise', 'graded')
    for i in range(n_random):
        yield ('random', f'random_{i}', S.random_patch(n_nodes, seed=seed + i,
                                                       process=procs[i % 4]))
    for tname, reps in (('square', 4), ('honeycomb', 3), ('kagome', 3), ('square_octagon', 3)):
        yield ('tiling', f'tiling_{tname}_r{reps}', S.seed_tiling(tname, reps))
    for r in (S.honeycomb(reps=3), S.kagome(reps=3)):
        yield ('basis', r['name'], r)
    for r in S.auxetic_motifs(reps=4):
        yield ('auxetic', r['name'], r)


def build(smoke=False, out=None, seed=0, n_random=24, n_nodes=120, verbose=True):
    """Sample every topology x geometry variant x k-field, label with the solver, save one npz."""
    warnings.simplefilter('ignore')
    rng_master = np.random.default_rng(seed)
    commit, dirty = _commit()
    samples, skipped = [], {}
    t0 = time.time()

    for family, topo_id, rec in topologies(smoke, n_random, n_nodes, seed):
        geo0, k0, fict = rec['geo'], rec['k0'], rec.get('is_fictional')
        if geo0 is None:
            continue
        if not MB.check_mesh_preconditions(geo0, periodic=True)[0]:
            # Never label a mesh the solver is WRONG (not merely inaccurate) on. This is the hole
            # that let an invalid mesh into `goal1`'s pool: `build_topologies` only checks areas>0
            # and never calls this gate. A dataset builder must not repeat it.
            skipped[topo_id] = 'mesh_preconditions'
            continue

        variants = [('base', geo0)]
        if family in ('cells', 'random'):
            for amp, st in ((0.06, 'correlated'), (0.12, 'correlated'), (0.10, 'white')):
                pts, gm = F.displace(geo0, np.random.default_rng(rng_master.integers(1 << 30)),
                                     amp=amp, structure=st)
                try:
                    g2 = C._periodic_delaunay(pts, *F.box_of(geo0))
                except Exception:                                        # noqa: BLE001
                    continue
                if MB.check_mesh_preconditions(g2, periodic=True)[0]:
                    variants.append((f'{st}_a{amp}', g2))

        for vname, geo in variants:
            k_specs = [('native', dict())] if fict is not None and fict.any() else []
            k_specs += [(f'{st}_{mg}', dict(structure=st, marginal=mg)) for st, mg in K_COMBOS]
            if family in ('cells', 'bravais', 'disordered', 'random'):
                k_specs += [(f'dilution_f{f}', dict(dilution=f)) for f in DILUTION_FRACS]

            for kname, spec in k_specs:
                rng = np.random.default_rng(rng_master.integers(1 << 30))
                if kname == 'native':
                    k, kmeta = k0 / k0.mean(), dict(structure='native_k0', marginal='bimodal',
                                                    contrast=float(k0.max() / k0.min()),
                                                    k_source='seed_k0')
                elif 'dilution' in spec:
                    k, kmeta = F.dilute(geo, rng, frac=spec['dilution'], k_soft=DILUTION_K_SOFT)
                else:
                    k, kmeta = F.k_field(geo, rng, **spec)
                    k = respect_fictional(k, k0, fict)
                    kmeta['contrast'] = float(k.max() / k.min())
                    kmeta['fictional_preserved'] = bool(fict is not None and fict.any())
                lab = solver_label(geo, k)
                if lab is None:
                    skipped[f'{topo_id}/{vname}/{kname}'] = 'non_finite'
                    continue
                samples.append(dict(**graph_of(geo, k, fict), **lab, **kmeta,
                                    family=family, topology_id=topo_id, geom_variant=vname,
                                    k_pattern=kname, seed=seed,
                                    traj_id=f'{topo_id}|{vname}|{kname}', traj_step=0,
                                    tiling_method='fan', n_threads=N_THREADS,
                                    commit=commit, dirty=dirty))
        if verbose and len(samples) % 200 < len(k_specs):
            print(f'  {len(samples):6d} samples  ({time.time()-t0:6.1f}s)  last: {topo_id[:40]}')

    out = out or os.path.join(OUT_DIR, 'dataset_smoke.npz' if smoke else 'dataset.npz')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    save(samples, out)
    if verbose:
        print(f'\n{len(samples)} samples -> {out}   ({time.time()-t0:.1f}s, {N_THREADS} thread(s))')
        if skipped:
            print(f'skipped {len(skipped)}: {sorted(set(skipped.values()))}')
    return samples, out


def save(samples, path):
    """Concatenate the variable-size graphs into flat arrays + offsets (the pointer scheme)."""
    if not samples:
        raise RuntimeError('no samples to save')
    d = {}
    for key in ('pts', 'bond_u', 'bond_v', 'bond_R', 'tri_bond', 'tri_verts', 'areas', 'k',
                'is_fictional'):
        d[key] = np.concatenate([np.atleast_1d(s[key]) for s in samples], axis=0)
        d[key + '_ptr'] = np.cumsum([0] + [len(np.atleast_1d(s[key])) for s in samples])
    for key in ('C6', 'nu_theta', 'E_theta'):
        d[key] = np.stack([s[key] for s in samples])
    for key in ('nu', 'E', 'Lx', 'Ly', 'w_max', 'min_eig', 'anisotropy', 'min_quality',
                'contrast', 'traj_step', 'seed', 'n_threads'):
        d[key] = np.array([s.get(key, np.nan) for s in samples], float)
    d['spd'] = np.array([s['spd'] for s in samples], bool)
    for key in ('family', 'topology_id', 'geom_variant', 'k_pattern', 'structure', 'marginal',
                'label_source', 'traj_id', 'tiling_method', 'commit', 'k_source'):
        d[key] = np.array([str(s.get(key, '')) for s in samples])
    np.savez_compressed(path, **d)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--smoke', action='store_true', help='small fast build for inspection')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--n_random', type=int, default=24)
    ap.add_argument('--n_nodes', type=int, default=120)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    build(smoke=a.smoke, out=a.out, seed=a.seed, n_random=a.n_random, n_nodes=a.n_nodes)
