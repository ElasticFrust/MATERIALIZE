"""Build the M2 v2 training set:  graph -> elastic tensor C6.

Implements `Phase 5/m2/M2_V2_PLAN.md` §3 (dataset) under decisions D2/D3/D4/D5/D8/D9.
REPLACES the v1 builder wholesale -- v1's labels are verified stale (37/41 drift, worst |dnu|=1.73,
audit A-19) and its sampling covered roughly one of the four `k` knobs.

WHAT IS LABELLED
----------------
The **PER-TRIANGLE TENSOR `C(s)`** -- an (n_tri, 6) field, not six bulk numbers (changed 2026-08-25;
the bulk-only scheme is preserved runnable in `build_dataset_bulk_legacy.py`).

It is the tensor and not the derived curves (D2): nu(theta), E(theta) are ratios/reciprocals of
quartics in `C`, so a model predicting 74 numbers directly can emit profiles **no positive-definite
`C` can produce**.  nu and E are stored, but as DERIVED diagnostics -- never as the target.

It is PER-TRIANGLE because that is what the section 2.1 head actually produces: it predicts `C(s)`
and averages to `C_eff`.  Supervising only the average lets local errors CANCEL -- one triangle too
stiff and another too soft scores zero bulk loss, so many wrong local fields give the right mean.
Measured (`Phase 5/results/m2_locality/M2_LOCALITY.md`): ~139x the raw numbers, but the EFFECTIVE
gain is much smaller and **depends on the size mix**, so quote it per dataset rather than as a
constant.  `C(s)` decorrelates in 2-3 hops and a 2-hop ball holds ~10 triangles, so a 228-triangle
network gives ~23 independent local samples against 1 -- while a network SMALLER than one ball gives
exactly 1 however many triangles it has.  The first smoke build measured **5.95x**, not 20x, because
its median was 16 triangles and 36 % of samples fell below one ball.  See the SIZE LADDER note.

`C_eff` remains the UNWEIGHTED mean of `C(s)` (`CLAUDE.md` section 3) and is stored alongside; the
two are asserted consistent at build time so they cannot drift apart.

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
#: CORRELATION LENGTH -- (q_max, spectrum) pairs for `fields.periodic_field`.
#:
#: `periodic_field`'s own docstring says sweeping these sweeps the correlation length, "the axis
#: section 3.1f says the plan had only the two endpoints of" -- and then every call in this builder
#: used the DEFAULTS (q_max=3, spectrum=-1.0). The axis was sampled at exactly one point. A
#: long-wavelength k field on an otherwise perfect lattice is disorder just as much as a positional
#: eta is, and it is the regime that actually probes the finite-hop receptive field: locally the
#: environment looks crystalline while the global response differs.
#:
#: q_max = 1 is a single box-scale mode (longest range the periodic cell admits); spectrum = 0 with
#: a large q_max approaches white noise. Named so `k_pattern` records which was used.
CORR_SPECS = (('xi_box',   dict(q_max=1, spectrum=-1.0)),
              ('xi_long',  dict(q_max=2, spectrum=-2.0)),
              ('xi_mid',   dict(q_max=4, spectrum=-1.0)),
              ('xi_short', dict(q_max=8, spectrum=0.0)))

#: ETA grid for the DISORDERED family, now crossed with the FULL (phi, psi) bravais grid rather
#: than a coarser 3x3 subset. Disorder is not a single amplitude: 0.05 is a barely-strained crystal,
#: 0.30 is well into the auxetic band (memory: eta drives nu from +1/3 to ~-0.11).
BRAVAIS_ETAS = (0.05, 0.10, 0.18, 0.30)

#: Long-wavelength POSITIONAL modulation applied to the ORDERED bravais lattices, so the dataset
#: contains "perfect crystal + long-range strain" -- previously impossible, because `displace`
#: variants were applied to `cells` and `random` only.
BRAVAIS_DISPLACE = ((0.05, dict(q_max=1, spectrum=-1.0)),
                    (0.10, dict(q_max=2, spectrum=-2.0)))

K_COMBOS = [('iid', 'uniform'),
            ('iid', 'lognormal'), ('iid', 'bimodal'), ('iid', 'heavy_tail'),
            ('correlated', 'lognormal'), ('correlated', 'bimodal'), ('correlated', 'heavy_tail'),
            ('gradient', 'lognormal'), ('gradient', 'bimodal'),
            ('orientation', 'lognormal'), ('orientation', 'bimodal'),
            ('sublattice', 'lognormal'), ('sublattice', 'bimodal'),
            ('length', 'lognormal')]

#: `correlated` crossed with the correlation-length grid. Kept separate from K_COMBOS so the
#: baseline combination list stays comparable with the earlier datasets.
K_CORR_COMBOS = [(mg, cname, ckw) for mg in ('lognormal', 'bimodal')
                 for cname, ckw in CORR_SPECS]

#: Dilution fractions. f = 0.40 puts live coordination at z ~ 3.6, BELOW the 2D isostatic point
#: z_c = 4 -- the rigidity region the v1 zoo never visited (every mesh there is a triangulation, so
#: live z = 6 exactly). Safe because `k_soft` stays inside the S0b bound; `fields.dilute` enforces it.
DILUTION_FRACS = (0.10, 0.20, 0.30, 0.40)

#: Write a partial snapshot every this many samples (see the partial-save note in `build`).
PART_EVERY = 2000


def PART_SUFFIX(out):
    """Path of the partial snapshot for `out`.

    It must END IN `.npz`: `np.savez_compressed` appends `.npz` when the name lacks it, so writing
    to `out + '.part'` silently produced `out + '.part.npz'` while the cleanup deleted `out.part`,
    which never existed -- leaving a 526 MB orphan beside every finished build."""
    return out[:-4] + '.part.npz' if out.endswith('.npz') else out + '.part.npz'
DILUTION_K_SOFT = 1e-6                      # 100x inside the measured 1e-8 boundary

#: Relative tensor gap above which a label is marked untrusted (`sim_ok=False`).  Samples are
#: FLAGGED, never dropped: the project's survivorship rule (audit A-11/A-12) is that every run is
#: kept and the denominator stays honest -- the trainer filters, the builder records.
SIM_GAP_TOL = 0.05

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

#: SIZE LADDER -- and the reason it is weighted the way it is.
#:
#: With PER-TRIANGLE supervision the useful quantity is not the sample count but the number of
#: EFFECTIVELY INDEPENDENT local samples, and `C(s)` decorrelates in 2-3 hops with a 2-hop ball
#: holding ~10 triangles (`Phase 5/results/m2_locality/M2_LOCALITY.md`).  So a network smaller than
#: one ball yields ONE effective sample however many triangles it has, and yield grows ~ n_tri/10
#: above that.
#:
#: Measured on the first smoke build: 848 samples -> 5044 effective, a 5.95x multiplier rather than
#: the ~20x quoted for a 228-triangle network, because the median was 16 triangles and **36 % of
#: samples were smaller than a single ball**.  `cells` was 51 % of the samples and 10 % of the
#: signal; `random` was 25 % of the samples and 51 % of the signal.
#:
#: Consequence, and it is in tension with D9 read naively: MINIMAL CELLS ARE GATES, NOT TRAINING
#: DATA.  They carry the analytic ground truth (`test_m2_head.py` uses exactly that) and they cost
#: almost nothing, so they stay -- but they are sampled thinly, and the training mass goes to sizes
#: that actually carry independent local environments.
RANDOM_SIZES = (60, 120, 240, 360)     # training bulk; effective yield scales with n_tri
LARGE_SIZES = (500,)                   # HELD OUT for size generalisation -- never trained on
N_LARGE = 20
CELL_N_CFG = 2                         # was 6; cells are the analytic band, not the training mass


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
        # `out['per_triangle']` is ALWAYS in INTERNAL units -- `physical_units` rescales only
        # `elastic_tensor` / `young`, not this field (verified: identical for both settings).
        # `region_tensor` applies the physical factor 8*n_tri/sum(areas) itself. Storing the two as
        # they come would put the per-triangle target and the bulk label in DIFFERENT unit systems,
        # a factor 18.475 apart on the regular lattice -- so convert here and keep everything
        # physical, which is the convention every stored result in this project uses.
        phys = 8.0 * prob.n_tri / float(np.asarray(prob.areas).sum())
        C6_per = np.asarray(out['per_triangle'], float) * phys       # (n_tri, 6) -- THE TARGET
        c6 = np.asarray(prob.region_tensor(out['per_triangle'], None), float)
        nu, E = (float(x) for x in c6_to_nuE(torch.as_tensor(c6)))
        nt, Et = (np.asarray(t) for t in c6_to_nuE_theta(torch.as_tensor(c6), ANG))
        wmax = float(np.abs(np.asarray(out['W'], float)).max())
    if not (np.isfinite(c6).all() and np.isfinite(C6_per).all()
            and np.isfinite(nu) and np.isfinite(E)):
        return None
    Cm = np.array([[c6[0], c6[1], c6[2]], [c6[1], c6[3], c6[4]], [c6[2], c6[4], c6[5]]])
    eig = np.linalg.eigvalsh(Cm)
    # bulk C_eff is the UNWEIGHTED mean of C(s) -- asserted, so the stored bulk label can never
    # drift from the per-triangle field it is derived from (an area weight here would bias nu on
    # unequal-area meshes; CLAUDE.md section 3).
    assert np.allclose(c6, C6_per.mean(0), rtol=1e-10, atol=1e-12), \
        'bulk C6 is not the unweighted mean of the per-triangle field'
    return dict(C6=c6, C6_per=C6_per, nu=nu, E=E, nu_theta=nt, E_theta=Et, w_max=wmax,
                spd=bool(eig.min() > 0), min_eig=float(eig.min()),
                anisotropy=float(Et.max() / max(Et.min(), 1e-300)),
                min_quality=float(np.min(POS.tri_shape_quality(geo))),
                label_source='solver')


def sim_check(geo, k):
    """Cross-check one label against the INDEPENDENT sim.  Returns (ok, relative gap, status).

    Section 3.1g requires this for dilution specifically -- "bounded, health-gated, and CROSS-CHECKED
    AGAINST THE INDEPENDENT SIM FAR MORE DENSELY THAN ELSEWHERE" -- and skipping it is what let wrong
    labels into the first full build.

    WHY A PARAMETER BOUND IS NOT ENOUGH, measured 2026-08-25.  S0b established `k_soft >= 1e-8` and I
    extended it to small cells on 16/16 agreement.  Both hold in the BULK of the regime and neither
    covers its NEAR-MECHANISM TAIL: regenerating that tail gave solver -7.82 vs sim +1.07, and solver
    -94.18 vs sim +0.55 -- opposite signs.  Worse, two cases with essentially the SAME `min_eig`
    (2.29e-06 and 2.26e-06) came out one exact and one wrong by a factor of 170, so no scalar health
    metric separates them.  The only thing that does is running the other code path.

    Compares the full bulk tensor via `_common.sim_bulk_C6` -> `physical_homog.virial_C`, which is
    genuinely independent (`CLAUDE.md` section 3); NOT `sim_region_C6`, which routes the sim's
    relaxation back through the solver's own contraction and would be self-verification."""
    g = dict(geo)
    C.apply_k_to_geo(g, np.asarray(k, float))
    try:
        c6_sim = np.asarray(C.sim_bulk_C6(g), float)
    except Exception as e:                                               # noqa: BLE001
        return False, float('nan'), type(e).__name__
    prob = DesignProblem.from_geo(g)
    with torch.no_grad():
        c6_slv = np.asarray(prob.region_tensor(
            prob.forward(torch.as_tensor(np.asarray(k, float)),
                         physical_units=True)['per_triangle'], None), float)
    denom = max(float(np.abs(c6_sim).max()), 1e-30)
    gap = float(np.abs(c6_slv - c6_sim).max() / denom)
    return gap <= SIM_GAP_TOL, gap, 'checked'


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

    for r in S.seed_cells(n_basis_range=range(S.N_BASIS_MIN, 13), n_cfg=CELL_N_CFG, seed=seed):
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
    # DISORDERED: the FULL (phi, psi) grid crossed with eta, not the coarser 3x3 subset it used to
    # be. A disordered crystal is a different material from its parent, so it earns the same
    # geometric resolution as the ordered family rather than a sparser one.
    for phi in BRAVAIS_PHI:
        for psi in BRAVAIS_PSI:
            for diag in S.BRAVAIS_DIAGONALS:
                for eta in BRAVAIS_ETAS:
                    for sd in DISORDER_SEEDS[:2]:
                        try:
                            r = S.bravais_lattice(phi, psi, reps=BRAVAIS_REPS, diagonal=diag,
                                                  eta=eta, seed=sd)
                        except (ValueError, AssertionError):
                            continue
                        yield ('disordered', r['name'], r)
    # LONG-RANGE disorder on ORDERED lattices -- its own family, and a physically distinct
    # regime: locally the environment is crystalline while the global response is strained. That is
    # precisely the case a finite-hop GNN should find hardest, and until now the dataset contained
    # none of it -- `displace` variants were applied to `cells` and `random` only.
    #
    # Applied through `bravais_lattice(disp_fn=...)` rather than `fields.displace`, because the
    # latter wraps positions with `np.mod` and this mesh has FROZEN connectivity carrying integer
    # image shifts from the ideal lattice (seeds.py documents the inverted-triangle failure).
    def _corr_disp(kw):
        def fn(pts, rng, Lx, Ly):
            return np.stack([F.periodic_field(pts, Lx, Ly, rng, **kw),
                             F.periodic_field(pts, Lx, Ly, rng, **kw)], 1)
        return fn

    for phi in BRAVAIS_PHI:
        for psi in BRAVAIS_PSI:
            for diag in S.BRAVAIS_DIAGONALS:
                for amp, ckw in BRAVAIS_DISPLACE:
                    try:
                        r = S.bravais_lattice(phi, psi, reps=BRAVAIS_REPS, diagonal=diag,
                                              eta=amp, seed=seed, disp_fn=_corr_disp(ckw),
                                              disp_tag='_corrq%d' % ckw['q_max'])
                    except (ValueError, AssertionError):
                        continue
                    yield ('longrange', r['name'], r)

    procs = ('uniform', 'poisson_disk', 'blue_noise', 'graded')
    # PROCESS AND SIZE MUST NOT SHARE A PERIOD. They both used `i % 4` with len(RANDOM_SIZES) == 4,
    # which ALIASED them completely: uniform was always n=60, graded always n=360, and only 4 of the
    # 16 (process, size) combinations existed. That also silently confounded the size-generalisation
    # holdout, whose ~500-node bin contains process/size pairs that appear nowhere in training.
    # Stepping the process every len(RANDOM_SIZES) samples visits all 16.
    for i in range(n_random):                    # training bulk, cycling the size ladder
        nn = RANDOM_SIZES[i % len(RANDOM_SIZES)]
        r = S.random_patch(nn, seed=seed + i,
                           process=procs[(i // len(RANDOM_SIZES)) % len(procs)])
        r['size_bin'] = 'train'
        yield ('random', r['name'], r)
    for j in range(N_LARGE):                     # HELD-OUT large bin -- size generalisation (3.1c)
        nn = LARGE_SIZES[j % len(LARGE_SIZES)]
        r = S.random_patch(nn, seed=seed + 500 + j, process=procs[j % 4])
        r['size_bin'] = 'large_holdout'
        yield ('random_large', r['name'], r)
    for tname, reps in (('square', 4), ('honeycomb', 3), ('kagome', 3), ('square_octagon', 3)):
        yield ('tiling', f'tiling_{tname}_r{reps}', S.seed_tiling(tname, reps))
    for r in (S.honeycomb(reps=3), S.kagome(reps=3)):
        yield ('basis', r['name'], r)
    for r in S.auxetic_motifs(reps=4):
        yield ('auxetic', r['name'], r)


#: Geometry families whose POSITIONS are ordered (a perfect lattice or an exact tiling).
ORDERED_GEOM = ('bravais', 'cells', 'anchor', 'basis', 'tiling', 'auxetic')

#: k patterns that leave every live bond at the same stiffness.
UNIFORM_K = ('iid_uniform',)


def disorder_class(family, geom_variant, k_pattern):
    """Which KIND of disorder a sample carries: 'ordered', 'k', 'geom', or 'both'.

    The family name alone is not an honest label. `bravais` is commented "ORDERED crystals", yet
    every bravais topology is crossed with 14 k-fields and 4 dilution fractions -- so the great
    majority of "ordered" samples are crystals with a DISORDERED STIFFNESS FIELD, which is disorder
    in every sense that matters to the elasticity: it breaks the translational symmetry and makes
    `W` nonzero. Recording the axes separately lets the holdout and the analysis be cut by what a
    sample actually is, rather than by which generator produced it.

    Positional disorder arrives two ways -- through the family (`disordered`, `longrange`,
    `random`) and through a `geom_variant` displacement applied to an otherwise ordered mesh."""
    geom = (family not in ORDERED_GEOM) or (geom_variant != 'base')
    kdis = not (k_pattern in UNIFORM_K or k_pattern == 'native')
    if geom and kdis:
        return 'both'
    if geom:
        return 'geom'
    if kdis:
        return 'k'
    return 'ordered'


def build(smoke=False, out=None, seed=0, n_random=24, n_nodes=120, verbose=True):
    """Sample every topology x geometry variant x k-field, label with the solver, save one npz."""
    warnings.simplefilter('ignore')
    rng_master = np.random.default_rng(seed)
    commit, dirty = _commit()
    samples, skipped = [], {}
    last_part = 0
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
        if family in ('cells', 'random'):        # not 'random_large': its labels are spent on
                                                 # validation, so it gets one variant, not four
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
            # k-DISORDER AT A CONTROLLED CORRELATION LENGTH (see CORR_SPECS)
            k_specs += [(f'correlated_{mg}_{cname}',
                         dict(structure='correlated', marginal=mg, **ckw))
                        for mg, cname, ckw in K_CORR_COMBOS]
            if family in ('cells', 'bravais', 'disordered', 'longrange', 'random'):
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
                # DENSE cross-check where the solver is known to be fragile (section 3.1g)
                if kmeta.get('structure') == 'dilution':
                    ok, gap, status = sim_check(geo, k)
                    lab.update(sim_ok=bool(ok), sim_gap=float(gap), sim_status=status)
                else:
                    lab.update(sim_ok=True, sim_gap=0.0, sim_status='not_checked')
                samples.append(dict(**graph_of(geo, k, fict), **lab, **kmeta,
                                    family=family, topology_id=topo_id, geom_variant=vname,
                                    k_pattern=kname, seed=seed,
                                    disorder_class=disorder_class(family, vname, kname),
                                    traj_id=f'{topo_id}|{vname}|{kname}', traj_step=0,
                                    size_bin=rec.get('size_bin', 'train'),
                                    tiling_method='fan', n_threads=N_THREADS,
                                    commit=commit, dirty=dirty))
        if verbose and len(samples) % 200 < len(k_specs):
            print(f'  {len(samples):6d} samples  ({time.time()-t0:6.1f}s)  last: {topo_id[:40]}')
        # PARTIAL SAVE. The builder used to write only at the very end, so a power cut 2.1 h into a
        # 2.4 h build produced NOTHING. Writing a `.part` every PART_EVERY samples caps the loss at
        # that interval; the partial is a valid dataset in its own right and is removed once the
        # real file lands.
        if out and len(samples) - last_part >= PART_EVERY:
            last_part = len(samples)
            try:
                save(samples, PART_SUFFIX(out))
                if verbose:
                    print(f'    [partial saved: {len(samples)} samples]')
            except Exception as e:                                       # noqa: BLE001
                print(f'    [partial save FAILED: {type(e).__name__}: {e}]')

    out = out or os.path.join(OUT_DIR, 'dataset_smoke.npz' if smoke else 'dataset.npz')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    save(samples, out)
    if os.path.exists(PART_SUFFIX(out)):
        os.remove(PART_SUFFIX(out))              # the real file supersedes it
    if verbose:
        checked = [x for x in samples if x.get('sim_status') == 'checked']
        if checked:
            bad = [x for x in checked if not x['sim_ok']]
            print(f'\nsim cross-check (dilution): {len(checked)} checked, {len(bad)} FLAGGED '
                  f'untrusted ({100.0*len(bad)/len(checked):.2f}%)')
            if bad:
                g = sorted(x['sim_gap'] for x in bad)
                print(f'  flagged gap: median {g[len(g)//2]:.3f}  max {g[-1]:.3f}')
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
                'is_fictional', 'C6_per'):
        d[key] = np.concatenate([np.atleast_1d(s[key]) for s in samples], axis=0)
        d[key + '_ptr'] = np.cumsum([0] + [len(np.atleast_1d(s[key])) for s in samples])
    for key in ('C6', 'nu_theta', 'E_theta'):
        d[key] = np.stack([s[key] for s in samples])
    for key in ('nu', 'E', 'Lx', 'Ly', 'w_max', 'min_eig', 'anisotropy', 'min_quality',
                'contrast', 'traj_step', 'seed', 'n_threads', 'sim_gap'):
        d[key] = np.array([s.get(key, np.nan) for s in samples], float)
    d['spd'] = np.array([s['spd'] for s in samples], bool)
    d['sim_ok'] = np.array([bool(s.get('sim_ok', True)) for s in samples], bool)
    for key in ('family', 'topology_id', 'geom_variant', 'k_pattern', 'disorder_class',
                'structure', 'marginal',
                'label_source', 'traj_id', 'tiling_method', 'commit', 'k_source', 'size_bin',
                'sim_status'):
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
