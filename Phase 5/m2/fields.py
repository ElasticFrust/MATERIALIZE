"""Sampling of the `k` field and the geometry field for the M2 v2 dataset.

Implements `Phase 5/m2/M2_V2_PLAN.md` §3.1f (disorder has a CORRELATION LENGTH), §3.1g (widen the
space by PHYSICS, not by more randomness) and §3.1h (the four `k` knobs), under the bound decision
D8 measured by S0b (`Phase 5/results/dilution_validity/DILUTION_VALIDITY.md`).

WHY THIS MODULE EXISTS
----------------------
`A(s)` is LINEAR in `k`, but `W` depends on `k` through `A^-1`.  So the `k` DISTRIBUTION is precisely
what exercises the part of the forward map the GNN actually has to learn: at uniform `k`, `W` is a
pure function of geometry, and it is contrast in `k` that makes it non-trivial.  The v1 dataset
varied roughly one of the four knobs below (an i.i.d. lognormal), which is why it is replaced rather
than extended.

THE FOUR KNOBS (§3.1h)
----------------------
    marginal shape      uniform / lognormal / bimodal / heavy-tailed
    correlation length  iid (xi=0) / correlated field / gradient (xi~L) / uniform (xi=inf)
    structure           keyed to the TOPOLOGY: orientation, sublattice, length, region
    contrast            k_max/k_min, bounded -- see CONTRAST_MAX

TWO THINGS THAT FALL OUT OF THE PHYSICS, AND ARE USED HERE
----------------------------------------------------------
1. ONLY THE SHAPE MATTERS.  Scaling every `k` by lambda sends `A -> lambda A`, `A^-1 -> A^-1/lambda`,
   so `W` is UNCHANGED and `C -> lambda C`.  nu is therefore invariant and only `E` moves.  Every
   field here is returned normalised to **mean(k) = 1**: one dimension fewer, for free, and the
   overall scale is restored (if wanted) by a single multiply.

2. THE SAFE DILUTION BOUND IS A *CONTRAST* BOUND.  S0b measured the boundary at `k_soft >= 1e-8`
   with the stiff bonds at 1.  Its root cause is the solver's regulariser `eps = 1e-12 * max|A3|`,
   which is RELATIVE to the global maximum -- so it scales with the stiff population, and the
   invariant quantity is the RATIO, not the absolute value.  That is what makes knob 1 and knob 4
   independent: normalising to mean 1 cannot move a design across the validity boundary.

A note on periodicity: spatial fields use wave vectors COMMENSURATE with the box,
`q = 2*pi*(m/Lx, n/Ly)` with integer m, n.  An incommensurate field would be discontinuous across the
periodic seam, so the network and the field it carries would have different periods -- the sample
would not be the crystal it claims to be.
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'Phase 2'))
import mesh_build as MB                                                   # noqa: E402

#: S0b (`DILUTION_VALIDITY.md`): `k_soft >= 1e-8` is safe at every dilution fraction tested up to
#: f = 0.40, including sub-isostatic z = 3.6; at `k_soft <= 1e-12` the solver returns nu of the
#: OPPOSITE SIGN and no geometric health gate can detect it.  Expressed as a contrast because the
#: solver's regulariser is relative (see the module docstring).  ~4 orders of margin on the measured
#: crossover at 2e-12.
CONTRAST_MAX = 1e8

K_MARGINALS = ('uniform', 'lognormal', 'bimodal', 'heavy_tail')
K_STRUCTURES = ('iid', 'correlated', 'gradient', 'orientation', 'sublattice', 'length', 'dilution')


# ---- geometry helpers (per-bond quantities the structured fields key on) -----------------------
def bond_midpoints(geo):
    """Per-bond midpoint, minimum-image correct.

    `bond_R` is the true periodic bond vector (it already carries the wrap), so the midpoint of a
    seam-crossing bond is `pts[u] + R/2` and NOT the average of the two stored endpoints -- the
    latter would land in the middle of the box for a bond that wraps around it."""
    pts = np.asarray(geo['pts'], float)
    u = np.asarray(geo['bond_u'], int)
    R = np.asarray(geo['bond_R'], float)
    return pts[u] + 0.5 * R


def bond_lengths(geo):
    R = np.asarray(geo['bond_R'], float)
    return np.hypot(R[:, 0], R[:, 1])


def bond_angles(geo):
    """Bond orientation in [0, pi) -- undirected, so theta and theta+pi are the same bond."""
    R = np.asarray(geo['bond_R'], float)
    return np.mod(np.arctan2(R[:, 1], R[:, 0]), np.pi)


def box_of(geo):
    return float(geo['BL1'][0]), float(geo['BL2'][1])


# ---- the correlated scalar field (the xi knob, shared by k and geometry) -----------------------
def periodic_field(xy, Lx, Ly, rng, n_modes=6, q_max=3, spectrum=-1.0):
    """Zero-mean scalar field  f(x) = sum_q A_q sin(q.x + phi_q)  on the periodic box.

    Wave vectors are drawn from the COMMENSURATE set `q = 2*pi*(m/Lx, n/Ly)`, |m|,|n| <= `q_max`,
    so the field has the box's own period.  `spectrum` sets the amplitude law `A_q ~ |q|^spectrum`:
    a negative exponent weights LONG wavelengths, giving a smooth, long-correlation field; a large
    `q_max` with `spectrum = 0` approaches white noise.  Sweeping these sweeps the correlation
    length, which is the axis §3.1f says the plan had only the two endpoints of.

    Returns the field sampled at `xy`, normalised to unit standard deviation so the caller controls
    the amplitude."""
    modes = [(m, n) for m in range(-q_max, q_max + 1) for n in range(-q_max, q_max + 1)
             if (m, n) != (0, 0)]
    pick = rng.choice(len(modes), size=min(n_modes, len(modes)), replace=False)
    out = np.zeros(len(xy))
    for idx in pick:
        m, n = modes[idx]
        q = 2.0 * np.pi * np.array([m / Lx, n / Ly])
        amp = np.linalg.norm(q) ** spectrum
        out += amp * np.sin(xy @ q + rng.uniform(0, 2 * np.pi))
    sd = out.std()
    return out / sd if sd > 0 else out


# ---- the k field ------------------------------------------------------------------------------
def _apply_marginal(z, marginal, rng, sigma=0.8, frac=0.4, ratio=10.0, tail=2.5):
    """Turn a standard-normal-ish driver `z` into a positive k with the requested MARGINAL shape.

    Keeping the driver separate from the marginal is what makes the two knobs independent: the same
    spatially-correlated `z` can be pushed through any marginal, so correlation length and
    distribution shape are not confounded."""
    if marginal == 'uniform':
        return np.ones_like(z)
    if marginal == 'lognormal':
        return np.exp(sigma * z)
    if marginal == 'bimodal':
        hi = z >= np.quantile(z, frac)
        return np.where(hi, ratio, 1.0).astype(float)
    if marginal == 'heavy_tail':
        # Designed k is measured HEAVY-TAILED -- max/median 17 on a healthy design and 147 on a
        # degenerate one (CLAUDE.md §3, the B-4 colour-scale finding). Training on lognormal and
        # deploying inside the design loop would be textbook distribution shift, so this is sampled
        # deliberately rather than hoped for. Pareto via the inverse CDF of the driver's rank.
        u = (np.argsort(np.argsort(z)) + 0.5) / len(z)
        return (1.0 - u) ** (-1.0 / tail)
    raise ValueError(f'unknown marginal {marginal!r}; expected one of {K_MARGINALS}')


def k_field(geo, rng, structure='iid', marginal='lognormal', contrast=None, **kw):
    """Per-bond stiffness `k`, normalised to mean 1.  Returns `(k, meta)`.

    `structure` selects the DRIVER (what the field is a function of) and `marginal` the DISTRIBUTION
    it is pushed through -- the two are deliberately independent (see `_apply_marginal`):

        iid          driver = white noise                     (xi = 0)
        correlated   driver = `periodic_field`                (xi swept by `spectrum`, `q_max`)
        gradient     driver = a single long-wavelength ramp   (xi ~ L)
        orientation  driver = cos(2*theta_bond)   -> ANISOTROPIC C, controllably, rather than by
                                                     waiting for anisotropy to appear by chance
        sublattice   driver = the bond's orientation CLASS (3 classes) -- breaks symmetry in a
                                                     DESIGNED way; pairs with the basis enumeration
        length       driver = log(bond length)     -> k ~ l^alpha, the fibre-network family
        dilution     see `dilute` -- the large-contrast corner of `bimodal`, bounded

    `contrast`, if given, rescales the field so `max(k)/min(k)` equals it, capped at CONTRAST_MAX.
    """
    nb = len(geo['bond_R'])
    Lx, Ly = box_of(geo)
    if structure == 'iid':
        z = rng.standard_normal(nb)
    elif structure == 'correlated':
        z = periodic_field(bond_midpoints(geo), Lx, Ly, rng,
                           n_modes=kw.get('n_modes', 6), q_max=kw.get('q_max', 3),
                           spectrum=kw.get('spectrum', -1.0))
    elif structure == 'gradient':
        xy = bond_midpoints(geo)
        z = np.sin(2.0 * np.pi * xy[:, kw.get('axis', 0)] / (Lx if kw.get('axis', 0) == 0 else Ly))
    elif structure == 'orientation':
        z = np.cos(2.0 * (bond_angles(geo) - kw.get('theta0', 0.0)))
    elif structure == 'sublattice':
        cls = np.floor(bond_angles(geo) / (np.pi / 3.0)).astype(int) % 3
        z = np.asarray(rng.standard_normal(3))[cls]
    elif structure == 'length':
        L = bond_lengths(geo)
        z = np.log(L / L.mean())
    else:
        raise ValueError(f'unknown structure {structure!r}; expected one of {K_STRUCTURES}')

    if structure != 'iid' and z.std() > 0:
        z = (z - z.mean()) / z.std()
    k = _apply_marginal(z, marginal, rng, **{a: kw[a] for a in ('sigma', 'frac', 'ratio', 'tail')
                                             if a in kw})
    k = np.maximum(np.asarray(k, float), 1e-300)
    if contrast is not None:
        k = _force_contrast(k, contrast)
    k = k / k.mean()                       # only the SHAPE matters (module docstring, point 1)
    return k, dict(structure=structure, marginal=marginal,
                   contrast=float(k.max() / k.min()), k_source='k_field')


def _force_contrast(k, contrast):
    """Rescale a positive field so max/min equals `contrast` (capped at CONTRAST_MAX).

    Done in log space so the ORDERING and the relative spacing of the field are preserved -- a
    clip would instead pile mass at the two ends and change the marginal we just chose."""
    contrast = min(float(contrast), CONTRAST_MAX)
    lo, hi = np.log(k.min()), np.log(k.max())
    if hi - lo < 1e-12:
        return k
    return np.exp((np.log(k) - lo) / (hi - lo) * np.log(contrast))


def dilute(geo, rng, frac=0.2, k_soft=1e-6, k_stiff=1.0):
    """BOND DILUTION -- a fraction `frac` of bonds set soft.  Returns `(k, meta)`, mean-normalised.

    Traverses coordination `z` toward and through the 2D isostatic point `z_c = 4`, which is the
    rigidity axis the v1 zoo never visited (every mesh there is a triangulation, so live z = 6
    exactly, with the tilings the only exception).

    NOT a separate axis from §3.1h: it is the LARGE-CONTRAST CORNER of the bimodal family, which is
    why it is bounded by the same `CONTRAST_MAX`.  Refuses a contrast beyond the S0b boundary rather
    than silently producing a label the solver gets WRONG -- 6 of 266 cases there returned nu of the
    opposite sign (worst: solver -0.189 vs sim +0.986), and §3 of that document shows the broken
    networks are structurally INDISTINGUISHABLE from the safe ones, so nothing downstream could
    catch it."""
    if k_stiff / k_soft > CONTRAST_MAX:
        raise ValueError(
            f'dilution contrast {k_stiff / k_soft:.1e} exceeds CONTRAST_MAX={CONTRAST_MAX:.0e}; '
            f'S0b measured the solver returning the WRONG SIGN of nu below that, undetectably. '
            f'Label this region with the sim, or do not sample it.')
    nb = len(geo['bond_R'])
    n_soft = int(round(frac * nb))
    k = np.full(nb, float(k_stiff))
    if n_soft:
        k[rng.choice(nb, size=n_soft, replace=False)] = float(k_soft)
    z_live = 2.0 * (nb - n_soft) / len(geo['pts'])
    k = k / k.mean()
    # Report the contrast of the field ACTUALLY produced, not the requested ratio: at frac=0 no bond
    # is soft and the field is uniform, so recording the nominal k_stiff/k_soft would stamp the
    # sample with a contrast it does not have. Provenance is read back as data downstream.
    return k, dict(structure='dilution', marginal='bimodal', frac=float(frac),
                   k_soft=float(k_soft), contrast=float(k.max() / k.min()),
                   contrast_requested=float(k_stiff / k_soft),
                   z_live=float(z_live), k_source='dilute')


# ---- the geometry field -----------------------------------------------------------------------
def node_min_altitude(geo):
    """Per node, the smallest ALTITUDE over its incident triangles — the local safety margin.

    The altitude from a vertex to its opposite edge is exactly the distance it may travel before the
    triangle FLATTENS and then inverts. It is NOT the edge length: a triangle can collapse into a
    sliver with every edge still at full length, which is why normalising a displacement by edge
    length (or by the mesh's mean bond length, as `displace` does) does not protect the mesh.

    Uses `mesh_build.edge_vec_orientation` so the edge vectors are the PERIODIC ones with the
    triangle's own corner order — `pts[j] - pts[i]` would be wrong across the wrap."""
    ei, sg = MB.edge_vec_orientation(geo['tri_bond'], geo['simplices'],
                                     geo['bond_u'], geo['bond_v'])
    bR = np.asarray(geo['bond_R'], float)
    sg = np.where(sg == 0.0, 1.0, sg)                      # self-loop: sign undefined, magnitude ok
    ev = bR[ei] * sg[..., None]                            # (n_tri, 3, 2): (0,1), (0,2), (1,2)
    area = 0.5 * np.abs(ev[:, 0, 0] * ev[:, 1, 1] - ev[:, 0, 1] * ev[:, 1, 0])
    L = np.linalg.norm(ev, axis=2)                         # |01|, |02|, |12|
    # altitude from corner c is 2*Area / |opposite edge|: corner0 -> |12|, 1 -> |02|, 2 -> |01|
    alt = 2.0 * area[:, None] / np.maximum(L[:, [2, 1, 0]], 1e-300)
    sm = np.asarray(geo['simplices'], np.int64)
    out = np.full(len(geo['pts']), np.inf)
    np.minimum.at(out, sm.ravel(), alt.ravel())
    return out


def first_inversion_scale(geo, d):
    """The EXACT scale `t*` at which displacing by `t*d` first flattens a triangle. Closed form.

    Along a fixed direction field `d`, each triangle's edge vectors are AFFINE in `t`, so twice its
    signed area is exactly QUADRATIC:

        2A_s(t) = cross(e1 + t*de1, e2 + t*de2) = c0 + c1*t + c2*t^2

    with `de_j` the difference of the two endpoint displacements. `t*` is the smallest positive root
    over all triangles — the true distance to the first inversion, orientation and all. No bound, no
    conservatism: a per-node altitude bound has to assume the worst relative orientation, which on an
    equilateral triangle costs a factor ~1.7 against the real limit.

    Vertex COLLISION needs no separate treatment: in a triangulation, two bonded nodes cannot meet
    without first flattening every triangle on that edge, so area-positivity already covers it.

    Periodicity is handled by construction — `de_j` is a difference of plain displacement vectors, so
    the constant image shift inside each edge vector cancels and nothing has to be wrapped."""
    ei, sg = MB.edge_vec_orientation(geo['tri_bond'], geo['simplices'],
                                     geo['bond_u'], geo['bond_v'])
    sg = np.where(sg == 0.0, 1.0, sg)
    ev = np.asarray(geo['bond_R'], float)[ei] * sg[..., None]       # (n_tri,3,2): (0,1),(0,2),(1,2)
    sm = np.asarray(geo['simplices'], np.int64)
    d = np.asarray(d, float)
    de1 = d[sm[:, 1]] - d[sm[:, 0]]                                  # change of edge (0,1)
    de2 = d[sm[:, 2]] - d[sm[:, 0]]                                  # change of edge (0,2)
    cr = lambda a, b: a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]          # noqa: E731
    c0, c2 = cr(ev[:, 0], ev[:, 1]), cr(de1, de2)
    c1 = cr(ev[:, 0], de2) + cr(de1, ev[:, 1])
    s = np.sign(c0)                                                  # keep each triangle's own sign
    a, b, c = s * c2, s * c1, s * c0                                 # want a t^2 + b t + c > 0
    # `c > 0` at t = 0 by construction (each triangle starts un-inverted), so a triangle inverts at
    # the smallest positive t where the parabola CROSSES zero. It crosses only if the discriminant
    # is positive -- with disc < 0 and the leading sign positive the triangle can never invert at
    # any scale. Clamping disc to 0 and taking the extremum as a root, which an earlier version did,
    # invents a crossing that does not exist and makes t* too SMALL: the gate caught it as
    # `1.001*t*` failing to invert anything.
    t = np.inf
    lin = np.abs(a) < 1e-300
    with np.errstate(invalid='ignore', divide='ignore'):
        tl = np.where(lin & (b < 0), -c / b, np.inf)
        disc = b * b - 4 * a * c
        rt = np.sqrt(np.where(disc > 0, disc, 0.0))
        real = (~lin) & (disc > 0)
        r1 = np.where(real, (-b - rt) / (2 * a), np.inf)
        r2 = np.where(real, (-b + rt) / (2 * a), np.inf)
    for r in (tl, r1, r2):
        r = np.where(np.isfinite(r) & (r > 0), r, np.inf)
        t = min(t, float(r.min()) if len(r) else np.inf)
    return t


def displace_safe(geo, rng, frac=0.5, structure='correlated', local_scale=True, **kw):
    """Perturb node positions so that NO TRIANGLE CAN INVERT. Returns `(pts, meta)`.

    `frac` is the fraction of the distance to the FIRST INVERSION, computed exactly by
    `first_inversion_scale` for this particular direction field. So `frac` is a dimensionless
    "how far toward breaking the mesh", `frac -> 1` approaches the true geometric limit, and
    `frac < 1` cannot invert anything. That is the user's "direction normalized fraction", made
    exact rather than bounded.

    An earlier version capped each node at `h_v / 3` (a third of its smallest incident altitude).
    That is a correct SUFFICIENT bound — vertex motion `d`, opposite-edge translation `d`, edge
    rotation `d` — but it must assume the worst relative orientation, and on an equilateral triangle
    it stops at 0.289 where vertices do not actually collide until 0.5. Solving the quadratic
    removes that factor ~1.7 and, more importantly, adapts to whatever the direction field happens
    to be.

    WHY A GUARANTEE RATHER THAN A FILTER. §3.1f prescribed "perturb, then accept/reject on shape
    quality, and report the acceptance rate". That works, but leaves a silently BIASED subset the
    moment the rate goes unreported — the same trap as an eta-sweep that does not say how many seeds
    survived. Scaling to a guaranteed-safe amplitude removes the rejection step, so nothing is
    discarded and nothing can be biased by discarding it.

    It also fixes the two defects `displace` has on FROZEN-CONNECTIVITY meshes:
      * it scales by `amp * mean(bond_length)`, a GLOBAL mean, so a node in a locally fine region is
        displaced by something comparable to its own neighbourhood and collapses it;
      * it ends with `np.mod(pts + d, box)`, and wrapping is WRONG here: triangles carry integer
        image shifts, so a wrapped node reconstructs its triangle a box away (`seeds.py`: 13 of 72
        triangles inverted, signed area −15.2, at eta = 0.05). This does not wrap.

    `meta['geom_eta_equiv']` reports the largest node displacement in units of the mean bond length,
    so a sweep can be quoted on the familiar eta scale (eta < 0.5 is the classical bound, and it is
    a COLLISION bound — which area-positivity subsumes)."""
    if not 0.0 <= frac < 1.0:
        raise ValueError('frac must be in [0, 1): it is a fraction of the distance to inversion')
    pts = np.asarray(geo['pts'], float).copy()
    Lx, Ly = box_of(geo)
    if structure == 'correlated':
        ux = periodic_field(pts, Lx, Ly, rng, **kw)
        uy = periodic_field(pts, Lx, Ly, rng, **kw)
        d = np.stack([ux, uy], 1)
    elif structure == 'white':
        a = rng.uniform(0, 2 * np.pi, len(pts))
        d = np.stack([np.cos(a), np.sin(a)], 1)
    else:
        raise ValueError(f'unknown structure {structure!r}; expected correlated / white')
    d = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-300)   # unit per node
    if local_scale:
        # SHAPE the field by each node's own margin before scaling it globally.
        #
        # Every triangle constrains all three of its vertices JOINTLY, so a per-vertex cap computed
        # independently is NOT safe: A, B and C can each be within their own limit and still flip
        # ABC by moving together. The guarantee therefore has to come from one global scale.
        #
        # But with a UNIT field that scale is set by the single worst triangle in the mesh, and every
        # other node is throttled to match it -- on a heterogeneous mesh the coarse regions then
        # barely move. Making the field proportional to each node's own smallest incident altitude
        # fixes that: fine regions ask for less, coarse regions ask for more, and the exact global
        # scale is no longer hostage to one fine spot. Each vertex is still limited by its tightest
        # incident triangle -- that is what `node_min_altitude` is -- it simply is not limited by
        # somebody else's.
        h = node_min_altitude(geo)
        d = d * (h / max(float(np.median(h)), 1e-300))[:, None]
    t_star = first_inversion_scale(geo, d)
    step = frac * t_star
    out = pts + step * d
    lbar = float(bond_lengths(geo).mean())
    moved = np.linalg.norm(out - pts, axis=1)
    return out, dict(geom_structure=structure, geom_frac=float(frac),
                     geom_bound='exact_first_inversion',
                     geom_local_scale=bool(local_scale),
                     geom_t_star=float(t_star),
                     geom_eta_equiv=float(moved.max() / lbar),
                     geom_eta_mean=float(moved.mean() / lbar),
                     # how much of its OWN budget the typical node actually used: the number that
                     # says whether the mesh moved uniformly or was throttled by its worst spot
                     geom_budget_used=float(np.median(moved / np.maximum(h if local_scale
                                                                         else node_min_altitude(geo),
                                                                         1e-300))))


def displace(geo, rng, amp=0.1, structure='correlated', **kw):
    """Perturbed node positions -- returns `(pts, meta)`.  Connectivity is NOT re-triangulated.

    `amp` is in units of the mean bond length, so it is comparable across cells of different size.
    That matters: §3.1f records that the familiar `eta <= 0.42` bound is a REGULAR-LATTICE
    measurement which does not transfer, because eta is an ABSOLUTE displacement and a mesh that
    already has short edges can collapse a triangle at a much smaller one.  The caller must still
    gate on SHAPE QUALITY and the health gate, and report the acceptance rate -- amplitude alone is
    not a safety criterion.

    `structure='correlated'` gives the long-wavelength field of §3.1f: locally a slightly-strained
    crystal, globally structured.  That is the regime that probes the receptive-field question --
    a finite-hop GNN sees a locally-crystalline environment while the global response differs.
    `structure='white'` recovers the zero-correlation-length corner."""
    pts = np.asarray(geo['pts'], float).copy()
    Lx, Ly = box_of(geo)
    scale = amp * float(bond_lengths(geo).mean())
    if structure == 'correlated':
        ux = periodic_field(pts, Lx, Ly, rng, **kw)
        uy = periodic_field(pts, Lx, Ly, rng, **kw)
        d = scale * np.stack([ux, uy], 1)
    elif structure == 'white':
        a = rng.uniform(0, 2 * np.pi, len(pts))
        d = scale * np.stack([np.cos(a), np.sin(a)], 1)
    else:
        raise ValueError(f'unknown structure {structure!r}; expected correlated / white')
    out = np.mod(pts + d, np.array([Lx, Ly]))
    return out, dict(geom_structure=structure, geom_amp=float(amp))
