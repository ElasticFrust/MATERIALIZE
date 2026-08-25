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
import numpy as np

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
