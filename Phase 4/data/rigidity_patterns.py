"""Spatially structured rigidity pattern samplers for Phase 4 training data.

Each function returns per-triangle rigidities (N_tri, 3) with spatial structure.
The patterns range from spatially uncorrelated (IID) to highly structured
(gradients, bands, percolation clusters). This diversity forces the GNN to
learn general structure-property relationships and gives the CVAE a rich
distribution of patterns to learn to generate.

All patterns produce log-normal-like values (k > 0) centered around k=1.
The spatial structure comes from evaluating a 2D field at edge midpoints,
then using it to modulate the log-rigidity.

For non-triangulated meshes, the caller (generate_dataset.py) freezes soft
edges after pattern sampling — patterns here only affect hard edges.
"""

import numpy as np
from scipy.spatial import KDTree


# ─────────────────────────────────────────────────────────────────────────────
# Helper: compute edge midpoints in solver (per-triangle) format
# ─────────────────────────────────────────────────────────────────────────────

def _edge_midpoints(points, simplices):
    """Compute midpoint of each triangle edge in solver convention.

    Returns (N_tri, 3, 2) array where [t, e, :] is the midpoint of edge e
    in triangle t. Edge ordering: (v0-v1, v0-v2, v1-v2).
    """
    v0 = points[simplices[:, 0]]  # (N_tri, 2)
    v1 = points[simplices[:, 1]]
    v2 = points[simplices[:, 2]]
    mid01 = 0.5 * (v0 + v1)
    mid02 = 0.5 * (v0 + v2)
    mid12 = 0.5 * (v1 + v2)
    return np.stack([mid01, mid02, mid12], axis=1)  # (N_tri, 3, 2)


def _mesh_center_and_radius(points):
    """Compute center and effective radius of a point cloud."""
    center = points.mean(axis=0)
    r = np.sqrt(np.sum((points - center) ** 2, axis=1))
    radius = np.percentile(r, 95)  # 95th percentile avoids outlier inflation
    return center, max(radius, 1e-8)


def _field_to_rigidities(field, sigma):
    """Convert a normalized field (mean~0, std~1) to rigidities.

    k = exp(sigma * field). This gives lognormal-like values with
    geometric mean ~1 and spread controlled by sigma.
    """
    return np.exp(sigma * field)


# ─────────────────────────────────────────────────────────────────────────────
# Pattern 1: IID Random (the existing baseline)
# ─────────────────────────────────────────────────────────────────────────────

def sample_iid(n_tri, sigma=1.0):
    """IID lognormal rigidities — each edge independent.

    This is the original sampling from generate_dataset.py, kept here for
    completeness so all patterns share the same interface.
    """
    return np.exp(np.random.randn(n_tri, 3) * sigma)


# ─────────────────────────────────────────────────────────────────────────────
# Pattern 2: Gaussian Random Field (GRF) — spatially correlated noise
# ─────────────────────────────────────────────────────────────────────────────

def sample_grf(points, simplices, sigma=1.0, correlation_length=2.0):
    """Spatially correlated rigidities via a Gaussian random field.

    Generates a smooth 2D random field using random Fourier features (RFF),
    then evaluates it at each edge midpoint. The correlation length xi
    controls the spatial scale of variation:
      xi=1: varies on the scale of individual bonds (nearly IID)
      xi=4: large smooth patches of stiff/soft regions

    Method: Random Fourier Features approximation to a Gaussian kernel.
    Sample N_feat random frequencies from N(0, 1/xi^2), compute
    cos(freq . x + phase) for each, and sum them. By CLT this
    approximates a Gaussian process with squared-exponential kernel.
    """
    mids = _edge_midpoints(points, simplices)  # (N_tri, 3, 2)
    n_tri = len(simplices)

    xi = max(correlation_length, 0.1)
    n_feat = 64  # enough features for smooth fields

    # Random Fourier features: freq ~ N(0, 1/xi^2)
    freqs = np.random.randn(n_feat, 2) / xi        # (n_feat, 2)
    phases = np.random.uniform(0, 2 * np.pi, n_feat)  # (n_feat,)

    # Evaluate at all edge midpoints: mids reshaped to (N_tri*3, 2)
    flat_mids = mids.reshape(-1, 2)                 # (N_tri*3, 2)
    # proj[i, f] = flat_mids[i] . freqs[f]
    proj = flat_mids @ freqs.T                      # (N_tri*3, n_feat)
    features = np.cos(proj + phases[None, :])       # (N_tri*3, n_feat)
    field = features.mean(axis=1)                   # (N_tri*3,)

    # Normalize to zero-mean, unit-std
    field = (field - field.mean()) / max(field.std(), 1e-8)
    field = field.reshape(n_tri, 3)

    return _field_to_rigidities(field, sigma)


# ─────────────────────────────────────────────────────────────────────────────
# Pattern 3: Linear Gradient
# ─────────────────────────────────────────────────────────────────────────────

def sample_gradient(points, simplices, sigma=1.0, alpha=None, theta=None):
    """Linear gradient of rigidity across the mesh.

    k = exp(sigma * alpha * (x*cos(theta) + y*sin(theta)))
    where the projection is normalized to [-1, 1] across the mesh.

    alpha: gradient steepness (default: random in [0.3, 1.5])
    theta: gradient direction in radians (default: random uniform)
    """
    mids = _edge_midpoints(points, simplices)
    n_tri = len(simplices)

    if theta is None:
        theta = np.random.uniform(0, 2 * np.pi)
    if alpha is None:
        alpha = np.random.uniform(0.3, 1.5)

    # Project midpoints onto gradient direction
    flat_mids = mids.reshape(-1, 2)
    proj = flat_mids[:, 0] * np.cos(theta) + flat_mids[:, 1] * np.sin(theta)

    # Normalize to [-1, 1]
    pmin, pmax = proj.min(), proj.max()
    if pmax - pmin > 1e-8:
        proj = 2 * (proj - pmin) / (pmax - pmin) - 1
    else:
        proj = np.zeros_like(proj)

    field = (alpha * proj).reshape(n_tri, 3)
    return _field_to_rigidities(field, sigma)


# ─────────────────────────────────────────────────────────────────────────────
# Pattern 4: Radial Pattern (stiff core / soft shell or vice versa)
# ─────────────────────────────────────────────────────────────────────────────

def sample_radial(points, simplices, sigma=1.0, sign=None, sharpness=None):
    """Radial rigidity pattern: center vs. boundary stiffness contrast.

    sign=+1: stiff core, soft boundary  ("inclusion")
    sign=-1: soft core, stiff boundary   ("cavity")

    sharpness controls the transition width:
      sharpness=2: smooth transition across ~half the mesh radius
      sharpness=8: sharp boundary between stiff/soft regions
    """
    mids = _edge_midpoints(points, simplices)
    n_tri = len(simplices)
    center, radius = _mesh_center_and_radius(points)

    if sign is None:
        sign = np.random.choice([-1, 1])
    if sharpness is None:
        sharpness = np.random.uniform(2.0, 8.0)

    flat_mids = mids.reshape(-1, 2)
    r = np.sqrt(np.sum((flat_mids - center) ** 2, axis=1))
    r_norm = r / radius  # 0 at center, ~1 at boundary

    # tanh transition centered at r/R = 0.5
    field = sign * np.tanh(sharpness * (0.5 - r_norm))
    field = field.reshape(n_tri, 3)

    return _field_to_rigidities(field, sigma)


# ─────────────────────────────────────────────────────────────────────────────
# Pattern 5: Binary Percolation
# ─────────────────────────────────────────────────────────────────────────────

def sample_percolation(n_tri, sigma=1.0, p_stiff=None, contrast=None):
    """Binary percolation: each edge is independently stiff or soft.

    p_stiff: probability of being in the stiff phase (default: random in [0.3, 0.9])
    contrast: log ratio between stiff and soft (default: random in [1.0, 3.0])

    At p_stiff near the percolation threshold (~0.5 for triangular lattices),
    the elastic response changes dramatically — critical for learning
    rigidity percolation phenomena.
    """
    if p_stiff is None:
        p_stiff = np.random.uniform(0.3, 0.9)
    if contrast is None:
        contrast = np.random.uniform(1.0, 3.0)

    is_stiff = np.random.random((n_tri, 3)) < p_stiff
    # Stiff edges: k = exp(+contrast/2), soft edges: k = exp(-contrast/2)
    # This centers the geometric mean around 1 when p_stiff = 0.5
    field = np.where(is_stiff, contrast / 2, -contrast / 2)

    return _field_to_rigidities(field, sigma)


# ─────────────────────────────────────────────────────────────────────────────
# Pattern 6: Stripe / Band Pattern
# ─────────────────────────────────────────────────────────────────────────────

def sample_stripes(points, simplices, sigma=1.0, width=None, theta=None,
                   sharpness=None):
    """Alternating bands of stiff and soft material along a random direction.

    width: band width in mesh units (default: random in [1.0, 4.0])
    theta: band orientation angle (default: random uniform)
    sharpness: transition sharpness (default: random in [2.0, 10.0])
      low sharpness: sinusoidal, smooth transition
      high sharpness: square-wave, sharp bands
    """
    mids = _edge_midpoints(points, simplices)
    n_tri = len(simplices)

    if width is None:
        width = np.random.uniform(1.0, 4.0)
    if theta is None:
        theta = np.random.uniform(0, np.pi)  # [0, pi) suffices for bands
    if sharpness is None:
        sharpness = np.random.uniform(2.0, 10.0)

    flat_mids = mids.reshape(-1, 2)
    proj = flat_mids[:, 0] * np.cos(theta) + flat_mids[:, 1] * np.sin(theta)

    # Periodic pattern with controllable sharpness
    # sin gives smooth bands; tanh(sharpness * sin) approaches square wave
    phase = 2 * np.pi * proj / max(width, 0.1)
    field = np.tanh(sharpness * np.sin(phase))
    field = field.reshape(n_tri, 3)

    return _field_to_rigidities(field, sigma)


# ─────────────────────────────────────────────────────────────────────────────
# Pattern 7: Virtual Distortion (adapted from Phase 3)
# ─────────────────────────────────────────────────────────────────────────────

def sample_virtual_distortion(points, simplices, sigma=1.0, eta=None, a=None):
    """Rigidities from virtual distortion of the mesh.

    Clone the mesh, perturb each node by a random displacement of
    magnitude eta (uniform random angle), then assign:
      k = exp(sigma * tanh(a * (l_deformed - l_original)))

    This creates spatially correlated heterogeneity that reflects the
    local geometry: compressed edges get stiffer, stretched edges get softer.

    eta: perturbation amplitude (default: random in [0.05, 0.3])
    a:   sensitivity parameter (default: random in [5, 15])
    """
    if eta is None:
        eta = np.random.uniform(0.05, 0.3)
    if a is None:
        a = np.random.uniform(5.0, 15.0)

    # Clone and perturb node positions
    deformed = points.copy()
    thetas = 2 * np.pi * np.random.rand(len(points))
    deformed[:, 0] += eta * np.cos(thetas)
    deformed[:, 1] += eta * np.sin(thetas)

    n_tri = len(simplices)

    # Compute original and deformed edge lengths per triangle
    # Edge ordering: (v0-v1, v0-v2, v1-v2)
    edge_pairs = [(0, 1), (0, 2), (1, 2)]
    l_orig = np.zeros((n_tri, 3))
    l_def = np.zeros((n_tri, 3))

    for e, (i, j) in enumerate(edge_pairs):
        vi_orig = points[simplices[:, i]]
        vj_orig = points[simplices[:, j]]
        vi_def = deformed[simplices[:, i]]
        vj_def = deformed[simplices[:, j]]
        l_orig[:, e] = np.sqrt(np.sum((vi_orig - vj_orig) ** 2, axis=1))
        l_def[:, e] = np.sqrt(np.sum((vi_def - vj_def) ** 2, axis=1))

    field = np.tanh(a * (l_def - l_orig))
    return _field_to_rigidities(field, sigma)


# ─────────────────────────────────────────────────────────────────────────────
# Pattern 8: Voronoi Cluster Pattern
# ─────────────────────────────────────────────────────────────────────────────

def sample_voronoi_clusters(points, simplices, sigma=1.0, n_seeds=None):
    """Piecewise-constant rigidity regions defined by Voronoi cells.

    Places n_seeds random points in the mesh domain. Each seed gets a
    random log-stiffness from N(0, 1). Each edge inherits the value
    from its nearest Voronoi seed, creating distinct patches.

    Mimics polycrystalline or composite materials with discrete regions
    of different stiffness.

    n_seeds: number of Voronoi regions (default: random in {3, 5, 8, 12})
    """
    mids = _edge_midpoints(points, simplices)
    n_tri = len(simplices)

    if n_seeds is None:
        n_seeds = np.random.choice([3, 5, 8, 12])

    # Place seeds uniformly in the bounding box of the mesh
    xmin, ymin = points.min(axis=0)
    xmax, ymax = points.max(axis=0)
    seeds = np.column_stack([
        np.random.uniform(xmin, xmax, n_seeds),
        np.random.uniform(ymin, ymax, n_seeds),
    ])

    # Each seed gets a random stiffness value (in log-space)
    seed_values = np.random.randn(n_seeds)

    # Assign each edge midpoint to its nearest seed
    tree = KDTree(seeds)
    flat_mids = mids.reshape(-1, 2)
    _, nearest = tree.query(flat_mids)
    field = seed_values[nearest].reshape(n_tri, 3)

    return _field_to_rigidities(field, sigma)


# ─────────────────────────────────────────────────────────────────────────────
# Unified dispatcher
# ─────────────────────────────────────────────────────────────────────────────

# Pattern name -> (function, needs_geometry)
# needs_geometry=True means the function requires (points, simplices) args.
# needs_geometry=False means it only needs n_tri.
RIGIDITY_PATTERNS = {
    'iid':                  (sample_iid,                 False),
    'grf':                  (sample_grf,                 True),
    'gradient':             (sample_gradient,             True),
    'radial':               (sample_radial,              True),
    'percolation':          (sample_percolation,         False),
    'stripes':              (sample_stripes,             True),
    'virtual_distortion':   (sample_virtual_distortion,  True),
    'voronoi_clusters':     (sample_voronoi_clusters,    True),
}

# Sampling weights for random pattern selection during data generation.
# IID and GRF get the highest weight as the most general patterns.
PATTERN_WEIGHTS = {
    'iid':                  0.25,
    'grf':                  0.20,
    'gradient':             0.10,
    'radial':               0.10,
    'percolation':          0.10,
    'stripes':              0.10,
    'virtual_distortion':   0.10,
    'voronoi_clusters':     0.05,
}


def sample_rigidity_pattern(points, simplices, pattern='iid', sigma=1.0,
                             **kwargs):
    """Dispatch to the appropriate rigidity pattern sampler.

    Args:
        points: (M, 2) node positions.
        simplices: (N_tri, 3) triangle vertex indices.
        pattern: pattern name (key in RIGIDITY_PATTERNS).
        sigma: overall magnitude/spread of the rigidities.
        **kwargs: pattern-specific parameters (correlation_length, p_stiff, etc.)

    Returns:
        (N_tri, 3) rigidities array.
    """
    if pattern not in RIGIDITY_PATTERNS:
        raise ValueError(f"Unknown pattern '{pattern}'. "
                         f"Available: {list(RIGIDITY_PATTERNS.keys())}")

    func, needs_geom = RIGIDITY_PATTERNS[pattern]
    if needs_geom:
        return func(points, simplices, sigma=sigma, **kwargs)
    else:
        n_tri = len(simplices)
        return func(n_tri, sigma=sigma, **kwargs)


def random_pattern_name():
    """Sample a random pattern name according to PATTERN_WEIGHTS."""
    names = list(PATTERN_WEIGHTS.keys())
    weights = np.array([PATTERN_WEIGHTS[n] for n in names])
    weights = weights / weights.sum()
    return np.random.choice(names, p=weights)


def random_pattern_params(pattern):
    """Sample random hyperparameters for a given pattern.

    Returns a dict of keyword arguments to pass to the pattern sampler.
    Each pattern has its own parameter distributions chosen to give good
    coverage of the pattern space.
    """
    if pattern == 'iid':
        return {}
    elif pattern == 'grf':
        return {'correlation_length': np.random.choice([1.0, 2.0, 4.0])}
    elif pattern == 'gradient':
        return {
            'alpha': np.random.uniform(0.3, 1.5),
            'theta': np.random.uniform(0, 2 * np.pi),
        }
    elif pattern == 'radial':
        return {
            'sign': np.random.choice([-1, 1]),
            'sharpness': np.random.uniform(2.0, 8.0),
        }
    elif pattern == 'percolation':
        return {
            'p_stiff': np.random.uniform(0.3, 0.9),
            'contrast': np.random.uniform(1.0, 3.0),
        }
    elif pattern == 'stripes':
        return {
            'width': np.random.uniform(1.0, 4.0),
            'theta': np.random.uniform(0, np.pi),
            'sharpness': np.random.uniform(2.0, 10.0),
        }
    elif pattern == 'virtual_distortion':
        return {
            'eta': np.random.uniform(0.05, 0.3),
            'a': np.random.uniform(5.0, 15.0),
        }
    elif pattern == 'voronoi_clusters':
        return {'n_seeds': int(np.random.choice([3, 5, 8, 12]))}
    else:
        return {}
