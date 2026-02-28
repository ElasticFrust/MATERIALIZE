"""Tests for rigidity pattern samplers.

Verifies that all 8 patterns produce valid (N_tri, 3) positive arrays
across multiple topology types, including non-triangulated meshes.
"""

import sys
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "Phase 4"))

from data.topology_generators import generate_topology
from data.rigidity_patterns import (
    RIGIDITY_PATTERNS, PATTERN_WEIGHTS,
    sample_rigidity_pattern, random_pattern_name, random_pattern_params,
    sample_iid, sample_grf, sample_gradient, sample_radial,
    sample_percolation, sample_stripes, sample_virtual_distortion,
    sample_voronoi_clusters,
)


# Representative topologies: one fully triangulated, one with hard/soft edges
TEST_TOPOS = [
    ('iso_crystal', (8, 8)),    # fully triangulated, all hard
    ('honeycomb', (8, 8)),      # non-triangulated, has soft edges
    ('kagome', (8, 8)),         # non-triangulated, z=4
    ('poisson_delaunay', (8, 8)),  # random, fully triangulated
]


def _get_mesh(topo_name, size):
    """Generate a mesh for testing."""
    return generate_topology(topo_name, size, seed=42)


def test_all_patterns_shape_and_positivity():
    """All patterns produce (N_tri, 3) arrays with all-positive values."""
    print("=== Test: shape and positivity ===")
    for topo_name, size in TEST_TOPOS:
        tri = _get_mesh(topo_name, size)
        n_tri = tri.n_tri
        print(f"\n  {topo_name}: {n_tri} triangles")

        for pattern_name in RIGIDITY_PATTERNS:
            params = random_pattern_params(pattern_name)
            rigs = sample_rigidity_pattern(
                tri.points, tri.simplices,
                pattern=pattern_name, sigma=1.0, **params,
            )
            assert rigs.shape == (n_tri, 3), \
                f"{pattern_name} on {topo_name}: wrong shape {rigs.shape}"
            assert np.all(rigs > 0), \
                f"{pattern_name} on {topo_name}: has non-positive values"
            assert np.all(np.isfinite(rigs)), \
                f"{pattern_name} on {topo_name}: has non-finite values"

            kmin, kmax = rigs.min(), rigs.max()
            print(f"    {pattern_name:25s}: k in [{kmin:.4f}, {kmax:.4f}]")

    print("\n  PASSED")


def test_sigma_controls_spread():
    """Higher sigma produces wider spread of rigidity values."""
    print("\n=== Test: sigma controls spread ===")
    tri = _get_mesh('iso_crystal', (8, 8))
    np.random.seed(42)

    for pattern_name in ['iid', 'grf', 'gradient', 'radial', 'stripes']:
        spreads = []
        for sigma in [0.3, 1.0, 2.0]:
            np.random.seed(42)
            params = random_pattern_params(pattern_name)
            np.random.seed(42)
            rigs = sample_rigidity_pattern(
                tri.points, tri.simplices,
                pattern=pattern_name, sigma=sigma, **params,
            )
            log_std = np.std(np.log(rigs))
            spreads.append(log_std)

        # Spread should increase with sigma (monotonic)
        assert spreads[-1] > spreads[0], \
            f"{pattern_name}: spread not increasing with sigma ({spreads})"
        print(f"  {pattern_name:25s}: log-std = {spreads} (increasing OK)")

    print("  PASSED")


def test_grf_correlation_length():
    """Longer correlation length produces smoother fields."""
    print("\n=== Test: GRF correlation length ===")
    tri = _get_mesh('iso_crystal', (10, 10))

    roughnesses = []
    for xi in [0.5, 2.0, 8.0]:
        np.random.seed(42)
        rigs = sample_grf(tri.points, tri.simplices, sigma=1.0,
                          correlation_length=xi)
        # Roughness: mean absolute difference between neighboring triangle edges
        log_rigs = np.log(rigs)
        roughness = np.mean(np.abs(np.diff(log_rigs, axis=1)))
        roughnesses.append(roughness)
        print(f"  xi={xi:.1f}: roughness={roughness:.4f}")

    # Longer correlation length -> smoother -> lower roughness
    assert roughnesses[-1] < roughnesses[0], \
        f"Roughness not decreasing with correlation length: {roughnesses}"
    print("  PASSED")


def test_percolation_respects_p_stiff():
    """Binary percolation fraction matches p_stiff statistically."""
    print("\n=== Test: percolation p_stiff ===")
    n_tri = 1000
    for p_target in [0.3, 0.5, 0.7, 0.9]:
        np.random.seed(42)
        rigs = sample_percolation(n_tri, sigma=1.0, p_stiff=p_target,
                                  contrast=2.0)
        # Stiff edges have k > 1 (since k = exp(contrast/2) for stiff)
        frac_stiff = np.mean(rigs > 1.0)
        assert abs(frac_stiff - p_target) < 0.05, \
            f"p_stiff={p_target}: actual fraction={frac_stiff:.3f}"
        print(f"  p_stiff={p_target}: actual={frac_stiff:.3f} (OK)")

    print("  PASSED")


def test_radial_center_vs_boundary():
    """Radial pattern with sign=+1 has stiffer center."""
    print("\n=== Test: radial pattern center vs boundary ===")
    tri = _get_mesh('iso_crystal', (10, 10))

    np.random.seed(42)
    rigs = sample_radial(tri.points, tri.simplices, sigma=1.0,
                         sign=1, sharpness=5.0)

    # Compute distance of each triangle centroid from mesh center
    centroids = tri.points[tri.simplices].mean(axis=1)
    center = tri.points.mean(axis=0)
    dist = np.sqrt(np.sum((centroids - center) ** 2, axis=1))
    median_dist = np.median(dist)

    inner_k = rigs[dist < median_dist].mean()
    outer_k = rigs[dist >= median_dist].mean()

    assert inner_k > outer_k, \
        f"sign=+1: inner k ({inner_k:.3f}) should be > outer k ({outer_k:.3f})"
    print(f"  sign=+1: inner={inner_k:.3f}, outer={outer_k:.3f} (stiff core OK)")

    # Opposite sign
    np.random.seed(42)
    rigs2 = sample_radial(tri.points, tri.simplices, sigma=1.0,
                          sign=-1, sharpness=5.0)
    inner_k2 = rigs2[dist < median_dist].mean()
    outer_k2 = rigs2[dist >= median_dist].mean()

    assert inner_k2 < outer_k2, \
        f"sign=-1: inner k ({inner_k2:.3f}) should be < outer k ({outer_k2:.3f})"
    print(f"  sign=-1: inner={inner_k2:.3f}, outer={outer_k2:.3f} (soft core OK)")
    print("  PASSED")


def test_stripes_periodicity():
    """Stripe pattern has periodic structure along its direction."""
    print("\n=== Test: stripe periodicity ===")
    tri = _get_mesh('iso_crystal', (10, 10))

    np.random.seed(42)
    rigs = sample_stripes(tri.points, tri.simplices, sigma=1.0,
                          width=2.0, theta=0.0, sharpness=10.0)

    # For theta=0 (vertical bands), group edges by x-coordinate
    mids_x = tri.points[tri.simplices].mean(axis=1)[:, 0]
    log_k = np.log(rigs.mean(axis=1))

    # Check that log_k oscillates with x
    bins = np.linspace(mids_x.min(), mids_x.max(), 20)
    bin_means = []
    for i in range(len(bins) - 1):
        mask = (mids_x >= bins[i]) & (mids_x < bins[i + 1])
        if mask.sum() > 0:
            bin_means.append(log_k[mask].mean())

    # Count sign changes in bin means (should have multiple oscillations)
    diffs = np.diff(bin_means)
    sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)
    assert sign_changes >= 2, \
        f"Expected periodic oscillation, got only {sign_changes} sign changes"
    print(f"  Sign changes across bins: {sign_changes} (periodic OK)")
    print("  PASSED")


def test_voronoi_piecewise_constant():
    """Voronoi clusters produce piecewise-constant regions."""
    print("\n=== Test: Voronoi piecewise-constant ===")
    tri = _get_mesh('iso_crystal', (10, 10))

    np.random.seed(42)
    rigs = sample_voronoi_clusters(tri.points, tri.simplices, sigma=1.0,
                                   n_seeds=5)
    unique_k = len(np.unique(np.round(np.log(rigs), 6)))

    # Should have exactly n_seeds distinct values (within triangle, all 3 edges
    # may differ if they fall in different Voronoi cells, so unique_k >= n_seeds)
    assert unique_k >= 3, \
        f"Expected at least 3 distinct k values, got {unique_k}"
    assert unique_k <= 15, \
        f"Expected at most ~15 distinct k values for 5 seeds, got {unique_k}"
    print(f"  Distinct log(k) values: {unique_k} (for 5 seeds, OK)")
    print("  PASSED")


def test_pattern_weights_sum_to_one():
    """Pattern weights sum to 1."""
    print("\n=== Test: pattern weights ===")
    total = sum(PATTERN_WEIGHTS.values())
    assert abs(total - 1.0) < 1e-10, f"Weights sum to {total}, not 1.0"
    assert set(PATTERN_WEIGHTS.keys()) == set(RIGIDITY_PATTERNS.keys()), \
        "PATTERN_WEIGHTS and RIGIDITY_PATTERNS have different keys"
    print(f"  Sum = {total:.6f}, keys match. PASSED")


def test_random_pattern_name_distribution():
    """random_pattern_name() roughly follows PATTERN_WEIGHTS."""
    print("\n=== Test: random pattern sampling distribution ===")
    np.random.seed(42)
    n = 10000
    counts = {}
    for _ in range(n):
        name = random_pattern_name()
        counts[name] = counts.get(name, 0) + 1

    for name in PATTERN_WEIGHTS:
        expected = PATTERN_WEIGHTS[name]
        actual = counts.get(name, 0) / n
        assert abs(actual - expected) < 0.03, \
            f"{name}: expected ~{expected:.2f}, got {actual:.3f}"
        print(f"  {name:25s}: expected={expected:.2f}, actual={actual:.3f}")

    print("  PASSED")


def test_dispatcher_rejects_invalid():
    """sample_rigidity_pattern rejects unknown pattern names."""
    print("\n=== Test: invalid pattern rejection ===")
    tri = _get_mesh('iso_crystal', (8, 8))
    try:
        sample_rigidity_pattern(tri.points, tri.simplices, pattern='nonexistent')
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"  Correctly raised: {e}")
    print("  PASSED")


if __name__ == '__main__':
    test_all_patterns_shape_and_positivity()
    test_sigma_controls_spread()
    test_grf_correlation_length()
    test_percolation_respects_p_stiff()
    test_radial_center_vs_boundary()
    test_stripes_periodicity()
    test_voronoi_piecewise_constant()
    test_pattern_weights_sum_to_one()
    test_random_pattern_name_distribution()
    test_dispatcher_rejects_invalid()
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
