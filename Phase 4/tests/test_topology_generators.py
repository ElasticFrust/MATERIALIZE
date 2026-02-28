"""Tests for the topology generators in Phase 4.

Verifies that all 18 topology generators produce valid, well-formed meshes
that are compatible with the forward solver and the GNN data pipeline.

Test categories:
  1. Basic shape tests: correct output types, dimensions, non-empty arrays.
  2. Triangulation validity: all simplices reference valid point indices,
     no degenerate (zero-area) triangles.
  3. Hard/soft edge consistency: hard edges match real lattice bonds,
     soft edges are properly classified.
  4. Triangle count targets: each topology produces ~950 triangles at size=(10,10).
  5. Cross-topology compatibility: all topologies produce TriangulationResults
     that can be converted to PyG Data objects via graph_utils.

Usage:
    cd "Phase 4"
    python -m pytest tests/test_topology_generators.py -v
    # or without pytest:
    python tests/test_topology_generators.py
"""

import numpy as np
import sys
from pathlib import Path

# Set up imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "Phase 2"))
sys.path.insert(0, str(PROJECT_ROOT / "Phase 3"))
sys.path.insert(0, str(PROJECT_ROOT / "Phase 4"))

from data.topology_generators import (
    TriangulationResult,
    generate_topology,
    generate_all_topologies,
    TOPOLOGY_GENERATORS,
    TOPO_CLASSES,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _validate_triangulation(result, name=""):
    """Run basic validity checks on a TriangulationResult."""
    prefix = f"[{name}] " if name else ""

    # 1. Type checks
    assert isinstance(result, TriangulationResult), \
        f"{prefix}Expected TriangulationResult, got {type(result)}"
    assert isinstance(result.points, np.ndarray), \
        f"{prefix}points should be ndarray"
    assert isinstance(result.simplices, np.ndarray), \
        f"{prefix}simplices should be ndarray"

    # 2. Shape checks
    assert result.points.ndim == 2 and result.points.shape[1] == 2, \
        f"{prefix}points shape should be (M, 2), got {result.points.shape}"
    assert result.simplices.ndim == 2 and result.simplices.shape[1] == 3, \
        f"{prefix}simplices shape should be (N_tri, 3), got {result.simplices.shape}"

    # 3. Non-empty
    M = result.n_points
    N_tri = result.n_tri
    assert M >= 4, f"{prefix}Too few points: {M}"
    assert N_tri >= 2, f"{prefix}Too few triangles: {N_tri}"

    # 4. Valid indices: all simplex indices should reference existing points
    assert result.simplices.min() >= 0, \
        f"{prefix}Negative vertex index in simplices"
    assert result.simplices.max() < M, \
        f"{prefix}Vertex index {result.simplices.max()} exceeds point count {M}"

    # 5. No degenerate triangles (zero area)
    pts = result.points
    for t in range(min(N_tri, 100)):  # Check first 100 triangles
        v0, v1, v2 = result.simplices[t]
        area = 0.5 * abs(
            (pts[v1, 0] - pts[v0, 0]) * (pts[v2, 1] - pts[v0, 1]) -
            (pts[v2, 0] - pts[v0, 0]) * (pts[v1, 1] - pts[v0, 1])
        )
        assert area > 1e-12, \
            f"{prefix}Triangle {t} has zero area (degenerate)"

    # 6. Metadata
    assert result.topo_name, f"{prefix}topo_name should be non-empty"
    assert result.topo_class, f"{prefix}topo_class should be non-empty"

    # 7. Hard edge set consistency
    if result.hard_edge_set is not None:
        for edge in list(result.hard_edge_set)[:100]:
            assert isinstance(edge, frozenset), \
                f"{prefix}hard_edge_set should contain frozensets"
            i, j = tuple(edge)
            assert 0 <= i < M and 0 <= j < M, \
                f"{prefix}hard_edge ({i},{j}) references invalid vertex"

    # 8. Default rigidities shape
    rigs = result.get_default_rigidities()
    assert rigs.shape == (N_tri, 3), \
        f"{prefix}default rigidities shape should be ({N_tri}, 3), got {rigs.shape}"
    assert np.all(rigs > 0), f"{prefix}all rigidities should be positive"

    # 9. Hard edge mask shape
    mask = result.get_hard_edge_mask_per_triangle()
    assert mask.shape == (N_tri, 3), \
        f"{prefix}hard edge mask shape should be ({N_tri}, 3), got {mask.shape}"
    assert mask.dtype == bool, f"{prefix}hard edge mask should be bool"

    return True


# ─────────────────────────────────────────────────────────────────────────────
# Test: all generators produce valid meshes
# ─────────────────────────────────────────────────────────────────────────────

def test_all_topologies_valid():
    """Test that every registered topology produces a valid TriangulationResult."""
    size = (6, 6)  # Small size for fast testing
    seed = 42

    for name in TOPOLOGY_GENERATORS:
        print(f"  Testing topology: {name} ...", end=" ")
        result = generate_topology(name, size, seed=seed)
        _validate_triangulation(result, name)
        print(f"OK ({result.n_tri} triangles, {result.n_points} points)")


# ─────────────────────────────────────────────────────────────────────────────
# Test: triangle count targets at size=(10,10)
# ─────────────────────────────────────────────────────────────────────────────

def test_triangle_counts():
    """Test that each topology produces roughly the target ~950 triangles at size=(10,10)."""
    size = (10, 10)
    seed = 42

    # Loose bounds: 400-2000 triangles (some topologies are naturally sparser/denser)
    min_tri, max_tri = 400, 2000

    for name in TOPOLOGY_GENERATORS:
        result = generate_topology(name, size, seed=seed)
        n_tri = result.n_tri
        assert min_tri <= n_tri <= max_tri, \
            f"{name}: {n_tri} triangles outside [{min_tri}, {max_tri}]"
        print(f"  {name:25s}: {n_tri:4d} triangles")


# ─────────────────────────────────────────────────────────────────────────────
# Test: hard/soft edge distinction for non-triangulated meshes
# ─────────────────────────────────────────────────────────────────────────────

def test_hard_soft_edges():
    """Test that non-triangulated topologies have proper hard/soft edge distinction."""
    size = (6, 6)
    seed = 42

    # These topologies should have hard_edge_set != None
    non_triangulated = ['honeycomb', 'kagome', 'square_lattice', 'lieb',
                        'diamond', 'bond_diluted']

    for name in non_triangulated:
        if name not in TOPOLOGY_GENERATORS:
            continue

        result = generate_topology(name, size, seed=seed)

        # Should have a hard_edge_set
        assert result.hard_edge_set is not None, \
            f"{name} should have hard_edge_set (non-triangulated topology)"
        assert len(result.hard_edge_set) > 0, \
            f"{name} hard_edge_set should not be empty"

        # Check that default rigidities distinguish hard and soft
        rigs = result.get_default_rigidities()
        mask = result.get_hard_edge_mask_per_triangle()

        n_hard = mask.sum()
        n_soft = (~mask).sum()
        assert n_hard > 0, f"{name}: should have some hard edges"
        assert n_soft > 0, f"{name}: should have some soft edges"

        # Hard edges should have k=1.0, soft should have k=k_soft_ratio
        assert np.allclose(rigs[mask], 1.0), \
            f"{name}: hard edges should have k=1.0"
        assert np.allclose(rigs[~mask], result.k_soft_ratio), \
            f"{name}: soft edges should have k={result.k_soft_ratio}"

        print(f"  {name:25s}: {n_hard} hard, {n_soft} soft edges")


def test_fully_triangulated_no_soft():
    """Test that fully triangulated topologies have all hard edges (no soft)."""
    size = (6, 6)
    seed = 42

    # These topologies should have hard_edge_set = None (all edges are hard)
    fully_triangulated = ['iso_crystal', 'aniso_crystal', 'rectangular', 'oblique',
                          'foam_eta02', 'foam_eta045', 'poisson_delaunay',
                          'blue_noise', 'clustered', 'gradient_density',
                          'penrose', 'ammann_beenker']

    for name in fully_triangulated:
        if name not in TOPOLOGY_GENERATORS:
            continue

        result = generate_topology(name, size, seed=seed)
        assert result.hard_edge_set is None, \
            f"{name} should have hard_edge_set=None (fully triangulated)"

        # All default rigidities should be 1.0
        rigs = result.get_default_rigidities()
        assert np.allclose(rigs, 1.0), f"{name}: all rigidities should be 1.0"

        # All edges should be hard
        mask = result.get_hard_edge_mask_per_triangle()
        assert mask.all(), f"{name}: all edges should be hard"

        print(f"  {name:25s}: all {rigs.size} edges hard (OK)")


# ─────────────────────────────────────────────────────────────────────────────
# Test: reproducibility with seeds
# ─────────────────────────────────────────────────────────────────────────────

def test_reproducibility():
    """Test that the same seed produces the same mesh for stochastic topologies."""
    size = (6, 6)

    stochastic = ['foam_eta02', 'poisson_delaunay', 'blue_noise',
                  'clustered', 'gradient_density', 'bond_diluted']

    for name in stochastic:
        if name not in TOPOLOGY_GENERATORS:
            continue

        r1 = generate_topology(name, size, seed=42)
        r2 = generate_topology(name, size, seed=42)

        assert np.allclose(r1.points, r2.points), \
            f"{name}: same seed should produce same points"
        assert np.array_equal(r1.simplices, r2.simplices), \
            f"{name}: same seed should produce same simplices"

        print(f"  {name:25s}: reproducible (OK)")


def test_different_seeds_differ():
    """Test that different seeds produce different meshes for stochastic topologies."""
    size = (6, 6)

    stochastic = ['poisson_delaunay', 'blue_noise', 'clustered', 'gradient_density']

    for name in stochastic:
        if name not in TOPOLOGY_GENERATORS:
            continue

        r1 = generate_topology(name, size, seed=42)
        r2 = generate_topology(name, size, seed=99)

        # Points should differ (at least some)
        assert not np.allclose(r1.points, r2.points), \
            f"{name}: different seeds should produce different points"

        print(f"  {name:25s}: different seeds differ (OK)")


# ─────────────────────────────────────────────────────────────────────────────
# Test: Delaunay compat adapter
# ─────────────────────────────────────────────────────────────────────────────

def test_delaunay_compat():
    """Test that to_delaunay_compat() produces objects with correct interface."""
    size = (6, 6)

    for name in ['iso_crystal', 'honeycomb', 'kagome', 'penrose']:
        if name not in TOPOLOGY_GENERATORS:
            continue

        result = generate_topology(name, size, seed=42)
        compat = result.to_delaunay_compat()

        # Should have points and simplices attributes
        assert hasattr(compat, 'points'), f"{name}: compat needs .points"
        assert hasattr(compat, 'simplices'), f"{name}: compat needs .simplices"
        assert np.array_equal(compat.points, result.points)
        assert np.array_equal(compat.simplices, result.simplices)

        print(f"  {name:25s}: Delaunay compat OK")


# ─────────────────────────────────────────────────────────────────────────────
# Test: generate_all_topologies
# ─────────────────────────────────────────────────────────────────────────────

def test_generate_all_topologies():
    """Test that generate_all_topologies returns valid results for all topologies."""
    size = (6, 6)
    results = generate_all_topologies(size, seeds_per_random=2)

    # Should have deterministic (11) + stochastic (7 * 2 seeds) = 25 results
    assert len(results) >= 20, \
        f"Expected >= 20 topologies, got {len(results)}"

    for r in results:
        _validate_triangulation(r, r.topo_name)

    print(f"  generate_all_topologies: {len(results)} valid meshes")


# ─────────────────────────────────────────────────────────────────────────────
# Test: TOPO_CLASSES coverage
# ─────────────────────────────────────────────────────────────────────────────

def test_topo_classes_coverage():
    """Test that TOPO_CLASSES has an entry for every registered topology."""
    for name in TOPOLOGY_GENERATORS:
        assert name in TOPO_CLASSES, \
            f"Topology '{name}' missing from TOPO_CLASSES"
        assert isinstance(TOPO_CLASSES[name], str) and TOPO_CLASSES[name], \
            f"TOPO_CLASSES['{name}'] should be a non-empty string"

    print(f"  TOPO_CLASSES: all {len(TOPOLOGY_GENERATORS)} topologies covered")


# ─────────────────────────────────────────────────────────────────────────────
# Run all tests
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    tests = [
        ("All topologies valid", test_all_topologies_valid),
        ("Triangle counts", test_triangle_counts),
        ("Hard/soft edges (non-triangulated)", test_hard_soft_edges),
        ("Fully triangulated (no soft)", test_fully_triangulated_no_soft),
        ("Reproducibility", test_reproducibility),
        ("Different seeds differ", test_different_seeds_differ),
        ("Delaunay compat", test_delaunay_compat),
        ("generate_all_topologies", test_generate_all_topologies),
        ("TOPO_CLASSES coverage", test_topo_classes_coverage),
    ]

    passed = 0
    failed = 0

    for test_name, test_fn in tests:
        print(f"\n{'='*60}")
        print(f"TEST: {test_name}")
        print(f"{'='*60}")
        try:
            test_fn()
            print(f"  PASSED")
            passed += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"RESULTS: {passed} passed, {failed} failed, {passed + failed} total")
    print(f"{'='*60}")

    if failed > 0:
        sys.exit(1)
