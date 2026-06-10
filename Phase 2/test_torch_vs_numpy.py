"""
Cross-validation: PyTorch forward solver vs NumPy (Disc_2_Cont_optimized).

Runs both solvers on identical random networks and compares every intermediate
and final output.  Also runs torch.autograd.gradcheck on small networks to
verify that the analytic gradients are correct.

Usage:
    python "Phase 2/test_torch_vs_numpy.py"
"""

import sys
import os
import time

import numpy as np
import torch

# Add parent dir so we can import both modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

import Disc_2_Cont_optimized as D2C
import forward_solver_torch as fst


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_network(size=(4, 4), eta=0.2, seed=None):
    """Generate a foam network with optional fixed seed."""
    if seed is not None:
        np.random.seed(seed)
    return D2C.generate_foam_points(size, eta)


def run_numpy(tri):
    """Run NumPy solver, return dict of results."""
    D2C.analyze_elastic_struct(tri)
    return {
        'bare': tri.BareElasticTensor,
        'W': tri.Ws,
        'per_triangle': tri.ActualElasticTensor,
        'elastic_tensor': tri.totalElasticTensor,
        'poisson': tri.PoissonsRatio,
        'young': tri.YoungsModulus,
    }


def run_torch(tri):
    """Run PyTorch solver, return dict of results (as numpy)."""
    solver, rigs, rl = fst.from_triangulation(tri)
    out = solver(rigs, rl, method='woodbury')
    return {k: v.detach().numpy() for k, v in out.items()}


def compare(label, np_val, pt_val, atol=1e-10, rtol=1e-8):
    """Compare two values and print result."""
    np_val = np.asarray(np_val)
    pt_val = np.asarray(pt_val)

    if np_val.shape != pt_val.shape:
        print(f"  {label}: SHAPE MISMATCH {np_val.shape} vs {pt_val.shape}")
        return False

    abs_err = np.max(np.abs(np_val - pt_val))
    denom = np.max(np.abs(np_val))
    rel_err = abs_err / denom if denom > 0 else abs_err

    ok = abs_err < atol or rel_err < rtol
    status = "OK" if ok else "FAIL"
    print(f"  {label:25s}  abs={abs_err:.2e}  rel={rel_err:.2e}  [{status}]")
    return ok


# ---------------------------------------------------------------------------
# Test 1: Output comparison across multiple networks
# ---------------------------------------------------------------------------

def test_output_comparison():
    """Compare NumPy and PyTorch outputs on several random networks."""
    print("=" * 65)
    print("TEST 1: Output comparison (NumPy vs PyTorch)")
    print("=" * 65)

    configs = [
        # (size, eta, seed)
        ((4, 4), 0.0,  42),
        ((4, 4), 0.1,  43),
        ((4, 4), 0.2,  44),
        ((4, 4), 0.3,  45),
        ((4, 4), 0.4,  46),
        ((5, 5), 0.2,  47),
        ((6, 6), 0.15, 48),
        ((3, 3), 0.35, 49),
        ((4, 4), 0.45, 50),
        ((7, 7), 0.1,  51),
    ]

    all_ok = True
    for size, eta, seed in configs:
        print(f"\n--- size={size}, eta={eta}, seed={seed} ---")
        tri = make_network(size, eta, seed)
        N = len(tri.simplices)
        print(f"    {N} triangles, {len(tri.points)} nodes")

        np_out = run_numpy(tri)

        # Reset the triangulation to clean state for torch
        tri2 = make_network(size, eta, seed)
        pt_out = run_torch(tri2)

        ok = True
        ok &= compare("bare elastic tensor", np_out['bare'], pt_out['bare'])
        ok &= compare("W (Woodbury solution)", np_out['W'], pt_out['W'])
        ok &= compare("per-triangle tensor", np_out['per_triangle'], pt_out['per_triangle'])
        ok &= compare("effective tensor", np_out['elastic_tensor'], pt_out['elastic_tensor'])
        ok &= compare("Poisson's ratio", np_out['poisson'], pt_out['poisson'])
        ok &= compare("Young's modulus", np_out['young'], pt_out['young'])
        all_ok &= ok

    print(f"\n{'ALL PASSED' if all_ok else 'SOME FAILED'}")
    return all_ok


# ---------------------------------------------------------------------------
# Test 2: Custom rigidities and rest lengths
# ---------------------------------------------------------------------------

def test_custom_parameters():
    """Verify agreement when using non-default rigidities and rest lengths."""
    print("\n" + "=" * 65)
    print("TEST 2: Custom rigidities and rest lengths")
    print("=" * 65)

    np.random.seed(100)
    tri = make_network((4, 4), 0.2, 100)
    N = len(tri.simplices)

    # Random rigidities and rest lengths
    rigs = np.random.uniform(0.5, 2.0, (N, 3))
    rl = np.random.uniform(0.8, 1.5, (N, 3))

    # Run NumPy manually: analyze_elastic_struct calls add_edges_to_triangulation
    # which resets rigidities/rest_lenghts, so we set them AFTER that call.
    D2C.add_edges_to_triangulation(tri)
    tri.rigidities = [list(r) for r in rigs]
    tri.rest_lenghts = [list(r) for r in rl]

    # Now run the pipeline steps that come after edge extraction
    tri.BareElasticTensor = D2C._compute_local_tensors_vectorized(tri)
    mean_tensor = np.mean(tri.BareElasticTensor, 0)
    tri.delta_tensor = tri.BareElasticTensor - mean_tensor
    A_blocks = D2C._batch_to_9x9(tri.BareElasticTensor)
    B_blocks = D2C._batch_to_9x9(tri.delta_tensor)
    dA_vecs = D2C._batch_to_9vec(tri.delta_tensor)
    tri.Ws = D2C._woodbury_solve(A_blocks, B_blocks, dA_vecs)
    tri.ActualElasticTensor = D2C._compute_actual_elastic_tensor_vectorized(
        tri.BareElasticTensor, tri.Ws
    )
    tri.totalElasticTensor = np.mean(tri.ActualElasticTensor, 0)
    C = tri.totalElasticTensor
    tri.PoissonsRatio = (C[2]*C[3] - C[1]*C[4]) / (C[0]*C[3] - C[1]**2)
    tri.YoungsModulus = (
        C[2]**2*C[3] - 2*C[1]*C[2]*C[4] + C[1]**2*C[5]
        + C[0]*(C[4]**2 - C[3]*C[5])
    ) / (C[1]**2 - C[0]*C[3])

    np_out = {
        'bare': tri.BareElasticTensor,
        'W': tri.Ws,
        'per_triangle': tri.ActualElasticTensor,
        'elastic_tensor': tri.totalElasticTensor,
        'poisson': tri.PoissonsRatio,
        'young': tri.YoungsModulus,
    }

    # PyTorch version
    tri2 = make_network((4, 4), 0.2, 100)
    solver, _, _ = fst.from_triangulation(tri2)
    rigs_t = torch.tensor(rigs, dtype=torch.float64)
    rl_t = torch.tensor(rl, dtype=torch.float64)
    pt_result = solver(rigs_t, rl_t, method='woodbury')
    pt_out = {k: v.detach().numpy() for k, v in pt_result.items()}

    ok = True
    ok &= compare("bare elastic tensor", np_out['bare'], pt_out['bare'])
    ok &= compare("W (Woodbury solution)", np_out['W'], pt_out['W'])
    ok &= compare("effective tensor", np_out['elastic_tensor'], pt_out['elastic_tensor'])
    ok &= compare("Poisson's ratio", np_out['poisson'], pt_out['poisson'])
    ok &= compare("Young's modulus", np_out['young'], pt_out['young'])

    print(f"\n{'PASSED' if ok else 'FAILED'}")
    return ok


# ---------------------------------------------------------------------------
# Test 3: Gradient correctness (autograd.gradcheck)
# ---------------------------------------------------------------------------

def test_gradient_correctness():
    """Use torch.autograd.gradcheck on a small network."""
    print("\n" + "=" * 65)
    print("TEST 3: Gradient correctness (torch.autograd.gradcheck)")
    print("=" * 65)

    # Small network for tractable finite-diff checking
    tri = make_network((2, 2), 0.15, 200)
    N = len(tri.simplices)
    print(f"  Network: {N} triangles")

    solver, _, default_rl = fst.from_triangulation(tri)

    def forward_poisson(rigs):
        """Scalar output for gradcheck."""
        out = solver(rigs, default_rl, method='woodbury')
        return out['poisson']

    def forward_young(rigs):
        """Scalar output for gradcheck."""
        out = solver(rigs, default_rl, method='woodbury')
        return out['young']

    rigs = torch.ones(N, 3, dtype=torch.float64, requires_grad=True)

    print("  Checking gradients of Poisson's ratio w.r.t. rigidities...")
    ok_p = torch.autograd.gradcheck(forward_poisson, (rigs,), eps=1e-6, atol=1e-4, rtol=1e-3)
    print(f"    Poisson gradcheck: {'OK' if ok_p else 'FAIL'}")

    print("  Checking gradients of Young's modulus w.r.t. rigidities...")
    ok_y = torch.autograd.gradcheck(forward_young, (rigs,), eps=1e-6, atol=1e-4, rtol=1e-3)
    print(f"    Young gradcheck:   {'OK' if ok_y else 'FAIL'}")

    # Also check rest_lengths gradient
    rigs2 = torch.ones(N, 3, dtype=torch.float64)
    rl2 = default_rl.clone().detach().requires_grad_(True)

    def forward_poisson_rl(rl):
        out = solver(rigs2, rl, method='woodbury')
        return out['poisson']

    print("  Checking gradients of Poisson's ratio w.r.t. rest_lengths...")
    ok_rl = torch.autograd.gradcheck(forward_poisson_rl, (rl2,), eps=1e-6, atol=1e-4, rtol=1e-3)
    print(f"    Rest-length gradcheck: {'OK' if ok_rl else 'FAIL'}")

    ok = ok_p and ok_y and ok_rl
    print(f"\n{'ALL PASSED' if ok else 'SOME FAILED'}")
    return ok


# ---------------------------------------------------------------------------
# Test 4: Performance benchmark
# ---------------------------------------------------------------------------

def test_performance():
    """Benchmark forward pass speed."""
    print("\n" + "=" * 65)
    print("TEST 4: Performance benchmark")
    print("=" * 65)

    for size in [(4, 4), (8, 8), (12, 12)]:
        tri = make_network(size, 0.2, 300)
        N = len(tri.simplices)
        solver, rigs, rl = fst.from_triangulation(tri)

        # Warmup
        solver(rigs, rl, method='woodbury')

        # Benchmark
        times = []
        for _ in range(20):
            t0 = time.perf_counter()
            solver(rigs, rl, method='woodbury')
            times.append(time.perf_counter() - t0)

        mean_ms = np.mean(times) * 1000
        std_ms = np.std(times) * 1000
        print(f"  size={size}, N={N:4d} triangles: {mean_ms:.1f} +/- {std_ms:.1f} ms")

    # Benchmark with gradient computation
    tri = make_network((8, 8), 0.2, 301)
    N = len(tri.simplices)
    solver, _, rl = fst.from_triangulation(tri)
    rigs = torch.ones(N, 3, dtype=torch.float64, requires_grad=True)

    # Warmup
    out = solver(rigs, rl, method='woodbury')
    out['poisson'].backward()

    times = []
    for _ in range(20):
        if rigs.grad is not None:
            rigs.grad.zero_()
        t0 = time.perf_counter()
        out = solver(rigs, rl, method='woodbury')
        out['poisson'].backward()
        times.append(time.perf_counter() - t0)

    mean_ms = np.mean(times) * 1000
    std_ms = np.std(times) * 1000
    print(f"\n  Forward + backward (N={N}): {mean_ms:.1f} +/- {std_ms:.1f} ms")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    ok1 = test_output_comparison()
    ok2 = test_custom_parameters()
    ok3 = test_gradient_correctness()
    test_performance()

    print("\n" + "=" * 65)
    if ok1 and ok2 and ok3:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
        sys.exit(1)
    print("=" * 65)
