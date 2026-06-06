"""
Verification of the KKT edge-compatibility correction to the forward solver.

Tests (in order):
  1. Backup exists
  2. Uniform mesh: δA=0 → W₀=0, KKT correction is zero, ν=1/3
  3. Constraint satisfaction: ‖JW‖ = 0 after correction, ‖JW₀‖ large for heterogeneous k
  4. Auxetic response vs η: compare old (NumPy/mean-field) vs new (KKT) for η ∈ 0..0.5
  5. Hexagonal tiling (η=0 crystal): both solvers should agree (δA=0 there)
  6. Gradient check: torch.autograd.gradcheck through the KKT solve

Usage:
    cd /home/user/MATERIALIZE
    python "Phase 2/test_kkt_correction.py"
"""

import sys, os
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

import Disc_2_Cont_optimized as D2C
import forward_solver_torch as fst_new

# ── helpers ─────────────────────────────────────────────────────────────────

def make_network(size=(4, 4), eta=0.0, seed=42):
    np.random.seed(seed)
    return D2C.generate_foam_points(size, eta)

def nu_numpy(tri, rigs=None, rest_lengths=None):
    """Run the old NumPy (mean-field) solver and return ν."""
    D2C.analyze_elastic_struct(tri)
    if rigs is not None:
        D2C.add_edges_to_triangulation(tri)
        tri.rigidities  = [list(r) for r in rigs]
        tri.rest_lenghts = [list(r) for r in (rest_lengths if rest_lengths is not None
                                               else np.sqrt(tri.actual_length2_np))]
        tri.BareElasticTensor = D2C._compute_local_tensors_vectorized(tri)
        mean_t = np.mean(tri.BareElasticTensor, 0)
        tri.delta_tensor = tri.BareElasticTensor - mean_t
        A_b = D2C._batch_to_9x9(tri.BareElasticTensor)
        B_b = D2C._batch_to_9x9(tri.delta_tensor)
        dA  = D2C._batch_to_9vec(tri.delta_tensor)
        tri.Ws = D2C._woodbury_solve(A_b, B_b, dA)
        tri.ActualElasticTensor = D2C._compute_actual_elastic_tensor_vectorized(
            tri.BareElasticTensor, tri.Ws)
        tri.totalElasticTensor = np.mean(tri.ActualElasticTensor, 0)
        C = tri.totalElasticTensor
        return (C[2]*C[3] - C[1]*C[4]) / (C[0]*C[3] - C[1]**2)
    return tri.PoissonsRatio

def nu_new(tri, rigs_t=None):
    """Run the new KKT-corrected solver and return ν."""
    solver, default_rigs, default_rl = fst_new.from_triangulation(tri)
    r = rigs_t if rigs_t is not None else default_rigs
    return solver(r, default_rl)['poisson'].item()

def section(title):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print('='*65)

# ── Test 1: backup ───────────────────────────────────────────────────────────

section("TEST 1: Backup file exists")
bckp = os.path.join(os.path.dirname(__file__), 'forward_solver_torch.py.bckp')
if os.path.exists(bckp):
    print("  forward_solver_torch.py.bckp  ✓")
else:
    print("  MISSING — run the backup command first")
bckp2 = os.path.join(os.path.dirname(__file__), '..', 'Disc_2_Cont_optimized.py.bckp')
if os.path.exists(bckp2):
    print("  Disc_2_Cont_optimized.py.bckp ✓")
else:
    print("  MISSING")

# ── Test 2: uniform mesh (η=0, uniform k=1) ──────────────────────────────────

section("TEST 2: Uniform mesh  δA=0 → W=0, ν=1/3")

tri_unif = make_network(size=(6, 6), eta=0.0, seed=0)
solver_u, rigs_u, rl_u = fst_new.from_triangulation(tri_unif)
out_u = solver_u(rigs_u, rl_u)

W_unif  = out_u['W'].detach().numpy()
nu_unif = out_u['poisson'].item()

print(f"  N triangles : {len(tri_unif.simplices)}")
print(f"  ‖W‖_max     : {np.abs(W_unif).max():.2e}   (expect ~0)")
print(f"  ν (KKT)     : {nu_unif:.6f}   (expect 0.333333...)")
print(f"  ν (analytic): 0.333333")
print(f"  {'PASS' if abs(nu_unif - 1/3) < 1e-4 else 'FAIL'}  (|Δν| = {abs(nu_unif - 1/3):.2e})")

# ── Test 3: constraint satisfaction ──────────────────────────────────────────

section("TEST 3: Edge-constraint satisfaction  ‖JW‖")

np.random.seed(7)
tri_het = make_network(size=(5, 5), eta=0.2, seed=7)
solver_h, rigs_h, rl_h = fst_new.from_triangulation(tri_het)

# Heterogeneous rigidities (binary pattern)
N_h = rigs_h.shape[0]
rigs_bin = torch.where(
    torch.rand(N_h, 3, dtype=torch.float64) > 0.5,
    torch.full((N_h, 3), 5.0, dtype=torch.float64),
    torch.full((N_h, 3), 0.1, dtype=torch.float64),
)
out_h = solver_h(rigs_bin, rl_h)

J = solver_h.J
if J is not None:
    # Compute W₀ (without correction) using the backup/numpy solver logic
    bare  = out_h['bare'].detach()
    delta = bare - bare.mean(0)
    A_b   = fst_new._batch_to_9x9(bare)
    B_b   = fst_new._batch_to_9x9(delta)
    dA    = fst_new._batch_to_9vec(delta)
    W0_no_kkt = fst_new._woodbury_solve(A_b, B_b, dA, J=None)  # old solve
    W_kkt     = out_h['W'].detach()

    r_before = (J @ W0_no_kkt.reshape(-1)).norm().item()
    r_after  = (J @ W_kkt.reshape(-1)).norm().item()
    print(f"  E_int       : {J.shape[0]//3}")
    print(f"  ‖J W₀‖      : {r_before:.4e}  (violation before KKT)")
    print(f"  ‖J W‖       : {r_after:.4e}   (violation after KKT, expect ~0)")
    print(f"  {'PASS' if r_after < 1e-8 else 'FAIL'}  (threshold 1e-8)")
else:
    print("  J is None — no interior edges found (tiny mesh?)")

# ── Test 4: Auxetic response vs η ────────────────────────────────────────────

section("TEST 4: Auxetic response ν(η)  — old vs new, uniform k=1")

etas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
nu_old_list, nu_new_list = [], []

print(f"\n  {'η':>5}  {'ν old (mean-field)':>20}  {'ν new (KKT)':>15}  {'Δν':>10}")
print(f"  {'-'*60}")

for eta in etas:
    np.random.seed(42)
    tri = make_network(size=(6, 6), eta=eta, seed=42)
    D2C.analyze_elastic_struct(tri)
    nu_o = tri.PoissonsRatio

    np.random.seed(42)
    tri2 = make_network(size=(6, 6), eta=eta, seed=42)
    nu_n = nu_new(tri2)

    nu_old_list.append(nu_o)
    nu_new_list.append(nu_n)
    print(f"  {eta:>5.1f}  {nu_o:>20.6f}  {nu_n:>15.6f}  {nu_n-nu_o:>+10.4f}")

# ── Test 5: Hexagonal (crystal, η=0) ─────────────────────────────────────────

section("TEST 5: Hexagonal crystal (η=0) — both solvers should agree")

np.random.seed(0)
tri_hex = make_network(size=(8, 8), eta=0.0, seed=0)
D2C.analyze_elastic_struct(tri_hex)
nu_hex_old = tri_hex.PoissonsRatio

np.random.seed(0)
tri_hex2 = make_network(size=(8, 8), eta=0.0, seed=0)
nu_hex_new = nu_new(tri_hex2)

print(f"  ν old (mean-field) : {nu_hex_old:.6f}")
print(f"  ν new (KKT)        : {nu_hex_new:.6f}")
print(f"  Δν                 : {nu_hex_new - nu_hex_old:+.2e}")
print(f"  {'PASS (agree to 1e-4)' if abs(nu_hex_new - nu_hex_old) < 1e-4 else 'DIFFER — check δA'}")

# ── Test 6: Gradient check ───────────────────────────────────────────────────

section("TEST 6: torch.autograd.gradcheck through KKT solve")

np.random.seed(99)
tri_small = make_network(size=(2, 2), eta=0.15, seed=99)
solver_s, _, rl_s = fst_new.from_triangulation(tri_small)
N_s = rl_s.shape[0]
rigs_gc = torch.ones(N_s, 3, dtype=torch.float64, requires_grad=True)

def fwd_nu(r):
    return solver_s(r, rl_s)['poisson']

print(f"  Mesh: {N_s} triangles, {solver_s.J.shape[0]//3 if solver_s.J is not None else 0} interior edges")
try:
    ok = torch.autograd.gradcheck(fwd_nu, (rigs_gc,), eps=1e-6, atol=1e-4, rtol=1e-3)
    print(f"  gradcheck: {'PASS' if ok else 'FAIL'}")
except Exception as e:
    print(f"  gradcheck: FAIL  ({e})")

# ── Summary ──────────────────────────────────────────────────────────────────

section("SUMMARY")
print("""
  Test 2 (uniform): W=0 and ν=1/3  → KKT correction is a no-op when δA=0
  Test 3 (‖JW‖):   violation drops to machine precision after correction
  Test 4 (η sweep): new ν(η) curve — report any auxetic shift vs mean-field
  Test 5 (crystal): old and new agree for uniform networks
  Test 6 (gradcheck): differentiability preserved through KKT solve
""")
print("  Plot ν(η) comparison: see nu_old_list, nu_new_list above.")
print(f"  η values : {etas}")
print(f"  ν old    : {[round(x, 4) for x in nu_old_list]}")
print(f"  ν new    : {[round(x, 4) for x in nu_new_list]}")
