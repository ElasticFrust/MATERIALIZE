"""
Crystal Lattice Poisson Ratio Diagnostic
=========================================
Investigates whether heterogeneous spring constants on a regular triangular
lattice can shift the Poisson ratio away from 1/3.

Tests:
1. Uniform k=1 baseline
2. Random heterogeneous k (100 samples, log-uniform 1e-3 to 1e3)
3. Structured heterogeneous k (directional bias)
4. Gradient analysis: d(nu)/d(k_i) at k=1
5. Simple gradient descent targeting nu=0

Theory: On a perfect equilateral triangular lattice all bare tensors A[n]
are identical when k is uniform, so delta=0 and W=0, giving nu=1/3.
The question is: can heterogeneous k break this?
"""

import sys
sys.path.insert(0, '/home/user/MATERIALIZE')
sys.path.insert(0, '/home/user/MATERIALIZE/Phase 2')
sys.path.insert(0, '/home/user/MATERIALIZE/Phase 3')

import numpy as np
import torch
import Disc_2_Cont_optimized as D2C
from forward_solver_torch import from_triangulation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)

print("=" * 80)
print("CRYSTAL LATTICE POISSON RATIO DIAGNOSTIC")
print("=" * 80)

# -------------------------------------------------------------------------
# 0. Create the crystal lattice
# -------------------------------------------------------------------------
print("\n--- Setting up 10x10 iso_crystal ---")
tri = D2C.generate_cryratl_points(size=(10, 10), shape=(1, 1), orientation=0)
solver, default_rigs, default_rl = from_triangulation(tri)

N_tri = default_rigs.shape[0]
print(f"Number of triangles: {N_tri}")
print(f"Number of edges (per-triangle, with sharing): {N_tri * 3}")
print(f"Default rigidities shape: {default_rigs.shape}")

# Examine edge vectors to understand geometry
edge_vecs = solver.edge_vecs.numpy()  # (N, 3, 2)
print(f"Edge vectors shape: {edge_vecs.shape}")

# Compute edge angles (for structured test later)
angles = np.arctan2(edge_vecs[:, :, 1], edge_vecs[:, :, 0])  # (N, 3)
angles_deg = np.degrees(angles) % 180  # fold to [0, 180)
print(f"Unique edge angles (deg, modulo 180): {np.unique(np.round(angles_deg, 1))}")

# -------------------------------------------------------------------------
# 1. Uniform k=1 baseline
# -------------------------------------------------------------------------
print("\n" + "=" * 80)
print("TEST 1: Uniform k=1")
print("=" * 80)

k_uniform = torch.ones(N_tri, 3, dtype=torch.float64, requires_grad=False)
result_uniform = solver(k_uniform)
nu_uniform = result_uniform['poisson'].item()
E_uniform = result_uniform['young'].item()
C_uniform = result_uniform['elastic_tensor'].detach().numpy()

print(f"Poisson ratio: {nu_uniform:.10f}")
print(f"Young's modulus: {E_uniform:.10f}")
print(f"Elastic tensor (6 components): {C_uniform}")
print(f"Deviation from 1/3: {abs(nu_uniform - 1/3):.2e}")

# Check the bare tensors and W
bare_uniform = result_uniform['bare'].detach().numpy()
W_uniform = result_uniform['W'].detach().numpy()
print(f"\nBare tensor stats:")
print(f"  Mean: {bare_uniform.mean(axis=0)}")
print(f"  Std:  {bare_uniform.std(axis=0)}")
print(f"  Max deviation from mean: {np.max(np.abs(bare_uniform - bare_uniform.mean(axis=0))):.2e}")
print(f"W stats:")
print(f"  Max |W|: {np.max(np.abs(W_uniform)):.2e}")
print(f"  Mean |W|: {np.mean(np.abs(W_uniform)):.2e}")

# -------------------------------------------------------------------------
# 2. Random heterogeneous k (100 samples)
# -------------------------------------------------------------------------
print("\n" + "=" * 80)
print("TEST 2: Random heterogeneous k (100 samples, log-uniform 1e-3 to 1e3)")
print("=" * 80)

np.random.seed(42)
nu_random = []
for i in range(100):
    log_k = np.random.uniform(-3, 3, size=(N_tri, 3))
    k_random = torch.tensor(10.0 ** log_k, dtype=torch.float64)
    result = solver(k_random)
    nu_random.append(result['poisson'].item())

nu_random = np.array(nu_random)
print(f"Poisson ratio statistics over 100 random samples:")
print(f"  Mean:   {nu_random.mean():.10f}")
print(f"  Std:    {nu_random.std():.10f}")
print(f"  Min:    {nu_random.min():.10f}")
print(f"  Max:    {nu_random.max():.10f}")
print(f"  Range:  [{nu_random.min():.10f}, {nu_random.max():.10f}]")
print(f"  Deviation from 1/3: min={abs(nu_random.min() - 1/3):.6f}, max={abs(nu_random.max() - 1/3):.6f}")

# -------------------------------------------------------------------------
# 3. Structured heterogeneous k
# -------------------------------------------------------------------------
print("\n" + "=" * 80)
print("TEST 3: Structured heterogeneous k")
print("=" * 80)

# Identify edge orientations
# angles_deg is (N, 3) with values in [0, 180)
# "horizontal-ish" = within 30 deg of 0 or 180
# "vertical-ish" = within 30 deg of 90

def classify_edges(angles_deg, target_angle, tolerance=30):
    """Return boolean mask (N, 3) for edges within tolerance degrees of target."""
    diff = np.abs(angles_deg - target_angle)
    diff = np.minimum(diff, 180 - diff)  # handle wraparound
    return diff < tolerance

horiz_mask = classify_edges(angles_deg, 0, 30) | classify_edges(angles_deg, 180, 30)
vert_mask = classify_edges(angles_deg, 90, 30)

print(f"Horizontal-ish edges: {horiz_mask.sum()} / {N_tri * 3}")
print(f"Vertical-ish edges: {vert_mask.sum()} / {N_tri * 3}")
print(f"Neither: {(~horiz_mask & ~vert_mask).sum()} / {N_tri * 3}")

# Test 3a: Horizontal edges stiff
print("\n--- Test 3a: Horizontal edges k=100, rest k=0.01 ---")
k_horiz = np.full((N_tri, 3), 0.01)
k_horiz[horiz_mask] = 100.0
k_horiz_t = torch.tensor(k_horiz, dtype=torch.float64)
result_horiz = solver(k_horiz_t)
nu_horiz = result_horiz['poisson'].item()
print(f"Poisson ratio: {nu_horiz:.10f}")
print(f"Young's modulus: {result_horiz['young'].item():.10f}")
print(f"Elastic tensor: {result_horiz['elastic_tensor'].detach().numpy()}")

# Test 3b: Vertical edges stiff
print("\n--- Test 3b: Vertical edges k=100, rest k=0.01 ---")
k_vert = np.full((N_tri, 3), 0.01)
k_vert[vert_mask] = 100.0
k_vert_t = torch.tensor(k_vert, dtype=torch.float64)
result_vert = solver(k_vert_t)
nu_vert = result_vert['poisson'].item()
print(f"Poisson ratio: {nu_vert:.10f}")
print(f"Young's modulus: {result_vert['young'].item():.10f}")
print(f"Elastic tensor: {result_vert['elastic_tensor'].detach().numpy()}")

# Test 3c: Alternating rows (every other triangle row has different k)
print("\n--- Test 3c: Alternating by triangle: even triangles k=100, odd k=0.01 ---")
k_alt = np.ones((N_tri, 3))
k_alt[0::2, :] = 100.0
k_alt[1::2, :] = 0.01
k_alt_t = torch.tensor(k_alt, dtype=torch.float64)
result_alt = solver(k_alt_t)
nu_alt = result_alt['poisson'].item()
print(f"Poisson ratio: {nu_alt:.10f}")

# Test 3d: Even more extreme - set by triangle centroid y position
print("\n--- Test 3d: Top-half k=1000, bottom-half k=0.001 ---")
centroids = np.array([tri.points[s].mean(axis=0) for s in tri.simplices])
k_spatial = np.ones((N_tri, 3))
top_half = centroids[:, 1] > 0
k_spatial[top_half] = 1000.0
k_spatial[~top_half] = 0.001
k_spatial_t = torch.tensor(k_spatial, dtype=torch.float64)
result_spatial = solver(k_spatial_t)
nu_spatial = result_spatial['poisson'].item()
print(f"Poisson ratio: {nu_spatial:.10f}")

# Test 3e: Per-edge within triangle: make edge 0 stiff, edges 1,2 soft (all triangles)
print("\n--- Test 3e: All triangles: edge[0] k=1000, edge[1,2] k=0.001 ---")
k_edge = np.full((N_tri, 3), 0.001)
k_edge[:, 0] = 1000.0
k_edge_t = torch.tensor(k_edge, dtype=torch.float64)
result_edge = solver(k_edge_t)
nu_edge = result_edge['poisson'].item()
print(f"Poisson ratio: {nu_edge:.10f}")
print(f"Elastic tensor: {result_edge['elastic_tensor'].detach().numpy()}")

# Test 3f: Same but edge[1]
print("\n--- Test 3f: All triangles: edge[1] k=1000, edge[0,2] k=0.001 ---")
k_edge2 = np.full((N_tri, 3), 0.001)
k_edge2[:, 1] = 1000.0
k_edge2_t = torch.tensor(k_edge2, dtype=torch.float64)
result_edge2 = solver(k_edge2_t)
nu_edge2 = result_edge2['poisson'].item()
print(f"Poisson ratio: {nu_edge2:.10f}")
print(f"Elastic tensor: {result_edge2['elastic_tensor'].detach().numpy()}")

# -------------------------------------------------------------------------
# 4. Gradient analysis: d(nu)/d(k_i) at k=1
# -------------------------------------------------------------------------
print("\n" + "=" * 80)
print("TEST 4: Gradient analysis at k=1")
print("=" * 80)

k_grad = torch.ones(N_tri, 3, dtype=torch.float64, requires_grad=True)
result_grad = solver(k_grad)
nu_val = result_grad['poisson']
print(f"nu at k=1: {nu_val.item():.10f}")

# Compute gradient
nu_val.backward()
grad = k_grad.grad.detach().numpy()

print(f"\nGradient d(nu)/d(k) statistics:")
print(f"  Shape: {grad.shape}")
print(f"  Norm: {np.linalg.norm(grad):.2e}")
print(f"  Max |gradient|: {np.max(np.abs(grad)):.2e}")
print(f"  Min |gradient|: {np.min(np.abs(grad)):.2e}")
print(f"  Mean gradient:  {grad.mean():.2e}")
print(f"  Std gradient:   {grad.std():.2e}")

# Check if all gradients are the same (which would mean the gradient points
# in the "all springs equal" direction and can't break symmetry)
grad_flat = grad.flatten()
print(f"\n  Are all gradients identical?")
print(f"  Max deviation from mean: {np.max(np.abs(grad_flat - grad_flat.mean())):.2e}")
print(f"  Relative std: {grad.std() / (abs(grad.mean()) + 1e-30):.2e}")

# Also check gradient of Young's modulus
k_grad2 = torch.ones(N_tri, 3, dtype=torch.float64, requires_grad=True)
result_grad2 = solver(k_grad2)
E_val = result_grad2['young']
E_val.backward()
grad_E = k_grad2.grad.detach().numpy()

print(f"\nGradient d(E)/d(k) statistics:")
print(f"  Norm: {np.linalg.norm(grad_E):.2e}")
print(f"  Max |gradient|: {np.max(np.abs(grad_E)):.2e}")
print(f"  Mean: {grad_E.mean():.2e}")
print(f"  Std:  {grad_E.std():.2e}")

# -------------------------------------------------------------------------
# 4b. Gradient at a NON-uniform point
# -------------------------------------------------------------------------
print("\n--- Gradient at a random heterogeneous k point ---")
np.random.seed(123)
k_init = 10.0 ** np.random.uniform(-1, 1, size=(N_tri, 3))
k_grad3 = torch.tensor(k_init, dtype=torch.float64, requires_grad=True)
result_grad3 = solver(k_grad3)
nu3 = result_grad3['poisson']
print(f"nu at random k: {nu3.item():.10f}")
nu3.backward()
grad3 = k_grad3.grad.detach().numpy()
print(f"Gradient d(nu)/d(k) at random k:")
print(f"  Norm: {np.linalg.norm(grad3):.2e}")
print(f"  Max |gradient|: {np.max(np.abs(grad3)):.2e}")
print(f"  Std:  {grad3.std():.2e}")
print(f"  Relative std: {grad3.std() / (abs(grad3.mean()) + 1e-30):.2e}")

# -------------------------------------------------------------------------
# 4c. Numerical gradient check for a few springs
# -------------------------------------------------------------------------
print("\n--- Numerical gradient check (finite differences) ---")
eps_fd = 1e-5
k_base = torch.ones(N_tri, 3, dtype=torch.float64)
nu_base = solver(k_base)['poisson'].item()
print(f"Base nu: {nu_base:.10f}")

# Check 5 random springs
np.random.seed(999)
check_indices = [(np.random.randint(N_tri), np.random.randint(3)) for _ in range(5)]
for (i, j) in check_indices:
    k_plus = k_base.clone()
    k_plus[i, j] += eps_fd
    nu_plus = solver(k_plus)['poisson'].item()

    k_minus = k_base.clone()
    k_minus[i, j] -= eps_fd
    nu_minus = solver(k_minus)['poisson'].item()

    fd_grad = (nu_plus - nu_minus) / (2 * eps_fd)
    print(f"  Spring ({i},{j}): d(nu)/dk = {fd_grad:.2e}  (nu+ = {nu_plus:.10f}, nu- = {nu_minus:.10f})")

# -------------------------------------------------------------------------
# 5. Gradient descent targeting nu=0
# -------------------------------------------------------------------------
print("\n" + "=" * 80)
print("TEST 5: Gradient descent targeting nu=0, starting from k=1")
print("=" * 80)

target_nu = 0.0
lr = 0.01
n_steps = 10000
log_every = 500

# Initialize
k_opt = torch.ones(N_tri, 3, dtype=torch.float64, requires_grad=True)
nu_history = []
loss_history = []
grad_norm_history = []
k_std_history = []

print(f"Target nu: {target_nu}")
print(f"Learning rate: {lr}")
print(f"Steps: {n_steps}")

for step in range(n_steps):
    result = solver(k_opt)
    nu = result['poisson']
    loss = (nu - target_nu) ** 2

    loss.backward()

    nu_history.append(nu.item())
    loss_history.append(loss.item())
    grad_norm_history.append(k_opt.grad.norm().item())
    k_std_history.append(k_opt.detach().std().item())

    # Simple gradient descent step
    with torch.no_grad():
        k_opt -= lr * k_opt.grad
        # Clamp to positive values
        k_opt.clamp_(min=1e-6)

    k_opt.grad.zero_()

    if step % log_every == 0 or step == n_steps - 1:
        print(f"  Step {step:5d}: nu = {nu_history[-1]:.10f}, "
              f"loss = {loss_history[-1]:.2e}, "
              f"grad_norm = {grad_norm_history[-1]:.2e}, "
              f"k_std = {k_std_history[-1]:.6f}")

print(f"\nFinal nu: {nu_history[-1]:.10f}")
print(f"Change in nu: {nu_history[-1] - nu_history[0]:.2e}")
print(f"Final k stats: mean={k_opt.detach().mean().item():.6f}, "
      f"std={k_opt.detach().std().item():.6f}, "
      f"min={k_opt.detach().min().item():.6f}, "
      f"max={k_opt.detach().max().item():.6f}")

# -------------------------------------------------------------------------
# 5b. Try with much larger learning rate
# -------------------------------------------------------------------------
print("\n--- Gradient descent with lr=1.0 ---")
k_opt2 = torch.ones(N_tri, 3, dtype=torch.float64, requires_grad=True)
nu_history2 = []

for step in range(n_steps):
    result = solver(k_opt2)
    nu = result['poisson']
    loss = (nu - target_nu) ** 2
    loss.backward()

    nu_history2.append(nu.item())

    with torch.no_grad():
        k_opt2 -= 1.0 * k_opt2.grad
        k_opt2.clamp_(min=1e-6)
    k_opt2.grad.zero_()

    if step % log_every == 0 or step == n_steps - 1:
        print(f"  Step {step:5d}: nu = {nu_history2[-1]:.10f}, "
              f"k_std = {k_opt2.detach().std().item():.6f}")

print(f"\nFinal nu (lr=1.0): {nu_history2[-1]:.10f}")
print(f"Change in nu: {nu_history2[-1] - nu_history2[0]:.2e}")

# -------------------------------------------------------------------------
# 5c. Try gradient descent on the LOSS = (C_xxxx - C_xxyy)^2 directly
#     to see if individual C components can be changed
# -------------------------------------------------------------------------
print("\n--- Can we change C_xxxx / C_xxyy ratio? ---")
k_opt3 = torch.ones(N_tri, 3, dtype=torch.float64, requires_grad=True)
for step in range(2000):
    result = solver(k_opt3)
    C = result['elastic_tensor']
    # Try to maximize C_xxxx / C_xxyy (make material stiffer in x)
    loss = -C[0]  # maximize C_xxxx
    loss.backward()

    with torch.no_grad():
        k_opt3 -= 0.1 * k_opt3.grad
        k_opt3.clamp_(min=1e-6)
    k_opt3.grad.zero_()

    if step % 500 == 0 or step == 1999:
        C_np = result['elastic_tensor'].detach().numpy()
        nu_now = result['poisson'].item()
        print(f"  Step {step}: C_xxxx={C_np[0]:.6f}, C_xxyy={C_np[2]:.6f}, "
              f"C_yyyy={C_np[5]:.6f}, nu={nu_now:.10f}")

# -------------------------------------------------------------------------
# 6. Theoretical analysis: WHY is nu stuck?
# -------------------------------------------------------------------------
print("\n" + "=" * 80)
print("THEORETICAL ANALYSIS")
print("=" * 80)

# On a perfect equilateral triangular lattice with uniform k:
# Each triangle has edges at 0, 60, 120 degrees (or similar set).
# The bare tensor for each triangle is:
# a0 = k/(16*l^2) * sum(vx^4)
# etc.
# All triangles are identical -> delta = 0, W = 0
# C_effective = mean(A) = A (uniform)

# With heterogeneous k:
# Each triangle's bare tensor = k_i * (geometric factor for edge i)
# Since the geometry is the same for all triangles of the same "type"
# (upward vs downward pointing), the tensor components are linear in k.

# The question: are the geometric factors the same for all triangles?
print("\nExamining bare tensor structure on uniform lattice:")
k_test = torch.ones(N_tri, 3, dtype=torch.float64)
result_test = solver(k_test)
bare = result_test['bare'].detach().numpy()

# Check unique bare tensors
unique_bare = np.unique(np.round(bare, 10), axis=0)
print(f"Number of unique bare tensors (at k=1): {len(unique_bare)}")
for i, ub in enumerate(unique_bare):
    count = np.sum(np.all(np.abs(bare - ub) < 1e-8, axis=1))
    print(f"  Type {i}: {ub} (count={count})")

# Check edge vectors per triangle
vx = solver.edge_vecs[:, :, 0].numpy()
vy = solver.edge_vecs[:, :, 1].numpy()
print(f"\nEdge vector patterns:")
edge_patterns = np.stack([vx, vy], axis=-1)  # (N, 3, 2)
unique_patterns = np.unique(np.round(edge_patterns.reshape(N_tri, -1), 8), axis=0)
print(f"Number of unique edge vector patterns: {len(unique_patterns)}")

# The KEY insight: on a crystal lattice, there are typically 2 types of
# triangles (upward and downward pointing). Within each type, the edge
# vectors are identical. So the bare tensor depends only on the TYPE
# of triangle and the 3 spring constants.
print("\nFor each unique edge pattern, show the bare tensor contribution:")
for i, pattern in enumerate(unique_patterns[:5]):
    mask = np.all(np.abs(edge_patterns.reshape(N_tri, -1) - pattern) < 1e-8, axis=1)
    count = mask.sum()
    evx = vx[mask][0]
    evy = vy[mask][0]
    angles_p = np.degrees(np.arctan2(evy, evx))
    print(f"  Pattern {i} (count={count}): angles={angles_p}")
    # Compute geometric factors per edge
    length2 = evx**2 + evy**2
    for j in range(3):
        f = 1.0 / length2[j] / 16.0
        a0_contrib = f * evx[j]**4
        a2_contrib = f * evx[j]**2 * evy[j]**2
        a4_contrib = f * evy[j]**4
        print(f"    Edge {j}: angle={angles_p[j]:.1f} deg, "
              f"a0_contrib={a0_contrib:.6f}, a2_contrib={a2_contrib:.6f}, "
              f"a4_contrib={a4_contrib:.6f}")

# -------------------------------------------------------------------------
# 7. The critical test: check if ALL BARE TENSORS remain proportional
#    when k varies per-edge but geometry is fixed
# -------------------------------------------------------------------------
print("\n" + "=" * 80)
print("CRITICAL TEST: Do bare tensors become non-proportional with hetero k?")
print("=" * 80)

# Set random k
np.random.seed(77)
k_hetero = 10.0 ** np.random.uniform(-2, 2, size=(N_tri, 3))
k_hetero_t = torch.tensor(k_hetero, dtype=torch.float64)
result_hetero = solver(k_hetero_t)
bare_hetero = result_hetero['bare'].detach().numpy()
W_hetero = result_hetero['W'].detach().numpy()

print(f"Bare tensor stats with heterogeneous k:")
print(f"  Mean: {bare_hetero.mean(axis=0)}")
print(f"  Std:  {bare_hetero.std(axis=0)}")
print(f"  Max delta from mean: {np.max(np.abs(bare_hetero - bare_hetero.mean(axis=0))):.6f}")

print(f"\nW stats with heterogeneous k:")
print(f"  Max |W|: {np.max(np.abs(W_hetero)):.6f}")
print(f"  Mean |W|: {np.mean(np.abs(W_hetero)):.6f}")

print(f"\nPoisson ratio: {result_hetero['poisson'].item():.10f}")

# Check the ratio a0:a2:a4 for each triangle
# For isotropic material, we need a0 = a4 and a2 = a0/3
ratios_a2_a0 = bare_hetero[:, 2] / bare_hetero[:, 0]
ratios_a4_a0 = bare_hetero[:, 4] / bare_hetero[:, 0]
ratios_a1_a0 = bare_hetero[:, 1] / bare_hetero[:, 0]
ratios_a3_a0 = bare_hetero[:, 3] / bare_hetero[:, 0]

print(f"\nBare tensor ratios (testing isotropy/proportionality):")
print(f"  a2/a0: mean={ratios_a2_a0.mean():.6f}, std={ratios_a2_a0.std():.6f}")
print(f"  a4/a0: mean={ratios_a4_a0.mean():.6f}, std={ratios_a4_a0.std():.6f}")
print(f"  a1/a0: mean={ratios_a1_a0.mean():.6f}, std={ratios_a1_a0.std():.6f}")
print(f"  a3/a0: mean={ratios_a3_a0.mean():.6f}, std={ratios_a3_a0.std():.6f}")
print(f"  (For isotropic: a2/a0 should be 1/3, a4/a0 should be 1)")

# -------------------------------------------------------------------------
# 8. Check: per-triangle actual elastic tensor
# -------------------------------------------------------------------------
print("\n" + "=" * 80)
print("PER-TRIANGLE ACTUAL ELASTIC TENSOR")
print("=" * 80)
actual_hetero = result_hetero['per_triangle'].detach().numpy()
actual_uniform = result_uniform['per_triangle'].detach().numpy()

print(f"With uniform k=1:")
print(f"  C_xxxx mean={actual_uniform[:, 0].mean():.6f}, std={actual_uniform[:, 0].std():.2e}")
print(f"  C_xxyy mean={actual_uniform[:, 2].mean():.6f}, std={actual_uniform[:, 2].std():.2e}")
print(f"  C_yyyy mean={actual_uniform[:, 5].mean():.6f}, std={actual_uniform[:, 5].std():.2e}")

print(f"\nWith heterogeneous k:")
print(f"  C_xxxx mean={actual_hetero[:, 0].mean():.6f}, std={actual_hetero[:, 0].std():.6f}")
print(f"  C_xxyy mean={actual_hetero[:, 2].mean():.6f}, std={actual_hetero[:, 2].std():.6f}")
print(f"  C_yyyy mean={actual_hetero[:, 5].mean():.6f}, std={actual_hetero[:, 5].std():.6f}")

# The key question: does the mean tensor still have nu=1/3 structure?
C_het = result_hetero['elastic_tensor'].detach().numpy()
print(f"\nHomogenized tensor (heterogeneous k):")
print(f"  C_xxxx = {C_het[0]:.6f}")
print(f"  C_xxxy = {C_het[1]:.6f}")
print(f"  C_xxyy = {C_het[2]:.6f}")
print(f"  C_xyxy = {C_het[3]:.6f}")
print(f"  C_xyyy = {C_het[4]:.6f}")
print(f"  C_yyyy = {C_het[5]:.6f}")
print(f"  nu = C_xxyy/C_xxxx = {C_het[2]/C_het[0]:.6f} (would be 1/3 for isotropic)")
print(f"  C_xxxx/C_yyyy = {C_het[0]/C_het[5]:.6f} (would be 1 for isotropic)")

# -------------------------------------------------------------------------
# Plot results
# -------------------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Plot 1: Random nu distribution
ax = axes[0, 0]
ax.hist(nu_random, bins=30, edgecolor='black', alpha=0.7)
ax.axvline(1/3, color='red', linestyle='--', linewidth=2, label='1/3')
ax.set_xlabel('Poisson ratio')
ax.set_ylabel('Count')
ax.set_title('Random heterogeneous k: nu distribution')
ax.legend()

# Plot 2: Gradient descent trajectory (lr=0.01)
ax = axes[0, 1]
ax.plot(nu_history, 'b-', linewidth=0.5)
ax.axhline(1/3, color='red', linestyle='--', label='1/3')
ax.axhline(target_nu, color='green', linestyle='--', label=f'target={target_nu}')
ax.set_xlabel('Step')
ax.set_ylabel('Poisson ratio')
ax.set_title(f'GD trajectory (lr=0.01)')
ax.legend()

# Plot 3: Gradient descent trajectory (lr=1.0)
ax = axes[0, 2]
ax.plot(nu_history2, 'r-', linewidth=0.5)
ax.axhline(1/3, color='red', linestyle='--', label='1/3')
ax.axhline(target_nu, color='green', linestyle='--', label=f'target={target_nu}')
ax.set_xlabel('Step')
ax.set_ylabel('Poisson ratio')
ax.set_title(f'GD trajectory (lr=1.0)')
ax.legend()

# Plot 4: Gradient norm over optimization
ax = axes[1, 0]
ax.semilogy(grad_norm_history, 'b-', linewidth=0.5)
ax.set_xlabel('Step')
ax.set_ylabel('Gradient norm')
ax.set_title('Gradient norm during GD (lr=0.01)')

# Plot 5: k spread over optimization
ax = axes[1, 1]
ax.plot(k_std_history, 'b-', linewidth=0.5)
ax.set_xlabel('Step')
ax.set_ylabel('Std of k values')
ax.set_title('Spring constant heterogeneity during GD')

# Plot 6: Loss over optimization
ax = axes[1, 2]
ax.semilogy(loss_history, 'b-', linewidth=0.5)
ax.set_xlabel('Step')
ax.set_ylabel('Loss = (nu - target)^2')
ax.set_title('Loss during GD (lr=0.01)')

plt.tight_layout()
plt.savefig('/home/user/MATERIALIZE/Phase 3/crystal_diagnostic2.png', dpi=150)
print(f"\nPlot saved to /home/user/MATERIALIZE/Phase 3/crystal_diagnostic2.png")

# -------------------------------------------------------------------------
# Summary
# -------------------------------------------------------------------------
print("\n" + "=" * 80)
print("SUMMARY OF FINDINGS")
print("=" * 80)
print(f"""
1. UNIFORM k=1: nu = {nu_uniform:.10f} (deviation from 1/3: {abs(nu_uniform - 1/3):.2e})

2. RANDOM HETEROGENEOUS k (100 samples):
   nu range: [{nu_random.min():.6f}, {nu_random.max():.6f}]
   Total variation: {nu_random.max() - nu_random.min():.6f}

3. STRUCTURED HETEROGENEOUS k:
   Horizontal stiff: nu = {nu_horiz:.6f}
   Vertical stiff:   nu = {nu_vert:.6f}
   Alternating:      nu = {nu_alt:.6f}
   Top/bottom:       nu = {nu_spatial:.6f}
   Edge[0] stiff:    nu = {nu_edge:.6f}
   Edge[1] stiff:    nu = {nu_edge2:.6f}

4. GRADIENT at k=1:
   Norm: {np.linalg.norm(grad):.2e}
   Are all gradients identical? Max deviation: {np.max(np.abs(grad_flat - grad_flat.mean())):.2e}

5. GRADIENT DESCENT (10000 steps):
   lr=0.01: nu went from {nu_history[0]:.10f} to {nu_history[-1]:.10f} (change: {nu_history[-1] - nu_history[0]:.2e})
   lr=1.0:  nu went from {nu_history2[0]:.10f} to {nu_history2[-1]:.10f} (change: {nu_history2[-1] - nu_history2[0]:.2e})
""")

if abs(nu_random.max() - nu_random.min()) < 0.01:
    print("CONCLUSION: Poisson ratio is essentially LOCKED at 1/3 on the crystal lattice.")
    print("Heterogeneous spring constants have negligible effect.")
    print("\nThis is because on a PERFECT equilateral triangular lattice:")
    print("- There are exactly 2 triangle types (up/down) with mirrored geometry")
    print("- Edge vectors within each type are related by 60-degree rotations")
    print("- The bare tensor a0:a2:a4 ratio is geometrically constrained to 1:1/3:1")
    print("- Even with heterogeneous k, the hexagonal symmetry of the GEOMETRY")
    print("  forces the homogenized tensor to remain isotropic with nu=1/3")
else:
    print("FINDING: Heterogeneous k CAN change the Poisson ratio on the crystal lattice!")
    print(f"The achievable range appears to be [{nu_random.min():.6f}, {nu_random.max():.6f}]")
