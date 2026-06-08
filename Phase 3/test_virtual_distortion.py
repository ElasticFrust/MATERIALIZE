"""Test virtual distortion rigidity assignment.

Creates a regular triangular lattice, applies the virtual distortion method
with eta=0.15 and a=10, then checks:
  1. Poisson's ratio differs from the uniform-rigidity baseline (1/3).
  2. The material remains isotropic (C_xxxx ≈ C_yyyy, off-diagonals ≈ 0).
  3. Rigidity statistics are sensible (all positive, spread around 1).
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
from sweep_utils import virtual_distortion_rigidities

# ── Parameters ───────────────────────────────────────────────────────────
ETA  = 0.15
A    = 10
SIZE = (14, 14)
SEED = 42

# ── Run ──────────────────────────────────────────────────────────────────
print(f"Virtual distortion test: eta={ETA}, a={A}, size={SIZE}, seed={SEED}")
print("=" * 70)

vd = virtual_distortion_rigidities(size=SIZE, eta=ETA, a=A, seed=SEED)

solver = vd['solver']
rigs   = vd['rigidities']
n_tri  = vd['n_tri']

# ── 1. Baseline: uniform rigidity on same lattice ────────────────────────
with torch.no_grad():
    baseline = solver(torch.ones(n_tri, 3, dtype=torch.float64))
    nu_base = baseline['poisson'].item()
    E_base  = baseline['young'].item()
    C_base  = baseline['elastic_tensor'].numpy()

print(f"\nBaseline (uniform k=1):")
print(f"  Poisson ratio = {nu_base:.6f}")
print(f"  Young modulus  = {E_base:.6f}")

# ── 2. Virtual distortion result ─────────────────────────────────────────
with torch.no_grad():
    result = solver(rigs)
    nu_vd = result['poisson'].item()
    E_vd  = result['young'].item()
    C_vd  = result['elastic_tensor'].numpy()

print(f"\nVirtual distortion (eta={ETA}, a={A}):")
print(f"  Poisson ratio = {nu_vd:.6f}")
print(f"  Young modulus  = {E_vd:.6f}")
print(f"  Delta Poisson  = {nu_vd - nu_base:+.6f}")

# ── 3. Isotropy check ────────────────────────────────────────────────────
# C = [C_xxxx, C_xxxy, C_xxyy, C_xyxy, C_xyyy, C_yyyy]
#      C[0]    C[1]    C[2]    C[3]    C[4]    C[5]
# Isotropy requires:
#   C_xxxx == C_yyyy              => C[0] ≈ C[5]
#   C_xxxy == 0, C_xyyy == 0     => C[1] ≈ 0, C[4] ≈ 0
#   C_xyxy == (C_xxxx - C_xxyy)/2 => C[3] ≈ (C[0] - C[2])/2

C = C_vd
scale = 0.5 * (abs(C[0]) + abs(C[5]))  # normalisation scale

aniso_C0_C5   = abs(C[0] - C[5]) / scale
aniso_C1       = abs(C[1]) / scale
aniso_C4       = abs(C[4]) / scale
shear_relation = abs(C[3] - (C[0] - C[2]) / 2) / scale

print(f"\nIsotropy diagnostics (should all be << 1):")
print(f"  |C_xxxx - C_yyyy| / scale  = {aniso_C0_C5:.6f}")
print(f"  |C_xxxy|          / scale  = {aniso_C1:.6f}")
print(f"  |C_xyyy|          / scale  = {aniso_C4:.6f}")
print(f"  |C_xyxy - (C_xxxx-C_xxyy)/2| / scale = {shear_relation:.6f}")

print(f"\n  Full tensor: {C}")

# ── 4. Rigidity statistics ────────────────────────────────────────────────
k_np = vd['rigidities_np']
print(f"\nRigidity statistics ({n_tri} triangles × 3 edges = {n_tri*3} values):")
print(f"  min  = {k_np.min():.6f}")
print(f"  max  = {k_np.max():.6f}")
print(f"  mean = {k_np.mean():.6f}")
print(f"  std  = {k_np.std():.6f}")

dl = vd['l_deformed'] - vd['l0']
print(f"\n  Edge length changes (l - l0):")
print(f"    min  = {dl.min():.6f}")
print(f"    max  = {dl.max():.6f}")
print(f"    mean = {dl.mean():.6f}")
print(f"    std  = {dl.std():.6f}")

# ── 5. Multi-seed consistency ─────────────────────────────────────────────
print(f"\nMulti-seed test (5 seeds, same eta={ETA}, a={A}):")
print(f"  {'Seed':>6s}  {'Poisson':>10s}  {'Young':>10s}  {'k_mean':>8s}  {'k_std':>8s}")
for s in [42, 137, 256, 999, 1337]:
    vd_s = virtual_distortion_rigidities(size=(8, 8), eta=ETA, a=A, seed=s)
    with torch.no_grad():
        r_s = vd_s['solver'](vd_s['rigidities'])
    print(f"  {s:6d}  {r_s['poisson'].item():10.6f}  {r_s['young'].item():10.6f}"
          f"  {vd_s['rigidities_np'].mean():8.4f}  {vd_s['rigidities_np'].std():8.4f}")

# ── 6. Assertions ─────────────────────────────────────────────────────────
print("\n" + "=" * 70)
ISOTROPY_TOL = 0.10  # 10% relative tolerance for isotropy (finite-size noise)

passed = True

# Poisson ratio should differ from baseline
if abs(nu_vd - nu_base) < 1e-6:
    print("FAIL: Poisson ratio unchanged from baseline")
    passed = False
else:
    print(f"PASS: Poisson ratio changed by {nu_vd - nu_base:+.6f}")

# Isotropy checks
if aniso_C0_C5 > ISOTROPY_TOL:
    print(f"FAIL: C_xxxx != C_yyyy  (relative diff = {aniso_C0_C5:.4f})")
    passed = False
else:
    print(f"PASS: C_xxxx ≈ C_yyyy  (relative diff = {aniso_C0_C5:.4f})")

if aniso_C1 > ISOTROPY_TOL:
    print(f"FAIL: C_xxxy != 0  (relative = {aniso_C1:.4f})")
    passed = False
else:
    print(f"PASS: C_xxxy ≈ 0  (relative = {aniso_C1:.4f})")

if aniso_C4 > ISOTROPY_TOL:
    print(f"FAIL: C_xyyy != 0  (relative = {aniso_C4:.4f})")
    passed = False
else:
    print(f"PASS: C_xyyy ≈ 0  (relative = {aniso_C4:.4f})")

if shear_relation > ISOTROPY_TOL:
    print(f"FAIL: shear relation violated  (relative = {shear_relation:.4f})")
    passed = False
else:
    print(f"PASS: shear relation  (relative = {shear_relation:.4f})")

# All rigidities positive
if k_np.min() <= 0:
    print(f"FAIL: some rigidities <= 0  (min = {k_np.min():.6f})")
    passed = False
else:
    print(f"PASS: all rigidities > 0  (min = {k_np.min():.6f})")

print("=" * 70)
if passed:
    print("ALL TESTS PASSED")
else:
    print("SOME TESTS FAILED")
    sys.exit(1)
