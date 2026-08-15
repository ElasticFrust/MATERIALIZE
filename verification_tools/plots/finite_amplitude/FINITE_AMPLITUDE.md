# What the solver's constitutive approximation costs at FINITE amplitude

**What.** Every check in the repo so far is *tangent*: linear solver vs linear oracle, both evaluated
at infinitesimal amplitude. None of them bounds the error a user incurs at a finite strain. This
measures it, and — the point of the experiment — **separates the two independent approximations**
that contribute.

**Why it is not obvious.** The solver is exact in GEOMETRY (Δg = FᵀF − ḡ) but its energy is a
quadratic form in Δg. Since q_e·Δg = ℓ′² − ℓ² *exactly*,

```
    U_sol = ½ Σ_e k_e (δℓ_e)² · ((ℓ′_e+ℓ_e)/2ℓ_e)²    vs    U_hooke = ½ Σ_e k_e (δℓ_e)²
```

so it differs from a real Hookean spring by ((ℓ′+ℓ)/2ℓ)² ≈ 1 + δℓ/ℓ — **first order in the strain,
not second.** That is a *constitutive* difference and it is invisible to every tangent check, because
it vanishes at λ→0.

**Method.** `verification_tools/finite_amplitude_check.py`. N=6 (72 tri), η=0.25, VD a=3. Three
relaxed energies of the SAME network under the SAME macro deformation F(λ), all with **exact
kinematics** (true node positions, L-BFGS over free nodal DOF):

| | energy | what it is |
|---|---|---|
| **U1** | ½ Σ k (ℓ′ − ℓ₀)² | real **linear Hookean springs** — the physical reference |
| **U2** | ½ Σ (k/4ℓ²)(ℓ′² − ℓ²)² | the solver's **energy functional**, relaxed over the same DOF |
| **U3** | ½ (Δg/2) : C_eff : (Δg/2) · A_tot | the solver's actual **prediction** — its linear response extrapolated to λ |

Then `U2 − U1` is the **constitutive** error alone (same relaxation, same kinematics, different
energy); `U3 − U2` is the **response-linearisation** error alone (same energy, linear vs relaxed
response); `U3 − U1` is the **total**, which is what a user of the solver actually incurs.

Terminology (`CLAUDE.md` §3): every model here has a LINEAR Hookean constitutive law except U2, whose
energy is quadratic in Δg. "Nonlinear" always means geometric/kinematic.

**Two convention factors in U3 were CALIBRATED on the η=0 crystal, not assumed** — this is the
A-0 class of bookkeeping and is not trusted to algebra alone:
(i) vec3 carries shear once but Δg:C:Δg counts the xy slot four times ⇒ `SH = diag(1,2,1)`;
(ii) `physical_units=True` returns C in the standard ½ε:C:ε convention and Δg = 2ε to leading order
⇒ contract with Δg/2. With both applied the residual ratio is **mode-independent** (uniaxial
4.618379 vs shear 4.618802 before (ii); exactly 1/4 after), and the crystal returns ν=1/3,
E=1.1547=2/√3. Δg itself is kept geometrically exact — only the *response* is linearised.

---

## Results

| mode | λ | constitutive `U2−U1` | response-lin `U3−U2` | **total `U3−U1`** |
|---|---|---|---|---|
| uniaxial x | 1e-3 | 1.23e-03 | 7.66e-04 | 4.69e-04 |
| uniaxial x | 1e-2 | 1.24e-02 | 7.65e-03 | 4.72e-03 |
| uniaxial x | **1e-1** | **1.25e-01** | **7.56e-02** | **4.98e-02** |
| uniaxial x | 2e-1 | 2.52e-01 | 1.49e-01 | 1.03e-01 |
| simple shear | 1e-3 | 9.94e-05 | 1.08e-04 | 8.93e-06 |
| simple shear | 1e-2 | 1.06e-03 | 1.08e-03 | 1.81e-05 |
| simple shear | **1e-1** | **1.77e-02** | **1.08e-02** | **6.91e-03** |
| simple shear | 2e-1 | 5.30e-02 | 2.30e-02 | 2.99e-02 |

(all relative to U1; figure `finite_amplitude.png`)

**1. Everything is first order in λ, as predicted.** Over λ = 1e-3 → 2e-1 (×200) the constitutive
error goes 1.23e-03 → 2.52e-01 (×204). Slope 1, not 2. The solver's constitutive approximation is
**not** a second-order effect that can be waved away — it enters at the same order as the strain.

**2. Both approximations vanish at λ→0**, confirming all three models share a tangent modulus — which
is why every existing tangent check is blind to this, and is also the correctness check on the
calibration above.

**3. The two errors partially CANCEL.** The total (4.98e-02 uniaxial at λ=0.1) is *smaller than
either* contributing term (1.25e-01 and 7.56e-02). The solver's stiffer constitutive law and its
unrelaxed linear response push the energy in opposite directions. This is a real effect, not luck of
one mesh — it holds at every λ and in both modes — but it is **not** a guarantee: it is a property of
this network family and should not be assumed to transfer.

**4. Practical numbers.** At **10% strain** the solver's total energy error vs real linear springs is
**~5.0% uniaxial, ~0.7% shear**; at 20%, ~10% and ~3.0%. Shear is consistently the milder channel.

## Why this matters beyond a number

The constitutive difference is the **only** one of the two that survives into the settings the
project is heading for. For **residual stress / incompatible ḡ** the reference state is prestressed
(T ≠ 0), the tangent stiffness picks up the geometric term (T/ℓ)(I − R̂⊗R̂), and the ((ℓ′+ℓ)/2ℓ)²
factor no longer sits at a stress-free reference. The current oracle cannot check any of that — it
assembles from a stress-free ḡ = I (`CLAUDE.md` §3, FUTURE_DIRECTIONS #1), and `assemble_K_faff`
takes no rest-length argument (audit **A-15**). This experiment is the tangent-free baseline that
work will have to be measured against.

## Limitations

- **One network** (N=6, η=0.25, VD a=3), one seed, two deformation modes. No family sweep, no size
  study — the cancellation in (3) especially needs more meshes before being relied on.
- Energies only. Per-triangle finite-amplitude fields, and ν/E at finite amplitude, are not measured.
- U1 is linear Hookean springs with exact kinematics — a *model*, not an experiment. It is the right
  reference for isolating the constitutive term, but it is not "reality" either.
- λ ≤ 0.2. Buckling / contact / bond inversion are not probed and would break U1 first.
- Flat, compatible, PBC only — same envelope as the rest of the verified stack.
