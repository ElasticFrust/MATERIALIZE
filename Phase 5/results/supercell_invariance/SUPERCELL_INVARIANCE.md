# S1 premise — is `C_eff` really INTENSIVE across supercells?

**What.** `M2_V2_PLAN.md` §3.1e proposes generating M2's training data as small **unit cells with
bases**, on the claim that for a crystal `C_eff` from the minimal cell **equals** `C_eff` from any
supercell of it. §3.1e also turns that claim into a **free, label-less gate** for the model (predict
on the 1×1 cell and on its 3×3 supercell; the answers must be identical). Both rest on a premise
that had never been measured.

**Why it came before any GNN code.** The premise needs **no model** to test. If the *solver* did not
reproduce `C` across supercells then "train small, deploy large" is unsound and the supercell gate is
meaningless — so it is checked first, at a cost of about a minute.

**Producer.** `Phase 5/verifications/supercell_invariance.py` · provenance in `run_provenance.txt` ·
verbatim stdout in `supercell.txt`.

---

## 1. Method

For each crystal family, build **the same crystal** at several cell sizes and compare the solver's
`C_eff`, ν and E. A perfect crystal at η = 0 must give identical numbers. The reported quantity is

```
    rel = max|C6(cell) - C6(first cell)| / max|C6(first cell)|
```

Four chains, chosen so the test is not degenerate:

| chain | why it is in the set |
|---|---|
| triangular, uniform k | the baseline; but ν = 1/3 could come out right *by accident* |
| **anisotropic ψ = 0.6, uniform k** | **the sharp one** — `C` is not a multiple of the identity, so an accidental match is not available |
| honeycomb tiling | soft fictional bonds ⇒ tests invariance under **large k contrast** (E ≈ 2.6e-03, i.e. near-mechanism) |
| kagome tiling | same, at a different coordination |

## 2. Result — invariant to machine precision

```
triangular crystal, uniform k (isotropic)
  cell        n_tri           nu            E   max|C6 - C6(first)|/|C6|
  half=2         32   0.33333333   1.15470054   0.000e+00
  half=3         72   0.33333333   1.15470054   1.709e-16
  half=4        160   0.33333333   1.15470054   1.709e-16
  half=6        336   0.33333333   1.15470054   5.128e-16

anisotropic crystal psi=0.6, uniform k  (SHARPER: C is not isotropic)
  cell        n_tri           nu            E   max|C6 - C6(first)|/|C6|
  half=2         64   0.56761413   1.17976381   0.000e+00
  half=3        144   0.56761413   1.17976381   3.721e-16
  half=4        256   0.56761413   1.17976381   9.302e-16
  half=6        576   0.56761413   1.17976381   7.441e-16

honeycomb tiling (soft fictional bonds -> tests under large k contrast)
  reps=2         48   0.99552128   0.00259096   0.000e+00
  reps=3        108   0.99552128   0.00259096   7.678e-14
  reps=4        192   0.99552128   0.00259096   1.344e-13

kagome tiling (soft fictional bonds -> tests under large k contrast)
  reps=2         64   0.33333333   0.57792762   0.000e+00
  reps=3        144   0.33333333   0.57792762   2.167e-13
  reps=4        256   0.33333333   0.57792762   6.781e-13
```

**ν and E are identical to all eight printed digits on every chain**, across a 10.5× growth in
triangle count. The residual grows only mildly with size (1.7e-16 → 5.1e-16; 7.7e-14 → 1.3e-13),
which is round-off accumulating in a larger mean, not a systematic drift. The tilings sit three
orders higher than the crystals because their soft fictional bonds put `A(s)` at large contrast —
still 1e-13, and still far below anything that matters.

## 3. What this establishes

1. **"Train small, deploy large" is sound** for the crystalline part of the space. The unit-cell
   generator of §3.1e samples the same physics the large cells have.
2. **The supercell gate is a real, label-free test of the model** — the physics satisfies it to
   1e-13, so any deviation a model shows is the model's, not the target's.
3. **Independent confirmation of the §3.1c pooling defect.** v1 pools global `mean + sum`; the `sum`
   branch is **extensive**, so on the triangular chain alone (32 → 336 triangles) its output would
   move by ~10.5× where the physics moves by 5e-16. A `sum` branch cannot satisfy a property the
   physics obeys exactly — which is why the v2 head assembles `C_eff = (1/N) Σ_s C(s)`, a mean, and
   satisfies it by construction.

## 4. An incidental cross-check (observation, not a gate)

The honeycomb tiling returns **ν = 0.99552**, against the **analytic** regular-honeycomb value
**ν = +1** from `test_hex_closed_form`'s closed form ν(r) = (4r²−1)/(3+4r−4r²) at d = 2. The 0.45 %
shortfall has the right sign and magnitude for the finite-hinge offset that gate already records
(~0.3–2.7 % at the designer's default, **first order in `k_spoke`**), i.e. it is the model
difference, not solver error.

**Stated as an observation, not a verified match:** the two constructions differ (a periodic tiling
with soft fictional bonds here, a single open hexagon with a rigid perimeter there), so the numbers
are not required to coincide. Making this quantitative — sweeping the tiling's fictional-bond
stiffness and checking the residual is first order, as the hexagon gate does — would turn it into a
third-path check on the tiling representation. Not done; noted as available.

## 5. Limitations

- **Crystals at η = 0 only.** That is the premise's own domain — "supercell" is only defined where
  there is a lattice to repeat — but it means nothing here transfers to disordered cells, where the
  §3.1c size-generalisation question is open and must be measured on the model.
- **This is the SOLVER's intensivity, not the sim's.** The independent oracle was not run; the claim
  under test is a property of the map M2 will be trained on, so the solver is the right instrument.
  If the solver were wrong in a size-dependent way this test would not see it — but `C_eff`'s
  agreement with the oracle is gated elsewhere (`test_forward_solver` [7]/[8]).
- **It does not test a model's size generalisation.** A GNN with a finite receptive field can still
  fail at large cells while the physics is exactly invariant (§3.1c's screening argument). This
  removes one explanation for such a failure; it does not remove the failure mode.
- Four chains, one k-pattern each (uniform, or the tiling's native `k0`). Invariance under a
  *spatially structured* k field that must itself be replicated coherently across the supercell was
  not tested.
