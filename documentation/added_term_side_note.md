# Side note — the "added term" alternative (δg = WΔg + δg⁰)

**Status: NOT used. Parked for a separate discussion.** The active formulation keeps the single ansatz
`δg_s = W_s Δg` (no added term). This note only records how things would differ if we added a
Δg-independent prestress field.

## The alternative
`δg_s = W_s:Δg + δg⁰_s`, with `δg⁰_s` a **Δg-independent** prestress fluctuation solving, on the same
operator/constraints,
`A_s:(δg⁰_s − δḡ_s) + [Cᵀκ]_s = 0`  (RHS `= +A_s:δḡ_s`).

## What changes
- **The response `W` — hence `C_eff, ν, E` — is identical.** `δg⁰` is Δg-independent, so it never enters
  the `W` solve. The two formulations differ *only* in the zero-applied-load (prestress) sector.
- **Prestress readout differs:**
  - **Current (no term):** at `Δg=0`, `δg_s=0`, so the prestress is the **raw, unrelaxed** mismatch
    `σ⁰_s = −A_s:δḡ_s`.
  - **Alternative (+δg⁰):** the fluctuation relaxes even at zero load, giving the **relaxed** residual
    stress `σ⁰_s = A_s:(δg⁰_s − δḡ_s)` — the network minimizes energy over the compatible fluctuation, so
    only the **incompatible** part of `δḡ` survives.
- **Compatible references (e.g. a uniform δḡ):** with `+δg⁰` the prestress would relax to ≈0 (compatible ⇒
  no residual stress); the current form instead reports a nonzero raw `−A δḡ`. So the added term is exactly
  what separates **relaxable (compatible)** from **frustrated (incompatible)** references — the
  relaxed-vs-unrelaxed distinction.

## Why it's worth a later look
For the finite-size / genuinely-frustrated cases (sphere cap, disclination), the *relaxed* residual stress
is the physical one, so the added term (or an equivalent relaxation step) is likely needed there — whereas
for the uniform-δḡ validation test it is deliberately left out to keep the equation identical to what we
documented and to isolate the raw source behaviour.
