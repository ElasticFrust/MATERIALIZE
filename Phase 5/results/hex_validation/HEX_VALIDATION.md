# hex_validation — the solver against a CLOSED-FORM solution

**Script** `Phase 5/verifications/hex_nu_linear.py`, `hex_response_plot.py` · gated by
`Phase 5/verifications/test_hex_closed_form.py`

## What

The only check in the repo with an **analytic** reference. Everything else compares two codes
(solver vs the independent sim); this compares the solver against a closed form, so it is the
strongest rung of the verification ladder.

A single hexagon — 7 nodes, 6 triangles, **open** (no periodicity) — with a rigid perimeter (six
unit edges) and free hinges. With `d` the separation of two opposite vertices and `r = d/2`:

$$\nu(r) = \frac{4r^2-1}{3+4r-4r^2}$$

**Derivation.** $|V_0V_1| = 1$ gives $y^2 = 1-(r-\tfrac12)^2$. Along the single soft mode the cell
dimensions are $L_x = 1+2r$ and $L_y = 2y$; differentiating gives the result. **The subtle step is
$L_x$** — it is the *lattice constant* $1+2r$, not the hexagon's own width $2r$; using the width
predicts $\nu = 2/3$ at the regular hexagon instead of $1$.

## Method · key numbers

| check | result |
|---|---|
| free-hinge solver vs closed form, `d ∈ [0.05, 2]`, 100 pts | **max \|Δν\| = 4.41e-06** |
| solver (default `k_spoke=1e-3`) vs the independent sim | **2.95e-11** |
| landmarks | ν(½) = −0.2 (re-entrant), ν(1) = 0, ν(2) = **+1** (regular hexagon, isotropic) |
| residual vs hinge stiffness | first order: 4.30e-02 → 4.48e-03 → 4.50e-04 for k = 1e-2/1e-3/1e-4 |

The first-order test matters as much as the agreement: it proves the ~0.3–2.7 % offset at the
designer's default `k_spoke` is the **model difference** (free vs finite hinges), not solver error.

**This gate is sensitive to the shear channel** — ν=1 at the regular hexagon depends on `C_xyxy` —
so it would have caught **A-0** (over-stiff by 29–90 %) immediately.

## Limitations

- Open mesh, single cell: it exercises the homogenisation on 6 triangles, not a disordered bulk.
- The closed form assumes a *rigid* perimeter and *free* hinges; the model has finite `k_spoke`, so
  the comparison is made in the limit (1e-8) and the residual checked for first-order scaling.
- ν diverges as d → 3, so the linear-axis figure stops at d = 2.

## Figures

- [`hex_nu_linear.png`](hex_nu_linear.png) — ν(d) on linear axes, d ≤ 2: closed form, free-hinge
  solver, default-stiffness solver, independent sim + the relative-deviation panel
- [`hex_response.png`](hex_response.png) — single hexagon vs the periodic lattice
