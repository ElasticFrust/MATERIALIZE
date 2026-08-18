# reentrant — the hexagon diameter family (re-entrant honeycomb by construction)

**Scripts** `Phase 5/verifications/dhex_family.py`, `reentrant_family.py`, `reentrant_move.py`

## What

Re-entrant honeycomb built by the **diameter algorithm** rather than by re-triangulating: a
triangular lattice viewed as hexagons each with a centre vertex, perimeter edges HARD (length 1,
held fixed so every hexagon keeps perimeter 6), radial centre→rim edges SOFT. One diameter is
squeezed from d = 2 (regular hexagon) down toward 0 (bow-tie) or out toward 3 (flat). **Topology is
fixed** — only the vertices move — so this isolates geometry from connectivity.

## Key numbers (solver, `dhex_family.py`, 96 nodes / 288 bonds / 192 soft)

| d | regime | ν | E |
|---|---|---|---|
| 2.00 | regular | +0.996 | 0.003 |
| 1.60 | squeezed | +1.367 | 0.005 |
| 1.20 | squeezed | +4.143 | 0.042 |
| 1.00 | squeezed | 0.000 | 0.501 |
| 0.80 | re-entrant | −5.040 | 0.051 |
| 0.50 | re-entrant | −2.557 | 0.011 |
| 0.30 | re-entrant | −2.040 | 0.007 |

d < 1 is auxetic, d = 1 is the sign change, d = 2 recovers the textbook honeycomb ν ≈ +1. These
match the closed form validated in [`../hex_validation/HEX_VALIDATION.md`](../hex_validation/HEX_VALIDATION.md).

## Limitations

- **E is tiny throughout** (0.003–0.5): the structure is soft because the radial spokes are soft by
  construction. These are near-mechanism networks, exactly the regime `CLAUDE.md` §3 flags for
  `A(s)` rank loss — the closed-form agreement is what justifies trusting them here.
- ν is large in magnitude (up to ±5) near d ≈ 1.2 and 0.8, where the response is stiffest against
  the soft mode; those points are the most sensitive to `k_spoke`.
- Fixed topology by design — this says nothing about what re-triangulation would reach.

## Figures

- [`dhex_family.png`](dhex_family.png) — the d-sweep, 3×3 tiled and cropped
- [`reentrant_family.png`](reentrant_family.png) · [`reentrant_by_move.png`](reentrant_by_move.png)
