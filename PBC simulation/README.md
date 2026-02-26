# Phase 4 — Periodic Boundary Condition (PBC) Simulation

## Motivation

Phase 3's validation revealed that the D2C analytical model and the KUBC simulation disagree significantly for heterogeneous-rigidity networks (50-70% error on Poisson ratio). Both methods have known limitations:

- **D2C analytical**: mean-field approximation ignores spatial correlations between neighbouring triangles.
- **KUBC simulation**: 38% of nodes are boundary-constrained, artificially stiffening the response.

A **PBC simulation** eliminates the boundary artifact entirely and directly matches the analytical model's assumption of an infinite periodic medium. This provides the fairest possible ground truth for evaluating the analytical method.

## Method

### Periodic boundary conditions on a triangular lattice

The key idea: the unit cell tiles the plane. Nodes on opposite edges of the bounding box are **paired**, and their displacements are constrained to differ by exactly the macroscopic strain:

```
u(x + L_x) = u(x) + eps . L_x_vec
u(y + L_y) = u(y) + eps . L_y_vec
```

where `eps` is the applied macroscopic strain tensor and `L_x_vec`, `L_y_vec` are the periodicity vectors.

### Implementation steps

1. **Generate a periodic mesh**: use the regular triangular lattice (already periodic by construction). For deformed meshes, perturb interior nodes only, or perturb all and then average boundary pairs to maintain periodicity.

2. **Identify periodic node pairs**: match nodes on left-right and top-bottom boundaries.

3. **Reduce DOFs**: eliminate one node from each pair (the "slave") and express its displacement in terms of the "master" node plus the macroscopic strain contribution.

4. **Minimise energy**: same spring energy as KUBC, but with the reduced DOF set. The only imposed displacements are the macroscopic strain — all non-affine relaxation is permitted.

5. **Extract stiffness**: same 6-test protocol as KUBC (exx, eyy, exy, and 3 combinations), but with PBC instead.

## Files

| File | Description |
|------|-------------|
| `pbc_simulation.py` | PBC mechanical simulation with periodic node pairing, constrained energy minimisation, and full Voigt stiffness extraction. Runs the same 4 configs (A-D) as the KUBC simulation for direct comparison. |
| `README.md` | This file. |

## Expected outcome

If the analytical model is correct for an infinite periodic medium, configs A and C should match to <1%. The key test is configs B and D — if PBC simulation and D2C still disagree by >20%, the analytical mean-field approximation is genuinely insufficient and a better theory is needed.
