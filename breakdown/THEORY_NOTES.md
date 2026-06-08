# Why the single-site mean field fails, and why a cluster fixes it

Working notes (June 2026 session). Companion to `derivation_edge_compatibility.pdf` and the
diagnostics in `breakdown/` (`pbc_dg_analysis.py`, `compat_projection.py`,
`analyze_overshoot.py`, `test_cluster_response.py`, `test_cluster_rigidity.py`).

## 1. What the derivation actually computes

The D2C mean field minimises the elastic energy
`E = ½ Σ_s A(s) (Δg + δg(s))²`
over the per-triangle non-affine metric changes `δg(s)`, subject to only two constraints:
- global normalisation `⟨δg⟩ = 0`, and
- **edge-length agreement** between neighbouring triangles (the edge-KKT constraint).

Eliminating the normalisation multiplier gives `χ = −⟨A⟩Δg`, i.e. every triangle is forced
to carry the **mean stress** `⟨A⟩Δg`, so (without the edge term)
`A(s)(Δg + δg(s)) = ⟨A⟩Δg  ⇒  δg(s) = −A(s)⁻¹ δA(s) Δg`.
This is a **uniform-stress (Reuss-type), single-site mean field**. The `B = ⟨δA·⟩/N`
operator is the mean-field self-consistency, solved in one linear pass by Woodbury
`(A−B)⁻¹` (it is self-consistent, but *linearly* — no iteration is needed).

## 2. Where it is approximate (it is the THEORY, not the code)

The implementation faithfully solves the equations above. The approximation is in the
**space being minimised over**:

- variables: `3N ≈ 6n` per-triangle metric components;
- minus edge agreement (`≈3n`) and normalisation (`3`)  ⇒  `≈ 3n − 3` admissible DOF;
- but the **physical** system has only `2n` DOF (the node displacements).

So the minimisation runs over a space **larger by `≈ n` dimensions** than is physically
realisable. Those extra dimensions are **incompatible metric fields**: every shared edge
length agrees, yet the triangles cannot be assembled into an actual flat sheet (the
discrete Gaussian curvature / vertex angle-sums do not close). No node configuration can
produce them — but the mean field is allowed to use them.

**Minimising energy over a strictly larger set gives a lower energy and a more compliant
response.** That is the systematic **≈1.4× overshoot** of `δg`. On the torus this is exact:
`½ Σ_s A(s)(strain_s)²` is precisely twice the true spring energy (each edge shared by two
triangles, counted once per triangle), so minimising the *same* energy over the
**compatible** (node-realisable) subspace would recover the simulation exactly. The
overshoot is entirely the extra incompatible DOF.

Two equivalent statements of the same deficiency:
- **kinematic:** the admissible field space is too big (edge agreement ⊊ full
  compatibility; the missing conditions are the per-vertex curvature/St-Venant ones);
- **mechanical:** each triangle is embedded in the *global-average* medium `⟨A⟩` (single
  site), not in its *actual* neighbours.

The overshoot is *not* a compatibility-projection artefact (projecting the MF field onto
compatible fields removes only ~10% of it) — it is the uniform-stress over-compliance.
Edge-KKT and even full edge+angle (vertex) KKT only shave ~0.1 off the 1.4 and the angle
block is numerically fragile (near-degenerate triangles ⇒ singular constraint Gram).

## 3. Why a local cluster fixes it

Relaxing a small node patch around each triangle (boundary held affine, the *actual* spring
physics) and reading the central triangle's `δg`:
- works in **node-displacement space ⇒ the field is automatically compatible** (no
  incompatible DOF can appear), and
- uses the **actual local neighbours' stiffness/geometry**, not the average.

It therefore cures both faces of the deficiency at once. Empirically (vs the PBC sim,
per-triangle `δg`):
- **geometric disorder:** even radius **d=1** gives corr 0.97–0.98 and overshoot ≈1.0 at
  *all* η (incl. η=0.5, where single-site MF decorrelates to −0.07). The response is
  ultra-local (the over-constrained z=6 lattice screens perturbations in ~1 hop).
- **rigidity contrast (regular lattice):** the cluster fixes the overshoot immediately
  (≈1.0 at d=1) and recovers direction as the radius grows; strong contrast needs larger
  clusters (corr at 100× contrast: d=1→0.81, d=6→0.95) because soft channels / stiff
  backbones have a longer correlation length.

**Unifying statement:** the required cluster radius ≈ the disorder correlation length —
~1 for geometric disorder, ~4–6 for strong rigidity contrast. The single-site MF is the
`radius→0` limit embedded in the average medium, which is exactly why it fails worst where
the correlation length is longest.

## 4. Consequences

- The fix is not a renormalisation knob (`÷1.4` is unprincipled and contrast-dependent) and
  not "more self-consistency iterations" — it is **the right space (compatible) and the
  right environment (actual local neighbours)**.
- A cluster forward model is a set of small, **non-iterative, loading-independent, local**
  linear solves (one per triangle, embarrassingly parallel) — see `Q1` below.
