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

## 5. The metric description IS complete — the only error is the zero-mean constraint

The cluster works in node (configuration) space. The open question was whether the
**metric** description can be made exact by imposing the right conditions on `W`, since
configuration and metric descriptions should be equivalent whenever the Gauss–Codazzi
(here just Gauss, i.e. discrete curvature) constraints hold. The answer is **yes**, and the
single thing that was breaking it is the normalisation constraint `⟨δg⟩ = 0`.

### 5.1 range(B) and ker(C)
Work in the space of per-triangle metric fields (`3·n_tri` numbers: `δg11, δg12, δg22` per
triangle).
- **`B`** maps a node-displacement field `u` to the per-triangle non-affine metric change it
  induces, `δg = B u` (with `δg_s = 2 sym(δU_s · E_ref,s⁻¹)`). So **`range(B)` = all metric
  fields that some actual node motion can produce** — the *compatible / realisable* fields.
  A field outside `range(B)` cannot be drawn as a deformed mesh: it carries discrete
  Gaussian curvature (disclinations), `inc(δg) ≠ 0`.
- **`C = [edge ; curvature ; mean]`** is the stack of constraint operators imposed in the
  metric-KKT route; **`ker(C)` = all metric fields satisfying every imposed constraint** —
  the subspace the KKT solve is confined to. The goal was `ker(C) = range(B)`.

The earlier rank test found `dim ker(C) = dim range(B)` (both 126 at `N=8`) — **same
dimension, but not the same subspace**. The mean rows tilt `ker(C)` away from `range(B)`.

### 5.2 Isolating the culprit (test_mean_isolation.py)
Change exactly one thing between two solves on the *same* compatible subspace
(`δg = B u`), with the *same* loading (the simulation's affine spring force):
- **V_A (the fix):** minimise the per-triangle metric energy `½ Σ_s δg_sᵀ H_s δg_s` over
  `range(B)`, **no mean constraint**. `H_s = Σ_edges (1/4l²) q qᵀ`, `q=[vx²,2vxvy,vy²]`.
- **V_B:** identical, but add only `(Mean·B) u = 0` via a Lagrange multiplier.

Result on the stored `N=40` data (one trial per η):

| η | V_A corr / overshoot | V_B (+mean) corr | V_B, spatial mean removed | sim ‖⟨δg⟩‖ |
|---|---|---|---|---|
| 0.1 | **0.9997 / 1.000** | 0.759 | 0.759 | 8.3e-2 |
| 0.2 | **0.9999 / 1.000** | 0.688 | 0.688 | 3.8e-1 |
| 0.3 | **0.9998 / 1.000** | 0.456 | 0.456 | 1.2e0 |
| 0.4 | **0.9999 / 1.000** | −0.300 | −0.300 | 5.1e0 |
| 0.5 | **1.0000 / 1.000** | 0.652 | 0.652 | 4.9e1 |

So **V_A reproduces the simulation exactly at every η, including large η** (corr ≈ 1,
overshoot 1.000). Adding *only* `⟨δg⟩ = 0` collapses it; and removing the spatial mean
afterward does **not** recover it (V_B with mean removed ≈ V_B) — the damage is not a
uniform offset, the constraint forces an entirely different compatible field. We are
therefore certain the zero-mean/normalisation condition is the failure.

### 5.3 How V_A is implemented *without* the mean — and where the loading comes from
The original single-site derivation (§1) *needed* `⟨δg⟩ = 0`: eliminating its multiplier is
what produced `χ = −⟨A⟩Δg` (every triangle carries the mean stress) and hence
`δg = −A⁻¹δA·Δg`. V_A is **not** that solve. It does not embed a triangle in an average
medium and never introduces a mean multiplier. Instead it is the *actual* global
equilibrium written in metric space:
- the field is parametrised as `δg = B u` (compatible by construction), and
- the macroscopic load enters as a **forcing term** — the affine residual spring force
  `f_aff` — not as a constraint on the metric average.

Concretely V_A solves `K_m u = −2 f_aff` on the free DOF, with `K_m = Bᵀ blkdiag(H_s) B`;
on the torus `K_m = 2K` so this is identical to the node-space equilibrium
`K u = −f_aff`. **The macroscopic state is fixed by the displacement gradient / periodic
boundary (a configuration quantity), not by the metric mean.** That is the whole point: the
correct macroscopic condition lives on the *gradient* `F` (periodicity), and `⟨δg⟩` is then
free to be whatever it is.

### 5.4 "But how do you enforce that ⟨δg⟩ ≠ 0? Is a self-consistent calculation needed?"
We enforce **nothing** on `⟨δg⟩`. We enforce (a) compatibility (`δg ∈ range B`) and (b) the
periodic/gradient loading. `⟨δg⟩` comes out nonzero on its own. No self-consistent iteration
is required for V_A — it is a single linear solve (and the cluster is its local truncation).

The "self-consistency" the single-site MF was chasing (average-medium closure) is the wrong
closure. If one insisted on staying in a *metric-only*, per-triangle-`W` language, the
correct closure is **not** `⟨δg⟩ = 0` but the nonlinear area/`det` identity of §5.5, which
couples triangles at second order. Working in node space (cluster, or V_A) makes that
closure automatic, which is why no iteration is needed.

### 5.5 The area question — it IS the area requirement, truncated to first order
Claim to check (user): the area computed from the global affine metric must equal the sum of
the per-triangle deformed areas — is that just the mean constraint, or is it an
expansion needing higher order? **Numerically the exact identity holds:**

`Σ_s A_ref,s · det(F_s) = det(F) · A_ref`  (rel. error 0 to η=0.4, 1.5e-4 at η=0.5),

where `det(F_s) = √det(g_def,s) = √det(I + Δg + δg_s)`. This is a genuine conservation law —
total area is set by `det(F)` (the periodic boundary), however the non-affine fluctuation
redistributes it among triangles.

Now expand `det(F_s)` in the strain:
`det(F_s) = √det(I + Δg + δg_s) ≈ 1 + ½ tr(Δg + δg_s) + O(strain²)`.
Summing with weights `A_ref,s` and using `det(F)·A_ref ≈ A_ref(1 + ½tr Δg)`:

  **first order ⇒ `Σ_s A_ref,s tr(δg_s) = 0`, i.e. the (trace part of the) `⟨δg⟩ = 0`
  constraint.**

So the linear zero-mean constraint is *exactly the first-order truncation of the exact area
requirement*. But the metric `δg` the simulation actually produces is the full nonlinear
object, whose area-weighted trace mean is the **second-order** term. Verified: the
area-weighted `⟨tr δg⟩` scales as `δ²` (strain amplitude `δ=1e-3`): ratio `⟨tr δg⟩/δ²` =
0.08, 0.35, 1.0, 3.7 for η = 0.1–0.4 (O(1)·δ², genuinely second order; η=0.5 leaves the
small-strain regime).

**Resolution (your last clause is correct):** the two area framings are the same — it is the
area requirement — *and* `⟨δg⟩ ≠ 0` precisely because the linear theory truncates an
expansion. The single-site MF is a linear-response theory, so it imposes the first-order
form `⟨δg⟩ = 0`, which **over-constrains** the true nonlinear field and is what breaks it.
Imposing periodicity on `F` (V_A / cluster) keeps the exact nonlinear `det(F_s)` law without
ever truncating, so it is correct to all orders.

### 5.6 The correct normalisation is AREA-WEIGHTED, and it is *implied by compatibility*
The MF imposes the **plain** mean `Σ_s δg_s = 0`. The simulation does not satisfy it. It
satisfies the **area-weighted** mean `Σ_s A_s δg_s = 0`. Measured on `dg_sim`
(typical `|δg| ~ δ = 1e-3`):

| η | ‖Σ δg_s‖ (plain) | ‖Σ A_s δg_s‖ (area) | rms‖δg‖ |
|---|---|---|---|
| 0.1 | 2.6e-5 | **5.8e-8** | 2.4e-4 |
| 0.2 | 1.2e-4 | **2.5e-7** | 5.3e-4 |
| 0.3 | 3.7e-4 | **7.2e-7** | 1.1e-3 |
| 0.4 | 1.6e-3 | **2.6e-6** | 4.0e-3 |
| 0.5 | 1.5e-2 | 2.1e-4 | 3.4e-1 |

The plain mean is *comparable to the signal* at high η; the area-weighted mean is 3–4 orders
**below** it (scaling as `δ²`). So the area-weighted mean vanishes; the plain one does not.

Why area-weighting is the right one — it is a discrete **divergence-theorem identity**, not a
modelling choice. As an operator on node displacements,
```
‖ (area-weighted mean) · B ‖ / ‖B‖ = 1.6e-16   (η≤0.3; 3e-3 at η=0.5)
‖ (plain mean)         · B ‖ / ‖B‖ = 0.19 … 1.12
```
i.e. **`Σ_s A_s B_s ≡ 0`**: the area-weighted average of the (linearised) strain over a
periodic cell is identically zero for *every* periodic displacement field
(`Σ_s A_s ε_s = sym Σ_s ∮ u⊗n = 0`). Compatibility *implies* the area-weighted mean — it is
redundant. The plain mean is **not** implied by compatibility, so adding it injects a
constraint inconsistent with the realisable fields.

Consequence, imposing each as a hard constraint on the otherwise-exact compatible solve
(corr vs sim):

| η | V_A (no mean) | + plain mean | + area-weighted mean |
|---|---|---|---|
| 0.2 | 0.9999 | 0.688 | **0.9998** |
| 0.3 | 0.9998 | 0.456 | **0.9983** |
| 0.4 | 0.9999 | −0.300 | **0.9994** |

The area-weighted mean is harmless (redundant); the plain mean is destructive. (At η=0.5 the
linear identity itself degrades — the `3e-3` residual — because the strain is no longer small
vs the disorder; that is the `δ²` term of §5.5 becoming visible.)

### 5.7 The intrinsic (configuration-free) derivation of the Hessian
This makes the metric-only theory precise — **no node displacements needed**. Minimise the
metric energy `½ Σ_s (Δg + δg_s)ᵀ H_s (Δg + δg_s)` over the per-triangle field `δg`, subject
to the **intrinsic** constraints
1. edge agreement,
2. **compatibility / zero discrete curvature** `inc(δg)=0` (the vertex angle-deficit
   operator — enforced by a stress-function/Airy multiplier, all in metric space), and
3. **area-weighted** normalisation `Σ_s A_s δg_s = 0`,

and **NOT** the plain mean. `ker([edge; curvature])` is exactly
`range(B) ⊕ {3 homogeneous modes}`; the 3 homogeneous modes are the macroscopic strain (the
loading), and the area-weighted normalisation removes precisely them, projecting onto
`range(B)` — so this intrinsic solve equals V_A equals the simulation, **to first order**.
The single-site MF differs by exactly two wrong choices: it *drops* (2) and it uses the plain
form of (3).

Everything is one **expansion**, not a free lunch: the exact closure is the nonlinear area
law `Σ_s A_s det(F_s) = det(F) A_ref` (§5.5). Its `O(δ)` term is the area-weighted mean
`Σ A_s δg_s = 0` (automatically satisfied by compatible fields); its `O(δ²)` term is the
nonzero remainder. The effective Hessian `C_eff = ⟨C_bare,s · W_s⟩` falls out order by order
from this expansion with the correct (area-weighted, compatibility-consistent) closure; it is
*not* obtained for free, and it does not require passing through the configuration.
