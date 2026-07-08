# The intrinsic metric-space forward solve — equations and constraints

Working notes, June 2026. Notation follows **Grossman & Boudaoud, PRR 2026
(arXiv:2309.07844)** and the companion `derivation_edge_compatibility.pdf`; new symbols are
introduced only where the paper has none, and are flagged as such. Implemented and verified in
`verification_tools/test_intrinsic_metric.py` (with `test_mean_isolation.py`, `test_curvature_operator.py`).

The aim of this note is to state **exactly** which energy is minimised, which constraints are
enforced, and how this differs from G&B — entirely in the metric (incompatible-elasticity)
description, **without passing through node displacements**.

---

## 1. Notation (G&B, plus three additions)

From G&B (unchanged):

- `n` vertices (2n positional DOF), `N` triangles, `E_int` interior edges, `n_int` interior
  vertices.
- Each triangle `s` carries a symmetric 2×2 metric `g(s)_{μν}`, split as
  $$g(s)_{\mu\nu} = \bar g_{\mu\nu} + \Delta g_{\mu\nu} + \delta g(s)_{\mu\nu},$$
  with `ḡ` the reference (rest) metric, `Δg` the uniform **affine** (macroscopic) loading, and
  `δg(s)` the **non-affine** fluctuation. Here `ḡ = I`.
- Linear response: `δg(s)_{μν} = W(s)^{αβ}_{μν} Δg_{αβ}` (the response tensor `W`, 9
  components/triangle).
- Per-triangle elastic stiffness `A(s)^{μναβ}` (major symmetry `A^{μναβ}=A^{αβμν}`).
  **In this network `A(s)` is the spring-energy metric Hessian** (no area prefactor):
  $$A(s)^{\mu\nu\alpha\beta}\ \longleftrightarrow\ H_s = \sum_{e\in s}\frac{k_e}{4\,\ell_e^2}\,q_e\,q_e^{\!\top},\qquad
  q_e^{\mu\nu}=\Delta x_e^{\mu}\,\Delta x_e^{\nu},$$
  where the sum is over the triangle's three reference edges `Δx_e` of length `ℓ_e` and stiffness
  `k_e`. (`q_e` is G&B's rank-1 edge carrier.)
- `δA(s) := A(s) − ⟨A⟩`, `⟨A⟩ = N⁻¹ Σ_s A(s)`.
- `J`: the `E_int × 3N` edge-compatibility matrix; row `e=(s1,s2,Δx_e)` carries `+q_e^⊤` on the
  three columns of `s1`, `−q_e^⊤` on those of `s2`.

New symbols (G&B has no equivalent; chosen with the user):

- **`S_s ≡ S_triangle`** — the (reference) area of triangle `s`. Used as the normalization weight.
- **`C`, `κ_v`** — the discrete **curvature / incompatibility** operator and its Lagrange
  multiplier (one scalar `κ_v` per interior vertex). Defined in §3.2.
- **`Π`** — the **compatibility map** from node displacements to per-triangle metric
  fluctuations, `δg = Π u`, `u ∈ ℝ^{2n}`. (G&B's `B` is taken for the mean-field operator, so we
  do **not** reuse it.) Used only in §6 to prove the metric solve equals the configuration solve.

---

## 2. The energy (identical to G&B §2)

`Δg` is held fixed (the load); we minimise over the fluctuation field `δg`:
$$E[\delta g] = \tfrac12 \sum_s A(s)^{\mu\nu\alpha\beta}\,
\bigl(\Delta g + \delta g(s)\bigr)_{\mu\nu}\,\bigl(\Delta g + \delta g(s)\bigr)_{\alpha\beta}.$$

---

## 3. The constraints

We enforce **three** constraints. Two coincide with — and one *corrects* — the paper; one is
**new** (the paper omits it).

### 3.1 Edge compatibility (λ) — identical to G&B §3.2
Adjacent triangles sharing reference edge `Δx_e` must agree on its length:
$$\bigl[\delta g(s1) - \delta g(s2)\bigr]_{\mu\nu}\,\Delta x_e^{\mu}\Delta x_e^{\nu} = 0,
\qquad\text{i.e. } (J\,\delta g)_e = 0,$$
one scalar per interior edge, multiplier `λ_e`.

### 3.2 Curvature / incompatibility (κ) — **NEW; absent from G&B**
Edge agreement alone is **not** full compatibility: a field can match every shared edge length
yet not be realisable as an actual flat triangulation (the vertex angles fail to close). The
missing condition is **zero discrete Gaussian curvature** (= zero disclination density = the
discrete St-Venant incompatibility `inc(δg)=0`).

For each interior vertex `v`, the linearised change of the angle sum around `v` must vanish:
$$(C\,\delta g)_v \;:=\; \sum_{s \ni v} a^{(s)}_{v}{}^{\mu\nu}\,\delta g(s)_{\mu\nu} \;=\; 0,
\qquad a^{(s)}_{v}{}^{\mu\nu} := \frac{\partial \theta^{(s)}_v}{\partial g(s)_{\mu\nu}},$$
where `θ^{(s)}_v` is the interior angle that triangle `s` contributes at `v`, and the sum runs
over triangles incident to `v`. `C` is `n_int × 3N`; multiplier `κ_v` (one scalar per interior
vertex). `κ_v` is the discrete analogue of an **Airy stress potential**.

### 3.3 Normalization (χ) — **CORRECTED: area-weighted**
G&B impose the *unweighted* `Σ_s δg(s)_{μν} = 0`. The correct condition — the one the
simulation actually satisfies, and the one *implied by compatibility* (§5–6) — is the
**`S_s`-weighted** mean:
$$\boxed{\;\sum_s S_s\,\delta g(s)_{\mu\nu} = 0\;}\qquad(\text{3 scalar conditions}),$$
multiplier `χ^{μν}` (symmetric, 3 components), as in G&B but with weight `S_s` instead of `1`.
Write this as `M_S δg = 0`, with `M_S` the `3 × 3N` area-weighted-mean matrix (row `μν` carries
`S_s` on the `μν` column of each triangle `s`).

---

## 4. Lagrangian and stationarity

$$\mathcal L = E
+ \chi^{\mu\nu}\sum_s S_s\,\delta g(s)_{\mu\nu}
+ \sum_{e=(s1,s2,\Delta x_e)} \lambda_e\,[\delta g(s1)-\delta g(s2)]_{\mu\nu}\,\Delta x_e^{\mu}\Delta x_e^{\nu}
+ \sum_v \kappa_v \sum_{s\ni v} a^{(s)}_v{}^{\mu\nu}\,\delta g(s)_{\mu\nu}.$$

Stationarity `∂𝓛/∂ δg(s)_{μν} = 0` gives the per-triangle equation (compare G&B's (★)):
$$\boxed{\,A(s)^{\mu\nu\alpha\beta}\bigl(\Delta g + \delta g(s)\bigr)_{\alpha\beta}
\;+\; S_s\,\chi^{\mu\nu}
\;+\; \sum_{e\ni s}\mathrm{sign}(s,e)\,\lambda_e\,q_e^{\mu\nu}
\;+\; \sum_{v\in s}\kappa_v\,a^{(s)}_v{}^{\mu\nu} \;=\; 0\,}\tag{$\star'$}$$

Two differences from G&B's (★): the normalization term is `S_s χ^{μν}` (was `χ^{μν}`), and there
is an extra curvature term `Σ_{v∈s} κ_v a^{(s)}_v`.

---

## 5. The system actually solved (matrix form)

Stack the per-triangle fluctuations into `δg ∈ ℝ^{3N}` and let `A = blkdiag(A(s))` be the
`3N×3N` **block-diagonal** stiffness (note: **block-diagonal — no mean-field `B`**; the global
coupling is carried by the constraints, not by `A−B`). Let `Δg_rep ∈ ℝ^{3N}` be the affine
3-vector replicated on every triangle. Then (★′) and the three constraints are the KKT system

$$
\begin{pmatrix}
A & J^{\!\top} & C^{\!\top} & M_S^{\!\top}\\
J & 0 & 0 & 0\\
C & 0 & 0 & 0\\
M_S & 0 & 0 & 0
\end{pmatrix}
\begin{pmatrix}\delta g\\ \lambda\\ \kappa\\ \chi\end{pmatrix}
=
\begin{pmatrix}-A\,\Delta g_{\mathrm{rep}}\\ 0\\ 0\\ 0\end{pmatrix}.
\tag{KKT}
$$

The driving term `−A Δg_rep` is the affine pre-stress. (Eliminating `χ` by the area-weighted
sum reproduces G&B's `−δA Δg` driving and the `A−B` operator; keeping `χ` explicit, as here,
leaves `A` block-diagonal and lets `χ` settle to the area-weighted mean stress
`χ^{μν} = −⟨A⟩_S^{μν\alpha\beta}Δg_{αβ} + …` self-consistently.) Solved as a sparse
saddle-point with a tiny multiplier regularisation `−εI` to absorb redundant constraint rows.

**Stripping `Δg`** (everything is linear in it) gives the loading-independent response, exactly
parallel to G&B §7 but with the corrected/added constraint blocks:
$$
\begin{pmatrix} A & J^{\!\top} & C^{\!\top} & M_S^{\!\top}\\ J&0&0&0\\ C&0&0&0\\ M_S&0&0&0\end{pmatrix}
\begin{pmatrix} W\\ \Lambda\\ \mathrm K\\ \mathrm X\end{pmatrix}
= \begin{pmatrix} -A\\ 0\\ 0\\ 0\end{pmatrix},
\qquad \delta g(s) = W(s)\,\Delta g .
$$
The homogenised tensor is the **UNWEIGHTED** mean of the per-triangle actual tensors
$$C_{\mathrm{eff}} = \frac{1}{N}\sum_s (\mathbb 1+W(s))^{\!\top} A(s)\,(\mathbb 1+W(s)),$$
from which `ν, E` follow. **Correction (post-fix, supersedes the June draft):** the final average is
*unweighted*, **not** area-weighted. `A(s)` carries no area prefactor, so the total energy is an
unweighted **sum** and its `Δg`-Hessian is the unweighted mean; this is the true physical
(energy = virial) effective tensor, and it matches the *physical* PBC simulation. An area weight in
the final average (the G&B convention originally written here) biases `ν`/`E` on disordered or
anisotropic meshes and over-states auxeticity. The weight `S_s` is retained **only** where it is
correct — the normalisation constraint `Σ_s S_s δg=0` of §3.3 — never in the homogenisation.
`physical_units=True` additionally rescales `C_eff → C_eff · 8N/Σ_s S_s` to physical stress units
(`ν` is scale-invariant, unaffected). Implemented at `forward_solver_torch.py::forward`
(`C = actual.mean(dim=0)`), verified against the virial/energy ground truth in `verification_tools/`.

---

## 6. Why it is correct: the metric solve *is* the configuration solve

Let `Π` map node displacements to triangle metric fluctuations, `δg = Π u`. Two facts close the
argument:

1. **Compatibility ⇒ the constraints hold automatically.** Any `δg = Π u` satisfies edge
   agreement (`J Π = 0`) and zero curvature (`C Π = 0`) by construction — these define
   realisability. So `range(Π) ⊆ ker[J;C]`.
2. **The area-weighted mean is *implied* by compatibility** — it is a discrete
   divergence-theorem identity, not an extra condition:
   $$M_S\,\Pi \equiv 0,\qquad\text{i.e.}\qquad \sum_s S_s\,\varepsilon(s) = \mathrm{sym}\sum_s \oint_{\partial s} u\otimes \hat n = 0$$
   for any periodic `u`. Verified numerically: `‖M_S Π‖/‖Π‖ = 1.6×10⁻¹⁶` (η≤0.3). The
   **unweighted** mean is *not* implied (`‖(unweighted)Π‖/‖Π‖ = 0.19–1.1`).

Consequently `ker[J;C] = range(Π) ⊕ {3 homogeneous modes}`; the homogeneous modes are the
macroscopic strain (the loading), and `M_S = 0` removes exactly them, projecting onto
`range(Π)`. Minimising `E` over this set is identical to the node-space equilibrium
$$\Pi^{\!\top} A\,\Pi\, u = -\,\Pi^{\!\top} A\,\Delta g_{\mathrm{rep}},\qquad
\Pi^{\!\top} A\,\Pi = 2K \ (\text{torus}),$$
i.e. `K u = −f_aff` — the **simulation**. Hence the intrinsic metric solve, the compatible
(configuration) solve, and the direct simulation coincide. No node displacements are used in
the metric solve itself; `Π` appears only in this proof.

---

## 7. Where G&B goes wrong, as one expansion

G&B impose **(edge λ) + (unweighted Σδg=0, χ)** and **omit the curvature constraint**. Both
choices are visible as truncation errors of a single exact law.

**The exact closure** is area conservation (verified to machine precision; `1.5×10⁻⁴` at η=0.5):
$$\sum_s S_s\,\det F_s = \det(F)\,\sum_s S_s,\qquad \det F_s = \sqrt{\det(I+\Delta g+\delta g_s)} .$$
Expanding `det F_s ≈ 1 + ½\,\mathrm{tr}(Δg+δg_s)`:

- **O(δ):** `Σ_s S_s δg(s) = 0` — the **area-weighted** normalization of §3.3 (automatically
  satisfied by compatible fields). G&B's claim that the *unweighted* trace
  `ḡ^{μν}Σ_s δg_{μν}(s)=0` is "area conservation" is the slip: the correct first-order area
  condition carries the weight `S_s`.
- **O(δ²):** a nonzero remainder — so the *actual* `⟨δg⟩` is second order, `∝ δ²`
  (`⟨tr δg⟩_S/δ² = O(1)`). Imposing `⟨δg⟩=0` (either weighting) is only a leading-order
  statement; the unweighted version is wrong already at O(δ) on a disordered (variable-`S_s`)
  mesh.

The **curvature constraint** is the other omission: without `C δg = 0`, G&B minimise over a
space ≈ `n_int` dimensions too large (incompatible fields with spurious disclinations), giving
the over-compliant ≈1.4× overshoot.

---

## 8. Numerical verification (`test_intrinsic_metric.py`)

> **Convention note:** the homogenised `ν, E` figures in the second table below were computed with
> the *area-weighted* homogenisation of the original June draft. The shipped solver now uses the
> **unweighted** physical mean (§5 correction), so its `ν, E` values differ numerically (they match
> the *physical*/virial simulation instead of the metric one) — but the conclusion is unchanged and
> weighting-independent: the accuracy lives in the per-triangle response `W` (first table, reproduced
> to 3–4 digits), and the same weighting is applied to solver and simulation alike. See
> `verification_tools/` and `Phase 2/SOLVER_GUIDE.md` for the current physical-units numbers.

**Per-triangle `δg` vs the PBC simulation (stored `N=40`):**

| η | constraints **edge+curv+`S_s`-mean** | edge+curv+**unweighted** mean (G&B-style) |
|---|---|---|
| 0.1–0.4 | **1.0000** | 0.945 → 0.887 |
| 0.5 | 0.830 | 0.038 |

**Homogenised `ν, E` vs η (3 macro modes, fresh meshes):**

| η | ν: sim / **intrinsic** / single-site MF | E: sim / **intrinsic** / MF |
|---|---|---|
| 0.2 | 0.251 / **0.251** / 0.256 | 0.0588 / **0.0588** / 0.0428 |
| 0.3 | 0.131 / **0.131** / 0.094 | 0.0532 / **0.0532** / 0.0198 |
| 0.4 | −0.060 / **−0.061** / −0.346 | 0.0438 / **0.0438** / 0.0005 |
| 0.5 | −0.333 / **−0.344** / −0.924 | 0.0306 / **0.0313** / 0.0000 |

The intrinsic metric solve reproduces the simulation's Hessian to 3–4 digits across the whole η
range; single-site MF collapses. The only residual error is at η=0.5, where the O(δ²) term of
§7 (the breakdown of the `M_S Π ≡ 0` identity, residual `3×10⁻³`) becomes visible — to remove
it one carries the next order of the area law, or uses the cluster (which never linearises).

---

## 9. Summary — the corrected constraint set

| | G&B | This note |
|---|---|---|
| energy | `½ Σ_s A(s)(Δg+δg(s))²` | same |
| edge (λ) | `J δg = 0` | same |
| curvature (κ) | — (absent) | **`C δg = 0`** (zero discrete Gaussian curvature) |
| normalization (χ) | `Σ_s δg(s) = 0` (unweighted) | **`Σ_s S_s δg(s) = 0`** (area-weighted) |
| operator | `(A−B)` mean field (χ eliminated) | block-diagonal `A` + constraints (`χ` kept) |
| result | overshoot ≈1.4×, `ν/E` collapse | matches simulation to 3–4 digits |

The metric (incompatible-elasticity) description **does** yield the correct effective Hessian on
its own, with no detour through the configuration — provided one (i) adds the curvature
constraint `C δg = 0` and (ii) replaces the unweighted normalization with the `S_s`-weighted
one. Both corrections are the first two orders of the exact area-conservation law.
