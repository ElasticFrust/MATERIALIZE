# Phase 5 — Goal 2: directional inverse design against PHYSICALLY-REALIZABLE targets

**Goal.** Design several *directional* cases — target both ν(θ) **and** E(θ) — where every target is
**physically realizable** and **non-trivial** (not a plain single cosine). For each case we optimise
**both** per-bond rigidities `k` **and** vertex positions, verify every design against an
**independent** full-PBC simulation (a different code path from the solver used to design), and
visualize the result.

Reproduce:
```
"C:\Users\doron\anaconda3\python.exe" "Phase 5\verifications\run_goal2.py"
```
Target generators live in `Phase 5/phys_targets.py`; the driver is
`Phase 5/verifications/run_goal2.py`. Machine-readable results: `results/goal2/summary.json`;
raw console log: `results/goal2/run_full.log`.

---

## 1. Why a target must be *derived from a valid tensor* (the physical constraint)

A directional response ν(θ), E(θ) is realizable **only if** it is the response of some valid 2-D
elastic tensor — a symmetric **positive-definite** stiffness. Elastic **reciprocity** (symmetry of
the compliance tensor) then *forces*

$$\frac{\nu(0)}{E(0)} = \frac{\nu(90^\circ)}{E(90^\circ)}.$$

**Consequence:** a plain `ν(θ) = A·cos(2θ)` with flat `E` is **impossible** — it demands
`ν(0)=+A`, `ν(90°)=−A` at equal `E`, so `ν(0)/E(0) = −ν(90°)/E(90°)`, violating reciprocity. Asking
the designer to hit such a target is asking for a material that cannot exist.

We therefore **guarantee** physical targets by **deriving every one from a valid stiffness tensor**,
never by writing down an ν(θ)/E(θ) curve by hand. Every kept target has reciprocity residual
`|ν(0)/E(0) − ν(90°)/E(90°)| < 1e-6` (checked in `phys_targets.is_physical`).

---

## 2. The three target sources (`phys_targets.py`)

1. **Crystal / lattice tensors** (`crystal_targets`, `topo_class = crystal_bravais`).
   Forward an anisotropic Bravais lattice (stretched `make_lattice(φ,ψ)`, some additionally
   **rotated** to tilt the principal axes and introduce a genuine C₁₆ shear coupling) through the
   solver with **uniform k=1**; use *its* bulk tensor's ν(θ), E(θ) as the target. Guaranteed
   physical (it is the response of a real network) and genuinely non-simple. **5 targets.**
2. **Random valid (SPD) tensors** (`random_spd_targets`, `topo_class = random_tensor`).
   Sample `Cv = A Aᵀ + εI` (SPD by construction), normalise the stiffness scale, read off ν(θ), E(θ),
   and **reject** any that fail the realizability filter (E out of range, or `|ν| ≥ 0.95` anywhere).
   **3 kept, 5 rejected.**
3. **Hand-designed orthotropic + filter** (`hand_designed_targets`, `topo_class = hand_orthotropic`).
   Build interpretable orthotropic compliances from `(Ex, Ey, ν_xy, G)`, some **rotated** for tilted
   multi-lobe profiles, and **validate** each as SPD + reciprocal + within the filter. **5 kept, 1
   rejected** (a deliberately non-SPD `ν_xy=1.1` spec).

Every target is stored with `label`, `provenance`, `topo_class`, the generating 6-vector `C6`, and
the length-37 ν(θ), E(θ) arrays on the canonical `ANG = linspace(0,π,37)` grid.

### Rejected candidates (non-physical — **do not count as designed cases**)

**6 of 19 candidate targets rejected:**

| candidate | source | reason |
|---|---|---|
| random_spd_s0_0 | random | `|ν| ≥ 0.95` (max |ν| = 0.996) |
| random_spd_s0_1 | random | `|ν| ≥ 0.95` (max |ν| = 1.934) |
| random_spd_s0_5 | random | `|ν| ≥ 0.95` (max |ν| = 1.016) |
| random_spd_s0_6 | random | `|ν| ≥ 0.95` (max |ν| = 1.002) |
| random_spd_s0_7 | random | `|ν| ≥ 0.95` (max |ν| = 1.519) |
| hand_Ex1.0_Ey1.0_nu1.1_G0.3 | hand | compliance **not SPD** (min eig −10.0) |

---

## 3. Design procedure (`run_goal2.py`)

- **Shared pool** (built once, reused across all cases): 6 stretched/sheared Bravais lattices
  (half=3), 4 random patches (Poisson-disk / blue-noise / uniform), and 2 **non-Delaunay** edge-flip
  variants — 12 small topologies (90–144 bonds).
- **Per case:** design `k` on every pool topology (differentiable solver, `n_iter=70`), verify each
  candidate with the **independent** sim, drop untrustworthy ones (`solver_sim_gap ≥ 0.05`), keep the
  best `KEEP=3` by **independent-sim** target error.
- **Vertex-position optimization** (`positions.design_with_positions`, SPSA + re-Delaunay) on the
  best topology, then re-verify. **Crucially, the position polish uses the SAME objective weights as
  the k-design** (`nu_weight=3.0, E_weight=1.0`) — passing the weights into the position budget was a
  real bug we fixed; without it the polish flattens the down-weighted E at the expense of the ν
  anisotropy. Rank-0 becomes the polished design iff it improves the independent-sim error **and**
  stays trustworthy.
- **Save** all kept designs to `networks/goal2/design_g2_<case>_<rank>.npz` with full metadata
  (provenance, topo_class, target ν/E arrays, generating C6, k-only & k+pos errors, gap).

Target error = `max( max_θ|ν_sim(θ)−ν*(θ)|, max_θ|E_sim(θ)−E*(θ)| )` from the **independent
simulation** (worst-angle L∞, the honest deliverable).

---

## 4. Results — per-case table

**13 physical cases designed; all 13 trustworthy (`solver_sim_gap < 0.05`).** k-only and k+positions
are both independent-sim errors; `Δ = k-only − k+pos` (positive ⇒ positions helped).

| case | provenance | topo_class | k-only err | k+pos err | Δ (help) | gap | best topology |
|---|---|---|---:|---:|---:|---:|---|
| crystal_phi1.3_psi1.0 | crystal | crystal_bravais | 0.0057 | 0.0057 | −0.0000 | 0.001 | random_poisson_disk |
| crystal_phi1.5_psi0.75 | crystal | crystal_bravais | 0.0163 | 0.0103 | +0.0060 | 0.000 | bravais_phi1.5_psi0.8+pos |
| crystal_phi0.7_psi1.3 | crystal | crystal_bravais | 0.0083 | 0.0072 | +0.0011 | 0.015 | bravais_phi0.7_psi1.3+pos |
| crystal_phi1.35_psi1.0_rot30 | crystal | crystal_bravais | 0.0112 | 0.0064 | +0.0048 | 0.002 | random_poisson_disk+pos |
| crystal_phi1.5_psi0.8_rot45 | crystal | crystal_bravais | 0.0402 | 0.0311 | +0.0091 | 0.011 | random_poisson_disk+pos |
| random_spd_s0_2 | random | random_tensor | 0.0381 | 0.0373 | +0.0008 | 0.005 | random_poisson_disk+pos |
| random_spd_s0_3 | random | random_tensor | 0.0868 | 0.0666 | +0.0202 | 0.004 | random_blue_noise+pos |
| random_spd_s0_4 | random | random_tensor | 0.0952 | 0.0834 | +0.0118 | 0.015 | random_poisson_disk+pos |
| hand_Ex1.6_Ey0.8_nu0.3_G0.45 | hand | hand_orthotropic | 0.0057 | 0.0039 | +0.0018 | 0.002 | random_blue_noise+pos |
| hand_Ex1.6_Ey0.8_nu0.3_G0.45_rot30 | hand | hand_orthotropic | 0.0031 | 0.0028 | +0.0003 | 0.002 | random_blue_noise+pos |
| hand_Ex1.2_Ey1.2_nu-0.3_G0.3 | hand | hand_orthotropic | 0.0810 | 0.0713 | +0.0097 | 0.005 | random_poisson_disk+pos |
| hand_Ex2.0_Ey0.7_nu0.1_G0.8_rot22 | hand | hand_orthotropic | 0.0315 | 0.0298 | +0.0017 | 0.009 | random_uniform+pos |
| hand_Ex1.0_Ey1.0_nu0.55_G0.2 | hand | hand_orthotropic | 0.1072 | 0.1028 | +0.0045 | 0.006 | random_poisson_disk+pos |

**Positions helped in 12 / 13 cases** (the one exception, crystal_phi1.3_psi1.0, was already at
0.0057 and the k-only design was retained). **Mean improvement `k-only − k+pos = +0.0055`.**

### Aggregate (k+pos independent-sim error)

- **Best: 0.0028**, **median: 0.0298**, **worst: 0.1028** across the 13 cases.
- Max solver-vs-sim gap over all kept designs: **0.015** (all `< 0.05`, i.e. every kept design is
  trustworthy — the solver's prediction is confirmed by the independent simulation).

### Grouped by provenance / topology class

| group | n | median k+pos err | range |
|---|---:|---:|---|
| crystal / crystal_bravais | 5 | 0.0072 | 0.0057 – 0.0311 |
| random / random_tensor | 3 | 0.0666 | 0.0373 – 0.0834 |
| hand / hand_orthotropic | 5 | 0.0298 | 0.0028 – 0.1028 |

Crystal-derived targets are the easiest (they came from real lattices, so a lattice-like network
reproduces them closely). Random-SPD and the more extreme hand cases (large or negative ν with soft
shear) are hardest — they sit near the edge of what small triangulated spring networks can reach.

---

## 5. Figures

- **Per-case ν(θ)/E(θ) overlays** (achieved independent-sim curves of the kept designs vs the dashed
  target): `results/goal2/response_<case>.png` and polar versions `results/goal2/polar_<case>.png`
  (13 each).
- **Achieved-vs-target scatter** aggregated over all cases & angles, for ν and E:
  `results/goal2/scatter_by_provenance.png` (coloured by crystal/random/hand) and
  `results/goal2/scatter_by_topo_class.png`.
- **Best-per-case network montage** (bonds coloured by actual designed k):
  `results/goal2/gallery_g2.png`.
- **High-DPI reusable elements** (each response + scatter + montage panel standalone):
  `results/goal2/elements/`.

---

## 6. Limitations & notes

- **Realizability filter is deliberately conservative.** We reject targets with `|ν| ≥ 0.95` anywhere.
  Such tensors are still *physical* (SPD) but far outside what small spring networks realize; the plan
  specifies this filter, and we report the rejects rather than chase unreachable targets.
- **Design tolerance is not uniform.** Crystal and mild orthotropic targets match to ~0.003–0.03;
  extreme random/hand targets plateau at ~0.07–0.10. These are honest independent-sim errors, not
  solver-optimism — the `< 0.05` gap on every kept design confirms the solver is trustworthy on these
  topologies.
- **Positions help modestly but consistently** (12/13 cases, mean +0.0055). The k lever does the bulk
  of the work; the position lever mostly polishes and does not overcome the topology/size-bound ceiling
  on the hardest targets.
- **Small cells** (half=3 lattices, 48-node patches) were used so the full 13-case run finishes in
  ~16 min. Larger cells would likely reduce the residual error on the hard cases.
- Best topologies are frequently the `+pos` (position-polished) variants and several are non-Bravais
  random patches, consistent with the pool exploring beyond the seed lattices.

---

## 7. Public API of `phys_targets.py`

```
ANG                                        canonical θ grid (from inverse_design)
voigt_to_c6(Cv) / c6_to_voigt(c6)          <-> 3x3 Voigt stiffness matrix
nuE_of_c6(c6, thetas=ANG)  -> (ν(37), E(37))   physical directional response
rotate_c6(c6, beta)                        rotate the material by β (radians); SPD-preserving
reciprocity_residual(c6)   -> float        |ν(0)/E(0) − ν(90)/E(90)|; ~0 for any physical tensor
is_physical(c6, nu_max=0.95, ...) -> (ok: bool, reason: str)   SPD + reciprocity + realizability
make_target(label, provenance, topo_class, c6, beta=0.0) -> Target
crystal_targets(specs=..., half=4)         -> list[Target]
random_spd_targets(n=8, seed=0, ...)       -> (kept: list[Target], rejected: list[dict])
hand_designed_targets(specs=..., nu_max=0.95) -> (kept: list[Target], rejected: list[dict])
all_targets(seed=0, half=4)                -> (targets: list[Target], reject_log: list[dict])
```
`Target = dict(label, provenance, topo_class, c6 (6,), beta, nu (37,), E (37,))`.


---

## 8. Failed-attempt overlays (selection story)
Enhanced per-case plots overlay the pool topologies that were designed but NOT kept, so the selection is visible: **kept** designs solid + named by topology & sim error, **dropped** (untrustworthy, solver-sim gap >= 0.05) dashed red, **out-ranked** (trustworthy but beaten by the top-3) dotted grey, **target** dashed-black-bold. Kept curves are reloaded from the saved designs; only the non-kept topologies were re-designed (k-only), and each is persisted under `networks/goal2_attempts/`.

Some failed attempts are **numerically degenerate near-mechanisms** whose homogenised ν/E blow up (|ν|≫1, E~1e40); these are **excluded from the curves** (plotting them would destroy the axes) but still counted as dropped, with an "N degenerate excluded" note on the figure. The "attempt err" column below is the **median over the physical (non-degenerate) attempts** — a mean would be poisoned by the ~1e39 degenerate errors.

Figures (add-only): `results/goal2/response_<case>_attempts.png` and `results/goal2/polar_<case>_attempts.png` for every case; counts in `results/goal2/attempts_summary.json`.

| case | provenance | kept | dropped | out-ranked | mean kept err | median attempt err (physical) |
|---|---|---:|---:|---:|---:|---:|
| crystal_phi1.3_psi1.0 | crystal | 3 | 3 | 6 | 0.0069 | 0.0221 |
| crystal_phi1.5_psi0.75 | crystal | 3 | 5 | 4 | 0.0417 | 0.1495 |
| crystal_phi0.7_psi1.3 | crystal | 3 | 4 | 5 | 0.0214 | 0.1585 |
| crystal_phi1.35_psi1.0_rot30 | crystal | 3 | 2 | 7 | 0.0126 | 0.0528 |
| crystal_phi1.5_psi0.8_rot45 | crystal | 3 | 4 | 5 | 0.0429 | 0.1846 |
| random_spd_s0_2 | random | 3 | 4 | 5 | 0.0489 | 0.3674 |
| random_spd_s0_3 | random | 3 | 5 | 4 | 0.0880 | 0.1964 |
| random_spd_s0_4 | random | 3 | 4 | 5 | 0.1003 | 0.3449 |
| hand_Ex1.6_Ey0.8_nu0.3_G0.45 | hand | 6 | 4 | 12 | 0.0105 | 0.0396 |
| hand_Ex1.6_Ey0.8_nu0.3_G0.45_rot30 | hand | 3 | 2 | 7 | 0.0105 | 0.0396 |
| hand_Ex1.2_Ey1.2_nu-0.3_G0.3 | hand | 3 | 4 | 5 | 0.0810 | 0.1851 |
| hand_Ex2.0_Ey0.7_nu0.1_G0.8_rot22 | hand | 3 | 4 | 5 | 0.0339 | 0.0700 |
| hand_Ex1.0_Ey1.0_nu0.55_G0.2 | hand | 3 | 4 | 5 | 0.1478 | 0.2624 |

**Totals:** kept=42, dropped(untrustworthy)=49, out-ranked=75. Failed attempts cluster measurably away from the target (median physical attempt error > mean kept error) in **13/13** cases, so the plots make the selection story clear.
