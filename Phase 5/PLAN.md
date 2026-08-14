# Phase 5 — Inverse designer for triangulated metamaterials (implementation spec)

> ## ⚠ STATUS 2026-08-14 — the saved designs are stale; the designer is sound
>
> A defect in the protected core's homogenisation contraction (`C_xyxy` over-stiff wherever
> `W ≠ 0`) was found and fixed — `documentation/shear_channel_defect.md`. The forward map now agrees
> with the independent physical oracle to ~1e-12, so **the designer machinery is trustworthy**. But
> every design in `Phase 5/networks/` was produced *and verified* against the old map:
>
> - **`goal2` / `goal2_attempts`: treat achieved numbers as INVALIDATED** — 39/39 and 98/102 designs
>   worsened; mean target error 0.05 → 0.48 and 0.20 → 0.71 (worst case 4.14). They target full
>   anisotropic tensors, so they leaned hardest on the broken component.
> - **`g1_2`: 21 of 53 saved designs now exceed `gap_tol`** against the true oracle.
> - `goal1` is the most robust (6/110 over tol). Full table: `Phase 5/results/shear_fix/SHEAR_FIX.md`.
> - `run_g1_2.py` saved only designs passing the *old* trustworthiness check, so its 58 rejected runs
>   are unrecoverable — **re-run, don't re-analyse.**
>
> **Known limits of the current verification** (audit register `documentation/AUDIT_2026-08.md`):
> `verify()`'s "independent sim" is **not** independent — it routes through the solver's own
> contraction (A-1), and the independent oracle covers **bulk only**, so per-triangle and regional
> `C(s)` are still ungated (A-9). §4 of this file's API notes have drifted from the code (C-6).
> Do not add designs on top of this until A-9 and A-7b are done.

> **Purpose of this document.** A self-contained, step-by-step spec an independent implementer can follow to build
> **Milestone 1 (M1)**: a *search-based inverse designer* that, given a target directional response **ν(θ), E(θ)** and a
> network size, produces **explicit triangulated networks (points + edges) + per-bond rigidities** that realise it, verified
> against an independent physical simulation. It also emits the labelled dataset that the **M2** neural net (a GNN edit-policy)
> will train on. **Do not modify `Phase 2/forward_solver_torch.py` (protected core).** All new code lives in `Phase 5/`.
> **First action: copy this file to `Phase 5/PLAN.md`.**

---

## 0. Environment & imports (use exactly this preamble in every Phase 5 script)

```python
import os, sys
import numpy as np
import torch
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))   # repo root from Phase 5/
sys.path.insert(0, os.path.join(REPO, 'Phase 3', 'verifications'))
import _common as C                 # sets up all other sys.path and imports the solver stack
from inverse_design import (DesignProblem, Objective, optimize, validate, ANG,
                            c6_to_nuE, c6_to_nuE_theta)
torch.set_default_dtype(torch.float64)   # REQUIRED — the whole stack is float64
```

- `ANG = np.linspace(0, np.pi, 37)` is the canonical θ grid; targets are arrays of length 37 (or scalars → flat profile).
- Everything is periodic (PBC). ν(θ),E(θ) is then a clean bulk property.

**Environment (verified):** run all scripts with `C:\Users\doron\anaconda3\python.exe` (numpy 2.3.5, scipy 1.16.3,
torch 2.12.1+cpu). `python`/`python3` on PATH are non-working Windows Store stubs — do not use them.

**Path-depth caveat (verified):** the `REPO = ...join(dirname(__file__), '..')` line above is correct only for scripts
**directly in `Phase 5/`** (`seeds.py`, `designer.py`, `dataset.py`, `gallery.py`, `triangulation.py`). For scripts one
level deeper — anything under **`Phase 5/verifications/`** — use `os.path.join(dirname(__file__), '..', '..')`. The
`import _common as C` line MUST precede `from inverse_design import ...` (importing `_common` is what puts
`Phase 3/` on `sys.path`).

---

## 1. Reusable API (the cheat-sheet — call these, do not reimplement)

Geometry (in `_common`, alias `C`):
- `C._periodic_delaunay(pts, Lx, Ly) -> geo` — periodic Delaunay of an (n,2) point cloud in `[0,Lx)×[0,Ly)`. Returns a
  `geo` dict with keys: `pts, simplices, edge_vecs, actual_len2, bond_u, bond_v, bond_R, tri_bond, areas, centroids,
  tri_verts, BL1, BL2`. **This is the ONLY thing needed to turn a point set into a solver-ready topology.**
- `C.make_lattice(phi, psi, half=10.0, seed=0, eta=0.0, half_y=None) -> geo` — seed lattices (φ=ψ=1 → regular triangular;
  eta>0 → disordered). Returns a `geo` dict.
- `geo['BL1'][0]`, `geo['BL2'][1]` are `Lx, Ly`.

Design (in `inverse_design`):
- `prob = DesignProblem.from_geo(geo)` — build a solver problem from a `geo` dict (needs `pts, simplices, edge_vecs,
  bond_R, areas, tri_bond, actual_len2` — all present from `_periodic_delaunay`).
- `Objective('nu_theta', target, thetas=ANG)` and `Objective('E_theta', target, thetas=ANG)` — directional targets.
  `target` is a length-37 array (a profile) or a scalar (broadcast to a flat profile). Optional `weight=`.
- `res = optimize(prob, objs, mode='k', optimizer='lbfgs', n_iter=80, n_restarts=1, seed=0, reg=0.02, verbose=False)`
  → `dict(k=<torch (n_bond,)>, l0=None, loss=float, history=list, raw=...)`. **Use `reg≈0.01–0.05`** (penalises k-VARIANCE → keeps k near a constant/uniform level, the level free to float,
  avoids floppy/unstable designs). `n_restarts>1` gives several distinct k-solutions (diversity).
- `validate(prob, res['k'], res['l0'], objs)` → list of dicts with `achieved`, `target`, `err`.
- Solver's own directional readout for a designed k:
  ```python
  out = prob.forward(res['k'])                    # physical_units=True by default
  C6  = prob.region_tensor(out['per_triangle'], None)     # (6,) torch, bulk physical tensor
  nu_th, E_th = c6_to_nuE_theta(C6, ANG)          # torch (37,), (37,)
  ```

Independent verification (in `_common`):
- `C.apply_k_to_geo(geo, res['k'])` — install designed k onto `geo` (sets `geo['bond_k']`, `geo['tri_k']`).
- `C6_per = C.sim_per_triangle_C6(geo)` — **independent** full-PBC relaxation → per-triangle physical tensor (nt,6).
  **This is a different code path from the solver — the honesty check.**
- `C6_bulk = C.region_phys_C6(geo, C6_per, None)` — bulk 6-vector from the sim.
- `nu_th_sim, E_th_sim = C.nu_E_theta(C6_bulk, ANG)` — sim's directional response (numpy (37,), (37,)).
- `nu_sim, E_sim = C.sim_region_nuE(geo)` — sim's isotropic-mean ν,E convenience.

Persistence (in `_common`):
- `C.save_network(path, geo, res['k'], C6_per=C6_per, **meta)` — save designed net (+ per-triangle tensor + JSON meta).
- `geo, bond_k, C6_per, meta = C.load_network(path)` — reload (rebuilds edge_vecs/actual_len2). Figures LOAD, never re-optimise.

---

## 2. Files to create (all under `Phase 5/`)

- `Phase 5/PLAN.md` — this document (copy verbatim, first).
- `Phase 5/seeds.py` — generate the seed + random periodic point sets (the three ingredients).
- `Phase 5/designer.py` — the search-based inverse designer (M1 core) + verification.
- `Phase 5/dataset.py` — scan many topologies/targets → the `(topology+k → ν(θ),E(θ))` labelled dataset (for M2).
- `Phase 5/gallery.py` — load saved networks → the postable gallery figure.
- `Phase 5/verifications/sanity.py` — the regular-lattice sanity gate.
- `Phase 5/networks/` — saved `.npz` designs (created at runtime).
- `Phase 5/triangulation.py` — **M1b only**: explicit-triangulation + periodic edge-flip (see §8).

---

## 3. `seeds.py` — produce periodic point sets (the three ingredients)

Each function returns `pts` (n,2) in `[0,Lx)×[0,Ly)` plus `(Lx, Ly)`. Use spacing ≈ 1 so edges ≈ 1 (matches the lattice
scale; ν is scale-invariant so absolute size only affects E magnitude).

The training set must be **rich and physically structured**, not just random. Five ingredients:

```python
def seed_bravais(phi_vals=(0.6,0.8,1.0,1.2,1.5), psi_vals=(0.6,0.8,1.0,1.2), etas=(0.0,0.15,0.30),
                 seeds=(0,1), half=8):
    """Scan the 2D BRAVAIS family: make_lattice(phi,psi) sweeps oblique/rectangular/square/
    hexagonal/sheared single-site lattices. eta>0 overlays disorder on each (several seeds)."""
    for phi in phi_vals:
        for psi in psi_vals:
            for eta in etas:
                for s in (seeds if eta > 0 else (0,)):
                    yield f'bravais_phi{phi}_psi{psi}_eta{eta}_s{s}', \
                          C.make_lattice(phi, psi, half=half, eta=eta, seed=s)

def seed_with_basis(bravais_v1, bravais_v2, basis, reps, Lx, Ly, eta=0.0, seed=0):
    """LATTICES WITH A COMPLEX BASIS (multi-atom cells → honeycomb, kagome, ...). Place every
    `basis` point at each Bravais site (m*v1 + n*v2), wrap into [0,Lx)×[0,Ly), optional disorder,
    then _periodic_delaunay the point cloud. Honeycomb basis = 2 points/cell; kagome = 3; etc."""
    m, n = np.meshgrid(range(reps), range(reps), indexing='ij')
    sites = (m.ravel()[:,None]*bravais_v1 + n.ravel()[:,None]*bravais_v2)
    pts = (sites[:,None,:] + np.asarray(basis)[None,:,:]).reshape(-1,2)
    pts[:,0] %= Lx; pts[:,1] %= Ly
    if eta > 0:
        rng = np.random.default_rng(seed); a = rng.uniform(0,2*np.pi,len(pts))
        pts = (pts + eta*np.stack([np.cos(a),np.sin(a)],1)); pts[:,0]%=Lx; pts[:,1]%=Ly
    return C._periodic_delaunay(pts, Lx, Ly)   # NOTE: triangulates the basis POINTS (see nuance below)

def seed_from_phase4():
    """Reuse Phase 4 point layouts (kagome/honeycomb/square/quasicrystal) where available:
    import Phase 4/data/topology_generators.py, take its POINT clouds, → _periodic_delaunay."""
    ...

def random_patch(n_nodes, seed, L=None, process='poisson_disk'):
    """Rich random periodic point cloud → geo. n_nodes ~30 up to SEVERAL HUNDRED (sparse path).
    `process` selects the POINT PROCESS (not just uniform):
      'uniform'      – rng.uniform (baseline).
      'poisson_disk' – Bridson periodic Poisson-disk sampling (min spacing r ≈ 0.8) → clean meshes.
      'blue_noise'   – uniform seed then a few Lloyd-relaxation iterations (periodic centroids).
      'graded'       – spatially varying density (e.g. denser in a band) for graded structures.
    Implement poisson_disk/blue_noise as small periodic samplers; wrap all points into [0,L)."""
    L = float(np.sqrt(n_nodes)) if L is None else L        # spacing ≈ 1
    pts = sample_points(n_nodes, L, seed, process)         # dispatch on `process`
    return C._periodic_delaunay(pts, L, L)

def seed_tiling(name, reps, eta=0.0, seed=0):
    """TILINGS as triangulations with SOFT 'fictional' edges. For non-triangular tilings
    (square, honeycomb, kagome, square-octagon 4.8.8, truncated-hex 3.6.3.6, snub-square, ...):
    generate the tiling's VERTICES and its NATIVE EDGES; then TRIANGULATE each non-triangular
    face by adding diagonals; give the ADDED edges near-zero 'fictional' rigidity so the reference
    network mechanically behaves like the tiling while staying a valid triangulation. Returns
    (geo, k0, is_fictional) where k0[native]=1.0, k0[added]=EPS (e.g. 1e-3).
    ROBUST build: triangulate faces EXPLICITLY and assemble via `geo_from_simplices` (the M1b
    helper) so native edges are guaranteed present; then tag each bond native/added by matching
    endpoints. (Delaunay-of-points may drop native edges — don't rely on it for tilings.)"""
    ...

def auxetic_motifs():
    """Force-include classic auxetic/mechanism seeds (re-entrant honeycomb, rotated-square) as
    tilings-with-soft-edges (via seed_tiling) — they seed the rare 'interesting' region."""
    ...
```

The full seed pool = `seed_bravais()` + `seed_with_basis()` + **`seed_tiling()` (square, honeycomb, kagome,
square-octagon 4.8.8, and more Archimedean tilings, with soft fictional triangulating edges)** + `seed_from_phase4()` +
`auxetic_motifs()` + `random_patch()`×many across **point processes (uniform / Poisson-disk / blue-noise / graded)**, **each
disorder-overlaid**. That spans Bravais lattices, complex-basis lattices, tilings, and disordered/random networks — a broad,
physically-grounded training set.

Notes: (a) `n_nodes`/`reps` up to **several hundred nodes**; >~300 tri use the sparse path. (b) Keep `seed` everywhere.
(c) **Fictional-edge mechanism (key):** every network is formally triangulated, but non-triangular tilings are represented by
adding diagonal edges at **near-zero rigidity (EPS)** in the reference state, so the mechanics match the tiling; those edges
are still real design DOF (the optimiser may stiffen them). **Store the `is_fictional` per-bond mask** in `save_network`'s
meta so (i) the seed's natural response uses native k=1 / fictional k=EPS, and (ii) the M2 GNN sees the *effective*
connectivity. (d) The same `geo_from_simplices` helper (M1b, §8) is what makes explicit face-triangulation possible — build
it early if tilings are needed before flips.

---

## 4. `designer.py` — the search-based inverse designer (M1 core)

**Goal:** given target profiles `nu_target(θ)`, `E_target(θ)` (length-37 arrays) and a candidate pool of topologies, return
the best network(s): explicit triangulation (`geo`) + rigidities `k`, verified.

```python
def design_on_topology(geo, nu_target, E_target, n_iter=120, n_restarts=3, reg=0.02):
    """Optimise k on ONE fixed topology toward the directional target. Returns best over restarts."""
    prob = DesignProblem.from_geo(geo)
    objs = [Objective('nu_theta', np.asarray(nu_target), thetas=ANG, weight=1.0),
            Objective('E_theta',  np.asarray(E_target),  thetas=ANG, weight=1.0)]
    res = optimize(prob, objs, mode='k', n_iter=n_iter, n_restarts=n_restarts, reg=reg, verbose=False)
    return prob, objs, res      # res['k'] is the designed per-bond stiffness

def search(nu_target, E_target, topology_pool, keep=5):
    """Scan a POOL of topologies; design k on each; return the `keep` best (lowest loss) as
    (geo, k, loss) — several distinct designs for the same target (the 'several versions')."""
    results = []
    for geo in topology_pool:
        prob, objs, res = design_on_topology(geo, nu_target, E_target)
        results.append((geo, res['k'], res['loss']))
    results.sort(key=lambda r: r[2])
    return results[:keep]

def verify(geo, k):
    """Independent-sim check + solver-vs-sim gap. Returns a report dict."""
    C.apply_k_to_geo(geo, k)
    C6_per  = C.sim_per_triangle_C6(geo)
    C6_bulk = C.region_phys_C6(geo, C6_per, None)
    nu_sim, E_sim = C.nu_E_theta(C6_bulk, ANG)              # numpy (37,)
    prob = DesignProblem.from_geo(geo)
    out  = prob.forward(torch.as_tensor(k))
    C6_solver = prob.region_tensor(out['per_triangle'], None)
    nu_slv, E_slv = (t.numpy() for t in c6_to_nuE_theta(C6_solver, ANG))
    eps_nu = 0.05                                                              # RELATIVE honesty check
    gap = float((np.abs(nu_sim - nu_slv) / (np.abs(nu_sim) + eps_nu)).max()
                + (np.abs(E_sim - E_slv) / np.maximum(np.abs(E_sim), 1e-12)).max())
    return dict(nu_sim=nu_sim, E_sim=E_sim, nu_solver=nu_slv, E_solver=E_slv,
                solver_sim_gap=gap, C6_per=C6_per)
```

**Topology pool for M1a (no flips yet):** the "serious scan" is over Delaunay-realizable topologies, produced by varying the
point set — this needs *no* new geometry code:
```python
def topology_pool_M1a(n_random=40, n_nodes=120, jitter_rounds=2):
    pool = [geo for _, geo in seeds.seed_lattices()]
    pool += [seeds.random_patch(n_nodes, seed=s) for s in range(n_random)]
    # local search: jitter the points of promising seeds and re-triangulate (changes topology via Delaunay flips)
    # (optional round: perturb pts by N(0,0.1), wrap into box, _periodic_delaunay again)
    return pool
```
Moving points changes the Delaunay connectivity, so this pool already scans many topologies. **Non-Delaunay topologies come
in M1b (§8) via explicit edge-flips.**

**Main M1 entry point:** for a requested target, run `search`, `verify` each kept design, `save_network` the good ones:
```python
for rank, (geo, k, loss) in enumerate(search(nu_target, E_target, topology_pool_M1a())):
    rep = verify(geo, k)
    C.apply_k_to_geo(geo, k)
    path = os.path.join(os.path.dirname(__file__), 'networks', f'design_{tag}_{rank}.npz')
    C.save_network(path, geo, k, C6_per=rep['C6_per'],
                   target_nu=list(nu_target), target_E=list(E_target),
                   loss=loss, solver_sim_gap=rep['solver_sim_gap'])
```

---

## 5. `dataset.py` — the labelled dataset for M2

Two ways to accumulate `(topology + k → ν(θ),E(θ))` pairs; do both:
1. **Forward scan (cheap, guarantees realisability):** for many random/seed topologies, set uniform or random `k`, run the
   solver, record the *achieved* ν(θ),E(θ). No optimisation.
2. **Design scan:** for a grid of targets (e.g. ν∈[-0.5,0.5]×5, E∈[0.5,2.0]×4), run `search`, record each kept
   `(geo, k, achieved ν(θ),E(θ), loss)`.
Save each record with `save_network` under `Phase 5/networks/` (the dataset *is* the collection of saved `.npz`). This is the
GNN's training set for M2.

**M1 minimal "collect relevant topologies" archive:** bin kept designs by `(mean_θ ν, mean_θ E)` on a coarse 2-D grid; keep
the lowest-loss design per bin. That is the smallest "find relevant topologies" mechanism (the full mixed-descriptor
CVT-MAP-Elites is deferred to the roadmap).

---

## 6. Verification gate — `verifications/sanity.py` (run FIRST, must pass)

```python
geo = C.make_lattice(1.0, 1.0, half=6)          # regular triangular
prob = DesignProblem.from_geo(geo)
out  = prob.forward(torch.ones(prob.n_bond))    # uniform k=1
nu, E = (float(x) for x in c6_to_nuE(prob.region_tensor(out['per_triangle'], None)))
assert abs(nu - 1/3) < 1e-2,  f"regular ν should be 1/3, got {nu}"
assert abs(E - 2/np.sqrt(3)) < 5e-2, f"regular E should be 2/√3≈1.1547, got {E}"
# and the independent sim agrees:
C.apply_k_to_geo(geo, torch.ones(prob.n_bond))
nu_s, E_s = C.sim_region_nuE(geo)
assert abs(nu_s - 1/3) < 1e-2 and abs(E_s - 2/np.sqrt(3)) < 5e-2
```
For every designed network also assert `solver_sim_gap < ~0.05` (larger gap ⇒ the solver's prediction on that topology is
untrustworthy — flag it, don't silently keep it).

---

## 7. `gallery.py` — the postable figure

Load the saved `.npz` designs (`C.load_network`), draw each network **in a square panel** with periodic data wrapped into
fractional `[0,1]²` coordinates (project convention: square plot regions), coloured by bond k, titled with achieved
ν(θ),E(θ) (or the ν,E scalars). One montage figure of the discovered topologies hitting a range of targets. High-DPI; save
each panel as its own image too (reusable elements).

---

## 8. `triangulation.py` — explicit edge-flips (M1b, the non-Delaunay extension)

Goal: reach triangulations Delaunay never produces (hubs, aligned/anisotropic connectivity) — the "serious scan of *all*
possibilities."

**Step 1 — factor `geo_from_simplices` out of `_periodic_delaunay`.** Everything in `_periodic_delaunay` *after* the
`Delaunay(tiled).simplices` call (lines building `canon, sft, edge_vecs, areas, tri_bond, bond_R, ... -> geo dict`) is a pure
function of an explicit triangle list `simp_t` (each row = 3 indices into the 3×3-tiled points) + `shift_of`. Extract it as
`geo_from_simplices(pts, simp_t, shift_of, Lx, Ly) -> geo`. `_periodic_delaunay` then = `Delaunay` + this helper. **Do this
by copy, in `Phase 5/triangulation.py`; do not edit `_common.py` beyond, at most, exposing the helper.**

**Step 2 — a periodic edge-flip.** Represent the current triangulation as the explicit tiled triangle list `simp_t` (indices
into tiled points) restricted to triangles whose centroid is in the box (as `_periodic_delaunay` keeps them). An interior edge
is shared by exactly two kept triangles. To flip:
- find the two triangles sharing edge `(a,b)`; let their opposite vertices be `c, d`;
- the flip is legal iff quad `a,c,b,d` is **convex** (both new triangles `a,c,d` and `c,b,d` have positive area — check
  signed areas on the tiled coordinates);
- replace the two triangles `{a,b,c},{a,b,d}` with `{a,c,d},{c,b,d}` in `simp_t`;
- rebuild `geo` via `geo_from_simplices`.

**Step 3 — use flips as extra topology moves** in the search pool (start from a Delaunay `geo`, apply K random legal flips →
a non-Delaunay candidate). Keep positions fixed during a flip; reject illegal flips. Validity is thus maintained by
construction.

Acceptance for M1b: a flipped (non-Delaunay) triangulation still passes the solver + independent-sim on a regular lattice
(same ν,E as Delaunay when geometry unchanged and it is just a re-triangulation of the same points), and the search can now
reach designs M1a plateaus on.

---

## 9. `M2` outline (next milestone — not built in M1)

Train the **GNN edit-policy**: a **simplicial GNN** (typed vertex/edge/triangle nodes; **invariant features** = edge lengths,
triangle angles/areas, k on edges — the solver's own metric quantities) that **reads the current triangulation** and,
conditioned on the target ν(θ),E(θ), outputs a **discrete edit** (which legal flip / point nudge / Δk). Train by imitating the
productive moves recorded in M1's dataset (§5), with k refined through the differentiable solver. Discrete output (which
edit) trained by ordinary continuous weight updates (classification/policy-gradient) — no continuous optimisation of the
topology. Several versions via sampled edits. Later: a generative-from-scratch decoder (non-GNN head on the same encoder).

---

## 10. Acceptance criteria for M1 (definition of done)

1. `verifications/sanity.py` passes (regular ν=1/3, E=2/√3; solver≈sim).
2. `designer.py` takes a target ν(θ),E(θ) and returns ≥1 explicit triangulated network + k whose **independent-sim** response
   matches the target within a stated tolerance, with `solver_sim_gap` small; **several** alternative designs for at least one
   target.
3. Designs saved as `.npz` under `Phase 5/networks/`; `gallery.py` reloads them (no re-optimising) into a square-panel figure.
4. `dataset.py` has produced a labelled `(topology+k → ν(θ),E(θ))` collection for M2.
5. M1b: at least some **non-Delaunay** (flipped) topologies appear among the designs, verified.

---

## 11. Guardrails (project rules)

- **Never modify** `Phase 2/forward_solver_torch.py`. Reuse `_common` / `inverse_design` as above; if a helper must be
  factored (e.g. `geo_from_simplices`), copy it into `Phase 5/`.
- Always **independently simulate** designs (`sim_per_triangle_C6`) — never "verify" with the same solver you optimised
  against. Track and report the solver-vs-sim gap; novel topologies are exactly where the solver is unvalidated.
- Always **save** designed networks; figures **load**, never re-optimise (random restarts ⇒ non-reproducible otherwise).
- New code only in `Phase 5/`; root `README.md` stays the stub. Keep every experiment + plot in the repo, not just /tmp.

---

## Deferred roadmap (context, NOT in scope now)

Quality-Diversity discovery with a mixed (mechanical × structural) descriptor archive; the discovery↔learning loop with the
GNN as QD's variation operator; a spatially-varying target field C\*(x) with an interpreter front-end; a forward surrogate for
scale; rest lengths ℓ₀ as extra outputs; non-periodic/open tiles; the shape / reference-metric (ḡ) track. Recorded so it's not
lost, explicitly out of near-term scope.
