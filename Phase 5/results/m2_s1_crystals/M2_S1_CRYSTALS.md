# S1, simplest case — can a shallow GNN learn the k=1 Bravais crystals?

**Answer: yes, to 0.1 % with NO message passing at all, and 0.025 % with one layer.**

**What, and why this is the right first test** (the user's framing). At `k = 1` on a perfect Bravais
lattice `W = 0` — measured, `max|W| = 2.8e-12` — so

```
    C(s) = A(s) = Σ_e  q_e q_eᵀ / (16 ℓ_e²)
```

is a **closed form in that triangle's own three edges**. No neighbour enters. A model that cannot
pass a single message should therefore already be exact, which isolates the **head and the features**
from the graph machinery entirely. If it fails here, nothing more complicated is worth running.

And **if `k` varies it is no longer a lattice** — a `k`-field breaks the translational symmetry — so
the crystal case is `k = 1` and only `k = 1`.

**Producer.** `Phase 5/verifications/m2_s1_crystals.py` → `crystals.png`, `crystals.npz`.
Threads pinned to 1, float64, seed 0.

---

## 1. Result

| layers | hidden | params | 400 epochs | **3000 epochs** |
|---|---|---|---|---|
| **0 — no message passing** | 32 | 4 902 | 0.0793 | **0.00107** |
| 1 | 32 | 11 270 | 0.0273 | **0.00025** |
| 2 | 32 | 17 638 | 0.0859 | 0.01404 |

(validation MAE / label std, 792 crystals, 20 % held out)

**The 0-layer model is the point.** It has no mechanism to see a neighbour, and it reaches 0.1 %.
So the head, the `Q`-basis and the angle features are together sufficient to represent and *learn* a
purely local target — the S1 precondition, which had not previously been demonstrated.

**Depth HURTS on a local target.** Two layers is 13× worse than zero, and its learning curve shows a
large instability spike near epoch 1000 before a slow recovery. Message passing adds input the target
does not depend on, plus optimisation difficulty. Depth should be justified by the target's
non-locality, not adopted by default.

**Not converged even at 3000.** Both shallow curves are still descending, so 0.00025 is an upper
bound on what this configuration reaches, not a floor.

## 2. How many samples this really is — 792, not 3.9 million

Every triangle of a Bravais crystal is equivalent by translation, and the two triangles of the
primitive cell are inversion-related while `C` is even under inversion. So a 96-triangle crystal
carries **exactly one distinct `C(s)`** — verified: 4 920 triangles over 50 crystals gave 50 distinct
values, 1.00 per crystal.

*(I had claimed 4 920 samples. That was wrong, and the correction is the user's.)*

Resolution therefore has to come from the **(φ, ψ) grid**, not from bigger cells — hence 33 × 24 and
a deliberately small model.

## 3. The parametrisation, measured rather than assumed

`a₁ = (1,0)`, `a₂ = (φ/2, ψ√3/2)`, bonded along `a₁`, `a₂` and **one** diagonal.

- **φ is NOT periodic at fixed diagonal.** ν runs monotonically 0 → +1/3 → 0 → −0.579 → −1.381 over
  φ ∈ [0,4]. An earlier period-2 claim of mine came from the **Delaunay** generator, which always
  takes the *shorter* diagonal and so silently re-folds φ into [0,1].
- **The diagonal flag is redundant with φ:** `(φ, a₁+a₂)` = `(φ+2, a₂−a₁)` to **0.00e+00** at every φ
  tested. They are one family shifted by 2, so one diagonal with φ swept wide covers both.
- **ψ ∈ (0.15, 4] is valid throughout** (A-17 clean), ν from +16.7 at ψ=0.15 to +0.25 at ψ=4, with
  almost all the variation below ψ ≈ 1 — hence logarithmic sampling.

## 4. What this experiment actually caught — a second unlearnable-scale bug

The first run gave 0.59–0.66 for **every** model size and got **worse** with depth. That is the
signature of a task that cannot be fit, not of a capacity shortfall — 792 samples against 17 k
parameters should *over*fit.

Cause: `build()` stored `out['per_triangle']` directly, but that field is **always in INTERNAL
units** — `physical_units=True` rescales only `elastic_tensor`/`young`. The target therefore sat in a
different unit system from the prediction, off by a **per-crystal factor of 18–220**, which no
scale-invariant model can absorb.

This is the **same class** as the morning's `Q`-normalisation bug, twelve hours apart: *the model was
asked for an output determined by a per-sample factor its inputs cannot see.* Both passed SPD,
equivariance, intensivity and expressiveness — those check the **form** of the head, not whether the
target is **reachable**. Both presented as "the model fails to learn".

**The guard that now exists:** `train_v2.oracle_check` pushes the analytic answer through the exact
pipeline before the first gradient step and **raises** unless it reproduces the stored target to
machine precision. Negative-tested — reintroduce either bug and it fires; the healthy pipeline passes
at 5.9e-16 median.

## 5. Limitations

- **W = 0 by construction, so this says nothing about the non-affine response.** It validates the
  head, the basis and the features on a local target; the whole difficulty of the real problem —
  learning `W` — is absent here by design.
- **Not converged**; the shallow curves were still falling at 3000 epochs.
- **One seed per configuration.** The spread across configs (0.00025 … 0.014) is large enough that
  some of it is optimisation noise, and that is not separated here.
- **The residual error is not uniform** — it concentrates in the large-|ν| tail and at low ψ, i.e.
  the extreme-anisotropy corner. Whether that is a real representational limit or just the hardest
  part of the fit is untested.
- ν reaches −59 on this grid, far outside anything the design work has used; those crystals are
  legitimate but extreme, and they dominate the error statistics.
