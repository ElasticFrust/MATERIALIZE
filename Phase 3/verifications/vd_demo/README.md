# VD on a regular lattice — is rigidity contrast alone auxetic?

Virtual-distortion (VD) rigidity on an **unchanged regular** triangular lattice: displace a *virtual*
copy of the nodes by `v → v + η(cosθ,sinθ)` (random θ per vertex, connectivity fixed, **no
re-triangulation**), read the virtual edge lengths `l`, and put `k = 1 + tanh(α(l − 1))` on the real
regular lattice (`l₀ = 1`). Homogenise ν, E by PBC **simulation** (and cross-check the forward solver).

## Conclusion
**Rigidity contrast alone makes the regular lattice auxetic — but only above a contrast threshold.**
- At the commonly-used `α = 5` it is *below* threshold: ν only drops from 1/3 toward ~0 (not auxetic).
  This initially (wrongly) suggested the disordered geometry was required — it is not.
- Sweeping α (`vd_alpha_sweep.py`): ν crosses 0 near **α ≈ 15–21** and reaches ~−0.05…−0.1 (10-seed
  mean) at α = 30. Higher η lowers the threshold.
- The auxeticity **weakens with system size** (finite-size self-averaging): the small N=14 system is
  noisier and more auxetic; at **16 200 triangles** the mean is milder (~−0.03…−0.05) but robust.
- The forward **solver reproduces the simulation** to ~0.001–0.002 across the whole α scan at 16k tri.

The earlier `a=5` η-sweep plots (`verification_tools/…VD_eta…`) are auxetic because their *actual
geometry* is deformed (disorder+VD); `vd_sweep_sim.py` shows the same VD `k` on the regular vs the
deformed geometry side by side — regular stays ν>0 at α=5, deformed reaches −0.54. Disorder just
**lowers the α threshold**; it is not required.

## Files
| file | what |
|---|---|
| `vd_regular.py` | single run (α=5, η=0.45): [rigidity k \| local ν \| local E] + saved network |
| `vd_decompose.py` | geometry × rigidity 2×2 (at α=5) — isolates the effect |
| `vd_sweep_sim.py` | ν,E vs η at α=5: same k on regular vs deformed geometry |
| `vd_alpha_sweep.py` | ν,E vs contrast α (η=0.1,0.15), 10 seeds, **simulation vs forward solver** |
| `vd_big.py` | large ~16 200-triangle run: α-sweep (sim vs solver, 3 seeds) + local maps |

All networks saved as `.npz` (geometry + k + per-triangle tensor + metadata) for reload.
