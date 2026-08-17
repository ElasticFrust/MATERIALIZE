

---

## 8. Failed-attempt overlays (selection story)
Enhanced per-case plots overlay the pool topologies that were designed but NOT kept, so the selection is visible: **kept** designs solid + named by topology & sim error, **dropped** (untrustworthy, solver-sim gap >= 0.05) dashed red, **out-ranked** (trustworthy but beaten by the top-3) dotted grey, **target** dashed-black-bold. Kept curves are reloaded from the saved designs; only the non-kept topologies were re-designed (k-only), and each is persisted under `networks/goal2_attempts/`.

Figures (add-only): `results/goal2/response_<case>_attempts.png` and `results/goal2/polar_<case>_attempts.png` for every case; counts in `results/goal2/attempts_summary.json`.

| case | provenance | kept | dropped | out-ranked | mean kept err | mean attempt err |
|---|---|---:|---:|---:|---:|---:|
| crystal_phi1.3_psi1.0 | crystal | 3 | 4 | 5 | 0.0083 | 0.0935 |
| crystal_phi1.5_psi0.75 | crystal | 3 | 5 | 4 | 0.0223 | 0.2564 |
| crystal_phi0.7_psi1.3 | crystal | 3 | 4 | 5 | 0.0192 | 0.0694 |
| crystal_phi1.35_psi1.0_rot30 | crystal | 3 | 2 | 7 | 0.0149 | 0.0547 |
| crystal_phi1.5_psi0.8_rot45 | crystal | 3 | 4 | 5 | 0.0245 | 0.2078 |
| random_spd_s0_2 | random | 3 | 4 | 5 | 0.1846 | 0.7086 |
| random_spd_s0_3 | random | 3 | 5 | 4 | 0.0562 | 0.8888 |
| random_spd_s0_4 | random | 3 | 5 | 4 | 0.4487 | 1.2018 |
| hand_Ex1.6_Ey0.8_nu0.3_G0.45 | hand | 6 | 4 | 5 | 0.0121 | 0.0767 |
| hand_Ex1.6_Ey0.8_nu0.3_G0.45_rot30 | hand | 3 | 3 | 6 | 0.0124 | 0.1139 |
| hand_Ex1.2_Ey1.2_nu-0.3_G0.3 | hand | 3 | 5 | 4 | 0.1189 | 0.3474 |
| hand_Ex2.0_Ey0.7_nu0.1_G0.8_rot22 | hand | 3 | 4 | 5 | 0.0834 | 0.8372 |
| hand_Ex1.0_Ey1.0_nu0.55_G0.2 | hand | 3 | 3 | 6 | 0.0302 | 0.0911 |

**Totals:** kept=42, dropped(untrustworthy)=52, out-ranked=65. Failed attempts cluster measurably away from the target (mean attempt error > mean kept error) in **13/13** cases, so the plots make the selection story clear.
