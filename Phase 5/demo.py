r"""
============================================================================================
 MATERIALIZE — Phase 5 inverse designer :: HANDS-ON DEMO  (read me, then run me)
============================================================================================

WHAT IS THIS?
-------------
This is a metamaterial *inverse designer*. Normally, physics goes FORWARD:
you build a spring network and then measure how stiff and how "squishy" it is.
Here we go BACKWARD ("inverse"): you *ask* for a mechanical behaviour, and the
program searches for a triangulated spring network (its shape/connectivity AND
its per-spring stiffnesses) that actually produces that behaviour.

Two numbers describe the behaviour we target:

  * E  = Young's modulus  -> how STIFF the material is (bigger = harder to stretch).
  * nu = Poisson's ratio  -> what happens SIDEWAYS when you stretch it:
           nu > 0  : normal materials get THINNER when stretched (like a rubber band).
           nu = 0  : width doesn't change (cork-like).
           nu < 0  : "auxetic" — the material gets FATTER when stretched. Weird and useful.
         A regular triangular lattice has nu = 1/3 (about 0.333); we will ask for
         something different and watch the designer find a network that delivers it.

(Advanced: E and nu can be *directional* — different along different pull-angles theta.
 This demo asks for the same value in every direction ("isotropic") to keep things simple.)

HOW THE DESIGNER WORKS (in one breath)
--------------------------------------
  1. It gathers a POOL of candidate network shapes (lattices, honeycomb/kagome-style
     tilings, random blue-noise foams, and some "flipped" non-standard topologies).
  2. For EACH shape it tunes the spring stiffnesses to get as close to your target as it can
     (using the fast differentiable solver).
  3. It keeps the best few — so you get SEVERAL different networks that all hit the target.
  4. It DOUBLE-CHECKS every winner with a completely separate physics simulation
     (not the solver it optimised against) — an honesty check, so the answer is trustworthy.
  5. It saves each design and can draw them into a picture.

HOW TO RUN
----------
  From a terminal (use the project's Python):

      "C:\Users\doron\anaconda3\python.exe" "Phase 5\demo.py"

  It prints a running commentary, saves the designs into  Phase 5/networks/ , and writes
  a picture  Phase 5/demo_gallery.png  you can open. Expect roughly 1-3 minutes
  (it does real optimisation + verification). Nothing here needs a GPU.

WANT TO EXPERIMENT?
-------------------
  Scroll down to the "CHANGE ME" block and edit TARGET_NU / TARGET_E, then re-run.
  Try TARGET_NU = -0.3 for a strongly auxetic material (harder target -> takes longer).
============================================================================================
"""

import os
import numpy as np

# `designer` is the Phase 5 inverse designer we built; `gallery` draws saved networks.
# (Importing designer also wires up the whole solver stack via its own header.)
import designer
import gallery


# ------------------------------------------------------------------------------------------
#  CHANGE ME  — this is the request you are making of the designer.
# ------------------------------------------------------------------------------------------
TARGET_NU = 0.0    # Poisson's ratio we want (0.0 = cork-like; try -0.3 for auxetic, +0.3 = normal)
TARGET_E  = 1.0    # Young's modulus we want (stiffness; ~1.0 is a natural scale for these lattices)
# ------------------------------------------------------------------------------------------


def main():
    print("=" * 90)
    print(f"  INVERSE DESIGN REQUEST:  find networks with  nu = {TARGET_NU:+.2f}   and   E = {TARGET_E:.2f}")
    print(f"  (a regular triangular lattice would give nu = +0.33 — we are asking for something else)")
    print("=" * 90)

    # --- Step 1: build a SMALL candidate pool -------------------------------------------------
    # A bigger pool explores more shapes but takes longer. We keep it small so the demo is quick.
    # The pool mixes structured lattices, tilings, random foams, and a few "flipped" (non-standard)
    # topologies that a plain Delaunay triangulation could never produce.
    print("\n[1/4] Assembling a small pool of candidate network shapes to try...")
    pool = designer.topology_pool(n_random=3, n_nodes=50, n_flip_variants=2, flips=5, seed=0)
    n_nonstd = sum(designer._is_nondelaunay(g) for g in pool)
    print(f"      -> {len(pool)} candidate shapes ({n_nonstd} of them non-standard 'flipped' topologies).")

    # --- Step 2+3+4: design, keep the best few, and verify each -------------------------------
    # `design(...)` does the whole job: tune stiffnesses on every shape, keep the best `keep`,
    # verify each with the independent simulation, and save them to Phase 5/networks/.
    # We use a modest optimiser budget (n_iter / n_restarts) so it finishes fast.
    print("\n[2/4] Designing (tuning spring stiffnesses on each shape toward the target)...")
    print("      ...then [3/4] keeping the best few, and [4/4] verifying each with an INDEPENDENT")
    print("      simulation (a different physics engine from the one we optimised against).")
    reports = designer.design(TARGET_NU, TARGET_E, tag="demo", pool=pool,
                              keep=3, n_iter=60, n_restarts=1, reg=0.02)

    # --- Show the results in plain language ---------------------------------------------------
    # `target_err_sim` = how far the INDEPENDENT simulation's result is from what you asked for
    #                    (smaller is better; this is the number that really matters).
    # `solver_sim_gap` = how much the fast solver and the honest simulation disagree
    #                    (should be tiny; if it's big, that design isn't trustworthy).
    print("\n" + "-" * 90)
    print("RESULTS — several different networks that all aim at your target:")
    print("-" * 90)
    print(f"{'#':>2}  {'network shape':32s}  {'sim nu':>8}  {'sim E':>7}  "
          f"{'off-target':>10}  {'honesty gap':>11}")
    for r in reports:
        print(f"{r['rank']:>2}  {r['seed_name'][:32]:32s}  "
              f"{r['nu_sim'].mean():>+8.3f}  {r['E_sim'].mean():>7.3f}  "
              f"{r['target_err_sim']:>10.4f}  {r['solver_sim_gap']:>11.4f}")
    print("-" * 90)

    best = min(reports, key=lambda r: r["target_err_sim"])
    print(f"\nBEST MATCH: the '{best['seed_name']}' network")
    print(f"    you asked for : nu = {TARGET_NU:+.3f},  E = {TARGET_E:.3f}")
    print(f"    it delivered  : nu = {best['nu_sim'].mean():+.3f},  E = {best['E_sim'].mean():.3f}"
          f"   (checked by the independent simulation)")
    if best["target_err_sim"] < 0.08:
        print("    --> Nailed it. The independent simulation confirms the target is met.")
    else:
        print("    --> Closest the small demo pool could get; a bigger pool / harder budget gets closer.")

    # --- Draw the winners ---------------------------------------------------------------------
    print("\nDrawing the designed networks (bond colour = spring stiffness)...")
    out_png = os.path.join(os.path.dirname(__file__), "demo_gallery.png")
    paths = [r["path"] for r in reports]
    gallery.gallery(paths, out_png, ncols=len(paths))
    print(f"    picture saved -> {out_png}")
    print(f"    raw designs   -> {os.path.join(os.path.dirname(__file__), 'networks')}\\design_demo_*.npz")

    print("\nDone!  Open the picture above to SEE the networks the designer invented for your request.")
    print("Edit TARGET_NU / TARGET_E near the top of this file and re-run to design something else.")


if __name__ == "__main__":
    main()
