# tiling_inspect — visual inspection of the tiling topology pool

**Script** `Phase 5/verifications/` tiling inspection helper

## What

A single overview figure of the tiling topologies used by `g1_2` / `goal1`, drawn to check by eye
that each generator produces the intended structure — the cheapest guard against a malformed
generator, and the one that would have caught **A-17** (non-manifold periodic meshes from cocircular
Delaunay tie-breaks) had it been read carefully.

## Status and limitation

**This is an inspection aid, not a verification.** It carries no numbers and asserts nothing. The
actual mesh-validity check is `Phase 2/mesh_build.check_mesh_preconditions` (combinatorial closure +
no inverted triangles), gated by `Phase 5/verifications/test_designer_surface.py` [5]; that is what
should be trusted, and a figure looking plausible is not evidence a mesh is sound — A-17's malformed
tilings looked perfectly reasonable when plotted.

## Figure

- [`tilings.png`](tilings.png)
