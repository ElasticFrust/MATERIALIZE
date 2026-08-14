r"""Re-entrant honeycomb the CORRECT way: build the hexagonal (honeycomb) topology ONCE at a
conventional angle (where the ribs ARE a valid triangulation), FREEZE the connectivity, then just
MOVE the vertices into the re-entrant configuration (never re-triangulate).  Because the topology is
frozen from the valid conventional mesh, the honeycomb ribs are never dropped -- so STRONG re-entrant
(which the Delaunay-of-points builder could not make) builds fine.

Sweeps the vertical-rib length v (diagonal vertical component 1-v): v<1 conventional, v>1 re-entrant.
Anchored (topology frozen) at v0=0.8.  nu is the solver readout with native ribs stiff / triangulating
diagonals soft (k0 from the conventional build).  Drawn 3x3-tiled + cropped.

Run:  "C:\Users\doron\anaconda3\python.exe" "Phase 5\verifications\reentrant_move.py"
"""
import os, sys
import numpy as np
import torch
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'Phase 3', 'verifications'))
import _common as C
from inverse_design import DesignProblem, c6_to_nuE
torch.set_default_dtype(torch.float64)
sys.path.insert(0, os.path.join(REPO, 'Phase 5'))
import seeds, positions, triangulation, gallery
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESDIR = os.path.join(REPO, 'Phase 5', 'results', 'reentrant')
NETDIR = os.path.join(REPO, 'Phase 5', 'networks', 'reentrant')
os.makedirs(RESDIR, exist_ok=True); os.makedirs(NETDIR, exist_ok=True)
W = np.sqrt(3.0)
REPS = 6
V0 = 0.8                                                       # conventional anchor (valid Delaunay)


def hpts(reps, v):
    """Honeycomb vertices for parameter v, in the FIXED (i,j,basis) order (no dedupe: for v!=1 there
    are no coincidences, so the ordering is v-independent -> index i is the same logical vertex)."""
    reps += reps % 2
    basis = [(0.0, 0.0), (0.0, v), (W / 2, 1.0), (W / 2, 1.0 + v)]
    raw = np.array([(i * W + cx, j * 2.0 + cy)
                    for i in range(reps) for j in range(reps) for cx, cy in basis], float)
    Lx, Ly = W * reps, 2.0 * reps
    return np.mod(raw, [Lx, Ly]), Lx, Ly


def signed_min_area_ratio(geo):
    """min signed-area / mean|area| over triangles (<=0 => a triangle has inverted = mesh tangled)."""
    tv = np.asarray(geo['tri_verts'], float)                  # (nt,3,2) image-correct vertices
    e1 = tv[:, 1] - tv[:, 0]; e2 = tv[:, 2] - tv[:, 0]
    sgn = 0.5 * (e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])   # signed area
    return float(sgn.min() / np.abs(sgn).mean())


def solver_nu(geo, k0):
    try:
        prob = DesignProblem.from_geo(geo)
        out = prob.forward(torch.as_tensor(np.asarray(k0, float)))
        nu, E = c6_to_nuE(prob.region_tensor(out['per_triangle'], None))
        return float(nu), float(E)
    except Exception:                                         # noqa: BLE001
        return float('nan'), float('nan')


# ---- freeze the honeycomb topology at the conventional anchor v0 ------------------------------
rec0 = seeds._reentrant_honeycomb(reps=REPS, v=V0)            # valid at v0 -> native/soft tags
geo0, k0, is_fic = rec0['geo'], rec0['k0'], rec0['is_fictional']
tris0 = positions._tris_from_geo(geo0)                        # FROZEN connectivity
Lx, Ly = float(geo0['BL1'][0]), float(geo0['BL2'][1])
p0, _, _ = hpts(REPS, V0)
assert len(p0) == len(geo0['pts']), f"index-order mismatch: hpts {len(p0)} vs geo0 {len(geo0['pts'])}"
assert np.allclose(np.sort(p0, 0), np.sort(np.asarray(geo0['pts']), 0), atol=1e-6), \
    "hpts(v0) does not reproduce the frozen build's vertex set"

vs = [0.7, 0.9, 1.2, 1.5, 1.8, 2.2, 2.6]
n = len(vs); ncols = 4; nrows = int(np.ceil(n / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(3.7 * ncols, 4.0 * nrows), squeeze=False)
print(f"{'v':>5} {'regime':13s} {'nu':>8} {'E':>8} {'min_sgn_area':>13} {'tangled':>8}")
for idx, v in enumerate(vs):
    pv, _, _ = hpts(REPS, v)
    geov = triangulation.geo_from_simplices(pv, tris0, Lx, Ly)   # MOVE vertices, FROZEN topology
    mar = signed_min_area_ratio(geov)
    tangled = mar <= 0.0
    nu, E = (solver_nu(geov, k0) if not tangled else (float('nan'), float('nan')))
    regime = 'conventional' if v < 1 else 're-entrant'
    ax = axes[idx // ncols][idx % ncols]
    gallery.draw_one_tiled(ax, geov, k0, meta={})
    tag = ' (auxetic)' if (np.isfinite(nu) and nu < -1e-3) else (' TANGLED' if tangled else '')
    ax.set_title(f"v={v} ({regime})\n" + r"$\nu$=" + (f"{nu:+.3f}{tag}" if np.isfinite(nu) else f"--{tag}"),
                 fontsize=9)
    print(f"{v:>5.2f} {regime:13s} {nu:>8.3f} {E:>8.3f} {mar:>13.3f} {str(tangled):>8}")
    if not tangled:
        C.apply_k_to_geo(geov, k0)
        C.save_network(os.path.join(NETDIR, f"reentrant_move_v{v}.npz"), geov, k0,
                       note=f"reentrant-by-move v={v} nu={nu:+.3f} (topology frozen from v0={V0})",
                       is_fictional=is_fic.tolist())
for idx in range(n, nrows * ncols):
    axes[idx // ncols][idx % ncols].axis('off')
fig.suptitle(f'Re-entrant by VERTEX MOVE at FROZEN honeycomb topology (anchored v0={V0}); 3x3 tiled+cropped',
             fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.96])
out = os.path.join(RESDIR, 'reentrant_by_move.png')
fig.savefig(out, dpi=200, bbox_inches='tight'); plt.close(fig)
print(f"\nsaved {out}")
