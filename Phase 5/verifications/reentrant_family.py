r"""Re-entrant honeycomb FAMILY, drawn as the RIB SKELETON directly (not via Delaunay), 3x3-tiled +
cropped so the motif reads with no boundary gaps.  Sweeps the vertical-rib length v: the diagonal
ribs go (0,v)->(W/2,1), vertical component 1-v, so
  v < 1 -> diagonals slope UP/out  -> conventional honeycomb
  v = 1 -> diagonals horizontal
  v > 1 -> diagonals slope DOWN/in -> RE-ENTRANT (bow-tie) honeycomb  (auxetic)
Vertical ribs drawn blue, diagonal ribs red, so the re-entrant transition is visible.

It ALSO tries the current seeds._reentrant_honeycomb triangulation per v and reports whether it
survives (it DROPS honeycomb ribs when they aren't Delaunay edges -- exactly the failure for strong
re-entrant v), so we can see the builder's valid range vs the true geometry.

Run:  "C:\Users\doron\anaconda3\python.exe" "Phase 5\verifications\reentrant_family.py"
"""
import os, sys
import numpy as np
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'Phase 3', 'verifications'))
import _common as C
sys.path.insert(0, os.path.join(REPO, 'Phase 5'))
import seeds
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

RESDIR = os.path.join(REPO, 'Phase 5', 'results', 'reentrant')
os.makedirs(RESDIR, exist_ok=True)
W = np.sqrt(3.0)


def build(reps, v):
    """Vertices + the honeycomb ribs (vertical + 2 diagonals) as REAL segments, for any v."""
    reps += reps % 2
    basis = [(0.0, 0.0), (0.0, v), (W / 2, 1.0), (W / 2, 1.0 + v)]
    raw = [(i * W + cx, j * 2.0 + cy) for i in range(reps) for j in range(reps) for cx, cy in basis]
    Lx, Ly = W * reps, 2.0 * reps
    box = np.array([Lx, Ly])
    pts = seeds._dedupe_pts(np.array(raw, float), Lx, Ly)
    key = {tuple(np.round(np.mod(p, box), 5)): i for i, p in enumerate(pts)}
    vert, diag = [], []                                       # real segments (end may exit the box)
    for p in pts:
        for d, bucket in [((0.0, v), vert), ((W / 2, v - 1.0), diag), ((-W / 2, v - 1.0), diag)]:
            q = p + np.array(d)
            if tuple(np.round(np.mod(q, box), 5)) in key:
                bucket.append((p, q))
    return pts, Lx, Ly, vert, diag


def draw_ribs(ax, segs, Lx, Ly, color, reps=3, lw=1.8, pad=0.06):
    r = reps // 2
    tiled = []
    for (p, q) in segs:
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                off = np.array([dx * Lx, dy * Ly])
                tiled.append([p + off, q + off])
    ax.add_collection(LineCollection(tiled, colors=color, linewidths=lw))
    ax.set_xlim(-pad * Lx, (1 + pad) * Lx); ax.set_ylim(-pad * Ly, (1 + pad) * Ly)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])


vs = [0.5, 0.7, 0.9, 1.0, 1.2, 1.5, 1.8, 2.2, 2.6]
n = len(vs); ncols = 3; nrows = int(np.ceil(n / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(3.7 * ncols, 4.0 * nrows), squeeze=False)
print(f"{'v':>5} {'regime':14s} {'builder(Delaunay)':20s}")
for idx, v in enumerate(vs):
    pts, Lx, Ly, vert, diag = build(6, v)
    ax = axes[idx // ncols][idx % ncols]
    draw_ribs(ax, vert, Lx, Ly, 'tab:blue', lw=2.0)
    draw_ribs(ax, diag, Lx, Ly, 'tab:red', lw=1.6)
    regime = 'conventional' if v < 1 else ('flat' if abs(v - 1) < 1e-9 else 're-entrant')
    try:
        seeds._reentrant_honeycomb(reps=6, v=v)
        builder = 'OK'
    except AssertionError as e:
        msg = str(e)
        builder = 'DROPS ribs' + (msg.split('dropped')[1].split('required')[0].strip()
                                  if 'dropped' in msg else '')
    ax.set_title(f"v={v}  ({regime})\nDelaunay-builder: {builder}", fontsize=9)
    print(f"{v:>5.2f} {regime:14s} {builder:20s}")
for idx in range(n, nrows * ncols):
    axes[idx // ncols][idx % ncols].axis('off')
fig.suptitle('Re-entrant honeycomb family — rib skeleton (blue=vertical, red=diagonal), 3x3 tiled+cropped',
             fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.97])
out = os.path.join(RESDIR, 'reentrant_family.png')
fig.savefig(out, dpi=200, bbox_inches='tight'); plt.close(fig)
print(f"\nsaved {out}")
