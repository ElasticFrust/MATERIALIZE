"""Plot the *deformed* virtual network, edges coloured by length."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import TwoSlopeNorm

from sweep_utils import virtual_distortion_rigidities

# ── Generate (same parameters as the rigidity plot) ──────────────────────
vd = virtual_distortion_rigidities(size=(10, 10), eta=0.15, a=10, seed=42)

deformed_pts = vd['deformed_points']
simplices    = vd['tri'].simplices
l_def        = vd['l_deformed']    # (N, 3)
l0           = vd['l0']            # (N, 3)

# ── Build segments on deformed positions, colour = edge length ───────────
segments, colors, lws = [], [], []

for ti, sv in enumerate(simplices):
    for ei, (a, b) in enumerate([(sv[0], sv[1]), (sv[0], sv[2]), (sv[1], sv[2])]):
        segments.append([deformed_pts[a], deformed_pts[b]])
        length = l_def[ti, ei]
        colors.append(length)
        lws.append(0.4 + 1.8 * abs(length - 1.0) / 0.3)

colors = np.array(colors)

# ── Plot ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(1, 1, figsize=(10, 9))

vmin = max(colors.min(), 1.0 - 0.35)
vmax = min(colors.max(), 1.0 + 0.35)
norm = TwoSlopeNorm(vmin=vmin, vcenter=1.0, vmax=vmax)
cmap = 'coolwarm'

lc = LineCollection(segments, cmap=cmap, norm=norm, linewidths=lws)
lc.set_array(colors)
ax.add_collection(lc)

ax.set_xlim(deformed_pts[:, 0].min() - 0.5, deformed_pts[:, 0].max() + 0.5)
ax.set_ylim(deformed_pts[:, 1].min() - 0.5, deformed_pts[:, 1].max() + 0.5)
ax.set_aspect('equal')

cb = plt.colorbar(lc, ax=ax, shrink=0.8, pad=0.02)
cb.set_label('Edge length  l  (l₀ = 1)', fontsize=11)

ax.set_title(
    f'Deformed Virtual Network  (η = 0.15)\n'
    f'Edges coloured by length  —  '
    f'mean l = {colors.mean():.4f},  std = {colors.std():.4f}',
    fontsize=13,
)
ax.set_xlabel('x')
ax.set_ylabel('y')

fig.tight_layout()
out = os.path.join(os.path.dirname(__file__), 'virtual_deformed_mesh.png')
fig.savefig(out, dpi=150)
print(f"Saved → {out}")
plt.close(fig)
