"""Plot the mesh coloured by virtual-distortion rigidities (eta=0.15, a=10)."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import TwoSlopeNorm

from sweep_utils import virtual_distortion_rigidities

# ── Generate ─────────────────────────────────────────────────────────────
vd = virtual_distortion_rigidities(size=(10, 10), eta=0.15, a=10, seed=42)
tri = vd['tri']
rigs = vd['rigidities_np']

with torch.no_grad():
    result = vd['solver'](vd['rigidities'])
nu = result['poisson'].item()
E  = result['young'].item()

# ── Build segments ───────────────────────────────────────────────────────
points = tri.points
simplices = tri.simplices
segments, colors, lws = [], [], []

for ti, sv in enumerate(simplices):
    for ei, (a, b) in enumerate([(sv[0], sv[1]), (sv[0], sv[2]), (sv[1], sv[2])]):
        segments.append([points[a], points[b]])
        k = rigs[ti, ei]
        colors.append(k)
        # Line width scales with rigidity
        lws.append(0.3 + 2.0 * min(k / 2.0, 1.0))

colors = np.array(colors)

# ── Plot ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(1, 1, figsize=(10, 9))

norm = TwoSlopeNorm(vmin=0.0, vcenter=1.0, vmax=2.0)
cmap = 'coolwarm'

lc = LineCollection(segments, cmap=cmap, norm=norm, linewidths=lws)
lc.set_array(colors)
ax.add_collection(lc)

ax.set_xlim(points[:, 0].min() - 0.5, points[:, 0].max() + 0.5)
ax.set_ylim(points[:, 1].min() - 0.5, points[:, 1].max() + 0.5)
ax.set_aspect('equal')

cb = plt.colorbar(lc, ax=ax, shrink=0.8, pad=0.02)
cb.set_label('Rigidity  k = 1 + tanh(a·(l−l₀))', fontsize=11)

ax.set_title(
    f'Virtual Distortion Rigidities\n'
    f'η = 0.15,  a = 10  |  ν = {nu:.4f},  E = {E:.5f}',
    fontsize=13,
)
ax.set_xlabel('x')
ax.set_ylabel('y')

fig.tight_layout()
out = os.path.join(os.path.dirname(__file__), 'virtual_distortion_mesh.png')
fig.savefig(out, dpi=150)
print(f"Saved → {out}")
plt.close(fig)
