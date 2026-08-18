r"""Re-entrant honeycomb by the DIAMETER algorithm (user's spec):

Triangular lattice viewed as hexagons each with a CENTER vertex: perimeter edges HARD (length 1,
kept fixed so every hexagon keeps perimeter 6), radial center->rim edges SOFT (length free).  Pick
ONE diameter direction (x = the antipodal rim pair on the x-axis); squeeze that diameter d from 2
(regular) down toward 0, keeping all 6 perimeter edges = 1.  Per-hexagon rim (all sides 1, x-diam d):
    y = sqrt(1 - ((1-d)/2)^2)
    rim = [(+d/2, 0), (+1/2, +y), (-1/2, +y), (-d/2, 0), (-1/2, -y), (+1/2, -y)]
d=2 -> regular hexagon; d=1 -> tall; d<1 -> RE-ENTRANT (rim x-vertices pull inward); d->0 -> bow-tie.

Topology is FIXED (fan of 6 triangles per hexagon from its centre); only the vertices move with d.
Radial spokes k=eps (soft), perimeter k=1 (hard).  nu is the solver readout; drawn 3x3 tiled+cropped.

Run:  "C:\Users\doron\anaconda3\python.exe" "Phase 5\verifications\dhex_family.py"
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
import triangulation, gallery
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESDIR = os.path.join(REPO, 'Phase 5', 'results', 'reentrant')
NETDIR = os.path.join(REPO, 'Phase 5', 'networks', 'reentrant')
os.makedirs(RESDIR, exist_ok=True); os.makedirs(NETDIR, exist_ok=True)


def build_dhex(nx, ny, d, eps=1e-3):
    """Periodic hexagon-with-centre lattice at x-diameter d.  Returns (geo, k0, is_soft)."""
    y = np.sqrt(1.0 - ((1.0 - d) / 2.0) ** 2)
    a1 = np.array([1.0 + d, 0.0])
    a2 = np.array([(1.0 + d) / 2.0, y])
    Lx, Ly = nx * (1.0 + d), ny * (2.0 * y)
    box = np.array([Lx, Ly])
    rim = np.array([(d / 2, 0.0), (0.5, y), (-0.5, y), (-d / 2, 0.0), (-0.5, -y), (0.5, -y)])

    centers = []
    for i in range(nx):
        for j in range(ny):
            centers.append(np.array([i * (1.0 + d), j * 2.0 * y]))          # centre type 0
            centers.append(np.array([i * (1.0 + d), j * 2.0 * y]) + a2)     # centre type 1 (stagger)
    centers = np.array(centers)

    # unique vertices by wrapped-coordinate key; remember which are centres
    key2idx, pts, is_center = {}, [], []

    def vid(real, center):
        w = np.mod(real, box)
        k = (round(float(w[0]), 5), round(float(w[1]), 5))
        if k not in key2idx:
            key2idx[k] = len(pts); pts.append(w); is_center.append(center)
        return key2idx[k]

    for c in centers:
        vid(c, True)
    tris = []                                                # (3,3): [base, sx, sy] per vertex
    for c in centers:
        cw = np.mod(c, box)
        ci = key2idx[(round(float(cw[0]), 5), round(float(cw[1]), 5))]
        cshift = np.round((c - cw) / box).astype(int)
        rimreal = c + rim                                    # 6 rim vertices (real, may exit box)
        ridx, rsh = [], []
        for rr in rimreal:
            w = np.mod(rr, box)
            ridx.append(vid(rr, False))
            rsh.append(np.round((rr - w) / box).astype(int))
        for a in range(6):
            b = (a + 1) % 6
            tris.append([[ci, cshift[0], cshift[1]],
                         [ridx[a], rsh[a][0], rsh[a][1]],
                         [ridx[b], rsh[b][0], rsh[b][1]]])
    pts = np.array(pts); is_center = np.array(is_center, bool)
    geo = triangulation.geo_from_simplices(pts, np.array(tris, np.int64), Lx, Ly)
    # radial (centre--rim) = soft; perimeter (rim--rim) = hard
    uc, vc = is_center[geo['bond_u']], is_center[geo['bond_v']]
    is_soft = uc ^ vc                                        # exactly one endpoint is a centre
    k0 = np.where(is_soft, eps, 1.0)
    return geo, k0, is_soft


def draw_dhex_tiled(ax, geo, is_soft, reps=3, pad=0.06):
    """Clean categorical draw: HARD perimeter edges solid navy, SOFT radial edges thin light-grey;
    3x3 tiled + cropped to the central cell (continuous, no boundary gaps)."""
    from matplotlib.collections import LineCollection
    Lx, Ly = float(geo['BL1'][0]), float(geo['BL2'][1])
    pts = np.asarray(geo['pts'], float)
    u = pts[geo['bond_u']]; v = u + np.asarray(geo['bond_R'], float)
    hard = ~np.asarray(is_soft, bool)
    r = reps // 2
    sh, ss = [], []
    for dx in range(-r, r + 1):
        for dy in range(-r, r + 1):
            off = np.array([dx * Lx, dy * Ly])
            s = np.stack([u + off, v + off], axis=1)
            sh.append(s[hard]); ss.append(s[~hard])
    ax.add_collection(LineCollection(np.concatenate(ss), colors='#c7ccd1', linewidths=0.9, zorder=1))
    ax.add_collection(LineCollection(np.concatenate(sh), colors='#1b3a6b', linewidths=2.2, zorder=2))
    ax.set_xlim(-pad * Lx, (1 + pad) * Lx); ax.set_ylim(-pad * Ly, (1 + pad) * Ly)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])


def solver_nu(geo, k0):
    try:
        prob = DesignProblem.from_geo(geo)
        out = prob.forward(torch.as_tensor(np.asarray(k0, float)))
        nu, E = c6_to_nuE(prob.region_tensor(out['per_triangle'], None))
        return float(nu), float(E)
    except Exception:                                        # noqa: BLE001
        return float('nan'), float('nan')


def main():
    """Sweep the diameter family, print the table, save the networks and the figure.

    Wrapped in main() + the __main__ guard (audit B-6): this body used to run AT IMPORT, so
    `from dhex_family import build_dhex` (hex_solver_validation.py:55) silently re-ran a
    7-point design sweep and rewrote 7 .npz + 1 .png. Two gates paid that cost on every run,
    and `git status` churned after merely running a test."""
    ds = [2.0, 1.6, 1.2, 1.0, 0.8, 0.5, 0.3]
    n = len(ds); ncols = 4; nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.7 * ncols, 4.0 * nrows), squeeze=False)
    print(f"{'d':>5} {'regime':13s} {'nu':>8} {'E':>8} {'n_node':>6} {'n_bond':>6} {'n_soft':>6}")
    for idx, d in enumerate(ds):
        geo, k0, is_soft = build_dhex(4, 4, d)
        nu, E = solver_nu(geo, k0)
        regime = 'regular' if abs(d - 2) < 1e-9 else ('re-entrant' if d < 1 else 'squeezed')
        ax = axes[idx // ncols][idx % ncols]
        draw_dhex_tiled(ax, geo, is_soft)                        # navy=hard perimeter, grey=soft radial
        tag = ' (auxetic)' if (np.isfinite(nu) and nu < -1e-3) else ''
        ax.set_title(f"d={d} ({regime})\n" + r"$\nu$=" + f"{nu:+.3f}{tag}", fontsize=9)
        print(f"{d:>5.2f} {regime:13s} {nu:>8.3f} {E:>8.3f} {len(geo['pts']):>6} "
              f"{len(geo['bond_u']):>6} {int(is_soft.sum()):>6}")
        C.apply_k_to_geo(geo, k0)
        C.save_network(os.path.join(NETDIR, f"dhex_d{d}.npz"), geo, k0,
                       note=f"diameter-hexagon d={d} nu={nu:+.3f} (perimeter hard, radial soft)",
                       is_fictional=is_soft.tolist())
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis('off')
    fig.suptitle('Hexagon-with-centre, squeeze x-diameter d (perimeter edges=1 fixed, radial soft); 3x3 tiled+cropped',
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(RESDIR, 'dhex_family.png')
    fig.savefig(out, dpi=200, bbox_inches='tight'); plt.close(fig)
    print(f"\nsaved {out}")



if __name__ == '__main__':
    main()
