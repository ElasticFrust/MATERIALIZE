"""
Phase 5 gallery — reload saved designed networks and render a montage figure.

Loads saved `.npz` designs (via `C.load_network`) and renders them through the root `plotting.py`
module (the single source of truth for figures; CLAUDE.md §3). Figures LOAD saved designs; they
never re-optimise (random restarts would make plots non-reproducible). Each network is drawn in a
SQUARE panel by the canonical tiled-continuous draw (bonds crossing the boundary render continuously,
no wrap gaps), coloured by per-bond rigidity k; panels are tiled into one montage AND each is also
saved as its own high-DPI element.
"""
import os, sys
import numpy as np
import torch
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))   # repo root from Phase 5/
sys.path.insert(0, REPO)                                    # root plotting.py
sys.path.insert(0, os.path.join(REPO, 'Phase 3', 'verifications'))
import _common as C                 # sets up all other sys.path and imports the solver stack
import plotting as P                # the SINGLE source of truth for figures
torch.set_default_dtype(torch.float64)

HERE = os.path.dirname(os.path.abspath(__file__))


# ---- title formatting ------------------------------------------------------------------------
def _scalar(v):
    """Reduce a scalar or a directional profile (length-37 list/array) to a single float (mean)."""
    a = np.asarray(v, float)
    return float(a.mean()) if a.size > 1 else float(a.reshape(-1)[0])


def _panel_title(meta):
    """Panel title: note + achieved/target nu,E (from meta if present) + loss/solver-sim gap."""
    if not isinstance(meta, dict):
        return ''
    lines = []
    note = meta.get('note') or meta.get('tag') or meta.get('name')
    if note:
        lines.append(str(note))
    ach = []
    for keys, lab in [(('achieved_nu', 'nu', 'nu_sim'), 'ν'), (('achieved_E', 'E', 'E_sim'), 'E')]:
        for k in keys:
            if k in meta and meta[k] is not None:
                ach.append(f"{lab}={_scalar(meta[k]):+.2f}")
                break
    if ach:
        lines.append('achieved ' + ' '.join(ach))
    tgt = []
    if meta.get('target_nu') is not None:
        tgt.append(f"ν*={_scalar(meta['target_nu']):+.2f}")
    if meta.get('target_E') is not None:
        tgt.append(f"E*={_scalar(meta['target_E']):+.2f}")
    if tgt:
        lines.append('target ' + ' '.join(tgt))
    tail = []
    if meta.get('loss') is not None:
        tail.append(f"loss={_scalar(meta['loss']):.2e}")
    if meta.get('solver_sim_gap') is not None:
        tail.append(f"gap={_scalar(meta['solver_sim_gap']):.3f}")
    if tail:
        lines.append(' '.join(tail))
    return '\n'.join(lines)


# ---- back-compat shim: the canonical draw now lives in plotting.draw_network ------------------
def draw_one_tiled(ax, geo, bond_k, meta=None, **kw):
    """Back-compat → `plotting.draw_network` (the canonical tiled-continuous cropped draw). Prefer
    calling `plotting.draw_network` directly in new code."""
    return P.draw_network(ax, geo, bond_k, title=(_panel_title(meta) if meta is not None else None))


# ---- montage ---------------------------------------------------------------------------------
def gallery(paths, out_path, ncols=4, tiled=True):
    """Load each `.npz` (C.load_network) and render a montage of canonical network panels + per-panel
    high-DPI elements, via `plotting.montage`. (`tiled` kept for back-compat — the canonical draw is
    always tiled-continuous now.) Figures LOAD saved designs; never re-optimise. Returns element paths."""
    paths = list(paths)
    if not paths:
        raise ValueError('gallery: no network paths given')
    items = []
    for p in paths:
        geo, bond_k, _C6, meta = C.load_network(p)
        items.append((geo, bond_k, _panel_title(meta)))
    return P.montage(items, out_path, ncols=ncols,
                     suptitle='Phase 5 gallery — designed triangulated networks (bond colour = rigidity k)')


# ---- demo network creation (only if none exist yet) ------------------------------------------
def _make_demo_networks(net_dir):
    """Create a couple of demo networks to have something to render (regular triangular lattice with
    uniform k, and the same with a spatially varied k). Returns the saved paths."""
    os.makedirs(net_dir, exist_ok=True)
    saved = []

    # demo0: regular triangular lattice, uniform k = 1
    geo = C.make_lattice(1.0, 1.0, half=6)
    nb = len(geo['bond_u'])
    k = torch.ones(nb)
    C.apply_k_to_geo(geo, k)
    C6 = C.sim_per_triangle_C6(geo)
    nu, E = C.sim_region_nuE(geo)
    p0 = os.path.join(net_dir, 'demo0.npz')
    C.save_network(p0, geo, k, C6_per=C6, note='demo0 uniform k', nu=nu, E=E)
    saved.append(p0)

    # demo1: same topology, spatially varied k (gradient in x -> visible colour variation)
    geo = C.make_lattice(1.0, 1.0, half=6)
    Lx = float(geo['BL1'][0])
    xmid = (geo['pts'][geo['bond_u']][:, 0] + geo['bond_R'][:, 0] / 2.0) / Lx
    k = torch.as_tensor(0.3 + 1.4 * (xmid % 1.0))            # k ranges ~0.3..1.7 across x
    C.apply_k_to_geo(geo, k)
    C6 = C.sim_per_triangle_C6(geo)
    nu, E = C.sim_region_nuE(geo)
    p1 = os.path.join(net_dir, 'demo1.npz')
    C.save_network(p1, geo, k, C6_per=C6, note='demo1 graded k', nu=nu, E=E)
    saved.append(p1)

    return saved


def main():
    net_dir = os.path.join(HERE, 'networks')
    paths = sorted(__import__('glob').glob(os.path.join(net_dir, '*.npz')))
    if not paths:
        print('No saved networks found — creating demo networks to render.', flush=True)
        paths = _make_demo_networks(net_dir)
    out_path = os.path.join(HERE, 'gallery.png')
    elems = gallery(paths, out_path, ncols=4)
    print(f'Rendered {len(paths)} network(s).', flush=True)
    print(f'Montage: {out_path}', flush=True)
    for e in elems:
        print(f'  element: {e}', flush=True)


if __name__ == '__main__':
    main()
