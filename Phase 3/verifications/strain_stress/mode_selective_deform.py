"""
strain_stress / mode_selective_deform — SHOW THE REGIONS DEFORM SELECTIVELY. Reloads a designed
mode-selective patch (mode_selective_small_<topo>.npz) and draws its DEFORMED geometry under an
increasing pull, at three amplitudes (0 / small / large), first for PULL +x then for PULL +y.

A uniaxial pull stretches EVERYTHING affinely (~ε on every region), which visually swamps the designed
selectivity (a ~0.26-per-unit-strain bulge). So we subtract the uniform background deformation
(best-fit affine of the relaxed field) and show the RESIDUAL — the non-affine, SELECTIVE motion —
exaggerated for visibility: under +x the DISC breathes outward (area up) while the triangle sits still;
under +y the TRIANGLE breathes while the disc sits still. Each region's residual area strain (the
selective ΔA/A, background removed) is annotated. Self-contained; loads a design, no re-run.

Field: u_fluct = PH.relax fluctuation under mode M (Δ=PH.DELTA); residual = u_fluct minus its best-fit
affine over the patch; residual node displacement scales linearly with ε and is drawn ×AMP.
"""
import os, sys, types
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _common as C

CASE = 'strain_stress'
TOPO = sys.argv[1] if len(sys.argv) > 1 else 'disorder_hi'
EPS = [0.0, 0.06, 0.15]                                  # 0 / small / large macro strain
DELTA = C.PH.DELTA
MODES = C.PH.MODES                                       # [xx, yy, xy]


def residual_field(pts, u_fluct):
    """u_fluct with its best-fit affine (A·X + b) removed -> the non-affine, localized part."""
    G = np.c_[pts, np.ones(len(pts))]                     # (nn,3)  [x, y, 1]
    coef, *_ = np.linalg.lstsq(G, u_fluct, rcond=None)    # (3,2)
    return u_fluct - G @ coef                             # (nn,2) residual (per unit DELTA)


def deform(geo, resid, amp):
    """Geometry deformed by ONLY the residual (selective) field, scaled by amp. Background static.
    Returns (bond segments (nb,2,2), deformed tri_verts (nt,3,2))."""
    pts, R = geo['pts'], geo['bond_R']
    bu, bv, sx, tv = geo['bond_u'], geo['bond_v'], geo['simplices'], geo['tri_verts']
    d = amp * resid                                       # residual displacement (nn,2)
    Pu = pts[bu] + d[bu]
    segs = np.stack([Pu, Pu + R + (d[bv] - d[bu])], axis=1)
    tvd = tv + d[sx]
    return segs, tvd


def full_deform(geo, u_fluct, M, eps):
    """FULL physical deformation (affine ε·M·X + relaxation fluctuation) at macro strain eps.
    Returns (bond segments, deformed tri_verts) — the real network at that strain."""
    pts, R = geo['pts'], geo['bond_R']
    bu, bv, sx, tv = geo['bond_u'], geo['bond_v'], geo['simplices'], geo['tri_verts']
    fl = (eps / DELTA) * u_fluct
    Pu = pts[bu] + eps * (pts[bu] @ M.T) + fl[bu]
    far = Pu + R + eps * (R @ M.T) + (fl[bv] - fl[bu])
    segs = np.stack([Pu, far], axis=1)
    tvd = tv + eps * (tv @ M.T) + fl[sx]
    return segs, tvd


def main():
    path = os.path.join(C.savedir(CASE), f'mode_selective_small_{TOPO}.npz')
    geo, kb, _, meta = C.load_network(path)
    disc_spec, tri_spec = meta['disc'], meta['triangle']
    ns = types.SimpleNamespace(centroids=geo['centroids'])
    disc, _ = C.region_shape(ns, disc_spec)
    tri, _ = C.region_shape(ns, tri_spec)
    nt = len(geo['simplices'])
    bg = np.setdiff1d(np.arange(nt), np.concatenate([disc, tri]))
    print(f"  [mode_selective_deform {TOPO}] tri={nt} disc={len(disc)} triangle={len(tri)}", flush=True)

    # relax the loaded network under the 3 macro modes -> node fluctuation fields; keep the RESIDUAL
    # (non-affine) part, which carries the selective breathing.
    u_modes = C.PH.relax(geo, np.arange(2, 2 * len(geo['pts'])), C.SA.assemble_K_faff)
    resid = {mk: residual_field(geo['pts'], u_modes[mk]) for mk in (0, 1)}

    # TRUE physical dilation (per unit strain) from the linear response -> honest excess-over-background
    eps, _ = C.unit_mode_response(geo)
    dil = {mk: 0.5 * (C.MO.vec3(eps[mk])[:, 0] + C.MO.vec3(eps[mk])[:, 2]) for mk in (0, 1)}

    def excess(mk, idx):                                  # region mean dilation minus background mean
        v = dil[mk][idx]; v = v[np.isfinite(v)]
        b = dil[mk][bg]; b = b[np.isfinite(b)]
        return float(v.mean() - b.mean())
    exc = {(mk, name): excess(mk, idx) for mk in (0, 1) for name, idx in (('disc', disc), ('tri', tri))}

    # one shared exaggeration: at the largest ε, make the biggest residual displacement ~22% of the
    # disc radius, so the selective motion is clearly visible (and comparable across the two pulls).
    r_disc = disc_spec['radius']
    rmax = max(np.abs(resid[mk]).max() for mk in (0, 1))
    AMP = 0.22 * r_disc / (rmax * (max(EPS) / DELTA))     # multiplies (ε/Δ)·resid

    plt.rcParams.update({'font.size': 12})
    fig, axes = plt.subplots(2, len(EPS), figsize=(5.0 * len(EPS), 10.0))
    pulls = [(0, '+x', 'DISC'), (1, '+y', 'TRIANGLE')]

    for row, (mk, plabel, active) in enumerate(pulls):
        for col, eps in enumerate(EPS):
            ax = axes[row, col]
            amp = AMP * (eps / DELTA)
            segs, tvd = deform(geo, resid[mk], amp)
            ax.add_collection(LineCollection(segs, colors='0.72', linewidths=0.35, zorder=1))
            for idx, fc, ec in ((disc, '#1f77b4', '#08306b'), (tri, '#ff7f0e', '#7f3f00')):
                ax.add_collection(PolyCollection(list(tvd[idx]), facecolors=fc, edgecolors=ec,
                                                 linewidths=0.25, alpha=0.80, zorder=3))
            # undeformed reference outline of the two regions (dashed)
            if eps > 0:
                for idx in (disc, tri):
                    ax.add_collection(PolyCollection(list(geo['tri_verts'][idx]), facecolors='none',
                                                     edgecolors='0.35', linewidths=0.4, linestyles=':', zorder=4))
            C.square_frame(ax, geo)
            if col == 0:
                ax.set_ylabel(f'PULL {plabel}\n(active: {active})', fontsize=13)
            ttl = 'ε = 0  (undeformed)' if eps == 0 else f'ε = {eps:.2f}'
            if eps > 0:
                # physical excess area-strain (background removed) at this ε: area ≈ 2·dilation
                d_ex, t_ex = 2 * eps * exc[(mk, 'disc')], 2 * eps * exc[(mk, 'tri')]
                ttl += f'\ndisc excess ΔA/A={d_ex:+.1%}   triangle={t_ex:+.1%}'
            ax.set_title(ttl, fontsize=10.5)

    fig.suptitle(f'Mode-selective patch — SELECTIVE deformation ({nt} tri, {TOPO}); uniform background '
                 f'stretch removed, residual ×{AMP:.0f} exaggerated\ndesigned: disc↔+x, triangle↔+y  '
                 f'(numbers = physical excess dilation, background removed; dotted = undeformed outline)',
                 fontsize=13.0, y=1.01)
    plt.tight_layout()
    out = os.path.join(C.savedir(CASE), f'mode_selective_deform_{TOPO}.png')
    plt.savefig(out, dpi=190, bbox_inches='tight'); plt.close()
    print(f'saved {os.path.basename(out)}')

    # ---- plain grayscale: the FULL deformed network at large strain (no color, shapes outlined) ----
    BIG = [0.15, 0.30]                                    # two large strain levels
    fig2, ax2 = plt.subplots(2, len(BIG), figsize=(5.2 * len(BIG), 10.4))
    for row, (mk, plabel) in enumerate([(0, '+x'), (1, '+y')]):
        for col, eps in enumerate(BIG):
            ax = ax2[row, col]
            segs, tvd = full_deform(geo, u_modes[mk], MODES[mk], eps)
            ax.add_collection(LineCollection(segs, colors='0.6', linewidths=0.3, zorder=1))
            for idx in (disc, tri):                       # shapes outlined in black (no fill, no colour)
                ax.add_collection(PolyCollection(list(tvd[idx]), facecolors='none',
                                                 edgecolors='k', linewidths=0.35, zorder=3))
            ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([]); ax.autoscale()
            if col == 0:
                ax.set_ylabel(f'PULL {plabel}', fontsize=13)
            ax.set_title(f'ε = {eps:.2f}', fontsize=12)
    fig2.suptitle(f'Mode-selective patch — FULL deformed network at large strain ({nt} tri, {TOPO}); '
                  'no colour, shapes outlined in black', fontsize=13, y=1.0)
    plt.tight_layout()
    out2 = os.path.join(C.savedir(CASE), f'mode_selective_deform_network_{TOPO}.png')
    fig2.savefig(out2, dpi=190, bbox_inches='tight'); plt.close()
    print(f'saved {os.path.basename(out2)}')

    # ---- plain, TO-SCALE uniaxial stretch exx = 0 / 0.15 / 0.30 (shared axes, no colour, no shapes) ----
    EXX = [0.0, 0.15, 0.30]
    segs_all = [full_deform(geo, u_modes[0], MODES[0], e)[0] for e in EXX]   # +x uniaxial (exx)
    allpts = np.concatenate([s.reshape(-1, 2) for s in segs_all])
    (x0, y0), (x1, y1) = allpts.min(0), allpts.max(0)
    mx, my = 0.03 * (x1 - x0), 0.03 * (y1 - y0)
    fig3, ax3 = plt.subplots(1, len(EXX), figsize=(5.4 * len(EXX), 5.6))
    for ax, e, segs in zip(ax3, EXX, segs_all):
        ax.add_collection(LineCollection(segs, colors='0.15', linewidths=0.4))
        ax.set_xlim(x0 - mx, x1 + mx); ax.set_ylim(y0 - my, y1 + my)   # SHARED limits -> to scale
        ax.set_aspect('equal', adjustable='box'); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f'εxx = {e:.2f}   (width ×{1 + e:.2f})', fontsize=13)
    fig3.suptitle(f'Mode-selective network under uniaxial stretch, to scale ({nt} tri, {TOPO})', fontsize=13, y=1.0)
    plt.tight_layout()
    out3 = os.path.join(C.savedir(CASE), f'mode_selective_deform_toscale_{TOPO}.png')
    fig3.savefig(out3, dpi=190, bbox_inches='tight'); plt.close()
    print(f'saved {os.path.basename(out3)}')


if __name__ == '__main__':
    main()
