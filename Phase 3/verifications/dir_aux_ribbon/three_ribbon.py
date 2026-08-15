"""
Case dir_aux_ribbon, STEP 2 — a large (~16k triangle) square split along x into THREE ribbons
(short side x, long side y): plain LEFT | strongly directional-auxetic CENTER | plain RIGHT. The
center ribbon is designed (independently, per [[glue-not-joint-design]]) to hit the REALIZABLE
strong-auxetic ν(θ≈0)=-1.3 target found in find_profile.py (narrow-window, |ν|>1: pulling along x,
it should expand in y MORE than it contracts in x). Left/right are plain undesigned matrix
(uniform k=1) on the same base topology, for contrast. The three independently-built patches are
glue()'d side by side, then cut into an open sheet and stretched along x (clamp x only on the two
ends; everything else free, same convention as two_region/ribbon.py).

Run TWICE, on two topologies built from the same base (aniso_str: anisotropic ordered rows, chosen
-- per instruction -- because an already-anisotropic base makes it easier for the optimiser to
reach a strongly directional response):
  ORDERED     : eta=0   (no positional disorder)
  DISORDERED  : eta=0.08 (small positional disorder, same connectivity story as elsewhere in the repo)

Outputs per topology: dir_aux_ribbon_{topo}.npz (glued network), dir_aux_ribbon_{topo}.png
(deformed shape + local nu(x)/eyy(x) profile). Also dir_aux_ribbon.csv (per-region nu, solo vs
glued, and the measured stretch response) across both topologies.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(HERE, '..', '..', '..', 'Phase 2'))
sys.path.insert(0, os.path.join(HERE, '..', '..', '..', 'verification_tools'))
import _common as C
from inverse_design import Objective
import metric_ops as MO

TOPO = 'aniso_str'
# aniso_str (psi=0.6, rows compressed in y) is ~1.7x denser per unit area than the regular topology
# used elsewhere for "large16k" -- half=11/half_y=33 (not half=14/half_y=42) is what actually lands
# each ribbon (width~22, height~66) x 3 near the ~16k-triangle scale (checked empirically: ~16900 total).
HALF_Y, HALF_X_RIBBON = 33.0, 11.0
NITER, REG = 250, 1e-4          # ~5600 tri/ribbon, adjoint path; a hard narrow-window anisotropic
N_RESTARTS = 3                  # target needs more than large16k's plain isotropic-nu iteration budget
NU_CENTER_TARGET = -1.3                         # realizable per find_profile.py (narrow window)
NARROW_THETAS = np.linspace(0.0, 0.20, 5)       # same window used there
RUNS = [(0.0, 'ordered'), (0.08, 'disordered')]


def strain(geo, u):
    tv = np.asarray(geo['tri_verts']); p0, p1, p2 = tv[:, 0], tv[:, 1], tv[:, 2]
    ev = np.stack([p1 - p0, p2 - p0, p2 - p1], 1)
    eps = MO.tri_metric_change(ev, np.asarray(geo['simplices']), np.eye(2), u)
    return eps[:, 0, 0], eps[:, 1, 1]           # exx, eyy per triangle


def profile(cen, val, nwt, nbin=30):
    """mean of a per-triangle scalar vs x (over the open triangles only)."""
    x = cen[:, 0]; edges = np.linspace(x.min(), x.max(), nbin + 1)
    xc = 0.5 * (edges[:-1] + edges[1:]); out = np.full(nbin, np.nan)
    for i in range(nbin):
        m = nwt & (x >= edges[i]) & (x < edges[i + 1])
        if m.any():
            out[i] = val[m].mean()
    return xc, out


def build_side(eta, seed):
    """Plain undesigned ribbon: uniform k=1 on the base topology."""
    geo = C.make_lattice(*C._TOPO_PARAMS[TOPO][:2], half=HALF_X_RIBBON, half_y=HALF_Y,
                         eta=eta, seed=seed)
    k = np.ones(len(geo['bond_u']))
    geo['bond_k'] = k; geo['tri_k'] = k[geo['tri_bond']]
    return geo, k


def build_center(eta, seed):
    """Center ribbon designed for a strong auxetic response at theta~0 (pulling along x)."""
    geo = C.make_lattice(*C._TOPO_PARAMS[TOPO][:2], half=HALF_X_RIBBON, half_y=HALF_Y,
                         eta=eta, seed=seed)
    prob = C.DesignProblem.from_geo(geo)
    r = C.optimize(prob, [Objective('nu_theta', NU_CENTER_TARGET, thetas=NARROW_THETAS)],
                   mode='k', n_iter=NITER, reg=REG, n_restarts=N_RESTARTS, verbose=False)
    k = r['k'].detach().numpy()
    print(f"    center design: final loss={r['loss']:.4e} over {N_RESTARTS} restarts", flush=True)
    return geo, k


def solo_nu0(geo, k):
    C.apply_k_to_geo(geo, k)
    C6 = C.sim_per_triangle_C6(geo)
    nu0 = C.nu_E_theta(C.region_phys_C6(geo, C6, None), np.array([0.0]))[0][0]
    return float(nu0)


def run_topology(eta, tag, csv_rows):
    print(f"=== {tag} (eta={eta}) ===", flush=True)
    geoL, kL = build_side(eta, seed=1)
    geoC, kC = build_center(eta, seed=2)
    geoR, kR = build_side(eta, seed=3)
    nuL = solo_nu0(geoL, kL); nuC = solo_nu0(geoC, kC); nuR = solo_nu0(geoR, kR)
    print(f"  solo nu(0): left={nuL:+.3f}  center={nuC:+.3f} (tgt {NU_CENTER_TARGET:+.2f})  right={nuR:+.3f}",
          flush=True)
    # persist the center ribbon's designed k (the only expensive-to-obtain piece: geometry is
    # deterministic and cheaply rebuildable from (eta, seed); left/right are always uniform k=1)
    # so re-plotting later never needs to re-run optimize().
    np.savez_compressed(os.path.join(HERE, f'dir_aux_ribbon_center_k_{tag}.npz'), kC=kC)

    Lx1, Ly = float(geoL['BL1'][0]), float(geoL['BL2'][1])
    assert abs(Ly - float(geoC['BL2'][1])) < 1e-6 and abs(Ly - float(geoR['BL2'][1])) < 1e-6
    Lx2 = float(geoC['BL1'][0]); Lx3 = float(geoR['BL1'][0])
    mid1, mid2 = Lx1, Lx1 + Lx2
    ptsL = np.asarray(geoL['pts'])
    ptsC = np.asarray(geoC['pts']) + [mid1, 0.0]
    ptsR = np.asarray(geoR['pts']) + [mid2, 0.0]
    pieces = [dict(pts_keep=ptsL, pts_full=ptsL, bond_u=geoL['bond_u'], bond_v=geoL['bond_v'],
                   bond_R=geoL['bond_R'], k=kL),
              dict(pts_keep=ptsC, pts_full=ptsC, bond_u=geoC['bond_u'], bond_v=geoC['bond_v'],
                   bond_R=geoC['bond_R'], k=kC),
              dict(pts_keep=ptsR, pts_full=ptsR, bond_u=geoR['bond_u'], bond_v=geoR['bond_v'],
                   bond_R=geoR['bond_R'], k=kR)]
    Lx = mid2 + Lx3
    geo, glued = C.glue(pieces, Lx, Ly)
    print(f"  GLUED: {geo['tri_bond'].shape[0]} tri, {glued.sum()} default/interface bonds "
          f"({glued.mean()*100:.1f}%)", flush=True)

    C6 = C.sim_per_triangle_C6(geo); cen = np.asarray(geo['centroids'])
    left = np.where(cen[:, 0] < mid1)[0]
    center = np.where((cen[:, 0] >= mid1) & (cen[:, 0] < mid2))[0]
    right = np.where(cen[:, 0] >= mid2)[0]
    nuL2 = C.c6_nuE(C.region_phys_C6(geo, C6, left))[0]
    nuC2 = C.c6_nuE(C.region_phys_C6(geo, C6, center))[0]
    nuR2 = C.c6_nuE(C.region_phys_C6(geo, C6, right))[0]
    print(f"  after gluing (isotropic-equivalent nu, NOT nu(theta=0)): left={nuL2:+.3f}  "
          f"center={nuC2:+.3f}  right={nuR2:+.3f}", flush=True)

    u, nwt = C.open_stretch(geo, axis=0, regularize=True)
    exx_raw, eyy_raw = strain(geo, u)
    exx, eyy = profile(cen, exx_raw, nwt)[1], profile(cen, eyy_raw, nwt)[1]
    xc = profile(cen, exx_raw, nwt)[0]
    zones = {'left': cen[:, 0] < mid1, 'center': (cen[:, 0] >= mid1) & (cen[:, 0] < mid2),
             'right': cen[:, 0] >= mid2}
    ex_zone = {n: np.nanmean(exx_raw[nwt & m]) for n, m in zones.items()}
    ey_zone = {n: np.nanmean(eyy_raw[nwt & m]) for n, m in zones.items()}
    el, ec, er = ey_zone['left'], ey_zone['center'], ey_zone['right']
    nu_practical = {n: -ey_zone[n] / ex_zone[n] for n in zones}      # -<eyy>/<exx> PER ZONE, the
    print(f"  under x-stretch <eyy>: left={el:+.4f}  center={ec:+.4f} (strongly auxetic, expect >> left/right)  "  # actual as-tested reading
          f"right={er:+.4f}", flush=True)
    print(f"  practical local nu=-<eyy>/<exx>: left={nu_practical['left']:+.3f}  "
          f"center={nu_practical['center']:+.3f}  right={nu_practical['right']:+.3f}", flush=True)
    C.save_network(os.path.join(HERE, f'dir_aux_ribbon_{tag}.npz'), geo, geo['bond_k'], C6,
                   mid1=float(mid1), mid2=float(mid2), eta=float(eta),
                   nu_solo=dict(left=nuL, center=nuC, right=nuR),
                   nu_glued_isotropic=dict(left=nuL2, center=nuC2, right=nuR2),
                   nu_practical=nu_practical)

    csv_rows.append((tag, eta, f'{nuL:.3f}', f'{nuC:.3f}', f'{nuR:.3f}',
                     f'{nuL2:.3f}', f'{nuC2:.3f}', f'{nuR2:.3f}', f'{el:.4f}', f'{ec:.4f}', f'{er:.4f}',
                     f"{nu_practical['left']:.3f}", f"{nu_practical['center']:.3f}", f"{nu_practical['right']:.3f}"))
    draw(geo, u, nwt, cen, mid1, mid2, xc, exx, eyy, nuL2, nuC2, nuR2, tag)
    plot_nu_theta(geoL, kL, geoC, kC, geoR, kR, geo, C6, mid1, mid2, nu_practical, tag)
    return geo, u, nwt, cen, mid1, mid2


def draw(geo, uc, nwt, cen, mid1, mid2, xc, exx, eyy, nuL, nuC, nuR, tag):
    fig, (a0, a1) = plt.subplots(2, 1, figsize=(15, 9.5), gridspec_kw={'height_ratios': [1.3, 1]})
    Lx, Ly = float(geo['BL1'][0]), float(geo['BL2'][1]); mg = 0.5 * Ly
    sx = np.asarray(geo['simplices'])[nwt]; tv = np.asarray(geo['tri_verts'])[nwt]
    scale = 2.0
    dtv = tv + scale * uc[sx]
    # colour each open triangle by its x-binned eyy (the ribbons are ~uniform along y within each
    # zone, so the 1-D x-profile is a fair per-triangle colour, and matches what the bottom panel plots)
    eyy_tri = np.interp(cen[nwt, 0], xc, eyy, left=eyy[0], right=eyy[-1])
    v = np.nanpercentile(np.abs(eyy), 92)
    cols = plt.cm.RdBu_r(0.5 + 0.5 * np.clip(np.nan_to_num(eyy_tri) / v, -1, 1))
    a0.add_collection(PolyCollection(list(tv), facecolors='none', edgecolors='0.88', lw=0.1))
    a0.add_collection(PolyCollection(list(dtv), facecolors=cols, edgecolors='0.5', lw=0.08, alpha=0.9))
    a0.axvline(mid1, color='k', ls='--', lw=1); a0.axvline(mid2, color='k', ls='--', lw=1)
    a0.set_xlim(-mg, Lx + mg); a0.set_ylim(-mg, Ly + mg)
    a0.set_aspect('equal'); a0.set_xticks([]); a0.set_yticks([])
    a0.set_title(f'Macroscopic deformed shape (x{scale:.0f}) under x-stretch — colour = lateral strain '
                 f'εyy (coarse, red=expands)\nLEFT plain ν={nuL:+.2f}  ·  CENTER strong-auxetic ν={nuC:+.2f} '
                 f'(target {NU_CENTER_TARGET:+.2f})  ·  RIGHT plain ν={nuR:+.2f}', fontsize=10)

    nu_local = np.clip(-eyy / exx, -2.2, 1.2)
    a1.axhline(0, color='k', lw=.6); a1.axvline(mid1, color='k', ls='--', lw=1); a1.axvline(mid2, color='k', ls='--', lw=1)
    a1.axvspan(cen[:, 0].min(), mid1, color='#7f7f7f', alpha=0.06)
    a1.axvspan(mid1, mid2, color='#d62728', alpha=0.08)
    a1.axvspan(mid2, cen[:, 0].max(), color='#7f7f7f', alpha=0.06)
    a1.plot(xc, nu_local, '-s', ms=4, color='#2c3e50', label='local ν(x) = -εyy/εxx (simulation)')
    a1.plot(xc, eyy, '-o', ms=3, color='#c0392b', alpha=0.6, label='lateral strain ⟨εyy⟩(x)')
    a1.axhline(NU_CENTER_TARGET, color='#d62728', ls=':', lw=1.2, label=f'center design target ν={NU_CENTER_TARGET:+.2f}')
    a1.axhline(-1.0, color='gray', ls=':', lw=1, label='|ν|=1')
    a1.set_xlabel('x  (stretch direction →)'); a1.set_ylabel('response')
    a1.set_ylim(-2.4, 1.4)
    a1.legend(fontsize=8, loc='lower right'); a1.grid(alpha=.3)
    a1.set_title('Measured response across the ribbon — center zone (red band) should dive to strongly '
                 'negative ν, well past the isotropic floor of −1', fontsize=10)
    fig.suptitle(f'dir_aux_ribbon — {tag} {TOPO} — plain | STRONG DIRECTIONAL AUXETIC | plain, '
                 f'clamp x of the two ends only', fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(HERE, f'dir_aux_ribbon_{tag}.png'), dpi=150, bbox_inches='tight')
    plt.close(); print(f'saved dir_aux_ribbon_{tag}.png')


def plot_nu_theta(geoL, kL, geoC, kC, geoR, kR, geo, C6, mid1, mid2, nu_practical, tag):
    """Three DIFFERENT, all-legitimate readings of nu(theta), theta in [0,180 deg), side by side
    for each region -- the point being that they need not agree, and each answers a different
    question:
      SOLO      : the ribbon's OWN periodic tensor, built and simulated in isolation (the
                  "as-designed / as-manufactured" material spec -- what you'd hand to a fab).
      EMBEDDED  : that same region's tensor measured FROM WITHIN the glued composite under a
                  GLOBAL periodic macro-strain (captures interface/kinematic-constraint coupling
                  with its neighbours -- the "as-installed structural" reading).
      PRACTICAL : a single point (theta=0 only) -- the DIRECTLY measured -<eyy>/<exx> from the
                  actual open-boundary x-stretch test (the "as-tested, pulling on the real panel"
                  reading; not a periodic-tensor quantity at all, so only exists at theta=0).
    The center panel also shows the narrow-window DESIGN TARGET the optimizer was actually given
    (a flat level over a small band near theta=0 -- nothing outside that band was ever constrained,
    so a wild-looking curve away from the red band is expected, not a bug)."""
    th = C.ANG; deg = np.degrees(th)

    def solo_profile(geo0, k0):
        C.apply_k_to_geo(geo0, k0)
        C6_0 = C.sim_per_triangle_C6(geo0)
        return C.nu_E_theta(C.region_phys_C6(geo0, C6_0, None), th)[0]

    nuL_solo, nuC_solo, nuR_solo = solo_profile(geoL, kL), solo_profile(geoC, kC), solo_profile(geoR, kR)

    cen = np.asarray(geo['centroids'])
    left = np.where(cen[:, 0] < mid1)[0]
    center = np.where((cen[:, 0] >= mid1) & (cen[:, 0] < mid2))[0]
    right = np.where(cen[:, 0] >= mid2)[0]
    nuL_emb, _ = C.nu_E_theta(C.region_phys_C6(geo, C6, left), th)
    nuC_emb, _ = C.nu_E_theta(C.region_phys_C6(geo, C6, center), th)
    nuR_emb, _ = C.nu_E_theta(C.region_phys_C6(geo, C6, right), th)

    fig, axes = plt.subplots(1, 3, figsize=(19, 6.2), sharey=True)
    panels = [('LEFT (plain)', nuL_solo, nuL_emb, nu_practical['left']),
              ('CENTER (designed)', nuC_solo, nuC_emb, nu_practical['center']),
              ('RIGHT (plain)', nuR_solo, nuR_emb, nu_practical['right'])]
    for ax, (name, nu_s, nu_e, nu_p) in zip(axes, panels):
        ax.axhline(0, color='k', lw=.5, ls=':'); ax.axhline(-1.0, color='gray', lw=1, ls=':')
        if name.startswith('CENTER'):
            ax.axvspan(np.degrees(NARROW_THETAS.min()), np.degrees(NARROW_THETAS.max()), color='#d62728', alpha=0.12,
                      label='design window (θ≈0)')
            ax.plot([0, np.degrees(NARROW_THETAS.max())], [NU_CENTER_TARGET] * 2, ls='--', color='#d62728',
                    lw=2, label=f'target ν={NU_CENTER_TARGET:+.2f}')
        ax.plot(deg, nu_s, '-', color='#2ca02c', lw=2, label='SOLO (isolated, as-designed)')
        ax.plot(deg, nu_e, '-', color='#9467bd', lw=2, label='EMBEDDED (in glued panel)')
        ax.plot([0], [nu_p], 'D', ms=11, color='#d62728', mec='k', mew=1.2, zorder=5,
               label='PRACTICAL (measured open x-stretch)')
        ax.set_title(name, fontsize=11); ax.set_xlabel('loading angle θ (deg)')
        ax.set_xlim(0, 180); ax.set_ylim(-3.0, 3.0)
        ax.grid(alpha=0.3); ax.legend(fontsize=7.5, loc='upper right')
    axes[0].set_ylabel('ν(θ)')
    fig.suptitle(f'dir_aux_ribbon — {tag} {TOPO} — SOLO vs EMBEDDED vs PRACTICAL ν(θ=0): three different, '
                 f'all-legitimate readings (θ=0 -> pull along x)', fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(os.path.join(HERE, f'dir_aux_ribbon_nutheta_{tag}.png'), dpi=150, bbox_inches='tight')
    plt.close(); print(f'saved dir_aux_ribbon_nutheta_{tag}.png')


def main():
    csv_rows = []
    for eta, tag in RUNS:
        run_topology(eta, tag, csv_rows)
    C.write_csv(os.path.join(HERE, 'dir_aux_ribbon.csv'),
                ['topology', 'eta', 'nu_left_solo', 'nu_center_solo', 'nu_right_solo',
                 'nu_left_glued_isotropic', 'nu_center_glued_isotropic', 'nu_right_glued_isotropic',
                 'eyy_left_stretch', 'eyy_center_stretch', 'eyy_right_stretch',
                 'nu_left_practical', 'nu_center_practical', 'nu_right_practical'], csv_rows)
    print('done')


if __name__ == '__main__':
    main()
