"""
ALL patch types on ALL topologies. For every topology x case we design k (isotropic scalar targets),
simulate, and SAVE the network to networks/ (so maps are rebuilt by loading, no re-optimising).

Cases (centered region):
  disc/square/triangle/ring auxetic : auxetic patch (nu=-0.3) in a normal matrix (nu=+0.3)
  normal_in_auxetic                 : normal disc (nu=+0.3) in an AUXETIC matrix (nu=-0.3)
  stiffE_in_soft / softE_in_stiff   : E-contrast disc
  decoupled                         : E differs only in R_E (left), nu only in R_nu (right)

make_maps.py then renders per-topology and per-case local nu/E maps with the region(s) marked.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _common as C

CASE, REG, N, NITER = 'auxetic_patch', 2e-3, 10, 120
csv_rows = []


def shape_spec(shape, Lx, Ly):
    cx, cy = 0.5 * Lx, 0.5 * Ly
    if shape == 'disc':
        return {'kind': 'circle', 'center': (cx, cy), 'radius': 0.20 * Lx}
    if shape == 'square':
        return {'kind': 'rect', 'center': (cx, cy), 'w': 0.36 * Lx, 'h': 0.36 * Ly}
    if shape == 'triangle':
        return {'kind': 'polygon', 'verts': C.triangle_verts(cx, cy, 0.24 * Lx)}
    if shape == 'ring':
        return {'kind': 'ring', 'center': (cx, cy), 'r_in': 0.12 * Lx, 'r_out': 0.24 * Lx}
    raise ValueError(shape)


# case -> (shape, (out_kind,out_val), (patch_kind,patch_val), field_to_show)
CASES = [
    ('disc_auxetic',     'disc',     ('nu', 0.30), ('nu', -0.30), 'nu'),
    ('square_auxetic',   'square',   ('nu', 0.30), ('nu', -0.30), 'nu'),
    ('triangle_auxetic', 'triangle', ('nu', 0.30), ('nu', -0.30), 'nu'),
    ('ring_auxetic',     'ring',     ('nu', 0.30), ('nu', -0.30), 'nu'),
    ('normal_in_auxetic', 'disc',    ('nu', -0.30), ('nu', 0.30), 'nu'),
    ('stiffE_in_soft',   'disc',     ('E', 0.70),  ('E', 1.60),  'E'),
    ('softE_in_stiff',   'disc',     ('E', 1.40),  ('E', 0.60),  'E'),
]


def run_case(topo, name, shape, out, patch, field, nd):
    prob, geo = C.make_case(topo, N)
    Lx, Ly = C.box(geo)
    p_idx, spec = C.region_shape(prob, shape_spec(shape, Lx, Ly))
    o_idx = np.setdiff1d(np.arange(prob.n_tri), p_idx)
    objs = [C.Objective(out[0], out[1], region=o_idx, weight=1.0),
            C.Objective(patch[0], patch[1], region=p_idx, weight=1.5)]
    res = C.optimize(prob, objs, mode='k', n_iter=NITER, reg=REG, verbose=False)
    C.apply_k_to_geo(geo, res['k']); C6 = C.sim_per_triangle_C6(geo)
    q = 0 if field == 'nu' else 1
    pin = C.c6_nuE(C.region_phys_C6(geo, C6, p_idx))[q]
    pout = C.c6_nuE(C.region_phys_C6(geo, C6, o_idx))[q]
    csv_rows.append((name, topo, field, patch[1], f'{pin:.3f}', out[1], f'{pout:.3f}'))
    C.save_network(os.path.join(nd, f'{name}__{topo}.npz'), geo, res['k'], C6, case=name, topo=topo,
                   N=N, field=field, region=spec, patch_target=patch[1], out_target=out[1],
                   patch_val=float(pin), out_val=float(pout))
    print(f"  {topo:12s} {name:18s} | patch {field}={pin:+.3f}(tgt{patch[1]})  out={pout:+.3f}", flush=True)


def run_decoupled(topo, nd):
    NU1, E1 = -0.30, 1.8                             # must match C.decoupled_ENu_design's recipe
    prob, geo = C.make_case(topo, N)
    res_k, C6, RE, RN, R_E, R_N, _, _ = C.decoupled_ENu_design(prob, geo, NITER, REG)
    rn = C.c6_nuE(C.region_phys_C6(geo, C6, R_N)); re = C.c6_nuE(C.region_phys_C6(geo, C6, R_E))
    csv_rows.append(('decoupled', topo, 'E&nu', NU1, f'{rn[0]:.3f}', E1, f'{re[1]:.3f}'))
    C.save_network(os.path.join(nd, f'decoupled__{topo}.npz'), geo, res_k, C6, case='decoupled',
                   topo=topo, N=N, field='decoupled', region=[RE, RN],
                   R_nu_nu=float(rn[0]), R_E_E=float(re[1]))
    print(f"  {topo:12s} decoupled          | R_nu nu={rn[0]:+.3f}(tgt{NU1})  R_E E={re[1]:.3f}(tgt{E1})", flush=True)


def main():
    nd = C.networks_dir(CASE)
    for topo in C.TOPO_IDS:
        for name, shape, out, patch, field in CASES:
            run_case(topo, name, shape, out, patch, field, nd)
        run_decoupled(topo, nd)
    C.write_csv(os.path.join(C.savedir(CASE), 'auxetic_patch_all.csv'),
                ['case', 'topology', 'field', 'patch_target', 'patch_sim', 'out_target', 'out_sim'], csv_rows)
    print('done')


if __name__ == '__main__':
    main()
