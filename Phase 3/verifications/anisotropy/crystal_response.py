"""
anisotropy / crystal_response — closed-form-checkable elastic response of a CRYSTALLINE lattice.

Build the single-site triangular crystal make_crystal(phi, psi): the regular lattice sheared/stretched
by M=[[1,(phi-1)/√3],[0,psi]] so v1=(1,0), v2=(phi/2, psi*sqrt3/2) — with the BOND TOPOLOGY PRESERVED
(the long sheared bond v2 stays a real spring; make_lattice would have Delaunay-reduced it away). Set
k=1 on every bond. Because it is a one-site Bravais lattice, an affine macro strain is already the
equilibrium — the non-affine fluctuation W vanishes — so the homogenised elastic tensor is EXACT on
an arbitrarily small periodic patch (no large-N average needed). We:

  1. verify W=0 empirically: under each unit macro-strain mode the per-triangle strain is spatially
     UNIFORM (std across triangles ~ 1e-15), i.e. every triangle strains affinely with the macro load;
  2. read the homogenised response two ways and confirm they agree:
       (a) the code's PBC relaxation  (sim_region_C6 -> nu_E_theta),
       (b) a by-hand affine (Born) sum  C_ijkl = (1/A) sum_b k_b r_i r_j r_k r_l / |r|^2
           over the lattice's own bond set, which is exact precisely because W=0.

Prints the Voigt stiffness, the engineering constants E_x,E_y,nu_xy,nu_yx,G, and the directional
nu(theta)/E(theta); saves crystal_response.png (network + nu(theta) & E(theta) polar).
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _common as C

CASE = 'anisotropy'
PHI, PSI, HALF = 3.0, 2.0, 4.0                 # crystalline lattice params; small square patch


def born_affine_C(geo):
    """By-hand affine (W=0) stiffness: C_ijkl = (1/A_cell) sum_bonds k r_i r_j r_k r_l/|r|^2, returned
    as the 2D Voigt 3x3 [xx,yy,xy]. Uses ONE primitive cell's worth of bonds = (#bonds)/(#sites)."""
    R = geo['bond_R']; k = geo['bond_k']                      # bond vectors & stiffnesses (k=1)
    L2 = (R ** 2).sum(1)
    A_box = abs(np.linalg.det(np.column_stack([geo['BL1'], geo['BL2']])))
    A_cell = A_box / len(geo['pts'])                          # area per lattice site = per primitive cell
    nb_per_cell = len(R) / len(geo['pts'])                    # bonds per site (=3 for triangular)
    # accumulate the fully-symmetric 4-tensor components (per box), then normalise to per unit area
    Cxxxx = (k * R[:, 0] ** 4 / L2).sum()
    Cyyyy = (k * R[:, 1] ** 4 / L2).sum()
    Cxxyy = (k * R[:, 0] ** 2 * R[:, 1] ** 2 / L2).sum()
    Cxxxy = (k * R[:, 0] ** 3 * R[:, 1] / L2).sum()
    Cyyxy = (k * R[:, 0] * R[:, 1] ** 3 / L2).sum()
    fac = 1.0 / A_box                                         # sum is already over the whole box
    Cv = np.array([[Cxxxx, Cxxyy, Cxxxy],
                   [Cxxyy, Cyyyy, Cyyxy],
                   [Cxxxy, Cyyxy, Cxxyy]]) * fac              # C_1212 = C_1122 (Cauchy relation)
    return Cv, A_cell, nb_per_cell


def engineering_from_Cv(Cv):
    """E_x,E_y,nu_xy,nu_yx,G from a 2D Voigt stiffness with tensor shear (C[2,2]=C_1212)."""
    S = np.linalg.inv(Cv)                                     # compliance (tensor convention)
    Ex, Ey = 1.0 / S[0, 0], 1.0 / S[1, 1]
    nu_xy = -S[0, 1] / S[0, 0]                                # load x -> contraction in y
    nu_yx = -S[0, 1] / S[1, 1]
    G = 1.0 / S[2, 2]                                         # tensor shear modulus C_1212
    return Ex, Ey, nu_xy, nu_yx, G


def main():
    geo = C.make_crystal(PHI, PSI, half=HALF)
    geo['bond_k'] = np.ones(len(geo['bond_R'])); geo['tri_k'] = geo['bond_k'][geo['tri_bond']]
    print(f"  [crystal_response] phi={PHI} psi={PSI}  sites={len(geo['pts'])} "
          f"bonds={len(geo['bond_R'])} tri={len(geo['simplices'])}", flush=True)

    # --- 1. verify W=0: per-triangle strain is uniform under each unit macro mode ---
    eps, _ = C.unit_mode_response(geo)                        # eps[k]: (nt,2,2) actual strain, mode k
    for kmode, nm in enumerate(['exx', 'eyy', 'exy']):
        e = eps[kmode].reshape(len(eps[kmode]), 4)
        print(f"    mode {nm}: per-triangle strain spread (max std over the 4 comps) = "
              f"{e.std(0).max():.2e}   (0 => affine => W=0)", flush=True)

    # --- 2a. code homogenised response (PBC relaxation) ---
    C6 = C.sim_region_C6(geo)                                 # physical homogenised 6-vector
    th = C.ANG
    nu_th, E_th = C.nu_E_theta(C6, th)
    idx = {a: int(np.argmin(np.abs(np.degrees(th) - a))) for a in (0, 45, 90, 135)}

    # --- 2b. by-hand affine (Born) response ---
    Cv, A_cell, nb = born_affine_C(geo)
    Ex, Ey, nu_xy, nu_yx, G = engineering_from_Cv(Cv)

    print(f"    cell area A={A_cell:.4f}  bonds/cell={nb:.1f}", flush=True)
    print("    Voigt stiffness C (by-hand affine, [xx,yy,xy]):", flush=True)
    for row in Cv:
        print("       [{:8.4f} {:8.4f} {:8.4f}]".format(*row), flush=True)
    print(f"    by-hand:  E_x={Ex:.4f}  E_y={Ey:.4f}  nu_xy={nu_xy:+.4f}  nu_yx={nu_yx:+.4f}  G={G:.4f}",
          flush=True)
    print(f"    code sim: E(0)={E_th[idx[0]]:.4f}  E(90)={E_th[idx[90]]:.4f}  "
          f"nu(0)={nu_th[idx[0]]:+.4f}  nu(90)={nu_th[idx[90]]:+.4f}", flush=True)
    print(f"    directional: nu(theta) in [{nu_th.min():+.4f},{nu_th.max():+.4f}]  "
          f"E(theta) in [{E_th.min():.4f},{E_th.max():.4f}]  E_max/E_min={E_th.max()/E_th.min():.2f}",
          flush=True)

    C.write_csv(os.path.join(C.savedir(CASE), 'crystal_response.csv'),
                ['theta_deg', 'nu_theta', 'E_theta'],
                [(f'{np.degrees(t):.1f}', f'{nu_th[i]:.4f}', f'{E_th[i]:.4f}') for i, t in enumerate(th)])

    # --- plot: network + nu(theta) line + E(theta) line (Cartesian) ---
    deg = np.degrees(th)
    fig = plt.figure(figsize=(15, 5))
    a0 = fig.add_subplot(1, 3, 1)
    C.draw_network(a0, geo, geo['bond_k'], cmap='viridis', lw_scale=3.0)
    a0.set_title(f'crystal  φ={PHI:g}, ψ={PSI:g}  (k=1)', fontsize=11)

    a1 = fig.add_subplot(1, 3, 2)
    a1.plot(deg, nu_th, color='#d62728', lw=2)
    a1.axhline(0, color='0.6', lw=.6)
    a1.set_xlabel('direction θ (deg)'); a1.set_ylabel('Poisson ratio ν(θ)')
    a1.set_xlim(0, 180); a1.set_xticks(range(0, 181, 45)); a1.grid(alpha=.3)
    a1.set_title(f'ν(θ)  [{nu_th.min():+.2f}, {nu_th.max():+.2f}]', fontsize=11)

    a2 = fig.add_subplot(1, 3, 3)
    a2.plot(deg, E_th, color='#2ca02c', lw=2)
    a2.set_xlabel('direction θ (deg)'); a2.set_ylabel('Young modulus E(θ)')
    a2.set_xlim(0, 180); a2.set_xticks(range(0, 181, 45)); a2.set_ylim(bottom=0); a2.grid(alpha=.3)
    a2.set_title(f'E(θ)  [{E_th.min():.2f}, {E_th.max():.2f}]  E$_{{max}}$/E$_{{min}}$='
                 f'{E_th.max()/E_th.min():.2f}', fontsize=11)

    fig.suptitle(f'{CASE} / crystal_response — Bravais crystal φ={PHI:g} ψ={PSI:g}, k=1, W=0 (affine, exact)',
                 fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(C.savedir(CASE), 'crystal_response.png')
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f'saved {os.path.basename(path)}')


if __name__ == '__main__':
    main()
