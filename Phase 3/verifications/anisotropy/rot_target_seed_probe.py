"""
anisotropy / rot_target_seed_probe — is the crystal's ν(θ) recreatable on disorder at ANY orientation?
Designs both the ORIGINAL crystal target (peak 25°) and the 90°-ROTATED target (peak 115°) on four
disorder_hi realisations (seeds 0-3) and reports the achieved sim ν range.

Finding (see rot_target_seed_probe_log.txt): ORIG fits on all seeds; ROT fits on 3/4 (seed 0 chokes,
seed 1 fits ROT even better than ORIG). Disorder is isotropic ON AVERAGE (any orientation reachable),
but a FIXED finite patch has weak quenched anisotropy that can disfavour a given orientation — seed 0
(used by fig1) happens to disfavour 115°, which is why designing the rotated target on it collapses to
flat. The clean way to reorient the response is to rotate the regular crystal lattice (fig1c, exact);
the disorder route also works, just not on every realisation. Run from Phase 3/verifications/.
"""
import os, sys
import numpy as np
sys.path.insert(0, '.')
import _common as C
TH = C.ANG
def rot(geo, deg):
    a=np.radians(deg); R=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]]); g=dict(geo)
    for k in ('pts','edge_vecs','bond_R','centroids','tri_verts'): g[k]=geo[k]@R.T
    g['BL1']=R@geo['BL1']; g['BL2']=R@geo['BL2']; g['actual_len2']=(g['edge_vecs']**2).sum(-1); return g
cg=C.make_crystal(4.0,1.0,half=4.0); cg['bond_k']=np.ones(len(cg['bond_R'])); cg['tri_k']=cg['bond_k'][cg['tri_bond']]
nu_orig=C.nu_E_theta(C.sim_region_C6(cg),TH)[0]
cg90=rot(cg,90.0); cg90['bond_k']=np.ones(len(cg90['bond_R'])); cg90['tri_k']=cg90['bond_k'][cg90['tri_bond']]
nu_t=C.nu_E_theta(C.sim_region_C6(cg90),TH)[0]
print(f"orig target peak {np.degrees(TH[np.argmax(nu_orig)]):.0f}deg, rot target peak {np.degrees(TH[np.argmax(nu_t)]):.0f}deg range[{nu_t.min():+.2f},{nu_t.max():+.2f}]",flush=True)
# ALSO design the ORIGINAL target on each seed as a control
for seed in [0,1,2,3]:
    for tag,tgt in [('ROT ',nu_t),('ORIG',nu_orig)]:
        prob,geo=C.make_case('disorder_hi',12,seed=seed)  # N=12 smaller/faster probe
        r=C.optimize(prob,[C.Objective('nu_theta',tgt)],mode='k',n_iter=350,n_restarts=3,reg=5e-3,verbose=False)
        C.apply_k_to_geo(geo,r['k'])
        nu=C.nu_E_theta(C.region_phys_C6(geo,C.sim_per_triangle_C6(geo),None),TH)[0]
        print(f"  seed{seed} {tag}: sim nu[{nu.min():+.2f},{nu.max():+.2f}] kmed={np.median(r['k'].detach().numpy()):.2f}",flush=True)
