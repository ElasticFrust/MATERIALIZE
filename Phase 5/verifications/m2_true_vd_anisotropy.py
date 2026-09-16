r"""TRUE virtual distortion, and the ANISOTROPY axis the prior art never recorded.

VD, properly: the real lattice stays REGULAR and is never moved. A VIRTUAL copy is displaced, its
edge lengths are read, and `k = 1 + tanh(a(l_virt - 1))` is put on the real regular lattice. Only the
stiffness varies; the geometry does not.

`Phase 3/verifications/vd_demo/` is the prior art for this and its README is the reference. An
earlier version of the A1/d sweep conflated true VD with actually-deformed geometry -- a confusion
that README explicitly warns about ("the earlier a=5 eta-sweep plots are auxetic because their ACTUAL
geometry is deformed").

WHAT IS NEW HERE. `vd_demo` records nu and E against alpha, establishing that rigidity contrast alone
makes the regular lattice auxetic above alpha ~ 15-21. It does NOT record ANISOTROPY -- and anisotropy
is the axis the coverage question turns on, because true VD ought to be the clean route to ISOTROPIC
auxetic precisely by leaving the geometry perfectly regular.

Result: it is not. Reading in `Phase 5/results/coverage_sweep/COVERAGE_SWEEP.md` section 6.

Run:
    python "Phase 5/verifications/m2_true_vd_anisotropy.py"
"""
import os, sys, numpy as np, torch, warnings
warnings.filterwarnings('ignore')
REPO=os.path.abspath('.')
sys.path.insert(0,os.path.join(REPO,'Phase 3','verifications'))
import _common as C
from inverse_design import DesignProblem, ANG, c6_to_nuE_theta
import mesh_build as MB
torch.set_default_dtype(torch.float64)

def resp(geo,k):
    g=dict(geo); g['bond_k']=np.asarray(k,float); g['tri_k']=np.asarray(k,float)[np.asarray(geo['tri_bond'],np.int64)]
    prob=DesignProblem.from_geo(g)
    with torch.no_grad():
        out=prob.forward(torch.as_tensor(np.asarray(k,float)),physical_units=True)
        nt,Et=c6_to_nuE_theta(prob.region_tensor(out['per_triangle'],None),ANG)
    nt,Et=nt.numpy(),Et.numpy()
    if not(np.isfinite(nt).all() and np.isfinite(Et).all() and Et.min()>0): return None
    return float(nt.mean()), float(Et.max()/Et.min())

print('TRUE VD, per the vd_demo recipe: real lattice stays REGULAR; k read from a VIRTUAL copy.')
print('  real = build_geometry(N, 0)   virt = build_geometry(N, eta, seed)   k = 1+tanh(a*(l_virt-1))')
print('Prior art (vd_demo/README) gives nu vs alpha. ANISOTROPY was never recorded -- that is the gap.')
print()
print('%-5s %-6s %-6s %6s  %-24s %-22s'%('N','n_tri','eta','alpha','nu median [min,max]','aniso median (max)'))
for N in (10,20):
    real=MB.build_geometry(N,0.0,seed=0)
    for eta in (0.15,0.30,0.45):
        for al in (5.,15.,30.,60.):
            nus,ans=[],[]
            for sd in range(6):
                virt=MB.build_geometry(N,eta,seed=sd)
                if len(virt['bond_R'])!=len(real['bond_R']): continue
                MB.set_VD(virt,al)                      # k from VIRTUAL lengths
                k=np.asarray(virt['bond_k'],float); k=k/k.mean()
                r=resp(real,k)                          # applied to the REAL regular lattice
                if r: nus.append(r[0]); ans.append(r[1])
            if not nus: continue
            nus,ans=np.array(nus),np.array(ans)
            print('%-5d %-6d %-6.2f %6.0f  %+.3f [%+.3f,%+.3f]   %8.2f (%.2f)'
                  %(N,len(real['tri_bond']),eta,al,np.median(nus),nus.min(),nus.max(),
                    np.median(ans),ans.max()))
