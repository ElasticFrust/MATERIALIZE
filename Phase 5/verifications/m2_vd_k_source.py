r"""Three k SOURCES at one fixed geometry -- VD applies to irregular lattices too (user, 2026-09-16).

Only the stiffness source changes:
    (1) uniform             k = 1
    (2) length-keyed REAL   k = 1 + tanh(a(l_real - 1))        <- what mesh_build.set_VD computes
    (3) further virtual     k = 1 + tanh(a(l_virt - l_real))   <- an EXTRA virtual distortion

Note `mesh_build.set_VD` can only express l0 = 1: it hardcodes `dl = |R| - 1.0`, so on a lattice
whose natural spacing is not 1 it keys off the wrong reference.

Reading: `Phase 5/results/coverage_sweep/COVERAGE_SWEEP.md` section 7.

Run:
    python "Phase 5/verifications/m2_vd_k_source.py"
"""
import os, sys, numpy as np, torch, warnings
warnings.filterwarnings('ignore')
REPO=os.path.abspath('.')
sys.path.insert(0,os.path.join(REPO,'Phase 3','verifications'))
import _common as C
from inverse_design import DesignProblem, ANG, c6_to_nuE_theta
import mesh_build as MB
sys.path.insert(0,os.path.join(REPO,'Phase 5')); sys.path.insert(0,os.path.join(REPO,'Phase 5','m2'))
import fields as F
torch.set_default_dtype(torch.float64)

def addbox(g):
    bu=np.asarray(g['bond_u'],np.int64); bv=np.asarray(g['bond_v'],np.int64); p=np.asarray(g['pts'],float)
    sh=np.abs(np.asarray(g['bond_R'],float)-(p[bv]-p[bu]))
    g['BL1']=np.array([sh[:,0][sh[:,0]>1e-9].min(),0.]); g['BL2']=np.array([0.,sh[:,1][sh[:,1]>1e-9].min()]); return g

def virtual_lengths(geo, pts):
    bu=np.asarray(geo['bond_u'],np.int64); bv=np.asarray(geo['bond_v'],np.int64)
    p0=np.asarray(geo['pts'],float); shift=np.asarray(geo['bond_R'],float)-(p0[bv]-p0[bu])
    R=np.asarray(pts,float)[bv]-np.asarray(pts,float)[bu]+shift
    return np.hypot(R[:,0],R[:,1])

def resp(geo,k):
    g=dict(geo); k=np.asarray(k,float)/np.asarray(k,float).mean()
    g['bond_k']=k; g['tri_k']=k[np.asarray(geo['tri_bond'],np.int64)]
    prob=DesignProblem.from_geo(g)
    with torch.no_grad():
        out=prob.forward(torch.as_tensor(k),physical_units=True)
        nt,Et=c6_to_nuE_theta(prob.region_tensor(out['per_triangle'],None),ANG)
    nt,Et=nt.numpy(),Et.numpy()
    if not(np.isfinite(nt).all() and np.isfinite(Et).all() and Et.min()>0): return None
    return float(nt.mean()), float(Et.max()/Et.min())

print('VD ON AN IRREGULAR LATTICE. Real geometry FIXED and irregular; only the k SOURCE changes.')
print('  (1) uniform            k = 1')
print('  (2) length-keyed REAL  k = 1+tanh(a(l_real - 1))     <- what mesh_build.set_VD does')
print('  (3) TRUE VD            k = 1+tanh(a(l_virt - l_real))  <- virtually distort the IRREGULAR lattice')
print()
print('%-6s %-5s %-24s %-24s %-24s'%('eta1','alpha','(1) uniform','(2) length-keyed real','(3) TRUE VD on irregular'))
print('%-6s %-5s %-24s %-24s %-24s'%('','','nu      aniso','nu      aniso','nu      aniso'))
for eta1 in (0.0,0.2,0.35):
    for al in (5.,15.,30.):
        rows=[[],[],[]]
        for sd in range(6):
            real=MB.build_geometry(12,eta1,seed=sd); MB.set_VD(real,0); addbox(real)
            lr=np.hypot(*np.asarray(real['bond_R'],float).T)
            pv,_=F.displace_safe(real,np.random.default_rng(100+sd),frac=0.6,structure='white')
            lv=virtual_lengths(real,pv)
            for i,k in enumerate((np.ones_like(lr),
                                  1.0+np.tanh(al*(lr-1.0)),
                                  1.0+np.tanh(al*(lv-lr)))):
                r=resp(real,k)
                if r: rows[i].append(r)
        out=[]
        for r in rows:
            if not r: out.append('   --        --  '); continue
            a=np.array(r); out.append('%+7.3f  %7.2f       '%(np.median(a[:,0]),np.median(a[:,1])))
        print('%-6.2f %-5.0f %s'%(eta1,al,''.join(out)))
