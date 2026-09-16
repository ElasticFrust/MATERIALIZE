r"""Does single-realization ANISOTROPY average out with cell size? (user, 2026-09-16)

The claim: on a disordered lattice anisotropy should average out. Tested two ways -- the SCALING of
(anisotropy - 1) against 1/sqrt(n_tri), which is what pure fluctuation of a statistically isotropic
ensemble must obey; and a Rayleigh test on the DIRECTION of the residual anisotropy, which separates
fluctuation (spread) from a systematic lattice or box artefact (clustered).

Reading: `Phase 5/results/coverage_sweep/COVERAGE_SWEEP.md` section 7.

Run:
    python "Phase 5/verifications/m2_anisotropy_scaling.py"
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
    g=dict(geo); k=np.asarray(k,float)/np.asarray(k,float).mean()
    g['bond_k']=k; g['tri_k']=k[np.asarray(geo['tri_bond'],np.int64)]
    prob=DesignProblem.from_geo(g)
    with torch.no_grad():
        out=prob.forward(torch.as_tensor(k),physical_units=True)
        nt,Et=c6_to_nuE_theta(prob.region_tensor(out['per_triangle'],None),ANG)
    nt,Et=nt.numpy(),Et.numpy()
    if not(np.isfinite(nt).all() and np.isfinite(Et).all() and Et.min()>0): return None
    return float(nt.mean()), float(Et.max()/Et.min()), float(ANG[int(np.argmax(Et))])

ETA=0.35
print('Does single-realization anisotropy AVERAGE OUT with cell size? eta=%.2f, k=1+tanh(a(l-1))'%ETA)
print('If it is pure fluctuation of an isotropic ensemble, (aniso-1) should fall like 1/sqrt(N_tri).')
print()
print('%-5s %-7s %-6s %10s %10s %14s %16s'%('N','n_tri','alpha','nu med','aniso med','(aniso-1)','(aniso-1)*sqrt(Ntri)'))
store={}
for al in (5.,30.):
    for N in (6,10,16,24):
        nus,ans,angs=[],[],[]
        for sd in range(10):
            g=MB.build_geometry(N,ETA,seed=sd)
            lr=np.hypot(*np.asarray(g['bond_R'],float).T)
            r=resp(g,1.0+np.tanh(al*(lr-1.0)))
            if r: nus.append(r[0]); ans.append(r[1]); angs.append(r[2])
        if not ans: continue
        nt=len(g['tri_bond']); am=float(np.median(ans))
        store[(al,N)]=(nt,am,np.array(angs))
        print('%-5d %-7d %-6.0f %10.3f %10.3f %14.3f %16.2f'
              %(N,nt,al,np.median(nus),am,am-1,(am-1)*np.sqrt(nt)))
print()
print('Is the residual anisotropy DIRECTION random across seeds (fluctuation) or clustered (systematic)?')
print('  Rayleigh test on the 2*theta of max E: R near 0 = uniform (fluctuation), near 1 = aligned.')
for (al,N),(nt,am,angs) in sorted(store.items()):
    z=np.exp(2j*angs)                      # theta and theta+pi are the same axis
    R=abs(z.mean())
    print('  alpha %2.0f  N=%-3d n_tri=%-5d  R = %.3f   %s'
          %(al,N,nt,R,'clustered -> systematic' if R>0.6 else 'spread -> fluctuation'))
