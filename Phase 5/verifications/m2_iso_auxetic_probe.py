r"""Does eta + VD reach ISOTROPIC AUXETIC, and does anisotropy self-average with cell size?

The focused half of A1/d. The broad sweep (`m2_coverage_sweep.py`) showed VD contrast spans
nu +0.993 to -1.193 but left the isotropic column empty -- which turned out to be two defects in
that sweep rather than physics: four of its five (phi, psi) lattices are intrinsically
anisotropic, and `bravais_lattice(1,1)` has TWO diagonals of which only `a2-a1` is the regular
triangular lattice (`a1+a2` has bond length sqrt(3)).

This probe fixes both -- triangular, `a2-a1` only -- and varies the CELL SIZE, because
anisotropy at fixed disorder is a finite-size effect: a small cell looks anisotropic by
fluctuation even when the ensemble is isotropic.

Result and reading: `Phase 5/results/coverage_sweep/COVERAGE_SWEEP.md` SS3.

Run:
    python "Phase 5/verifications/m2_iso_auxetic_probe.py"
"""
import os, sys, numpy as np, torch, warnings
warnings.filterwarnings('ignore')
REPO=os.path.abspath('.')
sys.path.insert(0,os.path.join(REPO,'Phase 3','verifications'))
import _common as C
from inverse_design import DesignProblem, ANG, c6_to_nuE_theta
import mesh_build as MB
sys.path.insert(0,os.path.join(REPO,'Phase 5')); sys.path.insert(0,os.path.join(REPO,'Phase 5','m2'))
import seeds as S, fields as F
sys.path.insert(0,os.path.join(REPO,'Phase 5','verifications'))
from m2_coverage_sweep import geo_moved, response
torch.set_default_dtype(torch.float64)

print('TRIANGULAR lattice, diagonal a2-a1 (the TRUE regular one, bond length 1).')
print('Does anisotropy fall toward 1 as the cell grows? -- i.e. does disorder self-average?')
print()
print('%-6s %-7s %-6s %6s  %-28s %-28s'%('reps','n_tri','frac','VD','nu  (median [min,max])','anisotropy (median, p90)'))
for reps in (6,10,16):
    r=S.bravais_lattice(1.0,1.0,reps=reps,diagonal='a2-a1',eta=0.0,seed=0); geo0=r['geo']
    for frac,vd in ((0.0,5.0),(0.6,5.0),(0.9,2.0),(0.9,5.0)):
        nus,ans=[],[]
        for sd in range(8):
            if frac>0:
                pts,_=F.displace_safe(geo0,np.random.default_rng(sd),frac=frac,structure='white')
            else: pts=np.asarray(geo0['pts'],float)
            geo,ma=geo_moved(geo0,pts)
            if ma<=0: continue
            MB.set_VD(geo,vd)
            k=np.asarray(geo['bond_k'],float); k=k/k.mean()
            res=response(geo,k)
            if res: nus.append(res['nu']); ans.append(res['aniso'])
        if not nus: continue
        nus,ans=np.array(nus),np.array(ans)
        print('%-6d %-7d %-6.1f %+6.1f  %+.3f [%+.3f,%+.3f]%s %8.2f  p90 %8.2f'
              %(reps,len(geo['tri_bond']),frac,vd,np.median(nus),nus.min(),nus.max(),' '*6,
                np.median(ans),np.percentile(ans,90)))
