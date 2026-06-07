"""Quick comparison: no-KKT / edge-KKT / full (edge+angle) KKT, standard and AW."""
import sys, numpy as np, time, torch
sys.path.insert(0, '.')
sys.path.insert(0, 'Phase 2')
import Disc_2_Cont_optimized as D2C
import forward_solver_torch as fst

ETA_VALUES = np.linspace(0.0, 0.5, 11)
N_TRIALS   = 10
SIZE       = (30, 30)
TRIM_FRAC  = 0.85

def directional_constants(C6):
    C0,C1,C2,C3,C4,C5 = C6
    C_V = np.array([[C0,C2,C1],[C2,C5,C4],[C1,C4,C3]])
    try: S = np.linalg.inv(C_V)
    except: return np.nan,np.nan,np.nan,np.nan
    def _s(n,d): return n/d if abs(d)>1e-15 else np.nan
    return _s(1,S[0,0]),_s(1,S[1,1]),_s(-S[1,0],S[0,0]),_s(-S[0,1],S[1,1])

CASES = [
    (False, False, False, 'Std'),
    (False, True,  False, 'Std+edge'),
    (False, True,  True,  'Std+full'),
    (True,  False, False, 'AW'),
    (True,  True,  False, 'AW+edge'),
    (True,  True,  True,  'AW+full'),
]

E_res  = {c[3]: [] for c in CASES}
nu_res = {c[3]: [] for c in CASES}

t0 = time.time()
for i_eta, eta in enumerate(ETA_VALUES):
    row_E  = {c[3]: [] for c in CASES}
    row_nu = {c[3]: [] for c in CASES}
    for trial in range(N_TRIALS):
        np.random.seed(100*i_eta + trial)
        DT = D2C.generate_foam_distort_first(SIZE, eta, trim_frac=TRIM_FRAC)
        solver, rigs, rl = fst.from_triangulation(DT)
        with torch.no_grad():
            for aw, kkt, ang, key in CASES:
                res = solver.forward(rigs, rl, area_weighted=aw,
                                     use_kkt=kkt, use_angle_kkt=ang)
                C6 = res['elastic_tensor'].numpy()
                Ex,Ey,nuxy,nuyx = directional_constants(C6)
                row_E [key].append(0.5*(Ex+Ey))
                row_nu[key].append(0.5*(nuxy+nuyx))
    for _,_,_,k in CASES:
        E_res [k].append(np.nanmedian(row_E [k]))
        nu_res[k].append(np.nanmedian(row_nu[k]))
    vals = "  ".join(
        f"{k}:E={np.nanmedian(row_E[k]):.4f}/nu={np.nanmedian(row_nu[k]):+.4f}"
        for _,_,_,k in CASES)
    print(f"eta={eta:.2f}  {vals}  [{time.time()-t0:.0f}s]", flush=True)

E0 = {k: E_res[k][0] for _,_,_,k in CASES}
print()
hdr = "  ".join(f"{k:>10}" for _,_,_,k in CASES)
print(f"{'eta':>4}  {hdr}")
print(f"           " + "  ".join(f"{'E/E0':>5}/{'nu':>5}" for _ in CASES))
for i, eta in enumerate(ETA_VALUES):
    row = "  ".join(
        f"{E_res[k][i]/E0[k]:5.3f}/{nu_res[k][i]:+5.3f}"
        for _,_,_,k in CASES)
    print(f"{eta:4.2f}  {row}")
print(f"\nDone in {time.time()-t0:.0f}s")
