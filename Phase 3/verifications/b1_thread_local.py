r"""Does BLAS THREAD COUNT change a design outcome? (`test_inverse_design` [4], test_local_region)

WHY THIS EXISTS
---------------
The 2026-08-22 B-1 overnight run (`b1_overnight.py`, arms alternating 1-thread / default) found
[4] failing on **11/11** 1-thread runs at exactly nu = -0.128 and passing on **10/10** default-thread
runs at exactly -0.150. But that harness alternates arms strictly (1,0,1,0,...), so thread count is
perfectly CONFOUNDED with run parity -- any alternating-in-time effect (thermal, cache, disk) would
produce an identical signature.

This script breaks the confound: it runs [4]'s problem ALONE -- no suite context, no alternation --
so the only difference between the two invocations is the thread count.

MEASURED 2026-08-23 (seed 4, N=16, eta=0.2, reg=0.02, n_iter=120):
    threads=1  achieved=-0.127929  err=0.096566  FAIL
    threads=4  achieved=-0.149881  err=0.000215  PASS
i.e. a ~1e-33 threading difference in the forward path (already on record) is amplified by 120
optimiser iterations into a DIFFERENT BASIN: nu moves by 0.022 and the objective error by 450x.

CONSEQUENCE FOR M2: a training set labelled by this design path is thread-count dependent, so the
dataset generator must PIN the thread count and record it as provenance.

Note `err` is max_theta |nu(theta) - target| while `achieved` is the theta-MEAN (see
`inverse_design.validate`), so the failing design is also angularly spread, not merely offset.

Run:  C:\Users\doron\anaconda3\python.exe "Phase 3/verifications/b1_thread_local.py" [n_threads]
        n_threads   1..N, or 0 to leave the process default. Default 0.
"""
import os
import sys

n = int(sys.argv[1]) if len(sys.argv) > 1 else 0
if n:                                     # must precede the torch import to bind the BLAS pools
    os.environ['OMP_NUM_THREADS'] = os.environ['MKL_NUM_THREADS'] = str(n)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C                                              # noqa: F401  (wires sys.path)
import torch

if n:
    torch.set_num_threads(n)

from inverse_design import DesignProblem, Objective, optimize, validate

# [4]'s problem, reproduced verbatim from test_inverse_design.test_local_region.
prob = DesignProblem.periodic(N=16, eta=0.2, seed=4)
centre = prob.centroids.mean(0)
span = prob.centroids[:, 0].max() - prob.centroids[:, 0].min()
region = prob.region_in_circle(centre, radius=0.20 * span)
objs = [Objective('nu', target=-0.15, region=region, weight=1.0)]

res = optimize(prob, objs, mode='k', n_iter=120, reg=0.02, verbose=False)
rep = validate(prob, res['k'], res['l0'], objs)[0]
print(f'threads={torch.get_num_threads():2d}  achieved={rep["achieved"]:+.6f}  '
      f'err={rep["err"]:.6f}  {"PASS" if rep["err"] < 0.03 else "FAIL"}')
