"""B-1 reproduction harness — is `test_inverse_design` [15] deterministic?

Audit B-1 recorded [15]'s `solver-vs-physical(tensor)` swinging over eleven orders between runs
(`1.6e-12, 1.6e-12, 1.4e-04, 1.6e-12, 1.17e-02(FAIL)`), from what was believed to be the same code.
It did not reproduce (register B-1 stage 3, 2026-08-16); this is the harness that established that,
kept so any recurrence can be checked against exactly the same probes rather than re-derived.

**The trap this script exists to avoid.** Test [15] loops over THREE cases and asserts on each:

    (phi, psi, eta, seed) = (1.0, 1.0, 0.0, 0) , (1.0, 1.0, 0.35, 1) , (1.0, 0.6, 0.0, 2)

The original B-1 bisect probed only case 1 — the regular lattice at eta=0, where **W ≡ 0**. That case
is degenerate and trivially stable (it is the same blindness `CLAUDE.md` §3 warns about for the
crystal gate), so it reported "clean" while never touching the case the failure was attributed to.
**Every mode here probes all three cases and reports the worst**, exactly as the test does.

Modes:
  fingerprint  build each case, print md5 of pts / simplices / k plus `vs` — checks the INPUT is
               identical across processes before blaming the arithmetic
  repeat N     run the probe N times in-process (arithmetic order) — expect bit-identical
  bisect       run the real suite's preceding tests in order, re-probing all 3 cases after each,
               to catch state left behind by an earlier test
  commits ...  run the probe against a list of git revisions via throwaway worktrees — the check
               that separates "nondeterminism" from "the tree was moving underneath the runs"

Run:  python b1_reproduce.py fingerprint
      python b1_reproduce.py repeat 30
      python b1_reproduce.py bisect
      python b1_reproduce.py commits 55850e9 201b2ab a522e42
"""
import hashlib
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
P3 = os.path.dirname(HERE)
REPO = os.path.dirname(P3)
sys.path.insert(0, P3)
sys.path.insert(0, HERE)

import _common as C                                     # noqa: E402  (wires the solver stack)
import physical_homog as PH                             # noqa: E402
import sim_assembly as SA                               # noqa: E402
from inverse_design import DesignProblem                # noqa: E402

torch.set_default_dtype(torch.float64)

# The three cases of test_inverse_design.test_homogenization ([15]), verbatim.
CASES = [(1.0, 1.0, 0.0, 0), (1.0, 1.0, 0.35, 1), (1.0, 0.6, 0.0, 2)]

# The suite's tests that run BEFORE [15], in the order test_inverse_design.py runs them.
PRECEDING = ['test_round_trip', 'test_property_auxetic', 'test_property_E', 'test_local_region',
             'test_mixed_global_local', 'test_large_N_adjoint', 'test_open_domain',
             'test_directional', 'test_constrain_isotropic', 'test_strain_stress_equivalence',
             'test_strain_stress_autograd', 'test_strain_stress_design',
             'test_homogeneity_regularizer']


def _h(a):
    """Short md5 of an array's bytes — for checking bit-identity, not for security."""
    return hashlib.md5(np.ascontiguousarray(a).tobytes()).hexdigest()[:8]


def build_case(phi, psi, eta, seed):
    """One of [15]'s cases, built exactly as the test builds it (k length depends on the mesh)."""
    geo = C.make_lattice(phi, psi, half=6.0, eta=eta, seed=seed)
    rng = np.random.default_rng(seed)
    k = 0.5 + rng.random(len(geo['bond_u']))
    geo['bond_k'] = k
    geo['tri_k'] = k[geo['tri_bond']]
    return geo, k


def probe_case(phi, psi, eta, seed):
    """[15]'s solver-vs-oracle tensor comparison for one case. Returns (vs, solver-C6 hash, geo, k)."""
    geo, k = build_case(phi, psi, eta, seed)
    free = np.arange(2, 2 * len(geo['pts']))
    cs = C.solver_region_C6(DesignProblem.from_geo(geo), torch.as_tensor(k))
    c_solver = np.array([[cs[0], cs[2], cs[1]], [cs[2], cs[5], cs[4]], [cs[1], cs[4], cs[3]]])
    c_phys = PH.energy_C(geo, free, SA.assemble_K_faff)          # INDEPENDENT oracle
    vs = float(np.abs(c_solver - c_phys).max() / np.abs(c_phys).max())
    return vs, _h(np.asarray(cs)), geo, k


def probe_all():
    """All three cases. Returns (list of vs, list of solver hashes) — worst is what [15] asserts on."""
    vs, hs = [], []
    for c in CASES:
        v, h, _, _ = probe_case(*c)
        vs.append(v); hs.append(h)
    return vs, hs


def _line(tag, vs, hs):
    flag = '   <-- EXCEEDS [15] assert (1e-2)' if max(vs) >= 1e-2 else ''
    return (f"{tag:<34} " + " ".join(f"c{i+1}={v:.3e}" for i, v in enumerate(vs))
            + f"  WORST={max(vs):.3e}  hashes={','.join(hs)}{flag}")


def mode_fingerprint():
    for i, c in enumerate(CASES):
        vs, hsol, geo, k = probe_case(*c)
        print(f"  case{i+1} eta={c[2]:<5} nb={len(geo['bond_u'])} ntri={len(geo['simplices'])} "
              f"pts={_h(geo['pts'])} simp={_h(geo['simplices'])} k={_h(k)} "
              f"solverC6={hsol} vs={vs:.4e}", flush=True)
    print(f"  threads: torch={torch.get_num_threads()} omp={os.environ.get('OMP_NUM_THREADS','unset')}")


def mode_repeat(n):
    """Repeat in-process. VALUE determinism and BIT determinism are reported SEPARATELY, because
    they are different findings: bit-level flips (B-1 stage 2) are expected — BLAS/allocator warm-up
    makes the first call differ from later ones — and are harmless as long as the VALUES hold. Only a
    spread in the values is a B-1 stage-3 recurrence. Do not read a hash difference as a defect."""
    vals, hashes, rows = set(), set(), []
    for i in range(n):
        vs, hs = probe_all()
        vals.add(tuple(f"{v:.9e}" for v in vs))
        hashes.add(tuple(hs))
        rows.append((i + 1, vs, hs))
        if i == 0:
            print(_line('run 1', vs, hs), flush=True)
    spread = max(max(r[1]) for r in rows) - min(max(r[1]) for r in rows)
    print(f"  {n} repeats -> {len(vals)} distinct VALUE set(s), {len(hashes)} distinct BIT pattern(s)")
    print(f"  worst-case spread across repeats: {spread:.2e}")
    if len(vals) == 1:
        print("  [VALUES DETERMINISTIC]" + ("" if len(hashes) == 1 else
              "  (bits differ run-to-run: B-1 stage 2, expected, harmless)"))
    else:
        print("  [VALUES DIFFER -- possible B-1 stage-3 recurrence, investigate]")
        for i, vs, hs in rows:
            print(f"    run {i}: " + " ".join(f"{v:.9e}" for v in vs))


def mode_bisect():
    import test_inverse_design as T
    torch.manual_seed(0)
    vs, hs = probe_all()
    print(_line('(nothing yet)', vs, hs), flush=True)
    for name in PRECEDING:
        try:
            getattr(T, name)()
        except Exception as e:                                   # a failing test is itself signal
            print(f"   ({name} raised {type(e).__name__}: {e})", flush=True)
        vs, hs = probe_all()
        print(_line(name, vs, hs), flush=True)


def mode_commits(revs):
    """Probe each revision in a throwaway worktree — separates nondeterminism from a moving tree."""
    child = os.path.join(HERE, os.path.basename(__file__))
    for rev in revs:
        wt = os.path.join(tempfile.gettempdir(), f'b1_wt_{rev}')
        subprocess.run(['git', 'worktree', 'remove', '--force', wt], cwd=REPO,
                       capture_output=True)
        r = subprocess.run(['git', 'worktree', 'add', '-q', '--detach', wt, rev], cwd=REPO,
                           capture_output=True, text=True)
        if r.returncode:
            print(f"  {rev}: worktree failed: {r.stderr.strip()}"); continue
        out = subprocess.run([sys.executable, child, 'fingerprint'],
                             cwd=os.path.join(wt, 'Phase 3', 'verifications'),
                             capture_output=True, text=True)
        worst = [ln for ln in out.stdout.splitlines() if 'vs=' in ln]
        date = subprocess.run(['git', 'log', '-1', '--format=%ad', '--date=format:%m-%d %H:%M', rev],
                              cwd=REPO, capture_output=True, text=True).stdout.strip()
        if worst:
            v = max(float(ln.split('vs=')[1]) for ln in worst)
            print(f"  {rev}  {date}  WORST={v:.3e}", flush=True)
        else:
            print(f"  {rev}  {date}  FAILED: {out.stderr.strip().splitlines()[-1:]}", flush=True)
        subprocess.run(['git', 'worktree', 'remove', '--force', wt], cwd=REPO, capture_output=True)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else 'fingerprint'
    print(f"B-1 reproduction harness -- mode={mode}")
    if mode == 'fingerprint':
        mode_fingerprint()
    elif mode == 'repeat':
        mode_repeat(int(sys.argv[2]) if len(sys.argv) > 2 else 10)
    elif mode == 'bisect':
        mode_bisect()
    elif mode == 'commits':
        mode_commits(sys.argv[2:])
    else:
        print(__doc__); sys.exit(2)


if __name__ == '__main__':
    main()
