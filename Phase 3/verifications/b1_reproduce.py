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
  suitectx N   RE-ESCALATION DIAGNOSTIC (2026-08-16). Reproduce the suite context (run the 13
               preceding tests), then probe N times logging the FULL tensors — per case, and
               component-wise. [15] reports only max-over-3-cases, so which case deviates, and
               whether the deviation is ONE tensor component or diffuse, has never been recorded.
               That is the sharpest unused clue: the anomaly appears only in suite context (the
               isolated probe has spread 0.00e+00), so `repeat` cannot see it and this can.

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


def probe_case_full(phi, psi, eta, seed):
    """Like probe_case, but returns the two 3x3 tensors so the deviation can be localised."""
    geo, k = build_case(phi, psi, eta, seed)
    free = np.arange(2, 2 * len(geo['pts']))
    cs = C.solver_region_C6(DesignProblem.from_geo(geo), torch.as_tensor(k))
    c_solver = np.array([[cs[0], cs[2], cs[1]], [cs[2], cs[5], cs[4]], [cs[1], cs[4], cs[3]]])
    c_phys = PH.energy_C(geo, free, SA.assemble_K_faff)
    return c_solver, c_phys


def mode_suitectx(n):
    """Suite context, then N fully-logged probes. Prints a line per (repeat, case); on any case whose
    error exceeds the isolated baseline by >100x, dumps the component-wise |dC|/max|C| matrix."""
    import test_inverse_design as T
    BASE = {0: 2.205e-13, 1: 1.604e-12, 2: 2.202e-13}          # isolated, measured
    torch.manual_seed(0)
    print('  establishing suite context (13 preceding tests)...', flush=True)
    for name in PRECEDING:
        try:
            getattr(T, name)()
        except Exception as e:
            print(f'   ({name} raised {type(e).__name__}: {e})', flush=True)
    print('  context established; probing\n', flush=True)
    hits = 0
    for r in range(n):
        for ci, case in enumerate(CASES):
            cs, cp = probe_case_full(*case)
            scale = np.abs(cp).max()
            vs = float(np.abs(cs - cp).max() / scale)
            flag = ''
            if vs > 100 * BASE[ci]:
                hits += 1
                flag = f'   *** {vs / BASE[ci]:.1e}x BASELINE ***'
            print(f'  rep {r+1:3d} case{ci+1} vs={vs:.4e}{flag}', flush=True)
            if flag:
                D = np.abs(cs - cp) / scale
                print('        component-wise |dC|/max|C|  (Voigt xx,yy,xy):', flush=True)
                for row in D:
                    print('          ' + '  '.join(f'{v:.3e}' for v in row), flush=True)
                print(f'        solver diag={np.diag(cs)}  phys diag={np.diag(cp)}', flush=True)
    print(f'\n  {n} repeats x 3 cases -> {hits} deviation(s) above 100x baseline')


STAGES = ('bare', 'W', 'per_triangle', 'elastic_tensor')      # the solver's pipeline, in order


def stage_hashes(phi, psi, eta, seed):
    """Hash EVERY stage of one forward solve, plus the inputs, in pipeline order.

    The point (user's suggestion, 2026-08-17): B-1 is only ever observed at the END of the pipeline,
    so we cannot tell WHERE the deviation is born. Hashing each stage localises it:

        geometry/k differ      -> the INPUT construction, not the solve
        bare differs           -> A(s) assembly
        W differs, bare same   -> the constrained SOLVE (the KKT / Woodbury path)
        per_triangle differs, W same -> the CONTRACTION (1+W)^T A (1+W)  <-- the A-0 channel
        elastic_tensor differs, per_triangle same -> the final average/reduction
    """
    geo, k = build_case(phi, psi, eta, seed)
    prob = DesignProblem.from_geo(geo)
    with torch.no_grad():
        out = prob.forward(torch.as_tensor(k), physical_units=True)
    h = {'geo': _h(geo['pts']) + _h(geo['simplices']), 'k': _h(k)}
    for s in STAGES:
        v = out[s]
        h[s] = _h(v.detach().cpu().numpy() if hasattr(v, 'detach') else np.asarray(v))
    return h


def mode_stages(n, ctx=True):
    """Repeat the stage-hash probe and report the FIRST stage that ever disagrees.

    With ctx=True the 13 preceding suite tests run first, because B-1 has never been observed
    without suite context (0 anomalies in 30 isolated processes)."""
    if ctx:
        import test_inverse_design as T
        torch.manual_seed(0)
        print('  establishing suite context (13 preceding tests)...', flush=True)
        for name in PRECEDING:
            try:
                getattr(T, name)()
            except Exception as e:
                print(f'   ({name} raised {type(e).__name__})', flush=True)
        print('  context established\n', flush=True)

    keys = ('geo', 'k') + STAGES
    seen = {c: {kk: {} for kk in keys} for c in range(len(CASES))}
    for r in range(n):
        for ci, case in enumerate(CASES):
            h = stage_hashes(*case)
            for kk in keys:
                seen[ci][kk].setdefault(h[kk], []).append(r + 1)
        if (r + 1) % 5 == 0:
            print(f'  {r+1}/{n}', flush=True)

    print()
    any_split = False
    for ci in range(len(CASES)):
        line, first_split = [], None
        for kk in keys:
            nv = len(seen[ci][kk])
            line.append(f'{kk}:{nv}')
            if nv > 1 and first_split is None:
                first_split = kk
        flag = ''
        if first_split:
            any_split = True
            flag = f'   <-- FIRST SPLIT AT: {first_split}'
        print(f'  case{ci+1}  ' + '  '.join(line) + flag)
        if first_split:
            for hv, runs in seen[ci][first_split].items():
                print(f'        {hv}  on runs {runs[:8]}{"..." if len(runs) > 8 else ""}')
    print()
    print('  distinct-value count per stage; 1 = deterministic across all repeats.')
    if not any_split:
        print(f'  NO stage split over {n} repeats x {len(CASES)} cases '
              f'-- B-1 did not fire; this is a BOUND, not an absence.')


def mode_imports():
    """Does IMPORT ORDER change the result? (user's hypothesis, 2026-08-17)

    Re-imports the stack in a different order in a fresh interpreter and compares stage hashes to
    the canonical order. A difference would make B-1 an import/initialisation-order effect."""
    import subprocess
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(os.path.dirname(here))
    # `_common` wires the rest of sys.path, so an order that imports something else FIRST must be
    # given the paths explicitly or it fails for the wrong reason (the first version of this test
    # did exactly that and reported spurious "DIFFERS" rows).
    paths = ";".join(f"sys.path.insert(0,r'{p}')" for p in
                     (os.path.join(repo, 'verification_tools'), os.path.join(repo, 'Phase 2'),
                      P3, here))
    prog = (
        "import sys,os;" + paths + ";"
        "{imports}"
        "import torch;torch.set_default_dtype(torch.float64);"
        "import b1_reproduce as B;"
        # WARM-UP first: run 1 differs from later runs in ANY order (BLAS/allocator warm-up), so
        # comparing first calls would measure warm-up, not import order. Report the SECOND call.
        "[B.stage_hashes(*c) for c in B.CASES];"
        "print('RESULT ' + '|'.join(B.stage_hashes(*c)['elastic_tensor'] for c in B.CASES))"
    )
    orders = {
        'canonical  (_common first)': "import _common;import physical_homog;import numpy;",
        'numpy first               ': "import numpy;import _common;import physical_homog;",
        'physical_homog first      ': "import physical_homog;import _common;import numpy;",
        'torch before everything   ': "import torch;import _common;import physical_homog;import numpy;",
    }
    ref = None
    for label, imp in orders.items():
        r = subprocess.run([sys.executable, '-c', prog.format(imports=imp)],
                           capture_output=True, text=True, cwd=P3)
        line = [l for l in r.stdout.splitlines() if l.startswith('RESULT ')]
        got = line[-1][7:] if line else f'ERR {r.stderr.strip()[-90:]}'
        same = '' if ref is None else ('  SAME' if got == ref else '  <-- DIFFERS')
        if ref is None:
            ref = got
        print(f'  {label}  {got}{same}')
    print('\n  identical hashes => import order is NOT the mechanism (warm-up controlled for).')


def mode_threads(n, arms=(1, 4)):
    """Does TORCH THREAD COUNT change the B-1 hit rate? Suite context once, then n probes per arm.

    Why this mode exists (2026-08-22): `mode_fingerprint` RECORDS `torch.get_num_threads()` but
    nothing ever VARIED it, so thread count was the one environmental knob never tested — while
    `imports` was tested and refuted. Measured directly: 40 identical `forward()` calls on the
    regular lattice give **2 distinct bit patterns at 4 threads and 1 (bit-exact) at 1 thread**, so
    threading demonstrably makes the forward path nondeterministic.

    That alone does NOT explain B-1: the variation is ~1e-33 against the 1.18e-02 excursion captured
    on 2026-08-22 (`b1_dumps/b1_anomaly_20260822T142234Z_eta0.0_s0.json`) — thirty orders apart, and
    consistent with the 1-2 ulp envelope already on record. The open question this mode answers is
    whether threading nevertheless changes the RATE of the large excursion, e.g. by seeding a
    divergence at an unstable branch. A rate difference implicates it; equal rates exonerate it.

    Cheap by construction: context is rebuilt ONCE (the 13 preceding tests, the expensive part) and
    then each probe is 3 small cases — seconds, against ~12 min for a full-suite run. That is the
    whole point: enough samples to MEASURE a rate rather than infer one from one or two events.

    NOTE `torch.set_num_threads` is process-global and applies to the probes only; the context is
    built once at whatever the process started with, so the arms share one context by design.
    """
    import test_inverse_design as T
    BASE = {0: 2.205e-13, 1: 1.604e-12, 2: 2.202e-13}          # isolated, measured
    torch.manual_seed(0)
    print(f'  establishing suite context once (13 preceding tests), then {n} probes x '
          f'{len(arms)} arms...', flush=True)
    for name in PRECEDING:
        try:
            getattr(T, name)()
        except Exception as e:
            print(f'   ({name} raised {type(e).__name__}: {e})', flush=True)
    print('  context established' + chr(10), flush=True)

    summary = {}
    for nt in arms:
        torch.set_num_threads(int(nt))
        hits, worst, bitsets = 0, 0.0, [set(), set(), set()]
        for r in range(n):
            for ci, case in enumerate(CASES):
                cs, cp = probe_case_full(*case)
                scale = np.abs(cp).max()
                vs = float(np.abs(cs - cp).max() / scale)
                bitsets[ci].add(cs.tobytes())
                worst = max(worst, vs / BASE[ci])
                if vs > 100 * BASE[ci]:
                    hits += 1
                    print(f'  threads={nt} rep {r+1:4d} case{ci+1} vs={vs:.4e}'
                          f'   *** {vs / BASE[ci]:.1e}x BASELINE ***', flush=True)
        summary[nt] = (hits, worst, [len(b) for b in bitsets])
        print(f'  threads={nt}: {n} reps x 3 cases -> {hits} hit(s) above 100x baseline; '
              f'worst {worst:.2e}x; distinct bit patterns per case {summary[nt][2]}', flush=True)

    print(chr(10) + '  RATE COMPARISON (hits per 3n probes):')
    for nt in arms:
        h, w, b = summary[nt]
        print(f'    threads={nt}: {h}/{3*n} hits, worst {w:.2e}x baseline, bit patterns {b}')
    if all(summary[nt][0] == 0 for nt in arms):
        print('  -> NO excursion in either arm. Threading not implicated at this sample size; the '
              'cheap harness may simply not reproduce it (then use an overnight full-suite loop).')
    return summary


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
    elif mode == 'suitectx':
        mode_suitectx(int(sys.argv[2]) if len(sys.argv) > 2 else 5)
    elif mode == 'threads':
        mode_threads(int(sys.argv[2]) if len(sys.argv) > 2 else 40)
    elif mode == 'stages':
        mode_stages(int(sys.argv[2]) if len(sys.argv) > 2 else 20,
                    ctx=(len(sys.argv) < 4 or sys.argv[3] != 'noctx'))
    elif mode == 'imports':
        mode_imports()
    elif mode == 'commits':
        mode_commits(sys.argv[2:])
    else:
        print(__doc__); sys.exit(2)


if __name__ == '__main__':
    main()
