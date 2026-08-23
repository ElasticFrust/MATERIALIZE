r"""B-1 overnight: repeat the FULL suite, paired 1-thread vs default, writing every verdict as it goes.

WHY THE FULL SUITE AND NOT THE CHEAP HARNESS
--------------------------------------------
`b1_reproduce.py threads 60` (2026-08-22) took **360 probes across 1- and 4-thread arms and saw ZERO
excursions** — every probe at exactly 1.00x baseline, where ~17 hits were expected at the nominal
1-in-21 rate. So `mode_suitectx`'s reconstructed context (replaying the 13 preceding tests once) is
**not sufficient** to produce the phenomenon. That fits B-1's documented signature — "clusters in
time with a persistent state transition" — and it leaves the expensive route as the only one: run the
real thing, many times.

The excursion this is hunting was captured 2026-08-22 during a routine gate run: `test_homogenization`
on the REGULAR lattice (phi=psi=1, eta=0, seed 0), **1.18e-02 against a 1.6e-12 baseline**, then 8/8
PASS on repeat in isolation. Dump: `b1_dumps/b1_anomaly_20260822T142234Z_eta0.0_s0.json`.

WHY PAIRED THREAD ARMS
----------------------
Threading demonstrably makes the forward path nondeterministic (40 `forward()` calls on the regular
lattice: 2 distinct bit patterns at 4 threads, 1 bit-exact at 1 thread) but only at **1.5e-33** —
thirty orders below the failure, consistent with the 1-2 ulp envelope on record. So threading is
**not the direct cause**; the open question is whether it changes the RATE, e.g. by seeding a
divergence at an unstable branch. Alternating arms answers that for free, and `set_num_threads(1)`
doubles as a control: if the 1-thread arm is bit-reproducible run to run, any excursion there is a
real state change rather than arithmetic noise.

INCREMENTAL BY DESIGN
---------------------
Every run appends one CSV row and flushes **before** the next starts, because a machine suspend must
not cost the night — one g1_2 design already recorded 18266 s of wall clock from exactly that. The
run log is written per iteration too, so a killed process leaves complete evidence for the runs it
finished. Re-running APPENDS: the CSV is the cumulative record across nights.

Determinism: each iteration is a fresh `python test_inverse_design.py` subprocess, so no state leaks
between runs; the thread arm is set by `OMP_NUM_THREADS`/`TORCH_NUM_THREADS` in the child env.

Run:  C:\\Users\\doron\\anaconda3\\python.exe "Phase 3/verifications/b1_overnight.py" [hours] [--arms 1,0]
        hours   wall-clock budget, default 10
        --arms  comma-separated thread counts to alternate; 0 = leave default. Default "1,0"
Out:  Phase 3/verifications/b1_dumps/overnight_<UTC>.csv   (+ .log)
"""
import csv
import datetime as dt
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
DUMPS = os.path.join(HERE, 'b1_dumps')
SUITE = os.path.join(REPO, 'Phase 3', 'test_inverse_design.py')
PY = sys.executable

# [15] prints this on success; anything else in that line is the signal we are hunting.
BASELINE_MARK = 'solver-vs-physical(tensor)=1.6e-12'


def one_run(threads, log):
    """One full-suite subprocess. Returns (verdict, seconds, worst_line)."""
    env = dict(os.environ, PYTHONIOENCODING='utf-8', KMP_DUPLICATE_LIB_OK='TRUE')
    if threads:                                     # 0 => leave the process default
        env['OMP_NUM_THREADS'] = str(threads)
        env['MKL_NUM_THREADS'] = str(threads)
        env['TORCH_NUM_THREADS'] = str(threads)
    t0 = time.time()
    p = subprocess.run([PY, SUITE], cwd=REPO, env=env, capture_output=True, text=True,
                       errors='replace')
    secs = time.time() - t0
    # stderr too: a native segfault or a traceback leaves NOTHING on stdout, and a run that dies
    # without evidence is a wasted hour of the budget (the sim is dense LAPACK -- documented as able
    # to hard-crash on near-singular geometry, which would otherwise land as a bare NO_VERDICT).
    err = (p.stderr or '').strip()
    out = ((p.stdout or '') + (f'\n[stderr]\n{err}' if err else '')
           + (p.returncode and f'\n[rc={p.returncode}]' or ''))
    log.write(out + '\n' + '=' * 78 + '\n')
    log.flush()
    fails = [l.strip() for l in out.splitlines() if 'FAIL' in l]
    homog = [l.strip() for l in out.splitlines() if 'homogenisation' in l or 'homogenization' in l]
    # [15]'s number, recorded on EVERY row. The first night returned 'FAIL' from an unrelated test
    # on 11 runs, which hid their [15] value in the CSV and forced a grep of the log to confirm the
    # 1-thread arm was clean. The verdict alone is not enough to classify a row.
    mark = next((l.split('solver-vs-physical(tensor)=')[1].split()[0]
                 for l in homog if 'solver-vs-physical(tensor)=' in l), '')
    if not mark and any('test_homogenization' in l for l in fails):
        mark = 'FAILED'
    if fails:
        return 'FAIL', secs, mark, fails[0][:200]
    if 'ALL PASSED' in out:
        # a pass that does NOT show the usual baseline is still worth flagging
        drift = homog and BASELINE_MARK not in homog[0]
        return ('PASS_DRIFT' if drift else 'PASS'), secs, mark, (homog[0][:200] if homog else '')
    return 'NO_VERDICT', secs, mark, (out.strip().splitlines() or [''])[-1][:200]


def main():
    hours = float(sys.argv[1]) if len(sys.argv) > 1 and not sys.argv[1].startswith('-') else 10.0
    arms = [1, 0]
    if '--arms' in sys.argv:
        arms = [int(x) for x in sys.argv[sys.argv.index('--arms') + 1].split(',')]

    os.makedirs(DUMPS, exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    csv_path = os.path.join(DUMPS, f'overnight_{stamp}.csv')
    log_path = os.path.join(DUMPS, f'overnight_{stamp}.log')
    deadline = time.time() + hours * 3600

    print(f'B-1 overnight: budget {hours:g} h, arms {arms} (0 = process default)', flush=True)
    print(f'  suite : {SUITE}', flush=True)
    print(f'  csv   : {csv_path}   (appended and flushed after EVERY run)', flush=True)

    with open(csv_path, 'w', newline='') as fh, open(log_path, 'w', encoding='utf-8') as log:
        w = csv.writer(fh)
        w.writerow(['i', 'utc', 'threads', 'verdict', 'secs', 'homog15', 'detail'])
        fh.flush()
        i = 0
        counts = {}
        while time.time() < deadline:
            threads = arms[i % len(arms)]
            i += 1
            verdict, secs, mark, detail = one_run(threads, log)
            counts[(threads, verdict)] = counts.get((threads, verdict), 0) + 1
            w.writerow([i, dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'),
                        threads, verdict, f'{secs:.1f}', mark, detail])
            fh.flush()                              # a suspend must not cost the night
            os.fsync(fh.fileno())
            flag = '   <<< HIT' if verdict in ('FAIL', 'PASS_DRIFT') else ''
            print(f'  [{i:3d}] threads={threads or "default"} {verdict:10} {secs:6.1f}s '
                  f'[15]={mark or "?":9}{flag}', flush=True)
            if flag:
                print(f'        {detail}', flush=True)

        print('\nSUMMARY (see the CSV for the full record):', flush=True)
        for (t, v), n in sorted(counts.items()):
            print(f'  threads={t or "default":>7}  {v:10} x{n}', flush=True)
        hits = sum(n for (t, v), n in counts.items() if v in ('FAIL', 'PASS_DRIFT'))
        print(f'  -> {hits} hit(s) in {i} runs', flush=True)
        if hits == 0:
            print('  No excursion. At ~1 in 21 that is unsurprising below ~60 runs; the CSV is '
                  'cumulative, so keep appending nights before concluding anything.', flush=True)


if __name__ == '__main__':
    main()
