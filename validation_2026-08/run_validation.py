r"""Clean re-run of the Phase 2/3/5 validations under the post-audit code, into ONE tree.

Why this exists. The August audit changed the forward map (**A-0** shear contraction, **A-10** ν
convention) and fixed two rendering defects (**B-4b** colour norm, **B-4c** polar zero crossings).
Every artifact produced before that is therefore suspect — that is audit **A-19**, and it was found
the hard way: `fig1_recreate_pointy.png` "regenerated" in 41 s and looked like a completed re-run,
when in fact it had hit a cached design frozen 2026-07-12 and re-rendered July's answer.

**So the single most important thing this runner does is move the design CACHES aside** (see
`CACHES`). Three scripts short-circuit on a saved `.npz`; without that step the sweep would look like
it re-ran everything while quietly reproducing stale results. Caches are MOVED to `attic/`, never
deleted — the July artifacts are evidence for A-19.

Layout produced:

    validation_2026-08/
        MANIFEST.md, manifest.csv     provenance + one row per script (exit, seconds, outputs)
        attic/                        design caches moved aside so scripts actually redesign
        phase2/<script>/  phase3/<script>/  phase5/<script>/
                                      stdout+stderr log, and every file the script wrote

Scope = gates + the headline validations `MATERIALIZE.md` §9 names (agreed with the user
2026-08-17). The one-off audit diagnostics and superseded probes in `Phase 3/verifications` are
deliberately NOT run: they would take days and would fill a "clean" tree with output from scripts we
have already superseded. They are inventoried for keep-or-retire triage instead.

Run:  python validation_2026-08/run_validation.py [--dry-run] [--only SUBSTR]
"""
import argparse
import csv
import datetime as _dt
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..'))
PY = r'C:\Users\doron\anaconda3\python.exe'          # the verified stack; never `python` (see §Env)

# Design caches that make a script SKIP its optimisation. Moved to attic/ before the run.
CACHES = [
    'Phase 3/verifications/showcase/fig1_pointy_network.npz',
    'Phase 3/verifications/showcase/fig1c_rot90_regular_network.npz',
    'Phase 3/verifications/showcase/fig1d_rot90_disorder_network.npz',
]

# (phase, path relative to REPO). Order: gates first — if a gate fails the rest is not worth running.
SCRIPTS = [
    ('phase2', 'Phase 2/test_forward_solver.py'),
    ('phase3', 'Phase 3/test_inverse_design.py'),
    ('phase5', 'Phase 5/verifications/test_designer_surface.py'),
    ('phase5', 'Phase 5/verifications/test_hex_closed_form.py'),
    ('phase5', 'Phase 5/verifications/sanity.py'),

    ('phase3', 'Phase 3/verifications/verify_lattice.py'),
    ('phase3', 'Phase 3/verifications/auxetic_sweep/design_and_verify.py'),
    ('phase3', 'Phase 3/verifications/auxetic_patch/design_and_verify.py'),
    ('phase3', 'Phase 3/verifications/auxetic_patch/make_maps.py'),
    ('phase3', 'Phase 3/verifications/anisotropy/design_and_verify.py'),
    ('phase3', 'Phase 3/verifications/anisotropy/make_maps.py'),
    ('phase3', 'Phase 3/verifications/two_region/demo.py'),
    ('phase3', 'Phase 3/verifications/strain_stress/design_and_verify.py'),

    ('phase3', 'Phase 3/verifications/showcase/fig1_recreate_pointy.py'),
    ('phase3', 'Phase 3/verifications/showcase/fig1b_recreate_pointy_regular.py'),
    ('phase3', 'Phase 3/verifications/showcase/fig1cd_rotated_substrate.py'),
    ('phase3', 'Phase 3/verifications/showcase/fig2_triangular_nu.py'),
    ('phase3', 'Phase 3/verifications/showcase/fig3_isotropize.py'),
    ('phase3', 'Phase 3/verifications/showcase/fig4_uniform_nu.py'),
    ('phase3', 'Phase 3/verifications/showcase/fig5_bullseye_strain.py'),
    ('phase3', 'Phase 3/verifications/showcase/fig6_different_nu.py'),

    ('phase5', 'Phase 5/verifications/run_goal1.py'),
    ('phase5', 'Phase 5/verifications/run_g1_2.py'),
    ('phase5', 'Phase 5/verifications/run_goal2.py'),
    ('phase5', 'Phase 5/verifications/run_goal2_attempts.py'),
    ('phase5', 'Phase 5/verifications/verify_positions.py'),
]

# Where produced files can appear. Walked before/after each script to attribute outputs.
WATCH = ['Phase 2', 'Phase 3', 'Phase 5']
SKIP_DIRS = {'__pycache__', '.git', 'validation_2026-08', 'networks_old'}


def slug(rel):
    """Destination folder name for a script — from its FULL relative path, not its basename.

    Four scripts are called `design_and_verify.py` and two `make_maps.py`; keying on the basename
    made them share one folder, and the second run's log OVERWROTE the first's (cost: the
    auxetic_sweep log — its numbers survived only because the script also writes a CSV)."""
    rel = os.path.splitext(rel)[0]
    for pre in ('Phase 3/verifications/', 'Phase 5/verifications/', 'Phase 2/', 'Phase 3/', 'Phase 5/'):
        if rel.startswith(pre):
            rel = rel[len(pre):]
            break
    return rel.replace('/', '__').replace('\\', '__')


def already_done(csv_path):
    """Scripts with a manifest row — so an interrupted sweep can resume instead of restarting."""
    if not os.path.exists(csv_path):
        return set()
    with open(csv_path, newline='', encoding='utf-8') as fh:
        return {r['script'] for r in csv.DictReader(fh) if r.get('script')}


def provenance():
    """(commit, dirty) — `dirty` means the tree had uncommitted changes, so commit alone does NOT
    reproduce the run. That distinction is exactly what made the B-1 investigation expensive."""
    def q(*a):
        return subprocess.run(['git', *a], cwd=REPO, capture_output=True, text=True).stdout.strip()
    return q('rev-parse', '--short', 'HEAD'), bool(q('status', '--porcelain'))


def snapshot():
    """{path: mtime} over the watch roots. Used to attribute outputs to the script that wrote them."""
    out = {}
    for root in WATCH:
        for dirpath, dirnames, filenames in os.walk(os.path.join(REPO, root)):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for fn in filenames:
                p = os.path.join(dirpath, fn)
                try:
                    out[p] = os.path.getmtime(p)
                except OSError:
                    pass
    return out


def stow_caches(dry):
    """MOVE (never delete) the design caches, so cached scripts actually redesign — audit A-19."""
    attic = os.path.join(HERE, 'attic')
    moved = []
    os.makedirs(attic, exist_ok=True)
    for rel in CACHES:
        src = os.path.join(REPO, rel)
        if os.path.exists(src):
            dst = os.path.join(attic, os.path.basename(rel))
            print(f'  stow {rel} -> attic/', flush=True)
            if not dry:
                shutil.move(src, dst)
            moved.append(rel)
    return moved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--only', default=None, help='run only scripts whose path contains this')
    args = ap.parse_args()

    todo = [s for s in SCRIPTS if not args.only or args.only in s[1]]
    commit, dirty = provenance()
    started = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec='seconds')
    print(f'commit {commit}{" (DIRTY)" if dirty else ""}   {len(todo)} scripts   {started}',
          flush=True)

    print('stowing design caches so nothing re-renders a stale design (A-19):', flush=True)
    stow_caches(args.dry_run)
    if args.dry_run:
        for ph, rel in todo:
            print(f'  would run [{ph}] {rel}', flush=True)
        return

    csv_path = os.path.join(HERE, 'manifest.csv')
    done_already = already_done(csv_path)
    if done_already:
        print(f'resuming: {len(done_already)} script(s) already recorded', flush=True)
    new = not os.path.exists(csv_path)
    with open(csv_path, 'a', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(['phase', 'script', 'exit', 'seconds', 'n_outputs', 'commit', 'dirty',
                        'finished_utc'])
        for i, (phase, rel) in enumerate(todo, 1):
            if rel in done_already:
                print(f'[{i}/{len(todo)}] {rel}\n   already in manifest — skipped (resume)', flush=True)
                continue
            stem = slug(rel)
            dest = os.path.join(HERE, phase, stem)
            os.makedirs(dest, exist_ok=True)
            src = os.path.join(REPO, rel)
            print(f'[{i}/{len(todo)}] {rel}', flush=True)
            if not os.path.exists(src):
                print('   MISSING — skipped', flush=True)
                w.writerow([phase, rel, 'MISSING', 0, 0, commit, dirty, ''])
                fh.flush()
                continue

            before = snapshot()
            t0 = time.perf_counter()
            env = dict(os.environ, PYTHONIOENCODING='utf-8', MPLBACKEND='Agg')
            proc = subprocess.run([PY, '-u', src], cwd=os.path.dirname(src), env=env,
                                  capture_output=True, text=True, errors='replace')
            dt = time.perf_counter() - t0
            after = snapshot()

            log = os.path.join(dest, f'{stem}.log')
            with open(log, 'w', encoding='utf-8') as lf:
                lf.write(f'# {rel}\n# commit {commit} dirty={dirty}\n'
                         f'# exit {proc.returncode}  {dt:.1f} s\n\n=== stdout ===\n{proc.stdout}\n'
                         f'=== stderr ===\n{proc.stderr}\n')

            produced = [p for p, m in after.items() if before.get(p, -1) < m]
            for p in produced:
                try:
                    rp = os.path.relpath(p, REPO).replace('\\', '/').replace('/', '__')
                    shutil.copy2(p, os.path.join(dest, rp))
                except OSError as e:
                    print(f'   copy failed {p}: {e}', flush=True)
            tail = (proc.stdout.strip().splitlines() or ['(no stdout)'])[-1][:110]
            print(f'   exit {proc.returncode}  {dt:7.1f} s  {len(produced)} outputs   {tail}',
                  flush=True)
            w.writerow([phase, rel, proc.returncode, f'{dt:.1f}', len(produced), commit, dirty,
                        _dt.datetime.now(_dt.timezone.utc).isoformat(timespec='seconds')])
            fh.flush()
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
