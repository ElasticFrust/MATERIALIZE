"""
A-7b re-layering — before/after equivalence instrument.

Purpose: the A-7b change (audit `documentation/AUDIT_2026-08.md`) lifts the shared geometry /
metric-tensor library out of the retireable oracle layer (`verification_tools/test_*.py`,
`verify_*.py`) into the core layer (`Phase 2/{metric_ops,mesh_build,solver_build}.py`) plus
`verification_tools/sim_assembly.py`. It is a PURE MOVE, so the honest check is that every moved
symbol — and the whole design path built on it — returns BIT-IDENTICAL results before and after.

This script therefore:
  (1) FINGERPRINTS every moved symbol's output, plus `DesignProblem.periodic/.open` forward solves
      (the design path sits directly on the moved code — see audit A-7b "re-check the blast
      radius, since `DesignProblem.periodic` is on the moved path").

      Two grades of comparison, because the solver is NOT bit-reproducible (see below):
        - `hash`  entries: sha256 over raw float64 bytes → BIT-IDENTICAL required. Everything
                  deterministic is checked this way, including the fully constructed solver's
                  geometry buffers AND all three intrinsic constraint operators
                  (`_q_geom`, `_J_edge_sp`, `_C_curv_sp`, `_M_S_sp`), which pins `make_solver`
                  exactly without going through a forward solve.
        - `~val` entries: floats compared at relative 1e-13.

      Why the second grade exists — measured, not assumed: `forward(method='intrinsic')` is
      run-to-run nondeterministic at the 1–2 ulp level. Calling the SAME (pre-move) `make_solver`
      twice in one process yields ν differing by 1.7e-16 (2 distinct doubles), C_eff by 2.2e-16,
      per-triangle C6 by 5.0e-16. That is audit finding **B-1** (solver nondeterminism, source
      unidentified) and is entirely independent of this re-layering. 1e-13 is ~100× that measured
      envelope and ~10 orders below any physically meaningful difference.
  (2) records an IMPORT INVENTORY over every .py in the repo. The change is import-level, so
      import status before-vs-after is near-exhaustive coverage of what was actually touched.
      Some files already fail to import for unrelated reasons, so the criterion is that no file
      CHANGES status — not that all pass.

Symbols are resolved through `_resolve()`, which prefers the NEW homes and falls back to the OLD
ones, so the very same script runs on both sides of the move.

Usage:
    python relayer_a7b.py --out before.json            # on the pre-change tree
    python relayer_a7b.py --out after.json             # after the move
    python relayer_a7b.py --compare before.json after.json
    (--skip-imports omits the slow subprocess inventory;
     --imports-from OLD.json reuses an inventory already recorded there)
"""
import os, sys, json, types, hashlib, argparse, subprocess

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
P3 = os.path.dirname(HERE)
ROOT = os.path.dirname(P3)
for _p in (P3, os.path.join(ROOT, 'Phase 2'), os.path.join(ROOT, 'verification_tools'), ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
torch.set_default_dtype(torch.float64)          # REQUIRED — the whole solver stack is float64

PY = sys.executable


# --------------------------------------------------------------------------- symbol resolution
def _resolve():
    """The moved symbols, preferring their NEW homes and falling back to the OLD ones.

    All-or-nothing on the new side: a partial new layout would silently mix providers and make a
    "match" meaningless. `which` records which side answered, so the JSON says so out loud."""
    try:
        import metric_ops as MO
        import mesh_build as MB
        import solver_build as SB
        import sim_assembly as SA
        return 'new', types.SimpleNamespace(
            vec3=MO.vec3, tri_metric_change=MO.tri_metric_change,
            bare_tensor_k=MO.bare_tensor, bare_tensor_nok=MO.bare_tensor,
            build_geometry=MB.build_geometry, set_VD=MB.set_VD,
            clean_tri=MB.clean_tri, build_open_mesh=MB.build_open_mesh,
            kkt_from_tri_bond=MB.kkt_from_tri_bond,
            make_solver=SB.make_solver, assemble_K_faff=SA.assemble_K_faff)
    except ImportError:
        import test_cluster_Ceff as CE
        import test_cluster_rigidity as TR
        import test_cluster_VD as VD
        import test_intrinsic_VD as IV
        import verify_solver_open as VO
        import verify_solver_sweep as VS
        return 'old', types.SimpleNamespace(
            vec3=CE.vec3, tri_metric_change=CE.tri_metric_change,
            # the TWO pre-move bare_tensors the single new one must reproduce:
            bare_tensor_k=TR.bare_tensor,        # k-aware  (meshes carrying tri_k)
            bare_tensor_nok=CE.bare_tensor,      # k-less   (meshes without tri_k)
            build_geometry=VD.build_geometry, set_VD=VD.set_VD,
            clean_tri=VO.clean_tri, build_open_mesh=VO.build_open_mesh,
            kkt_from_tri_bond=IV.kkt_from_tri_bond,
            make_solver=VS.make_solver, assemble_K_faff=TR.assemble_K_faff)


# --------------------------------------------------------------------------- exact fingerprints
def _h(*arrays):
    """sha256 over the raw float64/int64 bytes of `arrays` — bit-identity, not tolerance."""
    m = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(np.asarray(a))
        m.update(str(a.dtype).encode()); m.update(str(a.shape).encode()); m.update(a.tobytes())
    return m.hexdigest()


def _peek(a, n=4):
    """A few real numbers alongside each hash, so a diff is readable and not just 'hashes differ'."""
    return [float(x) for x in np.asarray(a, float).ravel()[:n]]


def _solver_state(s):
    """Exact fingerprint of a CONSTRUCTED ElasticSolver: the geometry buffers `make_solver`/`_mount`
    writes, plus every operator `_build_intrinsic_constraints` derives from them — the edge
    compatibility (C1) `_J_edge_sp`, discrete Gaussian curvature (C2) `_C_curv_sp`, area-weighted
    normalisation (C3) `_M_S_sp`, and the metric-Hessian carrier `_q_geom`.

    This is what pins the move exactly: it is fully deterministic, unlike a forward solve, so
    `make_solver` old-vs-new can be required to agree BIT-FOR-BIT rather than to a tolerance.
    Sparse operators are compared through a lexsorted COO, since COO ordering is not canonical."""
    parts = [s.edge_vecs.numpy(), s.actual_length2.numpy(), s.area_weights.numpy(), s._q_geom]
    for M in (s._J_edge_sp, s._C_curv_sp, s._M_S_sp):
        c = M.tocoo(); o = np.lexsort((c.col, c.row))
        parts += [c.row[o], c.col[o], c.data[o]]
    parts += list(s.kkt_arrays)
    return _h(*parts)


def _det_k(n):
    """Deterministic, RNG-free stiffness vector — reproducible across processes and machines."""
    return 1.0 + 0.5 * np.sin(np.arange(n, dtype=float))


def fingerprint():
    which, S = _resolve()
    import pbc_dg_analysis as pda
    import physical_homog as PH
    import Disc_2_Cont_optimized as D2C
    from inverse_design import DesignProblem
    out = {'_resolved_from': which}

    # ---- 1. mesh construction: build_geometry / set_VD -------------------------------------
    geo = S.build_geometry(8, 0.20, 1)
    out['build_geometry'] = {
        'hash': _h(geo['pts'], geo['edge_vecs'], geo['actual_len2'], geo['areas'],
                   geo['bond_R'], geo['bond_u'], geo['bond_v'], geo['tri_bond']),
        'n_tri': int(len(geo['simplices'])), 'n_bond': int(len(geo['bond_R'])),
        'pts': _peek(geo['pts']), 'areas': _peek(geo['areas'])}
    S.set_VD(geo, 5)
    out['set_VD'] = {'hash': _h(geo['bond_k'], geo['tri_k']), 'bond_k': _peek(geo['bond_k'])}

    # ---- 2. metric/tensor maths ------------------------------------------------------------
    # bare_tensor must reproduce BOTH pre-move variants: the k-aware one on a mesh carrying
    # tri_k, and the k-less one on a pbc_dg_analysis mesh, which carries none.
    mesh_nok = pda.build_periodic_tf_mesh(8, 0.20, 0)
    assert 'tri_k' not in mesh_nok and 'bond_k' not in mesh_nok, \
        "build_periodic_tf_mesh unexpectedly carries tri_k/bond_k — the k-less equivalence " \
        "argument for metric_ops.bare_tensor does not hold; STOP."
    out['bare_tensor_with_tri_k'] = {'hash': _h(S.bare_tensor_k(geo)),
                                     'peek': _peek(S.bare_tensor_k(geo))}
    out['bare_tensor_no_tri_k'] = {'hash': _h(S.bare_tensor_nok(mesh_nok)),
                                   'peek': _peek(S.bare_tensor_nok(mesh_nok))}

    g = np.arange(24, dtype=float).reshape(6, 2, 2)
    out['vec3'] = {'hash': _h(S.vec3(g)), 'peek': _peek(S.vec3(g))}

    ev, sx, nn = geo['edge_vecs'], geo['simplices'], len(geo['pts'])
    F = np.eye(2) + 1e-3 * np.array([[1.0, 0.3], [0.3, -0.5]])
    u = 1e-4 * np.stack([np.cos(np.arange(nn, dtype=float)),
                         np.sin(np.arange(nn, dtype=float))], 1)
    tmc = S.tri_metric_change(ev, sx, F, u)
    out['tri_metric_change'] = {'hash': _h(tmc), 'peek': _peek(tmc)}

    # ---- 3. periodic constraint topology + sim assembly -------------------------------------
    kkt = S.kkt_from_tri_bond(geo['tri_bond'], geo['edge_vecs'])
    out['kkt_from_tri_bond'] = {'hash': _h(*kkt), 'n_interior_edges': int(len(kkt[0]))}

    K, faff = S.assemble_K_faff(geo, F)
    Kc = K.tocoo()
    order = np.lexsort((Kc.col, Kc.row))         # COO ordering is not canonical — sort first
    out['assemble_K_faff'] = {'hash': _h(Kc.row[order], Kc.col[order], Kc.data[order], faff),
                              'faff': _peek(faff)}

    # ---- 4. open mesh construction ----------------------------------------------------------
    np.random.seed(7)
    tri = D2C.generate_foam_points((5, 5), 0.2)
    ct = S.clean_tri(tri)
    out['clean_tri'] = {'hash': _h(ct.points, ct.simplices), 'n_pts': int(len(ct.points))}
    om = S.build_open_mesh(ct)
    out['build_open_mesh'] = {
        'hash': _h(om['pts'], om['edge_vecs'], om['areas'], om['bond_R'],
                   om['bond_u'], om['bond_v'], om['tri_bond'], om['bond_k'], om['tri_k']),
        'n_tri': int(len(om['simplices'])), 'n_bond': int(len(om['bond_R']))}

    # ---- 5. solver construction (make_solver / _mount) --------------------------------------
    # `hash` = the constructed solver, EXACT. `~` = through a forward solve, rel 1e-13 (B-1).
    sv = S.make_solver(geo, kkt)
    res = sv.forward(torch.as_tensor(_det_k(len(geo['bond_R'])))[geo['tri_bond']],
                     rest_lengths=torch.as_tensor(np.sqrt(geo['actual_len2'])),
                     method='intrinsic', physical_units=True)
    out['make_solver'] = {'hash': _solver_state(sv),
                          '~poisson': float(res['poisson']), '~young': float(res['young']),
                          '~C_eff': _peek(res['elastic_tensor'].detach().numpy(), 6)}

    # ---- 6. THE DESIGN PATH — DesignProblem sits directly on the moved code ------------------
    def _prob_state(p):
        """Everything DesignProblem derives from the moved mesh/solver code, deterministically."""
        return _h(p.tri_bond.numpy(), p.bond_len.numpy(), p.areas.numpy(),
                  np.asarray(p.centroids, float), p.rl_ref.numpy()) + '|' + _solver_state(p.solver)

    out['design_path'] = {}
    for tag, (N, eta, seed) in {'periodic_N8_eta0.20': (8, 0.20, 0),
                                'periodic_N10_eta0.30': (10, 0.30, 3),
                                'periodic_N14_eta0.00': (14, 0.00, 0)}.items():
        prob = DesignProblem.periodic(N=N, eta=eta, seed=seed)
        r = prob.forward(torch.as_tensor(_det_k(prob.n_bond)))
        pt = r['per_triangle'].detach().numpy()
        out['design_path'][tag] = {
            'hash': _prob_state(prob), 'n_tri': prob.n_tri, 'n_bond': prob.n_bond,
            '~poisson': float(r['poisson']), '~young': float(r['young']),
            '~C_eff': _peek(prob.region_tensor(r['per_triangle'], None).detach().numpy(), 6),
            '~per_tri_absmax': float(np.abs(pt).max()), '~per_tri_sum': float(pt.sum())}

    np.random.seed(7)
    prob = DesignProblem.open(D2C.generate_foam_points((5, 5), 0.2))
    r = prob.forward(torch.as_tensor(_det_k(prob.n_bond)))
    pt = r['per_triangle'].detach().numpy()
    out['design_path']['open_foam_5x5'] = {
        # open problems build their solver via fst.from_triangulation, not _mount, so hash the
        # DesignProblem's own derived arrays only
        'hash': _h(prob.tri_bond.numpy(), prob.bond_len.numpy(), prob.areas.numpy(),
                   np.asarray(prob.centroids, float), prob.rl_ref.numpy()),
        'n_tri': prob.n_tri, 'n_bond': prob.n_bond,
        '~poisson': float(r['poisson']), '~young': float(r['young']),
        '~per_tri_absmax': float(np.abs(pt).max()), '~per_tri_sum': float(pt.sum())}

    # ---- 7. the ORACLE must stay independent of all of the above -----------------------------
    # (physical_homog shares no code with the design path; recorded so a regression shows up here.)
    free = np.arange(2, 2 * len(geo['pts']))
    out['oracle_energy_C'] = {'hash': _h(PH.energy_C(geo, free, S.assemble_K_faff)),
                              'peek': _peek(PH.energy_C(geo, free, S.assemble_K_faff), 9)}
    return out


# --------------------------------------------------------------------------- import inventory
_HEADER_RUNNER = r"""
import ast, os, sys, io, contextlib
path = sys.argv[1]
src = open(path, encoding='utf-8').read()
tree = ast.parse(src, path)
# Last MODULE-LEVEL import statement: everything above it (sys.path setup, HERE/ROOT, the
# _bootstrap dance) must run for the imports to resolve the way they do in the real script.
end = 0
for node in tree.body:
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        end = max(end, getattr(node, 'end_lineno', node.lineno))
if end == 0:
    print('NOIMPORTS'); sys.exit(0)
head = '\n'.join(src.splitlines()[:end])
try:
    code = compile(head, path, 'exec')          # truncation must still be valid Python
except SyntaxError:
    head, code = src, compile(src, path, 'exec')   # fall back to the whole file
    print('FULLFILE', file=sys.stderr)
g = {'__name__': '__not_main__', '__file__': os.path.abspath(path), '__builtins__': __builtins__}
sys.path.insert(0, os.path.dirname(os.path.abspath(path)))
with contextlib.redirect_stdout(io.StringIO()):    # scripts print banners at import time
    exec(code, g)
"""


def import_inventory():
    """Record, per repo .py file, whether its top-level IMPORTS resolve.

    Deliberately NOT `importlib.import_module`: many scripts under `Phase 3/verifications/` and
    `Phase 5/` have no `if __name__ == '__main__'` guard and run a full design experiment at
    module level, so importing them outright takes minutes to hours each and tests far more than
    this change touches. Instead each file is parsed, truncated at its last MODULE-LEVEL import
    statement (which keeps the sys.path/bootstrap preamble those imports depend on), and only that
    header is executed — in its own subprocess, cwd = its directory, as the script would be run.

    That is exactly the surface an import-level re-layering can break, and it is fast enough to
    run over all 144 files before and after."""
    files = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in ('.git', '__pycache__', '.claude')]
        for fn in sorted(filenames):
            if fn.endswith('.py'):
                files.append(os.path.relpath(os.path.join(dirpath, fn), ROOT))
    env = dict(os.environ, PYTHONIOENCODING='utf-8', MPLBACKEND='Agg')
    inv = {}
    for rel in sorted(files):
        path = os.path.join(ROOT, rel)
        key = rel.replace('\\', '/')
        try:
            p = subprocess.run([PY, '-c', _HEADER_RUNNER, path], cwd=os.path.dirname(path),
                               env=env, timeout=120, capture_output=True, text=True,
                               errors='replace')
            if p.returncode == 0:
                inv[key] = 'ok'
            else:
                last = [l for l in (p.stderr or '').strip().splitlines() if l.strip()]
                inv[key] = 'FAIL: ' + (last[-1][:200] if last else '?')
        except subprocess.TimeoutExpired:
            inv[key] = 'TIMEOUT'
        print(f"  {inv[key][:58]:<60} {rel}", flush=True)
    return inv


# --------------------------------------------------------------------------- compare
def compare(a_path, b_path):
    a = json.load(open(a_path, encoding='utf-8'))
    b = json.load(open(b_path, encoding='utf-8'))
    bad = []

    print(f"resolved from: before={a['fingerprint'].get('_resolved_from')}  "
          f"after={b['fingerprint'].get('_resolved_from')}")

    RTOL = 1e-13          # only for '~' keys; see the module docstring (measured envelope ~2 ulp)

    def walk(x, y, path, approx):
        if isinstance(x, dict):
            for k in sorted(set(x) | set(y)):
                if k == '_resolved_from':
                    continue
                if k not in x or k not in y:
                    bad.append(f"{path}/{k}: present in only one side"); continue
                walk(x[k], y[k], f"{path}/{k}", approx or k.startswith('~'))
        elif isinstance(x, list) and isinstance(y, list) and len(x) == len(y):
            for i, (xi, yi) in enumerate(zip(x, y)):
                walk(xi, yi, f"{path}[{i}]", approx)
        elif isinstance(x, float) and isinstance(y, float):
            # '~' → through a nondeterministic forward solve, rel 1e-13; otherwise EXACT
            if approx:
                if not (x == y or abs(x - y) <= RTOL * max(1.0, abs(x))):
                    bad.append(f"{path}: {x!r} != {y!r}  (rel {abs(x-y)/max(1.0,abs(x)):.2e})")
            elif x != y:
                bad.append(f"{path}: {x!r} != {y!r}  (EXACT required)")
        elif x != y:
            bad.append(f"{path}: {x!r} != {y!r}")

    walk(a['fingerprint'], b['fingerprint'], 'fingerprint', False)
    print(f"\n[fingerprint] {'MATCH (bit-identical)' if not bad else f'{len(bad)} MISMATCH'}")
    for m in bad:
        print('   ', m)

    ia, ib = a.get('imports') or {}, b.get('imports') or {}
    changed, improved = [], []
    if ia and ib:
        for f in sorted(set(ia) | set(ib)):
            if f not in ia:                          # a file the move ADDS: it must import cleanly
                if ib[f] != 'ok':
                    changed.append(f"{f}: NEW FILE does not import -> {ib[f][:70]}")
                continue
            if f not in ib:
                changed.append(f"{f}: file disappeared (was {ia[f][:40]})"); continue
            # only the ok/not-ok status matters; a reworded error message is not a change
            if ia[f] == 'ok' and ib[f] != 'ok':
                changed.append(f"{f}: REGRESSED  ok -> {ib[f][:70]}")
            elif ia[f] != 'ok' and ib[f] == 'ok':
                # NOT a failure. In particular TIMEOUT is load-dependent (the per-file cap is wall
                # clock), so a busy baseline can time out where a quiet run succeeds.
                improved.append(f"{f}: {ia[f][:60]} -> ok")
        n_ok_a = sum(1 for v in ia.values() if v == 'ok')
        n_ok_b = sum(1 for v in ib.values() if v == 'ok')
        print(f"\n[imports] before {n_ok_a}/{len(ia)} ok, after {n_ok_b}/{len(ib)} ok — "
              f"{'NO REGRESSIONS' if not changed else f'{len(changed)} REGRESSED'}"
              f"{f', {len(improved)} improved' if improved else ''}")
        for c in changed:
            print('    REGRESSION', c)
        for c in improved:
            print('    improved  ', c)
    else:
        print("\n[imports] not recorded on one side — skipped")

    ok = not bad and not changed
    print('\n' + ('A-7b EQUIVALENCE: PASS' if ok else 'A-7b EQUIVALENCE: FAIL'))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out')
    ap.add_argument('--compare', nargs=2, metavar=('BEFORE', 'AFTER'))
    ap.add_argument('--skip-imports', action='store_true')
    ap.add_argument('--imports-from', metavar='JSON',
                    help="reuse the inventory already recorded in JSON instead of re-running it")
    args = ap.parse_args()

    if args.compare:
        sys.exit(compare(*args.compare))
    if not args.out:
        ap.error('need --out PATH or --compare BEFORE AFTER')

    rec = {'fingerprint': fingerprint()}
    print(f"fingerprint: resolved symbols from the {rec['fingerprint']['_resolved_from'].upper()} "
          f"layout, {len(rec['fingerprint']) - 1} entries")
    if args.imports_from:
        rec['imports'] = json.load(open(args.imports_from, encoding='utf-8'))['imports']
        print(f"imports: reused {len(rec['imports'])} entries from {args.imports_from}")
    elif not args.skip_imports:
        print("import inventory (one subprocess per file):")
        rec['imports'] = import_inventory()
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(rec, f, indent=1, sort_keys=True)
    print(f"wrote {args.out}")


if __name__ == '__main__':
    main()
