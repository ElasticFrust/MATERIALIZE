r"""Export the PUBLIC DEMO BUNDLE -- the only code and weights that may leave this machine.

WHY THIS IS A SCRIPT AND NOT A COPY-PASTE. `documentation/PLAN_B.md` B.6 lists four things the
deployed clone must not carry: unpublished research code, any link back to this repository, an
unsanitised checkpoint, and a preset library still stamped with dataset ids. Hand-copying satisfies
none of those repeatably -- it satisfies them once, and then drifts the first time a model file is
edited. This script makes the bundle a FUNCTION of the repo, so it can be regenerated and, more
importantly, RE-CHECKED.

It exports, and nothing else:
    tensor_ops.py              the three tensor helpers the app calls, extracted
    model.py                   the network itself                (numpy + torch only)
    inference.py               bond_vectors / prepare / predict, EXTRACTED from train_v3.py so the
                               deployed pipeline is literally the trained one, not a retyping
    meshing.py                 mesh_build.py minus the solver-facing leaves
    checkpoint.pt              SANITISED -- architecture keys and weights, no training protocol

Three gates run on every export and the export FAILS on any of them:
    [1] scrub     no forbidden token survives in any emitted file (repo name, phase paths, solver
                  and oracle module names, test names, local paths, the author's name)
    [2] imports   the bundle imports nothing outside {numpy, scipy.spatial, torch, stdlib}
    [3] numeric   bundle predict == research predict to <= 1e-12 on real meshes, through the
                  sanitised checkpoint. This is the gate that actually protects the science: the
                  other two protect the secrets, this one protects the answer.

Run:
    python "Phase 5/m2/export_demo_bundle.py" --out C:/Users/doron/Documents/elastic-demo
"""
import argparse
import ast
import os
import re
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))

torch.set_default_dtype(torch.float64)

# --- what the bundle may import, checked by gate [2] -------------------------------------------
ALLOWED_TOP = {'numpy', 'scipy', 'torch', 'math', 'json', 'os', 'sys', 'types', 'typing',
               'dataclasses', 'functools', 'itertools', 'collections', 'warnings', 'time',
               'hashlib', 'random', 're', 'io', 'base64', 'argparse'}

# --- tokens that must NOT survive into the bundle, checked by gate [1] --------------------------
# The author's name and local paths are on this list for the same reason the repo name is: a
# docstring that says where a file came from is an attribution trail (PLAN_B.md B.6/2).
FORBIDDEN = [
    r'MATERIALIZE', r'Phase\s*[0-9]', r'forward_solver', r'inverse_design', r'_common',
    r'verification_tools', r'physical_homog', r'sim_assembly', r'metric_ops', r'solver_build',
    r'designer\.py', r'positions\.py', r'seeds\.py', r'fields\.py', r'build_dataset',
    r'train_v[23]', r'evaluate_v[12]', r'test_[a-z0-9_]+\.py', r'M2_[A-Z_]+\.md',
    r'[A-Z_]+\.md\b', r'dataset_v[0-9]', r'doron', r'[Cc]:\\', r'/Users/', r'github',
    # internal work-item ids. Narrow patterns on purpose: a bare `A-1` or `D6` could be real maths,
    # and a gate that cries wolf gets switched off. These five forms are the ones actually used.
    r'\bA0\.[0-9]', r'\baudit [AB]-[0-9]', r'\bFD #[0-9]', r'\bS[0-9] gate\b', r'\bD[0-9] wants\b',
]


def _scrub(text, rules):
    """Apply the explicit rewrite rules, then let gate [1] judge the result.

    Deliberately NOT a clever generic sanitiser. The rules are written out one by one and reviewed;
    anything they miss is caught by the forbidden-token gate and shows up as a failed export with
    the offending line quoted, not as a leak. A regex that silently rewrote whatever it thought was
    a path would fail the other way round, which is the wrong direction for this particular file."""
    for pat, rep in rules:
        text = re.sub(pat, rep, text)
    return text


def _span(text, name):
    """Exact (first_line, last_line) of a top-level `def name`, 1-based and inclusive.

    Uses `ast`, NOT a regex on `^def `. The first version of this ran a block from its `def` line to
    the next top-level `def`/`class`, which quietly swallowed anything sitting BETWEEN two functions
    -- and mesh_build.py keeps a deferred `from scipy.spatial import Delaunay` exactly there. The
    bundle exported, passed all three gates, and then died with `NameError: Delaunay` the first time
    a mesh was built. The parser knows where a function ends; a regex only guesses."""
    tree = ast.parse(text)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            first = min([node.lineno] + [d.lineno for d in node.decorator_list])
            return first, node.end_lineno
    raise SystemExit('no top-level function %r' % name)


def _drop_functions(text, names):
    """Remove top-level `def name(...)` blocks -- used to leave solver-facing leaves behind.

    Only valid for functions nothing else in the file calls; the caller asserts that separately."""
    for n in names:
        lines = text.splitlines(keepends=True)
        first, last = _span(text, n)
        text = ''.join(lines[:first - 1] + lines[last:])
    return text


def _extract(text, names):
    """Pull named top-level functions out of a module, in the order given, verbatim."""
    lines = text.splitlines(keepends=True)
    out = []
    for n in names:
        first, last = _span(text, n)
        out.append(''.join(lines[first - 1:last]).rstrip() + '\n')
    return '\n\n'.join(out)


# ------------------------------------------------------------------------------------------------
# the rewrite rules, one per reference that exists today
# ------------------------------------------------------------------------------------------------
COMMON = [
    (r'`Phase 5/m2/([A-Za-z0-9_]+)\.py([^`]*)`', r'`\1.py\2`'),
    (r'`Phase [0-9]/([A-Za-z0-9_]+)\.py([^`]*)`', r'`\1.py\2`'),
    (r'`Phase 5/m2/M2_V2_PLAN\.md[^`]*`', 'the design notes'),
    (r'`Phase 5/verifications/test_m2_constraints\.py`', 'the constraint gate'),
    (r'`Phase 5/verifications/test_m2_head\.py`', 'the head gate'),
    (r'`Phase 5/verifications/[A-Za-z0-9_]+\.py`', 'a verification script'),
    (r'`Phase 5/results/[A-Za-z0-9_/]+`', 'the measurements'),
    (r'`Phase 2/forward_solver_torch\._angle_gradient_vec`', 'the reference solver'),
    (r'`Phase 2/forward_solver_torch[^`]*`', 'the reference solver'),
    (r'`Phase 2/[A-Za-z0-9_]+\.py[^`]*`', 'the reference implementation'),
    (r'`Phase 5/[A-Za-z0-9_]+\.py[^`]*`', 'the reference implementation'),
    (r'`CLAUDE\.md[^`]*`', 'the project conventions'),
    (r'`[A-Z][A-Z0-9_]+\.md[^`]*`', 'the notes'),
    (r'\bPhase [0-9]\b', 'the reference implementation'),
    (r'`model\.py` \(v1\)', 'the first-generation model'),
    (r'`seeds\.py: [A-Za-z0-9_]+`', 'the generators'),
    (r'`seeds\.py`', 'the generators'),
    (r'`fields\.displace`', 'a displacement helper'),
    (r'`build_dataset[^`]*`', 'the dataset builder'),
    (r'`train_v2\.prepare`', "the previous model's prepare"),
    (r'`train_v2[^`]*`', 'the previous training module'),
    (r'`model_v3\.py:[0-9]+`', '`model_v3.py`'),
    (r'`test_[a-z0-9_]+\.py[^`]*`', 'a gate'),
    (r'\btest_[a-z0-9_]+\.py\b', 'a gate'),
    (r'`train_v3\.oracle_check`', 'a training-time gate'),
    (r'`evaluate_v2\.py` scored', 'an evaluation script scored'),
    (r'Three scripts now load checkpoints \(`evaluate_v2`, `m2_v3_report`,\n'
     r'    `m2_error_strata`\), so', 'Several scripts load checkpoints, so'),
    (r'see the project CLAUDE\.md [^)]*\)', 'a distinction kept deliberately)'),
    (r'\(`evaluate_v2\.geo_of`\)', '(in a downstream consumer)'),
    # internal work-item ids (plan stages, audit findings, decision numbers). They say nothing to a
    # reader of the bundle and everything about how the private project is organised, so they go --
    # the SENTENCE they annotate stays, because that is the part with the engineering content.
    (r' \(A0\.[0-9], [0-9]{4}-[0-9]{2}-[0-9]{2}\)', ''),
    (r' \(A0\.[0-9]\)', ''),
    (r'\(A0\.[0-9]; D[0-9] wants', '(opt-in because'),
    (r' \(audit A-[0-9]+[a-z]?\)', ''),
    (r'\bThis is the S[0-9] gate\.', 'This is the closed-form gate.'),
    # the re-layering note above the deferred Delaunay import: entirely about which private module
    # used to own these functions. The import stays; the archaeology goes.
    (r'# ---- lattice construction \(MOVED(?:.|\n)*?-+\n(?=from scipy)',
     '# ---- lattice construction ------------------------------------------------------------\n'),
]

# `graph_of` carries a long docstring about a dataset-provenance flag and the validation protocol it
# would corrupt -- neither of which exists in the bundle. The FUNCTION is kept verbatim (it is the
# adapter between the mesh builder's names and the model's); only the docstring is rewritten.
GRAPH_OF_DOC = (
    r'"""The stored graph\.(?:.|\n)*?re-derivation\)\."""',
    '"""Adapt a mesh into the graph the model consumes.\n\n'
    '    The only real work is naming: the mesh builder calls the triangle-vertex table\n'
    '    `simplices` and the model calls it `tri_verts`, and the periodic box arrives as two basis\n'
    '    vectors rather than two lengths. `is_fictional` marks edges added only to triangulate a\n'
    '    non-triangular face; it is provenance and never a model input -- the model already sees\n'
    '    such an edge through `log(k/k_mean)`, which reads about -7 against ~0 for a real rib."""')

# Renames applied to the emitted network module. The bundle has no "v2"/"v3": version-numbered
# filenames carry no meaning for a reader of the demo and carry a little history for anyone else.
# The import is narrowed at the same time -- the original pulled five names and used three, with a
# `noqa: F401` acknowledging that two were dead.
MODEL_RENAMES = [
    (r'from model_v2 import [^\n]*',
     'from tensor_ops import assemble, edge_carriers, sym3_to_c6'),
    (r'`model_v3\.py`', '`model.py`'),
    (r'\bmodel_v3\b', 'model'),
    (r'\bmodel_v2\b', 'tensor_ops'),
]

TENSOR_OPS_HEADER = '''"""The tensor algebra the network is built on: edge carriers, assembly, Voigt packing.

Three functions, each self-contained. They are separated from the network itself because they are
convention rather than architecture -- the Voigt ordering [xx, xy, yy] with no factor of two on
shear, and the per-triangle assembly C(s) = Q G Q^T -- and because the inference pipeline needs them
without needing the model.
"""
import numpy as np
import torch


'''


# meshing.py's module docstring is pure provenance -- where each function was lifted from, which
# file is protected, which audit moved it. None of that can be rewritten line by line into something
# meaningful, so it is REPLACED wholesale with a docstring written for the bundle.
MESHING_DOC = '''"""
Mesh construction -- periodic and open triangulated spring networks.

Turns a lattice specification or a point cloud into the geometry dict the model consumes (`pts`,
`simplices`, `edge_vecs`, `actual_len2`, `areas`, deduplicated `bond_*`, and the triangle->bond map
`tri_bond`). Pure geometry and topology bookkeeping: no physics, no torch.

Implements the discrete setting of Grossman & Boudaoud, PRR 2026 (arXiv:2309.07844) section II: a
planar triangulated network of central-force springs, periodic (torus) or open (free boundary).

Conventions:
  - Per-triangle edges are ordered **(e01, e02, e12)** everywhere.
  - Periodic edge vectors are **unwrapped** (image-corrected), so a bond crossing the box has its
    true vector, not the wrapped-short one.
  - A bond is canonicalised by (min node, max node, image offset), so the two triangles sharing it
    agree on one index -- this is what makes per-BOND stiffness well defined.
  - float64 throughout.
"""'''

# `model_v2.py` is NOT in this table. Copying it whole shipped ten definitions to deliver three --
# including `ForwardGNNv2`, a complete earlier architecture, and that generation's feature
# engineering. The three the app actually uses are extracted into `tensor_ops.py` instead (see
# TENSOR_OPS below); they are self-contained, 24 lines in total.
FILES = {
    'model.py': dict(src='Phase 5/m2/model_v3.py', rules=COMMON + MODEL_RENAMES, drop=[]),
    # the solver-facing leaves: nothing else in the file calls them (asserted in main)
    'meshing.py': dict(src='Phase 2/mesh_build.py', rules=COMMON, doc=MESHING_DOC,
                       drop=['kkt_from_tri_bond', 'build_open_mesh', 'clean_tri']),
}


def _replace_module_docstring(text, new_doc):
    """Swap a module's leading docstring for one written for the bundle."""
    m = re.match(r'\s*(?:r?"""(?:.|\n)*?""")', text)
    if not m:
        raise SystemExit('no module docstring to replace')
    return new_doc + text[m.end():]


def _say(s):
    """Print without dying on a non-ASCII character under a cp1252 console.

    This project has already lost a gate to exactly that: a Greek nu in a final report line made a
    passing 16/16 suite report as a hard FAIL whenever stdout was piped. A reporting path must not
    be able to fail the thing it reports on."""
    enc = sys.stdout.encoding or 'utf-8'
    sys.stdout.write(s.encode(enc, errors='replace').decode(enc) + '\n')

INFERENCE_HEADER = '''r"""The forward pipeline: a periodic triangulated spring network -> its predicted elastic tensor.

`prepare` and `predict` are EXTRACTED from the training module rather than retyped, so what runs
here is the pipeline the weights were trained with. That is not pedantry: an earlier version of this
code built a single sample without its batch keys, which silently switched one architectural channel
OFF at evaluation and ON during training -- the two disagreed by 7.8e-02 and nothing crashed.

The one deliberate difference from training: `C6_per` (the label) is optional here, because at
inference there is no label. Everything else is byte-identical, and an export gate checks the
prediction against the research code to 1e-12.
"""
import numpy as np
import torch

import tensor_ops as M2
import model as M3

torch.set_default_dtype(torch.float64)


def build_model(ck):
    """Rebuild the network from a sanitised checkpoint.

    Delegates to `model_v3.from_checkpoint`, which is the ONE place that knows how to map a
    checkpoint onto an architecture -- two of this model's channels are optional, and a model built
    from module defaults silently mismatches a checkpoint trained without them. Rebuilding that
    mapping here would be a second copy of exactly the rule that exists because a second copy once
    scored a checkpoint against the wrong architecture."""
    return M3.from_checkpoint(ck, eval_mode=True)

'''

# Exactly what `model_v3.from_checkpoint` reads, plus `head` (architecture-descriptive, not
# protocol). `tie` / `n_iter` are carried only when present: from_checkpoint infers both from the
# state dict otherwise, and that inference is the documented fallback, not a guess.
BULK_HELPERS = '''

def bulk_c6(c6_per):
    """The project's homogenisation: the UNWEIGHTED mean of the per-triangle tensors.

    Unweighted, not area-weighted. Area weighting is a constraint-side device; using it in the final
    average biases nu on meshes whose triangles differ in size, which is every disordered mesh."""
    return c6_per.mean(0)


def directional(c6_per, n_theta=37):
    """nu(theta) and E(theta) over [0, pi], plus the two scalars, from a per-triangle prediction.

    This is the ONLY thing the API returns. The per-triangle tensors stay on the server: the model
    predicts `G = (I + X) G_an (I + X)^T` against a truth of the form `C = (1 + W)^T A (1 + W)`, so a
    per-triangle field would let a reader recover `(1 + W)^T = Q (I + X) Q^-1` -- the model's internal
    decomposition -- which the bulk curves do not reveal. The angle grid matches the project's
    canonical one: 37 points over [0, pi]."""
    import numpy as _np
    th = _np.linspace(0.0, _np.pi, n_theta)
    nu, E = c6_to_nuE_theta(bulk_c6(c6_per), th)
    return (th, nu.detach().numpy(), E.detach().numpy(),
            float(nu.detach().numpy()[0]), float(E.detach().numpy()[0]))
'''

SANITISE_KEEP = ('ns', 'nt', 'hidden', 'layers', 'use_star', 'use_global', 'head', 'state')
SANITISE_OPT = ('tie', 'n_iter')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True, help='bundle directory (OUTSIDE this repository)')
    ap.add_argument('--ckpt', default=os.path.join(
        HERE, 'checkpoint_v3_res_bravais_w10_h1_L5_ns48_h64_e400_star_ms.pt'))
    ap.add_argument('--n_check', type=int, default=8, help='meshes for the numerical gate')
    ap.add_argument('--data', default=os.path.join(HERE, 'data', 'auxgen_breadth.npz'))
    a = ap.parse_args()

    out_app = os.path.join(a.out, 'app')
    if os.path.abspath(a.out).startswith(REPO):
        raise SystemExit('refusing to export INTO the research repository: %s' % a.out)
    os.makedirs(out_app, exist_ok=True)

    emitted = {}

    # ---- the copied modules -------------------------------------------------------------------
    for name, spec in FILES.items():
        text = open(os.path.join(REPO, spec['src']), encoding='utf-8').read()
        for fn in spec['drop']:                       # leaves only -- assert nothing calls them
            body = re.sub(r'^def %s\(.*?(?=^(?:def |class ))' % re.escape(fn), '', text,
                          flags=re.M | re.S)
            if re.search(r'(?<!def )\b%s\(' % re.escape(fn), body):
                raise SystemExit('%s is CALLED inside %s -- cannot drop it' % (fn, spec['src']))
        text = _drop_functions(text, spec['drop'])
        if spec.get('doc'):
            text = _replace_module_docstring(text, spec['doc'])
        emitted[name] = _scrub(text, spec['rules'])

    # ---- tensor_ops.py: only what the app calls -------------------------------------------
    m2 = open(os.path.join(HERE, 'model_v2.py'), encoding='utf-8').read()
    emitted['tensor_ops.py'] = TENSOR_OPS_HEADER + _scrub(
        _extract(m2, ['edge_carriers', 'assemble', 'sym3_to_c6']), COMMON)

    # ---- the inference module, extracted ------------------------------------------------------
    tv3 = open(os.path.join(HERE, 'train_v3.py'), encoding='utf-8').read()
    bds = open(os.path.join(HERE, 'build_dataset.py'), encoding='utf-8').read()
    # ONE function out of the design module -- textbook anisotropic elasticity (build the compliance
    # 4-tensor, contract with the axial and transverse directions), not unpublished method. Taken by
    # extraction rather than retyping so the site's curves are the project's curves, and so gate [3]
    # covers it. The MODULE is still excluded: it imports the solver stack, and gate [2] enforces
    # that nothing of the sort rides along.
    idn = open(os.path.join(REPO, 'Phase 3', 'inverse_design.py'), encoding='utf-8').read()
    body = _extract(tv3, ['bond_vectors', 'prepare', 'predict'])
    body += '\n\n' + _scrub(_extract(bds, ['graph_of']), [GRAPH_OF_DOC])
    body += '\n\n' + _scrub(_extract(idn, ['c6_to_nuE_theta']),
                            [(r'Matches _common\.nu_E_theta exactly: build', 'Build')])
    body += BULK_HELPERS
    # the label is not available at inference; everything else stays exactly as trained
    body = body.replace("             target=torch.as_tensor(g['C6_per']),\n",
                        "             target=(torch.as_tensor(g['C6_per']) if 'C6_per' in g\n"
                        "                     else None),\n")
    if 'C6_per' not in body:
        raise SystemExit('the target line moved -- re-check the extraction')
    # MODEL_RENAMES here too: the header and the extracted docstrings both name `model_v3`,
    # and a bundle that has no such file should not mention one.
    emitted['inference.py'] = _scrub(INFERENCE_HEADER + body, COMMON + MODEL_RENAMES)

    # ---- gate [0]: the emitted files must PARSE ----------------------------------------------
    for name, text in emitted.items():
        try:
            compile(text, name, 'exec')
        except SyntaxError as e:
            raise SystemExit('GATE [0] syntax FAILED: %s line %s: %s' % (name, e.lineno, e.msg))
    _say('[0] syntax    OK   -- %d files parse' % len(emitted))

    # ---- gate [1]: scrub ----------------------------------------------------------------------
    leaks = []
    for name, text in emitted.items():
        for i, line in enumerate(text.splitlines(), 1):
            for pat in FORBIDDEN:
                if re.search(pat, line, re.I):
                    leaks.append('%s:%d  /%s/  %s' % (name, i, pat, line.strip()[:90]))
    if leaks:
        _say('GATE [1] scrub FAILED -- %d surviving reference(s):' % len(leaks))
        for row in leaks[:40]:
            _say('   ' + row)
        raise SystemExit(1)
    _say('[1] scrub     OK   -- no forbidden token in %d files' % len(emitted))

    for name, text in emitted.items():
        with open(os.path.join(out_app, name), 'w', encoding='utf-8', newline='\n') as f:
            f.write(text)

    # ---- gate [2]: imports --------------------------------------------------------------------
    bad = []
    for name, text in emitted.items():
        for m in re.finditer(r'^\s*(?:from\s+([A-Za-z0-9_.]+)\s+import|import\s+([A-Za-z0-9_.]+))',
                             text, re.M):
            mod = (m.group(1) or m.group(2)).split('.')[0]
            if mod not in ALLOWED_TOP and mod not in {'tensor_ops', 'model', 'meshing',
                                                      'inference'}:
                bad.append('%s: %s' % (name, mod))
    if bad:
        raise SystemExit('GATE [2] imports FAILED: %s' % bad)
    _say('[2] imports   OK   -- bundle imports only numpy / scipy / torch / stdlib')

    # ---- the sanitised checkpoint -------------------------------------------------------------
    ck = torch.load(a.ckpt, map_location='cpu', weights_only=False)
    clean = {k: ck[k] for k in SANITISE_KEEP}
    clean.update({k: ck[k] for k in SANITISE_OPT if k in ck})
    dropped = sorted(set(ck) - set(clean))
    dst = os.path.join(out_app, 'checkpoint.pt')
    torch.save(clean, dst)
    _say('[ckpt]        %.2f MB, dropped %s' % (os.path.getsize(dst) / 1e6, dropped))
    back = torch.load(dst, map_location='cpu', weights_only=False)
    if set(back) - set(SANITISE_KEEP) - set(SANITISE_OPT):
        raise SystemExit('sanitised checkpoint still carries provenance keys: %s'
                         % sorted(set(back) - set(SANITISE_KEEP) - set(SANITISE_OPT)))

    # ---- gate [3]: the bundle must give the SAME ANSWER as the research code ------------------
    sys.path.insert(0, HERE)
    import train_v3 as T3                                                      # noqa: E402
    import model_v3 as M3research                                              # noqa: E402

    # the canonical loader, so the gate runs on samples shaped exactly as training saw them
    import train_v2 as T2                                                       # noqa: E402
    samples = T2.load(a.data)[:a.n_check]

    net_r = M3research.from_checkpoint(ck, eval_mode=True)

    sys.path.insert(0, out_app)
    for m in ('tensor_ops', 'model', 'inference'):     # import the BUNDLE copies, not the repo's
        sys.modules.pop(m, None)
    import importlib                                                            # noqa: E402
    inf = importlib.import_module('inference')
    net_b = inf.build_model(back)

    worst = 0.0
    with torch.no_grad():
        for g in samples:
            g = dict(g)
            a_r = T3.predict(net_r, T3.prepare(g)).numpy()
            a_b = inf.predict(net_b, inf.prepare(g)).numpy()
            rel = np.abs(a_r - a_b).max() / max(np.abs(a_r).max(), 1e-300)
            worst = max(worst, float(rel))
    if worst > 1e-12:
        raise SystemExit('GATE [3] numeric FAILED: worst relative difference %.3e' % worst)
    _say('[3] numeric   OK   -- bundle vs research predict, worst rel %.3e on %d meshes'
          % (worst, len(samples)))
    print('\nbundle written to %s' % out_app)


if __name__ == '__main__':
    main()
