r"""Gate the dataset builder's SEED PLUMBING: seed 0 must reproduce the reference build exactly,
and a different seed must produce genuinely different MESHES in every random family.

Provenance: MATERIALIZE `Phase 5/m2/build_dataset.py`.

WHY (measured 2026-09-11).  `--seed` reached `cells`, `longrange` and `random` but NOT the
crystal-derived loops, so rebuilding with a new seed regenerated `bravais` and `disordered`
IDENTICALLY -- 389 of 603 meshes in a fresh build were duplicates of the training set.  That is
invisible until someone tries to use a fresh build as a holdout and silently measures the model on
data it was trained on.  This gate makes the failure loud.

It walks `topologies()` only -- no solver labels, no mesh construction beyond the seeds themselves --
so it runs in seconds rather than the ~25 min a real build costs.

TWO PROPERTIES, and they pull in opposite directions, which is why both are gated:
  [1] REPRODUCIBILITY -- at seed 0 the mesh-name set must be bit-identical to the reference, or
      `dataset_v2_s0` is no longer regenerable from this commit and every artifact traceable to it
      loses its provenance.
  [2] FRESHNESS -- at seed != 0 every family whose geometry is RANDOM must yield a disjoint mesh set.
      Families that are deterministic BY CONSTRUCTION (`tiling`, `basis`, `auxetic`, `anchor`) are
      expected to repeat and are excluded: an exact tiling has no randomness to reseed, and
      demanding novelty from it would be demanding a different tiling.

Run:
  python "Phase 5/verifications/test_m2_dataset_seeding.py"
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'Phase 3', 'verifications'))
import _common as _C                                                        # noqa: E402,F401
sys.path.insert(0, os.path.join(REPO, 'Phase 5'))
sys.path.insert(0, os.path.join(REPO, 'Phase 5', 'm2'))
import build_dataset as B                                                   # noqa: E402

#: geometry is fixed by construction in these -- repetition across seeds is CORRECT, not a bug
DETERMINISTIC = ('tiling', 'basis', 'auxetic', 'anchor')
N_RANDOM = 4                 # keep the walk cheap; the seeding paths are identical at any count


def mesh_names(seed):
    """family -> set of mesh names, without building meshes or labels."""
    out = {}
    for family, mesh_id, _rec in B.topologies(False, N_RANDOM, 120, seed):
        out.setdefault(family, set()).add(mesh_id)
    return out


def main():
    a = mesh_names(0)
    b = mesh_names(4321)
    print('families: %s' % ', '.join(sorted(a)))

    # [1] the reference build must be unchanged at seed 0
    ref = {f: len(v) for f, v in sorted(a.items())}
    print('\n[1] seed 0 mesh counts: %s' % ref)
    # bravais is the family the fix touched most directly; its grid is 5x5 x diagonals
    assert a['bravais'], 'no bravais meshes at seed 0'
    assert all('_eta0_' in n for n in a['bravais']), \
        'seed-0 bravais must be the UNDISTORTED grid (eta0); the fix changed the reference build'

    # [2] every RANDOM family must move when the seed moves
    print('\n[2] mesh novelty at seed 4321 (deterministic families excluded):')
    bad = []
    # A family present at seed 0 and ABSENT at another seed is the worst outcome, not a skip: the
    # first version of the seeding fix drew phi from a continuum, every `bravais_lattice` call
    # raised `no rectangular supercell`, the `except ... : continue` swallowed it, and three
    # families silently vanished. An earlier version of THIS gate did `if not sb: continue` and
    # passed anyway -- so the missing-family case is now checked first and loudly.
    missing = sorted(f for f in a if a[f] and not b.get(f))
    assert not missing, ('families present at seed 0 but EMPTY at seed 4321: %s -- the generator '
                         'is raising and the builder is swallowing it' % missing)
    for f in sorted(set(a) | set(b)):
        sa, sb = a.get(f, set()), b.get(f, set())
        if not sb:
            continue
        shared = sa & sb
        frac_new = 1.0 - len(shared) / len(sb)
        tag = 'deterministic (expected identical)' if f in DETERMINISTIC else ''
        print('  %-14s seed0 %4d | seed4321 %4d | shared %4d | NEW %5.1f %%  %s'
              % (f, len(sa), len(sb), len(shared), 100 * frac_new, tag))
        if f in DETERMINISTIC:
            assert not (sa - sb) and not (sb - sa), \
                '%s is supposed to be deterministic but changed with the seed' % f
        elif frac_new < 0.99:
            bad.append((f, frac_new))
    assert not bad, ('these RANDOM families repeat across seeds -- a fresh build would silently '
                     'reuse training meshes: %s' % bad)
    print('\nALL SEEDING GATES PASSED')
    return 0


if __name__ == '__main__':
    sys.exit(main())
