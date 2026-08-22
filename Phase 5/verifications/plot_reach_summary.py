r"""What Poisson ratio has this project ACTUALLY achieved? — one figure over every experiment.

WHY
---
The reachable-nu evidence is scattered across four experiments with different setups, and one of
them (`goal1_frontier`, which holds the highest nu on record at **+0.906**) had **no figure at all** —
its result existed only as a table, which is a good way for a result to become invisible. This
collects them onto a single nu axis against the physical bound, so "how close to +-1 can we get, and
with what design freedom?" is answerable by looking.

SOURCES (all already computed; this script only LOADS and RENDERS — never re-optimises)
  - `Phase 5/results/goal1/results.csv`           k-design, 5 stiffness-contrast bands, 110 runs
  - `Phase 5/results/goal1_frontier/results.csv`  the high-nu probe, 24 runs (previously unplotted)
  - `Phase 5/results/g1_2/results.csv`            positions-only at k = 1 exactly, 110 runs
  - `Phase 3/verifications/auxetic_sweep/auxetic_sweep.csv`  the deep-auxetic sweep, 70 rows

CONVENTIONS
  - Every run is drawn. PALE interval = full reach; SOLID = the sub-range where solver and sim agree
    (`gap < 0.05`); points FILLED if trustworthy, HOLLOW if not. A design the gate rejects is a
    record of where the two code paths part company, not an absence (CLAUDE.md §3).
  - Rendering goes through the root `plotting.py` primitive `plot_ranges` — added for this figure
    rather than rolled locally, per the single-source-of-truth policy.

READ IT AS: contrast is the lever. Positions alone (k = 1) cannot pass the uniform-lattice +1/3 on
the high side; unrestricted k contrast reaches +0.906, and clamping contrast to f = 0.1 caps it near
+0.63. The same asymmetry holds on the auxetic side.

Determinism: pure rendering of saved CSVs — no RNG, no seed.

Run:  C:\\Users\\doron\\anaconda3\\python.exe "Phase 5/verifications/plot_reach_summary.py"
Out:  Phase 5/results/reach_summary/reach_summary.png
"""
# ---- §0 preamble (verbatim; this file lives in Phase 5/verifications/) ------------------------
import os, sys, csv
import numpy as np
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'Phase 3', 'verifications'))
import _common as C                                        # noqa: F401,E402  (wires sys.path)
sys.path.insert(0, REPO)
import plotting as P                                       # noqa: E402  — the ONLY render path
sys.path.insert(0, os.path.join(REPO, 'Phase 2'))
from persistence import _provenance                        # noqa: E402

OUT = os.path.join(REPO, 'Phase 5', 'results', 'reach_summary')
GAP_TOL = 0.05


def _rows(path):
    with open(path) as fh:
        return [r for r in csv.DictReader(fh)]


def _split(rows, nu_key, trust_fn):
    """dict(all=nu values, trust=boolean MASK over them), dropping non-finite/failed rows."""
    a, t = [], []
    for r in rows:
        try:
            nu = float(r[nu_key])
        except (KeyError, ValueError):
            continue
        if not np.isfinite(nu):
            continue
        a.append(nu)
        t.append(bool(trust_fn(r)))
    return dict(all=a, trust=t)


def collect():
    """Ordered groups, most permissive design freedom first."""
    g1 = _rows(os.path.join(REPO, 'Phase 5', 'results', 'goal1', 'results.csv'))
    fr = _rows(os.path.join(REPO, 'Phase 5', 'results', 'goal1_frontier', 'results.csv'))
    g12 = _rows(os.path.join(REPO, 'Phase 5', 'results', 'g1_2', 'results.csv'))
    ax = _rows(os.path.join(REPO, 'Phase 3', 'verifications', 'auxetic_sweep',
                            'auxetic_sweep.csv'))
    trust1 = lambda r: float(r['trustworthy']) >= 0.5              # noqa: E731
    groups = []
    # goal1 resolved by contrast band: the band IS the design-freedom axis
    for band, f in (('soft', '0'), ('large', '0.1'), ('medium', '0.5'),
                    ('small', '0.9'), ('none', '0.99')):
        sel = [r for r in g1 if r['band'] == band]
        if sel:
            groups.append((f'goal1  k-design, contrast f={f}', _split(sel, 'nu_sim_full', trust1)))
    groups.append(('goal1_frontier  high-nu probe (f=0, 0.1)', _split(fr, 'nu_sim_full', trust1)))
    groups.append(('g1_2  positions only, k=1 exactly', _split(g12, 'nu_achieved_sim', trust1)))
    # auxetic_sweep has no gap column; `stable` is its own screen, so treat stable as the solid arm
    groups.append(('auxetic_sweep  k-design, deep auxetic',
                   _split(ax, 'sim_nu', lambda r: str(r.get('stable', '')).strip()
                          in ('1', 'True', 'true'))))
    return groups


def main():
    groups = collect()
    os.makedirs(OUT, exist_ok=True)
    landmarks = [(1 / 3, 'uniform triangular +1/3'), (0.0, 'nu = 0'),
                 (-0.115, 'eta-disorder edge'), (1.0, 'hexagon closed form (d=2)')]

    def draw(ax):
        P.plot_ranges(groups, xlabel=r'achieved (independent-sim) $\nu$',
                      bounds=(-1.0, 1.0), landmarks=landmarks,
                      title=None, ax=ax)
        ax.plot([], [], color='0.25', lw=7, alpha=0.55, label='solver and sim agree (gap < 0.05)')
        ax.plot([], [], color='0.25', lw=7, alpha=0.30, label='full reach incl. gap > 0.05')
        ax.plot([], [], 'o', markersize=4.6, markerfacecolor='0.25', markeredgecolor='#0b0b0b',
                linestyle='none', label='run: agrees')
        ax.plot([], [], 'o', markersize=4.6, markerfacecolor='white', markeredgecolor='0.25',
                linestyle='none', label='run: gap > 0.05')
        # ABOVE the axes: every in-axes corner holds data on some row
        ax.legend(fontsize=7.5, loc='lower center', bbox_to_anchor=(0.5, 1.01), ncol=4,
                  frameon=False)
        ax.set_xlim(-1.08, 1.08)
        ax.set_title('Poisson ratio actually achieved, by design freedom',
                     fontsize=12, pad=42)      # pad clears the legend row above the axes

    path = P.save_element(draw, os.path.join(OUT, 'reach_summary.png'), figsize=(10.0, 5.6))
    commit, dirty, saved = _provenance()
    for name, d in groups:
        n, nt = len(d['all']), int(np.sum(d['trust']))
        print(f"  {name:44} n={n:3d} ({nt:3d} agree)  reach "
              f"[{min(d['all']):+.3f}, {max(d['all']):+.3f}]")
    print(f'\n-> {path}   [commit {commit[:7]} dirty={dirty}]')


if __name__ == '__main__':
    main()
