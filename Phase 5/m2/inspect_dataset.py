"""Look at the M2 v2 training set BEFORE building it at scale.

`M2_V2_PLAN.md` §3 specifies the dataset; this renders a REPRESENTATIVE SAMPLE of what the builder
actually produces, so the coverage can be judged by eye rather than from summary statistics alone.
That is a deliberate habit, not decoration: `NEXT_SESSION.md` records six plotter defects found by
eye and none by an exit code, and the whole point of inspecting before a 10 k build is that a
mis-sampled axis is cheap to fix now and expensive to fix afterwards.

Renders, all through the root `plotting.py` primitives (the single source of truth for figures):

    reps_networks.png   one network per FAMILY and per k-PATTERN, tiled-continuous and cropped,
                        bonds coloured by k -- what the GNN will actually be shown
    reps_response.png   nu(theta), E(theta) for the same representatives, cartesian + polar
    coverage.png        the FULL TENSOR: every C6 component pair, so coverage is judged on the
                        training target (D2) and not on a scalar summary of it
    dilution.png        the rigidity axis: C6 and E against live coordination z through z_c = 4

Run (after a build):
    python "Phase 5/m2/inspect_dataset.py" --data data/dataset_smoke.npz
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                          # noqa: E402
import numpy as np                                                       # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, REPO)
import plotting as P                                                     # noqa: E402

C6_NAMES = ('xxxx', 'xxxy', 'xxyy', 'xyxy', 'xyyy', 'yyyy')


def load(path):
    d = np.load(path)
    n = len(d['C6'])

    def graph(i):
        """Rebuild the geo dict `plotting.draw_network` expects from the flat pointer scheme."""
        g = {}
        for key, dst in (('pts', 'pts'), ('bond_u', 'bond_u'), ('bond_v', 'bond_v'),
                         ('bond_R', 'bond_R'), ('tri_bond', 'tri_bond'),
                         ('tri_verts', 'simplices'), ('areas', 'areas'), ('k', 'k'),
                         ('is_fictional', 'is_fictional')):
            p = d[key + '_ptr']
            g[dst] = d[key][p[i]:p[i + 1]]
        g['BL1'] = np.array([d['Lx'][i], 0.0])
        g['BL2'] = np.array([0.0, d['Ly'][i]])
        return g
    return d, n, graph


def pick(d, n, key, per=1):
    """Representative indices per distinct value of `key`.

    `per=1` takes the sample nearest that group's MEDIAN |C6| -- a typical member.  `per=3` takes
    the group's **REACH**: the min-nu, median-nu and max-nu members.  Reach is the right default for
    a coverage figure, because a median-only pick shows the `auxetic` family through a member with
    nu > 0 and so hides the one property the family exists to supply.  (Same reason the project rule
    is to read reach over ALL runs, never a filtered subset -- that filter concealed a real result
    four separate times.)"""
    out = []
    mag = np.abs(d['C6']).max(1)
    for v in sorted(set(d[key])):
        idx = np.flatnonzero(d[key] == v)
        if per == 1:
            order = idx[np.argsort(np.abs(mag[idx] - np.median(mag[idx])))][:1]
            out.extend((str(v), int(i)) for i in order)
            continue
        by_nu = idx[np.argsort(d['nu'][idx])]
        chosen = [('{} | min'.format(v), by_nu[0]),
                  ('{} | median'.format(v), by_nu[len(by_nu) // 2]),
                  ('{} | max'.format(v), by_nu[-1])]
        out.extend((lab, int(i)) for lab, i in chosen[:per])
    return out


def fig_networks(d, n, graph, sel, out, suptitle):
    items = []
    for label, i in sel:
        g = graph(i)
        panel_title = ('{}\n$\\nu$={:+.3f}  E={:.2e}  '
                       'k$_{{max}}$/k$_{{min}}$={:.0e}').format(
            label, d['nu'][i], d['E'][i], d['contrast'][i])
        # Log colour scale wherever the contrast is large: a linear norm pushes the whole live
        # population into the bottom few percent of viridis, and the correctly-dashed near-zero
        # bonds then read as a torn mesh (CLAUDE.md §3, the B-4 finding).
        scale = 'log' if d['contrast'][i] > 50 else 'linear'
        # DRAW EVERY EDGE THAT ENTERS THE COMPUTE (user, 2026-08-25). A fictional bond is still in
        # the solve -- it just carries almost no load -- so hiding it would misrepresent what was
        # computed. Mark it instead: solid = real rib, dashed = fictional. Passing the mask
        # explicitly also beats `draw_network`'s median-relative default, which stops dashing them
        # as soon as a k-field lifts them off zero.
        fict = np.asarray(g['is_fictional'], bool)
        solid = ~fict if fict.any() else None
        items.append((g, g['k'], panel_title, solid, scale))
    P.montage(items, out, ncols=3, suptitle=suptitle, cbar_label='k / mean(k)')
    return out


def fig_response(d, sel, out, title):
    th = np.linspace(0, np.pi, d['nu_theta'].shape[1])
    curves = [(d['nu_theta'][i], d['E_theta'][i]) for _, i in sel]
    fig = P.plot_directional(th, curves, labels=[str(l) for l, _ in sel], suptitle=title)
    P.save_fig(fig, out)
    return out


def fig_coverage(d, n, out):
    """The FULL TENSOR, component by component (decision D2).

    Coverage judged on a scalar nu would hide exactly what the model is trained to predict: an
    anisotropic C with nu(theta) running over several units can have a direction-averaged nu of 0.
    """
    fams = sorted(set(d['family']))
    cmap = plt.get_cmap('tab10')
    col = {f: cmap(i % 10) for i, f in enumerate(fams)}
    pairs = [(0, 5), (0, 3), (2, 3), (1, 4), (0, 2), (3, 5)]
    fig, axes = plt.subplots(2, 3, figsize=(P.STYLE.PANEL[0] * 3, P.STYLE.PANEL[1] * 2))
    for ax, (a, b) in zip(axes.ravel(), pairs):
        for f in fams:
            m = d['family'] == f
            ax.scatter(d['C6'][m, a], d['C6'][m, b], s=9, alpha=0.65, color=col[f], label=f,
                       linewidths=0)
        ax.set_xlabel(f'$C_{{{C6_NAMES[a]}}}$')
        ax.set_ylabel(f'$C_{{{C6_NAMES[b]}}}$')
        ax.grid(alpha=0.25)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=len(fams), frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f'Training target coverage — the full tensor, {n} samples '
                 f'({d["spd"].sum()}/{n} SPD)')
    fig.tight_layout()
    P.save_fig(fig, out)
    return out


def fig_dilution(d, out):
    """The rigidity axis the v1 zoo never visited: every mesh there is a triangulation, so live
    coordination is z = 6 exactly. Dilution walks z down through the 2D isostatic point z_c = 4."""
    m = np.char.startswith(d['k_pattern'].astype(str), 'dilution')
    if not m.any():
        return None
    frac = np.array([float(str(s).split('_f')[1]) for s in d['k_pattern'][m]])
    fig, axes = plt.subplots(1, 3, figsize=(P.STYLE.PANEL[0] * 3, P.STYLE.PANEL[1]))
    axes[0].semilogy(frac + np.random.default_rng(0).normal(0, .006, frac.size), d['E'][m],
                     'o', ms=4, alpha=.6)
    axes[0].set_xlabel('dilution fraction $f$'); axes[0].set_ylabel('$E$')
    axes[0].set_title('modulus collapses as bonds go soft')
    for j, nm in enumerate(C6_NAMES):
        axes[1].plot(frac, d['C6'][m, j], 'o', ms=3, alpha=.5, label=f'$C_{{{nm}}}$')
    axes[1].set_xlabel('dilution fraction $f$'); axes[1].set_ylabel('$C$ component')
    axes[1].set_title('the full tensor along the rigidity axis'); axes[1].legend(fontsize=7, ncol=2)
    axes[2].plot(frac, d['nu'][m], 'o', ms=4, alpha=.6)
    axes[2].axhline(0, color='k', lw=.8)
    axes[2].set_xlabel('dilution fraction $f$'); axes[2].set_ylabel(r'$\nu$ (direction-averaged)')
    axes[2].set_title(r'$\nu$ swings hard near isostatic')
    for ax in axes:
        ax.grid(alpha=.25)
    fig.suptitle('Bond dilution — the coordination axis (labels sim-verified, 16/16 to 4 decimals)')
    fig.tight_layout()
    P.save_fig(fig, out)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', default=os.path.join(HERE, 'data', 'dataset_smoke.npz'))
    ap.add_argument('--outdir', default=os.path.join(REPO, 'Phase 5', 'results', 'm2_dataset'))
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    d, n, graph = load(a.data)

    made = []
    made.append(fig_networks(d, n, graph, pick(d, n, 'family', per=3),
                             os.path.join(a.outdir, 'reps_by_family.png'),
                             'Each FAMILY at its min / median / max $\\nu$ — the REACH it supplies'))
    made.append(fig_networks(d, n, graph, pick(d, n, 'k_pattern'),
                             os.path.join(a.outdir, 'reps_by_kpattern.png'),
                             'One representative per k-PATTERN — the four knobs of §3.1h'))
    made.append(fig_response(d, pick(d, n, 'family'),
                             os.path.join(a.outdir, 'reps_response.png'),
                             r'Directional response $\nu(\theta), E(\theta)$ by family'))
    made.append(fig_coverage(d, n, os.path.join(a.outdir, 'coverage_tensor.png')))
    made.append(fig_dilution(d, os.path.join(a.outdir, 'dilution_axis.png')))
    for p in made:
        if p:
            print('wrote', p)


if __name__ == '__main__':
    main()
