r"""Angular-response companion plot for saved designs (pairs with gallery.py montages).

Project rule: whenever designed-network results are plotted, ALSO plot their angular response
nu(theta) / E(theta) with the target overlaid — the network picture shows the structure, the angular
response is the actual deliverable (especially for anisotropic targets). **All plotting goes through
the root `plotting.py` module** (CLAUDE.md 3): here we just load the saved designs and hand their
curves to `plotting.plot_directional` (cartesian MAIN + polar).

Usage:
    "C:\Users\doron\anaconda3\python.exe" "Phase 5\plot_responses.py" [tag]

Loads  Phase 5/networks/design_<tag>_*.npz  (default tag: 'aniso'), computes each design's
INDEPENDENT-simulation directional response from the stored per-triangle sim tensors (no re-run),
and draws all curves via plotting.plot_directional.  Saves  Phase 5/<tag>_response.png .
"""
import os, sys, glob
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, REPO)                                    # root plotting.py
sys.path.insert(0, os.path.join(REPO, 'Phase 3', 'verifications'))
import _common as C
from inverse_design import ANG
import plotting as P


def response_from_npz(path):
    """(nu_theta, E_theta, meta) of a saved design, from its stored SIM per-triangle tensors."""
    geo, bond_k, C6_per, meta = C.load_network(path)
    C6_bulk = C.sim_bulk_C6(geo)
    nu, E = C.nu_E_theta(C6_bulk, ANG)
    return np.asarray(nu), np.asarray(E), meta


def plot_responses(tag='aniso', out_path=None):
    here = os.path.dirname(__file__)
    paths = sorted(glob.glob(os.path.join(here, 'networks', f'design_{tag}_*.npz')))
    if not paths:
        raise SystemExit(f"no networks match design_{tag}_*.npz")
    out_path = out_path or os.path.join(here, f'{tag}_response.png')

    curves, meta = [], {}
    for p in paths:
        nu, E, meta = response_from_npz(p)
        curves.append((nu, E))

    fig = P.plot_directional(ANG, curves,
                             target_nu=meta.get('target_nu'), target_E=meta.get('target_E'),
                             polar=True, labels=[f"#{i}" for i in range(len(curves))],
                             suptitle=f"'{tag}' designs — independent-sim response")
    P.save_fig(fig, out_path)
    print(f"saved -> {out_path}   ({len(paths)} designs)")
    return out_path


if __name__ == '__main__':
    plot_responses(sys.argv[1] if len(sys.argv) > 1 else 'aniso')
