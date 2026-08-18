r"""Persistence for designed networks — the save-then-load contract.

MOVED here from `Phase 3/verifications/_common.py` (audit **A-7c**, 2026-08-18).
It lives in **Phase 2**, not Phase 5, because BOTH Phase 3's verification scripts and
Phase 5's production designer need it — a shared module must sit at or below the lower
consumer, or the dependency arrows invert. `_common` re-exports these names, so the ~100
verification scripts that call `C.save_network(...)` are untouched.

`save_network` stamps commit / dirty / saved_utc / seed (audit **B-3**). That stamp is what
makes stale artifacts *detectable*: anything without it predates the 2026-08 forward-map
fixes and is suspect (audit **A-19**, and `validation_2026-08/scan_provenance.py`).
"""
import json
import os
import subprocess

import numpy as np
import torch

_GIT_STATE = None            # memoised: git is shelled out ONCE per process, not once per save


def _provenance():
    """(commit, dirty, saved_utc) for the artifact-traceability stamp.

    The git query is cached for the process — a campaign saves hundreds of designs and the tree does
    not change under a running job. Never raises: a missing git, a detached checkout or a non-repo
    cwd must not be able to fail a long design run; provenance is best-effort and degrades to ''."""
    global _GIT_STATE
    import datetime
    import subprocess
    if _GIT_STATE is None:
        here = os.path.dirname(os.path.abspath(__file__))

        def _git(*a):
            try:
                return subprocess.run(('git',) + a, cwd=here, capture_output=True, text=True,
                                      timeout=10).stdout.strip()
            except Exception:                                # noqa: BLE001 — best-effort by design
                return ''
        _GIT_STATE = (_git('rev-parse', '--short', 'HEAD'), bool(_git('status', '--porcelain')))
    return (*_GIT_STATE,
            datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'))


def save_network(path, geo, bond_k, C6_per=None, seed=None, **meta):
    """Persist a DESIGNED network so later analysis/plots can reload it (load_network) WITHOUT
    re-running the minimizer. Stores geometry + designed per-bond k + (optional) per-triangle
    physical tensor + metadata (topo, size, target, region, achieved...).

    Also stamps PROVENANCE automatically — `commit`, `dirty`, `saved_utc`, and `seed` — so every
    saved artifact satisfies the charter's "traceable to (code version, config, seed)" rule
    (audit B-3: none of this was recorded). `dirty=True` means the tree had uncommitted changes when
    the design was produced, so `commit` alone does NOT reproduce it — that distinction is exactly
    what made the B-1 investigation expensive. Pass `seed=` explicitly; an unpassed seed is stored as
    None rather than silently invented."""
    import json
    k = bond_k.detach().numpy() if torch.is_tensor(bond_k) else np.asarray(bond_k)
    commit, dirty, saved = _provenance()
    meta.setdefault('commit', commit)
    meta.setdefault('dirty', dirty)
    meta.setdefault('saved_utc', saved)
    meta.setdefault('seed', seed)
    np.savez_compressed(path, pts=geo['pts'], tri_verts=geo['tri_verts'], centroids=geo['centroids'],
                        simplices=geo['simplices'], bond_u=geo['bond_u'], bond_v=geo['bond_v'],
                        bond_R=geo['bond_R'], tri_bond=geo['tri_bond'], areas=geo['areas'],
                        BL1=geo['BL1'], BL2=geo['BL2'], bond_k=k,
                        C6_per=(np.asarray(C6_per) if C6_per is not None else np.zeros(0)),
                        meta=json.dumps(meta))


def load_network(path):
    """Reload (geo, bond_k, C6_per, meta) written by save_network. geo has bond_k/tri_k installed,
    plus edge_vecs/actual_len2 rebuilt from tri_verts (needed by tri_metric_change/bare_tensor and
    the open_stretch* family) -- so it plugs straight into draw_network / fill_local_map /
    region_phys_C6 / nu_E_theta / open_stretch_nu with no extra per-caller reconstruction."""
    import json
    d = np.load(path, allow_pickle=True)
    geo = {k: d[k] for k in ['pts', 'tri_verts', 'centroids', 'simplices', 'bond_u', 'bond_v',
                             'bond_R', 'tri_bond', 'areas', 'BL1', 'BL2']}
    geo['bond_k'] = d['bond_k']; geo['tri_k'] = d['bond_k'][geo['tri_bond']]
    tv = geo['tri_verts']; p0, p1, p2 = tv[:, 0], tv[:, 1], tv[:, 2]
    geo['edge_vecs'] = np.stack([p1 - p0, p2 - p0, p2 - p1], 1)
    geo['actual_len2'] = (geo['edge_vecs'] ** 2).sum(2)
    C6 = d['C6_per']; C6 = None if C6.size == 0 else C6
    return geo, d['bond_k'], C6, json.loads(str(d['meta']))
