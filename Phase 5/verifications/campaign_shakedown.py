"""Campaign shakedown — run a CAMPAIGN DRIVER on a tiny subset, without touching it.

Stage 1 of `documentation/VERIFICATION_CAMPAIGN.md` is a shakedown: prove the pipeline works
end-to-end before spending ~3 h on the full sweep. The drivers take their grid from module-level
constants and write to fixed directories, so this wrapper **monkey-patches those constants and
redirects the output** rather than editing the driver — editing it would risk the shakedown's
settings leaking into the real run, and would dirty the tree the campaign requires to be clean.

What it validates (the things that would waste 3 h if broken):
  - the driver imports and runs under the post-audit stack;
  - a design completes, is verified against the INDEPENDENT sim, and is SAVED;
  - EVERY run is saved now, trustworthy or not (audit A-12) — untrusted ones marked;
  - a FAILED run is RECORDED rather than dropped (audit A-11), so the denominator is honest;
  - `_save()` survives heterogeneous rows (the latent crash found while fixing A-11);
  - artifacts carry commit / dirty / seed provenance (audit B-3).

Run:  python campaign_shakedown.py goal1
Out:  Phase 5/results/_shakedown_<driver>/ and Phase 5/networks/_shakedown_<driver>/
      (leading underscore: these are THROWAWAY, never campaign results)
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, 'Phase 3', 'verifications'))


def shakedown(name='goal1'):
    mod = __import__(f'run_{name}')

    out_res = os.path.join(REPO, 'Phase 5', 'results', f'_shakedown_{name}')
    out_net = os.path.join(REPO, 'Phase 5', 'networks', f'_shakedown_{name}')
    os.makedirs(out_res, exist_ok=True)
    os.makedirs(out_net, exist_ok=True)

    # --- the subset: one auxetic target and one positive, two bands, one rep = 4 runs ---------
    patches = dict(RESDIR=out_res, NETDIR=out_net)
    if hasattr(mod, 'NU_GRID'):
        patches['NU_GRID'] = np.array([-0.3, 0.2])
    if hasattr(mod, 'BANDS'):
        patches['BANDS'] = mod.BANDS[:2]
    if hasattr(mod, 'REPS_PER_COMBO'):
        patches['REPS_PER_COMBO'] = 1

    print(f"[shakedown] driver=run_{name}")
    for k, v in patches.items():
        print(f"[shakedown]   {k} := {v if not isinstance(v, str) else v}")
        setattr(mod, k, v)
    print(f"[shakedown] outputs -> {out_res}\n", flush=True)

    rc = mod.main()
    print(f"\n[shakedown] driver returned {rc!r}")
    return rc


if __name__ == '__main__':
    sys.exit(shakedown(sys.argv[1] if len(sys.argv) > 1 else 'goal1') or 0)
