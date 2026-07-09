"""
Shared bootstrap for verification_tools/ scripts: mounts this directory and Phase 2 onto sys.path
(so peer modules and forward_solver_torch import cleanly whether a script is run directly or
imported from elsewhere, e.g. Phase 3/verifications/_common.py) and sets the non-interactive 'Agg'
matplotlib backend BEFORE pyplot is imported anywhere. Import this before `import matplotlib.pyplot`:
    import _bootstrap
    import matplotlib.pyplot as plt
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _p in (HERE, os.path.join(ROOT, 'Phase 2')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib
matplotlib.use('Agg')
