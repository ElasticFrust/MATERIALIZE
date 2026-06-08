
import sys
from pathlib import Path
sys.path.insert(0, str(Path('/home/user/MATERIALIZE/Phase 4').resolve()))
sys.path.insert(0, str(Path('/home/user/MATERIALIZE/Phase 4').resolve().parent / 'Phase 2'))
sys.path.insert(0, str(Path('/home/user/MATERIALIZE/Phase 4').resolve().parent / 'Phase 3'))
import torch
torch.multiprocessing.set_sharing_strategy('file_system')

import sys as _sys
_sys.argv = ['gen']  # suppress argparse

from run_generate_dataset_resumable import generate_split

generate_split('test', 10000)
