"""PyTorch Geometric dataset wrapper for elastic mesh data.

Loads pre-generated .pt files (lists of Data objects) and optionally
generates data on-the-fly.
"""

import torch
from pathlib import Path

try:
    from torch_geometric.data import InMemoryDataset, Data
except ImportError:
    raise ImportError("torch_geometric is required. Install with: "
                      "pip install torch-geometric")


class ElasticMeshDataset(InMemoryDataset):
    """Dataset of elastic spring network graphs with labels.

    Can be initialized from:
      1. Pre-saved .pt file (list of Data objects)
      2. On-the-fly generation via generate_dataset()

    Usage:
        # From pre-generated file:
        dataset = ElasticMeshDataset(root='./data', split='train')

        # Generate and save:
        dataset = ElasticMeshDataset.generate_and_save(
            root='./data', split='train', n_samples=1000
        )
    """

    def __init__(self, root, split='train', transform=None, pre_transform=None):
        self.split = split
        super().__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return [f'{self.split}.pt']

    def process(self):
        # If processed file doesn't exist, check for raw .pt file
        raw_path = Path(self.root) / 'raw' / f'{self.split}.pt'
        if raw_path.exists():
            data_list = torch.load(raw_path, weights_only=False)
            self.save(data_list, self.processed_paths[0])
        else:
            # Generate a small default dataset
            from data.generate_dataset import generate_dataset
            print(f"No pre-generated data found. Generating small {self.split} set...")
            n = {'train': 500, 'val': 100, 'test': 100}.get(self.split, 100)
            data_list = generate_dataset(n)
            self.save(data_list, self.processed_paths[0])

    @classmethod
    def generate_and_save(cls, root, split='train', n_samples=1000,
                          n_workers=1, **kwargs):
        """Generate dataset and save in the expected format.

        Args:
            root: dataset root directory.
            split: 'train', 'val', or 'test'.
            n_samples: number of samples to generate.
            n_workers: parallel workers.

        Returns:
            ElasticMeshDataset instance.
        """
        from data.generate_dataset import generate_dataset

        root = Path(root)
        processed_dir = root / 'processed'
        processed_dir.mkdir(parents=True, exist_ok=True)

        data_list = generate_dataset(n_samples, n_workers=n_workers, **kwargs)

        save_path = processed_dir / f'{split}.pt'
        from torch_geometric.data import InMemoryDataset as _IMS
        # Use the InMemoryDataset save protocol
        torch.save(
            _IMS.collate(data_list),
            save_path,
        )
        print(f"Saved {len(data_list)} samples to {save_path}")

        return cls(root, split)
