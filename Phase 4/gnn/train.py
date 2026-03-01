"""Training script for the GNN forward surrogate.

Trains the ForwardGNN model to predict Poisson's ratio from spring network graphs.
The training loop includes:
  - AdamW optimizer with cosine annealing warm restarts
  - Gradient clipping (max norm 5.0) for stability
  - Early stopping based on validation MSE
  - Best model checkpointing

Supports chunked data loading: if {data_dir}/{split}_chunks/ exists with
chunk_*.pt files, data is streamed one chunk at a time to avoid loading
the full ~80k-sample dataset into memory at once.

Training target: MAE(nu) < 0.02 on the test set. This means the GNN predicts
Poisson's ratio within ±0.02 of the true value on average — sufficient for the
CVAE's physics loss (which uses the GNN as a differentiable surrogate).

Usage:
    # Quick test (small dataset):
    python -m gnn.train --data_dir ./processed --epochs 50 --batch_size 32

    # Full training:
    python -m gnn.train --data_dir ./processed --epochs 300 --batch_size 64
"""

import argparse
import random
import torch
import torch.nn.functional as F
import numpy as np
import time
from pathlib import Path

try:
    from torch_geometric.loader import DataLoader
except ImportError:
    raise ImportError("torch_geometric required: pip install torch-geometric")

from gnn.model import ForwardGNN, ForwardGNNLoss


def load_split(data_dir, split):
    """Load a data split from chunks or a single .pt file.

    Prefers chunk directory ({split}_chunks/) if it exists and contains files.
    Falls back to {split}.pt. Returns a list of Data objects.
    Use this for val/test splits that fit in memory (~10k samples).
    """
    data_dir = Path(data_dir)
    chunk_dir = data_dir / f'{split}_chunks'

    if chunk_dir.exists():
        chunk_files = sorted(chunk_dir.glob('chunk_*.pt'))
        if chunk_files:
            data_list = []
            for cf in chunk_files:
                data_list.extend(torch.load(cf, weights_only=False))
            print(f"  Loaded {split}: {len(data_list)} samples from {len(chunk_files)} chunks")
            return data_list

    single_file = data_dir / f'{split}.pt'
    if single_file.exists():
        data_list = torch.load(single_file, weights_only=False)
        print(f"  Loaded {split}: {len(data_list)} samples from {single_file.name}")
        return data_list

    raise FileNotFoundError(f"No data for split '{split}' in {data_dir}")


def get_chunk_files(data_dir, split='train'):
    """Return sorted list of chunk file paths, or None if not chunked."""
    chunk_dir = Path(data_dir) / f'{split}_chunks'
    if chunk_dir.exists():
        files = sorted(chunk_dir.glob('chunk_*.pt'))
        if files:
            return files
    return None


def count_chunk_samples(chunk_files):
    """Count total samples across chunk files without loading all into memory."""
    total = 0
    for cf in chunk_files:
        total += len(torch.load(cf, weights_only=False))
    return total


def train_epoch(model, loader, optimizer, loss_fn, device, grad_clip=5.0):
    """Train for one epoch from a DataLoader."""
    model.train()
    total_loss = 0
    n_graphs = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch)
        loss = loss_fn(pred, batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
        n_graphs += batch.num_graphs

    return total_loss / n_graphs


def train_epoch_chunked(model, chunk_files, batch_size, optimizer, loss_fn,
                        device, grad_clip=5.0):
    """Train for one epoch, streaming chunks to bound memory usage.

    Shuffles chunk order each epoch and shuffles within each chunk.
    Peak memory: one chunk (~2000 samples) rather than the full dataset.
    """
    model.train()
    total_loss = 0
    n_graphs = 0

    # Shuffle chunk order each epoch for better generalization
    file_order = list(chunk_files)
    random.shuffle(file_order)

    for cf in file_order:
        chunk_data = torch.load(cf, weights_only=False)
        chunk_loader = DataLoader(chunk_data, batch_size=batch_size, shuffle=True)

        for batch in chunk_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch)
            loss = loss_fn(pred, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
            n_graphs += batch.num_graphs

        del chunk_data, chunk_loader

    return total_loss / n_graphs


@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluate model, return dict of metrics."""
    model.eval()
    total_mse = 0
    total_mae = 0
    n_graphs = 0

    for batch in loader:
        batch = batch.to(device)
        pred = model(batch)
        nu_pred = pred['nu']
        nu_true = batch.y_nu

        total_mse += F.mse_loss(nu_pred, nu_true, reduction='sum').item()
        total_mae += (nu_pred - nu_true).abs().sum().item()
        n_graphs += batch.num_graphs

    return {
        'mse': total_mse / n_graphs,
        'mae': total_mae / n_graphs,
        'rmse': np.sqrt(total_mse / n_graphs),
    }


def train(data_dir, output_dir='./checkpoints', epochs=300, batch_size=64,
          lr=1e-3, weight_decay=1e-5, patience=30, hidden=32, n_layers=4,
          multitask=False, device=None):
    """Full training loop with early stopping.

    Args:
        data_dir: directory containing {split}.pt or {split}_chunks/ subdirs.
        output_dir: where to save model checkpoints.
        epochs: max training epochs.
        batch_size: graphs per batch.
        lr: initial learning rate.
        weight_decay: L2 regularization.
        patience: early stopping patience.
        hidden: GNN hidden dimension.
        n_layers: number of message-passing layers.
        multitask: enable auxiliary Young's modulus and tensor heads.
        device: 'cuda' or 'cpu'.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data — use chunked streaming for train, load val fully
    print("Loading data...")
    train_chunk_files = get_chunk_files(data_dir, 'train')
    use_chunked = train_chunk_files is not None

    if use_chunked:
        n_train = count_chunk_samples(train_chunk_files)
        print(f"  Train: {n_train} samples across {len(train_chunk_files)} chunks (streamed)")
        train_loader = None  # not used; train_epoch_chunked iterates chunks directly
    else:
        train_data = torch.load(data_dir / 'train.pt', weights_only=False)
        n_train = len(train_data)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        print(f"  Train: {n_train} samples")

    val_data = load_split(data_dir, 'val')
    val_loader = DataLoader(val_data, batch_size=batch_size)

    # Model
    model = ForwardGNN(
        node_in=8, edge_in=5, hidden=hidden, n_layers=n_layers,
        multitask=multitask,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                   weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2
    )
    loss_fn = ForwardGNNLoss(multitask=multitask)

    # Training loop with early stopping
    best_val_mse = float('inf')
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        if use_chunked:
            train_loss = train_epoch_chunked(
                model, train_chunk_files, batch_size, optimizer, loss_fn,
                device)
        else:
            train_loss = train_epoch(model, train_loader, optimizer, loss_fn,
                                     device)
        val_metrics = evaluate(model, val_loader, device)
        scheduler.step()

        dt = time.time() - t0
        lr_now = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch:3d}/{epochs} | "
              f"train_loss={train_loss:.6f} | "
              f"val_mse={val_metrics['mse']:.6f} | "
              f"val_mae={val_metrics['mae']:.4f} | "
              f"lr={lr_now:.2e} | "
              f"{dt:.1f}s")

        # Early stopping
        if val_metrics['mse'] < best_val_mse:
            best_val_mse = val_metrics['mse']
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_mse': best_val_mse,
                'val_mae': val_metrics['mae'],
                'hidden': hidden,
                'n_layers': n_layers,
            }, output_dir / 'best_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # Load best model and evaluate on test set if available
    try:
        test_data = load_split(data_dir, 'test')
        checkpoint = torch.load(output_dir / 'best_model.pt', weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])

        test_loader = DataLoader(test_data, batch_size=batch_size)
        test_metrics = evaluate(model, test_loader, device)

        print(f"\n=== Test Results ===")
        print(f"MSE:  {test_metrics['mse']:.6f}")
        print(f"RMSE: {test_metrics['rmse']:.4f}")
        print(f"MAE:  {test_metrics['mae']:.4f}")
    except FileNotFoundError:
        print("No test split found, skipping test evaluation.")

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./processed')
    parser.add_argument('--output_dir', type=str, default='./checkpoints')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden', type=int, default=32)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--multitask', action='store_true')
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    train(**vars(args))
