"""Training script for the CVAE inverse design model.

The CVAE training requires a pre-trained GNN forward surrogate (from gnn/train.py)
for the physics loss. The training proceeds in stages:

  Epochs 1-50:   Beta warmup — KL weight linearly increases from 0 to beta_max.
                 This prevents "posterior collapse" where the encoder ignores z
                 and the decoder learns a deterministic mapping.

  Epochs 1-200:  Physics loss uses the GNN surrogate (fast, differentiable).
                 gamma=10.0 ensures the decoded parameters actually produce
                 the target nu, not just match the training data's parameters.

  Epochs 201-300: Optionally switch to the actual solver for physics loss
                  (slower but more accurate — not yet implemented in this script).

The training monitors reconstruction loss (edge params), KL divergence (latent
regularization), and physics loss (nu accuracy) separately for diagnosis.

Supports chunked data loading: if {data_dir}/{split}_chunks/ exists with
chunk_*.pt files, training data is streamed one chunk at a time.

Usage:
    # Requires a trained GNN model:
    python -m cvae.train --data_dir ./processed --gnn_checkpoint ./checkpoints/best_model.pt
"""

import argparse
import random
import torch
import numpy as np
import time
from pathlib import Path

try:
    from torch_geometric.loader import DataLoader
except ImportError:
    raise ImportError("torch_geometric required: pip install torch-geometric")

from cvae.model import CVAE, CVAELoss
from gnn.model import ForwardGNN


# Reuse chunk loading utilities from gnn.train
from gnn.train import load_split, get_chunk_files, count_chunk_samples
from data.generate_dataset import generate_chunk_live


def train_epoch(cvae, gnn_surrogate, loader, optimizer, loss_fn, device,
                use_physics_loss=True, grad_clip=5.0):
    """Train CVAE for one epoch."""
    cvae.train()
    if gnn_surrogate is not None:
        gnn_surrogate.eval()

    total_loss = 0
    total_losses = {}
    n_graphs = 0

    for batch in loader:
        batch = batch.to(device)
        nu_target = batch.y_nu
        optimizer.zero_grad()

        # Forward pass
        k_pred, l0_pred, mu, logvar = cvae(batch, nu_target)

        # Physics loss via GNN surrogate (optional)
        nu_predicted = None
        if use_physics_loss and gnn_surrogate is not None:
            # Build a modified batch with predicted edge features
            # For simplicity, update edge_attr in-place (detached copy)
            batch_copy = batch.clone()
            n_unique = batch.n_unique_edges if hasattr(batch, 'n_unique_edges') else \
                batch.edge_attr.shape[0] // 2
            # Update edge features with predicted k, l0
            new_attr = batch.edge_attr.clone()
            new_attr[:n_unique, 0] = k_pred[:n_unique]
            new_attr[:n_unique, 1] = l0_pred[:n_unique]
            new_attr[n_unique:, 0] = k_pred[:n_unique]
            new_attr[n_unique:, 1] = l0_pred[:n_unique]
            # Recompute log factor
            log_factor = torch.log((k_pred[:n_unique] / (l0_pred[:n_unique] ** 2)).clamp(min=1e-20))
            new_attr[:n_unique, 3] = log_factor
            new_attr[n_unique:, 3] = log_factor
            batch_copy.edge_attr = new_attr

            # Recompute node features with new edge attrs
            from data.graph_utils import init_node_features
            batch_copy.x = init_node_features(
                batch_copy.pos, batch_copy.edge_index, new_attr, batch_copy.x.shape[0]
            )

            gnn_pred = gnn_surrogate(batch_copy)
            nu_predicted = gnn_pred['nu']

        loss, losses = loss_fn(k_pred, l0_pred, mu, logvar, batch,
                               nu_predicted, nu_target)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(cvae.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs
        n_graphs += batch.num_graphs
        for k, v in losses.items():
            total_losses[k] = total_losses.get(k, 0) + v * batch.num_graphs

    avg_losses = {k: v / n_graphs for k, v in total_losses.items()}
    return total_loss / n_graphs, avg_losses


def train_epoch_chunked(cvae, gnn_surrogate, chunk_files, batch_size,
                        optimizer, loss_fn, device, use_physics_loss=True,
                        grad_clip=5.0):
    """Train CVAE for one epoch, streaming chunks to bound memory usage."""
    cvae.train()
    if gnn_surrogate is not None:
        gnn_surrogate.eval()

    total_loss = 0
    total_losses = {}
    n_graphs = 0

    file_order = list(chunk_files)
    random.shuffle(file_order)

    for cf in file_order:
        chunk_data = torch.load(cf, weights_only=False)
        chunk_loader = DataLoader(chunk_data, batch_size=batch_size, shuffle=True)

        for batch in chunk_loader:
            batch = batch.to(device)
            nu_target = batch.y_nu
            optimizer.zero_grad()

            k_pred, l0_pred, mu, logvar = cvae(batch, nu_target)

            nu_predicted = None
            if use_physics_loss and gnn_surrogate is not None:
                batch_copy = batch.clone()
                n_unique = batch.n_unique_edges if hasattr(batch, 'n_unique_edges') else \
                    batch.edge_attr.shape[0] // 2
                new_attr = batch.edge_attr.clone()
                new_attr[:n_unique, 0] = k_pred[:n_unique]
                new_attr[:n_unique, 1] = l0_pred[:n_unique]
                new_attr[n_unique:, 0] = k_pred[:n_unique]
                new_attr[n_unique:, 1] = l0_pred[:n_unique]
                log_factor = torch.log((k_pred[:n_unique] / (l0_pred[:n_unique] ** 2)).clamp(min=1e-20))
                new_attr[:n_unique, 3] = log_factor
                new_attr[n_unique:, 3] = log_factor
                batch_copy.edge_attr = new_attr

                from data.graph_utils import init_node_features
                batch_copy.x = init_node_features(
                    batch_copy.pos, batch_copy.edge_index, new_attr, batch_copy.x.shape[0]
                )

                gnn_pred = gnn_surrogate(batch_copy)
                nu_predicted = gnn_pred['nu']

            loss, losses = loss_fn(k_pred, l0_pred, mu, logvar, batch,
                                   nu_predicted, nu_target)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(cvae.parameters(), grad_clip)
            optimizer.step()

            total_loss += loss.item() * batch.num_graphs
            n_graphs += batch.num_graphs
            for k, v in losses.items():
                total_losses[k] = total_losses.get(k, 0) + v * batch.num_graphs

        del chunk_data, chunk_loader

    avg_losses = {k: v / n_graphs for k, v in total_losses.items()}
    return total_loss / n_graphs, avg_losses


def train_epoch_renewable_cvae(cvae, gnn_surrogate, n_chunks, chunk_size,
                               batch_size, optimizer, loss_fn, device,
                               use_physics_loss=True, grad_clip=5.0,
                               n_workers=4):
    """Train CVAE for one epoch, generating *fresh data* for every chunk.

    Mirror of GNN's train_epoch_renewable but for the CVAE. Each chunk is
    drawn fresh from the full topology × pattern × sigma distribution, so
    the CVAE never sees the same edge-parameter realization twice. This
    prevents the decoder from memorising specific (topology, nu_target) pairs
    and forces it to learn a general latent-space representation.

    Args:
        cvae: CVAE model to train.
        gnn_surrogate: frozen ForwardGNN for physics loss (or None).
        n_chunks: fresh chunks generated per epoch.
        chunk_size: sample configs attempted per chunk (~90-95% yield).
        batch_size: graphs per mini-batch.
        optimizer: AdamW.
        loss_fn: CVAELoss.
        device: torch.device.
        use_physics_loss: whether to compute GNN-based physics loss.
        grad_clip: max gradient norm.
        n_workers: parallel workers for generate_chunk_live().
    """
    cvae.train()
    if gnn_surrogate is not None:
        gnn_surrogate.eval()

    total_loss = 0
    total_losses = {}
    n_graphs = 0

    for ci in range(n_chunks):
        chunk_data = generate_chunk_live(chunk_size, n_workers=n_workers)
        if not chunk_data:
            print(f"  [chunk {ci+1}/{n_chunks}] WARNING: empty chunk, skipping",
                  flush=True)
            continue
        chunk_loader = DataLoader(chunk_data, batch_size=batch_size, shuffle=True)

        chunk_loss = 0
        chunk_graphs = 0
        for batch in chunk_loader:
            batch = batch.to(device)
            nu_target = batch.y_nu
            optimizer.zero_grad()

            k_pred, l0_pred, mu, logvar = cvae(batch, nu_target)

            nu_predicted = None
            if use_physics_loss and gnn_surrogate is not None:
                batch_copy = batch.clone()
                n_unique = batch.n_unique_edges if hasattr(batch, 'n_unique_edges') else \
                    batch.edge_attr.shape[0] // 2
                new_attr = batch.edge_attr.clone()
                new_attr[:n_unique, 0] = k_pred[:n_unique]
                new_attr[:n_unique, 1] = l0_pred[:n_unique]
                new_attr[n_unique:, 0] = k_pred[:n_unique]
                new_attr[n_unique:, 1] = l0_pred[:n_unique]
                log_factor = torch.log((k_pred[:n_unique] / (l0_pred[:n_unique] ** 2)).clamp(min=1e-20))
                new_attr[:n_unique, 3] = log_factor
                new_attr[n_unique:, 3] = log_factor
                batch_copy.edge_attr = new_attr

                from data.graph_utils import init_node_features
                batch_copy.x = init_node_features(
                    batch_copy.pos, batch_copy.edge_index, new_attr, batch_copy.x.shape[0]
                )

                gnn_pred = gnn_surrogate(batch_copy)
                nu_predicted = gnn_pred['nu']

            loss, losses = loss_fn(k_pred, l0_pred, mu, logvar, batch,
                                   nu_predicted, nu_target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(cvae.parameters(), grad_clip)
            optimizer.step()

            total_loss += loss.item() * batch.num_graphs
            chunk_loss += loss.item() * batch.num_graphs
            chunk_graphs += batch.num_graphs
            n_graphs += batch.num_graphs
            for k, v in losses.items():
                total_losses[k] = total_losses.get(k, 0) + v * batch.num_graphs

        avg_chunk = chunk_loss / chunk_graphs if chunk_graphs > 0 else 0
        print(f"  [chunk {ci+1}/{n_chunks}] loss={avg_chunk:.4f} "
              f"({chunk_graphs} live samples)", flush=True)

        import gc
        del chunk_data, chunk_loader
        gc.collect()

    avg_losses = {k: v / n_graphs for k, v in total_losses.items()}
    return total_loss / n_graphs if n_graphs > 0 else 0, avg_losses


@torch.no_grad()
def evaluate_cvae(cvae, gnn_surrogate, loader, loss_fn, device):
    """Evaluate CVAE on validation set."""
    cvae.eval()
    total_loss = 0
    n_graphs = 0

    for batch in loader:
        batch = batch.to(device)
        nu_target = batch.y_nu
        k_pred, l0_pred, mu, logvar = cvae(batch, nu_target)
        loss, _ = loss_fn(k_pred, l0_pred, mu, logvar, batch)
        total_loss += loss.item() * batch.num_graphs
        n_graphs += batch.num_graphs

    return total_loss / n_graphs


def train(data_dir, gnn_checkpoint=None, output_dir='./checkpoints',
          epochs=300, batch_size=32, lr=5e-4, weight_decay=1e-5,
          patience=40, hidden=32, latent_dim=32, n_layers=4,
          beta_max=1.0, gamma=10.0, warmup_epochs=50, device=None,
          renewable=False, renewable_n_chunks=40, renewable_chunk_size=2000,
          renewable_n_workers=4):
    """Full CVAE training loop.

    Args:
        renewable: generate fresh training data each chunk instead of loading
                   pre-saved chunks. Provides effectively infinite training
                   data. Val split is still loaded from disk for stable eval.
        renewable_n_chunks: fresh chunks per epoch.
        renewable_chunk_size: sample configs attempted per chunk (~90-95% yield).
        renewable_n_workers: parallel workers for live data generation.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data — use chunked streaming for train, load val fully
    print("Loading data...")

    if renewable:
        n_train_per_epoch = int(renewable_n_chunks * renewable_chunk_size * 0.93)
        print(f"  Train: RENEWABLE — {renewable_n_chunks} chunks × "
              f"~{int(renewable_chunk_size * 0.93)} samples/chunk "
              f"≈ {n_train_per_epoch} fresh samples/epoch")
        train_chunk_files = None
        train_loader = None
        use_chunked = False
    else:
        train_chunk_files = get_chunk_files(data_dir, 'train')
        use_chunked = train_chunk_files is not None

        if use_chunked:
            n_train = count_chunk_samples(train_chunk_files)
            print(f"  Train: {n_train} samples across {len(train_chunk_files)} chunks (streamed)")
            train_loader = None
        else:
            train_data = torch.load(data_dir / 'train.pt', weights_only=False)
            n_train = len(train_data)
            train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
            print(f"  Train: {n_train} samples")

    val_data = load_split(data_dir, 'val')
    val_loader = DataLoader(val_data, batch_size=batch_size)

    # CVAE model
    cvae = CVAE(hidden=hidden, latent_dim=latent_dim, n_layers=n_layers).to(device)
    n_params = sum(p.numel() for p in cvae.parameters())
    print(f"CVAE parameters: {n_params:,}")

    # GNN surrogate for physics loss
    gnn_surrogate = None
    if gnn_checkpoint and Path(gnn_checkpoint).exists():
        print(f"Loading GNN surrogate from {gnn_checkpoint}")
        gnn_surrogate = ForwardGNN(hidden=hidden, n_layers=n_layers).to(device)
        ckpt = torch.load(gnn_checkpoint, weights_only=False)
        gnn_surrogate.load_state_dict(ckpt['model_state_dict'])
        gnn_surrogate.eval()
        for p in gnn_surrogate.parameters():
            p.requires_grad = False

    # Optimizer, scheduler, loss
    optimizer = torch.optim.AdamW(cvae.parameters(), lr=lr,
                                   weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = CVAELoss(beta_max=beta_max, gamma=gamma, warmup_epochs=warmup_epochs)

    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        loss_fn.set_epoch(epoch)

        use_physics = gnn_surrogate is not None
        if renewable:
            train_loss, train_losses = train_epoch_renewable_cvae(
                cvae, gnn_surrogate, renewable_n_chunks, renewable_chunk_size,
                batch_size, optimizer, loss_fn, device,
                use_physics_loss=use_physics,
                n_workers=renewable_n_workers,
            )
        elif use_chunked:
            train_loss, train_losses = train_epoch_chunked(
                cvae, gnn_surrogate, train_chunk_files, batch_size,
                optimizer, loss_fn, device, use_physics_loss=use_physics,
            )
        else:
            train_loss, train_losses = train_epoch(
                cvae, gnn_surrogate, train_loader, optimizer, loss_fn, device,
                use_physics_loss=use_physics,
            )
        val_loss = evaluate_cvae(cvae, gnn_surrogate, val_loader, loss_fn, device)
        scheduler.step()

        dt = time.time() - t0
        recon = train_losses.get('recon', 0)
        kl = train_losses.get('kl', 0)
        phys = train_losses.get('physics', 0)
        beta = train_losses.get('beta', 0)

        print(f"Epoch {epoch:3d}/{epochs} | "
              f"loss={train_loss:.4f} | "
              f"recon={recon:.4f} | "
              f"kl={kl:.4f} | "
              f"phys={phys:.4f} | "
              f"beta={beta:.3f} | "
              f"val={val_loss:.4f} | "
              f"{dt:.1f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': cvae.state_dict(),
                'val_loss': best_val_loss,
                'hidden': hidden,
                'latent_dim': latent_dim,
                'n_layers': n_layers,
            }, output_dir / 'best_cvae.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    return cvae


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./processed')
    parser.add_argument('--gnn_checkpoint', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='./checkpoints')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--hidden', type=int, default=32)
    parser.add_argument('--latent_dim', type=int, default=32)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--renewable', action='store_true',
                        help='Generate fresh data every chunk (no pre-saved dataset needed)')
    parser.add_argument('--renewable_n_chunks', type=int, default=40,
                        help='Fresh chunks generated per epoch in renewable mode')
    parser.add_argument('--renewable_chunk_size', type=int, default=2000,
                        help='Sample configs attempted per live chunk')
    parser.add_argument('--renewable_n_workers', type=int, default=4,
                        help='Parallel workers for live data generation')
    args = parser.parse_args()

    train(**vars(args))
