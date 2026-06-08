"""Full CVAE (Conditional Variational Autoencoder) for inverse elastic design.

The CVAE learns to generate edge parameter configurations (k, l0 per edge) that
produce a desired Poisson's ratio when evaluated by the forward solver. It combines:

  Encoder (training only): sees the full solution (graph + true k, l0 + nu_target)
           and compresses it to a latent code z ~ N(mu, sigma^2).

  Decoder (training + inference): sees only the graph topology + z + nu_target
           and generates per-edge (k, l0). At inference, z is sampled from N(0,I).

Loss function:
  L = L_recon + beta * L_KL + gamma * L_physics

  - L_recon: How well do the decoded (k, l0) match the true (k, l0)?
    Log-space MSE for rigidities (they span orders of magnitude).
    Normalized MSE for rest lengths (relative to l_actual).
    Only computed on hard (designable) edges, not soft regularization edges.

  - L_KL: KL divergence D_KL(q(z|x) || N(0,I)). Regularizes the latent space
    to be close to a standard normal, enabling sampling at inference time.
    Uses beta-warmup: beta linearly increases from 0 to beta_max over the first
    50 epochs. This prevents "posterior collapse" where the encoder ignores z.

  - L_physics: |nu_predicted - nu_target|^2. The predicted (k, l0) are fed
    through the GNN forward surrogate to check if they actually produce the
    target nu. This ensures physical consistency. Weighted by gamma=10.0.
    For epochs 1-200, uses the fast GNN surrogate. For epochs 201-300,
    can optionally switch to the actual solver for higher accuracy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from cvae.encoder import CVAEEncoder
from cvae.decoder import CVAEDecoder


class CVAE(nn.Module):
    """Conditional VAE for inverse design of elastic spring networks.

    Args:
        node_in: encoder node feature dim.
        edge_in: encoder edge feature dim (with k, l0).
        geom_node_in: decoder geometric node feature dim.
        geom_edge_in: decoder geometric edge feature dim.
        hidden: hidden dim for both encoder and decoder.
        latent_dim: latent code dimension.
        n_layers: NNConv layers in each component.
    """

    def __init__(self, node_in=8, edge_in=5, geom_node_in=4, geom_edge_in=4,
                 hidden=32, latent_dim=32, n_layers=4):
        super().__init__()
        self.latent_dim = latent_dim

        self.encoder = CVAEEncoder(
            node_in=node_in, edge_in=edge_in, hidden=hidden,
            latent_dim=latent_dim, n_layers=n_layers,
        )
        self.decoder = CVAEDecoder(
            latent_dim=latent_dim, geom_node_in=geom_node_in,
            geom_edge_in=geom_edge_in, hidden=hidden, n_layers=n_layers,
        )

    def reparameterize(self, mu, logvar):
        """Reparameterization trick: z = mu + sigma * eps."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, data, nu_target):
        """Full forward pass (training mode).

        Args:
            data: PyG Batch with full edge features.
            nu_target: (batch_size,) target Poisson ratios.

        Returns:
            k_pred, l0_pred: per-edge predicted parameters.
            mu, logvar: latent distribution parameters.
        """
        mu, logvar = self.encoder(data, nu_target)
        z = self.reparameterize(mu, logvar)
        k_pred, l0_pred = self.decoder(data, z, nu_target)
        return k_pred, l0_pred, mu, logvar

    def sample(self, data, nu_target, n_samples=1):
        """Sample designs from the prior (inference mode).

        Args:
            data: PyG Data/Batch with graph topology (no edge params needed).
            nu_target: (batch_size,) or scalar target Poisson ratio.
            n_samples: number of samples per graph.

        Returns:
            list of (k, l0) tuples, one per sample.
        """
        device = next(self.parameters()).device
        batch_size = data.batch.max().item() + 1 if hasattr(data, 'batch') else 1

        if isinstance(nu_target, (int, float)):
            nu_target = torch.full((batch_size,), nu_target,
                                   device=device, dtype=torch.float32)

        results = []
        for _ in range(n_samples):
            z = torch.randn(batch_size, self.latent_dim, device=device)
            k, l0 = self.decoder(data, z, nu_target)
            results.append((k.detach(), l0.detach()))

        return results


class CVAELoss(nn.Module):
    """CVAE loss with beta warmup and optional physics loss.

    Args:
        beta_max: maximum KL weight (reached after warmup).
        gamma: physics loss weight.
        warmup_epochs: epochs for linear beta warmup from 0 to beta_max.
    """

    def __init__(self, beta_max=1.0, gamma=10.0, warmup_epochs=50):
        super().__init__()
        self.beta_max = beta_max
        self.gamma = gamma
        self.warmup_epochs = warmup_epochs
        self._current_epoch = 0

    def set_epoch(self, epoch):
        self._current_epoch = epoch

    @property
    def beta(self):
        if self._current_epoch >= self.warmup_epochs:
            return self.beta_max
        return self.beta_max * self._current_epoch / self.warmup_epochs

    def reconstruction_loss(self, k_pred, l0_pred, data):
        """Log-space MSE for k, normalized MSE for l0.

        Log-space for k because rigidities span orders of magnitude.
        l0 is normalized by l_actual so errors are relative.
        """
        # Get true edge features from the data
        # edge_attr: (total_edges, 5) = [k, l0, l_actual, log_factor, is_real]
        n_unique = data.n_unique_edges if hasattr(data, 'n_unique_edges') else \
            data.edge_attr.shape[0] // 2

        # True values (first half of bidirectional edges = forward direction)
        k_true = data.edge_attr[:n_unique, 0]
        l0_true = data.edge_attr[:n_unique, 1]
        l_actual = data.edge_attr[:n_unique, 2]
        is_real = data.edge_attr[:n_unique, 4]

        # Only compute loss on designable (hard/real) edges
        mask = is_real > 0.5

        if mask.sum() == 0:
            return torch.tensor(0.0, device=k_pred.device)

        # Log-space loss for rigidities
        k_pred_masked = k_pred[:n_unique][mask]
        k_true_masked = k_true[mask]
        l_recon_k = F.mse_loss(
            torch.log(k_pred_masked.clamp(min=1e-8)),
            torch.log(k_true_masked.clamp(min=1e-8)),
        )

        # Normalized loss for rest lengths
        l0_pred_masked = l0_pred[:n_unique][mask]
        l0_true_masked = l0_true[mask]
        l_actual_masked = l_actual[mask].clamp(min=1e-8)
        l_recon_l0 = F.mse_loss(
            l0_pred_masked / l_actual_masked,
            l0_true_masked / l_actual_masked,
        )

        return l_recon_k + l_recon_l0

    def kl_loss(self, mu, logvar):
        """KL divergence: D_KL(q(z|x) || p(z)) where p(z) = N(0, I)."""
        return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    def forward(self, k_pred, l0_pred, mu, logvar, data,
                nu_predicted=None, nu_target=None):
        """Compute total CVAE loss.

        Args:
            k_pred, l0_pred: decoder outputs.
            mu, logvar: encoder outputs.
            data: PyG Batch with true edge features.
            nu_predicted: (batch_size,) from forward surrogate (optional).
            nu_target: (batch_size,) target Poisson ratio (optional).

        Returns:
            total_loss, dict of individual loss components.
        """
        l_recon = self.reconstruction_loss(k_pred, l0_pred, data)
        l_kl = self.kl_loss(mu, logvar)

        total = l_recon + self.beta * l_kl

        losses = {
            'recon': l_recon.item(),
            'kl': l_kl.item(),
            'beta': self.beta,
        }

        if nu_predicted is not None and nu_target is not None:
            l_physics = F.mse_loss(nu_predicted, nu_target)
            total = total + self.gamma * l_physics
            losses['physics'] = l_physics.item()

        losses['total'] = total.item()
        return total, losses
