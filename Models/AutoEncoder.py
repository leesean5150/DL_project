import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_LATENT_DIM = 64
DEFAULT_HIDDEN_DIMS = [256, 128]

# basic Autoencoder
class AutoEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = DEFAULT_LATENT_DIM,
        hidden_dims: list[int] | None = None,
        noise_std: float = 0.0,
    ):
        super().__init__()

        hidden_dims = hidden_dims or DEFAULT_HIDDEN_DIMS

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = list(hidden_dims)
        self.noise_std = noise_std

        encoder_layers = []
        prev_dim = input_dim
        for h in self.hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, h))
            encoder_layers.append(nn.GELU())
            prev_dim = h
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        prev_dim = latent_dim
        for h in reversed(self.hidden_dims):
            decoder_layers.append(nn.Linear(prev_dim, h))
            decoder_layers.append(nn.GELU())
            prev_dim = h
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def _maybe_add_noise(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.noise_std > 0:
            return x + torch.randn_like(x) * self.noise_std
        return x

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_noisy = self._maybe_add_noise(x)
        z = self.encoder(x_noisy)
        x_hat = self.decoder(z)
        return x_hat, z

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def reconstruction_error_vector(self, x: torch.Tensor) -> torch.Tensor:
        x_recon = self.reconstruct(x)
        return (x - x_recon).pow(2)

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        return self.reconstruction_error_vector(x).mean(dim=1)

    # def reconstruction_error(
    #     self,
    #     x: torch.Tensor,
    #     reduction: str = "mean",
    # ) -> torch.Tensor:
    #     x_hat = self.reconstruct(x)
    #     err = (x - x_hat) ** 2

    #     if reduction == "none":
    #         return err
    #     if reduction == "mean":
    #         return err.mean(dim=-1)
    #     if reduction == "sum":
    #         return err.sum(dim=-1)

    #     raise ValueError("reduction must be 'none', 'mean', or 'sum'")


# testing a VAE
class EnsembleVAE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = DEFAULT_LATENT_DIM,
        hidden_dims: list[int] | None = None,
        noise_std: float = 0.0,
        use_norm: bool = False,
    ):
        super().__init__()

        hidden_dims = hidden_dims or DEFAULT_HIDDEN_DIMS

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = list(hidden_dims)
        self.noise_std = noise_std
        self.use_norm = use_norm

        encoder_layers = []
        prev_dim = input_dim

        for h in self.hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, h))
            encoder_layers.append(nn.GELU())

            if use_norm:
                encoder_layers.append(nn.LayerNorm(h))

            prev_dim = h

        self.encoder_backbone = nn.Sequential(*encoder_layers)

        self.mu_head = nn.Linear(prev_dim, latent_dim)
        self.logvar_head = nn.Linear(prev_dim, latent_dim)

        decoder_layers = []
        prev_dim = latent_dim

        for h in reversed(self.hidden_dims):
            decoder_layers.append(nn.Linear(prev_dim, h))
            decoder_layers.append(nn.GELU())
            prev_dim = h

        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def _maybe_add_noise(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.noise_std > 0:
            return x + torch.randn_like(x) * self.noise_std
        return x

    def encode_dist(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self._maybe_add_noise(x)
        h = self.encoder_backbone(x)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x_noisy = self._maybe_add_noise(x)
        mu, logvar = self.encode_dist(x_noisy)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar, z

    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        mu, _ = self.encode_dist(x)
        z = mu
        return self.decode(z)

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        mu, _ = self.encode_dist(x)
        return mu

    @torch.no_grad()
    def reconstruction_error_vector(self, x: torch.Tensor) -> torch.Tensor:
        x_hat = self.reconstruct(x)
        return (x - x_hat).pow(2)

    @torch.no_grad()
    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        return self.reconstruction_error_vector(x).mean(dim=1)

    @torch.no_grad()
    def kl_per_sample(self, x: torch.Tensor) -> torch.Tensor:
        mu, logvar = self.encode_dist(x)
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        return kl.mean(dim=1)
    
def vae_loss(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    recon_weight: float = 1.0,
    kl_weight: float = 1e-3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    recon = F.mse_loss(x_hat, x, reduction="mean")

    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    kl = kl.sum(dim=1).mean()

    loss = recon_weight * recon + kl_weight * kl
    return loss, recon, kl