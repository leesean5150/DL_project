import torch
import torch.nn as nn

DEFAULT_LATENT_DIM = 64
DEFAULT_HIDDEN_DIMS = [256, 128]


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