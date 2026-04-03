import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_LATENT_DIM = 32

class AutoEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = DEFAULT_LATENT_DIM,
        hidden_dims: list[int] = [256, 128],
        noise_std: float = 0.0,   # 0 if no noising, set > 0 for denoising AE
    ):
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.noise_std = noise_std

        # Encoder
        encoder_layers = []
        prev_dim = input_dim

        for h in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, h))
            encoder_layers.append(nn.GELU())
            prev_dim = h

        encoder_layers.append(nn.Linear(prev_dim, latent_dim))

        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = []
        prev_dim = latent_dim

        for h in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(prev_dim, h))
            decoder_layers.append(nn.GELU())
            prev_dim = h

        decoder_layers.append(nn.Linear(prev_dim, input_dim))

        self.decoder = nn.Sequential(*decoder_layers)


    def forward(self, x: torch.Tensor):
        """
        x: [B, F]
        """
        # Optional denoising (injected noise)
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(x) * self.noise_std
            x_noisy = x + noise
        else:
            x_noisy = x

        z = self.encoder(x_noisy)
        x_hat = self.decoder(z)

        return x_hat, z

    def encode(self, x: torch.Tensor):
        return self.encoder(x)

    def reconstruct(self, x: torch.Tensor):
        z = self.encoder(x)
        return self.decoder(z)

    def reconstruction_error(self, x: torch.Tensor):
        x_hat = self.reconstruct(x)
        return ((x - x_hat) ** 2).mean(dim=-1)
    

class FeatureGate(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.gate = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        g = torch.sigmoid(self.gate(x))
        return x * g, g
    
class GatedAutoEncoder(AutoEncoder):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = DEFAULT_LATENT_DIM,
        hidden_dims: list[int] = [256, 128],
        noise_std: float = 0.0,
    ):
        super().__init__(input_dim, latent_dim, hidden_dims, noise_std)

        self.feature_gate = FeatureGate(input_dim)

        encoder_layers = []
        prev_dim = input_dim

        for h in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, h))
            encoder_layers.append(nn.GELU())
            prev_dim = h

        encoder_layers.append(nn.Linear(prev_dim, latent_dim))

        self.encoder = nn.Sequential(*encoder_layers)

    def forward(self, x: torch.Tensor):
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(x) * self.noise_std
            x_noisy = x + noise
        else:
            x_noisy = x

        gated_x, gates = self.feature_gate(x_noisy)
        z = self.encoder(gated_x)
        x_hat = self.decoder(z)

        return x_hat, z, gates

    def encode(self, x: torch.Tensor):
        gated_x, gates = self.feature_gate(x)
        z = self.encoder(gated_x)
        return z, gates

    def reconstruct(self, x: torch.Tensor):
        gated_x, _ = self.feature_gate(x)
        z = self.encoder(gated_x)
        return self.decoder(z)

