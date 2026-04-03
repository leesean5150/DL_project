import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_LATENT_DIM = 32
DEFAULT_HIDDEN_DIMS = [256, 128]


class AutoEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = DEFAULT_LATENT_DIM,
        hidden_dims: list[int] | None = None,
        noise_std: float = 0.0,   # 0 if no noising, set > 0 for denoising AE
    ):
        super().__init__()

        hidden_dims = hidden_dims or DEFAULT_HIDDEN_DIMS

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = list(hidden_dims)
        self.noise_std = noise_std

        # Encoder
        encoder_layers = []
        prev_dim = input_dim

        for h in self.hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, h))
            encoder_layers.append(nn.GELU())
            prev_dim = h

        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
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
            noise = torch.randn_like(x) * self.noise_std
            return x + noise
        return x

    def forward(self, x: torch.Tensor):
        """
        x: [B, F]
        Returns:
            x_hat: [B, F]
            z:     [B, latent_dim]
        """
        x_noisy = self._maybe_add_noise(x)
        z = self.encoder(x_noisy)
        x_hat = self.decoder(z)
        return x_hat, z

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)

    def reconstruction_error_vector(
        self,
        x: torch.Tensor,
        reduction: str = "abs",
    ) -> torch.Tensor:
        """
        Returns per-feature reconstruction error: [B, F]

        reduction:
            - 'abs'     -> |x - x_hat|
            - 'squared' -> (x - x_hat)^2
        """
        x_hat = self.reconstruct(x)

        if reduction == "abs":
            return (x - x_hat).abs()
        elif reduction == "squared":
            return (x - x_hat) ** 2
        else:
            raise ValueError("reduction must be 'abs' or 'squared'")

    def reconstruction_error(
        self,
        x: torch.Tensor,
        reduction: str = "squared",
    ) -> torch.Tensor:
        """
        Scalar reconstruction error per sample: [B]
        Keeps backward compatibility with your current code.
        """
        err = self.reconstruction_error_vector(x, reduction=reduction)
        return err.mean(dim=-1)

    def grouped_reconstruction_error(
        self,
        x: torch.Tensor,
        groups: list[list[int]] | tuple[tuple[int, ...], ...],
        reduction: str = "abs",
        group_reduce: str = "mean",
    ) -> torch.Tensor:
        """
        Returns grouped reconstruction error: [B, G]

        groups:
            e.g. [[0,1,2], [3,4], [5,6,7,8]]

        reduction:
            per-feature error type: 'abs' or 'squared'

        group_reduce:
            - 'mean'
            - 'sum'
            - 'max'
        """
        if not groups:
            raise ValueError("groups must be a non-empty list of index groups")

        err = self.reconstruction_error_vector(x, reduction=reduction)
        grouped = []

        for group in groups:
            if len(group) == 0:
                raise ValueError("group entries must not be empty")

            group_err = err[:, group]  # [B, group_size]

            if group_reduce == "mean":
                grouped.append(group_err.mean(dim=-1))
            elif group_reduce == "sum":
                grouped.append(group_err.sum(dim=-1))
            elif group_reduce == "max":
                grouped.append(group_err.max(dim=-1).values)
            else:
                raise ValueError("group_reduce must be 'mean', 'sum', or 'max'")

        return torch.stack(grouped, dim=-1)

    def anomaly_features(
        self,
        x: torch.Tensor,
        groups: list[list[int]] | None = None,
        include_latent: bool = False,
        include_scalar_error: bool = True,
        include_feature_error: bool = False,
        include_group_error: bool = False,
        reduction: str = "abs",
        group_reduce: str = "mean",
    ) -> dict[str, torch.Tensor]:
        """
        Convenience helper to return multiple anomaly views at once.
        """
        out: dict[str, torch.Tensor] = {}

        if include_latent:
            out["latent"] = self.encode(x)

        if include_scalar_error or include_feature_error or include_group_error:
            feat_err = self.reconstruction_error_vector(x, reduction=reduction)

            if include_feature_error:
                out["feature_error"] = feat_err

            if include_scalar_error:
                out["scalar_error"] = feat_err.mean(dim=-1, keepdim=True)

            if include_group_error:
                if groups is None:
                    raise ValueError("groups must be provided when include_group_error=True")

                grouped = []
                for group in groups:
                    if len(group) == 0:
                        raise ValueError("group entries must not be empty")

                    group_err = feat_err[:, group]

                    if group_reduce == "mean":
                        grouped.append(group_err.mean(dim=-1))
                    elif group_reduce == "sum":
                        grouped.append(group_err.sum(dim=-1))
                    elif group_reduce == "max":
                        grouped.append(group_err.max(dim=-1).values)
                    else:
                        raise ValueError("group_reduce must be 'mean', 'sum', or 'max'")

                out["group_error"] = torch.stack(grouped, dim=-1)

        return out


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
        hidden_dims: list[int] | None = None,
        noise_std: float = 0.0,
    ):
        super().__init__(input_dim, latent_dim, hidden_dims, noise_std)

        hidden_dims = hidden_dims or DEFAULT_HIDDEN_DIMS
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
        x_noisy = self._maybe_add_noise(x)
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