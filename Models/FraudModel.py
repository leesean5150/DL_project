import torch
import torch.nn as nn

from .AutoEncoder import AutoEncoder

from typing import List, Optional


class GatedLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, activation=nn.GELU):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim * 2)
        self.activation = activation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = self.fc(x)
        a, b = x_proj.chunk(2, dim=-1)
        return self.activation(a) * torch.sigmoid(b)


class MLPBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        gated: bool = True,
        dropout: float = 0.0,
        use_norm: bool = True,
    ):
        super().__init__()

        if gated:
            self.layer = GatedLayer(in_dim, out_dim)
        else:
            self.layer = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.GELU(),
            )

        self.norm = nn.LayerNorm(out_dim) if use_norm else nn.Identity()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer(x)
        x = self.norm(x)
        x = self.dropout(x)
        return x


class EncoderBackbone(nn.Module):
    def __init__(self, encoder: AutoEncoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder.encode(x)


class FraudMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Optional[List[int]] = None,
        gated: bool = True,
        dropout: float = 0.1,
        use_norm: bool = True,
        encoder: Optional[AutoEncoder] = None,
        freeze_encoder: bool = False,
        use_recon_error_vector: bool = False,  # backward-compatible default (So our ipynb's don't break)
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        self.encoder = encoder
        self.freeze_encoder = freeze_encoder
        self.use_recon_error_vector = use_recon_error_vector
        self.input_dim = input_dim

        if self.encoder is not None and self.freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.eval()

        if self.encoder is None:
            prev_dim = input_dim
        else:
            if self.use_recon_error_vector:
                # raw x + latent z + full reconstruction error vector
                prev_dim = input_dim + self.encoder.latent_dim + input_dim
            else:
                # raw x + latent z + scalar reconstruction error
                prev_dim = input_dim + self.encoder.latent_dim + 1

        layers = []
        for h in hidden_dims:
            layers.append(
                MLPBlock(
                    prev_dim,
                    h,
                    gated=gated,
                    dropout=dropout,
                    use_norm=use_norm,
                )
            )
            prev_dim = h

        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(prev_dim, 1)

        self.config = {
            "input_dim": input_dim,
            "hidden_dims": list(hidden_dims),
            "gated": gated,
            "dropout": dropout,
            "use_norm": use_norm,
            "has_encoder": encoder is not None,
            "freeze_encoder": freeze_encoder,
            "encoder_latent_dim": getattr(encoder, "latent_dim", None),
            "use_recon_error_vector": use_recon_error_vector,
        }

    def _get_encoder_features(self, x: torch.Tensor): # Our testing showed that this is not great...
        """
        Returns:
            z: latent vector, shape [B, latent_dim]
            err_feat:
                - shape [B, 1] if use_recon_error_vector=False
                - shape [B, input_dim] if use_recon_error_vector=True
        """
        z = self.encoder.encode(x)

        if self.use_recon_error_vector:
            if not hasattr(self.encoder, "reconstruction_error_vector"):
                raise AttributeError(
                    "Encoder is missing reconstruction_error_vector(x). "
                    "Please add this method to AutoEncoder."
                )
            err_feat = self.encoder.reconstruction_error_vector(x)
        else:
            err_feat = self.encoder.reconstruction_error(x).unsqueeze(1)

        return z, err_feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.encoder is not None:
            if self.freeze_encoder:
                with torch.no_grad():
                    z, err_feat = self._get_encoder_features(x)
            else:
                z, err_feat = self._get_encoder_features(x)

            x = torch.cat([x, z, err_feat], dim=1)

        x = self.backbone(x)
        logits = self.head(x)
        return logits.squeeze(-1)

    def train(self, mode: bool = True):
        super().train(mode)

        if self.encoder is not None and self.freeze_encoder:
            self.encoder.eval()

        return self