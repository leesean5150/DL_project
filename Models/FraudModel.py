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
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        self.encoder = encoder
        self.freeze_encoder = freeze_encoder

        if self.encoder is not None and self.freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.eval()

        if self.encoder is None:
            prev_dim = input_dim
        else:
            prev_dim = input_dim + self.encoder.latent_dim + 1
            # prev_dim = input_dim + 1

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
        }

    def forward(self, x):
        if self.encoder is not None:
            if self.freeze_encoder:
                with torch.no_grad():
                    z = self.encoder.encode(x)
                    err = self.encoder.reconstruction_error(x).unsqueeze(1)
            else:
                z = self.encoder.encode(x)
                err = self.encoder.reconstruction_error(x).unsqueeze(1)

            x = torch.cat([x, z, err], dim=1)
            # x = torch.cat([x, err], dim=1)

        x = self.backbone(x)
        logits = self.head(x)
        return logits.squeeze(-1)

    def train(self, mode: bool = True):
        super().train(mode)

        if self.encoder is not None and self.freeze_encoder:
            self.encoder.eval()

        return self