import torch
import torch.nn as nn
import torch.nn.functional as F

from .AutoEncoder import AutoEncoder, GatedAutoEncoder


class FrozenAnomalyEncoderAdapter(nn.Module):
    """
    Wraps a pretrained AE/GAE checkpoint and exposes per-token auxiliary features.

    Input:
        x: [B, S, F_full]

    Output:
        aux: [B, S, F_aux]
    """
    def __init__(
        self,
        checkpoint_path: str,
        freeze: bool = True,
        include_latent: bool = True,
        include_recon_error: bool = True,
        include_gates: bool = False,
        log1p_error: bool = True,
    ):
        super().__init__()

        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        model_type = ckpt["model_type"]
        input_dim = ckpt["input_dim"]
        latent_dim = ckpt["latent_dim"]
        hidden_dims = ckpt["hidden_dims"]
        noise_std = ckpt.get("noise_std", 0.0)

        if model_type == "AutoEncoder":
            model = AutoEncoder(
                input_dim=input_dim,
                latent_dim=latent_dim,
                hidden_dims=hidden_dims,
                noise_std=noise_std,
            )
            self.is_gated = False
        elif model_type == "GatedAutoEncoder":
            model = GatedAutoEncoder(
                input_dim=input_dim,
                latent_dim=latent_dim,
                hidden_dims=hidden_dims,
                noise_std=noise_std,
            )
            self.is_gated = True
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        self.model = model
        self.include_latent = include_latent
        self.include_recon_error = include_recon_error
        self.include_gates = include_gates and self.is_gated
        self.log1p_error = log1p_error

        mask = torch.as_tensor(ckpt["mask"], dtype=torch.bool)
        self.register_buffer("keep_mask", mask)

        self.full_feature_dim = len(ckpt["feature_names"])
        self.filtered_feature_dim = input_dim
        self.latent_dim = latent_dim

        aux_dim = 0
        if self.include_latent:
            aux_dim += latent_dim
        if self.include_recon_error:
            aux_dim += 1
        if self.include_gates:
            aux_dim += input_dim

        self.aux_dim = aux_dim

        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False
        
        self.trainable_params = any(p.requires_grad for p in self.model.parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, S, F_full]
        returns aux: [B, S, aux_dim]
        """
        if x.ndim != 3:
            raise ValueError(f"x must have shape [B, S, F], got {tuple(x.shape)}")

        B, S, F_full = x.shape
        if F_full != self.full_feature_dim:
            raise ValueError(
                f"Expected full feature dim {self.full_feature_dim}, got {F_full}"
            )

        x_masked = x[:, :, self.keep_mask]          # [B, S, F_filtered]
        x_flat = x_masked.reshape(B * S, -1)        # [B*S, F_filtered]

        with torch.set_grad_enabled(self.trainable_params):
            if self.is_gated:
                x_hat, z, gates = self.model(x_flat)
            else:
                x_hat, z = self.model(x_flat)
                gates = None

        parts = []

        if self.include_latent:
            parts.append(z)

        if self.include_recon_error:
            err = ((x_hat - x_flat) ** 2).mean(dim=1, keepdim=True)
            if self.log1p_error:
                err = torch.log1p(err)
            parts.append(err)

        if self.include_gates:
            parts.append(gates)

        aux = torch.cat(parts, dim=1)               # [B*S, aux_dim]
        aux = aux.view(B, S, -1)                    # [B, S, aux_dim]
        return aux


class TransformerFraudModel(nn.Module):
    def __init__(
        self,
        feature_dim: int = 111,   # raw features dimensions
        seq_len: int = 8,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        use_second_projection: bool = False,

        # AE / GAE supplement
        anomaly_checkpoint_path: str | None = None,
        freeze_anomaly_encoder: bool = True,
        include_ae_latent: bool = True,
        include_ae_recon_error: bool = True,
        include_ae_gates: bool = False,
    ):
        super().__init__()

        if d_model % nhead != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by nhead ({nhead})"
            )

        self.feature_dim = feature_dim
        self.seq_len = seq_len
        self.d_model = d_model

        # Optional frozen AE/GAE branch
        if anomaly_checkpoint_path is not None:
            self.anomaly_adapter = FrozenAnomalyEncoderAdapter(
                checkpoint_path=anomaly_checkpoint_path,
                freeze=freeze_anomaly_encoder,
                include_latent=include_ae_latent,
                include_recon_error=include_ae_recon_error,
                include_gates=include_ae_gates,
            )
            token_input_dim = feature_dim + self.anomaly_adapter.aux_dim
        else:
            self.anomaly_adapter = None
            token_input_dim = feature_dim

        encoder_layers = [
            nn.LayerNorm(token_input_dim),
            nn.Linear(token_input_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        ]

        if use_second_projection:
            encoder_layers.extend([
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
            ])

        self.transaction_encoder = nn.Sequential(*encoder_layers)

        self.position_embedding = nn.Parameter(torch.zeros(1, seq_len, d_model))
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.sequence_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"x must have shape (B, S, F), got {tuple(x.shape)}")

        if attention_mask.ndim != 2:
            raise ValueError(
                f"attention_mask must have shape (B, S), got {tuple(attention_mask.shape)}"
            )

        B, S, F = x.shape

        if F != self.feature_dim:
            raise ValueError(
                f"Expected feature_dim={self.feature_dim}, got input feature dim={F}"
            )

        if S != self.seq_len:
            raise ValueError(
                f"Expected seq_len={self.seq_len}, got input seq len={S}"
            )

        if self.anomaly_adapter is not None:
            aux = self.anomaly_adapter(x)           # [B, S, aux_dim]
            x_token = torch.cat([x, aux], dim=-1)  # [B, S, F + aux_dim]
        else:
            x_token = x

        z = self.transaction_encoder(x_token)
        z = z + self.position_embedding

        key_padding_mask = ~attention_mask.bool()
        h = self.sequence_encoder(z, src_key_padding_mask=key_padding_mask)

        # current transaction is last token due to left padding
        h_last = h[:, -1, :]
        logits = self.classifier(h_last).squeeze(-1)
        return logits


class LastTokenMLP(nn.Module):
    def __init__(
        self,
        feature_dim: int = 111,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        anomaly_checkpoint_path: str | None = None,
        freeze_anomaly_encoder: bool = True,
        include_ae_latent: bool = True,
        include_ae_recon_error: bool = True,
        include_ae_gates: bool = False,
    ):
        super().__init__()

        self.feature_dim = feature_dim

        if anomaly_checkpoint_path is not None:
            self.anomaly_adapter = FrozenAnomalyEncoderAdapter(
                checkpoint_path=anomaly_checkpoint_path,
                freeze=freeze_anomaly_encoder,
                include_latent=include_ae_latent,
                include_recon_error=include_ae_recon_error,
                include_gates=include_ae_gates,
            )
            input_dim = feature_dim + self.anomaly_adapter.aux_dim
        else:
            self.anomaly_adapter = None
            input_dim = feature_dim

        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"x must have shape (B, S, F), got {tuple(x.shape)}")

        if x.shape[-1] != self.feature_dim:
            raise ValueError(
                f"Expected feature_dim={self.feature_dim}, got input feature dim={x.shape[-1]}"
            )

        x_last = x[:, -1, :]  # [B, F]

        if self.anomaly_adapter is not None:
            aux = self.anomaly_adapter(x)[:, -1, :]  # only last token aux
            x_last = torch.cat([x_last, aux], dim=-1)

        logits = self.net(x_last).squeeze(-1)
        return logits