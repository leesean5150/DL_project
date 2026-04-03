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

class AttentionPooling(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.0):
        super().__init__()
        self.score = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, h: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # h: [B, S, D]
        # attention_mask: [B, S] with 1 for valid tokens, 0 for padding
        scores = self.score(h).squeeze(-1)  # [B, S]

        scores = scores.masked_fill(~attention_mask.bool(), float("-inf"))
        weights = torch.softmax(scores, dim=1)  # [B, S]

        pooled = torch.sum(h * weights.unsqueeze(-1), dim=1)  # [B, D]
        return pooled

class AEAttentionPoolingTransformerFraudModel(nn.Module):
    def __init__(
        self,
        feature_dim: int = 111,
        seq_len: int = 8,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        use_second_projection: bool = False,

        anomaly_checkpoint_path: str | None = None,
        freeze_anomaly_encoder: bool = True,
        include_ae_latent: bool = True,
        include_ae_recon_error: bool = True,
        include_ae_gates: bool = False,

        use_ae_token_residual: bool = True,
        use_ae_head_skip: bool = True,
        ae_hidden_dim: int | None = None,
        init_ae_residual_scale: float = 0.10,
    ):
        super().__init__()

        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")

        self.feature_dim = feature_dim
        self.seq_len = seq_len
        self.d_model = d_model
        self.use_ae_token_residual = use_ae_token_residual
        self.use_ae_head_skip = use_ae_head_skip

        if anomaly_checkpoint_path is not None:
            self.anomaly_adapter = FrozenAnomalyEncoderAdapter(
                checkpoint_path=anomaly_checkpoint_path,
                freeze=freeze_anomaly_encoder,
                include_latent=include_ae_latent,
                include_recon_error=include_ae_recon_error,
                include_gates=include_ae_gates,
            )
            aux_dim = self.anomaly_adapter.aux_dim
        else:
            self.anomaly_adapter = None
            aux_dim = 0

        raw_layers = [
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        ]
        if use_second_projection:
            raw_layers.extend([
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
        self.raw_encoder = nn.Sequential(*raw_layers)

        if aux_dim > 0:
            ae_hidden_dim = ae_hidden_dim or d_model
            self.ae_encoder = nn.Sequential(
                nn.LayerNorm(aux_dim),
                nn.Linear(aux_dim, ae_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(ae_hidden_dim, d_model),
                nn.Dropout(dropout),
            )
            self.ae_residual_scale = nn.Parameter(
                torch.tensor(init_ae_residual_scale, dtype=torch.float32)
            )
        else:
            self.ae_encoder = None
            self.ae_residual_scale = None

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

        self.attention_pool = AttentionPooling(d_model=d_model, dropout=dropout)

        head_input_dim = d_model
        if aux_dim > 0 and use_ae_head_skip:
            head_input_dim += d_model

        self.classifier = nn.Sequential(
            nn.LayerNorm(head_input_dim),
            nn.Linear(head_input_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)

    def _masked_mean(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).float()  # [B, S, 1]
        x_sum = (x * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return x_sum / denom

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"x must have shape (B, S, F), got {tuple(x.shape)}")
        if attention_mask.ndim != 2:
            raise ValueError(
                f"attention_mask must have shape (B, S), got {tuple(attention_mask.shape)}"
            )

        B, S, F = x.shape
        if F != self.feature_dim:
            raise ValueError(f"Expected feature_dim={self.feature_dim}, got {F}")
        if S != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, got {S}")

        z_raw = self.raw_encoder(x)  # [B, S, D]

        ae_proj = None
        if self.anomaly_adapter is not None:
            ae_aux = self.anomaly_adapter(x)       # [B, S, aux_dim]
            ae_proj = self.ae_encoder(ae_aux)      # [B, S, D]
            z = z_raw + self.ae_residual_scale * ae_proj
        else:
            z = z_raw

        z = z + self.position_embedding

        key_padding_mask = ~attention_mask.bool()
        h = self.sequence_encoder(z, src_key_padding_mask=key_padding_mask)

        pooled = self.attention_pool(h, attention_mask)  # [B, D]

        if ae_proj is not None and self.use_ae_head_skip:
            ae_pooled = self._masked_mean(ae_proj, attention_mask)  # [B, D]
            pooled = torch.cat([pooled, ae_pooled], dim=-1)

        logits = self.classifier(pooled).squeeze(-1)
        return logits
    
class GatedTransformerFraudModel(nn.Module):
    def __init__(
        self,
        feature_dim: int = 111,
        seq_len: int = 8,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        use_second_projection: bool = False,

        # raw feature gate
        gate_hidden_dim: int | None = None,
        gate_min_scale: float = 0.10,
        gate_reg_weight: float = 1e-3,
        use_gate_mlp: bool = False,

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

        if not (0.0 <= gate_min_scale <= 1.0):
            raise ValueError("gate_min_scale must be in [0, 1].")

        self.feature_dim = feature_dim
        self.seq_len = seq_len
        self.d_model = d_model
        self.gate_min_scale = gate_min_scale
        self.gate_reg_weight = gate_reg_weight

        self._last_gate_penalty = None
        self._last_gate_mean = None
        self._last_gate_tensor = None

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

        self.token_input_dim = token_input_dim

        # Raw feature gate
        if use_gate_mlp:
            hidden = gate_hidden_dim or token_input_dim
            self.feature_gate = nn.Sequential(
                nn.LayerNorm(token_input_dim),
                nn.Linear(token_input_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, token_input_dim),
            )
        else:
            self.feature_gate = nn.Sequential(
                nn.LayerNorm(token_input_dim),
                nn.Linear(token_input_dim, token_input_dim),
            )

        # Transaction encoder
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

        self._init_non_pretrained_weights()

    def _init_non_pretrained_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)

    def _apply_feature_gate(self, x_token: torch.Tensor) -> torch.Tensor:
        """
        x_token: [B, S, F_token]
        returns gated_x_token: [B, S, F_token]
        """
        raw_gate_logits = self.feature_gate(x_token)
        gate = torch.sigmoid(raw_gate_logits)

        # keep a floor so the model cannot zero everything out too aggressively
        gate_scale = self.gate_min_scale + (1.0 - self.gate_min_scale) * gate

        gated_x = x_token * gate_scale

        # regularizer: encourage smaller gates on average
        # penalty is based on gate_scale above the floor
        gate_penalty = gate.mean()

        self._last_gate_penalty = gate_penalty * self.gate_reg_weight
        self._last_gate_mean = gate_scale.mean().detach()
        self._last_gate_tensor = gate_scale.detach()

        return gated_x

    def get_aux_loss(self) -> torch.Tensor:
        if self._last_gate_penalty is None:
            return torch.tensor(0.0, device=self.position_embedding.device)
        return self._last_gate_penalty

    def get_gate_stats(self) -> dict[str, float]:
        if self._last_gate_mean is None:
            return {
                "gate_mean": float("nan"),
            }
        return {
            "gate_mean": float(self._last_gate_mean.item()),
        }

    def get_last_gate_tensor(self) -> torch.Tensor | None:
        return self._last_gate_tensor

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
            aux = self.anomaly_adapter(x)            # [B, S, aux_dim]
            x_token = torch.cat([x, aux], dim=-1)   # [B, S, F + aux_dim]
        else:
            x_token = x

        x_token = self._apply_feature_gate(x_token)

        z = self.transaction_encoder(x_token)
        z = z + self.position_embedding

        key_padding_mask = ~attention_mask.bool()
        h = self.sequence_encoder(z, src_key_padding_mask=key_padding_mask)

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