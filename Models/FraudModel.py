import torch
import torch.nn as nn

# B: batch size (TBD), S: seq len (8), F: feature dim (112)
# Our dataset will give us tensors of shape:
# x: (B, S, F) -- B number of Seqences of len S(8) each containing feature dimension of F(112)
# attention mask:  (B, S) -- B number of boolean masks of length S (8)
# y: (B) -- B number of a single label

# Model architechture
#
# [ENCODER AND EMBEDDING LAYER]
# We effectively want a small trasnsaction encoder inside the model (with a positional encoder)
# x:(B, S, F) => encoder:(B, S, d_model)
#
# [Visualisation]
#                   x:(B, S, F)
#                         |
#     base_encoding:(B, S, d_model) + positional_embedding:(1, S, d_model)
#                         |
#          Intermediate_embedding:(B, S, d_model)

# [SEQUENCING MODEL LAYER] (continuing from the embedding layer)
# Small transformer encoder:
# - num_layers = 2
# - nhead = 4
# - d_model = 64
# - dim_feedforward = 128 or 256
# - dropout = 0.1
#
# intermediate_embedding: (B, 8, 64)
#   -> transformer_encoder
#   -> transformer_output: (B, 8, 64)

# Since we use LEFT padding, the current transaction is always at the last position.
# So we take:
# last_hidden = transformer_output[:, -1, :]
# shape: (B, 64)

# [CLASSIFICATION HEAD] (we can modify this to along gating of the previous n number of outputs as they may be useful)
# last_hidden: (B, 64)
#   -> LayerNorm(64)
#   -> Linear(64, 64)
#   -> GELU
#   -> Dropout(0.1)
#   -> Linear(64, 1)
#   -> logits: (B, 1)
#   -> squeeze(-1)
#   -> logits: (B,)


class TransformerFraudModel(nn.Module):
    def __init__(
        self,
        feature_dim: int = 112,
        seq_len: int = 8,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        use_second_projection: bool = False,
    ):
        super().__init__()

        if d_model % nhead != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by nhead ({nhead})"
            ) # reject if we cannot even split the model dimensions among n heads

        self.feature_dim = feature_dim
        self.seq_len = seq_len
        self.d_model = d_model

        # Transaction encoder
        encoder_layers = [
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, d_model),
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

        # Learned positional embedding
        # Shape conceptually: (1, S, d_model)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, seq_len, d_model)
        )

        # Small initialization we can play around with different ones
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)

        # Transformer encoder
        # batch_first=True means inputs are (B, S, d_model)
        # src_key_padding_mask expects True = ignore
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

        # Classification head
        # We classify from the last token because with LEFT padding,
        # the current transaction is always at position -1.
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

        # 1) Transaction encoding
        # x -> (B, S, d_model)
        z = self.transaction_encoder(x)

        # 2) Add learned positional embedding
        z = z + self.position_embedding

        # 3) Transformer expects padding mask with True = ignore
        key_padding_mask = ~attention_mask

        # 4) Sequence encoding
        h = self.sequence_encoder(z, src_key_padding_mask=key_padding_mask,)  # (B, S, d_model)

        # 5) Current transaction is always the last token due to LEFT padding
        h_last = h[:, -1, :]  # (B, d_model)

        # 6) Classification head
        logits = self.classifier(h_last).squeeze(-1)  # (B,)

        return logits
    

import torch
import torch.nn as nn

# Baseline NN
class LastTokenMLP(nn.Module):

    def __init__(
        self,
        feature_dim: int = 112,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"x must have shape (B, S, F), got {tuple(x.shape)}")

        # Because of LEFT padding, current transaction is always last token
        x_last = x[:, -1, :]   # (B, F)
        logits = self.net(x_last).squeeze(-1)  # (B,)
        return logits