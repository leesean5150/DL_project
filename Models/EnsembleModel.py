import torch
import torch.nn as nn
from typing import List, Optional, Sequence


class ProbModelWrapper(nn.Module):
    """
    Base wrapper contract:
        x -> probability in [0, 1], shape [B]
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class TorchBinaryProbWrapper(ProbModelWrapper):
    """
    Wraps a PyTorch binary classifier that returns logits of shape [B] or [B, 1].
    Converts logits to probabilities with sigmoid.
    """
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        logits = self.model(x)
        if logits.ndim > 1:
            logits = logits.squeeze(-1)
        probs = torch.sigmoid(logits)
        return probs


class SklearnProbWrapper(ProbModelWrapper):
    """
    Optional wrapper for sklearn-like models with predict_proba().
    Keeps the same x -> probability API.

    Notes:
    - Expects x to be a torch tensor on any device.
    - Converts to CPU numpy internally.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_np = x.detach().cpu().numpy()
        probs_np = self.model.predict_proba(x_np)[:, 1]
        probs = torch.from_numpy(probs_np).to(x.device, dtype=torch.float32)
        return probs


class WeightedFraudEnsemble(nn.Module):
    """
    Simple weighted probability mixer.

    Input:
        x: base feature tensor [B, F]

    Output:
        blended fraud probability [B]
    """
    def __init__(
        self,
        members: Sequence[ProbModelWrapper],
        weights: Optional[Sequence[float]] = None,
    ):
        super().__init__()

        if len(members) == 0:
            raise ValueError("members must contain at least one model")

        self.members = nn.ModuleList(members)

        if weights is None:
            weights = [1.0 / len(members)] * len(members)

        if len(weights) != len(members):
            raise ValueError("weights length must match number of members")

        weight_tensor = torch.tensor(weights, dtype=torch.float32)
        if torch.any(weight_tensor < 0):
            raise ValueError("weights must be non-negative")

        total = weight_tensor.sum().item()
        if total <= 0:
            raise ValueError("weights must sum to a positive value")

        weight_tensor = weight_tensor / total
        self.register_buffer("weights", weight_tensor)

        self.config = {
            "num_members": len(members),
            "weights": weight_tensor.tolist(),
        }

    @torch.no_grad()
    def member_probs(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns:
            probs: [B, M]
        """
        probs = [member(x) for member in self.members]
        probs = torch.stack(probs, dim=1)
        return probs

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        probs = self.member_probs(x)  # [B, M]
        blended = (probs * self.weights.unsqueeze(0)).sum(dim=1)
        return blended


class StackingFraudMLP(nn.Module):
    """
    Small MLP that takes member probabilities as meta-features.

    Input:
        meta-features [B, M]

    Output:
        final logit [B]
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
        use_norm: bool = True,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [16, 8]

        layers: List[nn.Module] = []
        prev_dim = input_dim

        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.GELU())
            if use_norm:
                layers.append(nn.LayerNorm(h))
            layers.append(nn.Dropout(dropout))
            prev_dim = h

        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(prev_dim, 1)

        self.config = {
            "input_dim": input_dim,
            "hidden_dims": list(hidden_dims),
            "dropout": dropout,
            "use_norm": use_norm,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        logits = self.head(x)
        return logits.squeeze(-1)


class StackedFraudEnsemble(nn.Module):
    """
    End-to-end ensemble:
        base input x -> member probabilities -> stacker -> final probability

    This is useful for inference after the stacker has already been trained.
    """
    def __init__(
        self,
        members: Sequence[ProbModelWrapper],
        stacker: nn.Module,
    ):
        super().__init__()

        if len(members) == 0:
            raise ValueError("members must contain at least one model")

        self.members = nn.ModuleList(members)
        self.stacker = stacker

        self.config = {
            "num_members": len(members),
            "stacker_class": stacker.__class__.__name__,
            "stacker_config": getattr(stacker, "config", None),
        }

    @torch.no_grad()
    def member_probs(self, x: torch.Tensor) -> torch.Tensor:
        probs = [member(x) for member in self.members]
        probs = torch.stack(probs, dim=1)  # [B, M]
        return probs

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        meta_x = self.member_probs(x)
        logits = self.stacker(meta_x)
        return logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward_logits(x)
        probs = torch.sigmoid(logits)
        return probs


@torch.no_grad()
def collect_member_probs(
    members: Sequence[ProbModelWrapper],
    loader,
    device: torch.device,
) -> torch.Tensor:
    """
    Collect member probabilities for an entire loader.

    Returns:
        meta_x: [N, M]
    """
    all_meta = []

    for x, _ in loader:
        x = x.to(device)
        probs = [member(x) for member in members]
        probs = torch.stack(probs, dim=1)  # [B, M]
        all_meta.append(probs.cpu())

    return torch.cat(all_meta, dim=0)


@torch.no_grad()
def collect_labels(
    loader,
) -> torch.Tensor:
    ys = []
    for _, y in loader:
        ys.append(y.float().cpu())
    return torch.cat(ys, dim=0)


@torch.no_grad()
def collect_weighted_ensemble_probs(
    ensemble: WeightedFraudEnsemble,
    loader,
    device: torch.device,
) -> torch.Tensor:
    all_probs = []

    ensemble.eval()
    for x, _ in loader:
        x = x.to(device)
        probs = ensemble(x)
        all_probs.append(probs.cpu())

    return torch.cat(all_probs, dim=0)


@torch.no_grad()
def collect_stacked_ensemble_probs(
    ensemble: StackedFraudEnsemble,
    loader,
    device: torch.device,
) -> torch.Tensor:
    all_probs = []

    ensemble.eval()
    for x, _ in loader:
        x = x.to(device)
        probs = ensemble(x)
        all_probs.append(probs.cpu())

    return torch.cat(all_probs, dim=0)

class CatBoostAEWrapper(ProbModelWrapper):
    def __init__(self, cat_model, autoencoder):
        super().__init__()
        self.cat_model = cat_model
        self.autoencoder = autoencoder
        self.autoencoder.eval()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device

        # replicate extract_features()
        z = self.autoencoder.encoder(x)
        x_hat = self.autoencoder.decoder(z)

        diff = x - x_hat
        abs_err = torch.abs(diff)

        mse = (diff ** 2).mean(dim=1, keepdim=True)
        max_err = abs_err.max(dim=1, keepdim=True)[0]
        std_err = abs_err.std(dim=1, keepdim=True)
        l1_err = abs_err.mean(dim=1, keepdim=True)
        topk_err = torch.topk(abs_err, k=5, dim=1)[0].mean(dim=1, keepdim=True)

        features = torch.cat([
            x,
            z,
            mse,
            max_err,
            std_err,
            l1_err,
            topk_err
        ], dim=1)

        feats_np = features.detach().cpu().numpy()

        probs_np = self.cat_model.predict_proba(feats_np)[:, 1]
        probs = torch.from_numpy(probs_np).to(device).float()

        return probs