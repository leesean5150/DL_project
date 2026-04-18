from pathlib import Path
from typing import Sequence, Type
import joblib
import torch
import torch.nn as nn

project_root = Path(__file__).resolve().parents[1]

from Models.AutoEncoder import AutoEncoder, EnsembleVAE
from Models.EnsembleModel import (
    StackedFraudEnsemble,
    StackingFraudMLP,
    FeatureRichStackingFraudMLP,
    TorchBinaryProbWrapper,
    SklearnProbWrapper,
    ProbModelWrapper,
)
from Models.FraudModel import FraudMLP


def _load_checkpoint(path, device):
    return torch.load(path, map_location=device)


def load_autoencoder(path, device):
    ckpt = _load_checkpoint(path, device)

    model = AutoEncoder(
        input_dim=ckpt.get("input_dim", 776),
        latent_dim=ckpt.get("latent_dim", 16),
        hidden_dims=ckpt.get("hidden_dims", [128, 64]),
        noise_std=ckpt.get("noise_std", 0.0),
    )
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)

    model.to(device)
    model.eval()
    return model


def load_vae(path, device):
    ckpt = _load_checkpoint(path, device)

    model = EnsembleVAE(
        input_dim=ckpt["input_dim"],
        latent_dim=ckpt["latent_dim"],
        hidden_dims=ckpt["hidden_dims"],
        noise_std=ckpt.get("noise_std", 0.0),
        use_norm=ckpt.get("use_norm", False),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model


def load_torch_model(path, model_cls: Type[nn.Module], device):
    ckpt = _load_checkpoint(path, device)

    model_config = ckpt.get("model_config")
    if model_config is None:
        raise KeyError(f"No 'model_config' found in checkpoint: {path}")

    model = model_cls(**model_config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def load_fraud_mlp(path, device, encoder):
    ckpt = _load_checkpoint(path, device)

    model_config = ckpt.get("model_config")
    if model_config is None:
        raise KeyError(f"No 'model_config' found in checkpoint: {path}")

    model = FraudMLP(
        input_dim=model_config["input_dim"],
        hidden_dims=model_config["hidden_dims"],
        gated=model_config["gated"],
        dropout=model_config["dropout"],
        use_norm=model_config["use_norm"],
        encoder=encoder,
        freeze_encoder=model_config["freeze_encoder"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    return model


def load_stacking_mlp(path, device):
    return load_torch_model(path, StackingFraudMLP, device)


def load_feature_rich_stacking_mlp(path, device):
    return load_torch_model(path, FeatureRichStackingFraudMLP, device)


def load_prob_torch_wrapper(path, model_cls: Type[nn.Module], device):
    model = load_torch_model(path, model_cls, device)
    return TorchBinaryProbWrapper(model)


def load_fraud_mlp_wrapper(path, device):
    model = load_fraud_mlp(path, device)
    return TorchBinaryProbWrapper(model)


def load_sklearn_model(path):
    return joblib.load(path)


def load_sklearn_wrapper(path):
    model = load_sklearn_model(path)
    return SklearnProbWrapper(model)

def load_stacked_ensemble(path, members: Sequence[ProbModelWrapper], device):
    ckpt = torch.load(path, map_location=device)
    stacker = StackingFraudMLP(**ckpt["stacker_config"])
    stacker.load_state_dict(ckpt["stacker_state_dict"])
    model = StackedFraudEnsemble(
        members=members,
        stacker=stacker,
    )
    model.to(device)
    model.eval()
    return model