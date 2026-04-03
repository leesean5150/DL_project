# ChallengeChampionModel.py

from __future__ import annotations

import inspect
import json
import shutil
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_fscore_support,
    accuracy_score,
    fbeta_score,
)

from Models.FraudModel import (
    TransformerFraudModel,
    GatedTransformerFraudModel,
    AEAttentionPoolingTransformerFraudModel,
    LastTokenMLP,
)
from Utils.TrainUtils import build_dataloaders, get_device


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

DEFAULT_THRESHOLDS = np.linspace(0.01, 0.99, 100)

LENGTH_BUCKETS = {
    "len=1": lambda s: s == 1,
    "len=2": lambda s: s == 2,
    "len=3-4": lambda s: (s >= 3) & (s <= 4),
    "len=5-8": lambda s: (s >= 5) & (s <= 8),
    "len>=9": lambda s: s >= 9,
}


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def compute_metrics_from_logits(
    y_true: np.ndarray,
    logits: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(np.int64)
    logits = np.asarray(logits, dtype=np.float32)

    probs = sigmoid_np(logits)
    preds = (probs >= threshold).astype(np.int64)

    metrics: dict[str, float] = {}

    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, probs))
    except Exception:
        metrics["roc_auc"] = float("nan")

    try:
        metrics["pr_auc"] = float(average_precision_score(y_true, probs))
    except Exception:
        metrics["pr_auc"] = float("nan")

    metrics["accuracy"] = float(accuracy_score(y_true, preds))

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        preds,
        average="binary",
        zero_division=0,
    )

    metrics["precision"] = float(precision)
    metrics["recall"] = float(recall)
    metrics["f1"] = float(f1)
    metrics["f2"] = float(
        fbeta_score(y_true, preds, beta=2.0, average="binary", zero_division=0)
    )
    metrics["n"] = int(len(y_true))
    metrics["fraud_rate"] = float(y_true.mean())

    return metrics


def threshold_sweep_from_logits(
    y_true: np.ndarray,
    logits: np.ndarray,
    thresholds: np.ndarray = DEFAULT_THRESHOLDS,
) -> list[dict[str, float]]:
    probs = sigmoid_np(np.asarray(logits, dtype=np.float32))
    y_true = np.asarray(y_true).astype(np.int64)

    results = []

    for t in thresholds:
        preds = (probs >= t).astype(np.int64)

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, preds, average="binary", zero_division=0
        )
        f2 = fbeta_score(
            y_true, preds, beta=2.0, average="binary", zero_division=0
        )

        results.append({
            "threshold": float(t),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "f2": float(f2),
        })

    return results


def get_best_by_metric(
    sweep_results: list[dict[str, float]],
    metric: str = "f2",
) -> dict[str, float]:
    return max(sweep_results, key=lambda x: x[metric])


# ---------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------

def build_model_from_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    extra = ckpt.get("extra", {})

    model_class = extra.get("model_class")
    if model_class is None:
        raise ValueError(f"No model_class found in checkpoint extra for {checkpoint_path}")

    feature_dim = extra["feature_dim"]
    dropout = extra.get("dropout", 0.1)

    if model_class == "TransformerFraudModel":
        candidate_kwargs = {
            "feature_dim": feature_dim,
            "seq_len": extra.get("seq_len", 8),
            "d_model": extra.get("d_model", 64),
            "nhead": extra.get("nhead", 4),
            "num_layers": extra.get("num_layers", 3),
            "dim_feedforward": extra.get("dim_feedforward", 128),
            "dropout": dropout,
            "anomaly_checkpoint_path": extra.get("anomaly_checkpoint_path"),
            "freeze_anomaly_encoder": extra.get("freeze_anomaly_encoder", True),
            "include_ae_latent": extra.get("include_ae_latent", True),
            "include_ae_recon_error": extra.get("include_ae_recon_error", True),
            "include_ae_gates": extra.get("include_ae_gates", False),
        }
        cls = TransformerFraudModel

    elif model_class == "AEAttentionPoolingTransformerFraudModel":
        candidate_kwargs = {
            "feature_dim": feature_dim,
            "seq_len": extra.get("seq_len", 8),
            "d_model": extra.get("d_model", 64),
            "nhead": extra.get("nhead", 4),
            "num_layers": extra.get("num_layers", 3),
            "dim_feedforward": extra.get("dim_feedforward", 128),
            "dropout": dropout,
            "use_second_projection": extra.get("use_second_projection", False),

            "anomaly_checkpoint_path": extra.get("anomaly_checkpoint_path"),
            "freeze_anomaly_encoder": extra.get("freeze_anomaly_encoder", True),
            "include_ae_latent": extra.get("include_ae_latent", True),
            "include_ae_recon_error": extra.get("include_ae_recon_error", True),
            "include_ae_gates": extra.get("include_ae_gates", False),

            "use_ae_token_residual": extra.get("use_ae_token_residual", True),
            "use_ae_head_skip": extra.get("use_ae_head_skip", True),
            "ae_hidden_dim": extra.get("ae_hidden_dim"),
            "init_ae_residual_scale": extra.get("init_ae_residual_scale", 0.10),
        }
        cls = AEAttentionPoolingTransformerFraudModel

    elif model_class == "GatedTransformerFraudModel":
        candidate_kwargs = {
            "feature_dim": feature_dim,
            "seq_len": extra.get("seq_len", 8),
            "d_model": extra.get("d_model", 64),
            "nhead": extra.get("nhead", 4),
            "num_layers": extra.get("num_layers", 3),
            "dim_feedforward": extra.get("dim_feedforward", 128),
            "dropout": dropout,
            "use_second_projection": extra.get("use_second_projection", False),

            "gate_hidden_dim": extra.get("gate_hidden_dim"),
            "gate_min_scale": extra.get("gate_min_scale", 0.10),
            "gate_reg_weight": extra.get("gate_reg_weight", 1e-3),
            "use_gate_mlp": extra.get("use_gate_mlp", False),

            "anomaly_checkpoint_path": extra.get("anomaly_checkpoint_path"),
            "freeze_anomaly_encoder": extra.get("freeze_anomaly_encoder", True),
            "include_ae_latent": extra.get("include_ae_latent", True),
            "include_ae_recon_error": extra.get("include_ae_recon_error", True),
            "include_ae_gates": extra.get("include_ae_gates", False),
        }
        cls = GatedTransformerFraudModel

    elif model_class == "LastTokenMLP":
        candidate_kwargs = {
            "feature_dim": feature_dim,
            "hidden_dim": extra.get("hidden_dim", 64),
            "dropout": dropout,
            "anomaly_checkpoint_path": extra.get("anomaly_checkpoint_path"),
            "freeze_anomaly_encoder": extra.get("freeze_anomaly_encoder", True),
            "include_ae_latent": extra.get("include_ae_latent", True),
            "include_ae_recon_error": extra.get("include_ae_recon_error", True),
            "include_ae_gates": extra.get("include_ae_gates", False),
        }
        cls = LastTokenMLP

    else:
        raise ValueError(f"Unsupported model_class: {model_class}")

    sig = inspect.signature(cls.__init__)
    filtered_kwargs = {k: v for k, v in candidate_kwargs.items() if k in sig.parameters}

    model = cls(**filtered_kwargs).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model, ckpt, extra


# ---------------------------------------------------------------------
# Prediction collection
# ---------------------------------------------------------------------

@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()

    all_logits = []
    all_labels = []
    all_seq_lens = []

    cursor = 0
    seq_lengths = loader.dataset.seq_lengths

    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        y = batch["y"].cpu().numpy()

        logits = model(x, attention_mask).detach().cpu().numpy()

        batch_size = y.shape[0] if np.ndim(y) > 0 else 1
        batch_seq_lens = seq_lengths[cursor:cursor + batch_size]
        cursor += batch_size

        all_logits.append(logits)
        all_labels.append(y)
        all_seq_lens.extend([int(s) for s in batch_seq_lens])

    all_logits = np.concatenate(all_logits)
    all_labels = np.concatenate(all_labels)
    all_seq_lens = np.asarray(all_seq_lens, dtype=np.int64)

    return {
        "logits": all_logits,
        "labels": all_labels,
        "seq_lens": all_seq_lens,
    }


def evaluate_prediction_dict(pred_dict, threshold=0.5):
    return compute_metrics_from_logits(
        y_true=pred_dict["labels"],
        logits=pred_dict["logits"],
        threshold=threshold,
    )


def evaluate_by_buckets(pred_dict, buckets=LENGTH_BUCKETS, threshold=0.5):
    labels = pred_dict["labels"]
    logits = pred_dict["logits"]
    seq_lens = pred_dict["seq_lens"]

    results = {}

    for name, fn in buckets.items():
        mask = fn(seq_lens)
        if mask.sum() == 0:
            continue

        results[name] = compute_metrics_from_logits(
            y_true=labels[mask],
            logits=logits[mask],
            threshold=threshold,
        )

    return results


# ---------------------------------------------------------------------
# Champion registry
# ---------------------------------------------------------------------

@dataclass
class ChampionRecord:
    promoted_at_utc: str
    checkpoint_path: str
    model_class: str
    source_run_dir: str | None
    compare_metric: str
    compare_value: float
    test_threshold: float
    overall_metrics: dict[str, float]
    bucket_metrics: dict[str, dict[str, float]]
    extra: dict[str, Any]


def load_registry(registry_path: str | Path) -> dict[str, Any]:
    registry_path = Path(registry_path)
    if not registry_path.exists():
        return {
            "current_champion": None,
            "previous_champions": [],
        }

    with registry_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_registry(registry_path: str | Path, registry: dict[str, Any]) -> None:
    registry_path = Path(registry_path)
    registry_path.parent.mkdir(parents=True, exist_ok=True)

    with registry_path.open("w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)


def promote_candidate_if_better(
    candidate_record: ChampionRecord,
    registry_path: str | Path,
    min_improvement: float = 0.003,
) -> tuple[bool, dict[str, Any]]:
    registry = load_registry(registry_path)
    current = registry.get("current_champion")

    if current is None:
        registry["current_champion"] = asdict(candidate_record)
        save_registry(registry_path, registry)
        return True, registry

    current_metric = current["compare_value"]
    candidate_metric = candidate_record.compare_value

    if candidate_metric >= current_metric + min_improvement:
        registry["previous_champions"].append(current)
        registry["current_champion"] = asdict(candidate_record)
        save_registry(registry_path, registry)
        return True, registry

    return False, registry


# ---------------------------------------------------------------------
# Main evaluation pipeline
# ---------------------------------------------------------------------

def evaluate_checkpoint_on_test(
    checkpoint_path: str | Path,
    test_loader,
    device,
    metric_for_threshold_selection: str = "f2",
):
    model, ckpt, extra = build_model_from_checkpoint(checkpoint_path, device)
    preds = collect_predictions(model, test_loader, device)

    sweep = threshold_sweep_from_logits(
        y_true=preds["labels"],
        logits=preds["logits"],
        thresholds=DEFAULT_THRESHOLDS,
    )
    best = get_best_by_metric(sweep, metric=metric_for_threshold_selection)
    best_threshold = best["threshold"]

    overall = evaluate_prediction_dict(preds, threshold=best_threshold)
    buckets = evaluate_by_buckets(preds, threshold=best_threshold)

    return {
        "model": model,
        "checkpoint": ckpt,
        "extra": extra,
        "preds": preds,
        "threshold_sweep": sweep,
        "best_threshold_result": best,
        "best_threshold": best_threshold,
        "overall_metrics": overall,
        "bucket_metrics": buckets,
    }


def make_candidate_record(
    checkpoint_path: str | Path,
    evaluation: dict[str, Any],
    compare_metric: str = "f2",
) -> ChampionRecord:
    extra = evaluation["extra"]
    overall = evaluation["overall_metrics"]
    threshold = evaluation["best_threshold"]

    return ChampionRecord(
        promoted_at_utc=datetime.now(timezone.utc).isoformat(),
        checkpoint_path=str(checkpoint_path),
        model_class=extra.get("model_class", "unknown"),
        source_run_dir=str(Path(checkpoint_path).parent),
        compare_metric=compare_metric,
        compare_value=float(overall[compare_metric]),
        test_threshold=float(threshold),
        overall_metrics=overall,
        bucket_metrics=evaluation["bucket_metrics"],
        extra=extra,
    )


def print_summary(title: str, metrics: dict[str, float]) -> None:
    print(f"\n{title}")
    for k in ["n", "fraud_rate", "pr_auc", "roc_auc", "f2", "f1", "precision", "recall", "accuracy"]:
        v = metrics.get(k)
        if v is None:
            continue
        if isinstance(v, float):
            print(f"{k:>10s}: {v:.4f}")
        else:
            print(f"{k:>10s}: {v}")


# ---------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------

if __name__ == "__main__":
    device = get_device()
    print("Device:", device)

    train_loader, val_loader, test_loader = build_dataloaders(
        processed_dir="data/processed_fraud_normalized",
        seq_len=8,
        batch_size=128,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    candidate_checkpoint = "checkpoints/Transformer_AE_V5/best.pt"
    registry_path = "checkpoints/champion_registry.json"

    evaluation = evaluate_checkpoint_on_test(
        checkpoint_path=candidate_checkpoint,
        test_loader=test_loader,
        device=device,
        metric_for_threshold_selection="f2",
    )

    print_summary("Candidate overall metrics", evaluation["overall_metrics"])
    print("\nBest threshold result:", evaluation["best_threshold_result"])

    candidate_record = make_candidate_record(
        checkpoint_path=candidate_checkpoint,
        evaluation=evaluation,
        compare_metric="f2",
    )

    promoted, registry = promote_candidate_if_better(
        candidate_record=candidate_record,
        registry_path=registry_path,
        min_improvement=0.003,
    )

    print("\nPromoted to champion:", promoted)
    current = registry["current_champion"]
    print("Current champion model:", current["model_class"])
    print("Current champion path:", current["checkpoint_path"])
    print("Current champion compare value:", current["compare_value"])