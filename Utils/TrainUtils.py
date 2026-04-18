from __future__ import annotations


from typing import Dict, Iterable, Optional, Tuple, List, Any
from pathlib import Path
import json
import time
import torch


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_confusion_counts(
    probs: torch.Tensor,
    labels: torch.Tensor,
    threshold: float = 0.5,
) -> Dict[str, int]:
    
    preds = (probs >= threshold).to(torch.int64)
    labels = labels.to(torch.int64)

    tp = ((preds == 1) & (labels == 1)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    tn = ((preds == 0) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def compute_precision_recall(
    tp: int,
    fp: int,
    fn: int,
    eps: float = 1e-8,
) -> Tuple[float, float]:
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    return precision, recall


def compute_f1(
    tp: int,
    fp: int,
    fn: int,
    eps: float = 1e-8,
) -> float:
    precision, recall = compute_precision_recall(tp, fp, fn, eps=eps)
    return (2.0 * precision * recall) / (precision + recall + eps)


def compute_f2(
    tp: int,
    fp: int,
    fn: int,
    eps: float = 1e-8,
) -> float:
    precision, recall = compute_precision_recall(tp, fp, fn, eps=eps)
    return (5.0 * precision * recall) / (4.0 * precision + recall + eps)


def compute_accuracy(
    tp: int,
    fp: int,
    tn: int,
    fn: int,
    eps: float = 1e-8,
) -> float:
    return (tp + tn) / (tp + fp + tn + fn + eps)


def compute_metrics_from_probs(
    probs: torch.Tensor,
    labels: torch.Tensor,
    threshold: float = 0.5,
) -> Dict[str, float]:
    cm = compute_confusion_counts(probs, labels, threshold=threshold)

    precision, recall = compute_precision_recall(
        tp=cm["tp"],
        fp=cm["fp"],
        fn=cm["fn"],
    )

    f1 = compute_f1(
        tp=cm["tp"],
        fp=cm["fp"],
        fn=cm["fn"],
    )

    f2 = compute_f2(
        tp=cm["tp"],
        fp=cm["fp"],
        fn=cm["fn"],
    )

    accuracy = compute_accuracy(
        tp=cm["tp"],
        fp=cm["fp"],
        tn=cm["tn"],
        fn=cm["fn"],
    )

    return {
        "threshold": threshold,
        "tp": cm["tp"],
        "fp": cm["fp"],
        "tn": cm["tn"],
        "fn": cm["fn"],
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f2": f2,
        "accuracy": accuracy,
    }


def generate_thresholds(
    probs: torch.Tensor,
    num_thresholds: int = 200,
    min_threshold: Optional[float] = None,
    max_threshold: Optional[float] = None,
) -> torch.Tensor:
    """
    Generate candidate thresholds from predicted probabilities.
    Uses percentiles to focus on the model's actual score distribution.
    """
    probs = probs.detach().float().cpu()

    if min_threshold is None:
        min_threshold = max(0.001, float(torch.quantile(probs, 0.01)))
    if max_threshold is None:
        max_threshold = min(0.999, float(torch.quantile(probs, 0.999)))

    if min_threshold >= max_threshold:
        min_threshold, max_threshold = 0.001, 0.999

    thresholds = torch.linspace(min_threshold, max_threshold, steps=num_thresholds)
    return thresholds


def threshold_sweep(
    probs: torch.Tensor,
    labels: torch.Tensor,
    thresholds: Optional[Iterable[float]] = None,
    num_thresholds: int = 200,
    optimize_for: str = "f2",
) -> Tuple[Dict[str, float], list[Dict[str, float]]]:
    """
    Sweep thresholds and return:
      - best_metrics
      - all_metrics
    """
    valid_targets = {"f1", "f2", "accuracy"}
    if optimize_for not in valid_targets:
        raise ValueError(f"optimize_for must be one of {valid_targets}, got {optimize_for!r}")

    probs = probs.detach().float().cpu()
    labels = labels.detach().float().cpu()

    if thresholds is None:
        thresholds = generate_thresholds(probs, num_thresholds=num_thresholds).tolist()

    all_metrics: list[Dict[str, float]] = []
    best_metrics: Optional[Dict[str, float]] = None
    best_score = -1.0

    for threshold in thresholds:
        metrics = compute_metrics_from_probs(
            probs=probs,
            labels=labels,
            threshold=float(threshold),
        )
        all_metrics.append(metrics)

        score = metrics[optimize_for]
        if score > best_score:
            best_score = score
            best_metrics = metrics

    if best_metrics is None:
        raise RuntimeError("Threshold sweep failed to produce any metrics.")

    return best_metrics, all_metrics



def train_one_epoch(
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device).float()

        optimizer.zero_grad()

        logits = model(x)
        loss = criterion(logits, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def collect_probs_and_labels(
    model: torch.nn.Module,
    loader,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()

    all_probs = []
    all_labels = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device).float()

        logits = model(x)
        probs = torch.sigmoid(logits)

        all_probs.append(probs.cpu())
        all_labels.append(y.cpu())

    probs = torch.cat(all_probs, dim=0)
    labels = torch.cat(all_labels, dim=0)

    return probs, labels


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader,
    criterion: torch.nn.Module,
    device: torch.device,
    threshold: float = 0.5,
    sweep_thresholds: bool = True,
    num_thresholds: int = 200,
    optimize_for: str = "f2",
) -> Dict[str, float]:
    model.eval()

    total_loss = 0.0
    all_probs = []
    all_labels = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device).float()

        logits = model(x)
        loss = criterion(logits, y)
        probs = torch.sigmoid(logits)

        total_loss += loss.item()
        all_probs.append(probs.cpu())
        all_labels.append(y.cpu())

    probs = torch.cat(all_probs, dim=0)
    labels = torch.cat(all_labels, dim=0)

    base_metrics = compute_metrics_from_probs(
        probs=probs,
        labels=labels,
        threshold=threshold,
    )

    metrics = {
        "loss": total_loss / max(len(loader), 1),
        **base_metrics,
    }

    if sweep_thresholds:
        best_metrics, _ = threshold_sweep(
            probs=probs,
            labels=labels,
            num_thresholds=num_thresholds,
            optimize_for=optimize_for,
        )

        metrics.update({
            "best_threshold": best_metrics["threshold"],
            "best_precision": best_metrics["precision"],
            "best_recall": best_metrics["recall"],
            "best_f1": best_metrics["f1"],
            "best_f2": best_metrics["f2"],
            "best_accuracy": best_metrics["accuracy"],
            "best_tp": best_metrics["tp"],
            "best_fp": best_metrics["fp"],
            "best_tn": best_metrics["tn"],
            "best_fn": best_metrics["fn"],
        })

    return metrics


def _to_jsonable(obj):
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            return obj.item()
        return obj.detach().cpu().tolist()
    return obj


def _save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(data), f, indent=2)


def _check_overfitting(
    history: List[Dict[str, Any]],
    patience: int = 3,
    metric_name: str = "best_f2",
) -> Optional[str]:
    """
    Heuristic warning only, not a proof of overfitting.
    """
    if len(history) < patience + 1:
        return None

    recent = history[-(patience + 1):]

    train_losses = [x["train_loss"] for x in recent]
    val_losses = [x["val_loss"] for x in recent]
    metric_values = [x[metric_name] for x in recent]

    train_loss_down = all(train_losses[i] >= train_losses[i + 1] for i in range(len(train_losses) - 1))
    val_loss_up = all(val_losses[i] <= val_losses[i + 1] for i in range(len(val_losses) - 1))
    metric_not_improving = all(metric_values[i] >= metric_values[i + 1] for i in range(len(metric_values) - 1))

    if train_loss_down and val_loss_up:
        return (
            "[WARN] Possible overfitting detected: train loss is decreasing while "
            "validation loss is increasing over recent epochs."
        )

    if train_loss_down and metric_not_improving:
        return (
            f"[WARN] Possible overfitting detected: train loss is decreasing while "
            f"validation {metric_name} is not improving over recent epochs."
        )

    return None


def train_model(
    model: torch.nn.Module,
    train_loader,
    val_loader,
    epochs: int,
    lr: float,
    device: torch.device,
    pos_weight: Optional[float] = None,
    save_path: str = "default_run",
    optimize_for: str = "f2",
    num_thresholds: int = 200,
    checkpoint_root: str = "checkpoints",
    overfit_patience: int = 3,
) -> torch.nn.Module:
    model = model.to(device)

    if pos_weight is not None:
        criterion = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(pos_weight, dtype=torch.float32, device=device)
        )
    else:
        criterion = torch.nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.001)

    run_dir = Path(checkpoint_root) / save_path
    run_dir.mkdir(parents=True, exist_ok=True)

    best_model_path = run_dir / "best.pt"
    history_path = run_dir / "history.json"
    summary_path = run_dir / "summary.json"

    history: List[Dict[str, Any]] = []

    best_score = -1.0
    best_epoch = -1
    best_val_metrics: Optional[Dict[str, Any]] = None

    for epoch in range(epochs):
        epoch_start = time.time()

        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            threshold=0.5,
            sweep_thresholds=True,
            num_thresholds=num_thresholds,
            optimize_for=optimize_for,
        )

        epoch_time = time.time() - epoch_start
        score_to_track = val_metrics[f"best_{optimize_for}"]

        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_accuracy_at_0_5": val_metrics["accuracy"],
            "val_precision_at_0_5": val_metrics["precision"],
            "val_recall_at_0_5": val_metrics["recall"],
            "val_f1_at_0_5": val_metrics["f1"],
            "val_f2_at_0_5": val_metrics["f2"],
            "best_threshold": val_metrics["best_threshold"],
            "best_accuracy": val_metrics["best_accuracy"],
            "best_precision": val_metrics["best_precision"],
            "best_recall": val_metrics["best_recall"],
            "best_f1": val_metrics["best_f1"],
            "best_f2": val_metrics["best_f2"],
            "best_tp": val_metrics["best_tp"],
            "best_fp": val_metrics["best_fp"],
            "best_tn": val_metrics["best_tn"],
            "best_fn": val_metrics["best_fn"],
            "epoch_time_sec": epoch_time,
            "saved_best": False,
            "warning": None,
        }

        if score_to_track > best_score:
            best_score = score_to_track
            best_epoch = epoch + 1
            best_val_metrics = dict(val_metrics)
            epoch_record["saved_best"] = True
            checkpoint = {
                "model_class": model.__class__.__name__,
                "model_config": getattr(model, "config", None),
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch + 1,
                "best_score": best_score,
                "optimize_for": optimize_for,
                "val_metrics": val_metrics,
            }
            torch.save(checkpoint, best_model_path)
            print("[INFO] Saved best model")

        history.append(epoch_record)

        warning = _check_overfitting(
            history,
            patience=overfit_patience,
            metric_name=f"best_{optimize_for}",
        )
        if warning is not None:
            history[-1]["warning"] = warning
            print(warning)

        _save_json({"history": history}, history_path)

        print(f"\nEpoch {epoch + 1}/{epochs}")
        print(f"Train Loss:      {train_loss:.6f}")
        print(f"Val Loss:        {val_metrics['loss']:.6f}")
        print(
            f"Val @ 0.5 -> "
            f"Acc: {val_metrics['accuracy']:.4f} | "
            f"Prec: {val_metrics['precision']:.4f} | "
            f"Rec: {val_metrics['recall']:.4f} | "
            f"F1: {val_metrics['f1']:.4f} | "
            f"F2: {val_metrics['f2']:.4f}"
        )
        print(f"Best threshold:  {val_metrics['best_threshold']:.4f}")
        print(
            f"Best metrics -> "
            f"Acc: {val_metrics['best_accuracy']:.4f} | "
            f"Prec: {val_metrics['best_precision']:.4f} | "
            f"Rec: {val_metrics['best_recall']:.4f} | "
            f"F1: {val_metrics['best_f1']:.4f} | "
            f"F2: {val_metrics['best_f2']:.4f}"
        )
        print(
            f"Confusion @ best -> "
            f"TP: {val_metrics['best_tp']} | "
            f"FP: {val_metrics['best_fp']} | "
            f"TN: {val_metrics['best_tn']} | "
            f"FN: {val_metrics['best_fn']}"
        )
        print(f"Epoch time:      {epoch_time:.2f}s")

    summary = {
        "run_name": save_path,
        "checkpoint_dir": str(run_dir),
        "best_model_path": str(best_model_path),
        "best_epoch": best_epoch,
        "optimize_for": optimize_for,
        "best_score": best_score,
        "best_val_metrics": best_val_metrics,
        "epochs": epochs,
        "lr": lr,
        "pos_weight": pos_weight,
        "num_thresholds": num_thresholds,
        "device": str(device),
    }
    _save_json(summary, summary_path)

    print(f"\nBest validation {optimize_for.upper()}: {best_score:.6f}")
    print(f"[INFO] Saved history -> {history_path}")
    print(f"[INFO] Saved summary -> {summary_path}")
    print(f"[INFO] Best model path -> {best_model_path}")

    return model