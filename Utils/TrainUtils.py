import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_fscore_support,
    accuracy_score,
    fbeta_score
)

from .DataUtils import TransactionDataset, PREPROCESSED_DIRECTORY_PATH

def get_device() -> torch.device:
    cuda_available = torch.cuda.is_available()
    return torch.device("cuda" if cuda_available else "cpu")

def build_dataloaders(
    processed_dir: str = PREPROCESSED_DIRECTORY_PATH,
    seq_len: int = 8,
    batch_size: int = 64,
    num_workers: int = 0, # I would not reccomend changing this as i have had bad experiences with it in the past but will expose it as an option anyways
    pin_memory: bool = True,
    shuffle_train: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = TransactionDataset(processed_dir=processed_dir,
                                  seq_len=seq_len,
                                  split="train")
    
    val_ds = TransactionDataset(processed_dir=processed_dir,
                                seq_len=seq_len,
                                split="val")
    
    test_ds = TransactionDataset(processed_dir=processed_dir,
                                 seq_len=seq_len,
                                 split="test")
    
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, val_loader, test_loader

def compute_pos_weight_from_loader(train_loader: DataLoader) -> torch.Tensor:
    positive_count = 0.0
    total_count = 0.0

    for batch in train_loader:
        y = batch["y"]
        positive_count += float(y.sum().item())
        total_count += float(y.numel())

    negative_count = total_count - positive_count

    if positive_count <= 0:
        raise ValueError("Training set contains no positive sampels.") # This should not happen as we checked that there were at least some positive samples
    
    pos_weight = negative_count/positive_count
    return torch.tensor(pos_weight, dtype=torch.float32)


# METRICS

def _safe_metric(fn, *args, default: float = 0.0, **kwargs) -> float:
    try:
        return float(fn(*args, **kwargs))
    except Exception:
        return float(default)


def compute_binary_classification_metrics(
    y_true: np.ndarray,
    logits: np.ndarray,
    threshold: float = 0.5) -> Dict[str, float]:

    probs = torch.sigmoid(torch.as_tensor(logits)).cpu().numpy()
    preds = (probs >= threshold).astype(np.int64)

    metrics = {
        "roc_auc": _safe_metric(roc_auc_score, y_true, probs, default=-1.0),
        "pr_auc": _safe_metric(average_precision_score, y_true, probs, default=-1.0),
        "accuracy": _safe_metric(accuracy_score, y_true, preds, default=-1.0),
        "f2": _safe_metric(fbeta_score, y_true, preds, beta=2.0, average="binary", zero_division=0)
    }

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        preds,
        average="binary",
        zero_division=0,
    )

    metrics["precision"] = float(precision)
    metrics["recall"] = float(recall)
    metrics["f1"] = float(f1)

    return metrics

# CHECKPOINTING HELPERS

def save_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    history: Dict[str, list],
    best_metric: float,
    scheduler: Optional[Any] = None,
    extra: Optional[Dict[str, Any]] = None) -> None:
    ckp_path = Path(checkpoint_path)
    ckp_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "history": history,
        "best_metric": best_metric,
        "extra": extra or {},        
    }

    torch.save(payload, ckp_path)

def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any] = None,
    map_location: str = "cpu") -> Dict[str, Any]:

    checkpoint = torch.load(checkpoint_path, map_location=map_location)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint

# TRAINING LOOPS
def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
    max_grad_norm: Optional[float] = None,
    metric_threshold: float = 0.5) -> Dict[str, float]:

    model.train()

    running_loss = 0.0
    total_samples = 0

    all_logits = []
    all_targets = []

    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True).float()

        optimizer.zero_grad()

        logits = model(x, attention_mask)
        loss = criterion(logits, y)

        loss.backward()

        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()

        batch_size = y.size(0)
        running_loss += float(loss.item()) * batch_size
        total_samples += batch_size

        all_logits.append(logits.detach().cpu())
        all_targets.append(y.detach().cpu())

    epoch_loss = running_loss / max(total_samples, 1)

    all_logits_np = torch.cat(all_logits).numpy()
    all_targets_np = torch.cat(all_targets).numpy()

    metrics = compute_binary_classification_metrics(all_targets_np, all_logits_np, threshold=metric_threshold)
    metrics["loss"] = float(epoch_loss)
    return metrics

@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    metric_threshold: float = 0.5) -> Dict[str, float]:

    model.eval()

    running_loss = 0.0
    total_samples = 0

    all_logits = []
    all_targets = []

    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True).float()

        logits = model(x, attention_mask)
        loss = criterion(logits, y)

        batch_size = y.size(0)
        running_loss += float(loss.item()) * batch_size
        total_samples += batch_size

        all_logits.append(logits.detach().cpu())
        all_targets.append(y.detach().cpu())

    epoch_loss = running_loss / max(total_samples, 1)

    all_logits_np = torch.cat(all_logits).numpy()
    all_targets_np = torch.cat(all_targets).numpy()

    metrics = compute_binary_classification_metrics(all_targets_np, all_logits_np, threshold=metric_threshold)
    metrics["loss"] = float(epoch_loss)
    return metrics

# MONITORS
def _is_improved(current: float, best: float, mode: str) -> bool:
    if mode == "max":
        return current > best
    if mode == "min":
        return current < best
    raise ValueError(f"mode must be 'max' or 'min', got {mode}")


def _maybe_print_overfitting_warning(
    history: Dict[str, list],
    metric_name: str = "pr_auc",
    patience: int = 3,
    gap_threshold: float = 0.10) -> None:
    train_key = f"train_{metric_name}"
    val_key = f"val_{metric_name}"

    # Exit early if the training history isnt long enough
    if len(history[train_key]) < patience or len(history[val_key]) < patience:
        return

    recent_train = history[train_key][-patience:]
    recent_val = history[val_key][-patience:]

    train_improving = recent_train[-1] > recent_train[0]
    val_not_improving = recent_val[-1] <= max(recent_val[:-1])

    current_gap = history[train_key][-1] - history[val_key][-1]

    # If our training score keeps improving and our validation does not (and it is greater than some threshold) signals prob overfitting
    if train_improving and val_not_improving and current_gap > gap_threshold:
        print(
            f"[warning] Possible overfitting detected: "
            f"train_{metric_name}={history[train_key][-1]:.4f}, "
            f"val_{metric_name}={history[val_key][-1]:.4f}, "
            f"gap={current_gap:.4f}"
        )

def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    num_epochs: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    checkpoint_dir: str = "checkpoints/default_run",
    monitor: str = "val_pr_auc",
    monitor_mode: str = "max",
    scheduler: Optional[Any] = None,
    use_pos_weight: bool = True,
    max_grad_norm: Optional[float] = 1.0,
    early_stopping_patience: Optional[int] = 5,
    save_last: bool = True,
    extra_checkpoint_info: Optional[Dict[str, Any]] = None,
    metric_threshold: float = 0.5
) -> Dict[str, Any]:
    model = model.to(device)

    # change if we want but have faith in the adam
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    if use_pos_weight:
        pos_weight = compute_pos_weight_from_loader(train_loader).to(device)
        print(f"[info] Using pos_weight={float(pos_weight.item()):.4f}")
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = torch.nn.BCEWithLogitsLoss()

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_roc_auc": [],
        "val_roc_auc": [],
        "train_pr_auc": [],
        "val_pr_auc": [],
        "train_precision": [],
        "val_precision": [],
        "train_recall": [],
        "val_recall": [],
        "train_f1": [],
        "val_f1": [],
        "train_f2": [],
        "val_f2": [],
        "lr": [],
        "epoch_time_sec": [],
    }

    best_metric = -float("inf") if monitor_mode == "max" else float("inf")
    best_epoch = -1
    epochs_without_improvement = 0

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            max_grad_norm=max_grad_norm,
            metric_threshold=metric_threshold
        )

        val_metrics = evaluate_model(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            metric_threshold=metric_threshold
        )

        # Monitor logic
        current_monitor_value = {
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "train_pr_auc": train_metrics["pr_auc"],
            "val_pr_auc": val_metrics["pr_auc"],
            "train_roc_auc": train_metrics["roc_auc"],
            "val_roc_auc": val_metrics["roc_auc"],
            "train_f1": train_metrics["f1"],
            "val_f1": val_metrics["f1"],
            "train_f2": train_metrics["f2"],
            "val_f2": val_metrics["f2"],
            "train_recall": train_metrics["recall"],
            "val_recall": val_metrics["recall"],
            "train_precision": train_metrics["precision"],
            "val_precision": val_metrics["precision"]
        }.get(monitor)

        if current_monitor_value is None:
            raise ValueError(f"Unsupported monitor metric: {monitor}")

        if scheduler is not None:
            # updated just in case we need to perform ablation
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(current_monitor_value)
            else:
                scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - epoch_start

        # Store history
        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_roc_auc"].append(train_metrics["roc_auc"])
        history["val_roc_auc"].append(val_metrics["roc_auc"])
        history["train_pr_auc"].append(train_metrics["pr_auc"])
        history["val_pr_auc"].append(val_metrics["pr_auc"])
        history["train_precision"].append(train_metrics["precision"])
        history["val_precision"].append(val_metrics["precision"])
        history["train_recall"].append(train_metrics["recall"])
        history["val_recall"].append(val_metrics["recall"])
        history["train_f1"].append(train_metrics["f1"])
        history["val_f1"].append(val_metrics["f1"])
        history["train_f2"].append(train_metrics["f2"])
        history["val_f2"].append(val_metrics["f2"])
        history["lr"].append(current_lr)
        history["epoch_time_sec"].append(epoch_time)

        # Progress print
        print(f"\nEpoch {epoch}/{num_epochs}")
        print(
            f"train_loss={train_metrics['loss']:.4f}  "
            f"val_loss={val_metrics['loss']:.4f}"
        )
        print(
            f"train_pr_auc={train_metrics['pr_auc']:.4f}  "
            f"val_pr_auc={val_metrics['pr_auc']:.4f}"
        )
        print(
            f"train_roc_auc={train_metrics['roc_auc']:.4f}  "
            f"val_roc_auc={val_metrics['roc_auc']:.4f}"
        )
        print(
            f"train_f1={train_metrics['f1']:.4f}  "
            f"val_f1={val_metrics['f1']:.4f}"
        )
        print(
            f"train_f2={train_metrics['f2']:.4f}  "
            f"val_f2={val_metrics['f2']:.4f}"
        )
        print(
            f"train_recall={train_metrics['recall']:.4f}  "
            f"val_recall={val_metrics['recall']:.4f}"
        )
        print(
            f"train_acc={train_metrics['accuracy']:.4f}  "
            f"val_acc={val_metrics['accuracy']:.4f}"
        )
        print(f"lr={current_lr:.6f}  epoch_time={epoch_time:.2f}s")


        if _is_improved(current_monitor_value, best_metric, monitor_mode):
            best_metric = current_monitor_value
            best_epoch = epoch
            epochs_without_improvement = 0

            best_path = checkpoint_dir / "best.pt"
            save_checkpoint(
                checkpoint_path=str(best_path),
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                history=history,
                best_metric=best_metric,
                extra=extra_checkpoint_info,
            )
            print(f"[checkpoint] Saved new best model -> {best_path}")
        else:
            epochs_without_improvement += 1

        if save_last:
            last_path = checkpoint_dir / "last.pt"
            save_checkpoint(
                checkpoint_path=str(last_path),
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                history=history,
                best_metric=best_metric,
                extra=extra_checkpoint_info,
            )

        metric_key = monitor.split("_", 1)[1]

        _maybe_print_overfitting_warning(
            history=history,
            metric_name=metric_key,
            patience=3,
            gap_threshold=0.10,
        )

        if early_stopping_patience is not None and epochs_without_improvement >= early_stopping_patience:
            print(
                f"[early stopping] No improvement in {monitor} for "
                f"{epochs_without_improvement} epochs. Stopping."
            )
            break

    summary = {
        "history": history,
        "best_metric": best_metric,
        "best_epoch": best_epoch,
        "checkpoint_dir": str(checkpoint_dir),
        "monitor": monitor,
        "monitor_mode": monitor_mode,
        "run_config": extra_checkpoint_info
    }

    summary_path = checkpoint_dir / "training_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nTraining complete. Best {monitor}={best_metric:.4f} at epoch {best_epoch}")
    return summary