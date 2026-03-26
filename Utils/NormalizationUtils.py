import json
import shutil
from pathlib import Path
from typing import Dict, Any

import numpy as np

CATEGORICAL_LIKE_KEYS = {
    "ProductCD",
    "card4",
    "card6",
    "P_emaildomain",
    "R_emaildomain",
    "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9",
}


def get_train_transaction_indices(processed_dir: str) -> np.ndarray:
    processed_dir = Path(processed_dir)

    seq_lengths = np.load(processed_dir / "seq_lengths.npy", mmap_mode="r")
    train_uid_idx = np.load(processed_dir / "train_uid_indices.npy")

    uid_starts = np.cumsum(
        np.concatenate([np.array([0], dtype=np.int64), seq_lengths[:-1]])
    )
    uid_ends = uid_starts + seq_lengths

    txn_indices = []
    for uid_idx in train_uid_idx:
        uid_idx = int(uid_idx)
        start = int(uid_starts[uid_idx])
        end = int(uid_ends[uid_idx])
        txn_indices.append(np.arange(start, end, dtype=np.int64))

    return np.concatenate(txn_indices) if txn_indices else np.empty((0,), dtype=np.int64)


def fit_feature_normalizer(
    processed_dir: str,
    output_stats_path: str,
) -> Dict[str, Any]:
    processed_dir = Path(processed_dir)
    output_stats_path = Path(output_stats_path)

    features = np.load(processed_dir / "features.npy", mmap_mode="r")
    with open(processed_dir / "feature_keys.json", "r", encoding="utf-8") as f:
        feature_keys = json.load(f)

    train_indices = get_train_transaction_indices(str(processed_dir))
    train_features = np.asarray(features[train_indices], dtype=np.float32)

    normalize_mask = np.array(
        [
            not (
                k.endswith("_missing") 
                or k.endswith("_present")
                or k in CATEGORICAL_LIKE_KEYS
            )
            for k in feature_keys
        ],
        dtype=bool,
    )

    means = np.zeros(len(feature_keys), dtype=np.float32)
    stds = np.ones(len(feature_keys), dtype=np.float32)

    if normalize_mask.any():
        means[normalize_mask] = train_features[:, normalize_mask].mean(axis=0)
        stds[normalize_mask] = train_features[:, normalize_mask].std(axis=0)

    # Avoid divide-by-zero / tiny std explosions
    stds[stds < 1e-6] = 1.0

    stats = {
        "feature_keys": feature_keys,
        "means": means.tolist(),
        "stds": stds.tolist(),
        "normalize_mask": normalize_mask.tolist(),
    }

    output_stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    return stats


def apply_feature_normalizer(
    source_processed_dir: str,
    target_processed_dir: str,
    stats_path: str,
) -> None:
    source_processed_dir = Path(source_processed_dir)
    target_processed_dir = Path(target_processed_dir)
    stats_path = Path(stats_path)

    target_processed_dir.mkdir(parents=True, exist_ok=True)

    # Copy non-feature files as-is
    for filename in [
        "labels.npy",
        "seq_lengths.npy",
        "uids.json",
        "feature_keys.json",
        "metadata.json",
        "train_uid_indices.npy",
        "val_uid_indices.npy",
        "test_uid_indices.npy",
    ]:
        shutil.copy2(source_processed_dir / filename, target_processed_dir / filename)

    with open(stats_path, "r", encoding="utf-8") as f:
        stats = json.load(f)

    means = np.array(stats["means"], dtype=np.float32)
    stds = np.array(stats["stds"], dtype=np.float32)
    normalize_mask = np.array(stats["normalize_mask"], dtype=bool)

    features = np.load(source_processed_dir / "features.npy", mmap_mode="r")
    normalized = np.lib.format.open_memmap(
        target_processed_dir / "features.npy",
        mode="w+",
        dtype=np.float32,
        shape=features.shape,
    )

    chunk_size = 50000
    n = features.shape[0]

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = np.array(features[start:end], dtype=np.float32, copy=True)

        chunk[:, normalize_mask] = (
            chunk[:, normalize_mask] - means[normalize_mask]
        ) / stds[normalize_mask]

        normalized[start:end] = chunk

    del normalized