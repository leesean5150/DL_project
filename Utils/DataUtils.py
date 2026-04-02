import json
from pathlib import Path
from typing import Dict, Any, Optional, Literal

import numpy as np
import torch
from torch.utils.data import Dataset


PREPROCESSED_DIRECTORY_PATH = "data/processed_fraud"


class BaseFraudDataset(Dataset):
    """
    Shared base dataset for fraud data stored in the preprocessed format.

    Expected files inside processed_dir:
        - metadata.json
        - feature_keys.json
        - uids.json
        - features.npy
        - labels.npy
        - seq_lengths.npy
        - train_uid_indices.npy
        - val_uid_indices.npy
        - test_uid_indices.npy

    Important design choice:
    Splits are performed at the UID level to avoid leakage, then expanded into
    transaction indices. Any filtering (e.g. non-fraud only) should happen
    AFTER the split expansion, not before.
    """

    def __init__(
        self,
        processed_dir: str,
        split: Optional[str] = None,
        label_dtype: torch.dtype = torch.float32,
    ):
        self.processed_dir = Path(processed_dir)
        self.split = split
        self.label_dtype = label_dtype

        # Load metadata / auxiliary files
        with open(self.processed_dir / "metadata.json", "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        with open(self.processed_dir / "feature_keys.json", "r", encoding="utf-8") as f:
            self.feature_keys = json.load(f)

        with open(self.processed_dir / "uids.json", "r", encoding="utf-8") as f:
            self.uids = json.load(f)

        # Load arrays using memmap where appropriate
        self.features = np.load(self.processed_dir / "features.npy", mmap_mode="r")
        self.labels = np.load(self.processed_dir / "labels.npy", mmap_mode="r")
        self.seq_lengths = np.load(self.processed_dir / "seq_lengths.npy", mmap_mode="r")

        self.num_transactions = int(self.metadata["num_transactions"])
        self.feature_dim = int(self.metadata["feature_dim"])
        self.num_uids = int(self.metadata["num_uids"])

        self._validate_shapes()
        self._build_uid_segments()

    def _validate_shapes(self) -> None:
        if self.features.shape != (self.num_transactions, self.feature_dim):
            raise ValueError(
                f"features.npy shape mismatch: expected {(self.num_transactions, self.feature_dim)}, "
                f"got {self.features.shape}"
            )

        if self.labels.shape != (self.num_transactions,):
            raise ValueError(
                f"labels.npy shape mismatch: expected {(self.num_transactions,)}, got {self.labels.shape}"
            )

        if self.seq_lengths.shape != (self.num_uids,):
            raise ValueError(
                f"seq_lengths.npy shape mismatch: expected {(self.num_uids,)}, got {self.seq_lengths.shape}"
            )

        if len(self.uids) != self.num_uids:
            raise ValueError(
                f"uids.json length mismatch: expected {self.num_uids}, got {len(self.uids)}"
            )

    def _build_uid_segments(self) -> None:
        """
        Rebuild UID segment boundaries.

        Example:
            seq_lengths = [1, 5, 1]
            uid_starts  = [0, 1, 6]
            uid_ends    = [1, 6, 7]   # exclusive
        """
        self.uid_starts = np.cumsum(
            np.concatenate([np.array([0], dtype=np.int64), self.seq_lengths[:-1]])
        )
        self.uid_ends = self.uid_starts + self.seq_lengths

    def _build_transaction_indices_from_split(self, split: Optional[str]) -> np.ndarray:
        """
        Expand a UID-level split into global transaction indices.
        """
        if split is None:
            return np.arange(self.num_transactions, dtype=np.int64)

        split_map = {
            "train": "train_uid_indices.npy",
            "val": "val_uid_indices.npy",
            "test": "test_uid_indices.npy",
        }

        if split not in split_map:
            raise ValueError(
                f"split must be one of {list(split_map.keys())} or None, got {split}"
            )

        uid_indices = np.load(self.processed_dir / split_map[split])

        txn_ranges = []
        for uid_idx in uid_indices:
            uid_idx = int(uid_idx)
            start = int(self.uid_starts[uid_idx])
            end = int(self.uid_ends[uid_idx])
            txn_ranges.append(np.arange(start, end, dtype=np.int64))

        if not txn_ranges:
            return np.empty((0,), dtype=np.int64)

        return np.concatenate(txn_ranges)

    def _locate_uid_idx(self, global_txn_idx: int) -> int:
        """
        Find which UID segment contains this global transaction index.
        uid_ends is exclusive, so searchsorted(..., side='right') works well.
        """
        return int(np.searchsorted(self.uid_ends, global_txn_idx, side="right"))

    def _apply_label_filter(
        self,
        indices: np.ndarray,
        filter_mode: Literal["all", "fraud_only", "nonfraud_only"],
    ) -> np.ndarray:
        """
        Filter already-split transaction indices by label.
        """
        if filter_mode == "all":
            return indices

        label_view = self.labels[indices]

        if filter_mode == "fraud_only":
            mask = label_view == 1
        elif filter_mode == "nonfraud_only":
            mask = label_view == 0
        else:
            raise ValueError(
                "filter_mode must be one of: 'all', 'fraud_only', 'nonfraud_only'"
            )

        return indices[mask]


class TransactionDataset(BaseFraudDataset):
    """
    Rolling history dataset.
    Each sample is a padded history window ending at one transaction,
    and the label is only the last transaction's label.

    Good for:
        - current transformer baseline
        - sequence-to-one prediction
    """

    def __init__(
        self,
        processed_dir: str,
        seq_len: int = 8,
        split: Optional[str] = None,
        label_dtype: torch.dtype = torch.float32,
    ):
        super().__init__(processed_dir=processed_dir, split=split, label_dtype=label_dtype)
        self.seq_len = seq_len
        self.sample_indices = self._build_transaction_indices_from_split(split)

    def __len__(self) -> int:
        return len(self.sample_indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")

        # Map split-local sample index -> original global transaction index
        global_txn_idx = int(self.sample_indices[idx])

        # Find owning UID segment
        uid_idx = self._locate_uid_idx(global_txn_idx)
        uid = self.uids[uid_idx]

        seq_start = int(self.uid_starts[uid_idx])

        # Build rolling history window ending at global_txn_idx
        window_start = max(seq_start, global_txn_idx - self.seq_len + 1)
        window_end = global_txn_idx + 1

        window = self.features[window_start:window_end]  # shape [L, F]
        actual_len = window.shape[0]
        pad_len = self.seq_len - actual_len

        if actual_len <= 0:
            raise ValueError(
                f"Empty window encountered for idx={idx}, "
                f"global_txn_idx={global_txn_idx}, uid={uid}"
            )

        # Left pad with zeros so the target transaction stays at the end
        x = np.zeros((self.seq_len, self.feature_dim), dtype=np.float32)
        x[pad_len:] = window

        attention_mask = np.zeros((self.seq_len,), dtype=np.bool_)
        attention_mask[pad_len:] = True

        y = self.labels[global_txn_idx]

        return {
            "x": torch.from_numpy(x),
            "attention_mask": torch.from_numpy(attention_mask),
            "y": torch.tensor(y, dtype=self.label_dtype),
            "uid": uid,
            "txn_idx": global_txn_idx,
        }


class FlatTransactionDataset(BaseFraudDataset):
    """
    One transaction per sample.

    Good for:
        - MLP / tabular baselines
        - autoencoder / encoder-decoder pretraining
        - groupmates who only want single-transaction models

    filter_mode:
        - "all"
        - "fraud_only"
        - "nonfraud_only"
    """

    def __init__(
        self,
        processed_dir: str,
        split: Optional[str] = None,
        label_dtype: torch.dtype = torch.float32,
        filter_mode: Literal["all", "fraud_only", "nonfraud_only"] = "all",
        return_labels: bool = True,
    ):
        super().__init__(processed_dir=processed_dir, split=split, label_dtype=label_dtype)
        self.filter_mode = filter_mode
        self.return_labels = return_labels

        base_indices = self._build_transaction_indices_from_split(split)
        self.sample_indices = self._apply_label_filter(base_indices, filter_mode)

    def __len__(self) -> int:
        return len(self.sample_indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")

        global_txn_idx = int(self.sample_indices[idx])
        uid_idx = self._locate_uid_idx(global_txn_idx)
        uid = self.uids[uid_idx]

        x = np.asarray(self.features[global_txn_idx], dtype=np.float32)
        y = self.labels[global_txn_idx]

        item = {
            "x": torch.from_numpy(x),
            "uid": uid,
            "txn_idx": global_txn_idx,
        }

        if self.return_labels:
            item["y"] = torch.tensor(y, dtype=self.label_dtype)

        return item


class SequenceLabelDataset(BaseFraudDataset):
    """
    Rolling history dataset with labels at every valid timestep.

    Returns:
        - x:              [seq_len, feature_dim]
        - attention_mask: [seq_len] True for real tokens, False for padding
        - y_seq:          [seq_len] labels aligned to x
        - loss_mask:      [seq_len] same valid region as attention_mask
        - y_last:         scalar label for final timestep

    Good for:
        - future sequence labeling transformer
        - per-timestep supervision
        - causal/left-context training setups later on
    """

    def __init__(
        self,
        processed_dir: str,
        seq_len: int = 8,
        split: Optional[str] = None,
        label_dtype: torch.dtype = torch.float32,
    ):
        super().__init__(processed_dir=processed_dir, split=split, label_dtype=label_dtype)
        self.seq_len = seq_len
        self.sample_indices = self._build_transaction_indices_from_split(split)

    def __len__(self) -> int:
        return len(self.sample_indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")

        global_txn_idx = int(self.sample_indices[idx])

        uid_idx = self._locate_uid_idx(global_txn_idx)
        uid = self.uids[uid_idx]

        seq_start = int(self.uid_starts[uid_idx])

        window_start = max(seq_start, global_txn_idx - self.seq_len + 1)
        window_end = global_txn_idx + 1

        window_x = self.features[window_start:window_end]   # [L, F]
        window_y = self.labels[window_start:window_end]     # [L]

        actual_len = window_x.shape[0]
        pad_len = self.seq_len - actual_len

        if actual_len <= 0:
            raise ValueError(
                f"Empty window encountered for idx={idx}, "
                f"global_txn_idx={global_txn_idx}, uid={uid}"
            )

        x = np.zeros((self.seq_len, self.feature_dim), dtype=np.float32)
        x[pad_len:] = window_x

        attention_mask = np.zeros((self.seq_len,), dtype=np.bool_)
        attention_mask[pad_len:] = True

        y_seq = np.zeros((self.seq_len,), dtype=np.float32)
        y_seq[pad_len:] = window_y.astype(np.float32)

        loss_mask = np.zeros((self.seq_len,), dtype=np.bool_)
        loss_mask[pad_len:] = True

        y_last = self.labels[global_txn_idx]

        return {
            "x": torch.from_numpy(x),
            "attention_mask": torch.from_numpy(attention_mask),
            "y_seq": torch.tensor(y_seq, dtype=self.label_dtype),
            "loss_mask": torch.from_numpy(loss_mask),
            "y_last": torch.tensor(y_last, dtype=self.label_dtype),
            "uid": uid,
            "txn_idx": global_txn_idx,
        }