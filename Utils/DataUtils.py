import json
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

PREPROCESSED_DIRECTORY_PATH = "data/processed_fraud" 

class TransactionDataset(Dataset):
    def __init__(
        self,
        processed_dir: str,
        seq_len: int = 8,
        split: Optional[str] = None,   # "train", "test" and "val" sets (Use those strings as the maps of how to split). 
        label_dtype: torch.dtype = torch.float32, # we preprocessed so everything should be using floats
    ):
        self.processed_dir = Path(processed_dir)
        self.seq_len = seq_len
        self.label_dtype = label_dtype
        self.split = split

        # Load our Metadata
        with open(self.processed_dir / "metadata.json", "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        with open(self.processed_dir / "feature_keys.json", "r", encoding="utf-8") as f:
            self.feature_keys = json.load(f)

        with open(self.processed_dir / "uids.json", "r", encoding="utf-8") as f:
            self.uids = json.load(f)

        # Load arrays
        self.features = np.load(self.processed_dir / "features.npy", mmap_mode="r")
        self.labels = np.load(self.processed_dir / "labels.npy", mmap_mode="r")
        self.seq_lengths = np.load(self.processed_dir / "seq_lengths.npy", mmap_mode="r")

        self.num_transactions = int(self.metadata["num_transactions"])
        self.feature_dim = int(self.metadata["feature_dim"])
        self.num_uids = int(self.metadata["num_uids"])

        # Basic consistency checks
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

        # Rebuild UID segments
        # Example:
        # seq_lengths = [1, 5, 1]
        # uid_starts  = [0, 1, 6]
        # uid_ends    = [1, 6, 7]   (exclusive)
        self.uid_starts = np.cumsum(
            np.concatenate([np.array([0], dtype=np.int64), self.seq_lengths[:-1]])
        )
        self.uid_ends = self.uid_starts + self.seq_lengths

        # Build sample indices for split
        self.sample_indices = self._build_sample_indices(split)

    def _build_sample_indices(self, split: Optional[str]) -> np.ndarray:
        if split is None:
            return np.arange(self.num_transactions, dtype=np.int64)

        split_map = {
            "train": "train_uid_indices.npy",
            "val": "val_uid_indices.npy",
            "test": "test_uid_indices.npy",
        }

        if split not in split_map:
            raise ValueError(f"split must be one of {list(split_map.keys())} or None, got {split}")

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

    # Note:
    # Splits are performed at the UID level to avoid leakage, but each transaction
    # still becomes one training sample. Therefore dataset length is the number of
    # transactions/windows in the selected split, not the number of UID sequences.
    # A UID with T transactions contributes T rolling-window samples. Which is basically the same as the number of actual transactions:
    # Example: Given a sequence S of 3 transactions: [T1, T2 , T3] we make a rollign window of:
    # S_1 => [T1, T2, T3] 
    # S_2 => [T2, T3]
    # S_3 => [T3] 
    # The number of sequences generated is equal to the number of transactions
    def __len__(self) -> int:
        return len(self.sample_indices)

    def _locate_uid_idx(self, global_txn_idx: int) -> int:
        """
        Find which UID segment contains this global transaction index.
        uid_ends is exclusive, so searchsorted(..., side='right') works well.
        """
        return int(np.searchsorted(self.uid_ends, global_txn_idx, side="right"))

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")

        # Map split-local sample index -> original global transaction index
        global_txn_idx = int(self.sample_indices[idx])

        # Find owning UID segment
        uid_idx = self._locate_uid_idx(global_txn_idx)
        uid = self.uids[uid_idx]

        seq_start = int(self.uid_starts[uid_idx])
        seq_end = int(self.uid_ends[uid_idx])  # exclusive, not used directly here

        # Build rolling history window ending at global_txn_idx
        window_start = max(seq_start, global_txn_idx - self.seq_len + 1)
        window_end = global_txn_idx + 1

        window = self.features[window_start:window_end]  # shape [L, F]
        actual_len = window.shape[0]
        pad_len = self.seq_len - actual_len

        if actual_len <= 0:
            raise ValueError(
                f"Empty window encountered for idx={idx}, global_txn_idx={global_txn_idx}, uid={uid}"
            )

        # Left pad with zeros (we left pad because we want the final value in the sequence to be the thing we predict)
        # For example if we had a transaction sequence of 3 it would look like::
        # S = [PAD, PAD, PAD, T1, T2, T3] <- this way T3 which is the thing we want to predict is the last value in the sequence
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
        }