import json
from typing import List, Dict, Optional, Any
import torch
from torch.utils.data import Dataset


class TransactionDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str,
        seq_len: int = 8,
        feature_keys: Optional[List[str]] = None,
        label_key: str = "isFraud",
        encoder=None,
        sort_key: Optional[str] = None,
    ):
        self.seq_len = seq_len
        self.label_key = label_key
        self.feature_keys = feature_keys or []
        self.samples = []
        self.encoder = encoder

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                uid = record["UID"]
                transactions = record["transactions"]

                if not transactions:
                    continue

                if sort_key is not None:
                    transactions = sorted(
                        transactions,
                        key=lambda x: x.get(sort_key, 0)
                    )

                for i in range(len(transactions)):
                    if self.label_key not in transactions[i]:
                        raise KeyError(
                            f"Missing label key '{self.label_key}' for uid={uid}, idx={i}"
                        )

                    start = max(0, i - self.seq_len + 1)
                    window = transactions[start:i + 1]

                    feature_window = []
                    for txn in window:
                        txn_cpy = dict(txn)
                        txn_cpy.pop(self.label_key, None)
                        feature_window.append(txn_cpy)

                    self.samples.append({
                        "uid": uid,
                        "window": feature_window,
                        "label": transactions[i][self.label_key],
                    })

    def __len__(self):
        return len(self.samples)

    def _encode_transaction(self, transaction: Dict[str, Any]):
        if self.encoder is not None:
            encoded = self.encoder.encode(transaction)

            if isinstance(encoded, torch.Tensor):
                encoded = encoded.detach().cpu().flatten().tolist()

            if not isinstance(encoded, list):
                raise TypeError(
                    f"Encoder must return list or 1D torch.Tensor, got {type(encoded)}"
                )

            return encoded

        features = []
        for key in self.feature_keys:
            value = transaction.get(key, 0.0)

            if value is None:
                value = 0.0

            if isinstance(value, (int, float)):
                features.append(float(value))
            else:
                features.append(0.0)

        return features

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        window = sample["window"]

        encoded_window = [self._encode_transaction(txn) for txn in window]

        if len(encoded_window) == 0:
            raise ValueError(f"Empty encoded window at idx={idx}")

        feature_dim = len(encoded_window[0])
        pad_len = self.seq_len - len(encoded_window)

        if pad_len > 0:
            padding = [[0.0] * feature_dim for _ in range(pad_len)]
            encoded_window = padding + encoded_window

        x = torch.tensor(encoded_window, dtype=torch.float32)
        y = torch.tensor(sample["label"], dtype=torch.float32)
        attention_mask = torch.tensor(
            [0] * pad_len + [1] * len(window),
            dtype=torch.bool
        )

        return {
            "x": x,
            "attention_mask": attention_mask,
            "y": y,
            "uid": sample["uid"],
        }