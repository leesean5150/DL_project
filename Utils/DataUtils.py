from __future__ import annotations

import sys
from pathlib import Path

project_root = Path().resolve().parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split

from .Preprocess import load_data, preprocess


class FraudAEDataset(Dataset):
    """
    PyTorch Dataset for autoencoder training on flat fraud features.

    Returns:
        x               if return_labels=False
        (x, y)          if return_labels=True
    """

    def __init__(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        dtype: torch.dtype = torch.float32,
        return_labels: bool = False,
    ) -> None:
        self.X = torch.tensor(
            X.to_numpy(dtype=np.float32, copy=True),
            dtype=dtype
        )
        self.x_shape = self.X.shape
        

        self.y = None
        self.y_shape = None

        if y is not None:
            self.y = torch.tensor(y.to_numpy(copy=True), dtype=torch.float32)
            self.y_shape = self.y.shape

        self.return_labels = return_labels

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        x = self.X[idx]

        if self.return_labels and self.y is not None:
            return x, self.y[idx]

        return x




def build_ae_datasets(
    mode="ae",
    train_only_nonfraud=True,
    return_labels=True,
    val_split=0.1,
    test_split=0.1,
    random_state=42,
):
    train_df, _ = load_data()  # ignore Kaggle test for now (maybe we use later for random bootstrapping)

    X, y = preprocess(train_df, mode=mode)


    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=val_split + test_split,
        random_state=random_state,
        stratify=y,
    )


    val_ratio = val_split / (val_split + test_split)

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=(1 - val_ratio),
        random_state=random_state,
        stratify=y_temp,
    )


    if train_only_nonfraud:
        mask = (y_train == 0)
        X_train = X_train.loc[mask]
        y_train = y_train.loc[mask]

    X_train = X_train.reset_index(drop=True)
    X_val = X_val.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)

    y_train = y_train.reset_index(drop=True)
    y_val = y_val.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)


    train_ds = FraudAEDataset(X_train, y_train, return_labels=return_labels)
    val_ds = FraudAEDataset(X_val, y_val, return_labels=return_labels)
    test_ds = FraudAEDataset(X_test, y_test, return_labels=return_labels)

    input_dim = X_train.shape[1]

    print(f"[INFO] Train: {X_train.shape}")
    print(f"[INFO] Val: {X_val.shape}")
    print(f"[INFO] Test: {X_test.shape}")

    return train_ds, val_ds, test_ds, input_dim


def build_ae_dataloaders(
    batch_size: int = 256,
    mode: str = "ae",
    train_only_nonfraud: bool = False,
    return_labels: bool = True,
    val_split: float = 0.1,
    num_workers: int = 0,
):
    train_ds, val_ds, test_ds, input_dim = build_ae_datasets(
        mode=mode,
        train_only_nonfraud=train_only_nonfraud,
        return_labels=return_labels,
        val_split=val_split,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False) if val_ds else None
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, input_dim