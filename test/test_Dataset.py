import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

import json
import numpy as np
import torch
from torch.utils.data import DataLoader

from Utils import TransactionDataset, PREPROCESSED_DIRECTORY_PATH


PROCESSED_DIR = Path(PREPROCESSED_DIRECTORY_PATH)
SEQ_LEN = 8



def load_metadata(processed_dir: Path):
    with open(processed_dir / "metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    with open(processed_dir / "feature_keys.json", "r", encoding="utf-8") as f:
        feature_keys = json.load(f)

    with open(processed_dir / "uids.json", "r", encoding="utf-8") as f:
        uids = json.load(f)

    seq_lengths = np.load(processed_dir / "seq_lengths.npy", mmap_mode="r")
    return metadata, feature_keys, uids, seq_lengths


def compute_uid_boundaries(seq_lengths: np.ndarray):
    uid_starts = np.cumsum(
        np.concatenate([np.array([0], dtype=np.int64), seq_lengths[:-1]])
    )
    uid_ends = uid_starts + seq_lengths
    return uid_starts, uid_ends


def test_basic_loading():
    print("=== test_basic_loading ===")
    ds = TransactionDataset(
        processed_dir=str(PROCESSED_DIR),
        seq_len=SEQ_LEN,
        split=None,
    )

    print(f"Dataset length: {len(ds)}")
    print(f"Feature dim   : {ds.feature_dim}")
    print(f"Num UIDs      : {ds.num_uids}")
    print(f"Num txns      : {ds.num_transactions}")

    assert len(ds) == ds.num_transactions
    assert ds.features.shape == (ds.num_transactions, ds.feature_dim)
    assert ds.labels.shape == (ds.num_transactions,)
    assert ds.seq_lengths.shape == (ds.num_uids,)

    print("PASS\n")


def test_split_loading():
    print("=== test_split_loading ===")
    train_ds = TransactionDataset(str(PROCESSED_DIR), seq_len=SEQ_LEN, split="train")
    val_ds = TransactionDataset(str(PROCESSED_DIR), seq_len=SEQ_LEN, split="val")
    test_ds = TransactionDataset(str(PROCESSED_DIR), seq_len=SEQ_LEN, split="test")

    print(f"Train samples: {len(train_ds)}")
    print(f"Val samples  : {len(val_ds)}")
    print(f"Test samples : {len(test_ds)}")

    total = len(train_ds) + len(val_ds) + len(test_ds)
    print(f"Total split samples: {total}")

    full_ds = TransactionDataset(str(PROCESSED_DIR), seq_len=SEQ_LEN, split=None)
    assert total == len(full_ds), "Split sample counts do not sum to full dataset length"

    print("PASS\n")


def test_sample_structure():
    print("=== test_sample_structure ===")
    ds = TransactionDataset(
        processed_dir=str(PROCESSED_DIR),
        seq_len=SEQ_LEN,
        split="train",
    )

    sample = ds[0]

    print("Keys:", sample.keys())
    print("x shape:", sample["x"].shape)
    print("attention_mask shape:", sample["attention_mask"].shape)
    print("y:", sample["y"])
    print("uid:", sample["uid"])
    print("attention_mask:", sample["attention_mask"])

    assert isinstance(sample["x"], torch.Tensor)
    assert isinstance(sample["attention_mask"], torch.Tensor)
    assert isinstance(sample["y"], torch.Tensor)
    assert isinstance(sample["uid"], str)

    assert sample["x"].shape == (SEQ_LEN, ds.feature_dim)
    assert sample["attention_mask"].shape == (SEQ_LEN,)
    assert sample["attention_mask"].dtype == torch.bool

    print("PASS\n")


def test_attention_mask_consistency(): # keep in mind we are left padding to keep the last value in there the transaction we want to predict
    print("=== test_attention_mask_consistency ===")
    ds = TransactionDataset(
        processed_dir=str(PROCESSED_DIR),
        seq_len=SEQ_LEN,
        split="train",
    )

    for i in [0, 1, 2, 10, 100, len(ds) // 2]:
        sample = ds[i]
        mask = sample["attention_mask"].numpy()

        # mask should be [0,0,...,1,1,...,1]
        first_one = np.argmax(mask) if np.any(mask) else len(mask)

        if np.any(mask):
            assert np.all(mask[first_one:]), f"Inconsistent mask at idx={i}: {mask}"
            assert not np.any(mask[:first_one] & ~np.zeros_like(mask[:first_one], dtype=bool))

        num_valid = int(mask.sum())
        assert 1 <= num_valid <= SEQ_LEN, f"Invalid valid token count at idx={i}"

        print(f"idx={i}, valid_len={num_valid}, mask={mask}")

    print("PASS\n")


def test_no_uid_leakage():
    print("=== test_no_uid_leakage ===")
    train_ds = TransactionDataset(str(PROCESSED_DIR), seq_len=SEQ_LEN, split="train")
    val_ds = TransactionDataset(str(PROCESSED_DIR), seq_len=SEQ_LEN, split="val")
    test_ds = TransactionDataset(str(PROCESSED_DIR), seq_len=SEQ_LEN, split="test")

    train_uids = set(train_ds.uids[int(uid_idx)] for uid_idx in np.load(PROCESSED_DIR / "train_uid_indices.npy"))
    val_uids = set(val_ds.uids[int(uid_idx)] for uid_idx in np.load(PROCESSED_DIR / "val_uid_indices.npy"))
    test_uids = set(test_ds.uids[int(uid_idx)] for uid_idx in np.load(PROCESSED_DIR / "test_uid_indices.npy"))

    assert train_uids.isdisjoint(val_uids), "Train/Val UID leakage detected"
    assert train_uids.isdisjoint(test_uids), "Train/Test UID leakage detected"
    assert val_uids.isdisjoint(test_uids), "Val/Test UID leakage detected"

    print(f"Train UIDs: {len(train_uids)}")
    print(f"Val UIDs  : {len(val_uids)}")
    print(f"Test UIDs : {len(test_uids)}")
    print("PASS\n")


def test_first_transaction_padding():
    print("=== test_first_transaction_padding ===")
    ds = TransactionDataset(
        processed_dir=str(PROCESSED_DIR),
        seq_len=SEQ_LEN,
        split=None,
    )

    uid_starts = ds.uid_starts

    # test the first few UID starts
    for global_txn_idx in uid_starts[:10]:
        global_txn_idx = int(global_txn_idx)

        # Need to map back from global txn idx to local dataset idx for split=None
        local_idx = global_txn_idx
        sample = ds[local_idx]
        mask = sample["attention_mask"].numpy()

        # first txn of a UID should have exactly one valid token
        assert mask.sum() == 1, f"Expected one valid token for first txn, got {mask.sum()}"
        assert mask[-1] == 1, f"Last position should be valid for first txn, got mask={mask}"

        print(f"global_txn_idx={global_txn_idx}, uid={sample['uid']}, mask={mask}")

    print("PASS\n")


def test_window_boundary_respect():
    print("=== test_window_boundary_respect ===")
    ds = TransactionDataset(
        processed_dir=str(PROCESSED_DIR),
        seq_len=SEQ_LEN,
        split=None,
    )

    # pick a few UID starts and verify windows don't cross boundaries
    for uid_idx in range(min(10, ds.num_uids)):
        seq_start = int(ds.uid_starts[uid_idx])
        seq_end = int(ds.uid_ends[uid_idx])

        # test first transaction of UID
        sample = ds[seq_start]
        mask = sample["attention_mask"].numpy()
        assert mask.sum() == 1, f"Boundary violation at UID {uid_idx}, first txn mask={mask}"

        # test last transaction of UID if sequence has length > 1
        if seq_end - seq_start > 1:
            last_idx = seq_end - 1
            sample_last = ds[last_idx]
            mask_last = sample_last["attention_mask"].numpy()
            valid_len = int(mask_last.sum())

            expected_len = min(SEQ_LEN, seq_end - seq_start)
            assert valid_len == expected_len, (
                f"Boundary issue at UID {uid_idx}: expected {expected_len}, got {valid_len}"
            )

            print(
                f"UID {uid_idx}: start={seq_start}, end={seq_end}, "
                f"last_valid_len={valid_len}, expected={expected_len}"
            )

    print("PASS\n")


def test_dataloader_batch():
    print("=== test_dataloader_batch ===")
    ds = TransactionDataset(
        processed_dir=str(PROCESSED_DIR),
        seq_len=SEQ_LEN,
        split="train",
    )

    loader = DataLoader(
        ds,
        batch_size=16,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )

    batch = next(iter(loader))

    print("Batch x shape:", batch["x"].shape)
    print("Batch attention_mask shape:", batch["attention_mask"].shape)
    print("Batch y shape:", batch["y"].shape)
    print("Batch uid count:", len(batch["uid"]))

    assert batch["x"].shape == (16, SEQ_LEN, ds.feature_dim)
    assert batch["attention_mask"].shape == (16, SEQ_LEN)
    assert batch["y"].shape == (16,)
    assert len(batch["uid"]) == 16

    print("PASS\n")


def main():
    print("Running TransactionDataset tests...\n")

    test_basic_loading()
    test_split_loading()
    test_sample_structure()
    test_attention_mask_consistency()
    test_no_uid_leakage()
    test_first_transaction_padding()
    test_window_boundary_respect()
    test_dataloader_batch()

    print("All tests passed!")


if __name__ == "__main__":
    main()