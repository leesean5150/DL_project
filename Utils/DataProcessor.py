import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _coerce_to_float(value: Any) -> float:
    if value is None:
        return 0.0

    if isinstance(value, (int, float, bool, np.integer, np.floating)):
        out = float(value)
        if np.isnan(out):
            return 0.0
        return out

    if isinstance(value, str):
        try:
            out = float(value)
            if np.isnan(out):
                return 0.0
            return out
        except ValueError:
            return 0.0

    return 0.0


def _make_uid_splits(
    num_uids: int,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0")

    rng = np.random.default_rng(seed)
    uid_indices = np.arange(num_uids)
    rng.shuffle(uid_indices)

    train_end = int(num_uids * train_ratio)
    val_end = train_end + int(num_uids * val_ratio)

    train_idx = uid_indices[:train_end]
    val_idx = uid_indices[train_end:val_end]
    test_idx = uid_indices[val_end:]

    return train_idx, val_idx, test_idx


def _scan_missing_counts(
    jsonl_path: Path,
    feature_keys: List[str],
    label_key: str,
    sentinel_missing_value: float,
    sort_key: Optional[str] = None,
) -> Tuple[int, int, Dict[str, int]]:
    """
    Returns:
        num_uids
        num_transactions
        missing_counts: count of sentinel occurrences per feature
    """
    num_uids = 0
    num_transactions = 0
    missing_counts = {k: 0 for k in feature_keys}

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            transactions = record.get("transactions", [])
            if not transactions:
                continue

            if sort_key is not None:
                transactions = sorted(transactions, key=lambda x: x.get(sort_key, 0))

            num_uids += 1
            num_transactions += len(transactions)

            for txn in transactions:
                if label_key not in txn:
                    raise KeyError(
                        f"Missing label key '{label_key}' at line {line_num}"
                    )

                for key in feature_keys:
                    value = _coerce_to_float(txn.get(key, 0.0))
                    if value == sentinel_missing_value:
                        missing_counts[key] += 1

    return num_uids, num_transactions, missing_counts


def _build_feature_policy(
    feature_keys: List[str],
    missing_counts: Dict[str, int],
    num_transactions: int,
    flag_only_threshold: float,
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Returns:
        keep_only_value_keys
        keep_value_and_missing_flag_keys
        keep_only_present_flag_keys
        final_feature_keys
    """
    keep_only_value_keys = []
    keep_value_and_missing_flag_keys = []
    keep_only_present_flag_keys = []

    for key in feature_keys:
        missing_ratio = missing_counts[key] / num_transactions if num_transactions > 0 else 0.0

        if missing_counts[key] == 0:
            keep_only_value_keys.append(key)
        elif missing_ratio >= flag_only_threshold:
            keep_only_present_flag_keys.append(key)
        else:
            keep_value_and_missing_flag_keys.append(key)

    final_feature_keys = []
    final_feature_keys.extend(keep_only_value_keys)
    final_feature_keys.extend(keep_value_and_missing_flag_keys)
    final_feature_keys.extend([f"{k}_missing" for k in keep_value_and_missing_flag_keys])
    final_feature_keys.extend([f"{k}_present" for k in keep_only_present_flag_keys])

    return (
        keep_only_value_keys,
        keep_value_and_missing_flag_keys,
        keep_only_present_flag_keys,
        final_feature_keys,
    )


def _encode_transaction_with_policy(
    transaction: Dict[str, Any],
    keep_only_value_keys: List[str],
    keep_value_and_missing_flag_keys: List[str],
    keep_only_present_flag_keys: List[str],
    sentinel_missing_value: float,
) -> np.ndarray:
    values: List[float] = []

    # 1) raw values only
    for key in keep_only_value_keys:
        value = _coerce_to_float(transaction.get(key, 0.0))
        values.append(value)

    # 2) value + missing flag
    missing_flags: List[float] = []
    for key in keep_value_and_missing_flag_keys:
        value = _coerce_to_float(transaction.get(key, 0.0))
        if value == sentinel_missing_value:
            values.append(0.0)
            missing_flags.append(1.0)
        else:
            values.append(value)
            missing_flags.append(0.0)

    # 3) present flag only
    present_flags: List[float] = []
    for key in keep_only_present_flag_keys:
        value = _coerce_to_float(transaction.get(key, 0.0))
        if value == sentinel_missing_value:
            present_flags.append(0.0)
        else:
            present_flags.append(1.0)

    return np.array(values + missing_flags + present_flags, dtype=np.float32)


def preprocess_jsonl_to_disk(
    jsonl_path: str,
    output_dir: str,
    feature_keys: Optional[List[str]] = None,
    label_key: str = "isFraud",
    sort_key: Optional[str] = None,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    split_seed: int = 42,
    sentinel_missing_value: float = -999.0,
    flag_only_threshold: float = 0.90,
) -> None:
    """
    Preprocess grouped JSONL transactions into flat memory-mappable arrays.

    Missing-value policy:
      - no sentinel missing => keep raw value
      - sentinel missing ratio < flag_only_threshold => keep value with sentinel replaced by 0.0 + add <feature>_missing
      - sentinel missing ratio >= flag_only_threshold => drop value, keep <feature>_present only
    """
    jsonl_path = Path(jsonl_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    inferred_keys = set(feature_keys or [])

    # infer schema (we should have processed the jsonl enough so we should be able to just infer without more settings)
    if feature_keys is None:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                record = json.loads(line)
                transactions = record.get("transactions", [])
                if not transactions:
                    continue

                for txn in transactions:
                    for k in txn.keys():
                        if k != label_key:
                            inferred_keys.add(k)

        feature_keys = sorted(inferred_keys)

    if not feature_keys:
        raise ValueError("No feature keys found.")

    # scan missing counts and dataset sizes
    num_uids, num_transactions, missing_counts = _scan_missing_counts(
        jsonl_path=jsonl_path,
        feature_keys=feature_keys,
        label_key=label_key,
        sentinel_missing_value=sentinel_missing_value,
        sort_key=sort_key,
    )

    if num_uids == 0 or num_transactions == 0:
        raise ValueError("No usable UID records / transactions found.")

    (
        keep_only_value_keys,
        keep_value_and_missing_flag_keys,
        keep_only_present_flag_keys,
        final_feature_keys,
    ) = _build_feature_policy(
        feature_keys=feature_keys,
        missing_counts=missing_counts,
        num_transactions=num_transactions,
        flag_only_threshold=flag_only_threshold,
    )

    final_feature_dim = len(final_feature_keys)

    print(f"Found {num_uids} UID records")
    print(f"Found {num_transactions} total transactions")
    print(f"Original feature count: {len(feature_keys)}")
    print(f"Final feature count   : {final_feature_dim}")
    print(f"Keep raw only         : {len(keep_only_value_keys)}")
    print(f"Keep value+missing    : {len(keep_value_and_missing_flag_keys)}")
    print(f"Keep present-flag only: {len(keep_only_present_flag_keys)}")

    if keep_only_present_flag_keys:
        print("Flag-only features:")
        for k in keep_only_present_flag_keys:
            ratio = missing_counts[k] / num_transactions
            print(f"  {k}: missing_ratio={ratio:.4f}")

    if keep_value_and_missing_flag_keys:
        print("Value+missing-flag features:")
        for k in keep_value_and_missing_flag_keys:
            ratio = missing_counts[k] / num_transactions
            print(f"  {k}: missing_ratio={ratio:.4f}")

    # Allocate arrays
    features = np.lib.format.open_memmap(
        output_dir / "features.npy",
        mode="w+",
        dtype=np.float32,
        shape=(num_transactions, final_feature_dim),
    )
    labels = np.lib.format.open_memmap(
        output_dir / "labels.npy",
        mode="w+",
        dtype=np.float32,
        shape=(num_transactions,),
    )
    seq_lengths = np.lib.format.open_memmap(
        output_dir / "seq_lengths.npy",
        mode="w+",
        dtype=np.int64,
        shape=(num_uids,),
    )

    uids: List[str] = []

    # Write our encoded data
    txn_cursor = 0
    uid_cursor = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            uid = record["UID"]
            transactions = record.get("transactions", [])

            if not transactions:
                continue

            if sort_key is not None:
                transactions = sorted(transactions, key=lambda x: x.get(sort_key, 0))

            uids.append(uid)
            seq_lengths[uid_cursor] = len(transactions)

            for txn in transactions:
                if label_key not in txn:
                    raise KeyError(
                        f"Missing label key '{label_key}' for uid={uid} at line {line_num}"
                    )

                features[txn_cursor] = _encode_transaction_with_policy(
                    transaction=txn,
                    keep_only_value_keys=keep_only_value_keys,
                    keep_value_and_missing_flag_keys=keep_value_and_missing_flag_keys,
                    keep_only_present_flag_keys=keep_only_present_flag_keys,
                    sentinel_missing_value=sentinel_missing_value,
                )
                labels[txn_cursor] = float(txn[label_key])
                txn_cursor += 1

            uid_cursor += 1

    # flush memmaps
    del features
    del labels
    del seq_lengths

    with open(output_dir / "uids.json", "w", encoding="utf-8") as f:
        json.dump(uids, f)

    with open(output_dir / "feature_keys.json", "w", encoding="utf-8") as f:
        json.dump(final_feature_keys, f, indent=2)

    metadata = {
        "num_uids": num_uids,
        "num_transactions": num_transactions,
        "original_feature_dim": len(feature_keys),
        "feature_dim": final_feature_dim,
        "label_key": label_key,
        "sort_key": sort_key,
        "storage_format": "flat_transaction_arrays",
        "sentinel_missing_value": sentinel_missing_value,
        "flag_only_threshold": flag_only_threshold,
        "keep_only_value_keys": keep_only_value_keys,
        "keep_value_and_missing_flag_keys": keep_value_and_missing_flag_keys,
        "keep_only_present_flag_keys": keep_only_present_flag_keys,
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    train_uid_idx, val_uid_idx, test_uid_idx = _make_uid_splits(
        num_uids=num_uids,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=split_seed,
    )

    np.save(output_dir / "train_uid_indices.npy", train_uid_idx)
    np.save(output_dir / "val_uid_indices.npy", val_uid_idx)
    np.save(output_dir / "test_uid_indices.npy", test_uid_idx)

    print("Preprocessing complete.")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    data_dir = (base_dir / "../data").resolve()

    jsonl_path = data_dir / "ieee_fraud_sequences_full_embeddings.jsonl"
    output_dir = data_dir / "processed_fraud"

    print(f"Reading from: {jsonl_path}")
    print(f"Writing to : {output_dir}")

    preprocess_jsonl_to_disk(
        jsonl_path=str(jsonl_path),
        output_dir=str(output_dir),
        feature_keys=None,
        label_key="isFraud",
        sort_key="TransactionDT",
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        split_seed=100,
        sentinel_missing_value=-999.0,
        flag_only_threshold=0.90,
    )