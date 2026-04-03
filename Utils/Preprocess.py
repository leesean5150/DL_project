import pandas as pd
import numpy as np
import os
from typing import Tuple, List, Dict

TRAIN_CSV_PATH = ""



def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_dir = os.path.join(project_root, "data")

    train_path = os.path.join(data_dir, "train_transaction.csv")
    test_path = os.path.join(data_dir, "test_transaction.csv")

    print(f"[INFO] Project root: {project_root}")
    print(f"[INFO] Data dir: {data_dir}")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Train file not found: {train_path}")

    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test file not found: {test_path}")

    print(f"[INFO] Loading train data from: {train_path}")
    train_df = pd.read_csv(train_path)

    print(f"[INFO] Loading test data from: {test_path}")
    test_df = pd.read_csv(test_path)

    print(f"[INFO] Train shape: {train_df.shape}")
    print(f"[INFO] Test shape: {test_df.shape}")

    return train_df, test_df

def preprocess(
    df: pd.DataFrame,
    mode: str = "baseline",
    sentinel_value: int = -999,
    features_to_drop: List[str] = None
):
    df = df.copy()

    if features_to_drop is None:
        features_to_drop = ['TransactionID']

    if 'TransactionDT' in df.columns:
        df['TransactionHour'] = (df['TransactionDT'] // 3600) % 24
        df['TransactionDay'] = df['TransactionDT'] // (3600 * 24)
        df['TransactionWeekday'] = df['TransactionDay'] % 7

    df.drop(
        columns=[c for c in features_to_drop if c in df.columns],
        inplace=True
    )

    y = df['isFraud'] if 'isFraud' in df.columns else None

    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    if 'isFraud' in numeric_cols:
        numeric_cols.remove('isFraud')

    # MODE: BASELINE (Selina implementation)
    if mode == "baseline":
        df[numeric_cols] = df[numeric_cols].fillna(sentinel_value)

        df[categorical_cols] = df[categorical_cols].fillna('Unknown')
        for col in categorical_cols:
            df[col] = df[col].astype('category').cat.codes

    elif mode == "ae":
        for col in numeric_cols:
            df[col + "_missing"] = df[col].isna().astype(int)

        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())

        df[numeric_cols] = (df[numeric_cols] - df[numeric_cols].mean()) / (df[numeric_cols].std() + 1e-6)


        df[categorical_cols] = df[categorical_cols].fillna('Unknown')
        for col in categorical_cols:
            df[col] = df[col].astype('category').cat.codes


    elif mode == "transformer":
        df[numeric_cols] = df[numeric_cols].fillna(0)
        df[categorical_cols] = df[categorical_cols].fillna('Unknown')

    else:
        raise ValueError(f"Unknown mode: {mode}")


    X = df.drop(columns=['isFraud']) if y is not None else df

    return X, y