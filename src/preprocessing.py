"""Data preprocessing for antibacterial polymer MIC prediction."""

import re
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(filepath: str | Path) -> pd.DataFrame:
    """Load the dataset from CSV file."""
    df = pd.read_csv(filepath)
    return df


def parse_mic_value(value: str | float) -> float:
    """
    Parse MIC values handling ranges and censored values.

    - Ranges ("32-64", "64-128"): Take geometric mean
    - Censored (">128", ">256"): Impute as threshold value
    - Numeric: Return as-is
    """
    if pd.isna(value):
        return np.nan

    value_str = str(value).strip()

    # Handle censored values (e.g., ">128", ">256")
    if value_str.startswith(">"):
        threshold = float(value_str[1:])
        return threshold

    # Handle ranges (e.g., "32-64")
    if "-" in value_str:
        parts = value_str.split("-")
        if len(parts) == 2:
            try:
                low, high = float(parts[0]), float(parts[1])
                # Geometric mean
                return np.sqrt(low * high)
            except ValueError:
                pass

    # Handle numeric values
    try:
        return float(value_str)
    except ValueError:
        return np.nan


def clean_mic_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Clean all MIC columns in the dataframe."""
    mic_cols = ["MIC_PAO1", "MIC_SA", "MIC_PAO1_PA"]
    df = df.copy()

    for col in mic_cols:
        if col in df.columns:
            df[col] = df[col].apply(parse_mic_value)

    return df


def bin_mic_values(y: np.ndarray) -> np.ndarray:
    """
    Bin MIC values into 3 classes for classification.

    Classes:
        0 = Low (Active): MIC <= 64
        1 = Medium (Moderate): 64 < MIC <= 128
        2 = High (Inactive): MIC > 128

    Args:
        y: Array of continuous MIC values

    Returns:
        Array of class labels (0, 1, or 2)
    """
    bins = np.zeros_like(y, dtype=int)
    bins[(y > 64) & (y <= 128)] = 1
    bins[y > 128] = 2
    return bins


MIC_CLASS_NAMES = ["Low (<=64)", "Medium (64-128)", "High (>128)"]


def get_base_features(df: pd.DataFrame) -> list[str]:
    """Return list of base feature column names."""
    return [
        "composition_A",
        "composition_B1",
        "composition_B2",
        "composition_C",
        "Number of blocks",
        "dpn",
        "Dispersity",
        "cLogP_predicted",
    ]


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full preprocessing pipeline.

    1. Clean MIC values
    2. Handle missing values
    3. Ensure correct data types
    """
    df = df.copy()

    # Clean MIC columns
    df = clean_mic_columns(df)

    # Ensure numeric columns are properly typed
    numeric_cols = [
        "composition_A", "composition_B1", "composition_B2", "composition_C",
        "Number of blocks", "dpn", "Dispersity", "cLogP_predicted",
        "Target", "NMR", "GPC"
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def split_data(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into train/test sets.

    Returns:
        X_train, X_test, y_train, y_test
    """
    # Remove rows with missing target values
    valid_mask = ~df[target_col].isna()
    df_valid = df[valid_mask]

    X = df_valid[feature_cols]
    y = df_valid[target_col]

    # Create bins for stratification (quartiles)
    y_bins = pd.qcut(y, q=4, labels=False, duplicates="drop")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y_bins,
    )

    return X_train, X_test, y_train, y_test


def save_processed_data(df: pd.DataFrame, output_path: str | Path) -> None:
    """Save processed dataframe to CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    # Test preprocessing
    data_path = Path(__file__).parent.parent / "Dataset final scix.xlsx - Dataset_Complete_modified.csv"

    df = load_data(data_path)
    print(f"Loaded {len(df)} samples")

    df = preprocess_data(df)
    print(f"Processed data shape: {df.shape}")

    # Check MIC distributions
    for col in ["MIC_PAO1", "MIC_SA", "MIC_PAO1_PA"]:
        print(f"\n{col}:")
        print(f"  Non-null: {df[col].notna().sum()}")
        print(f"  Min: {df[col].min():.2f}, Max: {df[col].max():.2f}")
        print(f"  Mean: {df[col].mean():.2f}, Median: {df[col].median():.2f}")
