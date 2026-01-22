"""Output directory utilities for organizing experiment results."""

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def clear_outputs_directory(output_dir: Path) -> None:
    """Delete and recreate the outputs directory for a fresh run."""
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def get_next_session_number(output_dir: Path) -> int:
    """Scan existing session_NNN folders and return next number."""
    if not output_dir.exists():
        return 1

    existing_sessions = []
    for folder in output_dir.iterdir():
        if folder.is_dir() and folder.name.startswith("session_"):
            try:
                session_num = int(folder.name.split("_")[1])
                existing_sessions.append(session_num)
            except (IndexError, ValueError):
                continue

    if not existing_sessions:
        return 1
    return max(existing_sessions) + 1


def create_session_folder(output_dir: Path) -> tuple[Path, int]:
    """Create and return session_NNN folder and session number."""
    output_dir.mkdir(parents=True, exist_ok=True)
    session_num = get_next_session_number(output_dir)
    session_folder = output_dir / f"session_{session_num:03d}"
    session_folder.mkdir(parents=True, exist_ok=True)
    return session_folder, session_num


def create_experiment_folder(
    session_dir: Path, model: str, features: str, target: str
) -> Path:
    """Create and return path: {model}_{features}_{target}/"""
    folder_name = f"{model}_{features}_{target}"
    folder_path = session_dir / folder_name
    folder_path.mkdir(parents=True, exist_ok=True)
    # Create plots subdirectory
    (folder_path / "plots").mkdir(exist_ok=True)
    return folder_path


def save_experiment_config(folder: Path, config: dict) -> None:
    """Save config.json with model type, features, hyperparameters."""
    config_path = folder / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, default=str)


def save_fold_metrics(folder: Path, fold_metrics: list[dict]) -> None:
    """Save fold_metrics.csv with one row per fold."""
    df = pd.DataFrame(fold_metrics)
    df.index.name = "fold"
    df.to_csv(folder / "fold_metrics.csv")


def save_predictions(folder: Path, fold_data: list[dict]) -> None:
    """Save predictions.csv with columns: fold, actual, predicted."""
    rows = []
    for fold_idx, data in enumerate(fold_data):
        y_true = data.get("y_true", [])
        y_pred = data.get("y_pred", [])
        for actual, predicted in zip(y_true, y_pred):
            rows.append({
                "fold": fold_idx,
                "actual": actual,
                "predicted": predicted,
            })
    df = pd.DataFrame(rows)
    df.to_csv(folder / "predictions.csv", index=False)


def save_experiment_summary(folder: Path, result: Any) -> None:
    """Save summary.json with aggregated metrics."""
    summary = {
        "experiment_id": result.experiment_id,
        "model_type": result.model_type,
        "feature_set": result.feature_set,
        "target": result.target,
        "n_folds": result.n_folds,
        "is_classification": result.is_classification,
        "metrics_mean": result.metrics_mean,
        "metrics_std": result.metrics_std,
    }
    summary_path = folder / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)


def update_aggregate_results(
    output_dir: Path,
    session_results: list[Any],
    session_num: int,
) -> None:
    """
    Update both detailed and summary aggregate CSV files.

    Args:
        output_dir: Root output directory
        session_results: List of CVResult objects from this session
        session_num: Current session number
    """
    detailed_path = output_dir / "aggregate_results_detailed.csv"
    summary_path = output_dir / "aggregate_results_summary.csv"

    # Build rows for this session
    session_rows = []
    for r in session_results:
        row = {
            "session": session_num,
            "experiment": r.experiment_id,
            "model": r.model_type,
            "features": r.feature_set,
            "target": r.target,
            "n_folds": r.n_folds,
            "type": "classification" if r.is_classification else "regression",
        }
        # Add all metrics with mean and std
        for key in r.metrics_mean:
            row[f"{key}_mean"] = r.metrics_mean[key]
            row[f"{key}_std"] = r.metrics_std[key]
        session_rows.append(row)

    session_df = pd.DataFrame(session_rows)

    # Update detailed CSV (append new session rows)
    if detailed_path.exists():
        existing_detailed = pd.read_csv(detailed_path)
        detailed_df = pd.concat([existing_detailed, session_df], ignore_index=True)
    else:
        detailed_df = session_df

    detailed_df.to_csv(detailed_path, index=False)

    # Update summary CSV (recalculate averages across all sessions)
    _update_summary_csv(detailed_df, summary_path)


def _update_summary_csv(detailed_df: pd.DataFrame, summary_path: Path) -> None:
    """Recalculate summary statistics across all sessions."""
    # Group by model, features, target, type
    group_cols = ["model", "features", "target", "type"]

    # Find metric columns (those ending in _mean or _std)
    mean_cols = [c for c in detailed_df.columns if c.endswith("_mean")]
    std_cols = [c for c in detailed_df.columns if c.endswith("_std")]

    summary_rows = []
    for group_key, group_df in detailed_df.groupby(group_cols):
        row = dict(zip(group_cols, group_key))
        row["n_sessions"] = len(group_df)

        # For each metric, compute average of means and stds across sessions
        # Also compute std of means across sessions (variance between runs)
        for mean_col in mean_cols:
            metric_name = mean_col  # e.g., "rmse_mean"
            base_name = mean_col.replace("_mean", "")  # e.g., "rmse"
            std_col = f"{base_name}_std"

            # Average of per-session means
            row[mean_col] = group_df[mean_col].mean()

            # Average of per-session stds (within-session variance)
            if std_col in group_df.columns:
                row[std_col] = group_df[std_col].mean()

            # Std of means across sessions (between-session variance)
            row[f"{base_name}_across_sessions_std"] = group_df[mean_col].std()

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    # Reorder columns for readability
    leading_cols = ["model", "features", "target", "type", "n_sessions"]
    other_cols = [c for c in summary_df.columns if c not in leading_cols]
    summary_df = summary_df[leading_cols + sorted(other_cols)]

    summary_df.to_csv(summary_path, index=False)
