"""Evaluation metrics and visualization for MIC prediction experiments."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import StratifiedKFold


@dataclass
class ExperimentResult:
    """Store results from a single experiment."""

    experiment_id: str
    model_type: str
    feature_set: str
    target: str
    rmse: float
    mae: float
    r2: float
    y_true: np.ndarray
    y_pred: np.ndarray
    train_losses: Optional[list[float]] = None
    val_losses: Optional[list[float]] = None
    feature_importance: Optional[dict[str, float]] = None


@dataclass
class CVResult:
    """Store results from cross-validation."""

    experiment_id: str
    model_type: str
    feature_set: str
    target: str
    n_folds: int
    is_classification: bool
    # Mean and std for each metric
    metrics_mean: dict[str, float]
    metrics_std: dict[str, float]
    # Per-fold metrics for detailed analysis
    fold_metrics: list[dict[str, float]]


def aggregate_cv_results(fold_metrics: list[dict[str, float]]) -> tuple[dict[str, float], dict[str, float]]:
    """
    Aggregate per-fold metrics into mean and std.

    Args:
        fold_metrics: List of metric dicts from each fold

    Returns:
        Tuple of (mean_dict, std_dict)
    """
    if not fold_metrics:
        return {}, {}

    # Get all metric keys from first fold
    metric_keys = fold_metrics[0].keys()

    means = {}
    stds = {}
    for key in metric_keys:
        values = [m[key] for m in fold_metrics]
        means[key] = np.mean(values)
        stds[key] = np.std(values)

    return means, stds


def cross_validate_model(
    model_fn,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    is_classification: bool = False,
    random_state: int = 42,
) -> tuple[dict[str, float], dict[str, float], list[dict[str, float]]]:
    """
    Perform stratified k-fold cross-validation.

    Args:
        model_fn: Function that returns a fresh model instance
        X: Features array
        y: Target array
        n_splits: Number of CV folds (default 5)
        is_classification: Whether this is classification task
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (metrics_mean, metrics_std, fold_metrics)
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # For regression, stratify on binned y
    if not is_classification:
        # Use qcut with duplicates='drop' to handle repeated values
        y_stratify = pd.qcut(y, q=min(n_splits, len(np.unique(y))), labels=False, duplicates='drop')
    else:
        y_stratify = y

    fold_metrics = []
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y_stratify)):
        model = model_fn()  # Fresh model each fold
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Create validation split from training data for models that need it
        n_val = int(len(X_train) * 0.2)
        val_indices = np.random.permutation(len(X_train))
        val_idx_local, train_idx_local = val_indices[:n_val], val_indices[n_val:]

        X_tr, X_val = X_train[train_idx_local], X_train[val_idx_local]
        y_tr, y_val = y_train[train_idx_local], y_train[val_idx_local]

        # Fit model
        try:
            # Try to fit with validation set (for NN, XGBoost)
            model.fit(X_tr, y_tr, X_val, y_val)
        except TypeError:
            # Fall back to simple fit for sklearn models
            model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # Calculate metrics
        if is_classification:
            metrics = calculate_classification_metrics(y_test, y_pred)
        else:
            metrics = calculate_metrics(y_test, y_pred)

        fold_metrics.append(metrics)

    # Aggregate results
    metrics_mean, metrics_std = aggregate_cv_results(fold_metrics)

    return metrics_mean, metrics_std, fold_metrics


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Calculate regression metrics."""
    return {
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


def calculate_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Calculate classification metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
    }


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str] = None,
    title: str = "Confusion Matrix",
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Plot confusion matrix for classification results."""
    cm = confusion_matrix(y_true, y_pred)

    if class_names is None:
        class_names = [f"Class {i}" for i in range(len(cm))]

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )

    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(title, fontsize=14)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_learning_curves(
    train_losses: list[float],
    val_losses: list[float],
    title: str = "Learning Curves",
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Plot training and validation loss curves."""
    fig, ax = plt.subplots(figsize=(10, 6))

    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, "b-", label="Training Loss", linewidth=2)
    if val_losses:
        ax.plot(epochs, val_losses, "r-", label="Validation Loss", linewidth=2)

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss (MSE)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Find best epoch
    if val_losses:
        best_epoch = np.argmin(val_losses) + 1
        best_loss = min(val_losses)
        ax.axvline(x=best_epoch, color="g", linestyle="--", alpha=0.7, label=f"Best: epoch {best_epoch}")
        ax.scatter([best_epoch], [best_loss], color="g", s=100, zorder=5)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Predicted vs Actual",
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Plot predicted vs actual values."""
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(y_true, y_pred, alpha=0.6, edgecolor="k", linewidth=0.5)

    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Perfect prediction")

    # Calculate metrics for annotation
    metrics = calculate_metrics(y_true, y_pred)
    annotation = f"RMSE: {metrics['rmse']:.2f}\nMAE: {metrics['mae']:.2f}\nR2: {metrics['r2']:.3f}"
    ax.text(
        0.05, 0.95, annotation,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    ax.set_xlabel("Actual MIC", fontsize=12)
    ax.set_ylabel("Predicted MIC", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_feature_importance(
    importance: dict[str, float],
    title: str = "Feature Importance",
    top_n: int = 15,
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Plot feature importance bar chart."""
    # Sort by importance
    sorted_importance = sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
    features, values = zip(*sorted_importance)

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in values]
    bars = ax.barh(range(len(features)), values, color=colors, alpha=0.8)

    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features)
    ax.invert_yaxis()
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_model_comparison(
    results: list[ExperimentResult],
    metric: str = "rmse",
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Compare models across feature sets."""
    # Create comparison dataframe
    data = []
    for r in results:
        data.append({
            "Model": r.model_type.upper(),
            "Feature Set": r.feature_set.title(),
            "RMSE": r.rmse,
            "MAE": r.mae,
            "R2": r.r2,
        })
    df = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(12, 6))

    metric_col = metric.upper() if metric.lower() != "r2" else "R2"
    x = np.arange(len(df["Feature Set"].unique()))
    width = 0.25

    models = df["Model"].unique()
    for i, model in enumerate(models):
        model_data = df[df["Model"] == model]
        values = [model_data[model_data["Feature Set"] == fs][metric_col].values[0]
                  for fs in df["Feature Set"].unique()]
        ax.bar(x + i * width, values, width, label=model, alpha=0.8)

    ax.set_xlabel("Feature Set", fontsize=12)
    ax.set_ylabel(metric_col, fontsize=12)
    ax.set_title(f"Model Comparison: {metric_col}", fontsize=14)
    ax.set_xticks(x + width)
    ax.set_xticklabels(df["Feature Set"].unique())
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_cv_model_comparison(
    results: list["CVResult"],
    metric: str = "rmse",
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Compare models across feature sets with error bars for CV results."""
    # Filter results for the appropriate task type
    if metric in ["accuracy", "f1_macro", "f1_weighted"]:
        results = [r for r in results if r.is_classification]
    else:
        results = [r for r in results if not r.is_classification]

    if not results:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(0.5, 0.5, f"No data for metric: {metric}", ha='center', va='center')
        return fig

    # Create comparison dataframe
    data = []
    for r in results:
        row = {
            "Model": r.model_type.upper(),
            "Feature Set": r.feature_set.title(),
        }
        # Add all metrics
        for key, val in r.metrics_mean.items():
            row[key] = val
            row[f"{key}_std"] = r.metrics_std.get(key, 0)
        data.append(row)
    df = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(12, 6))

    feature_sets = sorted(df["Feature Set"].unique())
    x = np.arange(len(feature_sets))
    width = 0.2

    models = sorted(df["Model"].unique())
    colors = plt.cm.Set2(np.linspace(0, 1, len(models)))

    for i, model in enumerate(models):
        model_data = df[df["Model"] == model]
        values = []
        errors = []
        for fs in feature_sets:
            fs_data = model_data[model_data["Feature Set"] == fs]
            if len(fs_data) > 0:
                values.append(fs_data[metric].values[0])
                errors.append(fs_data[f"{metric}_std"].values[0])
            else:
                values.append(0)
                errors.append(0)

        ax.bar(x + i * width, values, width, label=model, alpha=0.8,
               color=colors[i], yerr=errors, capsize=3)

    ax.set_xlabel("Feature Set", fontsize=12)
    metric_label = metric.upper() if metric not in ["r2", "accuracy"] else {"r2": "R²", "accuracy": "Accuracy"}[metric]
    ax.set_ylabel(metric_label, fontsize=12)
    ax.set_title(f"Model Comparison: {metric_label} (with CV std)", fontsize=14)
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(feature_sets)
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_entropy_vs_mic(
    df: pd.DataFrame,
    entropy_col: str,
    mic_col: str,
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Plot entropy feature vs MIC values."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Color by number of blocks
    scatter = ax.scatter(
        df[entropy_col],
        df[mic_col],
        c=df["Number of blocks"],
        cmap="viridis",
        alpha=0.7,
        edgecolor="k",
        linewidth=0.5,
        s=80,
    )

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Number of Blocks", fontsize=11)

    ax.set_xlabel(entropy_col.replace("_", " ").title(), fontsize=12)
    ax.set_ylabel(mic_col, fontsize=12)
    ax.set_title(f"{entropy_col.replace('_', ' ').title()} vs {mic_col}", fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def create_results_table(results: list[ExperimentResult]) -> pd.DataFrame:
    """Create a summary table of all experiment results."""
    data = []
    for r in results:
        data.append({
            "Experiment": r.experiment_id,
            "Model": r.model_type,
            "Features": r.feature_set,
            "Target": r.target,
            "RMSE": f"{r.rmse:.2f}",
            "MAE": f"{r.mae:.2f}",
            "R2": f"{r.r2:.3f}",
        })
    return pd.DataFrame(data)


def save_results_csv(results: list[ExperimentResult], path: str | Path) -> None:
    """Save experiment results to CSV."""
    df = create_results_table(results)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def generate_report(
    results: list[ExperimentResult],
    output_dir: str | Path,
) -> None:
    """Generate full experiment report with all plots and tables."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save results table
    save_results_csv(results, output_dir / "results.csv")

    # Group by target for comparison plots
    targets = set(r.target for r in results)

    for target in targets:
        target_results = [r for r in results if r.target == target]

        # Model comparison plot
        plot_model_comparison(
            target_results,
            metric="rmse",
            save_path=output_dir / f"{target}_comparison_rmse.png",
        )

        plot_model_comparison(
            target_results,
            metric="r2",
            save_path=output_dir / f"{target}_comparison_r2.png",
        )

        # Individual prediction plots
        for r in target_results:
            plot_predictions(
                r.y_true,
                r.y_pred,
                title=f"{r.model_type.upper()} - {r.feature_set} - {r.target}",
                save_path=output_dir / f"{r.experiment_id}_predictions.png",
            )

            # Learning curves for NN
            if r.train_losses:
                plot_learning_curves(
                    r.train_losses,
                    r.val_losses,
                    title=f"Learning Curves: {r.experiment_id}",
                    save_path=output_dir / f"{r.experiment_id}_learning_curves.png",
                )

            # Feature importance for XGBoost
            if r.feature_importance:
                plot_feature_importance(
                    r.feature_importance,
                    title=f"Feature Importance: {r.experiment_id}",
                    save_path=output_dir / f"{r.experiment_id}_importance.png",
                )

    plt.close("all")


if __name__ == "__main__":
    # Test with synthetic data
    np.random.seed(42)
    y_true = np.random.uniform(32, 256, 50)
    y_pred = y_true + np.random.randn(50) * 20

    metrics = calculate_metrics(y_true, y_pred)
    print("Metrics:", metrics)

    # Test plots
    fig = plot_predictions(y_true, y_pred, title="Test Predictions")
    plt.show()
