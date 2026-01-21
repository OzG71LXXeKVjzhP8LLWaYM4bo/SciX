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
