"""
SciX: Antibacterial Polymer MIC Prediction Pipeline

Main entry point for running experiments comparing composition vs entropy features
for predicting Minimum Inhibitory Concentration (MIC) of antibacterial polymers.
"""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.preprocessing import load_data, preprocess_data, split_data, get_base_features
from src.features import add_entropy_features, get_feature_sets
from src.models import get_model, NeuralNetworkRegressor, XGBoostRegressor
from src.evaluation import (
    ExperimentResult,
    calculate_metrics,
    generate_report,
    save_results_csv,
    plot_entropy_vs_mic,
)


def run_experiment(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_type: str,
    feature_names: list[str],
    experiment_id: str,
    feature_set: str,
    target: str,
    verbose: bool = True,
) -> ExperimentResult:
    """Run a single experiment and return results."""
    input_dim = X_train.shape[1]

    # Create model
    model = get_model(model_type, input_dim=input_dim)

    # Create validation split from training data
    n_val = int(len(X_train) * 0.2)
    indices = np.random.permutation(len(X_train))
    val_idx, train_idx = indices[:n_val], indices[n_val:]

    X_tr, X_val = X_train[train_idx], X_train[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]

    # Train model
    if verbose:
        print(f"\nTraining {model_type} on {feature_set} features for {target}...")

    if model_type == "nn":
        history = model.fit(
            X_tr, y_tr, X_val, y_val,
            epochs=200,
            batch_size=16,
            patience=30,
            verbose=verbose,
        )
        train_losses = history.train_losses
        val_losses = history.val_losses
    else:
        model.fit(
            X_tr, y_tr, X_val, y_val,
            feature_names=feature_names,
            verbose=verbose,
        )
        train_losses = None
        val_losses = None

    # Predictions
    y_pred = model.predict(X_test)

    # Metrics
    metrics = calculate_metrics(y_test, y_pred)

    # Feature importance (for XGBoost)
    feature_importance = None
    if isinstance(model, XGBoostRegressor):
        feature_importance = model.get_feature_importance()

    result = ExperimentResult(
        experiment_id=experiment_id,
        model_type=model_type,
        feature_set=feature_set,
        target=target,
        rmse=metrics["rmse"],
        mae=metrics["mae"],
        r2=metrics["r2"],
        y_true=y_test,
        y_pred=y_pred,
        train_losses=train_losses,
        val_losses=val_losses,
        feature_importance=feature_importance,
    )

    if verbose:
        print(f"  RMSE: {result.rmse:.2f}, MAE: {result.mae:.2f}, R2: {result.r2:.3f}")

    return result


def run_all_experiments(
    df: pd.DataFrame,
    targets: list[str] = None,
    model_types: list[str] = None,
    feature_sets: list[str] = None,
    random_state: int = 42,
    verbose: bool = True,
) -> list[ExperimentResult]:
    """Run the full experiment matrix."""
    np.random.seed(random_state)

    if targets is None:
        targets = ["MIC_PAO1", "MIC_SA", "MIC_PAO1_PA"]

    if model_types is None:
        model_types = ["nn", "xgboost", "ridge"]

    if feature_sets is None:
        feature_sets = ["composition", "entropy", "combined"]

    all_feature_sets = get_feature_sets()
    results = []
    experiment_counter = 1

    for target in targets:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Target: {target}")
            print(f"{'='*60}")

        for feature_set_name in feature_sets:
            feature_cols = all_feature_sets[feature_set_name]

            # Check all features exist
            missing = [f for f in feature_cols if f not in df.columns]
            if missing:
                print(f"Warning: Missing features {missing} for {feature_set_name}")
                continue

            # Split data
            X_train, X_test, y_train, y_test = split_data(
                df, target, feature_cols, random_state=random_state
            )

            # Convert to numpy
            X_train_np = X_train.values
            X_test_np = X_test.values
            y_train_np = y_train.values
            y_test_np = y_test.values

            for model_type in model_types:
                experiment_id = f"E{experiment_counter}_{model_type}_{feature_set_name}_{target}"

                result = run_experiment(
                    X_train_np,
                    y_train_np,
                    X_test_np,
                    y_test_np,
                    model_type=model_type,
                    feature_names=feature_cols,
                    experiment_id=experiment_id,
                    feature_set=feature_set_name,
                    target=target,
                    verbose=verbose,
                )

                results.append(result)
                experiment_counter += 1

    return results


def main():
    parser = argparse.ArgumentParser(description="SciX: Antibacterial Polymer MIC Prediction")
    parser.add_argument(
        "--data",
        type=str,
        default="Dataset final scix.xlsx - Dataset_Complete_modified.csv",
        help="Path to dataset CSV",
    )
    parser.add_argument(
        "--target",
        type=str,
        nargs="+",
        default=None,
        help="Target columns (default: all MIC targets)",
    )
    parser.add_argument(
        "--model",
        type=str,
        nargs="+",
        default=None,
        choices=["nn", "xgboost", "ridge"],
        help="Model types to run",
    )
    parser.add_argument(
        "--features",
        type=str,
        nargs="+",
        default=None,
        choices=["composition", "entropy", "combined"],
        help="Feature sets to use",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs",
        help="Output directory for results",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output",
    )

    args = parser.parse_args()

    # Load and preprocess data
    print("Loading data...")
    df = load_data(args.data)
    print(f"  Loaded {len(df)} samples")

    print("Preprocessing...")
    df = preprocess_data(df)

    print("Adding entropy features...")
    df = add_entropy_features(df)

    # Save processed data
    processed_path = Path("data/processed/polymer_data_processed.csv")
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_path, index=False)
    print(f"  Saved processed data to {processed_path}")

    # Run experiments
    print("\nRunning experiments...")
    results = run_all_experiments(
        df,
        targets=args.target,
        model_types=args.model,
        feature_sets=args.features,
        random_state=args.seed,
        verbose=not args.quiet,
    )

    # Generate report
    output_dir = Path(args.output)
    print(f"\nGenerating report in {output_dir}...")
    generate_report(results, output_dir)

    # Print summary
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)

    # Group by target
    for target in set(r.target for r in results):
        print(f"\n{target}:")
        target_results = [r for r in results if r.target == target]

        # Sort by RMSE
        target_results.sort(key=lambda x: x.rmse)

        for r in target_results:
            print(f"  {r.model_type:8s} | {r.feature_set:12s} | RMSE: {r.rmse:6.2f} | R2: {r.r2:+.3f}")

    # Best overall
    print("\n" + "-"*60)
    print("Best models by target:")
    for target in set(r.target for r in results):
        target_results = [r for r in results if r.target == target]
        best = min(target_results, key=lambda x: x.rmse)
        print(f"  {target}: {best.model_type} + {best.feature_set} (RMSE: {best.rmse:.2f})")

    # Entropy vs Composition comparison
    print("\n" + "-"*60)
    print("Entropy vs Composition (average RMSE improvement):")
    for model in set(r.model_type for r in results):
        comp_results = [r for r in results if r.model_type == model and r.feature_set == "composition"]
        ent_results = [r for r in results if r.model_type == model and r.feature_set == "entropy"]

        if comp_results and ent_results:
            comp_rmse = np.mean([r.rmse for r in comp_results])
            ent_rmse = np.mean([r.rmse for r in ent_results])
            improvement = (comp_rmse - ent_rmse) / comp_rmse * 100
            better = "entropy" if improvement > 0 else "composition"
            print(f"  {model:8s}: {better} is better by {abs(improvement):.1f}%")

    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    main()
