"""
SciX: Antibacterial Polymer MIC Prediction Pipeline

Main entry point for running experiments comparing composition vs entropy features
for predicting Minimum Inhibitory Concentration (MIC) of antibacterial polymers.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.preprocessing import load_data, preprocess_data, bin_mic_values
from src.features import add_entropy_features, get_feature_sets
from src.models import get_model
from src.evaluation import (
    CVResult,
    cross_validate_model,
    plot_cv_model_comparison,
)


def run_cv_experiments(
    df: pd.DataFrame,
    targets: list[str] = None,
    model_types: list[str] = None,
    feature_sets: list[str] = None,
    n_folds: int = 5,
    random_state: int = 42,
    verbose: bool = True,
) -> list[CVResult]:
    """Run cross-validation experiments for all model/feature/target combinations."""
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
            print(f"Target: {target} ({n_folds}-fold CV)")
            print(f"{'='*60}")

        for feature_set_name in feature_sets:
            feature_cols = all_feature_sets[feature_set_name]

            # Check all features exist
            missing = [f for f in feature_cols if f not in df.columns]
            if missing:
                print(f"Warning: Missing features {missing} for {feature_set_name}")
                continue

            # Get full dataset (no split - CV handles it)
            X = df[feature_cols].values
            y_raw = df[target].values

            for model_type in model_types:
                experiment_id = f"CV{experiment_counter}_{model_type}_{feature_set_name}_{target}"
                is_classification = model_type == "logistic"

                # Prepare y (bin for classification)
                if is_classification:
                    y = bin_mic_values(y_raw)
                else:
                    y = y_raw

                if verbose:
                    print(f"\nCV: {model_type} | {feature_set_name} | {target}...")

                # Create model factory function
                input_dim = X.shape[1]
                def make_model(mt=model_type, dim=input_dim):
                    return get_model(mt, input_dim=dim)

                # Run cross-validation
                metrics_mean, metrics_std, fold_metrics = cross_validate_model(
                    model_fn=make_model,
                    X=X,
                    y=y,
                    n_splits=n_folds,
                    is_classification=is_classification,
                    random_state=random_state,
                )

                result = CVResult(
                    experiment_id=experiment_id,
                    model_type=model_type,
                    feature_set=feature_set_name,
                    target=target,
                    n_folds=n_folds,
                    is_classification=is_classification,
                    metrics_mean=metrics_mean,
                    metrics_std=metrics_std,
                    fold_metrics=fold_metrics,
                )

                # Print results
                if verbose:
                    if is_classification:
                        acc = metrics_mean.get("accuracy", 0)
                        acc_std = metrics_std.get("accuracy", 0)
                        f1 = metrics_mean.get("f1_macro", 0)
                        f1_std = metrics_std.get("f1_macro", 0)
                        print(f"  Acc: {acc:.3f} ± {acc_std:.3f} | F1: {f1:.3f} ± {f1_std:.3f}")
                    else:
                        rmse = metrics_mean.get("rmse", 0)
                        rmse_std = metrics_std.get("rmse", 0)
                        r2 = metrics_mean.get("r2", 0)
                        r2_std = metrics_std.get("r2", 0)
                        print(f"  RMSE: {rmse:.2f} ± {rmse_std:.2f} | R²: {r2:.3f} ± {r2_std:.3f}")

                results.append(result)
                experiment_counter += 1

    return results


def print_cv_summary(results: list[CVResult], output_dir: Path) -> None:
    """Print summary of cross-validation results and generate plots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print("CROSS-VALIDATION SUMMARY")
    print("="*60)

    # Separate classification and regression results
    classification_results = [r for r in results if r.is_classification]
    regression_results = [r for r in results if not r.is_classification]

    # Print regression results
    if regression_results:
        print("\n--- Regression Models ---")
        for target in sorted(set(r.target for r in regression_results)):
            n_folds = regression_results[0].n_folds
            print(f"\n{target} ({n_folds}-fold CV):")
            target_results = [r for r in regression_results if r.target == target]
            # Sort by mean RMSE
            target_results.sort(key=lambda x: x.metrics_mean.get("rmse", float("inf")))

            for r in target_results:
                rmse = r.metrics_mean.get("rmse", 0)
                rmse_std = r.metrics_std.get("rmse", 0)
                r2 = r.metrics_mean.get("r2", 0)
                r2_std = r.metrics_std.get("r2", 0)
                # Flag high variance (std > 10% of mean)
                high_var = " ⚠" if rmse_std > rmse * 0.1 else ""
                print(f"  {r.model_type:8s} | {r.feature_set:12s} | RMSE: {rmse:5.2f} ± {rmse_std:4.2f} | R²: {r2:+.3f} ± {r2_std:.3f}{high_var}")

            # Generate comparison plots for this target
            plot_cv_model_comparison(
                target_results,
                metric="rmse",
                save_path=output_dir / f"{target}_comparison_rmse.png",
            )
            plot_cv_model_comparison(
                target_results,
                metric="r2",
                save_path=output_dir / f"{target}_comparison_r2.png",
            )

    # Print classification results
    if classification_results:
        print("\n--- Classification Models ---")
        for target in sorted(set(r.target for r in classification_results)):
            n_folds = classification_results[0].n_folds
            print(f"\n{target} ({n_folds}-fold CV, 3-class):")
            target_results = [r for r in classification_results if r.target == target]
            # Sort by mean accuracy (descending)
            target_results.sort(key=lambda x: x.metrics_mean.get("accuracy", 0), reverse=True)

            for r in target_results:
                acc = r.metrics_mean.get("accuracy", 0)
                acc_std = r.metrics_std.get("accuracy", 0)
                f1 = r.metrics_mean.get("f1_macro", 0)
                f1_std = r.metrics_std.get("f1_macro", 0)
                # Flag high variance
                high_var = " ⚠" if acc_std > acc * 0.1 else ""
                print(f"  {r.model_type:8s} | {r.feature_set:12s} | Acc: {acc:.3f} ± {acc_std:.3f} | F1: {f1:.3f} ± {f1_std:.3f}{high_var}")

            # Generate comparison plots for this target
            plot_cv_model_comparison(
                target_results,
                metric="accuracy",
                save_path=output_dir / f"{target}_comparison_accuracy.png",
            )

    # Best models
    print("\n" + "-"*60)
    print("Best models by target:")

    if regression_results:
        for target in sorted(set(r.target for r in regression_results)):
            target_results = [r for r in regression_results if r.target == target]
            best = min(target_results, key=lambda x: x.metrics_mean.get("rmse", float("inf")))
            rmse = best.metrics_mean.get("rmse", 0)
            rmse_std = best.metrics_std.get("rmse", 0)
            print(f"  {target} (regression): {best.model_type} + {best.feature_set} (RMSE: {rmse:.2f} ± {rmse_std:.2f})")

    if classification_results:
        for target in sorted(set(r.target for r in classification_results)):
            target_results = [r for r in classification_results if r.target == target]
            best = max(target_results, key=lambda x: x.metrics_mean.get("accuracy", 0))
            acc = best.metrics_mean.get("accuracy", 0)
            acc_std = best.metrics_std.get("accuracy", 0)
            print(f"  {target} (classification): {best.model_type} + {best.feature_set} (Acc: {acc:.3f} ± {acc_std:.3f})")

    # Save CV results to CSV
    save_cv_results_csv(results, output_dir / "cv_results.csv")
    print(f"\nResults saved to {output_dir}/cv_results.csv")
    print(f"Plots saved to {output_dir}/")


def save_cv_results_csv(results: list[CVResult], path: Path) -> None:
    """Save CV results to CSV file."""
    data = []
    for r in results:
        row = {
            "Experiment": r.experiment_id,
            "Model": r.model_type,
            "Features": r.feature_set,
            "Target": r.target,
            "N_Folds": r.n_folds,
            "Type": "classification" if r.is_classification else "regression",
        }
        # Add all metrics with mean and std
        for key in r.metrics_mean:
            row[f"{key}_mean"] = r.metrics_mean[key]
            row[f"{key}_std"] = r.metrics_std[key]
        data.append(row)

    df = pd.DataFrame(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


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
        choices=["nn", "xgboost", "ridge", "logistic"],
        help="Model types to run (logistic is a classifier)",
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
        "--cv",
        type=int,
        default=5,
        help="Number of CV folds (default: 5)",
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

    output_dir = Path(args.output)

    # Run cross-validation experiments
    print(f"\nRunning {args.cv}-fold cross-validation experiments...")
    cv_results = run_cv_experiments(
        df,
        targets=args.target,
        model_types=args.model,
        feature_sets=args.features,
        n_folds=args.cv,
        random_state=args.seed,
        verbose=not args.quiet,
    )

    # Print CV summary and generate plots
    print_cv_summary(cv_results, output_dir)


if __name__ == "__main__":
    main()
