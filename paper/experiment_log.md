# SciX Experiment Log

This document tracks all experiments run for the antibacterial polymer MIC prediction project.

---

## Experiment Overview

### MIC_PAO1 Experiments

| ID | Date | Model | Features | Target | RMSE | MAE | R² | Notes |
|----|------|-------|----------|--------|------|-----|-----|-------|
| E1 | 2026-01-21 | NN | composition | MIC_PAO1 | 52.32 | 38.29 | -0.052 | Overfitting |
| E2 | 2026-01-21 | XGBoost | composition | MIC_PAO1 | 43.76 | 21.66 | 0.264 | Best XGB |
| E3 | 2026-01-21 | Ridge | composition | MIC_PAO1 | 45.53 | 28.19 | 0.203 | |
| E4 | 2026-01-21 | NN | entropy | MIC_PAO1 | 43.89 | 29.15 | 0.260 | Improved vs comp |
| E5 | 2026-01-21 | XGBoost | entropy | MIC_PAO1 | 51.24 | 27.10 | -0.009 | Poor |
| E6 | 2026-01-21 | Ridge | entropy | MIC_PAO1 | 44.05 | 25.88 | 0.254 | |
| E7 | 2026-01-21 | NN | combined | MIC_PAO1 | 49.69 | 34.17 | 0.051 | |
| E8 | 2026-01-21 | XGBoost | combined | MIC_PAO1 | 55.88 | 20.79 | -0.200 | Worst |
| E9 | 2026-01-21 | Ridge | combined | MIC_PAO1 | 41.50 | 25.96 | 0.338 | **BEST** |

### MIC_SA Experiments

| ID | Date | Model | Features | Target | RMSE | MAE | R² | Notes |
|----|------|-------|----------|--------|------|-----|-----|-------|
| E10 | 2026-01-21 | NN | composition | MIC_SA | 43.16 | 32.07 | 0.470 | |
| E11 | 2026-01-21 | XGBoost | composition | MIC_SA | 40.35 | 25.42 | 0.537 | |
| E12 | 2026-01-21 | Ridge | composition | MIC_SA | 36.46 | 25.59 | 0.622 | |
| E13 | 2026-01-21 | NN | entropy | MIC_SA | 56.15 | 42.77 | 0.103 | Poor NN |
| E14 | 2026-01-21 | XGBoost | entropy | MIC_SA | 38.66 | 26.25 | 0.575 | |
| E15 | 2026-01-21 | Ridge | entropy | MIC_SA | 31.37 | 20.12 | 0.720 | Strong |
| E16 | 2026-01-21 | NN | combined | MIC_SA | 40.40 | 30.03 | 0.536 | |
| E17 | 2026-01-21 | XGBoost | combined | MIC_SA | 35.15 | 23.47 | 0.648 | |
| E18 | 2026-01-21 | Ridge | combined | MIC_SA | 29.36 | 20.46 | 0.755 | **BEST OVERALL** |

### MIC_PAO1_PA Experiments

| ID | Date | Model | Features | Target | RMSE | MAE | R² | Notes |
|----|------|-------|----------|--------|------|-----|-----|-------|
| E19 | 2026-01-21 | NN | composition | MIC_PAO1_PA | 48.80 | 35.38 | 0.085 | |
| E20 | 2026-01-21 | XGBoost | composition | MIC_PAO1_PA | 51.30 | 25.63 | -0.012 | |
| E21 | 2026-01-21 | Ridge | composition | MIC_PAO1_PA | 47.58 | 29.77 | 0.130 | |
| E22 | 2026-01-21 | NN | entropy | MIC_PAO1_PA | 43.80 | 29.82 | 0.263 | Improved |
| E23 | 2026-01-21 | XGBoost | entropy | MIC_PAO1_PA | 47.28 | 20.97 | 0.141 | |
| E24 | 2026-01-21 | Ridge | entropy | MIC_PAO1_PA | 39.82 | 25.44 | 0.391 | **BEST** |
| E25 | 2026-01-21 | NN | combined | MIC_PAO1_PA | 52.53 | 38.46 | -0.060 | |
| E26 | 2026-01-21 | XGBoost | combined | MIC_PAO1_PA | 54.79 | 23.42 | -0.154 | |
| E27 | 2026-01-21 | Ridge | combined | MIC_PAO1_PA | 43.39 | 25.59 | 0.276 | |

---

## Detailed Experiment Records

### Run 1: Initial Baseline

**Date**: [To be filled]

**Configuration**:
```bash
uv run python main.py --seed 42
```

**Hyperparameters**:
- Neural Network:
  - Hidden dims: [64, 32, 16]
  - Dropout: 0.3
  - Learning rate: 0.001
  - Weight decay: 0.01
  - Epochs: 200
  - Patience: 30
  - Batch size: 16

- XGBoost:
  - n_estimators: 200
  - max_depth: 5
  - learning_rate: 0.05
  - subsample: 0.8
  - colsample_bytree: 0.8

- Ridge:
  - alpha: 1.0

**Results**:
- Total experiments: 27 (9 per target × 3 targets)
- Best overall: Ridge + Combined on MIC_SA (R² = 0.755)
- Entropy improvement: 11.1% average RMSE reduction for Ridge

**Observations**:
1. Ridge regression consistently outperformed NN and XGBoost
2. Small dataset (111 samples) favored simpler models
3. Entropy features most beneficial for MIC_PAO1_PA target
4. Neural networks showed signs of overfitting despite regularization
5. XGBoost feature importance highlighted entropy metrics as predictive

---

## Hyperparameter Tuning Log

### NN Architecture Search

| Hidden Dims | Dropout | Val Loss | Notes |
|-------------|---------|----------|-------|
| [64, 32, 16] | 0.3 | - | Baseline |
| [32, 16] | 0.2 | - | Simpler |
| [128, 64, 32] | 0.4 | - | Deeper |

### XGBoost Tuning

| n_est | max_depth | lr | Val RMSE | Notes |
|-------|-----------|-----|----------|-------|
| 200 | 5 | 0.05 | - | Baseline |
| 100 | 3 | 0.1 | - | Smaller |
| 500 | 7 | 0.01 | - | Larger |

---

## Feature Importance Rankings

### MIC_PAO1 (XGBoost)

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | - | - |
| 2 | - | - |
| 3 | - | - |

### MIC_SA (XGBoost)

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | - | - |
| 2 | - | - |
| 3 | - | - |

---

## Anomalies and Issues

*Document any unexpected behaviors, errors, or edge cases encountered during experiments.*

---

## Key Insights

*Summary of important findings from experiments.*

1. **Shannon Entropy features improve Ridge regression by 11.1%** - This is the most significant finding, supporting the hypothesis that sequence randomness has predictive value.

2. **Simple models beat complex models** - Ridge regression outperformed both neural networks and XGBoost, demonstrating that feature engineering matters more than model complexity for small datasets.

3. **Target-dependent behavior** - MIC_SA was most predictable (R² = 0.755), while MIC_PAO1 and MIC_PAO1_PA showed lower predictability. This suggests different mechanisms or data quality issues across targets.

4. **Entropy features most valuable for MIC_PAO1_PA** - The best model for this target uses entropy-only features (R² = 0.391 vs 0.130 for composition).

5. **Neural networks underperformed** - Despite regularization (dropout, weight decay, early stopping), NNs struggled with the small dataset size.

6. **Combined features help for some targets** - MIC_SA and MIC_PAO1 benefited from combined features, suggesting complementary information.

---

## Commands Reference

```bash
# Run all experiments
uv run python main.py

# Run specific model
uv run python main.py --model nn

# Run specific feature set
uv run python main.py --features entropy

# Run specific target
uv run python main.py --target MIC_PAO1

# Quiet mode (less verbose)
uv run python main.py --quiet

# Custom output directory
uv run python main.py --output results/run_001
```
