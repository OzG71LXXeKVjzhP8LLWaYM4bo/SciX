# SciX Experiment Log

This document tracks experiments for the antibacterial polymer MIC prediction project.

---

## Experiment Configuration

### Models
- **Neural Network**: 64→32→16→1, ReLU, Dropout=0.3, Adam optimizer
- **XGBoost**: 200 estimators, max_depth=5, lr=0.05
- **Ridge Regression**: L2 regularization, alpha=1.0
- **Logistic Regression**: 3-class classification, C=1.0

### Feature Sets
| Set | Features | Count |
|-----|----------|-------|
| Composition | blocks, dpn, Dispersity, cLogP, Target, NMR, GPC + compositions | 11 |
| Entropy | blocks, dpn, Dispersity, cLogP, Target, NMR, GPC + entropy metrics | 11 |
| Combined | All features | 15 |

### Evaluation
- **Method**: 5-fold stratified cross-validation
- **Sessions**: 5 independent runs with different random seeds (42, 123, 456, 789, 1000)
- **Metrics**: RMSE, MAE, R² (regression); Accuracy, F1 macro/weighted (classification)

---

## Results Summary (5 Sessions Aggregated)

### Classification Results (Logistic Regression)

| Target | Features | Accuracy | Std | F1 Macro | Entropy Benefit |
|--------|----------|----------|-----|----------|-----------------|
| **MIC_SA** | **composition** | **93.9%** | 4.2% | 0.85 | baseline |
| MIC_SA | entropy | 93.7% | 4.3% | 0.84 | -0.2% |
| MIC_SA | combined | 93.7% | 4.0% | 0.84 | -0.2% |
| **MIC_PAO1** | **entropy** | **80.9%** | 3.6% | 0.63 | **+6.5%** |
| MIC_PAO1 | combined | 79.5% | 4.4% | 0.61 | +5.1% |
| MIC_PAO1 | composition | 74.4% | 6.0% | 0.53 | baseline |
| **MIC_PAO1_PA** | **entropy** | **80.0%** | 4.9% | 0.62 | **+4.5%** |
| MIC_PAO1_PA | combined | 78.9% | 4.3% | 0.60 | +3.4% |
| MIC_PAO1_PA | composition | 75.5% | 5.1% | 0.55 | baseline |

### Regression Results (Best Models)

| Target | Model | Features | R² | Std | RMSE | Std |
|--------|-------|----------|-----|-----|------|-----|
| **MIC_SA** | Ridge | combined | **0.724** | 0.144 | 28.0 | 7.5 |
| MIC_SA | XGBoost | combined | 0.707 | 0.161 | 28.6 | 9.0 |
| MIC_SA | XGBoost | entropy | 0.703 | 0.149 | 29.1 | 8.1 |
| MIC_SA | Ridge | entropy | 0.702 | 0.150 | 29.2 | 7.8 |
| MIC_SA | XGBoost | composition | 0.687 | 0.158 | 29.5 | 8.5 |
| MIC_SA | Ridge | composition | 0.684 | 0.110 | 30.6 | 6.0 |
| MIC_SA | NN | combined | 0.612 | 0.177 | 33.5 | 8.1 |
| MIC_SA | NN | entropy | 0.589 | 0.198 | 34.4 | 8.5 |
| MIC_SA | NN | composition | 0.567 | 0.158 | 35.8 | 7.0 |
| MIC_PAO1_PA | XGBoost | entropy | 0.183 | 0.261 | 48.7 | 10.2 |
| MIC_PAO1_PA | XGBoost | combined | 0.131 | 0.328 | 49.7 | 10.8 |
| MIC_PAO1 | XGBoost | combined | 0.118 | 0.209 | 50.6 | 8.6 |
| MIC_PAO1 | XGBoost | entropy | 0.086 | 0.278 | 51.2 | 9.7 |

---

## Key Findings

### 1. Shannon Entropy Impact

**Classification** (entropy vs composition, 5-session average):
- MIC_PAO1: 74.4% → 80.9% **(+6.5%)**
- MIC_PAO1_PA: 75.5% → 80.0% **(+4.5%)**
- MIC_SA: 93.9% → 93.7% (no improvement, composition marginally better)

**Regression** (MIC_SA):
- Combined features R² = 0.724 vs Composition R² = 0.684 (+0.04)
- Entropy features improve regression when combined with composition

### 2. Best Models by Target

| Target | Task | Best Model | Performance |
|--------|------|------------|-------------|
| MIC_SA | Regression | Ridge + combined | R² = 0.724 ± 0.144 |
| MIC_SA | Classification | Logistic + composition | 93.9% ± 4.2% |
| MIC_PAO1 | Classification | Logistic + entropy | 80.9% ± 3.6% |
| MIC_PAO1_PA | Classification | Logistic + entropy | 80.0% ± 4.9% |

### 3. Model Performance Ranking

**Regression** (MIC_SA):
1. Ridge/XGBoost (R² = 0.68-0.72)
2. Neural Network (R² = 0.57-0.61)

**Classification** (all targets):
- Logistic regression provides stable performance across all targets
- Entropy features critical for Pseudomonas targets

---

## Class Distribution

| Target | Low (≤64) | Medium (64-128) | High (>128) |
|--------|-----------|-----------------|-------------|
| MIC_PAO1 | 21 (18.9%) | 75 (67.6%) | 15 (13.5%) |
| MIC_SA | 1 (0.9%) | 83 (74.8%) | 27 (24.3%) |
| MIC_PAO1_PA | 21 (18.9%) | 75 (67.6%) | 15 (13.5%) |

---

## Cross-Session Variability

Results aggregated from 5 independent sessions (seeds: 42, 123, 456, 789, 1000):

| Metric | Cross-Session Std |
|--------|-------------------|
| Classification accuracy | 1.3-2.3% |
| Regression R² | 0.05-0.11 |
| Regression RMSE | 1.0-3.6 |

Low cross-session variability indicates robust and reproducible results.

---

## Commands Reference

```bash
# Run all experiments with 5-fold CV
uv run python main.py --model nn xgboost ridge logistic

# Run multiple sessions with different seeds
uv run python main.py --model nn xgboost ridge logistic --seed 42
uv run python main.py --model nn xgboost ridge logistic --seed 123
uv run python main.py --model nn xgboost ridge logistic --seed 456
uv run python main.py --model nn xgboost ridge logistic --seed 789
uv run python main.py --model nn xgboost ridge logistic --seed 1000
```

---

## Conclusions

1. **Shannon entropy improves classification for Pseudomonas targets** (+4.5-6.5%)
2. **MIC_SA is highly predictable** (93.9% classification, R²=0.72 regression)
3. **Classification is more robust than regression** for this dataset
4. **Combined features improve regression** (R² = 0.72 vs 0.68 for composition alone)
5. **Results are reproducible** across 5 independent sessions
